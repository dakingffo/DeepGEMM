# 预编译 Kernel Wheel — DeepGEMM 混合 AOT+JIT 方案

## 动机

DeepGEMM 采用 JIT（即时编译）策略：每次遇到新的 (shape, tiling, config) 组合，都需在运行时调用 NVCC/NVRTC 编译 CUDA kernel。这赋予系统极致灵活性，但**冷启动延迟**高达数秒至数十秒。

本方案的目标是：**将常用算子持久化编译进 wheel 包**，使部署机器实现热启动（毫秒级 CUBIN 加载）；同时**保留现有 JIT 路径作为回退**，对未见过的 shape 自动触发即时编译。

---

## 设计概览

### 核心原则：文件系统即清单

不引入独立数据库、不依赖 JSON 索引。目录树**本身就是**清单：

```
deep_gemm/precompiled/
└── sm{架构主版本}/             # 9 = H100, 10 = B200
    └── {kernel 家族}/          # bf16_gemm / fp8_gemm / fp8_fp4_gemm
        └── {编译期维度}/        # 例: "n=4096,k=14336"
            ├── B128_N256_K64_C2_S7.cubin   # config 签名
            └── B128_N256_K64_C2_S7.header   # 头文件 hash
```

**目录层级：**

| 层级 | 示例 | 必须匹配？ |
|------|------|-----------|
| SM 架构 | `sm90/` | **是** — SASS 不跨架构兼容 |
| Kernel 家族 | `bf16_gemm/` | **是** — 不同 kernel 不同代码 |
| 编译期维度 | `n=4096,k=14336/` | **是** — JIT 烘焙的维度常量 |
| Config 签名 | `B128_N256_K64_C2_S7.cubin` | **是** — block 大小、流水线级数等 |

**为什么用文件系统查找：** `stat()` 在本地存储上是微秒级 O(1) 操作，precompiled 树位于已安装的 Python 包内 (`site-packages/deep_gemm/precompiled/`)。

### 拦截点

拦截发生在 `Compiler::build()` 内部（[csrc/jit/compiler.hpp](csrc/jit/compiler.hpp)）。新增重载 `build(name, code, key)` 由 GEMM kernel 调用；原始 `build(name, code)` 保留给非 GEMM kernel（attention、layout 等）。

```
build(name, code, key):
  1. L1 内存缓存         ← 不变
  2. 预编译查找          ← stat() + header hash 校验
     → 命中:  拷贝 CUBIN 到 L2 缓存，加载，返回
     → 未命中: 继续往下
  3. 原始 JIT 路径       ← L2 磁盘缓存 + NVCC/NVRTC 编译
  4. 如果 DG_PERSISTENT_COMPILE=1：
     → 将编译好的 CUBIN + header hash 写入 precompiled 树
```

### Header Hash 校验

每个 `.cubin` 旁有一个 `.header` 文件，存储编译时的 `IncludeParser` 哈希值。加载时对比当前哈希：

- **匹配** → CUBIN 有效，直接加载
- **不匹配** → `.cuh` 头文件已变更（库升级），CUBIN 过期，回退到 JIT

这是内容寻址的版本控制——无需手动维护版本号。

---

## 实现细节

### 新增类型

**`PrecompiledKey`**（[csrc/jit/compiler.hpp](csrc/jit/compiler.hpp)）——轻量结构体，编码 kernel 身份：

```cpp
struct PrecompiledKey {
    int arch_major;           // 9 或 10
    std::string name;         // "bf16_gemm" / "fp8_gemm" / ...
    std::string compiled_dims; // "nk" / "mn"
    int m, n, k;              // shape（0 = 运行期变量）
    int block_m, block_n, block_k;
    int cluster_size;
    int num_stages;

    static PrecompiledKey from(const GemmDesc&, const GemmConfig&, const std::string& name);
    std::string dim_path() const;       // "n=4096,k=14336"
    std::string config_filename() const; // "B128_N256_K64_C2_S7"
    std::filesystem::path cubin_path(const std::filesystem::path& root) const;
    std::filesystem::path header_path(const std::filesystem::path& root) const;
};
```

### 新增 Compiler 成员

| 成员 | 用途 |
|------|------|
| `Compiler::precompiled_root` | 静态路径。默认 `{lib_root}/precompiled`。可通过 `DG_PERSISTENT_OUTPUT` 覆盖 |
| `build(name, code, key)` | 新重载，包含预编译拦截逻辑 |
| `try_load_precompiled(code, key, dir_path)` | 私有方法。查找 CUBIN，校验 header，注入 L2 缓存 |
| `persist_to_precompiled(code, key)` | 私有方法。将编译好的 CUBIN + header 写入 precompiled 树 |

### 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `DG_PERSISTENT_COMPILE` | 开启编译主机模式（写入预编译 CUBIN） | 0（关闭） |
| `DG_PERSISTENT_OUTPUT` | 预编译输出/输入目录 | `{lib_root}/precompiled` |

### setup.py 改动

1. **`package_data`**：增加 `'precompiled/**/*'`
2. **`merge_precompiled()`**（`CustomBuildPy` 内）：若 `DG_PERSISTENT_OUTPUT` 指向外部目录，在打包前将其内容合并到 build 树中

### 调用点改动

每个使用标准 `GemmDesc` + `GemmConfig` 的 GEMM kernel 函数，现在构造 `PrecompiledKey` 并调用新重载：

```cpp
// 改前：
const auto code = Runtime::generate(args);
const auto runtime = compiler->build("sm90_bf16_gemm", code);

// 改后：
const auto code = Runtime::generate(args);
const auto key = PrecompiledKey::from(desc, config, "bf16_gemm");
const auto runtime = compiler->build("sm90_bf16_gemm", code, key);
```

---

## 改动文件清单

### 核心 JIT 基础设施

| 文件 | 改动 |
|------|------|
| [csrc/jit/compiler.hpp](csrc/jit/compiler.hpp) | `PrecompiledKey` 结构体、`precompiled_root`、`build(name,code,key)`、`try_load_precompiled()`、`persist_to_precompiled()` |

### GEMM Kernel 实现（23 个调用点）

| 文件 | Kernel 家族 | 调用点 |
|------|------------|--------|
| [csrc/jit_kernels/impls/sm90_bf16_gemm.hpp](csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) | `bf16_gemm` | 6 |
| [csrc/jit_kernels/impls/sm100_bf16_gemm.hpp](csrc/jit_kernels/impls/sm100_bf16_gemm.hpp) | `bf16_gemm` | 6 |
| [csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp](csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp) | `fp8_gemm` | 2 |
| [csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp](csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp) | `fp8_gemm` | 4 |
| [csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp](csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp) | `fp8_fp4_gemm` | 5 |

### 打包相关

| 文件 | 改动 |
|------|------|
| [setup.py](setup.py) | `package_data` + `merge_precompiled()` |
| [.gitignore](.gitignore) | 忽略 `*.cubin`、`*.header` |
| [deep_gemm/precompiled/.gitkeep](deep_gemm/precompiled/.gitkeep) | 占位目录 |

### 不受影响的文件

以下 kernel 不使用标准 `GemmDesc` + `GemmConfig` 模式，保持原有 `build(name, code)` 路径，正常回退到 JIT 编译：

- `sm100_fp8_gemm_1d1d.hpp` — 使用不同 `GemmConfig` 结构体
- `smxx_layout.hpp` — layout kernel
- `smxx_fp8_*_mqa_logits.hpp` — attention kernel
- `smxx_fp8_*_paged_mqa_logits.hpp` — paged attention kernel
- `sm90_bmk_bnk_mn.hpp` / `sm100_bmk_bnk_mn.hpp` — BMK/BNK kernel
- `sm90_tf32_hc_prenorm_gemm.hpp` / `sm100_tf32_hc_prenorm_gemm.hpp` — TF32 prenorm
- `sm100_fp8_fp4_mega_moe.hpp` — Mega MoE
- `smxx_clean_logits.hpp` — logits 清理

---

## 工作流

### 编译主机（CI）

```bash
# 1. 可编辑安装
pip install -e .

# 2. 开启编译模式，枚举 shape × 硬件变体
export DG_PERSISTENT_COMPILE=1

for num_sms in 132 114; do
    python -c "
import deep_gemm, torch
deep_gemm.set_num_sms($num_sms)

a = torch.randn((7168, 14336), dtype=torch.bfloat16, device='cuda')
b = torch.randn((4096, 14336), dtype=torch.bfloat16, device='cuda')
d = torch.zeros((7168, 4096), dtype=torch.bfloat16, device='cuda')

deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims='nk')
# ↑ JIT 正常执行，同时 DG_PERSISTENT_COMPILE=1 将
#   CUBIN + .header 写入 deep_gemm/precompiled/
"
done

# 3. 打 wheel（precompiled/ 自动打包）
python setup.py bdist_wheel
# → dist/deep_gemm-2.5.0+...whl
```

### 部署机器

```bash
# 先将 wheel 文件传输到目标机器（scp / NFS / 对象存储），
# 然后在目标机器上安装：
pip install /shared/wheels/deep_gemm-2.5.0+...whl

# 正常使用 — 预编译 kernel 在 <1ms 内加载
python my_training_script.py
```

无需 API 变更，无需设置环境变量。预编译 CUBIN 对用户透明；未见过的 shape 自动回退到 JIT。

### 多编译主机并发

多台 CI 机器可并发向同一 precompiled 目录写入。每个 CUBIN 文件独立；编译循环使用原子 `rename(tmp, final)`。若两台主机编译了相同的 kernel，第二台主机的 `rename` 会无害失败，以先到者为准。无需任何锁机制。

---

## 性能模型

| 场景 | 首次调用 | 第二次调用（相同 shape） | 相同 N,K；不同 M |
|------|---------|----------------------|-----------------|
| 纯 JIT（冷启动） | 5–30s NVCC | 0s L1 | 0s L1 |
| 预编译（wheel） | ~1ms CUBIN 加载 | 0s L1 | 0s L1 |
| 缺失 shape（wheel） | 5–30s NVCC | 0s L1 | 0s L1 |

预编译方案消除了所有打入 wheel 的 shape 的冷启动延迟，同时 JIT 回退保证任意 shape 都能执行。
