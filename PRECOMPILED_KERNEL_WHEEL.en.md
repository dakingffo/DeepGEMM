# Precompiled Kernel Wheel — Hybrid AOT+JIT Scheme for DeepGEMM

## Motivation

DeepGEMM relies on JIT (Just-In-Time) compilation: every unique (shape, tiling, config) combination triggers an NVCC/NVRTC invocation at runtime. This provides extreme flexibility but incurs a **cold-start latency** of seconds to tens of seconds the first time a shape is encountered.

The goal of this scheme is to **persist commonly-used kernels into the wheel**, enabling hot-start (sub-millisecond CUBIN load) on deploy machines while **retaining the existing JIT path as a fallback** for unseen shapes.

---

## Design Overview

### Principle: File System as Manifest

No external database, no JSON index. The directory tree **is** the manifest:

```
deep_gemm/precompiled/
└── sm{arch_major}/              # 9 = H100, 10 = B200
    └── {kernel_family}/         # bf16_gemm, fp8_gemm, fp8_fp4_gemm
        └── {compiled_dims}/     # e.g. "n=4096,k=14336"
            ├── B128_N256_K64_C2_S7.cubin   # config signature
            └── B128_N256_K64_C2_S7.header   # include hash
```

**Directory tiers:**

| Tier | Example | Must Match? |
|------|---------|-------------|
| SM architecture | `sm90/` | **Yes** — SASS is not cross-architecture |
| Kernel family | `bf16_gemm/` | **Yes** — different kernels are different |
| Compiled dims | `n=4096,k=14336/` | **Yes** — JIT-specialized values must align |
| Config signature | `B128_N256_K64_C2_S7.cubin` | **Yes** — block sizes, pipeline stages etc. |

**Why file-system lookup works:** `stat()` is O(1) microsecond-level on local storage, and the precompiled tree lives inside the installed Python package (`site-packages/deep_gemm/precompiled/`).

### Interception Point

The interception happens inside `Compiler::build()` in [csrc/jit/compiler.hpp](csrc/jit/compiler.hpp). A new overload `build(name, code, key)` is called by GEMM kernels; the original `build(name, code)` is kept for non-GEMM kernels (attention, layout, etc.).

```
build(name, code, key):
  1. L1 in-memory cache   ← unchanged
  2. Precompiled lookup   ← stat() + header hash verify
     → hit:  copy CUBIN into L2 cache, load, return
     → miss: fall through
  3. Original JIT path     ← L2 disk cache + NVCC/NVRTC
  4. If DG_PERSISTENT_COMPILE=1:
     → copy compiled CUBIN + header hash into precompiled tree
```

### Header Hash Verification

A `.header` file alongside each `.cubin` stores the `IncludeParser` hash at compile time. At load time, the current hash is compared:

- **Match** → CUBIN is valid, load directly
- **Mismatch** → `.cuh` headers have changed (library upgraded), CUBIN is stale, fallback to JIT

This is content-addressed version control — no manual version bump needed.

---

## Implementation

### New Types

**`PrecompiledKey`** ([csrc/jit/compiler.hpp](csrc/jit/compiler.hpp)) — a lightweight struct encoding kernel identity:

```cpp
struct PrecompiledKey {
    int arch_major;           // 9 or 10
    std::string name;         // "bf16_gemm", "fp8_gemm", ...
    std::string compiled_dims; // "nk", "mn"
    int m, n, k;              // shape (0 = runtime variable)
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

### New Compiler Members

| Member | Purpose |
|--------|---------|
| `Compiler::precompiled_root` | Static path. Default: `{lib_root}/precompiled`. Override: `DG_PERSISTENT_OUTPUT` |
| `build(name, code, key)` | New overload with precompiled interception |
| `try_load_precompiled(code, key, dir_path)` | Private. Looks up CUBIN, verifies header, ingests into L2 |
| `persist_to_precompiled(code, key)` | Private. Copies compiled CUBIN + header to precompiled tree |

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DG_PERSISTENT_COMPILE` | Enable compile-host mode (write precompiled CUBINs) | 0 (off) |
| `DG_PERSISTENT_OUTPUT` | Precompiled output/input directory | `{lib_root}/precompiled` |

### setup.py Changes

1. **`package_data`**: includes `'precompiled/**/*'`
2. **`merge_precompiled()`** in `CustomBuildPy`: if `DG_PERSISTENT_OUTPUT` points to an external directory, copies its contents into the build tree before wheel packaging

### Call Site Changes

Each GEMM kernel function that uses standard `GemmDesc` + `GemmConfig` now constructs a `PrecompiledKey` and calls the new overload:

```cpp
// Before:
const auto code = Runtime::generate(args);
const auto runtime = compiler->build("sm90_bf16_gemm", code);

// After:
const auto code = Runtime::generate(args);
const auto key = PrecompiledKey::from(desc, config, "bf16_gemm");
const auto runtime = compiler->build("sm90_bf16_gemm", code, key);
```

---

## Affected Files

### Core JIT Infrastructure

| File | Change |
|------|--------|
| [csrc/jit/compiler.hpp](csrc/jit/compiler.hpp) | `PrecompiledKey` struct, `precompiled_root`, `build(name,code,key)`, `try_load_precompiled()`, `persist_to_precompiled()` |

### GEMM Kernel Implementations (23 call sites)

| File | Kernel Family | Sites |
|------|---------------|-------|
| [csrc/jit_kernels/impls/sm90_bf16_gemm.hpp](csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) | `bf16_gemm` | 6 |
| [csrc/jit_kernels/impls/sm100_bf16_gemm.hpp](csrc/jit_kernels/impls/sm100_bf16_gemm.hpp) | `bf16_gemm` | 6 |
| [csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp](csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp) | `fp8_gemm` | 2 |
| [csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp](csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp) | `fp8_gemm` | 4 |
| [csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp](csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp) | `fp8_fp4_gemm` | 5 |

### Packaging

| File | Change |
|------|--------|
| [setup.py](setup.py) | `package_data` + `merge_precompiled()` |
| [.gitignore](.gitignore) | Ignore `*.cubin`, `*.header` |
| [deep_gemm/precompiled/.gitkeep](deep_gemm/precompiled/.gitkeep) | Placeholder directory |

### Not Affected

Kernels that do **not** use the standard `GemmDesc` + `GemmConfig` pattern remain on the original `build(name, code)` path:

- `sm100_fp8_gemm_1d1d.hpp` — uses a different `GemmConfig` struct
- `smxx_layout.hpp` — layout kernels
- `smxx_fp8_*_mqa_logits.hpp` — attention kernels
- `smxx_fp8_*_paged_mqa_logits.hpp` — paged attention kernels
- `sm90_bmk_bnk_mn.hpp` / `sm100_bmk_bnk_mn.hpp` — BMK/BNK kernels
- `sm90_tf32_hc_prenorm_gemm.hpp` / `sm100_tf32_hc_prenorm_gemm.hpp` — TF32 prenorm
- `sm100_fp8_fp4_mega_moe.hpp` — Mega MoE
- `smxx_clean_logits.hpp` — logits cleanup

These use the original `build(name, code)` and fall through to JIT compilation normally.

---

## Workflow

### Compile Host (CI)

```bash
# 1. Editable install
pip install -e .

# 2. Enable compile mode, enumerate shapes × hardware variants
export DG_PERSISTENT_COMPILE=1

for num_sms in 132 114; do
    python -c "
import deep_gemm, torch
deep_gemm.set_num_sms($num_sms)

a = torch.randn((7168, 14336), dtype=torch.bfloat16, device='cuda')
b = torch.randn((4096, 14336), dtype=torch.bfloat16, device='cuda')
d = torch.zeros((7168, 4096), dtype=torch.bfloat16, device='cuda')

deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims='nk')
# ↑ JIT runs normally, but DG_PERSISTENT_COMPILE=1 also writes
#   the CUBIN + .header to deep_gemm/precompiled/
"
done

# 3. Build wheel (precompiled/ is packaged automatically)
python setup.py bdist_wheel
# → dist/deep_gemm-2.5.0+...whl
```

### Deploy Machine

```bash
# After copying the wheel to the target machine (scp, NFS, object storage, etc.),
# install from wherever it landed:
pip install /shared/wheels/deep_gemm-2.5.0+...whl

# Use as normal — precompiled kernels load in <1ms
python my_training_script.py
```

No API changes, no environment variables needed. Precompiled CUBINs are transparently loaded; unrecognized shapes fall back to JIT.

### Concurrent Compile Hosts

Multiple CI machines can write to the same precompiled directory concurrently. Each CUBIN file is independent; the compile loop uses atomic `rename(tmp, final)`. If two hosts compile the same kernel, the second `rename` fails harmlessly and the first writer wins. No locking required.

---

## Performance Model

| Scenario | First Call | Second Call (same shape, same M) | Same N,K; different M |
|----------|-----------|----------------------------------|-----------------------|
| Pure JIT (cold) | 5–30s NVCC | 0s L1 | 0s L1 |
| Precompiled (wheel) | ~1ms CUBIN load | 0s L1 | 0s L1 |
| Missing shape (wheel) | 5–30s NVCC | 0s L1 | 0s L1 |

The precompiled scheme eliminates cold-start latency for all shapes baked into the wheel, while the JIT fallback ensures full coverage.
