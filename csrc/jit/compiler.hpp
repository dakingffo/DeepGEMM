#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
#include <regex>
#include <string>

#include "../jit_kernels/heuristics/config.hpp"
#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"
#include "include_parser.hpp"

namespace deep_gemm {

// Precompiled kernel key — used for filesystem-based lookup of AOT-compiled CUBINs.
// Encodes the kernel identity, shape dimensions baked at compile time, and the
// specific config (tiling, pipeline) chosen by the heuristic.
struct PrecompiledKey {
    int arch_major;
    std::string name;          // e.g. "bf16_gemm"
    std::string compiled_dims; // e.g. "nk"
    int m, n, k;               // full shape (0 = not compiled, runtime variable)
    int block_m, block_n, block_k;
    int cluster_size;
    int num_stages;

    static PrecompiledKey from(const GemmDesc& desc, const GemmConfig& config,
                                const std::string& name) {
        return {
            device_runtime->get_arch_major(),
            name,
            desc.compiled_dims,
            desc.m, desc.n, desc.k,
            config.layout.block_m, config.layout.block_n, config.layout.block_k,
            config.layout.get_cluster_size(),
            config.pipeline_config.num_stages
        };
    }

    // Subdirectory from compiled dims: e.g. compiled_dims="nk" → "n=4096,k=14336"
    std::string dim_path() const {
        std::string result;
        for (const char c: compiled_dims) {
            if (!result.empty()) result += ",";
            int val = 0;
            switch (c) {
                case 'm': val = m; break;
                case 'n': val = n; break;
                case 'k': val = k; break;
            }
            result += fmt::format("{}={}", c, val);
        }
        return result.empty() ? "all_runtime" : result;
    }

    // Config filename: e.g. "B128_N256_K64_C2_S7"
    std::string config_filename() const {
        return fmt::format("B{}_N{}_K{}_C{}_S{}",
            block_m, block_n, block_k, cluster_size, num_stages);
    }

    std::filesystem::path cubin_path(const std::filesystem::path& root) const {
        return root / fmt::format("sm{}", arch_major) / name / dim_path()
            / (config_filename() + ".cubin");
    }

    std::filesystem::path header_path(const std::filesystem::path& root) const {
        return root / fmt::format("sm{}", arch_major) / name / dim_path()
            / (config_filename() + ".header");
    }
};

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::filesystem::path cuda_home;
    static std::filesystem::path cuobjdump_path;
    static std::filesystem::path precompiled_root;

    static void prepare_init(const std::string& library_root_path,
                             const std::string& cuda_home_path_by_python) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::cuobjdump_path = Compiler::cuda_home / "bin" / "cuobjdump";

        // Precompiled root: env var or default to package-relative path
        if (const auto env_path = get_env<std::string>("DG_PERSISTENT_OUTPUT");
            not env_path.empty()) {
            precompiled_root = env_path;
        } else {
            precompiled_root = Compiler::library_root_path / "precompiled";
        }
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not cuda_home.empty());
        DG_HOST_ASSERT(not cuobjdump_path.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_JIT_CPP_STANDARD", 20));
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))
            flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    static void fsync_path(const std::filesystem::path& path) {
        const auto fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
    }

    // Recursively fsync a directory: files and subdirectories first (bottom-up), then the directory itself
    // NOTES: ensures data and directory entries are visible on other nodes in distributed filesystems
    static void fsync_dir(const std::filesystem::path& dir_path) { // NOLINT(*-no-recursion)
        for (const auto& entry: std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_directory())
                fsync_dir(entry.path());
            else if (entry.is_regular_file())
                fsync_path(entry.path());
        }
        fsync_path(dir_path);
    }

    static void put(const std::filesystem::path& path, const std::string& data) {
        std::ofstream out(path, std::ios::binary);
        DG_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // NOTES: fsync to ensure the data is visible to other processes (e.g., NVCC)
        // on distributed filesystems, where `close()` alone does not guarantee persistence
        fsync_path(path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Compile into a temporary directory, then atomically rename the whole directory
        // NOTES: renaming a directory is atomic on both local and distributed filesystems,
        // avoiding the stale inode issue that occurs when renaming individual files
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        make_dirs(tmp_dir_path);

        // Compile into the temporary directory
        const auto tmp_cubin_path = tmp_dir_path / "kernel.cubin";
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_PTX")) {
            const auto tmp_ptx_path = tmp_dir_path / "kernel.ptx";
            compile(code, tmp_dir_path, tmp_cubin_path, tmp_ptx_path);
        } else {
            compile(code, tmp_dir_path, tmp_cubin_path);
        }

        // Disassemble if needed
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_SASS")) {
            const auto tmp_sass_path = tmp_dir_path / "kernel.sass";
            disassemble(tmp_cubin_path, tmp_sass_path);
        }

        // Fsync before rename to ensure visibility on distributed filesystems
        fsync_dir(tmp_dir_path);

        // Atomically rename the temporary directory to the final cache path
        // NOTES: if another rank already created dir_path, rename will fail — that's fine
        make_dirs(dir_path.parent_path());
        std::error_code error_code;
        std::filesystem::rename(tmp_dir_path, dir_path, error_code);
        if (error_code) {
            // Another rank beat us, then clean up our dir and use the existing one
            // NOTES: avoid `std::filesystem::remove_all` here — it can segfault on
            // distributed filesystems, when concurrent processes operate
            // on the same parent directory, causing stale directory entries
            safe_remove_all(tmp_dir_path);
        }

        // Put into the runtime cache
        const auto runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    // ─── Precompiled (AOT) kernel support ────────────────────────────
    //
    // Overload of build() that intercepts the JIT path with a precompiled
    // CUBIN lookup.  The PrecompiledKey carries the shape/config metadata
    // that the original JIT hash would have embedded in the generated code
    // string, but in a structured form that can be matched across machines.
    //
    // Flow:
    //   1. L1 in-memory cache (same as original)
    //   2. Precompiled filesystem cache – lookup by (arch, name, dims, config)
    //      → hit  + valid header hash  → copy into L2, load, return
    //      → miss / stale              → fall through
    //   3. Original JIT path (L2 disk cache + NVCC/NVRTC compilation)
    //   4. If DG_PERSISTENT_COMPILE=1  → copy compiled CUBIN + header hash
    //      into the precompiled directory for packaging
    //
    // The original build(name, code) is kept for kernels that do not carry
    // GemmDesc/GemmConfig metadata (attention, layout, etc.).
    // ─────────────────────────────────────────────────────────────────

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code,
                                          const PrecompiledKey& key) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" /
            fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // L1 memory cache
        if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Try precompiled CUBIN
        if (const auto runtime = try_load_precompiled(code, key, dir_path); runtime != nullptr)
            return runtime;

        // Fall through to original JIT compilation
        auto runtime = build(name, code);

        // Persist to precompiled directory if in compile-host mode
        if (get_env<int>("DG_PERSISTENT_COMPILE", 0))
            persist_to_precompiled(code, name, key);

        return runtime;
    }

private:
    // Look up a precompiled CUBIN, verify the header hash, and if valid
    // ingest it into the JIT L2 cache so subsequent calls hit L1/L2 directly.
    // Returns nullptr on miss or stale hash.
    std::shared_ptr<KernelRuntime> try_load_precompiled(
            const std::string& code, const PrecompiledKey& key,
            const std::filesystem::path& dir_path) const {
        const auto pc_cubin = key.cubin_path(precompiled_root);
        const auto pc_header = key.header_path(precompiled_root);

        if (not std::filesystem::exists(pc_cubin))
            return nullptr;

        // Verify header hash if present
        if (std::filesystem::exists(pc_header)) {
            std::ifstream hf(pc_header);
            std::string stored_hash;
            std::getline(hf, stored_hash);
            if (stored_hash != include_parser->get_hash_value(code, true)) {
                if (get_env<int>("DG_JIT_DEBUG"))
                    printf("Precompiled CUBIN header hash mismatch: %s (stored) vs %s (current), "
                           "falling back to JIT\n", stored_hash.c_str(),
                           include_parser->get_hash_value(code, true).c_str());
                return nullptr;
            }
        }

        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Loading precompiled CUBIN: %s\n", pc_cubin.c_str());

        // Ingest into L2 cache: copy CUBIN + write code, then load
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        make_dirs(tmp_dir_path);
        std::filesystem::copy_file(pc_cubin, tmp_dir_path / "kernel.cubin");
        put(tmp_dir_path / "kernel.cu", code);
        fsync_dir(tmp_dir_path);

        make_dirs(dir_path.parent_path());
        std::error_code error_code;
        std::filesystem::rename(tmp_dir_path, dir_path, error_code);
        if (error_code)
            safe_remove_all(tmp_dir_path);

        return kernel_runtime_cache->get(dir_path);
    }

    // Copy the just-compiled CUBIN from the JIT L2 cache into the precompiled
    // directory, along with the current header hash.
    void persist_to_precompiled(const std::string& code,
                                 const std::string& name,
                                 const PrecompiledKey& key) const {
        // Recompute the L2 cache path (same formula as build())
        const auto kernel_signature = fmt::format("{}$${}$${}$${}",
            name, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" /
            fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        const auto src_cubin = dir_path / "kernel.cubin";
        if (not std::filesystem::exists(src_cubin))
            return;  // compilation was a no-op (e.g. another rank built it)

        const auto dst_cubin = key.cubin_path(precompiled_root);
        const auto dst_header = key.header_path(precompiled_root);

        // Atomic write of CUBIN + header.
        // IMPORTANT: write tmp to the SAME filesystem as the destination,
        // then rename.  POSIX rename is only atomic within a single
        // filesystem; across filesystems (e.g. overlayfs → NFS) it fails
        // with EXDEV.  The UUID prefix prevents concurrent writers from
        // overwriting each other's tmp files.
        make_dirs(dst_cubin.parent_path());
        const auto tmp_dir = dst_cubin.parent_path() / ".tmp";
        make_dirs(tmp_dir);
        const auto tmp_base = get_uuid() + "_" + key.config_filename();
        const auto tmp_cubin = tmp_dir / (tmp_base + ".cubin");
        const auto tmp_header = tmp_dir / (tmp_base + ".header");
        std::filesystem::copy_file(src_cubin, tmp_cubin);
        put(tmp_header, include_parser->get_hash_value(code, true));

        std::error_code ec;
        std::filesystem::rename(tmp_cubin, dst_cubin, ec);
        // If rename failed, another writer already placed this CUBIN
        // (or some other error) — keep the existing file.
        std::filesystem::rename(tmp_header, dst_header, ec);

        // Always clean up tmp files
        if (std::filesystem::exists(tmp_cubin))
            std::filesystem::remove(tmp_cubin);
        if (std::filesystem::exists(tmp_header))
            std::filesystem::remove(tmp_header);

        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Persisted precompiled CUBIN: %s\n", dst_cubin.c_str());
    }

public:
    static void disassemble(const std::filesystem::path &cubin_path, const std::filesystem::path &sass_path) {
        // Disassemble the CUBIN file to SASS
        const auto command = fmt::format("{} --dump-sass {} > {}", cuobjdump_path.c_str(), cubin_path.c_str(), sass_path.c_str());
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running cuobjdump command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("cuobjdump failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "cuobjdump failed");
        }
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::optional<std::filesystem::path> &ptx_path = std::nullopt) const = 0;
};

DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuobjdump_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, precompiled_root);

class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto command = std::string(nvcc_path) + " --version";
        const auto [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }

public:
    NVCCCompiler() {
        // Override the compiler signature
        nvcc_path = cuda_home / "bin" / "nvcc";
        if (const auto env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // The override the compiler flags
        // Only NVCC >= 12.9 supports arch-specific family suffix
        const auto arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), arch);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Compile
        // Avoid cwd files shadowing C++ standard library headers
        const auto compile_dir = make_tmp_dir();
        const auto command = fmt::format("cd {} && {} {} -cubin -o {} {}",
            compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Compile to PTX if needed
        if (ptx_path.has_value()) {
            const auto ptx_command = fmt::format("cd {} && {} {} -ptx -o {} {}",
                compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), ptx_path->c_str(), flags);
            if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
                printf("Running NVCC PTX command: %s\n", ptx_command.c_str());
            const auto [ptx_return_code, ptx_output] = call_external_command(ptx_command);
            if (ptx_return_code != 0) {
                printf("NVCC PTX compilation failed: %s\n", ptx_output.c_str());
                DG_HOST_ASSERT(false and "NVCC PTX compilation failed");
            }
        }

        // Check local memory usage
        if (get_env("DG_JIT_PTXAS_CHECK", 0))
            DG_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

class NVRTCCompiler final: public Compiler {
public:
    NVRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        DG_NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");

        // Build include directories list
        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());

        // Add PCH support for version 12.8 and above
        // NOTES: PCH is vital for compilation speed
        std::string pch_flags;
        if (major > 12 or minor >= 8) {
            pch_flags = "--pch ";
            if (get_env<int>("DG_JIT_DEBUG", 0))
                pch_flags += "--pch-verbose=true ";
        }

        // Override the compiler flags
        // Only NVRTC >= 12.9 supports arch-specific family suffix
        const auto arch = device_runtime->get_arch(false, major > 12 or minor >= 9);
        flags = fmt::format("{} {}--gpu-architecture=sm_{} -default-device {} --device-int128",
                            flags, include_dirs, arch, pch_flags);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Parse compilation options
        std::istringstream iss(flags);
        std::vector<std::string> options;
        std::string option;
        while (iss >> option)
            options.push_back(option);

        // Convert to C-style string array for NVRTC
        std::vector<const char*> option_cstrs;
        for (const auto& opt: options)
            option_cstrs.push_back(opt.c_str());

        // Print compiler command if requested
        if (get_env<int>("DG_JIT_DEBUG", 0) or get_env<int>("DG_JIT_PRINT_COMPILER_COMMAND", 0)) {
            printf("Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt: options)
                printf("%s ", opt.c_str());
            printf("\n");
        }

        // Create NVRTC program and compile
        nvrtcProgram program;
        DG_NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto compile_result = nvrtcCompileProgram(program, static_cast<int>(option_cstrs.size()), option_cstrs.data());

        // Get and print compiler log
        size_t log_size;
        DG_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s\n", compilation_log.c_str());
            }
        }

        if (ptx_path.has_value()) {
            // Get PTX size and data if needed
            size_t ptx_size;
            DG_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
            std::string ptx_data(ptx_size, '\0');
            DG_NVRTC_CHECK(nvrtcGetPTX(program, ptx_data.data()));

            // Write into the file system
            put(ptx_path.value(), ptx_data);
        }

        // Get CUBIN size and data
        size_t cubin_size;
        DG_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
        std::string cubin_data(cubin_size, '\0');
        DG_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));

        // Write into the file system
        put(cubin_path, cubin_data);

        // Cleanup
        DG_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    } else {
        return std::make_shared<NVCCCompiler>();
    }
});

} // namespace deep_gemm
