#!/usr/bin/env python3
"""Benchmark: precompiled (hot) vs JIT (cold) kernel loading.

Usage (each phase is a separate process to clear L1 in-memory cache):
  python benchmark_precompiled.py compile   # Phase 1: persist precompiled CUBINs
  python benchmark_precompiled.py hot       # Phase 2: load from precompiled (fresh process)
  python benchmark_precompiled.py cold      # Phase 3: pure JIT (precompiled moved aside)
  python benchmark_precompiled.py all       # Run all three sequentially via subprocess
"""

import os, sys, time, shutil, glob, subprocess
import torch
import deep_gemm


SCRIPT = __file__
PCD = os.path.join(os.path.dirname(deep_gemm.__file__), 'precompiled')
CACHE = os.path.expanduser("~/.deep_gemm/cache")

# ── Shapes ──
S1 = (7168, 4096, 14336)    # Phase 1 (compile + persist)
S2 = (2048, 4096, 14336)    # Phase 2 (same N,K as S1 → should hit precompiled)
S3 = (4096, 2048,  8192)    # Phase 3 (completely new → cold JIT)


def clear_disk_cache():
    if os.path.exists(CACHE):
        shutil.rmtree(CACHE)


def run_gemm(m, n, k):
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
    d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    t0 = time.perf_counter()
    deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims='nk')
    return time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
# Phase implementations (each runs in its own process)
# ═══════════════════════════════════════════════════════════════

def phase_compile():
    """Compile host mode: JIT compile two shapes + persist to precompiled/."""
    os.environ['DG_PERSISTENT_COMPILE'] = '1'
    os.environ['DG_JIT_DEBUG'] = '1'
    clear_disk_cache()

    t = run_gemm(*S1)
    print(f"COMPILE: S1 (JIT+persist) {t*1000:.0f}ms")

    run_gemm(*S2)   # also persist S2's config (same CUBIN, different M)
    t = run_gemm(*S2)
    print(f"COMPILE: S2 (JIT+persist) {t*1000:.0f}ms")

    # Show what was persisted
    for c in sorted(glob.glob(os.path.join(PCD, "**/*.cubin"), recursive=True)):
        print(f"  PERSISTED: {os.path.relpath(c, PCD)}")


def phase_hot():
    """Deploy mode: load from precompiled wheel (fresh process, no L1)."""
    # DG_PERSISTENT_COMPILE is OFF
    os.environ['DG_JIT_DEBUG'] = '1'
    clear_disk_cache()

    t = run_gemm(*S2)
    print(f"HOT: S2 (precompiled load) {t*1000:.1f}ms")


def phase_cold():
    """Pure JIT cold start (precompiled moved aside)."""
    os.environ['DG_JIT_DEBUG'] = '1'
    backup = PCD + '.backup'
    os.rename(PCD, backup)
    clear_disk_cache()

    t = run_gemm(*S3)
    print(f"COLD: S3 (pure JIT) {t*1000:.0f}ms")

    os.rename(backup, PCD)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def get_env():
    """Return the env for subprocess (inherit current, force CUDA_HOME)."""
    env = os.environ.copy()
    # Ensure correct CUDA for NVCC
    env['CUDA_HOME'] = '/usr/local/cuda-12.9'
    return env


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if phase == 'compile':
        phase_compile()
    elif phase == 'hot':
        phase_hot()
    elif phase == 'cold':
        phase_cold()
    elif phase == 'all':
        env = get_env()
        for p, label in [('compile', 'COMPILE HOST'), ('hot', 'DEPLOY HOT'), ('cold', 'JIT COLD')]:
            print(f"\n{'='*50}\n  PHASE: {label}\n{'='*50}")
            subprocess.run([sys.executable, SCRIPT, p], env=env)
    else:
        print(f"Usage: {SCRIPT} [compile|hot|cold|all]")
        sys.exit(1)
