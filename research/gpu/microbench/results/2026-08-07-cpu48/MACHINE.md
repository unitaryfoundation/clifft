# Run 3 — 48-core CPU (measures what Run 2a projected)

- **Date:** 2026-08-07
- **Platform:** brev.nvidia, AWS `m8i.24xlarge`, $6.10/hr
- **CPU:** Intel Xeon 6975P-C (Granite Rapids), **48 physical cores** / 96
  threads, **2 MB L2 per core** (96 MB total), 480 MB L3, AVX-512.
  The 2 MB/core L2 matches the Xeon 8468 slice that Run 2a extrapolated from,
  so the cache-residency mechanism is preserved.
- **Code:** branch `clifft-research`, commit 9e5497b
- **Files:**
  - `cpu48_scaling.csv` — `bench_cpu 16 16 12,16,18 4` at
    `OMP_NUM_THREADS = 1,2,4,8,16,24,32,48` (`OMP_PROC_BIND=spread`)
  - `clifft_xcheck.txt` — REAL clifft (`clifft.sample`) on compiled circuits,
    normalised to Gamp/s via `research/gpu/opcode_census.py`
  - `cpu48.log` — topology, build log, per-thread-count timings

## Why this run mattered

Run 2a projected the 48-core CPU baseline from 8 cores. This measures it, and
also compares the microbenchmark's portable model kernels against clifft's own
hand-written AVX-512 paths on the same box.
