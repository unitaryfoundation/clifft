# H200 debug session - 2026-08-24

- **Platform:** brev.nvidia (org Unitary-Foundation-Brev), nebius
  `gpu-h200-sxm.1gpu-16vcpu-200gb`, $5.40/hr, instance `clifft-bench`
- **GPU:** NVIDIA H200 SXM, 141 GB HBM3e, driver 580.173.02, CUDA 13.0.88
  (nvcc), sm_90
- **CPU:** 16 vCPU x86_64 (weak baseline - GPU-vs-CPU ratios here are NOT
  decision-grade; the 48-core Granite Rapids comparison needs its own run)
- **Code:** branch `research/sampling-gpu-bench`; first build at 62d3726a
  compiled with ZERO nvcc errors; session fixes: block-global concurrency
  cap + sticky-error clear (final remote commit 6b86db89)
- **Command:** `bench_sampling --shots 100000 --threads 1,0 --validate 64`
  over the census corpus (see files)

## Files

- `results_h200.csv` / `.log` - full-corpus run at 100k shots, first six
  circuits (killed during rand_t15's single-thread CPU row, which would
  have taken ~1h; every completed row and validation is valid)
- `results_h200_wide.csv` / `.log` - rand_t15 (w=20) + surface at 10k
  shots; GPU rows for rand_t15 absent (the two bugs below)
- `results_h200_w20gpu.csv` / `.log` - rand_t15 GPU-only rerun after the
  fixes: block_global 3,561 shots/s, 64 replay rows OK

## Session findings

1. **All validations green.** Every tier on every circuit: 64 GPU record
   rows replayed reachable on the CPU executor, no marginal mismatches.
2. **BlockShared wins where it fits** (w <= 13), e.g. hidden_ccz_t4
   (w=12): 4.10M shots/s vs 525K CPU-16vcpu (7.8x) vs 325K thread-per-shot.
3. **Real-program w=20 throughput matches the 2026-08 microbench**: 3,561
   shots/s (block_global, real compiled rand_t15) vs 3,461 shots/s
   (synthetic MIMD real-mix, same GPU class) - within 3%.
4. **Two real bugs found and fixed** (invisible to syntax checks):
   block-global sized its slab pool by free memory and starved the launch;
   a failed tier's sticky CUDA error was reported by the next tier.
