# Run 2a — H200 (GPU-side numbers; not the Grace baseline)

- **Date:** 2026-08-06
- **Platform:** brev.nvidia (org Unitary-Foundation-Brev), nebius
  `gpu-h200-sxm.1gpu-16vcpu-200gb`, $5.40/hr
- **GPU:** NVIDIA H200 SXM, 141 GB HBM3e, driver 580.159.04, CUDA 13.0.88
  (nvcc), sm_90 — the documented FP64 stand-in for GH200's Hopper GPU
  (identical FP64 rate/HBM class; NVLink-C2C transfer behavior NOT
  represented, PCIe path instead)
- **CPU:** 16 vCPU x86_64 (OpenMP, 16 threads), Ubuntu 24.04, g++ 13.3,
  `-O3 -march=native`
- **Code:** branch `clifft-research`, commit 75af10c
- **Command:** `./run.sh 10 30 12,16,18,20 4`
- **Validation:** all five validator families OK (max err ~1e-14..1e-17 vs
  1e-9 bar; zero rng / frame-word mismatches); MIMD shared-memory variant
  runs only at k=12 (227 KB opt-in limit), global elsewhere — see
  run_stderr.log
- **Caveat:** the CPU rows here are a 16-vCPU x86 box, NOT the 72-core Grace
  baseline the precommitted go/no-go verdict calls for. GPU-vs-GPU numbers
  (architecture fork, occupancy, crossover) are decision-grade; GPU-vs-CPU
  ratios are provisional until the GH200 session.
