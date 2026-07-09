#!/usr/bin/env bash
# Build and run the clifft active-block GPU microbenchmark.
# Builds whichever GPU targets the host toolchain provides (bench_gpu with
# nvcc on the GH200, bench_gpu_hip with hipcc on the MI300X); locally just
# bench_cpu.
set -euo pipefail
cd "$(dirname "$0")"

# Single-statevector sweep allocates 2^(KMAX+1) complex128 (EXPAND headroom):
# KMAX=28 -> ~8 GB, KMAX=30 -> ~32 GB (fine on GH200's HBM; bump if you want k=30).
KMIN="${1:-10}"
KMAX="${2:-28}"
BKS="${3:-12,16,18,20}"
LAYERS="${4:-4}"

CUDA_ARCH="${CUDA_ARCH:-90}"
HIP_ARCH="${HIP_ARCH:-gfx942}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH="${CUDA_ARCH}" -DHIP_ARCH="${HIP_ARCH}"
cmake --build build -j

echo "== CPU =="
./build/bench_cpu "$KMIN" "$KMAX" "$BKS" "$LAYERS" | tee cpu.csv >/dev/null
echo "wrote cpu.csv"

if [[ -x ./build/bench_gpu ]]; then
  echo "== GPU (CUDA) =="
  ./build/bench_gpu "$KMIN" "$KMAX" "$BKS" "$LAYERS" | tee gpu.csv >/dev/null
  echo "wrote gpu.csv"
fi

if [[ -x ./build/bench_gpu_hip ]]; then
  echo "== GPU (HIP) =="
  ./build/bench_gpu_hip "$KMIN" "$KMAX" "$BKS" "$LAYERS" | tee gpu_hip.csv >/dev/null
  echo "wrote gpu_hip.csv"
fi

if [[ ! -x ./build/bench_gpu && ! -x ./build/bench_gpu_hip ]]; then
  echo "no GPU benchmark built (no CUDA or HIP compiler found) -- run on the GH200/MI300X."
fi
