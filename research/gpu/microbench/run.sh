#!/usr/bin/env bash
# Build and run the clifft active-block GPU microbenchmark.
# On the GH200 this builds both bench_cpu and bench_gpu; locally just bench_cpu.
set -euo pipefail
cd "$(dirname "$0")"

# Single-statevector sweep allocates 2^(KMAX+1) complex128 (EXPAND headroom):
# KMAX=28 -> ~8 GB, KMAX=30 -> ~32 GB (fine on GH200's HBM; bump if you want k=30).
KMIN="${1:-10}"
KMAX="${2:-28}"
BKS="${3:-12,16,18,20}"
LAYERS="${4:-4}"

CUDA_ARCH="${CUDA_ARCH:-90}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH="${CUDA_ARCH}"
cmake --build build -j

echo "== CPU =="
./build/bench_cpu "$KMIN" "$KMAX" "$BKS" "$LAYERS" | tee cpu.csv >/dev/null
echo "wrote cpu.csv"

if [[ -x ./build/bench_gpu ]]; then
  echo "== GPU =="
  ./build/bench_gpu "$KMIN" "$KMAX" "$BKS" "$LAYERS" | tee gpu.csv >/dev/null
  echo "wrote gpu.csv"
else
  echo "bench_gpu not built (no CUDA compiler found) -- run this on the GH200."
fi
