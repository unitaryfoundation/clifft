#!/usr/bin/env bash
# Syntax-checks the CUDA backend with clang on machines without a CUDA
# toolkit, using the stub header in cuda_stub/. Both compilation passes must
# be clean before spending GPU time. Run from the repo root after a CPU
# configure (the stim include comes from the FetchContent checkout).
set -euo pipefail

STIM_INC=$(find build/_deps/stim-src -maxdepth 3 -name stim.h | head -1 | xargs dirname)
COMMON=(-std=c++20 -x cuda -nocudainc -nocudalib -fsyntax-only
        -Isrc -Ibuild/generated -I"$STIM_INC" -Iresearch/sampling_gpu/cuda_stub)

clang++ "${COMMON[@]}" --cuda-host-only src/clifft/sampling/cuda/sampler.cu
clang++ "${COMMON[@]}" --cuda-device-only --cuda-gpu-arch=sm_90 \
    src/clifft/sampling/cuda/sampler.cu
echo "CUDA syntax check: host and device passes clean"
