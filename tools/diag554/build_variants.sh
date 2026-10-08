#!/bin/bash
# Build the canary benchmark binary for each executor variant, mirroring the
# performance canary: fixed source and build paths, Release, ThinLTO, lld,
# OpenMP, x86-64-v2.
set -euo pipefail

ROOT=${GITHUB_WORKSPACE:-$PWD}
SOURCE_SHA=${SOURCE_SHA:?SOURCE_SHA names the clifft revision to patch}
mkdir -p "$ROOT/canary-bin"
for variant in control local compact; do
    rm -rf "$ROOT/benchmark-source"
    mkdir -p "$ROOT/benchmark-source"
    git -C "$ROOT" archive "$SOURCE_SHA" | tar -x -C "$ROOT/benchmark-source"
    if [[ $variant != control ]]; then
        git -C "$ROOT/benchmark-source" apply "$ROOT/tools/diag554/variants/$variant.patch"
    fi
    cmake -E remove_directory "$ROOT/benchmark-build"
    cmake -S "$ROOT/benchmarks" -B "$ROOT/benchmark-build" \
        -DCLIFFT_SOURCE_DIR="$ROOT/benchmark-source" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DCLIFFT_OPENMP=ON \
        -DCLIFFT_CPU_BASELINE=x86-64-v2
    cmake --build "$ROOT/benchmark-build" --target clifft_benchmarks --parallel
    grep --quiet -- '-flto=thin' "$ROOT/benchmark-build/compile_commands.json"
    grep --quiet -- '-fuse-ld=lld' "$ROOT/benchmark-build/build.ninja"
    cp "$ROOT/benchmark-build/clifft_benchmarks" "$ROOT/canary-bin/$variant"
done
ls -la "$ROOT/canary-bin"
