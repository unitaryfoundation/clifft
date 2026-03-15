# UCC VM Throughput Optimization Plan

## Status: In Progress

The UCC VM is currently 3.7x slower than Qiskit-Aer on QV-20 (6.37s vs 1.7s
per shot). The bottleneck is instruction dispatch overhead and redundant array
passes -- not cache misses (the 16MB QV-20 statevector fits in our 35.8MB L3).

This plan targets three concrete optimizations, each measured against committed
benchmark circuits. A prior cache-blocking attempt (branch
`feat/cache-block-phase1`) was abandoned after adding 29% overhead.

## Machine Specifications

- Intel Xeon Platinum 8259CL @ 2.50GHz, 2 cores, 1 thread/core
- L1d: 32KB, L2: 1MB, L3: 35.8MB
- 7.2GB RAM, no swap
- GCC C++20, `-O2 -march=native -ffast-math`

## Benchmark Circuits

| Circuit | Qubits | peak_rank | Bytecode | Baseline/shot |
|---------|--------|-----------|----------|---------------|
| QV-10 | 10 | 10 | 2258 | ~1.03ms |
| Cultivation d=5 | 42 | 10 | 1518 | ~0.095ms |
| QV-20 (ref) | 20 | 20 | 8987 | ~6.37s |

Benchmark harness: `tests/test_benchmarks.cc` (Catch2 `[bench]` tag).
Run: `ctest --test-dir build -R Bench` or
`./build/tests/ucc_tests "[bench]" --benchmark-samples 5`.

---

## Step 0: Benchmark Harness -- DONE

Added Catch2 `BENCHMARK` tests for QV-10 (100 shots, ~103ms) and cultivation
d=5 (1000 shots, ~95ms). Fixture circuits committed to `tests/fixtures/`.

## Step 1: Transparent Huge Pages

Add `madvise(v_, bytes, MADV_HUGEPAGE)` hint after statevector allocation.
The 2^k array spans thousands of 4KB pages; 2MB huge pages reduce TLB
pressure for free.

- **File:** `src/ucc/svm/svm.cc` (SchrodingerState constructor)
- **Difficulty:** Trivial
- **Impact:** Modest for rank-10 circuits, potentially significant at rank-20+

## Step 2: Single-Axis CISC Fusion -- OP_ARRAY_U2 -- DONE

Replace chains of 1-qubit gates on the same virtual axis (e.g., Rz-H-Rz)
with a single fused 2x2 matrix sweep. QV circuits are 77% ARRAY_H + PHASE_ROT
instructions, mostly in per-qubit-per-layer sequences. Fusing these eliminates
redundant array passes and dispatch overhead.

- **New opcode:** `OP_ARRAY_U2` with constant pool for precomputed 2x2 matrices
  across all 4 Pauli frame input states (I, X, Z, Y)
- **New pass:** `SingleAxisFusionPass` (`src/ucc/optimizer/single_axis_fusion_pass.cc`)
- **Heuristic:** Fuse if sequence has >= 3 array ops, OR >= 2 with a rotation.
  Isolated H+T pairs are cheaper unfused (specialized scalar loops beat dense
  2x2 matrix sweeps for lightweight ops).
- **VM execution:** Read local frame bits, select matrix from constant pool,
  sweep array with 2x2 multiply, update frame bits and gamma. Matrix elements
  hoisted to stack locals to avoid aliasing-induced reloads.
- **Difficulty:** Medium
- **Impact:** Measured via A/B benchmark (Python bindings, Release build):

| Circuit | Instrs (before) | Instrs (fused) | Reduction | ms/shot (before) | ms/shot (fused) | Speedup |
|---------|-----------------|----------------|-----------|------------------|-----------------|--------|
| QV-10 (500 shots) | 2,258 | 814 | 63.9% | 1.22 | 0.79 | 1.54x |
| QV-20 (5 shots) | 8,967 | 3,089 | 65.6% | 6,370 | 3,402 | 1.87x |
| Cultivation d=5 (5000 shots) | 1,518 | 1,518 | 0% | 0.089 | 0.089 | 1.00x |

Cultivation shows no effect because its instruction stream is dominated by
QEC operations (CNOT, measurements, noise) with no fusible single-axis sequences.

---

## Potential Future Steps

These warrant profiling before committing to implementation.

### OpenMP Parallelization

Add `#pragma omp parallel for` to embarrassingly-parallel array sweeps.
All loop iterations access disjoint memory via `scatter_bits`. Prototyped
on the 2-core dev machine and found that the OpenMP runtime overhead
(~80% slowdown even with `if(iters >= 4096)` guard and `OMP_NUM_THREADS=1`)
makes it harmful for current workloads. Only viable on 4+ core machines
with rank-13+ statevectors where the per-iteration work justifies thread
spawning cost.

- Difficulty: Low
- Prerequisite: Test on a machine with 4+ cores and larger circuits

### AVX2 SIMD Intrinsics

Hand-written `__m256d` intrinsics for `exec_array_h` and `exec_array_u2`.
Compilers often fail to auto-vectorize `std::complex<double>` due to strict
IEEE-754 requirements. Protect with `#if defined(__AVX2__)` with scalar
fallback.

- Difficulty: High
- Prerequisite: Confirm auto-vectorization is actually failing via disassembly

### Spatial Fusion -- Multi-Target FWHT

Merge parallel layers of identical 1Q gates (e.g., H on multiple distinct axes)
into a single array sweep using Fast Walsh-Hadamard Transform butterfly.

- Difficulty: Very High
- Prerequisite: Measure actual parallel gate density in target circuits

---

## Abandoned: Cache-Blocked Execution

Branch `feat/cache-block-phase1` (9 commits, not merged to main).

Approach: Loop-inversion meta-instruction (`OP_CACHE_BLOCK`) that processed
the statevector in L1-sized chunks, applying a mini-program per chunk.

Results at QV-20:
- 8.22s/shot vs 6.37s baseline (29% slower)
- 116 ARRAY_SWAPs each sweeping 8MB = ~1GB extra memory traffic
- Mini-VM dispatch caused 2.9x branch explosion
- 16MB statevector already fits in 35.8MB LLC

Lesson: For sub-LLC workloads, reducing array passes (CISC fusion) and
dispatch overhead is more impactful than cache locality. Cache blocking
only helps when the statevector exceeds LLC capacity (QV-25+, 32MB+).
