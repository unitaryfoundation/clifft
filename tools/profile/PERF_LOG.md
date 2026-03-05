# UCC SVM Performance Log

**Environment:** Linux 6.8, GCC 13, x86-64 VM (4 vCPU), Stim SSE2 (128-bit SIMD).
**Build:** `RelWithDebInfo` (`-O2 -g`) for profiling, `Release` for benchmarks.
**Benchmark workloads:**

| Workload | Circuit | Qubits | Shots | Key Characteristic |
|---|---|---|---|---|
| QEC | `tools/bench/target_qec.stim` | 17 | 100k | Surface code: measurements, detectors, noise |
| Deep Clifford | Synthetic: 5000 random Cliffords | 50 | 100k | Pure Clifford, rank=0, 50 measurements |
| T-gate Mixed | Synthetic: 500 Cliffords + 10 T | 50 | 100k | Non-Clifford, rank=10 |

---

## Current Performance: UCC vs Stim (pytest-benchmark, 100k shots)

| Benchmark | Stim | UCC | Ratio |
|---|---|---|---|
| **Compile QEC** | 72 us | 2,032 us | 28x slower |
| **Compile Deep Clifford** | 1,146 us | 2,129 us | 1.9x slower |
| **Sample QEC** | 11.1 ms | 150.3 ms | 13.6x slower |
| **Sample Deep Clifford** | 21.7 ms | 730.6 ms | 33x slower |

Stim's advantage comes from its SIMD frame simulator processing 128
shots per circuit pass (SSE2). UCC runs each shot independently through
a scalar Schrodinger VM.

## Native Profiler Timing (100k shots, RelWithDebInfo)

| Workload | Total | Per-shot |
|---|---|---|
| Pure Clifford | 802 ms | 8.0 us |
| QEC | 157 ms | 1.6 us |
| T-gate (rank=10) | 1,165 ms | 11.6 us |

---

## Hotspot Analysis (perf record, Pure Clifford)

The profile is now well-distributed across bit manipulation and dispatch:

| % | Source | What |
|---|---|---|
| 20% | `bit_get` (.val access) | Pauli frame bit reads |
| 17% | DISPATCH macro sites | Computed goto indirect jumps |
| 13% | `bit_xor` (.val access) | Pauli frame bit updates |
| 11% | Label dispatch (FRAME_CZ) | Per-opcode goto targets |
| 10% | Label dispatch (others) | Per-opcode goto targets |
| 5% | `exec_frame_cz` body | CZ anticommutation check |
| 5% | MEAS_DORMANT_STATIC | Measurement handlers |
| 5% | MEAS_DORMANT_RANDOM | Measurement handlers |

No single hotspot exceeds 20%. Zero L1 cache misses across all workloads.

## Hotspot Analysis (perf record, QEC)

| % | Source | What |
|---|---|---|
| 36% | DISPATCH macro | Computed goto indirect jumps |
| 10% | Bytecode loop iteration | pc increment + bounds check |
| 8% | `exec_noise` early return | Gap sampling fast-path (skip check) |
| 5% | `bit_get` (.val) | Pauli frame reads |
| 2% | MEAS_DORMANT_STATIC | Measurement output |
| 2% | Xoshiro256++ RNG | `operator()` |
| 2% | Noise gap advance | `draw_next_noise` |

## Hotspot Analysis (perf record, T-gate rank=10)

| % | Source | What |
|---|---|---|
| 15% | DISPATCH macro | Computed goto indirect jumps |
| 10% | Bytecode iteration | pc increment + bounds check |
| 8% | `bit_get` (.val) | Pauli frame reads |
| 6% | T-gate inner loop | `arr[i] *= kExpIPiOver4` |
| 4% | Meas probability | `prob_b0 += norm(arr[i])` |
| 3% | `std::complex` multiply | T-gate complex arithmetic |
| 3% | `exec_expand` | Array doubling |
| 3% | `exec_swap` inner loop | Array element swaps |
| 3% | Meas compaction | Array half-copy |

---

## Latest Optimization: Threaded Dispatch + Direct Bit Ops (PR #60)

Three changes targeting the two dominant bottlenecks:

### 1. Direct `.val` Bit Access

Replaced Stim's `bitword<64>` SIMD shift operators with direct `.val`
access on the underlying `uint64_t`. The old `bit_get` passed bitwords
by value and used `operator>>` which triggered SIMD lane-crossing
shuffles. The new version compiles to a single `bt` or `shr`+`and`.

### 2. Force-Inline All Handlers

Marked all `exec_*` functions `static inline` to ensure the compiler
flattens them into the dispatch loop, eliminating call/return overhead.

### 3. Computed Goto Dispatch

Replaced the `for`/`switch` dispatch loop with GCC/Clang computed gotos
(`goto *dispatch_table[opcode]`). Each opcode gets its own indirect
branch site, so the CPU branch predictor can independently track the
most likely successor for each opcode type.

MSVC fallback retains the original switch loop under `#else`.

### Results

| Workload | Before | After | Speedup |
|---|---|---|---|
| **Pure Clifford** | 1,077 ms (10.8 us/shot) | 802 ms (8.0 us/shot) | **1.34x** |
| **QEC** | 187 ms (1.9 us/shot) | 157 ms (1.6 us/shot) | **1.19x** |
| **T-gate** | 1,391 ms (13.9 us/shot) | 1,165 ms (11.6 us/shot) | **1.19x** |

| Benchmark (vs Stim) | Before | After | Improvement |
|---|---|---|---|
| **Sample QEC** | 15.7x slower | 13.6x slower | 1.15x faster |
| **Sample Deep Clifford** | 47x slower | 33x slower | 1.40x faster |

Pure Clifford saw the largest gain (1.34-1.40x) because it is 100%
frame opcode dispatch — exactly what the threaded dispatch optimizes.
QEC and T-gate workloads saw 1.19x gains; their time includes noise
sampling and array arithmetic that are unaffected by dispatch changes.

---

## Remaining Optimization Opportunities

1. **Dispatch overhead still ~25-35%** — consider merging common opcode
   sequences (e.g. FRAME_CZ + FRAME_CZ) into fused super-instructions.
2. **T-gate array loops not vectorized** — the `if (i & v_bit)` branch
   defeats auto-vectorization. Branchless bit-weaving (like ARRAY_CNOT
   already uses) would help.
3. **Measurement probability loops** — `prob_b0 += norm(arr[i])` could
   use SIMD horizontal reduction.
4. **Per-shot `reset()` reseeds Xoshiro** — 4-word SplitMix expansion
   is cheap but still called 100k times. Counter-based seeding would
   be even cheaper.
