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

## Bytecode Optimization Passes (PR #92)

Four bytecode optimization passes applied post-lowering, pre-execution.
All passes operate on the `vector<Instruction>` bytecode in-place.

### Pass 1: NoiseBlockPass

Collapses runs of individual `OP_NOISE` instructions into a single
`OP_NOISE_BLOCK` that samples a geometric distribution to skip over
non-firing noise sites in bulk. For the d=5 circuit, 3471 noise
instructions collapse to 24 blocks.

### Pass 2: MultiGatePass

Detects star-graph patterns where multiple CNOTs (or CZs) share a
common target (or control) axis. Fuses them into `OP_ARRAY_MULTI_CNOT`
or `OP_ARRAY_MULTI_CZ` using a 64-bit bitmask encoding.
For d=5: 191 individual CNOTs become 44 CNOT + 45 MULTI_CNOT.

### Pass 3: ExpandTPass

Fuses adjacent `OP_EXPAND` + `OP_PHASE_T/T_DAG` on the same axis into
`OP_EXPAND_T/T_DAG`, performing the array duplication and phase rotation
in a single loop. 24 pairs fused in the d=5 circuit.

### Pass 4: SwapMeasPass

Fuses adjacent `OP_ARRAY_SWAP` + `OP_MEAS_ACTIVE_INTERFERE` into
`OP_SWAP_MEAS_INTERFERE`, performing the swap and Born measurement
in a single pass. 15 pairs fused in the d=5 circuit.

### Results (d=5 surface code, 100k shots, RelWithDebInfo -O2 -march=native)

**Default pipeline** (NoiseBlockPass + ExpandTPass + SwapMeasPass):

| Metric | Before (main) | After (optimized) | Change |
|---|---|---|---|
| **Instructions** | 5,111 | 1,625 | 3.15x fewer |
| **Per-shot time** | ~103 us | ~98 us | **~5% faster** |
| **Optimization cost** | -- | 0.7 ms (one-time) | Negligible |

**With MultiGatePass** (opt-in via `UCC_ENABLE_MULTI_GATE=1`):

| Metric | Before (main) | After (all passes) | Change |
|---|---|---|---|
| **Instructions** | 5,111 | 1,518 | 3.37x fewer |
| **Per-shot time** | ~103 us | ~88 us | **~15% faster** |

MultiGatePass is excluded from the default pipeline because its
popcount-based parity branch creates ~50/50 branch mispredictions in
the inner loop (33.9% of runtime when enabled). At rank=10 (d=5), the
dispatch savings still outweigh the misprediction penalty, but at
higher ranks the O(2^k) branch penalty will dominate. It will be
re-enabled once the inner loop is rewritten with branchless AVX2 blends.

### Instruction Profile (1625 total, default pipeline)

| Category | Count | % | Cost |
|---|---|---|---|
| FRAME ops | 764 | 47.0% | ~2 XORs each |
| MEAS_DORMANT | 215 | 13.2% | bit read |
| APPLY_PAULI | 135 | 8.3% | XOR + popcount |
| NOISE_BLOCK | 24 | 1.5% | geometric skip |
| DETECTOR | 107 | 6.6% | parity XOR |
| ARRAY_CNOT | 191 | 11.8% | 2^(k-2) swaps |
| ARRAY_CZ | 6 | 0.4% | 2^(k-2) negations |
| PHASE_T/T_DAG | 48 | 3.0% | 2^(k-1) multiplies |
| EXPAND_T | 24 | 1.5% | 2^(k-1) copy+multiply |
| SWAP_MEAS | 15 | 0.9% | fused swap+Born+fold |
| Other | 96 | 5.9% | misc |

### SwapMeasPass: Single-Pass Memory Fusion

The `OP_SWAP_MEAS_INTERFERE` handler now performs the logical swap and
X-basis fold in a single O(2^k) memory pass, eliminating the redundant
O(2^k) array permutation that separate ARRAY_SWAP + MEAS_ACTIVE_INTERFERE
would require. Each output index is mapped directly to its unswapped
source indices using bit extraction, and the in-place fold is provably
hazard-free without any temporary buffer.

### Hotspot Analysis (perf record, d=5 circuit, with MultiGatePass)

| % of runtime | Operation | Notes |
|---|---|---|
| **33.9%** | MULTI_CNOT inner loop | Naive bit-insertion + unpredictable popcount branch |
| **19.6%** | EXPAND_T inner loop | Already fused, inherent work |
| **8.6%** | ARRAY_CNOT inner loop | Uses PDEP, well-optimized |
| **7.0%** | PHASE_T inner loop | Uses PDEP, well-optimized |
| **5.6%** | MEAS_ACTIVE_INTERFERE | Born probability + fold |
| **~5%** | Frame/meas/detector | O(1) per instruction |
| **<1%** | Dispatch overhead | Computed goto |

---

## Remaining Optimization Opportunities

1. **MULTI_CNOT branchless/PDEP** (33.9% of runtime) — the popcount
   parity branch is ~50/50 unpredictable. A branchless conditional
   swap or restructured index enumeration could help significantly.
2. **AVX vectorization** — the inner loops process one complex<double>
   (16 bytes) at a time. AVX2 could process 2 per cycle.
3. **Measurement probability loops** — `prob_b0 += norm(arr[i])` could
   use SIMD horizontal reduction.
4. **Per-shot `reset()` reseeds Xoshiro** — 4-word SplitMix expansion
   is cheap but still called 100k times. Counter-based seeding would
   be even cheaper.

---

## QV-20 Profile: Post-Fusion (SingleAxisFusionPass)

**Config:** `RelWithDebInfo` (-O2 -g -march=native), 3 shots, perf record -F 9999
**Circuit:** QV-20 (20 qubits, peak_rank=20, 16MB statevector, 2^19 pairs/sweep)
**Bytecode:** 8987 -> 3089 instructions after fusion
**Per-shot:** ~4,784 ms (RelWithDebInfo); ~3,402 ms (Release)

### Top-Level

99.73% of cycles in `ucc::execute()`. Compilation is negligible.

### Hotspot Breakdown by Logical Operation

| % of Runtime | Operation | Source | Notes |
|---|---|---|---|
| **~73.5%** | **exec_array_u2** inner loop | svm.cc:814-822 | 2x2 matrix butterfly sweep |
| **~7.0%** | **exec_array_cz** inner loop | svm.cc:429-431 | v[idx] = -v[idx] negate |
| **~5.3%** | **exec_array_cnot** inner loop | svm.cc:408-410 | std::swap butterfly |
| **~1.8%** | **exec_phase_rot** inner loop | svm.cc:744-745 | arr[i] *= z diagonal |
| **~1.8%** | **scatter_bits** (PDEP) | bmi2intrin.h:71 | Index computation |
| **~7.2%** | Dispatch + misc | svm.cc | Goto, frame ops, measurement |

### exec_array_u2 Assembly Analysis

The compiler emits scalar `vmulsd`/`vfmadd231sd` — one `double` at a time.
It does NOT auto-vectorize to 256-bit AVX despite `-ffast-math`. Each
complex multiply = 4 scalar muls + 2 adds. The loop body per iteration:

| % | What | Instructions |
|---|---|---|
| 29.3% | complex multiply (4x) | vmulsd + vfnmadd/vfmadd |
| 13.3% | Load arr[idx1] | 2x vmovsd from memory |
| 11.2% | complex add (2x) | vaddsd |
| 6.0% | complex __rep() | Register shuffles |
| 6.0% | Store arr[idx1] | 2x vmovsd to memory |
| 4.1% | Loop counter + PDEP | inc + pdep for scatter index |
| 3.5% | Load arr[idx0] | 2x vmovsd from memory |
| 2.9% | Store arr[idx0] | 2x vmovsd to memory |

### Optimization Opportunities

1. **AVX2 SIMD for U2** (~73.5% of runtime) — hand-written `__m256d`
   intrinsics could process real+imag in parallel or batch 2 pairs.
   Expected ~1.5-2x on inner loop, ~1.3-1.5x system.

2. **Branchless CNOT/CZ** (12.3%) — already memory-bandwidth bound but
   AVX prefetching or batched patterns might help.

3. **Axis-locality reordering** — scatter_bits PDEP creates non-sequential
   access. High axis bits = full-array strides. Reordering virtual axes
   so frequently-touched axes get low bit positions improves locality.

4. **Measurement loops** (<2%) — negligible for QV circuits.
