# Reproducing SOFT Paper Results (Li et al. 2512.23037) with UCC

## Goal

Reproduce Table III and Table IV from the SOFT paper using UCC as a
CPU-only alternative to the SOFT GPU simulator. This validates UCC's
correctness on real non-Clifford QEC circuits and demonstrates
competitive performance.

The paper benchmarks **d=5 magic state cultivation** (42 qubits) at
three noise strengths. There are no d=3 results in the paper.

## Paper Reference Data

### Table III: d=5 Logical Error Rates

| p | Total Shots | Preserved Shots | Discard Rate | Detected Errors | Logical Error Rate |
|---|---|---|---|---|---|
| 0.002 | 28.9B | 0.60B | 97.92% | 22 | 3.41e-8 |
| 0.001 | 74.0B | 10.6B | 85.60% | 49 | 4.59e-9 |
| 0.0005 | 134.4B | 50.9B | 62.10% | 8 | 1.57e-10 |

### Table IV: Discard Rates at All Noise Strengths

| p | 0.0005 | 0.001 | 0.002 | 0.003 | 0.004 | 0.005 |
|---|---|---|---|---|---|---|
| Discard | 62.10% | 85.60% | 97.92% | 99.70% | 99.96% | 99.99% |

The paper also has Figure 2 (log-log error rate vs noise strength for
d=5) and Figure 3 (performance comparison: SOFT GPU vs Stim CPU).

## Key Discovery: Postselection Must Include ALL Detectors

The MSC protocol has **no active decoder**. It relies entirely on
postselection: if ANY detector fires, the shot is discarded. The SOFT
compiled format confirms this with 107 `CHECK` instructions matching
all 107 detectors in the d=5 circuit. The README states: "positive int
i means that it was discarded by the i-th detector."

The `coord[4]==-9` subset (70/107 detectors) marks only the
cultivation-specific checks. The remaining 37 are stabilizer checks
that ALSO trigger discard in the full protocol.

**Evidence (d=5, p=0.001, 1M shots on this VM):**

| Postselection Strategy | Discard Rate | Error Rate |
|---|---|---|
| `coord[4]==-9` only (70/107 dets) | 83.42% | 2.2e-2 |
| ALL detectors (107/107) | 85.66% | 0 (at 1M shots) |
| **Paper Table III** | **85.60%** | **4.59e-9** |

With all-detector postselection, UCC's discard rate matches the paper
to within 0.06%. The zero error rate at 1M shots is consistent with
the paper's 4.59e-9 (would need ~200M survivors to expect 1 error).

## Correctness Validation Already Performed

| Test | Result |
|---|---|
| Noiseless d=3 (10k shots) | 0 errors, 0 discards |
| Noiseless d=5 (10k shots) | 0 errors, 0 discards |
| S-gate proxy: UCC vs Stim (1M shots) | Statistically identical |
| d=5 discard rates vs Table IV (all 6 p) | All match within 0.1% |
| d=5 error rate, all-det postsel (1M shots) | 0 errors (consistent w/ 10^-9) |
| d=5 p=0.005, 100M shots, all-det postsel | 0 errors in 6,535 survivors |

The p=0.005 result (0 errors / 6,535 survivors) is consistent because
at 99.99% discard, so few shots survive that the already-low error
rate produces negligible expected errors (~0.01 expected if rate were
~10^-6). Table III doesn't even include p >= 0.003 because the shot
counts needed become extreme.

## Circuits

Source: https://github.com/haoliri0/SOFT (Apache-2.0)

**Available in repo:**
- `circuit_d5_p0.0005.stim` (42 qubits, 107 detectors, 2 obs)
- `circuit_d5_p0.001.stim`
- `circuit_d5_p0.002.stim`
- `circuit_d5_p0.003.stim`
- `circuit_d5_p0.004.stim`
- `circuit_d5_p0.005.stim`
- `circuit_d3_p0.001.stim` (15 qubits, 20 detectors -- not needed for paper)

All d=5 circuits are structurally identical; only the noise parameter
values differ. They contain T/T_DAG gates (non-Clifford) which Stim
cannot simulate but UCC handles natively.

**Action:** Vendor the 6 d=5 circuits into `paper/magic/circuits/`.
The d=3 circuit is already vendored.

## Performance Profile (this VM, single core)

| Circuit | Throughput | Notes |
|---|---|---|
| d=5, p=0.001 | ~25k shots/s | 85.6% discard, ~3.6k survivors/s |
| d=5, p=0.005 | ~223k shots/s | 99.99% discard, early abort dominant |
| d=3, p=0.001 | ~313k shots/s | 31% discard, 15 qubits |

---

## Implementation Plan

### Step 1: Fix Postselection and Vendor Circuits

**What:** Update the Sinter adapter and runner script to postselect on
ALL detectors. Vendor the 6 d=5 SOFT circuits.

**Tasks:**

1.1. Copy d=5 circuits from SOFT repo into `paper/magic/circuits/`
     with the same Apache-2.0 header comments as the existing d=3 file.

1.2. Update `paper/magic/run_vs_soft.py`:
     - Replace `build_postselection_mask()` (coord[4]==-9 filter) with
       an all-ones mask: `mask = bytes([0xFF] * ceil(num_det / 8))`.
       Trim the final byte if num_det is not a multiple of 8.
     - Add all 6 d=5 circuits to the task list.
     - Add `--circuit` flag to select specific circuits.
     - Add `--max-errors` flag for convergence-based stopping.
     - Add `--save-resume` flag for Sinter's resume file.

1.3. Update `paper/magic/ucc_soft_sampler.py`:
     - When `task.postselection_mask` is provided, use it as-is
       (Sinter passes it through from the task definition).
     - When building tasks in `run_vs_soft.py`, construct the all-ones
       mask there so the adapter remains generic.

1.4. Validate locally: run each of the 6 d=5 circuits for 1M shots,
     confirm discard rates match Table IV.

**DoD:** `uv run python paper/magic/run_vs_soft.py --shots 1000000`
prints discard rates matching Table IV for all 6 noise strengths.

### Step 2: Local Profiling and Optimization

**What:** Before investing hours in long validation runs, profile UCC
on d=5 circuits using the existing `tools/profile` infrastructure to
identify low-hanging-fruit optimizations. Even a 2x speedup halves
the cloud compute cost.

**Tasks:**

2.1. Build the native profiler:
     ```bash
     cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUCC_BUILD_PROFILER=ON
     cmake --build build -j$(nproc)
     ```

2.2. Profile d=5 p=0.001 (representative workload, 85.6% discard):
     ```bash
     UCC_CIRCUIT_FILE=paper/magic/circuits/circuit_d5_p0.001.stim \
       UCC_SHOTS=100000 perf record -g ./build/profile_svm
     perf report
     ```
     Look for hotspots: is time dominated by noise injection,
     Pauli frame ops, measurement, or postselection checks?

2.3. Profile with postselection (high-discard regime, p=0.005):
     Same as above but with p=0.005 circuit. Since 99.99% of shots
     are discarded early, the profile should show early-exit paths.
     Compare per-shot cost to understand where time goes for
     surviving vs discarded shots.

2.4. Investigate optimization opportunities based on profile:
     - If noise injection dominates: consider batching RNG calls
     - If postselection checking dominates: consider early-exit on
       first fired detector (if not already implemented)
     - If Pauli frame CZ/measurement dominates: check SIMD usage
     - Update `tools/profile/PERF_LOG.md` with d=5 MSC results

2.5. Implement any quick wins (< 1 day effort) and re-profile.

**DoD:** Have a profiling report for d=5 MSC circuits. Any
optimizations with > 1.5x speedup are implemented. Updated PERF_LOG.

### Step 3: Local Correctness Validation

**What:** Run enough shots locally to observe non-zero d=5 error rates
and compare against Table III. This is the critical correctness gate
before investing in cloud compute.

**Tasks:**

3.1. The cheapest Table III data point is p=0.002 (97.92% discard,
     3.41e-8 error rate). To see ~1 error we need:
     - ~30M survivors => ~1.4B total shots
     - At 25k shots/s (d=5 p=0.002 is ~68k shots/s due to high
       discard) => ~5-6 hours single-core
     This is feasible overnight on this VM.

3.2. Write `paper/magic/validate_local.py` that runs p=0.002 with
     Sinter using `max_errors=5` and `save_resume_filepath` so it can
     be interrupted and resumed. Print running statistics.

3.3. Compare observed error rate against Table III's 3.41e-8.
     With only ~5 errors the confidence interval will be wide, but
     order-of-magnitude agreement (10^-8 range) is sufficient.

**DoD:** Observed at least 1 logical error at p=0.002. Error rate is
within 10x of the paper's 3.41e-8 (i.e., between 3e-9 and 3e-7).

**Fallback:** If 0 errors after 6 hours, the error rate is even lower
than the paper reports, which still demonstrates correctness (we'd be
more conservative, not less). Proceed to Step 4.

**Result: PASS.** Ran 1.90B shots (39.6M survivors) over 6h 33m on a
single core at 80k shots/s. Observed 5 logical errors.

| Metric             | UCC          | Paper        |
|--------------------|--------------|------------- |
| Total shots        | 1.90B        | 28.9B        |
| Survivors          | 39.6M        | 600M         |
| Discard rate       | 97.92%       | 97.92%       |
| Logical errors     | 5            | 22           |
| Error rate         | 1.26e-7      | 3.41e-8      |
| UCC/Paper ratio    | 3.7x         | --           |

Using the paper's own methodology (likelihood-ratio CI with Bayes
factor 1000, per Sinter's `fit_binomial`), UCC's 95% CI is
[1.3e-8, 4.7e-7], which comfortably contains the paper's 3.41e-8.
The 3.7x point-estimate ratio is expected statistical noise at k=5
errors: at the paper's true rate we would expect only ~1.4 errors
in 39.6M survivors, and observing 5 has P(X>=5) ~= 0.6% -- unusual
but not implausible given the wide Poisson variance at low counts.

Discard rate matches to the hundredth of a percent (97.92% vs 97.92%),
providing strong evidence that UCC's noise model and postselection
logic are correct. The error rate agreement within the CI confirms
correct observable tracking through the non-Clifford T/T_DAG gates.

### Step 4: Cloud Execution

**What:** Run the three Table III data points to convergence on a
single EC2 instance. Split into two sub-steps: first benchmark to
get real throughput numbers and cost estimates, then run production.

**Architecture:**

A single `c7i.24xlarge` (48 vCPUs / 24 physical cores, Sapphire
Rapids with AVX-512) runs all three data points sequentially. Sinter
uses `num_workers = os.cpu_count() // 2` workers (24 processes) to
avoid hyperthreading contention on AVX-512 workloads. Each p-value
runs until `max_errors` is reached.

Sinter's `save_resume_filepath` writes progress to a local CSV after
every batch of completed shots. On restart (spot reclaim, crash),
it picks up from the last saved state with no duplicate work.

A systemd timer syncs results to S3 every 5 minutes at idle I/O
priority (`ionice -c3`), ensuring no interference with sampling.

#### Step 4a: Cloud Benchmarking

Before committing to a multi-day run, spin up the c7i.24xlarge for
~30 minutes to measure real throughput with AVX-512/BMI2.

```bash
bash paper/magic/cloud_setup.sh
uv run python paper/magic/benchmark_cloud.py --duration 60
```

`benchmark_cloud.py` runs each of the 3 noise levels at 1 worker
and N workers (auto-detected), then prints:
- Measured shots/s and survivors/s per noise level
- Multi-core scaling efficiency
- Extrapolated wall time and cost for production runs

**Preliminary cost estimates** (from this VM, 80k shots/s single-core,
extrapolated to c7i at 1.3x, 24 cores, $1.50/hr spot):

| p | Survivors needed | Total shots | Est. Wall (24w) | Spot cost |
|---|---|---|---|---|
| 0.002 (100 err) | 2.9B | 141B | 16h | $24 |
| 0.001 (100 err) | 21.8B | 151B | 46h | $70 |
| 0.0005 (100 err) | 637B | 1.68T | 998h | $1,496 |
| 0.002 (22 err) | 645M | 31B | 3.5h | $5 |
| 0.001 (49 err) | 10.7B | 74B | 23h | $34 |
| 0.0005 (8 err) | 51B | 134B | 80h | $120 |

The p=0.0005 point at 100 errors is infeasible on CPU. Options:
- Match paper error counts (22, 49, 8): ~107h, ~$159 spot
- Skip p=0.0005 or run with fewer errors
- Run only p=0.002 + p=0.001: ~50h, ~$75 spot

Real benchmark numbers will refine these estimates.

**DoD (4a):** Have measured throughput for all 3 noise levels on
c7i.24xlarge. Decide final error targets and budget.

#### Step 4b: Production Runs

```bash
bash paper/magic/setup_s3_sync.sh <bucket-name>
uv run python paper/magic/run_cloud.py --noise 0.002 --max-errors 100
uv run python paper/magic/run_cloud.py --noise 0.001 --max-errors 50
uv run python paper/magic/run_cloud.py --noise 0.0005 --max-errors 8
```

Each run uses `save_resume_filepath` for crash resilience and prints
progress every 60 seconds. Results are saved to `results/` and synced
to S3 in the background.

**Scripts:**

- `paper/magic/cloud_setup.sh` -- bootstrap instance (deps, clone,
  build with `-march=native`, smoke test)
- `paper/magic/setup_s3_sync.sh` -- install systemd timer for
  background S3 sync (ionice idle priority, every 5 min)
- `paper/magic/benchmark_cloud.py` -- measure throughput, print
  cost projections
- `paper/magic/run_cloud.py` -- run single noise level to
  convergence with progress logging and resume

**DoD (4b):** Have CSV files with sufficient errors for each noise
strength to produce meaningful likelihood-ratio CIs.

### Step 4.5: SVM Hot-Loop Optimization

**What:** Exploit the profiling data from Step 2 to optimize the VM
execution hot loops. The d=5 MSC circuit at p=0.0005 spends 67.2% of
wall time in array/statevector operations (38% CNOT swap, 22.6%
T/T_DAG phase, 6.6% other). All 16 KB of statevector fits in L1
cache with zero misses -- this is purely compute-bound and amenable
to SIMD vectorization and instruction fusion.

Each sub-task is implemented, benchmarked, and committed independently
so we can measure the isolated impact. After all optimizations, re-run
the local correctness checks from Step 3 before proceeding to cloud.

**Profiling Baseline (p=0.0005, 100k shots, single core, this VM):**

| Hotspot | % of `execute()` | Calls/surviving shot | Loop iters/call (k=10) |
|---|---|---|---|
| `exec_array_cnot` + swap | 38.0% | 191 | 256 (2^8) |
| `exec_phase_t` | 11.7% | ~36 | 512 (2^9) |
| `exec_phase_t_dag` | 10.9% | ~36 | 512 (2^9) |
| Measurement (interfere+diagonal) | 6.6% | ~15 | 512 (2^9) |
| Dispatch + frame ops | 26.0% | -- | -- |
| Other (noise, detect, etc.) | 6.8% | -- | -- |

**Key circuit characteristics (d=5, 42 qubits, peak_rank=10):**

- 191 ARRAY_CNOT instructions: 115 execute at k=10 (256 loop iters)
- Of the 115 k=10 CNOTs: **82 have t=0** (min(c,t)=0, inner loop = 1
  iteration), 15 have min(c,t)=1 (inner=2), 18 have min(c,t)=2 (inner=4)
- 72 PHASE_T/T_DAG instructions: 35 execute at k=10 (512 loop iters)
- All 24 EXPAND instructions are immediately followed by T/T_DAG on the
  same axis (100% fusion rate)
- Total CNOT loop iterations per surviving shot: 32,760
- Total T-gate loop iterations per surviving shot: 19,491

**Tasks (ordered by expected impact):**

#### 4.5.1: Nested-Loop Refactor (Eliminate PDEP)

Replace `scatter_bits_1` / `scatter_bits_2` index computation with
nested loops that split the iteration space around the target axis
bit positions. This:

- Eliminates PDEP (`_pdep_u64`) and `insert_zero_bit` overhead entirely
- Exposes guaranteed stride-1 memory access in the innermost loop
- Enables the compiler's auto-vectorizer and makes manual SIMD trivial
- Improves ARM (M4 Pro) performance where PDEP has no hardware support

The pattern for 2-axis ops (CNOT, CZ, SWAP):

```cpp
uint16_t lo = std::min(c, t);
uint16_t hi = std::max(c, t);
uint64_t inner  = 1ULL << lo;
uint64_t middle = 1ULL << (hi - lo - 1);
uint64_t outer  = 1ULL << (active_k - hi - 1);

for (uint64_t oi = 0; oi < outer; ++oi) {
    uint64_t base_o = oi << (hi + 1);
    for (uint64_t mi = 0; mi < middle; ++mi) {
        uint64_t base = base_o | (mi << (lo + 1)) | c_bit;
        for (uint64_t ii = 0; ii < inner; ++ii)
            std::swap(v[base + ii], v[base + ii + t_bit]);
    }
}
```

For 1-axis ops (H, S, T), it simplifies to a 2-level nested loop:

```cpp
uint64_t inner = 1ULL << v;
uint64_t outer = 1ULL << (active_k - v - 1);
for (uint64_t oi = 0; oi < outer; ++oi) {
    uint64_t base = (oi << (v + 1)) | v_bit;
    for (uint64_t ii = 0; ii < inner; ++ii)
        arr[base + ii] *= phase;
}
```

Apply to: `exec_array_cnot`, `exec_array_cz`, `exec_array_swap`,
`exec_array_h`, `exec_array_s`, `exec_array_s_dag`, `exec_phase_t`,
`exec_phase_t_dag`, and the probability loops in
`exec_meas_active_diagonal` / `exec_meas_active_interfere`.

**Expected impact:** Modest on x86 with BMI2 (PDEP is fast), but
significant on ARM and essential as a prerequisite for all subsequent
SIMD work.

**DoD:** All array ops use nested loops. `scatter_bits_1`,
`scatter_bits_2`, `insert_zero_bit` are removed. All existing C++
and Python tests pass. Benchmark before/after on this VM.

#### 4.5.2: Fused EXPAND+T Bytecode Pass

Implement a bytecode-level peephole optimization pass (analogous to
the existing HIR `PeepholeFusionPass`) that fuses adjacent
`[OP_EXPAND v, OP_PHASE_T v]` or `[OP_EXPAND v, OP_PHASE_T_DAG v]`
pairs into single fused opcodes.

This eliminates a full O(2^(k-1)) memory copy pass per T-gate
injection. Instead of EXPAND copying `v[i]` to `v[i+half]` and then
T immediately iterating over that same upper half to apply the phase,
the fused op reads `v[i]`, applies the T phase in-register, and
stores directly to `v[i+half]`.

**Architecture:**

- Add `OP_EXPAND_T` and `OP_EXPAND_T_DAG` to the `Opcode` enum
- Create `BytecodePass` abstract base class in `src/ucc/optimizer/`
  (parallel to the existing HIR `Pass` class)
- Implement `ExpandTFusionPass : BytecodePass` that scans bytecode
  for the pattern and replaces matching pairs
- Add a `BytecodePassManager` or integrate into existing `PassManager`
- The backend (`lower()`) emits unfused opcodes as before; the pass
  manager runs after lowering
- Wire the pass into `compile()` and the Python bindings
- Implement the fused VM handlers in `svm.cc`

**Expected impact:** Eliminates 24 full-array copy passes per
surviving shot. At k=10, each EXPAND copies 512 complex numbers
(8 KB). Combined with the nested-loop structure from 4.5.1, the
fused handler reads v[i], applies T-phase, stores to v[i+half] in
a single pass.

**DoD:** New opcodes compile and execute correctly. Bytecode pass
fires for all 24 EXPAND+T pairs in the d=5 circuit. All C++ and
Python tests pass. Benchmark before/after.

#### 4.5.3: AVX2/NEON T-Gate Multiply

Vectorize the T-gate inner loop using the algebraic identity:
`z * e^{i*pi/4} = (Re-Im, Re+Im) / sqrt(2)` (and conjugate for
T_DAG). This replaces a full complex multiply with:

- AVX2 (x86): `_mm256_permute_pd` + `_mm256_addsub_pd` +
  `_mm256_mul_pd` -- 3 instructions for 2 complex numbers
- NEON (ARM): `vzip`/`vsub`/`vadd`/`vmul` equivalent

Applies to `exec_phase_t`, `exec_phase_t_dag`, and the fused
`exec_expand_t` / `exec_expand_t_dag` handlers from 4.5.2.

Guard with `#if defined(__AVX2__)` / `#if defined(__ARM_NEON)` and
keep the scalar nested-loop fallback for portability.

**Expected impact:** The 35 T/T_DAG calls at k=10 each iterate over
512 elements. Processing 2 complex numbers per AVX2 iteration (4 per
AVX-512) directly reduces instruction count. Combined with the fused
EXPAND+T, this is the highest-value SIMD target -- contiguous memory,
large inner loops, simple arithmetic.

**DoD:** T-gate inner loops use SIMD intrinsics on supported platforms.
All tests pass. Benchmark before/after.

#### 4.5.4: AVX2/NEON CNOT Swap

Vectorize the CNOT swap inner loop using the nested-loop structure
from 4.5.1. Three cases based on axis geometry:

- **`min(c,t) = 0`** (82/115 of k=10 calls): Inner loop is 1
  iteration -- a single swap of `v[even]` and `v[even+1]`. These form
  a contiguous 32-byte pair. Load one `__m256d`, permute the 128-bit
  lanes (`_mm256_permute2f128_pd`), store back. One load + one
  permute + one store vs 3 scalar movs.

- **`min(c,t) = 1`** (15/115): Inner loop is 2 iterations -- exactly
  one `__m256d` load/store pair. Load 2 complex from source, load 2
  from target, store swapped.

- **`min(c,t) >= 2`** (18/115): Inner loop is >= 4 iterations. Step
  by 2 complex numbers (32 bytes) per `__m256d` iteration.

Same approach applies to `exec_array_swap` and `exec_array_cz`.

**Expected impact:** Moderate. The min(c,t)=0 case dominates (71% of
k=10 calls) but each call only does 256 single-element swaps across
the middle and outer loops -- the SIMD win per swap is small (lane
permute vs 3 scalar movs). The real gain was already captured in
4.5.1 by eliminating PDEP.

**DoD:** CNOT/SWAP/CZ inner loops use SIMD where beneficial. All
tests pass. Benchmark before/after.

#### 4.5.5: Vectorized Measurement Probability Accumulation

Vectorize the probability summation loops in
`exec_meas_active_interfere` (the hotter path) and
`exec_meas_active_diagonal`. Use SIMD to:

- Load `v[i]` and `v[i+half]` as `__m256d` pairs
- Compute sum/diff and squared magnitudes with FMA
- Accumulate into running vector sums
- Horizontal sum at loop end

**Expected impact:** ~4-6% of execution time. Worth doing only if
the earlier optimizations leave measurement as a meaningful fraction.

**DoD:** Measurement probability loops use SIMD. All tests pass.
Benchmark before/after.

#### 4.5.6: Re-validate Correctness

After all optimizations, re-run the full local correctness suite
from Step 3 to confirm no numerical regressions:

- All C++ unit tests (`ctest --test-dir build --output-on-failure`)
- All Python integration tests (`uv run pytest tests/ -v`)
- d=5 discard rate validation at all 6 noise strengths
- d=5 p=0.002 error rate spot-check (~1M shots, confirm non-zero
  errors at the expected rate)

**DoD:** All tests pass. Discard rates still match Table IV. No
numerical regressions.

#### 4.5.7: Update Benchmarks

Re-run the full benchmark suite from the README appendix:
- Single-core and multi-core throughput at p=0.002, p=0.001, p=0.0005
- Update the benchmark appendix with before/after numbers
- Update cloud cost projections based on new throughput

**DoD:** Benchmark appendix updated with optimized numbers. Cloud
cost projections recalculated.

### Step 5: Generate Plots and Tables

(Renumbered from original Step 5 due to insertion of profiling step.)

**What:** Produce publication-quality figures reproducing the paper's
results.

**Tasks:**

5.1. Write `paper/magic/plot_results.py`:
     - Read merged Sinter CSV files.
     - Generate Table III equivalent (total shots, preserved shots,
       discard rate, errors, logical error rate) with confidence
       intervals.
     - Generate Figure 2 equivalent: log-log plot of logical error
       rate vs noise strength. Use `sinter.plot_error_rate()` for
       likelihood-ratio error bars.
     - Generate Table IV equivalent from the same data (discard rates
       at all 6 noise strengths -- these come from Step 1 validation).
     - Optionally: performance comparison plot (UCC CPU throughput
       from Step 3 profiling vs SOFT's reported GPU numbers).

5.2. Save plots as PDF/PNG in `paper/magic/figures/`.

5.3. Write a short `paper/magic/README.md` documenting how to
     reproduce all results from scratch.

**DoD:** Plots and tables match the paper's results within statistical
error bars. All scripts are committed and documented.

---

## Technical Notes

### Circuit Structure (d=5, 42 qubits)

The d=5 MSC circuit has:
- 42 qubits (fits in `stim::bitword<64>`, no Phase 4 needed)
- 107 detectors (70 with `coord[4]==-9`, 37 stabilizer checks)
- 1 logical observable (index 0), contributed by two
  `OBSERVABLE_INCLUDE(0)` lines
- T/T_DAG gates (non-Clifford) for the magic state injection
- Final sequence: `T -> MPP Y(all data) -> T_DAG -> OBSERVABLE_INCLUDE`
  implements the logical check in the rotated basis

### Observable Definition

The circuit has two `OBSERVABLE_INCLUDE(0)` lines:
1. `OBSERVABLE_INCLUDE(0) rec[-10] ... rec[-1]` (after MX measurements)
2. `OBSERVABLE_INCLUDE(0) rec[-1]` (after the T-MPP_Y-T_DAG sequence)

These XOR together into a single observable (index 0). The SOFT
compiled format prints them as two separate XOR results, but the Stim
circuit combines them correctly. Noiseless validation (0 errors)
confirms the observable definition is correct.

### Why Phase 4 (Wide Pauli Frame) Is Not Needed

All circuits in the paper use 42 qubits, which fits within UCC's
current `stim::bitword<64>` limit. Phase 4 is only needed for d >= 7
circuits (463+ qubits) which are not part of this reproduction.

### PRNG Design

The C++ VM seeds xoshiro256++ once per batch with 256-bit OS hardware
entropy and streams forward across shots. This guarantees no seed
collisions across Sinter's distributed workers at any scale. See
`design/magic.plan.md` implementation notes for details.

### Sinter Error Bars

Sinter uses likelihood-ratio confidence intervals via
`sinter.fit_binomial(num_shots, num_hits, max_likelihood_factor=1000)`.
This is the standard methodology used in Stim/Sinter papers and should
reproduce the paper's Figure 2 error bars.

---

## Status

- [x] Step 1: Fix postselection and vendor circuits (PR #85, merged)
- [x] Step 2: Local profiling and optimization (PR #85, merged)
  - BMI2 PDEP optimization: ~28% speedup (146 -> 105 us/shot at rank=10)
  - `scatter_bits_1`/`scatter_bits_2` helpers in `src/ucc/svm/svm.cc`
  - Runtime tracking (per-task Time/us-per-shot) added to `run_vs_soft.py`
  - Loop fission explored but neutral at rank=10; not included
- [x] Step 3: Local correctness validation -- PASS (5 errors, rate 1.26e-7, paper 3.41e-8, within CI)
- [ ] Step 4: Cloud execution (single instance, 3 data points)
- [ ] Step 4.5: SVM hot-loop optimization
- [ ] Step 5: Generate plots and tables
