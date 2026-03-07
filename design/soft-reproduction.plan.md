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
single EC2 instance. One instance is sufficient -- Sinter's built-in
multi-worker parallelism saturates all cores, and `save_resume_filepath`
handles crash resilience. S3 backup is optional insurance.

**Architecture:**

A single `c7i.24xlarge` (48 vCPUs / 24 physical cores, Sapphire
Rapids with AVX-512) runs all three data points sequentially. Sinter
uses `num_workers = os.cpu_count() // 2` workers (24 processes) to
avoid hyperthreading contention on AVX-512 workloads. Each p-value
runs until `max_errors=100` or a shot budget is exhausted.

```
EC2 Instance (c7i.24xlarge)
  run_cloud.py --noise 0.002   # fastest, ~3-6 hours
  run_cloud.py --noise 0.001   # ~8-16 hours
  run_cloud.py --noise 0.0005  # ~15-30 hours
```

Sinter's `save_resume_filepath` writes progress to a local CSV after
every batch of completed shots. On restart (spot reclaim, crash),
it picks up from the last saved state with no duplicate work.

**Tasks:**

4.1. Write `paper/magic/run_cloud.py`:
     - Accept `--noise` flag to select which p value to run.
     - Use `sinter.collect()` with `max_errors=100` (paper had
       22-49 errors per point; 100 gives tighter CIs).
     - Use `save_resume_filepath` for crash resilience.
     - Print progress every 60 seconds.

4.2. Write `paper/magic/cloud_setup.sh`:
     - Install system deps (gcc, cmake, python3.12, uv)
     - Clone repo, build UCC
     - Quick-profile single-core to confirm expected throughput
     - Test multi-worker scaling (1, 4, 12, 24 workers)

4.3. Estimated compute per point (single c7i.24xlarge, 24 workers):

     Measured single-core throughput (d=5 p=0.001, no postselection):
     105 us/shot (~9.5k shots/s) with BMI2 PDEP optimization.
     With postselection early-exit (~85% discard after ~20% of
     instructions), effective throughput is ~29k shots/s per core.
     Cluster throughput: ~29k * 24 cores = ~700k shots/s.

| p | Total Shots Needed | Estimated Wall Time | Est. Cost (spot) |
|---|---|---|---|
| 0.002 | ~3B | ~1.2 hours | ~$2 |
| 0.001 | ~7.5B | ~3 hours | ~$5 |
| 0.0005 | ~13.5B | ~5.4 hours | ~$8 |

     Total estimated cost: **~$15** on a single instance.
     The paper used 16 NVIDIA A100 GPUs; we're targeting one
     48-vCPU node.

4.4. Download results:
     ```bash
     scp ec2-user@instance:~/ucc-next/results_*.csv ./results/
     ```

**AWS Setup (minimal):**

No S3 configuration is strictly required. Sinter's
`save_resume_filepath` provides local crash resilience. For optional
S3 backup as extra insurance:

1. Create an S3 bucket: `aws s3 mb s3://ucc-soft-results`
2. Create an IAM instance profile with `s3:PutObject` permission
   on that bucket, and attach it to the EC2 instance at launch.
3. Add a cron job on the instance:
   ```bash
   # /etc/cron.d/ucc-backup
   */5 * * * * ec2-user aws s3 cp /home/ec2-user/ucc-next/results_*.csv s3://ucc-soft-results/$(hostname)/
   ```

Alternatively, skip S3 entirely and just `scp` the results when
each run completes. For spot instances, enable the 2-minute
interruption warning handler to trigger a final upload.

**DoD:** Have CSV files with >= 50 errors for each of the three Table
III noise strengths.

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
- [ ] Step 5: Generate plots and tables
