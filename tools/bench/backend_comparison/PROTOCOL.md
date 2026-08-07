# Legacy versus symbolic-coordinate sampling baseline

This is the proposed measurement matrix and protocol for the scalar
symbolic-coordinate baseline. It is measurement work only: it does not change
the default backend or any production implementation.

## Adaptive execution gates

The complete matrix below is the eventual validation protocol, not a
prerequisite for initial attribution. The first pass uses three balanced blocks
on seven triage cells:

| ID | Reason |
|---|---|
| `surface_d7_r7_aggregate` | representative surface-code workload |
| `cultivation_d3_aggregate` | representative cultivation workload |
| `distillation_aggregate` | representative distillation workload |
| `coherent_d3_r3_aggregate` | representative coherent workload |
| `regime_k12_l512` | controlled dense/long-stream diagnostic |
| `cultivation_d3_k0` | importance-sampling capability sentinel |
| `noncomp_d17_r5_low_leak` | noncomputational component sentinel |

Three blocks give six process-level samples per backend and are used only to
screen for clear, material gaps. Close results and numbers intended for final
reporting expand to five blocks, giving ten samples per backend. The two
largest material execution regressions and largest compile gap are profiled
immediately after triage.

Each profile reports the dominant cost and its measured share, likely
mechanism, approximate Amdahl upper bound, smallest plausible intervention,
and exact before/after benchmark. The attribution and proposed intervention
are posted for review before any production optimization is implemented. If an
approved optimization succeeds on the triage set, the remaining paper matrix,
raw-output subset, EXP_VAL, importance, noncomputational, QV-20, and coherent
d5 r5 cells become broader validation.

## Source identity

- Clifft: `3fdafa411e8e4eca812ff17df2a8f30c584fdc03`.
- Paper corpus: `unitaryfoundation/clifft-paper` commit
  `db7dc9f13a2c2854690e92390c779048a1ac1400`.
- Output-relevant regime-map generator: corrected attribution branch
  `paper/2607.28600-symft-cpu-bench` at
  `adfe94a51e0819d9a2a917f8c932addea640e8c2`.
- Local Clifft fixtures: `tests/fixtures/qv10.stim` and
  `tools/bench/fixtures/qv20_seed42.stim` from the Clifft commit above.

Every raw result records the SHA-256 digest of the exact circuit text. The
paper circuit SHA-256 digests at the pinned corpus commit are:

| Circuit | SHA-256 |
|---|---|
| surface d7 r7 | `30d1940101d70e05a63f0d2f877756ffaeaba7e8e17e94cd2ea40fce04b99583` |
| cultivation d3 | `90a7d841e003e5ee38137cd9a3eb6529bb552e49c424bc6b0932a27d97cdb41f` |
| cultivation d5 | `c2b4566917bd9bf27a5705284dac02700ef0dcc7c03c91066670db376d633a6d` |
| distillation | `188bd53c48dbc21f840fb297df6f41c61f5bad6a856bba621f00ff42078921c1` |
| coherent d3 r1 | `9b439238478b15977829c1015ee47dfe401976b3882092d9560ba321fb0f510a` |
| coherent d3 r3 | `87d1308c83894e87c60aeb2dc31b74be89b3460a951929951f2c3ac92606827d` |
| coherent d5 r1 | `2707188abe8912f693fe4f910db8c4b6bd795c71a8fba38cc153da90ee5910b8` |
| coherent d5 r5 | `54088bbd5f06b441596e414f9fa99d8eeaff8a4a1a862b911e0ad40092c5e549` |

## Matched compilation and output contracts

All ordinary and importance-sampling cells compile the identical source text
with the default HIR pass manager and `normalize_syndromes=True`. Legacy also
runs its default bytecode passes; that backend-specific lowering is part of
legacy compile time. Both backends use the same postselection mask, expected
detector and observable normalization, shots, seed multiset, thread count,
and output mode.

Primary QEC aggregate cells postselect every detector and call
`sample_survivors(..., keep_records=False)`. The separate raw-record subset
uses no postselection and calls `sample()`, retaining every measurement,
detector, observable, and expectation-value result. Controlled diagnostics
have no postselection and use counts-only `sample_survivors()`, except QV and
EXP_VAL, which use raw `sample()` so their outputs are retained and checked.

### Primary paper workloads

| ID | Output | Shots per process |
|---|---|---:|
| `surface_d7_r7_aggregate` | all-detector survivors | 200,000 |
| `cultivation_d3_aggregate` | all-detector survivors | 500,000 |
| `cultivation_d5_aggregate` | all-detector survivors | 50,000 |
| `distillation_aggregate` | all-detector survivors | 200,000 |
| `coherent_d3_r1_aggregate` | all-detector survivors | 1,000,000 |
| `coherent_d3_r3_aggregate` | all-detector survivors | 300,000 |
| `coherent_d5_r1_aggregate` | all-detector survivors | 15,000 |
| `coherent_d5_r5_aggregate` | all-detector survivors | 1, extended run |
| `surface_d7_r7_raw` | unpostselected raw records | 50,000 |
| `cultivation_d3_raw` | unpostselected raw records | 200,000 |
| `coherent_d3_r3_raw` | unpostselected raw records | 200,000 |

### Controlled diagnostics

The regime fixtures are generated deterministically from the corrected
output-relevant generator. Every 16-qubit Clifford/noise block contributes to
observable 0, and the active core also contributes to observable 0. The
selected short and long streams separate dense intercept from stream cost.

| ID | Output | Shots per process |
|---|---|---:|
| `regime_k4_l32` | aggregate | 2,000,000 |
| `regime_k4_l512` | aggregate | 150,000 |
| `regime_k8_l32` | aggregate | 400,000 |
| `regime_k8_l512` | aggregate | 120,000 |
| `regime_k12_l32` | aggregate | 25,000 |
| `regime_k12_l512` | aggregate | 20,000 |
| `qv10_raw` | raw records | 10,000 |
| `qv20_raw` | raw records, extended dense-state run | 1 |

The EXP_VAL controls are deterministic fixtures: the existing 20-qubit,
200-probe peak-rank-zero case and an 8-qubit, 200-probe case with eight live
non-Clifford coordinates at the probes.

| ID | Output | Shots per process |
|---|---|---:|
| `exp_val_k0_200_raw` | raw records and probe values | 500,000 |
| `exp_val_k8_200_raw` | raw records and probe values | 15,000 |

### Capability sentinels

Importance cells use the paper cultivation circuits, all-detector
postselection, counts-only output, and `sample_k_survivors()`.

| ID | Forced k | Shots per process |
|---|---:|---:|
| `cultivation_d3_k0` | 0 | 400,000 |
| `cultivation_d3_k1` | 1 | 400,000 |
| `cultivation_d3_k2` | 2 | 500,000 |
| `cultivation_d5_k0` | 0 | 10,000 |
| `cultivation_d5_k1` | 1 | 20,000 |
| `cultivation_d5_k2` | 2 | 25,000 |

Noncomputational cells reuse the existing deterministic d17 r5 repetition
code, classifier, and models. Both return and check the complete raw result.

| ID | Model | Shots per process |
|---|---|---:|
| `noncomp_d17_r5_lossless` | no transitions | 500,000 |
| `noncomp_d17_r5_low_leak` | S-hook leakage/loss at p=0.01 | 1,000 |

A continuation-heavier model is not in the initial matrix. It will be added
only if the low-leak result or profile shows continuation work is material.

## Process protocol

- One Release build, native CPU baseline, with exact compiler path/version,
  CMake cache, compile flags, loaded extension, Clifft version, and SVM ISA
  recorded. Profiling uses a second Release build that preserves the same
  optimization, native-ISA, and fast-math flags while adding debug symbols and
  frame pointers.
- `OMP_NUM_THREADS=1`, `clifft.set_num_threads(1)`, and one process pinned to
  CPU 3. Topology, affinity, governor availability, virtualization, OS, and
  relevant thread/ISA environment variables are captured once per run.
- Every timing sample is a fresh worker process. It compiles once, records
  compile time and static metadata, warms the selected API, then times one
  fixed-shot call. Noncomputational APIs compile internally, so their timed
  call is explicitly labeled as combined compilation and execution. Outputs
  are consumed into shape, count, and checksum summaries before the process
  exits.
- Three balanced blocks per initial triage cell, expanding close or final cells
  to five blocks. Even blocks run legacy/symbolic/symbolic/legacy; odd blocks
  run symbolic/legacy/legacy/symbolic. Mirrored positions use two new seeds per
  block: ABBA runs A(seed0), B(seed0), B(seed1), A(seed1), while BAAB mirrors
  that assignment. This gives six initial or ten expanded process-level
  samples per backend with paired seed multisets. Checksums prove that outputs
  were consumed; they are not required to match because backend RNG schedules
  may legitimately differ.
- Before cross-backend cells, run the same surface aggregate call as legacy A
  versus legacy A under the identical balanced schedule. This estimates the
  machine and harness noise floor without a code change. If its paired ratio
  envelope exceeds roughly +/-5 percent or shows drift, add an unchanged-code
  k=12, L=512 dense A/A control before interpreting backend gaps.
- Run all mandatory cells serially. Run coherent d5 r5 and QV-20 only after
  the main matrix and label them extended. No SOFT, Stim, tsim, random-circuit,
  or dashboard comparison is part of this run.

## Raw data and analysis

The incrementally written JSON records the full manifest, environment,
circuit IDs/sources/hashes, backend and output contract, compile and sample
times, attempted and accepted throughput, discard fraction, logical and
observable counts, output summaries, peak RSS, and available program metadata
including legacy peak rank/instruction counts and symbolic action counts.
A scratch C++ metadata extractor calls `plan_sampling()` directly, without a
production binding change, to record symbolic initial/max active width, width
at EXP_VAL actions, action-class counts, expression-term counts, and predicted
dense passes. The active EXP_VAL sentinel must assert that every probe executes
with eight live coordinates.

Peak RSS from process rusage is labeled as whole-process peak, including
imports, compilation, warmup, and execution. It is useful as a screening
signal only; an interesting difference receives a phase-isolated memory
follow-up before being attributed to executor storage.

Report medians with IQR, MAD, min/max, and paired backend ratios. Compare the
A/A ratio distribution first, and compare discard and conditional logical
rates with uncertainty intervals. Profile only the largest representative
gaps that exceed that noise floor and have material absolute cost, separating
compile and execute gaps. Cap the first pass at three execution gaps plus the
largest compile gap.

Use `/usr/lib/linux-tools-6.8.0-106/perf`, which has been verified on this VM
for hardware-counter statistics, call-graph recording, and report decoding
despite the `/bin/perf` kernel-package wrapper mismatch. Collect at least
cycles, instructions, IPC, branches, branch misses, and cache events with
`perf stat`; use `perf record` and flat/call-graph reports for attribution.
Before accepting profiles, compare symbolic/legacy ratios from the
Release-with-symbols build against the canonical Release build on surface,
cultivation, and k=12 dense representatives. No production optimization is
made before review of the baseline and profiles.
