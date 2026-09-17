# Structured residual states: first diagnostic results

This is the first measurement milestone for
[the representation and execution study](https://github.com/unitaryfoundation/clifft/issues/495),
not a completed backend evaluation. **One-round circuits are excluded from
the practical corpus and recommendations.** The main repeated-round target
here is coherent-noise surface-code memory at distance 5 for five rounds.

The current evidence does not justify implementing either a product-component
backend or a straightforward exact MPS replacement for that target. Cultivation
also needs a more specific mechanism than independent components. This is not
a rejection of tensor methods with different orderings or Clifford frames.

No production planner, executor, public API, or circuit semantics change in
this experiment. The new tools are opt-in and consume the existing semantic
plan after the production HIR passes. They do not parse human-facing inspection
strings or discover topology inside hot execution.

## Product-component diagnostic

Track a partition of active coordinates. Promotion adds an independent
one-coordinate factor. A rotation merges every factor touched by its Pauli.
A measurement merges touched factors, removes its planned pivot, and compacts
the remaining coordinate indices. A Pauli expectation factors without merging.
An unknown initial residual starts monolithic. There is no speculative split
after a collapse. Instruments and continuations explicitly decline analysis.

This is conservative structural separability in Clifft's current frame, not
physical-qubit separability. Pauli signs can depend on faults and measurement
outcomes without invalidating the partition. It reuses the premise of
[the existing product-component investigation](https://github.com/unitaryfoundation/clifft/issues/314).

The visit model uses `predicted_dense_passes` times the larger input/output
dimension, versus the touched component dimension. Each merge additionally
charges one output write per tensor-product coefficient. It excludes fusion,
merge-input reads, dispatch, SIMD efficiency, classical work, scratch, allocation,
and rejected-shot lifetimes. Thus its ratios **are not speedups or rigorous
runtime bounds**. Live coefficient counts exclude the old factors retained
while forming a merge, allocator retention, and scratch; they are not peak RSS.

| Workload | Peak active width | Largest component | Dense / factored peak live coefficients | Visit ratio including merge outputs |
| --- | ---: | ---: | ---: | ---: |
| Cultivation d3, p=0.001 | 4 | 4 | 16 / 16 | 1.004 |
| Cultivation d5, p=0.001 | 10 | 10 | 1,024 / 1,024 | 1.023 |
| Surface d7, seven rounds | 0 | 0 | 1 / 1 | no dense work |
| Five-block color-code distillation | 5 | 4 | 32 / 16 | 2.067 |
| Coherent surface d3, three rounds | 5 | 5 | 32 / 32 | 1.229 |
| Coherent surface d5, five rounds | 13 | 13 | 8,192 / 8,192 | 1.329 |
| Quantum Volume 10 | 10 | 10 | 1,024 / 1,024 | 1.099 |
| Quantum Volume 20 | 20 | 20 | 1,048,576 / 1,048,576 | 1.039 |

The favorable distillation case is already tiny. Its reduced coefficient work
does not by itself justify backend complexity. The five-round coherent case
retains some local-work savings but loses the large storage benefit as factors
merge. A conservative product implementation should not be selected on peak
physical-qubit count or on one-round results.

Two generated controls isolate the distinction: twelve independent magic
states need 24 factored coefficients instead of 4,096; adding a shared Pauli
rotation makes this conservative model merge all twelve. These controls have
no terminal measurements: measurement-aware production scheduling otherwise
reduces their active widths before a representation change is needed.

## Residual Schmidt ranks

The native tool creates validated semantic-plan prefixes at up to sixteen
evenly spaced dense-action checkpoints, with four sampled states per prefix.
It removes output probes/postselection, preserves feedback and readout noise,
and fills unneeded suffix record slots with constants. It then uses the
unchanged production executor. These are **independently sampled unconditional
prefixes**, not four continuous trajectories or accepted-shot ensembles.
Snapshot construction and SVD are outside all reported timing intervals.

NumPy evaluates Schmidt spectra across every contiguous cut in ascending
active-coordinate order. No coefficient or singular value is truncated in
execution. Numerical rank is reported at relative thresholds 1e-8, 1e-10,
1e-12, and 1e-14; these observations neither prove exact algebraic rank nor
bound all fault patterns. All snapshot norms and structurally predicted
product partitions are checked independently with linear algebra.

| Case | Largest observed numerical rank at 1e-12 | Interpretation |
| --- | ---: | --- |
| Independent magic-state control | 1 | Product diagnostic behaves as intended |
| Shared-rotation control | 2 | Product factors can merge while MPS rank stays small |
| Coherent d5, five rounds | 64 | Later residuals reach the maximum rank of a balanced cut through 12 coordinates |
| Local corrected cultivation d5 fixture | 32 | Some sampled cuts reach full rank at width 10 |
| Flagged d5a19f13, noise-free file | 15 | Some compressibility, but far from a constant-sized logical residual in this ordering |
| Flagged d5a19f13, p=0.001 | 22 | Noiseless profiles do not capture all observed noisy ranks |

For five-round coherent memory the largest rank is 58 even at 1e-8, and 64
at all three tighter thresholds. For the other cases the maxima are unchanged
across the four thresholds. The maximum product-partition tail norm is below
9e-15 in these samples. This validates the observed factorization, not every
possible circuit or fault branch.

An MPS implementation must pay for applying gates, measurement, ordering
changes, and exact factorization. Ranks near the dense limit weaken its case
at these small active widths. These measurements leave open improved
coordinate orderings, compiler-selected disentangling Cliffords, other tensor
topologies, and shot-dependent frames.

## Compilation granularity

Timings below are indicative scale checks on this VM, not competitive benchmark
claims: GCC 13.3, Release/native, OpenMP disabled, one scalar executor, seed 42,
one untimed warmup shot, median of three independent tool invocations. Practical
cases use 32 attempted shots per invocation and postselect every detector.
No CPU affinity is imposed; short microsecond timings and compile timings vary
with VM scheduling. Executor allocation is reported separately in the data;
the shot interval excludes construction and public API/result-collection costs.

| Case | Parse + compile + lower | Hot attempted shot | Compile cost / hot-shot cost |
| --- | ---: | ---: | ---: |
| Cultivation d5 | 38.5 ms | 10.0 us | about 3,850 shots |
| Coherent d5, five rounds | 11.6 ms | 329 us | about 35 shots |
| QV20 | 38.7 ms | 163 ms | about 0.24 shots |

For the first two cases, rerunning the *current full pipeline* on every shot
would cost more than even eliminating the current shot execution entirely.
This does not rule out cheap specialization after shared preprocessing, or a
restricted proxy/interpreter with much cheaper analysis. For QV20 the arithmetic
leaves room if specialization could remove roughly a quarter of execution
cost, but these diagnostics provide no structural mechanism for doing so.

The compile/hot-shot ratio is an amortization scale, not the break-even point
against an unimplemented alternative. The required comparison remains
`C_common + N * E[C_specialize + T_specialized]` against
`C_shared + N * E[T_shared]`, with allocation, discarded shots, and public output
costs included. Fault weight, accepted-trajectory costs, retained capacity,
cache reuse, and alternate interpreters remain unmeasured.

## Corpus and paper coverage

The eight practical cases above come from the hash-checked
[clifft-bench corpus](https://github.com/unitaryfoundation/clifft-bench/tree/4058cf344c143ca217106ada6f64841eaeb0ef00),
excluding its two one-round cases. The production baseline is Clifft commit
`faa53693cef399197e27bd39c325a07acd438c68`; separate planner optimizations must
be remeasured before attributing their effects to a new representation.
The local manifest is suite 0.2.1: all artifact paths and hashes were also
checked against the pinned 0.2.2 manifest. Only the suite-version field and
one compatible-adapter list differ; no circuit input differs.

[Chan et al.](https://arxiv.org/abs/2609.17706) supply additional corrected and
flagged circuits in
[cliffordea](https://github.com/timchan0/cliffordea/tree/3e3be604595b2b12343a45dd4ba2b03a532b21a0).
The study locally acquired d3a6, d3a6f2, d5a19, and d5a19f13 in noise-free and
p=0.001 forms. Following the repository's
[simulation instructions](https://github.com/timchan0/cliffordea/blob/3e3be604595b2b12343a45dd4ba2b03a532b21a0/cliffordea/sim/README.md),
only S/S_DAG instruction names become T/T_DAG. The selected d5 files already
contain the required growth correction; `_uncorrected` files are excluded.
Detector, feedback, flag, gate-order, and noise content otherwise stays intact.
The noise-free files omit noise instructions; they are not the retained-noise,
zero-probability convention used by the MPS paper.

These eight additions have peak widths 4 or 10 and all merge to that width.
Noisy d3 variants have visit ratio 1.004 and noisy d5 variants 1.058. They add
important fault/flag coverage but do not make the product-factor case stronger.
Snapshots here are diagnostics, not an independent validation of the papers'
logical-error rates. Circuit redistribution permission was not established,
so only source references, hashes, transformations, and derived diagnostics
are checked in; the downloaded circuit bytes remain local research inputs.

[Wan and Zapirain](https://arxiv.org/abs/2609.18922) motivate using corrected
cultivation as an exact validation target. The local
`tests/fixtures/cultivation_d5.stim` matches their Appendix A hash
`859d925c74712efaae8bab1d944f9e5be4e042e226eeafcc6d7bef9d6fbfbae5`.
It is a separate input from the benchmark-manifest circuit; this study does
not interchange their noise rates or claim agreement with published response
tables.

The crucial outstanding gap is the actual Reg3/Reg5 and rotated T-cultivation
workloads in [Hartweg and Pineiro Orioli](https://arxiv.org/abs/2609.19116).
Their reported Reg5 width 22 and CAMPS bond dimension 5 motivate a different
frame/representation, but do not show that Clifft's present residual has that
rank. Published Clifft 0.7.0 timings are not a current baseline.
The admissible Clifford-stabilizer protocols in
[Takada, Bartlett, and Williamson](https://arxiv.org/abs/2609.16929) remain
promising candidates for a restricted exact output-preserving reduction;
their applicability conditions must be established on concrete circuit inputs.

## Recommendation and next experiment

1. Defer a production product backend for repeated-round coherent memory and
   current cultivation. Continue the scoped product investigation only if
   larger independent-block practical workloads justify it. Keep the new
   diagnostic as a cheap way to test future candidates.
2. Defer a straightforward exact MPS backend in the existing coordinate order.
   First test reorderings and compiler-chosen frames on the five-round target
   and actual Reg5 circuits; record noisy and fixed-fault rank distributions.
3. Prioritize acquiring those fold-based workloads and proving a small
   Clifford-stabilizer/proxy or complete-check interface. Its first contract
   should state eligible preparation, gates, Pauli noise, measurements,
   requested outputs, and exact fallback. Do not assume passing a syndrome
   check restores an arbitrary noisy physical state to the codespace.
4. Measure shared preprocessing plus per-shot/block specialization against the
   current pipeline before choosing compilation granularity. An interpreter
   or trajectory-dependent topology requires a separate architectural decision
   under `AGENTS.md`; none is introduced by these diagnostic tools.

This milestone does not select a backend, establish a practical speedup,
complete the composition matrix, or close the research issue. The evidence
narrows the next experiments without treating a synthetic or one-round result
as an adoption criterion.

## Reproduce

Build against the current checkout; the profiling target needs no test suite:

```bash
cmake -S . -B build-research -DCMAKE_BUILD_TYPE=Release \
  -DCLIFFT_BUILD_TESTS=OFF -DCLIFFT_BUILD_PROFILER=ON -DCLIFFT_OPENMP=OFF
cmake --build build-research --target profile_structure -j4
python3 tools/profile/factored_state.py \
  --binary build-research/profile_structure \
  --manifest /path/to/clifft-bench/manifests/workloads.v1.json \
  --exclude-one-round --postselect --shots 32 --repeats 3 \
  --output /tmp/practical.json
```

Each artifact is hash-checked against the supplied manifest. Explicit
`--circuit PATH` arguments can be repeated to study local fixtures. Timing
sampling is skipped above width 22 or for instruments, while the semantic
analysis still runs. Set larger shot counts for meaningful timing comparisons.
The native positional interface is
`CIRCUIT [SHOTS=128] [MAX_WIDTH=22] [POSTSELECT=0] [SNAPSHOT_DIR]`.

For rank diagnostics, install NumPy in the chosen Python environment and use
the unchanged compiled plan to generate bounded snapshots:

```bash
build-research/profile_structure \
  /path/to/clifft-bench/workloads/circuits/coherent_d5_r5.stim \
  128 22 0 /tmp/d5r5-snapshots > /tmp/d5r5-plan.json
OPENBLAS_NUM_THREADS=1 python3 tools/profile/residual_ranks.py \
  --directory /tmp/d5r5-snapshots --plan /tmp/d5r5-plan.json \
  --output /tmp/d5r5-ranks.json
python3 -m unittest discover -s tools/profile -p 'test_*.py'
```

Snapshot mode declines instruments and peak widths above 14 before allocating
snapshot executors. It does not sample rare-fault strata or certify exact rank.
Use separate output directories for different circuits.

To prepare the additional cultivation inputs from a local checkout at the
pinned cliffordea revision:

```bash
python3 tools/profile/prepare_cultivation_study.py \
  --checkout /path/to/cliffordea --output /tmp/paper-circuits
```

The script reads committed bytes, checks the revision and expected converted
gate counts, and records original/derived hashes. Pass its `T_*.stim` files as
individual `--circuit` arguments. It does not fetch files or silently modify
the source checkout.

[Recorded data](factored_state_data.json) retains input hashes, binary hashes,
per-invocation timings, model summaries, and all sampled cut ranks. Full
coefficient snapshots and per-action factor traces can be regenerated with
the commands above; they are not retained in the repository.
