# Fault-conditioned compilation: a low-reuse opportunity

The second experiment for [the representation study](https://github.com/unitaryfoundation/clifft/issues/495)
finds a narrow positive result: **sampling one or a few cultivation shots from
a fresh circuit can be faster after resolving its physical Pauli faults**.
This comes from avoiding expensive shared symbolic preparation, not a smaller
active state. Recompiling every shot loses at larger shot counts. Five-round
coherent memory does not benefit in this experiment.

This is an opt-in research harness. No production API, planner, or executor
changes. One-round circuits are excluded throughout.

## What the prototype does

Parse the physical circuit once and prepare its independent quantum noise
sites. For each realization, sample a mutually exclusive outcome at every site,
replace it with its physical X/Y/Z gates (or remove the identity outcome), and
run the unchanged trace, HIR optimization, coordinate planning, lowering, and
scalar execution pipeline. Phase errors retain their physical gate location;
they are not propagated through a gadget using an assumed Pauli-only rule.

Supported channels are X/Y/Z_ERROR, DEPOLARIZE1/2/3, PAULI_CHANNEL_1/2/3, and
CORRELATED_ERROR/ELSE_CORRELATED_ERROR chains. Each correlated chain is one
mutually exclusive site with the original conditional probabilities converted
to absolute outcome probabilities. No fault-weight truncation is performed.
Preparation declines loss, leakage, level transitions, and unsupported noise.

Measurements, resets, Pauli feedback, detectors, observables, rotations, and
readout noise remain in the circuit. Asymmetric readout stays state-dependent
at its original record position. The experiment uses ordinary circuit initial
states; arbitrary supplied residuals and noncomputational continuations have
not been investigated. Final-state query, expectation, and importance-sampling
integration are not claimed by these timing results.

The original shared plan is also bound to each identical quantum fault pattern
through its existing presampled-symbol interface. Site counts and ordered
probability lists are checked against the physical preparation. This mapping
depends on the current frontend/planner ordering and is a diagnostic interface,
not a proposed stable API. The independent conformance cases exercise the
ordering, including all two- and three-qubit Pauli channel outcomes.

Full per-shot compilation is outside ordinary executor dispatch. There is no
new runtime tableau evolution or topology discovery inside the executor, and
no architectural exception is required for this experiment.

## Corpus and controls

Use the hash-checked benchmark cultivation d5 and coherent d5/five-round cases,
plus the corrected flagged d5a19f13 T circuit at p=0.001 from the pinned
[Chan et al. source](https://github.com/timchan0/cliffordea/tree/3e3be604595b2b12343a45dd4ba2b03a532b21a0).
Acquisition and S-to-T conversion are described in the
[first report](factored_state.md); the original and derived hashes are retained.

For each circuit, study 32 quantum-fault patterns drawn normally and from the
exact k=0,1,2 quantum-site strata. The existing `KFaultSampler` handles the
conditioned site distribution, followed by the original conditional Pauli
distribution within each fired site. **Readout faults are not included in k.**
This differs from public fixed-k APIs that also count readout faults.

Each pattern gets 128 measurement/readout samples with every detector
postselected. A separate comparison disables early rejection for 16 patterns
in the natural and zero-quantum-fault ensembles. The same compiled specialized
plan is reused for these samples only to measure conditional execution cost;
the hypothetical per-shot strategy must pay compilation on every shot.

## Structure and sustained execution

| Circuit | Shared peak width | Specialized width across all 128 patterns | Full-plan unfused coefficient visits, shared / specialized |
| --- | ---: | ---: | ---: |
| Coherent d5, five rounds | 13 | 15 | 2,587,680 / 3,181,452 |
| Benchmark cultivation d5 | 10 | 10 | 62,822 / 24,350 |
| Flagged cultivation d5a19f13 | 10 | 10 | 66,146 / 25,162 |

Fixing faults does not produce a smaller dense residual here. In the coherent
case, removing noise changes the optimizer's schedule and increases width.
This illustrates why fewer circuit instructions do not guarantee less dense
work. The cultivation cases save symbolic and coefficient work while retaining
the same peak dense allocation.

For natural fault samples with postselection, approximate mean specialized
compilation costs are 3.94 ms, 0.87 ms, and 1.19 ms respectively. The ordinary
shared executor's attempted-shot means are about 146 us, 7.9 us, and 5.1 us.
The specialized hot-shot means are about 313 us, 3.0 us, and 2.6 us. This
rules out amortizing full specialization separately on every shot in a
high-reuse sampling job for these cases.

Without early rejection, cultivation's hot execution drops from roughly
20 us to 8--11 us; coherent memory increases from roughly 515 us to 768 us.
These VM timings are indicative, not tuned throughput results. Importantly,
the diagnostic **shared presampled path is not the fastest production baseline**:
assigning every supplied noise symbol costs much more than automatic lazy
noise sampling on cultivation. Its 36--77 us timings must not be presented as
the production baseline to inflate the specialization benefit.

In the 32 ordinary patterns, cultivation produces 32 distinct materialized
circuits in each family. Coherent memory produces eight, with frequent
zero-fault repeats, but the specialized execution itself is slower there.
These observations give no compelling case for a fault-pattern cache. Equal
widths or visit counts are not a sufficient cache key, and a no-fault fast
path requires an exactly conditioned fallback rather than an unconditional
resample that would count no-fault trajectories twice.

## Fresh-process latency

Unlike the conditional execution measurements, each latency invocation does
only one candidate pipeline. The timer includes parsing, common preparation
where applicable, quantum fault draws, materialization, all compilation and
allocation, scalar shots, discard/observable-0 count collection, and teardown.
It excludes process startup, dynamic-library loading, and input-file I/O.
The specialized path does not first compile the original shared plan.

Each cell below is the median of five fresh-process runs, alternating strategy
order. Both strategies use all-detector postselection and the same physical
noise model; they need not produce identical rows from their different RNG
streams. Seeds are fixed across timing repeats. Hardware is the same 8-vCPU
AMD EPYC 9554P VM, GCC 13.3 Release/native, OpenMP disabled. CPU affinity is
not pinned, and OS caches are not flushed.

| Circuit | Shots | Shared total ms | Specialized total ms | Shared / specialized |
| --- | ---: | ---: | ---: | ---: |
| Coherent d5, five rounds | 1 | 13.28 | 13.23 | 1.00 |
| | 4 | 9.75 | 31.32 | 0.31 |
| | 16 | 7.09 | 80.13 | 0.09 |
| | 64 | 13.31 | 287.18 | 0.05 |
| Cultivation d5 | 1 | 33.87 | 6.20 | 5.47 |
| | 4 | 35.88 | 15.55 | 2.31 |
| | 16 | 29.10 | 28.44 | 1.02 |
| | 64 | 17.57 | 64.04 | 0.27 |
| Flagged d5a19f13 | 1 | 28.28 | 8.20 | 3.45 |
| | 4 | 33.29 | 15.26 | 2.18 |
| | 16 | 33.70 | 41.84 | 0.81 |
| | 64 | 21.07 | 101.68 | 0.21 |

The non-monotonic shared times expose substantial VM/allocator/cache variation.
For one-shot cultivation, the specialized ranges (5.8--8.8 ms and 7.6--11.4 ms)
remain below the corresponding shared ranges (20.6--41.3 ms and 19.5--50.2 ms).
The exact crossover near 16 shots is not established: only a small grid was
measured, and those ranges overlap. By 64 shots, shared compilation wins.

Adding separately measured warm compile and execution components had suggested
a coherent one-shot benefit. The fresh-process experiment did not confirm it.
This is why the adoption decision should use complete candidate pipelines,
not a sum of favorable warm microbenchmarks.

This result is relevant to interactive debugging, smoke tests, and other
low-reuse calls. It does not speed up the large shot counts normally needed
to estimate rare cultivation failures and does not establish best-in-class
performance against external simulators or optimized alternative planners.

## Residual ranks and correctness

Snapshot the first realization in each k=0,1,2 stratum, with four unconditional
prefix samples at sixteen spaced checkpoints plus the first maximum-width
checkpoint. Snapshot mode now permits width 16 to cover the specialized
coherent plan's width 15. Readout remains stochastic. These few selected
patterns are not a distributional bound or an accepted-shot ensemble.

Maximum observed numerical Schmidt ranks at relative threshold 1e-12 are:

| Circuit | Natural shared-plan prefixes | Specialized k=0 / k=1 / k=2 prefixes |
| --- | ---: | ---: |
| Coherent d5, five rounds | 64 | 64 / 64 / 64 |
| Benchmark cultivation d5 | 22 | 12 / 12 / 12 |
| Flagged d5a19f13 | 22 | 15 / 15 / 15 |

The shared and specialized columns use different sampled prefixes and frames;
this is exploratory evidence, not a paired-state proof of rank reduction.
At tighter threshold 1e-14 these maxima are unchanged. Some cultivation
prefixes look more compressible, but a dense width-10 state is already small.
This does not establish an MPS speed advantage or justify a tensor backend.

Validation includes exhaustive visible-record probabilities on small circuits,
marginalizing hidden reset records and comparing specialized execution with
the original shared plan conditioned on the same quantum faults. Independent
checks use Qiskit Aer for non-Clifford fault conjugation and Stim for stochastic
Clifford/else-correlated behavior. Tests also cover channel argument order,
readout-dependent feedback, fixed-k weighting, serialization round trips,
fresh-path output collection, and unsupported-instrument rejection. Snapshot
norms and predicted product partitions are checked by linear algebra. Large
circuit aggregate samples are diagnostics, not rare-logical-error validation.

## Recommendation

**Proceed with a narrowly scoped low-reuse sampling investigation**, coordinated
with [the limited-reuse issue](https://github.com/unitaryfoundation/clifft/issues/494).
Keep shared compilation for sustained sampling. A potential initial interface
would explicitly request a small shot count from a circuit, use a conservative
Pauli-noise eligibility check, and retain the existing compiled path as fallback.
Avoid automatic dispatch based on a hard-coded crossover from this VM.

Before production adoption, repeat the complete latency test with controlled
CPU placement, broader circuits and seeds, the public output contract, and
applicable compiler improvements. Measure preparation memory and fallback
overhead. Any shared common-IR or block-specialization shortcut must preserve
fault-location and feedback semantics; similar plan shapes alone do not prove
it safe. Integration with fixed-fault/importance sampling and continuations
requires its own explicit contract.

For the original large-throughput goal, this experiment deprioritizes naive
per-shot recompilation on the tested workloads. The actual Reg5/fold-based
inputs and restricted whole-check/Clifford-proxy eligibility remain outstanding.
This report does not close the wider representation study.

## Reproduce

Build with the same profiling CMake options as the first report:

```bash
cmake --build build-research --target profile_fault_specialization profile_structure -j4
python3 tools/profile/run_fault_study.py \
  --binary build-research/profile_fault_specialization \
  --circuit /path/to/coherent_d5_r5.stim \
  --circuit /path/to/msc_d5_inject_cultivate_p1e-3.stim \
  --circuit /path/to/T_d5a19f13_inject+cultivate_p1e-3.stim \
  --patterns 32 --shots 128 --output /tmp/fault-study
```

This defaults to k=-1 (ordinary),0,1,2 and saves every materialized circuit,
input/derived hashes, and raw measurements. For the no-early-rejection control,
use `--patterns 16 --k -1 0 --postselect 0` and a different output directory.

```bash
python3 tools/profile/run_fault_study.py \
  --binary build-research/profile_fault_specialization \
  --circuit /path/to/msc_d5_inject_cultivate_p1e-3.stim \
  --latency --latency-shots 1 4 16 64 --repeats 5 \
  --output /tmp/fault-latency
```

The native diagnostic interface is
`CIRCUIT [PATTERNS=16] [SHOTS=128] [K=-1] [OUTPUT_DIR] [POSTSELECT=1]`.
The independent latency interface is
`--latency shared|specialized CIRCUIT SHOTS --postselect-all`.
Sampling refuses widths above 22 in latency mode; the main diagnostic instead
skips timing for those widths. Exact small-circuit oracles require at most
twelve total records, width fourteen, and no stochastic readout.

Use `profile_structure` and `residual_ranks.py` on the exported `0.stim` files
for the rank experiment, following the first report's commands. Run the tests
with NumPy, Stim, Qiskit, and Qiskit Aer available:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tools/profile -p 'test_*.py'
```

[Recorded data](fault_specialization_data.json) includes all per-pattern
metrics and input hashes, all five fresh-process latency repeats, and rank
summaries. Full coefficient snapshots and spectra are reproducible local
artifacts, not checked-in circuit copies.
