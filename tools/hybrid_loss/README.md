# Online Clifford coordinates with Clifft's dense active state

This standalone research executor investigates skip-gate loss without continuation
compilation. It uses Clifft's parsed circuits, resolved five-level model,
measurement classifier, native Pauli strings, and production dense-state kernels.
The prototype leaves the public sampling API unchanged. Its production integration
is now available as the explicit `noncomp` execution mode `"online"`; automatic
strategy selection still chooses fixed or trajectory execution.

The experiment deliberately permits runtime coordinate planning and temporary
allocation. It is an explicitly authorized architecture experiment, not an
implementation of the production executor's allocation-free or topology-free
contract. Unsupported instructions fail explicitly. This is not a release-ready
backend.

## Measured result

On the AMD EPYC 9554P VM, one core, at base revision
`49b568528f5323627f16d06ad2ee73d9fec5598a`, the prototype beats QDK 1.33.1
on all four matched skip-gate surface-code cases. Times are microseconds per
shot. QDK ranges below span the estimates from two timing runs, not confidence
intervals. The hybrid column uses the final isolated repeat.

| Distance | QDK | Hybrid | Hybrid speedup |
| --- | ---: | ---: | ---: |
| 5 | 300-321 | 63.8 | 4.7-5.0x |
| 7 | 780-876 | 221 | 3.5-4.0x |
| 9 | 1,642-1,732 | 610 | 2.7-2.8x |
| 11 | 3,049-3,072 | 1,617 | 1.9x |

The d=7 instrumented trajectory profile spent 64% of sampling time in coordinate
planning, 11% in tracing, 9% in executable preparation, 6% in rewriting, and
0.2% in execution. A rewrite-only improvement would leave the dominant costs.
The initial online prototype took 3,165 us/shot at d=11; fusing the dormant
measurement's frame transformations reduced this to 1,617 us/shot.

On the adapted cultivation circuits with loss, the hybrid took 25.3 us/shot at
d=3 and 194 us/shot at d=5. The established trajectory sampler took 231 and
10,714 us/shot in the representative comparison, respectively: about 9x and 55x
improvements. Observed active widths matched the reference, reaching 6 and 13.
Those reference timings are whole-call averages; the final hybrid estimates use
slopes. They are not a matched-model QDK comparison.

The neutral-atom workloads also passed, including decoded acceptance and logical
outcomes. The exact LDU case reaches width 18 in both backends and remains
millisecond-scale. Its existing trajectory profile spends about 97% of time in
quantum execution, demonstrating a different bottleneck from surface-code loss.

Existing fixed execution must be retained. For example, the d=3 cultivation
control with full partner depolarization takes about 0.29 us/shot on existing
automatic packed execution, versus tens of microseconds on the online prototype.
An online executor alone is not a universal replacement for compiled sampling.

Validation completed: 84 independent Aer/Stim/dense-Kraus checks; ten
representative comparisons against current Clifft; six direct QDK joint-law
checks; and lossless parity-readout adaptation checks at both cultivation
distances. The fused and initial online versions produced identical output bytes
on all 120,288 representative shots. ASan, UBSan, and assertions passed selected
representative cases; LeakSanitizer could not run under the sandbox's ptrace
restriction and was disabled.

See [timings](results/timings.csv), [run metadata](results/metadata.json), and
[phase measurements](results/continuation_phases.json). The results directory
also retains raw timing calls and validation summaries. The full 22-case sweep
is exploratory; six headline hybrid cases were repeated in isolation.

## Integrated API measurements

The production integration was remeasured on base `28faf8ff` with Clang 18,
Release optimization, and assertions enabled. These are medians of three
prepared public API calls on one pinned core, including output conversion.
Preparation is reported separately in [the raw report](results/integrated.json).
Shot counts differ between modes to keep the sweep bounded; each mode uses the
same circuit and model. This is an initial throughput comparison, not the full
event-rate, memory, and shot-count crossover study.

| Workload | Existing mode | Existing us/shot | Online us/shot |
| --- | --- | ---: | ---: |
| Surface d5 skip loss | trajectory | 10,515 | 69.7 |
| Surface d7 skip loss | trajectory | 121,392 | 250 |
| Cultivation d3 skip loss | trajectory | 299 | 25.1 |
| Cultivation d5 skip loss | trajectory | 14,027 | 204 |
| Neutral two-row exact | trajectory | 216 | 53.8 |
| Neutral LDU exact | trajectory | 11,690 | 5,017 |
| Cultivation d3 full partner depolarization | fixed | 0.329 | 25.1 |
| Cultivation d5 full partner depolarization | fixed | 87.3 | 217 |

Reproduce after generating the frozen workloads below, using a Python environment
that imports the integrated build:

```bash
python tools/hybrid_loss/benchmark_integrated.py \
  --workloads /tmp/hybrid-workloads --output /tmp/integrated.json \
  --cpu 2 --build-description "describe the compiler and build flags here"
```

The earlier QDK results above are historical prototype measurements; QDK was not
rerun for this integrated sweep. The maintained production tests live in the
ordinary C++ and Python suites. This directory retains the exploratory harness
and its original reports, whose source hashes describe the files at run time,
before production integration and formatting/type-check cleanup.

## Representation

The quantum state is `C (|phi>_k tensor |0>_(n-k))`. Each shot maintains the inverse
Clifford map `C^-1` and a dense coefficient array with `2^k` entries. Physical
Clifford gates update a few inverse-tableau rows; Pauli noise changes their signs.
Non-Clifford axes are mapped into the current coordinates as needed.

If a mapped axis has X support on a dormant coordinate, a controlled-Pauli basis
change makes it the next active X coordinate. This basis change fixes the prior
state because its control is dormant zero. Clifft's promotion/rotation kernels
then update the coefficients. An active measurement uses Clifft's probability and
collapse kernels and removes its measured coordinate. A dormant measurement
updates the Clifford frame without increasing active width. Its basis change and
sampled correction are fused into one traversal of the inverse tableau.

Loss samples a hidden computational measurement, parks the now unobserved factor
at zero, and updates the level ledger. Subsequent gates consult that ledger.
Named transitions retain source/destination correlations and apply the exact
no-jump filter when requested. Leaked levels, recapture, partner Pauli effects,
leakage spreading, readout confusion, heralds, and reset restoration use the same
model specification as the trajectory reference.

This is the online-coordinate component of a hybrid design. It does not yet
splice compiled segments into dynamic execution. The existing fixed/packed
strategy remains necessary for its favorable workloads.

## Build and run

From the repository root, with a C++20 compiler and Python available:

```bash
cmake -S tools/hybrid_loss -B build-hybrid -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build-hybrid --target hybrid_loss -j4

python tools/hybrid_loss/experiment.py --output /tmp/hybrid-workloads
python tools/hybrid_loss/validate.py \
  --binary "$PWD/build-hybrid/hybrid_loss" --output /tmp/hybrid-validation
python tools/hybrid_loss/compare.py \
  --binary "$PWD/build-hybrid/hybrid_loss" \
  --workloads /tmp/hybrid-workloads --output /tmp/hybrid-comparison
python tools/hybrid_loss/benchmark.py \
  --binary "$PWD/build-hybrid/hybrid_loss" --backend hybrid \
  --workloads /tmp/hybrid-workloads --output /tmp/hybrid-benchmark
```

Dependencies for workload generation and comparisons: NumPy, Stim, and the
benchmarked Clifft Python build. Independent validation also uses Qiskit Aer.
The optional `--backend qdk` comparison uses QDK 1.33.1. Run each environment
separately, verify the imported Clifft version, and pin timing runs to an idle
core. The scripts currently use CPU 2 for timings and CPU 3 for validation;
adjust those affinities on hosts with fewer CPUs. Do not overlap validation
with timing runs used for conclusions.

`experiment.py` serializes the model as a simple text file for the native tool.
A direct invocation is:

```bash
build-hybrid/hybrid_loss --mode hybrid --circuit circuit.stim \
  --model model.txt --shots 1024 --seed 42 --capacity 20 --output /tmp/sample
```

The JSON result separates preparation and sampling, and reports active width,
continuations, and Linux peak RSS. Optional output files contain row-major byte
arrays for measurements, heralds, detectors, observables, and final status.
`--final-status 1` requests the last array. File I/O is outside the sampling timer;
native result allocation and materialization are inside it. The coefficient
budget is explicit; exceeding it raises an error without approximation.

`--mode trajectory` runs the established driver from the same source revision.
The build generates a locally instrumented copy of that driver and reports
rewrite, trace, optimization, coordinate planning, executable preparation,
execution, and classical-processing times to stderr. Its counters are intended
for these single-worker experiments. Production sources are not instrumented.

## Workload contract

`experiment.py` writes a manifest with hashes of every circuit and model.

- Surface-code memory: d=5,7,9,11, rounds=d, with loss 0, 1e-4, or 1e-3 at the
  original reproduction's sites. Ordinary Pauli/readout rates are zero for this
  QDK comparison. Both tools skip ideal gates with an inactive operand; resets
  restore loss. Compare ternary measurement records and ignore arbitrary binary
  placeholders in heralded slots. Ancilla reuse follows reset; data readout is
  terminal. QDK's behavior is not assumed equivalent for arbitrary circuits.
- T-state cultivation: the repository's d=3 and d=5 circuits, with their noise
  channels set to p=0.001. MPP readout is expanded into a fresh preparation of a
  reusable ancilla, controlled-Pauli couplings, and X readout. This preserves the
  lossless instrument; under loss it explicitly defines a physical readout
  circuit. Added loss follows listed physical unitaries, including those new
  couplings, and excludes virtual record-controlled corrections. These are
  benchmark extensions, not published loss results. Noise on inactive operands
  follows Clifft semantics, so these results are not a matched QDK comparison.
- Neutral atoms: checked-in two-row and LDU schedules with the tutorial's
  rebalanced and exact five-level models. Published decoding and acceptance
  postprocessing are retained. QDK is not a reference for these leakage models.
- Controls: cultivation without added loss and with full partner depolarization
  exercise Clifft's existing fixed execution, including its packed path.

The benchmark uses three shot counts and three repetitions. It reports the slope
between median endpoint times and all three repetition slopes. This removes
per-call setup from throughput estimates, including QDK's QIR overhead. It is a
native research harness versus public Python APIs, not a packaged-backend
end-to-end comparison. Output formats differ and that remains part of the scope.

## Correctness and limits

`validate.py` checks 40 random non-Clifford unitary circuits against exact Aer
probabilities, 20 noisy Clifford circuits with reset/feed-forward against Stim,
and 24 five-level circuits against the independent dense Kraus enumerator. The
last group includes T gates, entanglement, leakage, loss, recapture, and both
no-jump policies. Joint output probabilities and impossible outcomes are tested.

`compare.py` checks representative joint herald/measurement statistics, detector
and observable records, final statuses, and decoded acceptance/outcomes against
the pinned established trajectory sampler. These statistical checks are evidence
for the exercised workloads, not a proof of every channel combination. Large
joint laws are checked through selected correlations rather than enumerated in
full. Timing runs are separate from these correctness samples.

Current experiment limits include single-worker execution, an explicit active
width cap of at most 24, no MPP/expectation API, no U3 or three-qubit noise channel
interpreter, and no production-compatible per-shot RNG stream contract. Seeded
runs are repeatable within this executor. Some Clifford-valued rotation spellings
can use the dense path unnecessarily. The implementation can allocate during
runtime frame changes and has not undergone production hardening.

The method retains exponential dependence on genuine active width. Exact LDU
back-action reaches width 18 in both this executor and the trajectory reference.
Avoiding continuations cannot remove that cost. Existing compiled/packed paths
are also much faster on several no-loss and depolarizing-partner controls.

## Provenance

The compact state and quantum kernels come from Clifft's existing factored-state
implementation and its documented SOFT/SymFT lineage. Microsoft QDK supplied the
skip-gate performance/differential reference and motivated investigating dynamic
Clifford coordinates. See [acknowledgments](../../docs/acknowledgments.md).
