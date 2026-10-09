# Native Profiling Tools

Native C++ harnesses isolate production compile, sampling, and
strong-simulation costs for `perf` or another sampling profiler:

- `profile_compile` repeatedly runs parse, trace and HIR optimization,
  coordinate planning, and executable-plan preparation.
- `profile_probability` compiles a unitary circuit and repeatedly queries
  `clifft::basis_probabilities()` over a batch of bitstrings.
- `profile_sample` compiles a circuit once and repeatedly samples it through
  the public C++ path.
- `profile_optimizer_allocations` counts allocations within the default HIR
  optimizer, excluding parsing, tracing, planning, and execution.

`profile_symbolic_stabilizers.py` measures measured CSS preparation, Pauli
feedback, and stabilizer slicing through the complete default compiler pipeline.
It includes faulty-preparation and readout controls, a phase-pass ablation, and
independent Aer/Stim checks.

Add `--wide` to time each default optimizer pass on rotated surface-code memory
at distances 3, 11, and 21 with equally many rounds. These stress cases replace
each depolarizing location with `R_Z(0.02)` on the same targets, retaining
measurement flips and optionally the original Pauli noise. They measure pass
time and rotation removals without lowering or sampling the wide state. The
distance-21 cases can take several minutes; compare identical cases and build
settings in separate environments. These are scaling probes, not validated
fault-tolerant coherent-noise models.

## Build

The harnesses are opt-in. `RelWithDebInfo` retains call stacks while preserving
the optimized code paths used for profiling.

```bash
cmake -B build-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCLIFFT_BUILD_PROFILER=ON
cmake --build build-profile --target profile_compile profile_probability profile_sample profile_optimizer_allocations -j$(nproc)
```

`just profile-build` builds the timing harnesses. Build the allocation probe
explicitly when needed.

## Compilation

`profile_compile` runs `default_hir_pass_manager()` and reports parse, trace,
optimization, scheduling, plan, prepare, and total time separately. File I/O
is outside the timed loop.

```bash
CLIFFT_COMPILE_ITERATIONS=200 \
  CLIFFT_CIRCUIT_FILE=tests/fixtures/cultivation_d5.stim \
  ./build-profile/profile_compile

CLIFFT_COMPILE_ITERATIONS=200 \
  CLIFFT_CIRCUIT_FILE=tests/fixtures/cultivation_d5.stim \
  perf record -F 9999 -g --call-graph dwarf \
  -o perf-compile.data ./build-profile/profile_compile
```

| Variable | Default | Description |
|---|---:|---|
| `CLIFFT_CIRCUIT_FILE` | generated circuit | Input `.stim` file |
| `CLIFFT_COMPILE_ITERATIONS` | 20 | Number of complete compilations |
| `CLIFFT_NUM_QUBITS` | 50 | Qubits in the generated circuit |
| `CLIFFT_CLIFFORD_DEPTH` | 5000 | Clifford gates in the generated circuit |
| `CLIFFT_T_GATES` | 0 | T gates appended to the generated circuit |
| `CLIFFT_POSTSELECT_ALL` | unset | Mark every detector for postselection |
| `CLIFFT_ACTIVE_WIDTH_SCHEDULE` | `0` | Enable the opt-in scheduler and report its time and work separately |

Enable scheduling to time it separately:

```bash
CLIFFT_ACTIVE_WIDTH_SCHEDULE=1 CLIFFT_COMPILE_ITERATIONS=100 \
  CLIFFT_CIRCUIT_FILE=tests/fixtures/coherent_d5_r5.stim \
  taskset -c 0 perf record -e cycles:u -F 499 -g --call-graph dwarf \
  -o perf-width.data ./build-profile/profile_compile
perf report --stdio --no-children -i perf-width.data
```

The scheduler reports `swept_ops` and `classification_probes` separately. The
execution budget does not bound classification probes, which can grow
quadratically for wide ready sets. The total compile time also includes
parsing, production passes, planning, and executable preparation.

### Optimizer allocations

```bash
./build-profile/profile_optimizer_allocations tests/fixtures/cultivation_d5.stim
```

The standalone probe replaces C++ `new` and `delete` only in its own executable.
It counts requested payload bytes allocated during the single-threaded default
HIR optimizer, including live output allocations. Peak live bytes exclude
preexisting HIR storage, allocator metadata, and direct C allocation calls.
This is an optimizer allocation measurement, not process RSS or isolated
symbolic-analysis memory. Use the ordinary harness for timings; the probe's
allocation headers change allocator behavior.

### Measured preparation and feedback

Run from the repository root with the development dependencies installed:

```bash
taskset -c 0 .venv/bin/python tools/profile/profile_symbolic_stabilizers.py \
  --repeats 11 --shots 65536 --corpus --small-batches \
  --output /tmp/symbolic-stabilizers.json
```

The JSON output includes circuits, optimized T counts, active widths, action
counts, plans, compilation stages, sampling throughput, and independent
validation results. `--small-batches` adds sampling timings for 1, 16, 256,
and 4,096 shots; `--corpus` adds compilation of existing repository fixtures.
Use `--skip-validation` when repeating timing-only runs. Compare builds in
separate environments with the same inputs, retained outputs, thread count,
and CPU affinity; alternate their execution order to limit timing drift.

The Reed-Muller workload prepares a logical plus state using ten measured Z
checks and Pauli feedback, then applies transversal T. It retains preparation
records, logical outputs, and expectation probes. The slicing workload checks
arbitrary-angle cancellation, unequal rotations, and sign-changing faults.
The 45-qubit transversal-CCZ case is a compilation scaling control without a
decoder, acceptance rule, or fault-tolerant ancilla preparation.

For a fresh compilation, compare compilation plus sampling time at the desired
shot count. When reusing a compiled program, amortize compilation separately.
The no-phase ablation distinguishes reductions enabled by HIR optimization
from facts already exploited by the sampling planner.

## Sampling

`profile_sample` keeps parsing and compilation outside the measured interval.
Executor construction, state initialization, hot execution, and result
collection remain inside it, matching the end-to-end cost of repeated calls to
the public sampling API.

```bash
CLIFFT_CIRCUIT_FILE=tools/bench/fixtures/qv20_seed42.stim \
  CLIFFT_PROFILE_API=sample \
  CLIFFT_PROFILE_SHOTS=1 \
  CLIFFT_PROFILE_THREADS=1 \
  ./build-profile/profile_sample
```

| Variable | Default | Description |
|---|---:|---|
| `CLIFFT_CIRCUIT_FILE` | required | Input `.stim` file |
| `CLIFFT_PROFILE_SHOTS` | 1 | Shots per measured sample call |
| `CLIFFT_PROFILE_THREADS` | 1 | Total worker budget; `0` selects auto |
| `CLIFFT_PROFILE_SHOT_WORKERS` | unset | Explicit cross-shot workers; set with intra-shot workers |
| `CLIFFT_PROFILE_INTRA_SHOT_WORKERS` | unset | Explicit per-shot workers; set with shot workers |
| `CLIFFT_PROFILE_INTRA_SHOT_MIN_ACTIVE_WIDTH` | 18 | Expert kernel threshold; requires an explicit layout |
| `CLIFFT_PROFILE_WARMUPS` | 2 | Untimed sample calls |
| `CLIFFT_PROFILE_REPETITIONS` | 20 | Timed sample calls |
| `CLIFFT_PROFILE_API` | `sample` | Public API to profile: `sample`, `sample_survivors`, `sample_k`, or `sample_k_survivors` |
| `CLIFFT_PROFILE_BATCH_SIZE` | auto | Select the conservative automatic policy; a positive integer forces that capacity and `1` selects scalar execution |
| `CLIFFT_PROFILE_KEEP_RECORDS` | unset | Retain surviving rows for either survivor API |
| `CLIFFT_PROFILE_FIXED_K` | 1 | Fault count for either fixed-fault API |
| `CLIFFT_PROFILE_POSTSELECTION` | `none` | Survivor detector mask: `none`, `all`, `first-half`, `last-half`, or `alternating` |
| `CLIFFT_PROFILE_AGGREGATE_SURVIVORS` | unset | Legacy alias selecting `sample_survivors` when `CLIFFT_PROFILE_API` is unset |
| `CLIFFT_PROFILE_POSTSELECT_ALL` | unset | Legacy alias for `CLIFFT_PROFILE_POSTSELECTION=all` |
| `CLIFFT_PROFILE_GENERATED_WIDTH` | unset | Generate a rotation-heavy circuit of this width instead of loading a file |
| `CLIFFT_PROFILE_GENERATED_DEPTH` | 20 | Layers in the generated circuit |

The profiler prints the planner's estimated coefficient visits per lane and a
final `RESULT` line with that estimate, the requested batch setting, effective
lane capacity and worker count, timing, survival rate, and retained row count.
For example, this compares scalar and explicit packed capacities for the
aggregate survivor path with every detector postselected. Automatic mode is
intentionally scalar for a postselected plan.

```bash
for batch in 1 256 1024; do
  env CLIFFT_CIRCUIT_FILE=tests/fixtures/surface_d7_r7_p001.stim \
    CLIFFT_PROFILE_API=sample_survivors \
    CLIFFT_PROFILE_KEEP_RECORDS=0 \
    CLIFFT_PROFILE_POSTSELECTION=all \
    CLIFFT_PROFILE_SHOTS=100000 \
    CLIFFT_PROFILE_BATCH_SIZE="$batch" \
    ./build-profile/profile_sample
done
```

To run the complete public-API matrix and retain the raw results as CSV:

```bash
python3 tools/profile/run_sampling_mode_matrix.py \
  --output /tmp/clifft-sampling-mode-matrix.csv
```

The matrix covers ordinary and fixed-fault sampling, aggregate and retained
survivor output, with and without postselection. Explicit capacities can be
changed with `--batches`; scalar (`1`) and `auto` are always required. Use
`--apis`, `--keep-records`, and `--postselection` to run a focused subset.

`tools/profile/fixtures/active_width5_transient.stim` and
`active_width5_sustained.stim` have the same peak active width but different
coefficient-state lifetimes. They exercise the automatic work cutoff without
assuming that peak width alone predicts whether batching is profitable.

## External measured preparation

`profile_external_feedback.py` compares the default policy with explicit phase
ceilings of 32 and 64. It records individual pass times, cap and expansion
counters, T counts, and peak active width. At width at most 16 it also records
full compilation and the first sampling batch, including executor preparation.
Larger circuits use `trace`, HIR passes, and `active_width_trace` only.

The checked-in `tests/fixtures/merlin_bt27.stim` is the unmodified circuit from
[Merlin](https://github.com/mark-koch/merlin) revision
`097380fac1a3968ca47925146e211fe990f4c396`, generated by
`benchmarks/protocols/code_switching.py` with
`build_code_switching_case("bt27", p_phys=0, target_scoring=False)`. Supplying
that checkout verifies the fixture and adds BT81 and original inverse-logical-CCZ
target scoring. The Z-only variant replaces pre-CCZ depolarization and is a
modified diagnostic workload. Zero events in the target-scoring sample are
only a sanity check; the test suite separately checks decoded logical outputs
and a syndrome jointly against Aer and an exact convolution of Stim fault maps.

```bash
taskset -c 0 .venv/bin/python tools/profile/profile_external_feedback.py \
  --merlin-checkout /path/to/pinned/merlin \
  --repeats 5 --shots 256 --output /tmp/external-feedback.json
```

The optional `--ibm-input` accepts a generated gauging control. The reviewed d3
input comes from [IBM gauging](https://github.com/IBM/gauging_clifford_measurement)
revision `ce90c09c239b2173612c22cc12a2bbf5a7356680`: call `ColorCode(3)`,
`unitary_prep(3, basis="Y")`, `mid_circ_meas(3, basis="Y")`, `stab(3)`, then
`mid_circ_meas(3, basis="Y")`. Follow the upstream conversion convention by
replacing S/S_DAG with T/T_DAG and omitting empty-target Clifford operations.
This control currently has no reduction; its blocker is unresolved.

Use matching Release build settings, one pinned worker, and separate Python
environments when comparing revisions. The profiler also includes Reed-Muller,
stabilizer slicing, the CCZ skeleton, cultivation, coherent memory, and
measurement barriers as controls. Results are local measurements, not portable
performance targets.

BT81's natural phase support has 87 variables, beyond the 64-bit parity
representation. The bounded policy reports cap hits and keeps small synthesis
blocks there. A separate follow-up should evaluate multiword parity keys and
sparse cubic algebra on that 87-variable case with explicit work budgets and
the existing rewrite cost guards. No conditional corrections or execution
architecture changes are needed for the current policy.

## BT27 fault specialization

`profile_bt27_fault_specialization.py` fixes the pre-CCZ Pauli-fault realizations
in the pinned BT27 fixture while keeping measurements and feedback live. It
checks every single fault plus reproducible multiple-fault patterns, measures
complete planned width, and samples only bounded-width cases. Original
reference parities are retained across specializations. This is a diagnostic
for conditional reduction, not a production execution or caching strategy.

`validate_bt27_fault_specialization.py` optionally compares raw record and
syndrome correlations and conditional logical Pauli probes with a separately
installed Merlin. It reconstructs detector parities from raw records to avoid
normalizing each fixed fault against a faulty reference.

See [the research record](../../research/conditional_clifford/README.md) for
commands, pinned revisions, results, limitations, and subsequent decisions.

`analyze_bt27_phase_corrections.py` extracts an exact physical phase polynomial
and fixed-fault Clifford correction controls. `--structure` also fingerprints
ordinary optimized variants; those fingerprints are not executable-equivalence
classes. `validate_bt27_phase_corrections.py` checks small operator identities
with Aer, physical correction composition with Stim, and full-fixture fault
relocation with Clifft differential sampling.

The opt-in CMake target `profile_bt27_shared_analysis` tests a separate shortcut:
reconstructing variants from relative frames inferred from compiled single
faults. It records exact HIR mismatches, including negative results, before any
planning or executable preparation. This is an offline diagnostic, not a
specialization backend. Reproduction commands and retained results are in the
same research record.

`profile_bt27_boundary_reuse` is a separate opt-in native target that composes
exact physical corrections with one optimized prefix ending before the decoder.
`study_bt27_boundary_reuse.py` prepares its inputs and validates complete outputs
against fresh compilation and optionally Merlin. It also records the common
prefix actions, varying decoder actions, and exploratory costs. Each suffix is
still planned offline; no executor state or runtime topology is shared.

`study_circuit_noise.py` broadens the diagnostic to synthetic gate/readout noise
throughout BT27 and BT81, the existing noisy cultivation fixture, and a Clifford
QEC control. It measures exact-history repetition separately from fixed-history
compilation, validates bounded-width cases against Merlin and the Clifford
control against Stim, and inspects wide cases without execution. It does not
implement a cache. The research record documents noise policies and limitations.

`bt27_circuit_corrections.py` precomputes the physical fault dependencies for the
full synthetic BT27 noise model. `study_bt27_circuit_reuse.py` evaluates them as
boundary Cliffords, record flips, and final Pauli frames, then uses the existing
native boundary diagnostic to reuse the ideal reduced core. It checks complete
outputs against fresh compilation and Merlin, small phase operators against
Aer, and a Clifford control's signed flows against Stim. Construction, suffix
planning, and output restoration remain offline research operations.

`profile_bt27_scored_sampling` and `study_bt27_scored_sampling.py` extend that
experiment through Merlin's complete logical target-scoring tail: six logical
CCZ gates, nine X measurements, and nine observables. The Python research host
draws a new categorical fault history for every shot, composes both the phase
correction and the decoder Pauli before scoring, and returns all 135 records,
72 detectors, and nine observables. Each variant is planned before calling the
ordinary executor; there is no history cache or production sampling API change.

The driver checks fixed histories against fresh compilation and Merlin, checks
an isolated decoder fault against a nine-qubit Aer oracle, and compares fresh
stochastic samples on identical complete sources. The original pre-CCZ noise
and the synthetic gate/readout-noise model both retain Merlin's ideal scoring
tail. Timings include every per-shot fault draw, composition, planning,
execution, and output restoration; setup is reported separately. Linux memory
high-water marks are recorded separately for the Python and native processes.

```bash
cmake --build build-profile --target profile_bt27_scored_sampling -j4
.venv/bin/python tools/profile/study_bt27_scored_sampling.py \
  --binary build-profile/profile_bt27_scored_sampling \
  --merlin-checkout /path/to/pinned/merlin \
  --shots 4096 --fixed-shots 512 --output /tmp/bt27-scored-sampling.json
```

Add `--smoke --shots 128 --fixed-shots 128` for a smaller validation run. The
study requires the pinned Merlin checkout and installed `merlin-sim`, Stim,
Qiskit, and Aer. It limits native active width to 16 before allocation; this is
a research budget, not a production simulator limit. Finite distribution
checks are not precise estimates of rare undetected logical errors.

`validate_bt27_scored_equivalence.py` strengthens the fixed-history check. It
uses the native tool's optional `--export-clifford` diagnostic to reoptimize
both the composed HIR and the original-location reference. Stim supplies their
exact stabilizer flows; eliminating hidden reset outcomes yields a canonical
affine law for every visible record, detector, and logical observable jointly.
The exporter is checked against independently specified Clifford circuits,
and deliberately omitted corrections exercise the comparison's sensitivity.
Both BT27 reductions still depend on Clifft's optimizer, so equality here is
conditional on those rewrites; it is not a fully independent proof of the
unreduced non-Clifford circuits. Sample support checks also do not establish
uniform sampling over the allowed outcomes.

`compare_bt27_scored_rates.py` separately draws a predeclared number of fresh
histories and compares three rates using exact binomial confidence intervals
with simultaneous coverage. It declares absolute equivalence margins before
sampling, checkpoints completed independent chunks, and never extends the
budget based on observed outcomes. Its defaults use 262,144 shots per backend
across eight seeds. Concurrent workers accelerate validation; their timings
are not a performance comparison. SciPy is an additional dependency.

```bash
.venv/bin/python tools/profile/validate_bt27_scored_equivalence.py \
  --binary build-profile/profile_bt27_scored_sampling \
  --merlin-checkout /path/to/pinned/merlin \
  --output /tmp/bt27-scored-exact-equivalence.json

.venv/bin/python tools/profile/compare_bt27_scored_rates.py \
  --binary build-profile/profile_bt27_scored_sampling \
  --merlin-checkout /path/to/pinned/merlin \
  --output /tmp/bt27-scored-rate-equivalence.json --plan-only
.venv/bin/python tools/profile/compare_bt27_scored_rates.py \
  --binary build-profile/profile_bt27_scored_sampling \
  --merlin-checkout /path/to/pinned/merlin \
  --output /tmp/bt27-scored-rate-equivalence.json
```

Use the same binary and protocol when resuming a rate comparison. To test
different margins, inputs, or builds, start a separately declared study with a
new output path. A confidence interval outside a requested equivalence margin
means equivalence to that tolerance was not established; it does not by itself
prove a simulator defect.

`bt27_scored_clifford.py` derives the complete scored Clifford circuit directly
from the existing fault controls. It checks the physical/scoring phase identity
on the ideal preparation's computational support, then precomputes the
quadratic finite-difference responses to the nine-bit logical X shift.
Mixed shift products supply essential Z terms. The resulting physical Clifford
source retains all measurements, feedback, record restoration, detectors, and
scored logical outputs; ordinary Clifft traces, lowers, and samples it without
invoking the HIR optimizer. Construction and planning remain offline research
operations before each ordinary sampling call.

`study_bt27_direct_clifford.py` certifies all 512 shift responses, checks small
operator identities with Aer, matches the 398 retained exact joint output laws
using Stim, and compares fresh noisy samples against Merlin. It exercises
negative controls for unsupported preparation, missing mixed terms, and an
omitted decoder-X contribution. The fresh-shot comparisons are bug checks;
they do not repeat the preceding study's predeclared rate-equivalence margins.
Timings include new circuit construction and planning per shot, with setup
reported separately. There is no history cache.

```bash
.venv/bin/python tools/profile/study_bt27_direct_clifford.py \
  --merlin-checkout /path/to/pinned/merlin \
  --shots 4096 --fixed-shots 256 \
  --output /tmp/bt27-direct-scored-clifford.json
```

This driver requires the same pinned Merlin checkout and installed `merlin-sim`,
Stim, Qiskit, and Aer, plus the retained scored-sampling and exact-equivalence
JSON artifacts. It uses the installed Clifft Python extension and does not
require the native research target. Add `--smoke --shots 128 --fixed-shots 128`
to restrict the fixed-history matrix to five representative cases. The exact
Boolean and small Aer checks run in full even in smoke mode.

`study_bt27_clifford_plans.py` audits all 398 retained histories for changes to
the random/determined measurement pattern and verifies the exact-law ranks.
This detects an obstruction to reusing the ideal affine plan with only sign
changes; matching diagnostic action shapes would not prove compatibility.
It also builds a compact record-parity converter, verifies it on a complete
linear basis, and compares old/new host outputs exactly for the same fresh
histories and seeds. Stage timings separate parsing, tracing, lowering,
sampling, and output restoration. It adds no production plan instructions.

`study_bt81_scored_clifford.py` applies the same reduction to the larger BT81
preparation and decoder with full synthetic physical gate/readout noise and
ideal target scoring. It certifies all 512 logical shifts, exercises a failing
preparation control, and checks fixed histories against direct Stim and
original-location Merlin samples. Fresh stochastic comparisons are bug checks,
not a rate-equivalence study. Both drivers assume all-zero circuit-entry state
and execute the full preparation normally.

```bash
.venv/bin/python tools/profile/study_bt27_clifford_plans.py \
  --merlin-checkout /path/to/pinned/merlin --shots 1024 \
  --output /tmp/bt27-clifford-plans.json
.venv/bin/python tools/profile/study_bt81_scored_clifford.py \
  --merlin-checkout /path/to/pinned/merlin --shots 1024 --fixed-shots 128 \
  --output /tmp/bt81-scored-clifford.json
```

The same pinned Merlin checkout, installed Clifft extension, NumPy, Stim, and
`merlin-sim` are required. The BT27 audit also consumes the retained scored
sampling and exact-law artifacts. These drivers need no new native build.

`scored_clifford_core.py` factors the direct physical Clifford circuit into a
fixed preparation-record law, determined Z readouts, and a smaller diagonal
Clifford on the X-readout coordinates. Exact signed output constraints certify
the decoded product state on every preparation branch. The factorization
preserves the complete output law after the existing fault-dependent record
restoration. BT27 has a 33-qubit core and BT81 an 87-qubit core; both remain
Clifford, with no dense non-Clifford active state.

`study_conditional_clifford_maps.py` matches the retained joint laws and checks
all combinations of a bounded set of binary Pauli-fault switches. It compiles
the resulting affine samplers into Boolean coefficient expressions, validates
their full support and rank, and records expression growth. An exhaustive
three-qubit check compares all 512 diagonal Clifford settings against Aer.
The map evaluator performs only precomputed Boolean and parity operations.
Setup enumerates every selected setting; this is not a scalable compiler for
the full noise model or a production execution change.

```bash
.venv/bin/python tools/profile/study_conditional_clifford_maps.py \
  --merlin-checkout /path/to/pinned/merlin --switches 8 \
  --output /tmp/conditional-clifford-maps.json
```

Use `--switches 2` for a smaller circuit slice; the complete small Aer check
and all saved-history law checks still run. The driver requires the existing
BT27/BT81 JSON references, NumPy, Stim, Qiskit, Aer, the installed Clifft
extension, and the pinned Merlin generator checkout. No new native build is
needed. Formula-evaluation timings exclude fault drawing, correction evaluation,
preparation sampling, output restoration, and exhaustive offline construction.

## Automatic conditional simplification across circuits

`automatic_specialization.py` is a research host around the existing default
algebraic HIR pipeline. It materializes sampled Pauli faults at their original
locations, without identifying circuit families or logical operations. An
optional second path samples the maximal initial Clifford prefix with Stim,
reconstructs its conditional state, retains its records with MPAD, and compiles
the suffix. This tests one preparation boundary; it cannot resume an arbitrary
non-Clifford state at every later measurement.

The noise host accepts independent X/Y/Z errors, depolarizing channels, biased
one- and two-qubit Pauli channels, and symmetric readout flips, with different
probabilities at different locations. Two-qubit channels are categorical.
Inter-location correlated channels and unsupported instructions are rejected.
Sampling and compilation occur outside the existing executor. There is no
history cache, postselection, or fault-count truncation.

`study_automatic_specialization.py` compares ordinary noisy compilation,
noise-free compilation, sampled faults, and sampled preparation outcomes on
distillation, cultivation, BT, and constructed controls. Direct-X variants
retain actual logical sampling without an ideal verification tail. It records
T counts, active widths, compiler limits, bounded stress histories, fresh-shot
costs, batch costs, and Merlin acceptance and timings. Wide plans are inspected
without allocating their dense execution state. Timing shots do not establish
statistical equivalence.

```bash
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_automatic_specialization.py \
  --merlin-checkout /path/to/pinned/merlin --shots 32 --batch-shots 1024 \
  --output /tmp/automatic-specialization.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_automatic_specialization.py \
  --merlin-checkout /path/to/pinned/merlin \
  --output /tmp/automatic-specialization-validation.json
```

The validator checks exact small noise mixtures and measurement-conditioned
laws against Aer, complete joint record laws for selected distillation fault
histories against Aer, wire renaming, an unreduced noncommuting control, and
stochastic Clifford behavior against Stim. Larger cultivation checks compare
fixed-history moments with Merlin and are explicitly not equivalence proofs.
The same pinned Merlin checkout and the existing development dependencies are
required. No native build or production execution change is introduced.

## Reusing conditional phase analysis

`shared_phase_specialization.py` is a research compiler for a Clifford
preparation followed by CNOT and diagonal phase operations with terminal Pauli
readouts. It derives affine support and a weighted phase polynomial, then
precomputes fault/outcome-dependent Clifford responses. It recognizes no circuit
family or encoder/decoder pattern. Each shot evaluates those responses and
traces/lowers only the reduced computation, retaining any fixed non-Clifford
remainder. No phase optimization, history lookup, or tableau planning occurs
in Clifft's executor; all conditional construction is in the research host.

`study_shared_phase_specialization.py` records shared setup, fresh per-shot
stages, ordinary batch costs, explicit fallbacks, and a measured cost comparison.
The comparison is a research diagnostic, not a production selection policy.
`validate_shared_phase_specialization.py` checks small general preparations and
wire renamings against Aer, full distillation record laws against Aer, prefix
fault responses against Stim, and scored BT27 conditional laws against the
existing optimizer/exporter. It normalizes deterministic MPAD before requesting
flow constraints to avoid the installed Stim 1.16.0 sign-ordering issue; a
256-pattern literal control checks the normalized oracle. Timing-shot moment
checks do not establish rate equivalence.

```bash
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_shared_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin --shots 256 \
  --output /tmp/shared-phase-specialization.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_shared_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin \
  --binary build-profile/profile_bt27_scored_sampling \
  --output /tmp/shared-phase-validation.json
```

The same pinned Merlin checkout and Python development dependencies are
required. The validation additionally uses the existing native scored-sampling
diagnostic's `--export-source` mode. The default panel omits BT81; `--cases`
selects named panel entries. Execution checks width before lowering (default
budget 12), and unsupported resets or noncommuting operations inside a phase
region produce an explicit fallback. See the
[research report](../../research/conditional_clifford/SHARED_PHASE_SPECIALIZATION.md)
for the supported interface and limits of each oracle.

## Quantum continuations after regional reduction

`regional_phase_specialization.py` discovers the first eligible phase region
after a Clifford preparation and stops before the first incompatible operation.
It uses the shared analysis in state-preserving mode: restore the physical
wire labels, affine encoding, and fault/outcome-dependent offsets, retain
preparation records, and append the remaining circuit. Later measurements,
feedback, resets, and noncommuting gates run in ordinary Clifft. This is
source composition in the research host, without native executor continuation
or a second reduction on an unsupported non-Clifford input state.

`validate_regional_phase_specialization.py` compares full joint record/state
density matrices at the exit and after the continuation against Aer, and
checks exact visible-record probabilities using Clifft replay. It includes
complete physical statevectors for bounded protocol examples, intentional
coherence/correlation errors that a weaker oracle would miss, and conditional
scored BT27 output laws relative to the existing optimizer/exporter.

`study_regional_phase_specialization.py` records region and complete-circuit
widths, stress histories, fresh-shot stages, and explicit budget stops. A
constructed parity-echo family varies the continuation's non-Clifford work to
show both retained and lost width savings. Wide programs are inspected before
lowering; sampled cases retain every original output record.

```bash
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_regional_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin \
  --binary build-profile/profile_bt27_scored_sampling \
  --output /tmp/regional-phase-validation.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_regional_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin --shots 64 \
  --output /tmp/regional-phase-specialization.json
```

Dependencies are the same as the shared-phase study above. Re-run
`validate_shared_phase_specialization.py` when changing their shared renderer.
The [regional report](../../research/conditional_clifford/REGIONAL_PHASE_SPECIALIZATION.md)
distinguishes state-interface evidence, large-circuit conditional law checks,
and fresh execution diagnostics from statistical rate equivalence.

## Chaining regions through Clifford exits

`chained_phase_specialization.py` reuses the first regional analysis, then
rebuilds later analyses only after an exact constructive Clifford-exit
certificate. Faults are drawn once at their original locations. Earlier
measurement records remain fixed, while new prefix uncertainty receives fresh
explicit sampler seeds. A partial non-Clifford exit, unsupported next region,
or region budget stops chaining and retains the complete continuation.
All analysis and sampling of preparation outcomes occur in the research host.

`validate_chained_phase_specialization.py` enumerates conditional branches and
compares the full joint classical/quantum output against Aer, with Clifft replay
for the complete visible-record law. It checks branch-dependent second regions,
noise mixtures, retained records, conservative fallbacks, budgets, and fresh
sampler seeds. Protocol fallback checks require exactly the existing
single-region source; they do not constitute new physical-state proofs for BT.

`study_chained_phase_specialization.py` compares ordinary, one-region, and
chained widths on matched sources. Constructed two-region circuits test where
one reduction remains too wide and chaining permits complete execution.
Stage costs include the currently repeated analysis of later regions.

```bash
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_chained_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin \
  --output /tmp/chained-phase-validation.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_chained_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin --shots 32 \
  --output /tmp/chained-phase-specialization.json
```

The same Python dependencies and pinned Merlin checkout are required; this
validator needs no native exporter. The default limits are eight regions and
execution width twelve. `--max-regions`, `--max-width`, and `--cases` control
the bounded study. See the
[chaining report](../../research/conditional_clifford/CHAINED_PHASE_SPECIALIZATION.md)
for exact-check scope, costs, and the remaining non-Clifford-entry limitation.

## Measurement deferral and realistic boundaries

`deferred_phase_specialization.py` commutes supported phase operations and
independent Pauli channels before earlier measurements only when their quantum
and record dependencies permit it. It retains actual measurements and maps
each original fault site into the new schedule, then invokes the unchanged
state-preserving regional reducer. All work remains in the research host.

`validate_deferred_phase_specialization.py` checks the permutation using
independent wire/record projections, complete small Aer instruments, exhaustive
categorical-noise mixtures, and Clifft record replay. Large scored output laws
are compared with the preceding terminal shared-phase implementation; this is
not an independent large-state equivalence proof.

`study_deferred_phase_specialization.py` compares full scored/direct-X BT,
distillation, and cultivation, with all-gate-noise variants and a BT81 growth
probe. It diagnoses cultivation's logical input using Bell correlations whose
physical representatives exclude explicitly purified reset environments.
Setup, full per-shot stages, response-mask payload, and available Merlin
comparisons are reported separately. The default is 64 shots and execution
width twelve; `--cases`, `--shots`, and `--max-width` bound the run. BT81's
analysis requires substantially more memory and setup time than BT27.

```bash
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_deferred_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin \
  --output /tmp/deferred-phase-validation.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_deferred_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin --shots 64 \
  --output /tmp/deferred-phase-specialization.json
```

The Python dependencies and unchanged pinned Merlin checkout are the same as
the preceding studies. No native exporter is needed. See the
[assessment](../../research/conditional_clifford/DEFERRED_PHASE_SPECIALIZATION.md)
for the remaining non-Clifford-input restriction and validation scope.

## One Pauli rotation through a measured bridge

`one_core_phase_specialization.py` requires the first regional reduction to
leave one T rotation. It carries that rotation on a stabilizer reference through
Clifford gates, commuting measurements, and resets whose wire can be cleared
using a commuting stabilizer. Unsupported crossings retain the conditional
state and ordinary continuation. A diagonal representative can enter a second
phase region. A complete raw-width/T-count check rejects an expanded synthesis
without resampling records already observed by the host.

`validate_one_core_phase_specialization.py` exhaustively enumerates bounded
preparation, measurement, and hidden-reset branches. Independent Aer matrices
check the complete visible-record/physical-state interface. Clifft replay checks
record probabilities before and after ordinary optimization. Noise mixtures,
fallbacks, phase corruption, synthesis expansion, and explicit RNG seeds are
covered. This validator does not require Merlin or a native exporter.

`study_one_core_phase_specialization.py` measures complete cultivation d3/d5,
distillation controls, and two small witnesses. The final source always goes
through the existing HIR optimizer. `--carrier-only` skips the second research
synthesis and tests whether transport alone exposes the useful structure.
The default is 256 shots and execution width twelve; `--cases`, `--shots`, and
`--max-width` bound the study.

```bash
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_one_core_phase_specialization.py \
  --output /tmp/one-core-phase-validation.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_one_core_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin --shots 256 \
  --output /tmp/one-core-phase-specialization.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_one_core_phase_specialization.py \
  --merlin-checkout /path/to/pinned/merlin --shots 256 --carrier-only \
  --output /tmp/one-core-carrier-only.json
```

The shared analyzer's one-wire MPAD fix also requires the preceding full
shared-phase regression command, including its pinned Merlin checkout and
native exporter; use a fresh output path such as
`/tmp/one-core-shared-regression.json`. The
[assessment](../../research/conditional_clifford/ONE_CORE_PHASE_SPECIALIZATION.md)
separates the demonstrated entry capability from synthesis expansion, remaining
large-circuit validation limits, and substantial per-shot host overhead.

## Automatic conditional reduction assessment

`conditional_phase_frontend.py` accepts a complete raw circuit from all-zero
input. It uses dependency-based phase discovery, chains certified Clifford
exits, transports one surviving Pauli rotation when supported, and preserves
an ordinary conditional continuation otherwise. It has no protocol labels,
code matrices, history cache, or backend timing heuristic. Use the existing
HIR optimizer on its complete output before lowering.

Adjacent correlated-error alternatives are kept as one categorical noise
location and one scheduling block, including the union of their physical
supports. The implementation remains a Python research host before native
execution.

The factory panel is exported by the independent benchmark worktree at local
commit `1bcbf960`; its `research/conditional_clifford/BENCHMARK_HANDOFF.md`
documents regeneration. Pass the exported directory and that checkout below.
The validator verifies pinned reference file contents and preserves the
independent rational D/E oracle. It feeds original physical circuits to the
new frontend, including separate coarse-syndrome and state-probe circuits.

```bash
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_conditional_phase_frontend.py \
  --output /tmp/conditional-frontend-validation.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_factory_frontend.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --reference-checkout /path/to/quadcycle-factory-study \
  --merlin-checkout /path/to/pinned/merlin --tail-limit 32 \
  --output /tmp/factory-frontend-validation.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_factory_frontend.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 128 \
  --output /tmp/factory-frontend-study.json
```

The timing driver uses a fresh child process for each backend and input. It
compares ordinary optimized Clifft, per-shot fault materialization, optional
initial Clifford-prefix sampling, this frontend, and Merlin where supported.
A width budget is checked before dense allocation. Whole-process peak RSS
includes Python/import/input costs and is not coefficient-array memory alone.
Every attempted history counts, including width failures; finite record/parity
moment comparisons are bug checks rather than full distribution certificates.
Run the preceding deferred-phase validator as a regression after changing the
scheduler. See the
[assessment](../../research/conditional_clifford/FACTORY_FRONTEND_ASSESSMENT.md)
for results, validation scope, and remaining applicability limits.

## Fixed-preparation compilation reuse

`compiled_prefix_reuse.py` extends the preceding frontend by optimizing its
fixed, unmeasured preparation once. The opt-in `export_optimized_prefix` tool
exports the optimized Pauli T rotations and complete final Clifford frame;
Stim synthesizes that frame before any per-shot execution. All-zero circuit
entry is assumed. Fault corrections, prefix records, and the full continuation
remain trajectory-specific. A Clifford-only tail permits omitting the repeated
phase pass; later magic retains it. Existing Clifford/carrier/fallback routes
are unchanged. No native executor or public API changes are involved.

```bash
cmake --build build-profile --target export_optimized_prefix -j2
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_compiled_prefix_reuse.py \
  --exporter build-profile/export_optimized_prefix \
  --output /tmp/compiled-prefix-validation.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_factory_frontend.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --reference-checkout /path/to/quadcycle-factory-study \
  --merlin-checkout /path/to/pinned/merlin --tail-limit 32 \
  --prefix-exporter build-profile/export_optimized_prefix \
  --output /tmp/compiled-prefix-factory-validation.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_compiled_prefix_reuse.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 128 \
  --exporter build-profile/export_optimized_prefix \
  --output /tmp/compiled-prefix-study.json
```

Regenerate the opt-in build configuration from the Build section first if the
exporter target is absent. Dependencies and benchmark/reference checkouts are
the same as the preceding assessment. The small validator checks complete
preparation states and record/state instruments against Aer. The large
validator keeps its independent reference while changing only the candidate
route. It records how often phase optimization is actually omitted.

The study compares fresh compilation, reuse with all passes, reuse with the
conditional phase-pass omission, and omission without reuse. It alternates
mode order, checks width before allocation, and retains failed attempts.
Timings include complete shot construction and sampling but exclude setup
and correctness audits. `--cases`, `--shots`, and `--max-width` bound the panel.
Peak RSS spans all modes and imported validation dependencies; child RSS
can include inherited parent memory. Neither is isolated native-state storage.

Reproduce the motivating native compilation profile with:

```bash
env OMP_NUM_THREADS=1 CLIFFT_COMPILE_ITERATIONS=256 \
  CLIFFT_CIRCUIT_FILE=research/conditional_clifford/compiled-prefix-profile.stim \
  taskset -c 0 perf record -e cpu-clock:u -F 499 --call-graph dwarf \
  -o /tmp/conditional-compile.perf.data build-profile/profile_compile
perf report -i /tmp/conditional-compile.perf.data --stdio \
  --children --call-graph none --percent-limit 1
```

On the study VM, the generic `perf` launcher lacked matching kernel tools;
`/usr/lib/linux-tools-6.8.0-106/perf` supported the software event successfully.
See the [assessment](../../research/conditional_clifford/COMPILED_PREFIX_REUSE.md)
for ablations, correctness scope, and the remaining per-shot planning cost.

## Parsed and traced prefix reuse

`PreparedPhase` in `prefix_trace_reuse.py` directly emits the stored optimized
preparation and reuses its known gate count and wire mapping. This preserves
the preceding frontend's conditional source and routes while avoiding the
reconstruction and reparsing of the larger original preparation. It can feed
the existing Python compiler without using the native diagnostic below.

`profile_prefix_trace_reuse` is a persistent research worker comparing fresh
compilation, reuse of the parsed prefix, and composition of separately traced
prefix/continuation HIR. The prefix must be unmeasured and deterministic; all
Pauli faults must already be materialized. Composition transforms signed tail
axes through the prefix's inverse frame and preserves the complete final
physical frame. No production tracer, planner, or executor changes are needed.

```bash
cmake --build build-profile --target profile_prefix_trace_reuse -j2
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_prefix_trace_reuse.py \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/prefix-trace-validation.json
env OMP_NUM_THREADS=1 taskset -c 0 .venv/bin/python \
  tools/profile/study_prefix_trace_reuse.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 128 \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/prefix-trace-study.json
```

Regenerate the opt-in CMake configuration first if the worker target is absent.
The exporter and Python dependencies are the same as the preceding study.
Small validation checks complete native states and record laws against Aer,
original-source record/state instruments, and unsupported-interface rejection.

The timing driver alternates rendering and compilation order for matched
histories/seeds. Exact full-HIR comparisons and parity audits run separately
from timed requests. Reported complete costs include the worker IPC boundary;
they are paired within this harness and should not be directly subtracted
from the earlier Python-only costs. Width is checked before dense allocation.
`--cases`, `--shots`, and `--max-width` bound the panel. Native memory comes
from Linux VmHWM rather than the Python parent's inherited child-process peak.
See the [assessment](../../research/conditional_clifford/PREFIX_TRACE_REUSE.md)
for the limited benefit of prefix-only tracing reuse and remaining continuation
and planning costs.

## Fixed Clifford-continuation reuse

`ContinuationPhase` in `continuation_trace_reuse.py` extends `PreparedPhase`
with a fixed continuation template and fault-response controls. Its native
worker mode traces the encoder/Clifford tail once, removes setup-only Pauli
probes, and precomputes packed operation-sign responses. Each shot supplies
the sampled diagonal Clifford boundary correction, selected Pauli generators,
and prefix/readout record flips. Full signed axes, feedback, hidden resets,
annotations, and the outgoing frame are preserved before ordinary planning.

This path retains the preceding frontend when its preparation reuse is
ineligible or the tail is non-Clifford. Noise keeps the original categorical
draw law, including correlated outcomes. There is no history cache, fault-weight
truncation, or native executor change. The optional fourth worker argument is
the setup probe circuit; the existing fresh/parsed/traced modes still work.

```bash
cmake --build build-profile --target export_optimized_prefix profile_prefix_trace_reuse -j2
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_continuation_trace_reuse.py \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/continuation-trace-validation.json
env OMP_NUM_THREADS=1 taskset -c 2 .venv/bin/python \
  tools/profile/study_continuation_trace_reuse.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 128 \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/continuation-trace-study.json
```

The driver compares matched histories and seeds in alternating order against
fresh tracing after direct prefix rendering. It includes complete host/IPC and
native costs, checks width before coefficient allocation, and separately checks
exact HIR equality. Selected stress histories include an outcome at every site;
ordinary timing draws are unconditioned. Validation covers Aer full states,
complete record laws with final-state tomography, exact Clifford laws from
Stim, frontend instruments, and rejection/fallback controls. Rerun
`validate_prefix_trace_reuse.py` for the preceding worker modes as well.

See the [assessment](../../research/conditional_clifford/CONTINUATION_TRACE_REUSE.md)
for setup/storage cost, measured factory/D/E gains, the BT slowdown, and the
remaining per-shot composition and planning cost. This experiment does not
automatically choose a backend or expand the single-rotation carrier.

## Diagonal boundary composition

The `diagonal` mode of `profile_prefix_trace_reuse` applies the sampled
boundary's S powers and CZ terms in a single traversal of each forward
Clifford-frame row. It preserves signed Y-containing axes through the native
raw Pauli phase convention. The inverse frame, fault/readout responses,
continuation composition, optimizer, planner, and executor retain their
preceding behavior. Both older modes remain available for paired comparisons.

```bash
cmake --build build-profile --target profile_prefix_trace_reuse -j2
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_boundary_composition.py \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/boundary-algebra-validation.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_continuation_trace_reuse.py --mode diagonal \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/boundary-continuation-validation.json
env OMP_NUM_THREADS=1 taskset -c 2 .venv/bin/python \
  tools/profile/study_continuation_trace_reuse.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 128 \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --modes fresh continuation diagonal \
  --output /tmp/boundary-composition-study.json
```

The timing driver rotates mode order and reports native boundary, patch,
axis/assembly, and frame-product substages in addition to whole-shot costs.
Pass `mode="diagonal"` to `ContinuationWorker.instantiate` to use the new path;
the default remains `continuation` for reproducing the preceding experiment.
The dedicated validator exhausts the 512 three-qubit diagonal Clifford
corrections, including canceling gates, and checks physical widths 65, 129,
and 193 against complete fresh traces. The existing continuation and prefix
validators cover state/record equivalence and earlier worker modes.

See the [assessment](../../research/conditional_clifford/BOUNDARY_COMPOSITION.md)
for the formula, gains on all four eligible benchmark inputs, and the checkpoint
before further optimization/planning reuse. This adds no production compiler
or executor behavior and does not expand the phase frontend's applicability.

## Optimization and planning reuse assessment

`study_planning_reuse.py` measures the existing `diagonal` construction and
separately requests `audit` snapshots from the worker. Those snapshots carry
synthetic operation provenance through each pass, exact per-pass changed flags,
diagnostic HIR fingerprints, width transitions, and untruncated plan actions.
Comparisons distinguish full actions, actions without constant Boolean terms,
and action-kind/width sequences. No result is used to select or reuse a plan.

```bash
cmake --build build-profile --target profile_prefix_trace_reuse -j2
env OMP_NUM_THREADS=1 taskset -c 2 .venv/bin/python \
  tools/profile/study_planning_reuse.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 128 \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/planning-reuse-study.json
env OMP_NUM_THREADS=1 .venv/bin/python tools/profile/validate_planning_reuse.py \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/planning-reuse-controls.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_continuation_trace_reuse.py --mode audit \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/planning-reuse-validation.json
```

The same driver can profile one native worker without diagnostic requests:

```bash
env OMP_NUM_THREADS=1 taskset -c 2 .venv/bin/python \
  tools/profile/study_planning_reuse.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 256 \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --cases quadcycle-noisy.stim \
  --perf /usr/lib/linux-tools-6.8.0-106/perf \
  --perf-data /tmp/planning-reuse.perf.data \
  --output /tmp/planning-reuse-profile.json
/usr/lib/linux-tools-6.8.0-106/perf report \
  -i /tmp/planning-reuse.perf.data --stdio --children \
  --call-graph none --percent-limit 1 --sort symbol
```

Use an available kernel-compatible `perf` binary on other machines. Profile
mode omits audit requests and records no equivalence-check count; correctness
comes from the separate assessment and validators. It includes native setup
once and excludes Python host work. Inclusive sample shares overlap. Timing
artifacts and native profiles serve different purposes.

See the [assessment](../../research/conditional_clifford/PLANNING_REUSE_ASSESSMENT.md)
for the observed invariant schedules, equal-width plan counterexamples, and
the diagonal-boundary predicate proposed for a future scheduling certificate.
The noisy and no-fault groups retain the preceding noise/outcome laws. Rerun
`validate_prefix_trace_reuse.py` to check older worker paths as well.

## Guarded squeeze schedule reuse

`mode="squeeze"` in `ContinuationWorker.instantiate` uses a setup-time
operation permutation when the diagonal-boundary axis certificate holds and
an exact per-shot comparison shows the earlier optimizers preserved its
scheduling inputs. It otherwise runs ordinary squeezing. Planning remains
fresh in either case. The `squeeze_status` field distinguishes actual reuse,
boundary-predicate fallback, and optimizer-change fallback.

```bash
cmake --build build-profile --target profile_prefix_trace_reuse -j2
env OMP_NUM_THREADS=1 taskset -c 2 .venv/bin/python \
  tools/profile/study_continuation_trace_reuse.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /path/to/pinned/merlin --shots 256 \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --modes fresh diagonal squeeze --output /tmp/squeeze-reuse-study.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_continuation_trace_reuse.py --mode squeeze \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/squeeze-reuse-validation.json
env OMP_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_boundary_composition.py --mode squeeze \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/squeeze-reuse-boundary.json
env OMP_NUM_THREADS=1 .venv/bin/python tools/profile/validate_planning_reuse.py \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/squeeze-reuse-controls.json
env OMP_NUM_THREADS=1 .venv/bin/python tools/profile/validate_prefix_trace_reuse.py \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/squeeze-reuse-prefix-regression.json
```

Reference requests in squeeze mode compare the final optimized HIR and Clifford
frame with a full fresh pipeline, as well as checking raw composition. Ordinary
timed requests omit this diagnostic work. `optimizer_seconds` reports the
permutation application under `StatevectorSqueezePass`; `squeeze_guard_seconds`
separately reports snapshot/comparison/destruction cost, also included in total
optimization time. Common setup reports schedule construction and stored/moved
operation counts. The study retains paired savings in consecutive 32-shot blocks.

The boundary validator's squeeze mode uses a prefix whose rotations survive
the earlier optimizers, and asserts actual reuse over the complete diagonal
group. The diagnostic validator separately tests both fallback conditions,
including an optimizer change with unchanged operation count.
See the [assessment](../../research/conditional_clifford/SQUEEZE_SCHEDULE_REUSE.md)
for measured savings, certificate scope, and the remaining planning cost.

## Probability queries

`profile_probability` uses a unitary-only circuit because measurements,
feedback, noise, and instruments are not eligible for basis-state queries.

```bash
CLIFFT_QUERIES=2000 \
  perf record -F 9999 -g --call-graph dwarf \
  -o perf-prob.data ./build-profile/profile_probability
```

Its generated-circuit defaults are 20 qubits, Clifford depth 200, and 20 T
gates. `CLIFFT_CIRCUIT_FILE`, `CLIFFT_NUM_QUBITS`,
`CLIFFT_CLIFFORD_DEPTH`, and `CLIFFT_T_GATES` override them.

## Inspecting a profile

```bash
perf report -i perf-compile.data --stdio --no-children -n --percent-limit 0.5
perf report -i perf-compile.data --stdio --no-children --sort=srcline --percent-limit 1
perf script -i perf-compile.data > profile.linux-perf.txt
```

Use `perf annotate -i perf-compile.data --stdio --symbol=<symbol>` to inspect
one hot function's generated assembly, and `perf stat -d <command>` for
hardware-counter totals.
