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
