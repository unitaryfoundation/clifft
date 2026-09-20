# Native Profiling Tools

The [physical cultivation study](research/cultivation_corpus.md) adds a pinned
author circuit corpus, independent reference checks, source-mapped active-width
diagnostics, and public-API throughput measurements. Its optional
`profile_cultivation` target is built with `CLIFFT_BUILD_PROFILER=ON`.

The [SOFT d7 investigation](research/soft_cultivation.md) adds a pinned physical
T workload, an exact noiseless detector witness, and offline trajectory-state
diagnostics using the optional `snapshot_cultivation` target. Its cultivation
contract remains unverified; the report separates that limitation from timing.

The [coherent Clifford gadget experiment](research/clifford_gadget.md) reduces
the physical parity gadgets to two coherent Clifford terms and checks complete
fault-conditioned histories. Its offline reference demonstrates the reduction
but is slower than ordinary Clifft; Cirq remains an optional research dependency.

The [fixed-contraction follow-up](research/gadget_contraction.md) compiles the
terminal interference and sampling marginals into small reusable schedules.
It adds the optional `profile_gadget_contraction` arithmetic benchmark and
validates generated records against native full-circuit replay. Those kernel
timings exclude the prefix and compiler handoff.

The [native handoff and sampler](research/native_terminal.md) connects ordinary
Clifft prefix execution to the terminal schedules with compiler-certified
boundary expressions and precompiled fault maps. Its optional
`sample_terminal_gadget` target measures complete attempts with fresh noise;
it is an experimental profiling path, not a public sampling mode.

The [author-corpus coverage audit](research/terminal_coverage.md) finds that
the terminal prototype currently declines all four validated author inputs.
It identifies the missing logical measurement, feedback, and flag support,
and separates a distance-scaling benchmark from rare-error estimation.

The [geometric gadget sweep](research/gadget_family.md) constructs a reproducible
d3/d5/d7/d9 family with ideal encoded inputs and noisy physical parity trees.
It measures the crossover against ordinary Clifft, validates small cases
against dense physical projectors, and bounds dense baseline allocation.

The [complete-protocol source audit](research/protocol_sources.md) preserves
the proposed Takada f3/f5/f7 reconstruction, exact source revisions, noise
and acceptance conventions, and known export traps. The geometric sweep is
not a complete cultivation protocol.

The [reconstructed fold-transversal f3 protocol](research/folded_msc.md)
now includes physical injection, growth, noisy logical checks and terminal
error detection. It adds an independent Aer trajectory audit, exhaustive
single-fault checks, and an ordinary-Clifft baseline.

The [complete f5 extension](research/folded_msc_f5.md) adds validated unitary
growth and flagged folded checks. Ordinary Clifft reaches active width 22;
an offline coherent-Clifford oracle agrees on complete physical histories.
Production compilation of this reduction, f7 growth, and decoder-based
final error correction remain future work.

The [scheduled-baseline refresh](research/scheduled_baseline.md) compares
identical folded and synthetic inputs with PR 471 explicitly enabled, using
both bounded and unbounded search. It records compilation, complete-attempt
throughput, isolated-process memory and physical replay checks, superseding
the earlier default-pipeline-only performance comparisons.

The [folded-check contraction prototype](research/folded_contraction.md)
samples two coherent folded checks and returns a logical-state continuation
using fixed native arithmetic. Complete physical histories and continued
logical phases are checked for f3/f5. Fault binding is still offline, so its
kernel timings are not complete-attempt speedups.

The [complete native folded sampler](research/folded_protocol.md) now includes
the Clifft prefix, fresh physical noise, native table binding, noisy syndrome
continuation and final logical evaluation. It validates 144 fresh complete
histories and compares both full and early-rejected attempts against PR 471.
The f5 gain survives that comparison; f3 still needs the ordinary path.

The [complete f7 cultivation extension](research/folded_protocol_f7.md)
tests the larger circuit before fallback or production consolidation. It
includes d5-to-d7 growth, the eight-qubit verified cat, and the complete
physical cultivation prefix. Fresh f7 attempts pass independent history
checks while avoiding an ordinary width-44 coefficient array.

The [other non-Clifford candidate audit](research/non_clifford_candidates.md)
compares available distillation and Clifford-measurement families, identifies
Clifford surrogate exports, and records an ordinary-Clifft active-width check
of the existing 85-qubit distillation baseline.

The [folded sampler memory experiment](research/folded_memory.md) shares
execution workspaces and compacts fixed lookup tables. It reduces the f7
numeric payload from 79.06 MiB to 10.43 MiB, preserves independently checked
physical histories, and records the runtime tradeoff against the same
PR-471-enabled sampler before compression.

The [native f5-to-f7 integration](research/folded_native_growth.md) connects
both folded kernels through a compiler-certified noisy growth transfer.
Complete f7 histories pass independent replay, the dense prefix allocation
is removed, and full attempts improve to about 194 ms at about 40 MiB RSS.
Use `--native-growth` when compiling an f7 research bundle.

The [folded block cleanup](research/folded_cleanup.md) separates blocks from
protocol orchestration, emits compact plans directly, and documents the
remaining work for a transparent Clifft specialization pass. Regenerate research
bundles with the current compiler; the cleanup uses explicit versioned formats.

Three native C++ harnesses isolate production compile, sampling, and
strong-simulation costs for `perf` or another sampling profiler:

- `profile_compile` repeatedly runs parse, trace and HIR optimization,
  coordinate planning, and executable-plan preparation.
- `profile_probability` compiles a unitary circuit and repeatedly queries
  `clifft::basis_probabilities()` over a batch of bitstrings.
- `profile_sample` compiles a circuit once and repeatedly samples it through
  the public C++ path.

## Build

The harnesses are opt-in. `RelWithDebInfo` retains call stacks while preserving
the optimized code paths used for profiling.

```bash
cmake -B build-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCLIFFT_BUILD_PROFILER=ON
cmake --build build-profile --target profile_compile profile_probability profile_sample -j$(nproc)
```

The equivalent build command is `just profile-build`.

## Compilation

`profile_compile` reports parse, trace and optimization, plan, prepare, and
total time separately. File I/O is outside the timed loop.

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
