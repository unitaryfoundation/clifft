# Native Profiling Tools

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
| `CLIFFT_PROFILE_BATCH_SIZE` | auto | Force a positive packed lane capacity; `1` selects scalar execution |
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
For example, this compares scalar and automatic execution of the aggregate
survivor path with every detector postselected:

```bash
for batch in 1 auto; do
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
