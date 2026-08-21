# Compilation and Probability Profiling Tools

Two native C++ harnesses isolate production compile and strong-simulation
costs for `perf` or another sampling profiler:

- `profile_compile` repeatedly runs parse, trace and HIR optimization,
  coordinate planning, and executable-plan preparation.
- `profile_probability` compiles a unitary circuit and repeatedly queries
  `clifft::basis_probabilities()` over a batch of bitstrings.

## Build

The harnesses are opt-in. `RelWithDebInfo` retains call stacks while preserving
the optimized code paths used for profiling.

```bash
cmake -B build-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCLIFFT_BUILD_PROFILER=ON
cmake --build build-profile --target profile_compile profile_probability -j$(nproc)
```

The equivalent build command is `just profile-build`.

## Compilation

`profile_compile` reports parse, trace and optimization, plan, prepare, and
total time separately. It also prints the executor-independent plan feature
histogram and peak active width used by experimental backends to check workload
coverage. File I/O is outside the timed loop.

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
