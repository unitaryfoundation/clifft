# SVM Profiling Tool

A native C++ harness for profiling Clifft's Schrödinger Virtual Machine with
`perf` (Linux) or other sampling profilers. It compiles a circuit and runs
many shots so the hot loops accumulate enough samples for meaningful analysis.

## Build

The profiler is opt-in (not part of default or coverage builds). Use
`-DCLIFFT_BUILD_PROFILER=ON` and `RelWithDebInfo` for debug symbols at full
optimization (`-O2 -g`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCLIFFT_BUILD_PROFILER=ON
cmake --build build -j$(nproc)
```

Or use the just recipe:

```bash
just profile-build
```

## Quick start

```bash
# Default: 50-qubit random Clifford circuit, 100k shots
./build/profile_svm

# T-gate workload (exercises SVM branch/collide/measure inner loops)
CLIFFT_T_GATES=10 CLIFFT_SHOTS=100000 ./build/profile_svm

# Load a real circuit file
CLIFFT_CIRCUIT_FILE=tests/fixtures/target_qec.stim CLIFFT_SHOTS=100000 ./build/profile_svm

# QV-20: large statevector workload (peak rank 20, array ops dominate)
CLIFFT_CIRCUIT_FILE=tools/bench/fixtures/qv20_seed42.stim CLIFFT_SHOTS=10 ./build/profile_svm
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CLIFFT_CIRCUIT_FILE` | *(none)* | Path to a `.stim` file. Overrides random generation. |
| `CLIFFT_NUM_QUBITS` | 50 | Number of qubits for random circuit |
| `CLIFFT_CLIFFORD_DEPTH` | 5000 | Number of random Clifford gates |
| `CLIFFT_T_GATES` | 0 | Number of T-gates to append |
| `CLIFFT_SHOTS` | 100000 | Number of shots to sample |

## Profiling with `perf`

### Record a profile

```bash
# Deep Clifford (AG_PIVOT dominated)
perf record -F 9999 -g --call-graph dwarf -o perf.data ./build/profile_svm

# QEC circuit
CLIFFT_CIRCUIT_FILE=tests/fixtures/target_qec.stim CLIFFT_SHOTS=100000 \
  perf record -F 9999 -g --call-graph dwarf -o perf.data ./build/profile_svm

# T-gate circuit (SVM inner loop dominated)
CLIFFT_T_GATES=10 CLIFFT_CLIFFORD_DEPTH=500 CLIFFT_SHOTS=100000 \
  perf record -F 9999 -g --call-graph dwarf -o perf.data ./build/profile_svm
```

### View results

```bash
# Flat function-level hotspots (most useful first pass)
perf report -i perf.data --stdio --no-children -n --percent-limit 0.5

# Interactive TUI
perf report -i perf.data

# Annotated assembly for a specific function
perf annotate -i perf.data --stdio \
  --symbol="clifft::execute(clifft::CompiledModule const&, clifft::SchrodingerState&)"

# Map hot addresses to source lines
perf report -i perf.data --stdio --no-children --sort=srcline --percent-limit 1

# Export for external tools (e.g. Firefox Profiler, speedscope)
perf script -i perf.data > profile.linux-perf.txt
```

### Hardware counter stats (no recording overhead)

```bash
perf stat -d ./build/profile_svm
```

### Troubleshooting

If `perf` says "not found for kernel X.Y.Z", the installed `linux-tools`
package doesn't match the running kernel. You can often use the available
version directly:

```bash
# Find the installed perf binary
ls /usr/lib/linux-tools/*/perf

# Use it explicitly
/usr/lib/linux-tools/6.8.0-101-generic/perf record ...
```
