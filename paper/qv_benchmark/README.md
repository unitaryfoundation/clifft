# Quantum Volume Benchmark

Benchmarks Clifft against Qiskit-Aer, Qulacs, qsim, and Qrack on random
**Quantum Volume** circuits, scaling qubit count from 6 to 26+. These
circuits are dense with non-Clifford gates, showcasing the regime where
Clifft's compiled stabilizer VM excels.

## Dependencies

```
numpy
pandas
matplotlib
clifft
qiskit
qiskit-aer
qulacs
qsimcirq
qiskit-qrack
```

Not all simulators are required — use `--simulators` to select a subset.

## Running

All commands run from the `paper/` directory (see top-level
[`README.md`](../README.md) for setup).

```bash
# Quick test
uv run python -m qv_benchmark --min-q 6 --max-q 12 --repeats 1 --simulators clifft,qiskit

# Specific qubit counts
uv run python -m qv_benchmark --qubits 10,14,18,22

# Single simulator
uv run python -m qv_benchmark --simulators clifft

# Recommended EC2 run (24-core, 96 GB RAM)
uv run python -m qv_benchmark \
    --min-q 6 --max-q 32 --step 2 \
    --mem-limit-gb 80 --timeout 600 \
    --repeats 3
```

Results are written to `results.csv` in this directory.

Each `(simulator, N, seed)` combination runs in an isolated subprocess to
capture peak RSS memory and prevent GC drift between runs.

## Plotting

```bash
python plot_qv.py
```

Produces a log-scale execution time vs qubit count plot.

## Validation

```bash
python validate_hop.py
```

Validates Clifft's statevector output against Qiskit-Aer using fidelity
checks and Heavy Output Probability (HOP) computation.

## CLI options (run_benchmark.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--min-q` | `6` | Minimum qubit count |
| `--max-q` | `26` | Maximum qubit count |
| `--step` | `2` | Step size for qubit range |
| `--qubits` | — | Explicit comma-separated qubit counts (overrides min/max) |
| `--repeats` | `3` | Repetitions per configuration |
| `--simulators` | `clifft,qiskit,qulacs,qsim,qrack` | Comma-separated backends |
| `--mem-limit-gb` | `6.0` | Per-worker memory cap (RLIMIT_AS) |
| `--timeout` | `300` | Per-worker timeout in seconds |
| `--output` | `results.csv` | Output CSV path |
| `--seed` | `42` | Base RNG seed |

## Files

| File | Description |
|------|-------------|
| `run_benchmark.py` | Orchestrator — spawns subprocess workers, collects CSV |
| `worker.py` | Subprocess worker for a single (simulator, N, seed) run |
| `generator.py` | Random QV circuit generation (QASM 2.0) |
| `qasm_adapter.py` | Converts QASM to Clifft/stim, Qulacs, Cirq formats |
| `plot_qv.py` | Publication-ready scaling plot |
| `validate_hop.py` | Statevector fidelity + HOP validation |
