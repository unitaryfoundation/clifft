<!--pytest-codeblocks:skipfile-->

# Scaling Experiment: Clifft and Qiskit Aer

Clifft's factored-state architecture means simulation cost scales with the
circuit's active width $k$, not the total qubit count $N$. The corresponding
active-state dimension is $2^k$. This page compares Clifft against Qiskit Aer's
statevector simulator on two parameter sweeps that isolate each scaling axis.
It is a locally reproducible experiment, not a release benchmark or a
cross-simulator leaderboard. Canonical published comparisons will live in the
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench) project.

## Circuit Design

The benchmark circuit has three parameters:

- **N** — total physical qubits
- **k** — active width (the number of qubits that receive non-Clifford T-gates in this benchmark)
- **t** — total T-gates applied

The circuit places T-gates interleaved with Hadamard and CNOT gates on the
first $k$ qubits, then pads the remaining $N - k$ qubits with a Clifford
entangling layer: Hadamards followed by a CNOT chain across all $N$ qubits.

A dense statevector simulator like Qiskit Aer must allocate $2^N$ complex
amplitudes regardless of circuit structure. Clifft's compiler recognizes that
the Clifford padding can be absorbed into an offline Clifford frame $U_C$, so
its executor only allocates $2^k$ amplitudes.

## Two Sweeps

### Qubit Scaling

Fix $k = 12$ and $t = 20$, sweep $N$ from 16 to 29.

This sweep checks the architectural distinction directly. Clifft's active
array remains $2^{12}$, while Qiskit Aer allocates a dense state over all $N$
qubits. Front-end and fixed per-circuit costs mean measured runtime need not be
perfectly flat or double at every step.

### Active-Width Scaling

Fix $N = 24$ and $t = 40$, sweep $k$ from 8 to 25.

This sweep holds Qiskit Aer's dense state at $2^{24}$ amplitudes while
increasing Clifft's active state from $2^8$ toward the dense limit. Clifft's
coefficient storage and dense kernel work grow as $O(2^k)$.

The experiment demonstrates why active width, rather than total qubit count,
is the useful first-order predictor for Clifft. It does not establish which
simulator is faster for an arbitrary application circuit.

## Prerequisites

```bash
pip install clifft qiskit qiskit-aer matplotlib
```

## Running the Benchmark

The benchmark script is self-contained. It generates circuits in Qiskit,
converts them to Stim format for Clifft, runs both simulators in isolated
subprocesses for clean memory measurement, and produces the plot.

```bash
# Run the benchmark and generate a local CSV and plot
python docs/guide/scripts/run_benchmark.py

# Re-plot after a completed local run
python docs/guide/scripts/run_benchmark.py --plot-only

# Custom output path
python docs/guide/scripts/run_benchmark.py -o my_plot.png
```

The script applies a 120-second timeout and a 6.5 GiB worker memory limit, so
larger dense-state points may time out or run out of memory. Before publishing
results, record the Clifft revision, package versions, CPU, execution target,
thread count, and operating system alongside the generated CSV.

### Why Clifft Is Faster at Low Active Width

The key insight is Clifft's factored-state representation:

$$|\psi\rangle \sim U_C \, P \, (|\phi\rangle_A \otimes |0\rangle_D)$$

Here $\sim$ denotes equality up to global phase.

The compiler absorbs Clifford evolution into an offline frame and chooses the
stabilizer coordinates needed by later operations. Only the active state over
those coordinates (dimension $2^k$) is stored and evolved by the executor.
Noise, measurements, and conditional operations use prepared affine Boolean
expressions; the runtime does not evolve the Clifford tableau. Dormant
Clifford structure therefore adds no coefficient storage.

Qiskit Aer has no such factorization: it must allocate and evolve a full
$2^N$ statevector for every circuit.

!!! note "Interpreting results"
    Exact timings depend on hardware, compiler, package versions, and the
    selected Clifft kernels. Use this experiment to inspect scaling on one
    controlled machine; use `clifft-bench` for maintained published
    comparisons.
