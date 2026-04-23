# Simulation

Clifft's Schrodinger Virtual Machine (SVM) executes compiled programs to produce measurement results or state vectors.

## Sampling

`clifft.sample()` runs a compiled program for multiple shots and returns a `SampleResult` with measurement, detector, and observable outcomes:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    M 0 1
""")

# Sample 10,000 shots
result = clifft.sample(program, shots=10000, seed=42)

# result.measurements is a 2D array: (shots x num_measurements)
print(result.measurements.shape)  # (10000, 2)
print(result.measurements[:5])    # First 5 shots
```

`clifft.sample()` returns a `SampleResult` object with `.measurements`, `.detectors`, `.observables`, and `.exp_vals` attributes (each a numpy array). For circuits without detectors, observables, or expectation value probes, those arrays will have zero columns. Tuple unpacking (`m, d, o = clifft.sample(...)`) is supported for backward compatibility.

## State Vector Extraction

For exact state inspection (without measurement collapse), use `execute()` and `get_statevector()`:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
""")

# Create a state object sized to the program
state = clifft.State(
    peak_rank=program.peak_rank,
    num_measurements=program.num_measurements,
    num_detectors=program.num_detectors,
    num_observables=program.num_observables,
)

# Execute the program
clifft.execute(program, state)

# Extract the full state vector
sv = clifft.get_statevector(program, state)

# sv is a numpy array of complex amplitudes
print(sv)  # [0.707+0j, 0+0j, 0+0j, 0.707+0j]
```

!!! warning "State vector scales exponentially"
    `get_statevector()` expands the factored state into a dense $2^n$ vector.
    This is only practical for circuits with a moderate number of qubits.

## Post-Selection (Survivor Sampling)

For circuits with post-selection (e.g., magic state distillation), compile with a `postselection_mask` and use `sample_survivors()`:

!!! important "Mask format: one flag per detector, not bit-packed"
    `postselection_mask` is a **flat list of `uint8` flags with exactly one
    element per detector**.  Set `mask[i] = 1` to post-select on detector *i*,
    or `0` to leave it as a normal detector.  This is **not** a bit-packed
    byte array — each element maps directly to one detector index.

    Sinter uses a different (bit-packed) convention.  If you are converting
    from a Sinter `postselection_mask`, unpack it first with
    `numpy.unpackbits(..., count=num_det, bitorder="little")`.

<!--pytest.mark.skip-->

```python
import clifft

# Mark detectors 0 and 2 for post-selection (one flag per detector)
program = clifft.compile(circuit_text, postselection_mask=[1, 0, 1])

# Only returns stats for shots that pass post-selection
result = clifft.sample_survivors(program, shots=1_000_000, seed=42)
print(f"Survival rate: {result.passed_shots / result.total_shots:.4f}")
print(f"Logical errors: {result.logical_errors}")
```

The returned `SampleResult` object contains:

- `total_shots` — number of shots attempted
- `passed_shots` — number that survived post-selection
- `discards` — number discarded
- `logical_errors` — count of logical errors
- `observable_ones` — numpy array of per-observable error counts

Pass `keep_records=True` to also get the raw `detectors` and `observables` arrays for surviving shots.

This is critical for distillation circuits with >99% discard rates — doomed shots are fast-failed immediately, avoiding wasted computation.

## Detector and Observable Results

Circuits with `DETECTOR` and `OBSERVABLE_INCLUDE` annotations automatically produce detector and observable results alongside measurements:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-1]
""")

result = clifft.sample(program, shots=10000, seed=42)
# result.detectors shape: (10000, num_detectors)
# result.observables shape: (10000, num_observables)
```

### Syndrome Normalization

By default, detector and observable values are raw measurement parities. For QEC workflows where decoders expect `0` = "no error", use `normalize_syndromes=True` at compile time:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile(
    circuit_text,
    normalize_syndromes=True,
    hir_passes=clifft.default_hir_pass_manager(),
    bytecode_passes=clifft.default_bytecode_pass_manager(),
)

result = clifft.sample(program, shots=10000, seed=42)
# result.detectors and result.observables are now XOR-normalized against the noiseless reference
```

See [Compiling Circuits](compiling.md#syndrome-normalization) for details.

## Expectation Values

`EXP_VAL` is a non-destructive probe that computes the expectation value of a Pauli product operator on the current state, without collapsing it. This is useful for observing properties of the quantum state mid-circuit without affecting subsequent operations.

```python
import clifft
import numpy as np

program = clifft.compile("""
    H 0
    CNOT 0 1
    EXP_VAL X0*X1 Z0*Z1
    M 0 1
""")

result = clifft.sample(program, shots=1000, seed=42)

# result.exp_vals is a 2D array: (shots x num_exp_vals)
print(result.exp_vals.shape)  # (1000, 2)
print(np.mean(result.exp_vals, axis=0))  # [1.0, 1.0] for Bell state
```

Each `EXP_VAL` instruction takes one or more Pauli product strings (e.g., `X0`, `Z0*Z1`, `X0*Y1*Z2`). Each product produces one column in `result.exp_vals`, with values in the range [-1, +1].

Key properties:

- **Non-destructive**: `EXP_VAL` does not collapse the state or consume qubits. Measurements after an `EXP_VAL` are unaffected.
- **Pauli frame aware**: In noisy circuits, the expectation value accounts for the current Pauli frame. For example, a `Z_ERROR` before `EXP_VAL X0` will flip the sign because Z anti-commutes with X.
- **Per-shot values**: Each shot produces an independent expectation value. For Clifford states this is deterministic ($\pm 1$ or $0$); for non-Clifford states (after T gates) or noisy circuits, values vary across shots.

The `Program` object reports `program.num_exp_vals` for the total number of probes. Circuits without `EXP_VAL` produce an empty array with shape `(shots, 0)`.

## Deterministic Seeds

All sampling functions accept an optional `seed` parameter for reproducible results:

```python
import clifft

program = clifft.compile("H 0\nM 0")
r1 = clifft.sample(program, 100, seed=42)
r2 = clifft.sample(program, 100, seed=42)
assert (r1.measurements == r2.measurements).all()  # Identical
```

If `seed` is omitted (or `None`), Clifft uses 256-bit OS hardware entropy.

## Importance Sampling (Forced k-Faults)

For circuits where logical errors are extremely rare (e.g., QEC at low physical error rates), standard Monte Carlo requires an impractical number of shots. Clifft provides **stratified importance sampling** via `sample_k` and `sample_k_survivors`, which force exactly `k` physical faults per shot and weight the results by the exact Poisson-Binomial probability $P(K = k)$.

<!--pytest.mark.skip-->

```python
import clifft

result = clifft.sample_k_survivors(prog, shots=50_000, k=3, seed=42)
# Returns SampleResult with survivor metadata and surviving-shot arrays
```

Key API:

- **`clifft.sample_k(program, shots, k, seed=None)`** -- Like `sample()`, but forces exactly `k` faults. Only valid for programs without post-selection; post-selected programs must use `sample_k_survivors()`. Returns a `SampleResult` with `.measurements`, `.detectors`, and `.observables`.
- **`clifft.sample_k_survivors(program, shots, k, seed=None, keep_records=False)`** -- Like `sample_survivors()`, but forces exactly `k` faults. Returns a `SampleResult` whose arrays contain only surviving shots plus survivor metadata.
- **`program.noise_site_probabilities`** -- 1D numpy array of per-site fault probabilities (quantum noise sites followed by readout noise entries). Use for computing the Poisson-Binomial PMF.

Results from these functions must be combined across strata with $P(K=k)$ weights. See the [Importance Sampling Tutorial](importance-sampling.md) for a complete walkthrough.

## Performance

Simulation speed depends on the peak active dimension $k$ (number of simultaneously active non-Clifford qubits, exposed as `program.peak_rank`), not the total qubit count. The bytecode optimizer significantly reduces per-shot cost by fusing instructions -- see [Optimization Passes](../reference/passes.md) for the full list.

## Simulation Limits

The SVM can handle circuits with many more physical qubits than a naive simulator -- the factored state representation means only $2^k$ amplitudes are stored, where $k$ is the peak number of simultaneously active (non-Clifford) qubits.
