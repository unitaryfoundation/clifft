# Simulation

Clifft's Schrödinger Virtual Machine (SVM) executes compiled programs. The main simulation APIs are:

- `sample()` for ordinary shot-based sampling
- `sample_survivors()` for post-selected sampling
- `execute()` and `get_statevector()` for inspecting small final states
- `sample_k()` and `sample_k_survivors()` for stratified importance sampling

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

`clifft.sample()` returns a `SampleResult` object with `.measurements`, `.detectors`, `.observables`, and `.exp_vals` attributes, each represented as a NumPy array. For circuits without detectors, observables, or expectation-value probes, the corresponding arrays have zero columns.

For Stim-like compatibility, tuple unpacking is also supported:

<!--pytest-codeblocks:cont-->

```python
measurements, detectors, observables = clifft.sample(program, shots=10000, seed=42)
```

Terminology follows Stim's model:

- **Measurements** are the raw results produced by `M`, `MX`, `MY`, and related measurement instructions.
- **Detectors** are declared parity checks over previous measurements using `DETECTOR`.
- **Observables** are logical observable parities declared with `OBSERVABLE_INCLUDE`.

All three are returned per shot. Detectors and observables are empty arrays when the circuit does not declare them.

## State Vector Extraction

For debugging and small circuits, `execute()` and `get_statevector()` let you inspect the final dense state vector:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
""")

state = clifft.State(
    peak_rank=program.peak_rank,
    num_measurements=program.num_measurements,
    num_detectors=program.num_detectors,
    num_observables=program.num_observables,
)

clifft.execute(program, state)
sv = clifft.get_statevector(program, state)

print(sv)  # [0.707+0j, 0+0j, 0+0j, 0.707+0j]
```

!!! warning "State vector extraction is for small circuits"
    `get_statevector()` expands Clifft's factored representation into a dense
    $2^n$ state vector over all physical qubits. This is useful for debugging
    and validation, but it is not the scalable simulation path.

## Detectors, Observables, and Post-Selection

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

By default, detector and observable values are raw measurement parities. This matches the circuit definition, but some QEC workflows expect `0` to mean "matches the noiseless reference" and `1` to mean "differs from the noiseless reference."

Use `normalize_syndromes=True` at compile time to XOR detector and observable outputs against a noiseless reference:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile(
    circuit_text,
    normalize_syndromes=True,
)

result = clifft.sample(program, shots=10000, seed=42)
```

This is often useful before passing detector data to decoders. It also composes with post-selection: detectors that fire in the noiseless reference will not cause spurious discards after normalization.

You can also supply explicit reference parities if you've computed them yourself:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile(
    circuit_text,
    expected_detectors=[1, 0, 0, 1],
    expected_observables=[1],
)
```

!!! note
    `normalize_syndromes=True` is mutually exclusive with manually passing
    `expected_detectors` or `expected_observables`.

See [Compiling Circuits](compilation.md#advanced-reference-syndrome-computation) for computing reference syndromes directly.

### Post-Selection / Survivor Sampling

For circuits with post-selection, compile with a `postselection_mask` and sample with `sample_survivors()`. The mask has one entry per detector: set `mask[i] = 1` to discard shots where detector `i` fires.

!!! important "Mask format"
    `postselection_mask` is a flat list of flags with one element per detector.
    It is not bit-packed. If you are converting a bit-packed Sinter mask, unpack
    it first with `numpy.unpackbits(..., count=num_det, bitorder="little")`.

<!--pytest.mark.skip-->

```python
import clifft

# Mark detectors 0 and 2 for post-selection
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
- `observable_ones` — NumPy array of per-observable error counts

Pass `keep_records=True` to also get the raw `detectors` and `observables` arrays for surviving shots.

Post-selection is implemented as survivor sampling. Marked detectors are checked during execution, and shots are discarded as soon as Clifft can determine that they fail the post-selection condition. This avoids spending full simulation time on shots that cannot contribute to the surviving sample.

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

Each `EXP_VAL` instruction takes one or more Pauli product strings, such as `X0`, `Z0*Z1`, or `X0*Y1*Z2`. Each product produces one column in `result.exp_vals`, with values in `[-1, +1]`.

`EXP_VAL` is non-destructive: it does not collapse the state or affect later measurements. It is also Pauli-frame aware, so noisy operations that change the current frame are reflected in the reported value.

The `Program` object reports `program.num_exp_vals`. Circuits without `EXP_VAL` produce an empty array with shape `(shots, 0)`.

## Deterministic Seeds

All sampling functions accept an optional `seed` parameter for reproducible results:

```python
import clifft

program = clifft.compile("H 0\nM 0")
r1 = clifft.sample(program, 100, seed=42)
r2 = clifft.sample(program, 100, seed=42)
assert (r1.measurements == r2.measurements).all()  # Identical
```

If `seed` is omitted or set to `None`, Clifft uses hardware entropy from the operating system.

## Importance Sampling (Forced k-Faults)

For circuits where logical errors are rare, standard Monte Carlo can require an impractical number of shots. Clifft provides stratified importance sampling via `sample_k` and `sample_k_survivors`, which force exactly `k` physical faults per shot. Results from different `k` strata must be combined using the corresponding Poisson-binomial probability $P(K = k)$.

<!--pytest.mark.skip-->

```python
import clifft

result = clifft.sample_k_survivors(prog, shots=50_000, k=3, seed=42)
# Returns SampleResult with survivor metadata and surviving-shot arrays
```

Key API:

- **`clifft.sample_k(program, shots, k, seed=None)`** -- Like `sample()`, but forces exactly `k` faults. Only valid for programs without post-selection; post-selected programs must use `sample_k_survivors()`. Returns a `SampleResult` with `.measurements`, `.detectors`, and `.observables`.
- **`clifft.sample_k_survivors(program, shots, k, seed=None, keep_records=False)`** -- Like `sample_survivors()`, but forces exactly `k` faults. Returns a `SampleResult` whose arrays contain only surviving shots plus survivor metadata.
- **`program.noise_site_probabilities`** -- 1D NumPy array of per-site fault probabilities, with quantum noise sites followed by readout noise entries. Use this for computing the Poisson-binomial PMF.

See the [Importance Sampling Tutorial](importance-sampling.md) for a complete walkthrough.

## Performance and Limits

Clifft's simulation cost is controlled primarily by the peak active dimension `program.peak_rank`, not by the total number of physical qubits. The SVM stores and updates a dense active state of size $2^k$, where $k$ is the number of simultaneously active qubits in Clifft's factored representation.

This means Clifft can handle circuits with many physical qubits when non-Clifford effects remain localized. It also means performance degrades as `program.peak_rank` grows: circuits with large sustained active dimension approach the cost of dense state-vector simulation.
