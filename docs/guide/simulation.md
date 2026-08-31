# Sampling and Results

!!! tip "Not sure which API to use?"
    Start with [Choose a Workflow](../getting-started/choosing-a-workflow.md).

This page covers the common fixed-plan sampling workflows:

- Use `clifft.sample()` when every requested shot should produce one output row.
- Compile with a detector `postselection_mask` and use
  `clifft.sample_survivors()` when rejected shots must be discarded.

Both functions consume the reusable `Program` returned by `clifft.compile()`
and return a `SampleResult`. Exact queries, fixed-fault importance sampling,
and noncomputational trajectories are separate scientific workflows linked
under [Specialized workflows](#specialized-workflows).

Batching, threading, and hardware backends are execution choices. They do not
change which sampling function matches the circuit semantics.

## Ordinary Sampling

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

## Detectors and Observables

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

See [Compiling Circuits](compilation.md#reference-syndrome-computation) for
computing reference syndromes directly.

## Post-Selection / Survivor Sampling

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

With the default `keep_records=False`, the per-shot `.measurements`,
`.detectors`, `.observables`, and `.exp_vals` arrays are empty. Pass
`keep_records=True` to retain those arrays with one row per surviving shot.

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

Set an explicit seed when exact replay is useful for debugging or tests. If
`seed` is omitted or set to `None`, Clifft uses operating-system entropy; this
is normally preferable for independent production simulations.

Seeded replay is guaranteed within the same sampling workflow and execution
configuration.
Changing the thread count alone does not change its rows, but scalar and packed
execution use separate random streams, and different packed capacities may do
so as well. Results from every supported execution strategy remain
statistically equivalent.

## CPU Execution Settings

The fixed-plan samplers support automatic packed batching, cross-shot workers,
and intra-shot OpenMP workers. Most users should keep `batch_size="auto"` and
set only a total `threads` budget when more CPU parallelism is needed.

See [CPU Execution and Tuning](cpu-execution.md) for the compatibility matrix,
automatic policies, expert overrides, reproducibility boundaries, and memory
tradeoffs.

## Specialized Workflows

These workflows answer different scientific questions. They are not CPU
execution modes, and their compatibility rules determine which API to call.

### State Vector Extraction

`clifft.get_statevector()` expands the final pure unitary state over all
physical qubits. It is a debugging and validation path, is currently limited to
10 qubits, and returns the state only up to global phase. See the
[Quick Start](../getting-started/quickstart.md#state-vector-access) for a minimal
example.

### Exact Probabilities

Use `clifft.basis_probabilities()` to query computational-basis probabilities
for a unitary program without measurements. Use
`clifft.record_probabilities()` for exact joint probabilities of measurement
records in a noiseless circuit. Both reject noise, detectors, observables, and
post-selection. See [Strong Simulation: Exact Probabilities](strong-simulation.md)
for examples, detailed limits, and cost tradeoffs.

### Importance Sampling (Forced k-Faults)

`clifft.sample_k()` and `clifft.sample_k_survivors()` condition every shot on
exactly `k` physical faults. Use the survivor variant when the program has a
post-selection mask. Results from different strata must be combined with their
fault-count probabilities; a single stratum is not an unconditional error-rate
estimate. See the [Importance Sampling Tutorial](importance-sampling.md) for the
complete statistical workflow.

### Leakage and Loss Trajectories

Experimental `clifft.noncomp.sample()` accepts a circuit and noncomputational
model together. It compiles trajectory-specific continuations internally, so
there is no separate `clifft.compile()` step. See
[Leakage and Loss](leakage-and-loss.md) for supported models and limits.

## Performance and Limits

Clifft's simulation cost is controlled primarily by `program.peak_active_width`,
rather than by the total number of physical qubits. The executor stores and
updates a dense active state of dimension $2^k$, where $k$ is the number of
simultaneously active stabilizer coordinates.

This means Clifft can handle circuits with many physical qubits when
non-Clifford effects remain localized. It also means performance degrades as
`program.peak_active_width` grows: circuits with large sustained active width approach
the cost of dense state-vector simulation.
