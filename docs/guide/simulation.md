# Sampling and Results

!!! tip "Not sure which API to use?"
    Start with [Choose a Workflow](../getting-started/choosing-a-workflow.md).

This page covers ordinary sampling and post-selected survivor sampling, then
explains the fields returned in a `SampleResult`. Exact probabilities,
importance sampling, and leakage or loss answer different scientific questions
and have their own guides.

## Choose sampling behavior

Both sampling functions consume the reusable `Program` returned by
`clifft.compile()`:

- Use `clifft.sample()` when every requested shot should produce one row.
- Compile with a detector `postselection_mask` and use
  `clifft.sample_survivors()` when rejected shots should be discarded.

Threading and packed batching change how these functions execute, not their
statistical meaning. See [CPU Execution and Tuning](cpu-execution.md) after
choosing the sampling behavior you need.

### Ordinary sampling

`clifft.sample()` returns one row per shot:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    M 0 1
""")

result = clifft.sample(program, shots=10_000, seed=42)
assert result.measurements.shape == (10_000, 2)
```

The optional `seed` is mainly useful for repeatable tests and debugging. Leave
it unset to use hardware entropy. A fixed seed replays results when the
execution configuration is unchanged; changing packed-batch settings can
change individual rows. See [Reproducibility](cpu-execution.md#reproducibility)
for the execution details.

For Stim-style compatibility, the measurement, detector, and observable arrays
can also be obtained by tuple unpacking:

```python
import clifft

program = clifft.compile("H 0\nM 0")
measurements, detectors, observables = clifft.sample(
    program,
    shots=100,
    seed=42,
)
assert measurements.shape == (100, 1)
assert detectors.shape == (100, 0)
assert observables.shape == (100, 0)
```

### Post-selected survivor sampling

The `postselection_mask` has one flag per detector. A `1` discards a shot when
that detector fires; a `0` leaves the detector out of the post-selection
condition.

!!! important "Mask format"
    The mask is a flat list of flags, not a bit-packed value. To convert a
    bit-packed Sinter mask, use
    `numpy.unpackbits(..., count=num_det, bitorder="little")`.

```python
import clifft

program = clifft.compile(
    "H 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    postselection_mask=[1],
)
result = clifft.sample_survivors(program, shots=10_000, seed=42)

assert result.total_shots == 10_000
assert result.passed_shots + result.discards == result.total_shots
print(result.passed_shots / result.total_shots)
```

The result includes aggregate survivor information:

- `total_shots`: shots attempted
- `passed_shots`: shots that survived post-selection
- `discards`: shots rejected
- `logical_errors`: number of logical errors among survivors
- `observable_ones`: per-observable error counts

By default, the per-shot arrays are empty. Pass `keep_records=True` to retain
measurement, detector, observable, and expectation-value rows for each
survivor.

Marked detectors are checked during execution. Clifft discards a shot as soon
as it can determine that the shot fails, avoiding later work that cannot
contribute to the surviving sample.

## Understand `SampleResult`

A `SampleResult` exposes four NumPy arrays:

| Field | Contents |
|---|---|
| `measurements` | Raw results from `M`, `MX`, `MY`, and related instructions. |
| `detectors` | Parities declared by `DETECTOR` instructions. |
| `observables` | Logical parities declared by `OBSERVABLE_INCLUDE`. |
| `exp_vals` | Values produced by non-destructive `EXP_VAL` probes. |

For ordinary sampling, every array has one row per shot. An array has zero
columns when the circuit does not declare that result type. Survivor sampling
retains rows only when `keep_records=True`.

### Detectors and observables

By default, detector and observable values are the measurement parities written
in the circuit. Some QEC workflows instead expect `0` to mean "matches the
noiseless reference" and `1` to mean "differs from the noiseless reference."
Set `normalize_syndromes=True` at compile time to use that convention:

```python
import clifft

program = clifft.compile(
    "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    normalize_syndromes=True,
)
result = clifft.sample(program, shots=10, seed=42)

assert not result.detectors.any()
assert not result.observables.any()
```

If the reference values are already known, supply them explicitly:

```python
import clifft

program = clifft.compile(
    "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    expected_detectors=[1],
    expected_observables=[1],
)
result = clifft.sample(program, shots=10, seed=42)

assert not result.detectors.any()
assert not result.observables.any()
```

`normalize_syndromes=True` cannot be combined with `expected_detectors` or
`expected_observables`. Automatic normalization samples one reference shot at
seed 0 after removing stochastic and readout noise. It does not check that
detector or observable parities are deterministic. Use it when the remaining
circuit has fixed parities. Callers that need to inspect the reference can use
[`compute_reference_syndrome()`](../reference/python-api.md#clifft.compute_reference_syndrome).

If those parities vary, the reference is one possible outcome. Its values can
change when optimization reorders random draws, even when the output
distribution is unchanged. Postselection then selects for that sampled
reference, which may differ from the protocol's intended acceptance condition.

Coherent error rotations remain in the reference circuit: an ordinary unitary
gate does not identify whether it implements the protocol or an error. When
building a circuit with coherent errors, retain its intended ideal counterpart
and derive the reference from that circuit.

??? example "Derive references from an ideal circuit"

    Here the intended circuit prepares and measures `|+>`. The added `R_Z`
    rotation models a coherent error, so it belongs only in the circuit being
    sampled.

    ```python
    import clifft

    ideal = """
        RX 0
        MX 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    with_errors = """
        RX 0
        R_Z(0.02) 0
        MX 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    reference = clifft.compute_reference_syndrome(clifft.trace(clifft.parse(ideal)))
    program = clifft.compile(
        with_errors,
        expected_detectors=reference["detectors"],
        expected_observables=reference["observables"],
        postselection_mask=[1],
    )
    assert reference == {"detectors": [0], "observables": [0]}
    ```

    The ideal circuit must have fixed detector and observable parities with the
    same output ordering as the circuit with errors. Preserve it before injecting
    errors or fusing gates; Clifft cannot recover that distinction from a combined
    rotation. If the intended reference values are already known, supply them
    directly through `expected_detectors` and `expected_observables`.

### Expectation values

`EXP_VAL` measures the expectation value of a Pauli product without collapsing
the state or affecting later operations:

```python
import clifft
import numpy as np

program = clifft.compile("""
    H 0
    CNOT 0 1
    EXP_VAL X0*X1 Z0*Z1
    M 0 1
""")
result = clifft.sample(program, shots=100, seed=42)

assert result.exp_vals.shape == (100, 2)
assert np.allclose(result.exp_vals, 1.0)
```

Each Pauli product passed to `EXP_VAL`, such as `X0`, `Z0*Z1`, or
`X0*Y1*Z2`, produces one column with values in `[-1, +1]`. The program reports
the number of columns as `program.num_exp_vals`.
