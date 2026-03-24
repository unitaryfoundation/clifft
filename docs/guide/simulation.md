# Simulation

UCC's Schrodinger Virtual Machine (SVM) executes compiled programs to produce measurement results or statevectors.

## Sampling

`ucc.sample()` runs a compiled program for multiple shots and returns measurement, detector, and observable outcomes:

```python
import ucc

program = ucc.compile("""
    H 0
    CNOT 0 1
    M 0 1
""")

# Sample 10,000 shots
meas, det, obs = ucc.sample(program, shots=10000, seed=42)

# meas is a 2D array: (shots x num_measurements)
print(meas.shape)  # (10000, 2)
print(meas[:5])    # First 5 shots
```

`ucc.sample()` always returns a tuple of three numpy arrays: `(measurements, detectors, observables)`. For circuits without detectors or observables, those arrays will have zero columns.

## Statevector Extraction

For exact state inspection (without measurement collapse), use `execute()` and `get_statevector()`:

```python
import ucc

program = ucc.compile("""
    H 0
    CNOT 0 1
""")

# Create a state object sized to the program
state = ucc.State(
    peak_rank=program.peak_rank,
    num_measurements=program.num_measurements,
    num_detectors=program.num_detectors,
    num_observables=program.num_observables,
)

# Execute the program
ucc.execute(program, state)

# Extract the full statevector
sv = ucc.get_statevector(program, state)

# sv is a numpy array of complex amplitudes
print(sv)  # [0.707+0j, 0+0j, 0+0j, 0.707+0j]
```

!!! warning "Statevector scales exponentially"
    `get_statevector()` expands the factored state into a dense $2^n$ vector.
    This is only practical for circuits with a moderate number of qubits.

## Post-Selection (Survivor Sampling)

For circuits with post-selection (e.g., magic state distillation), compile with a `postselection_mask` and use `sample_survivors()`:

<!--pytest.mark.skip-->

```python
import ucc

# Mark detector 0 for post-selection
program = ucc.compile(circuit_text, postselection_mask=[1])

# Only returns stats for shots that pass post-selection
result = ucc.sample_survivors(program, shots=1_000_000, seed=42)
print(f"Survival rate: {result['passed_shots'] / result['total_shots']:.4f}")
print(f"Logical errors: {result['logical_errors']}")
```

The returned dict contains:

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
import ucc

program = ucc.compile("""
    H 0
    CNOT 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-1]
""")

meas, det, obs = ucc.sample(program, shots=10000, seed=42)
# det shape: (10000, num_detectors)
# obs shape: (10000, num_observables)
```

### Syndrome Normalization

By default, detector and observable values are raw measurement parities. For QEC workflows where decoders expect `0` = "no error", use `normalize_syndromes=True` at compile time:

<!--pytest.mark.skip-->

```python
import ucc

program = ucc.compile(
    circuit_text,
    normalize_syndromes=True,
    hir_passes=ucc.default_hir_pass_manager(),
    bytecode_passes=ucc.default_bytecode_pass_manager(),
)

meas, det, obs = ucc.sample(program, shots=10000, seed=42)
# det and obs are now XOR-normalized against the noiseless reference
```

See [Compiling Circuits](compiling.md#syndrome-normalization) for details.

## Deterministic Seeds

All sampling functions accept an optional `seed` parameter for reproducible results:

```python
import ucc

program = ucc.compile("H 0\nM 0")
meas1, _, _ = ucc.sample(program, 100, seed=42)
meas2, _, _ = ucc.sample(program, 100, seed=42)
assert (meas1 == meas2).all()  # Identical
```

If `seed` is omitted (or `None`), UCC uses 256-bit OS hardware entropy.

## Performance

Simulation speed depends on the peak rank $k$ (number of simultaneously active non-Clifford qubits), not the total qubit count. The bytecode optimizer significantly reduces per-shot cost by fusing instructions — see [Optimization Passes](../reference/passes.md) for the full list.

## Simulation Limits

The SVM can handle circuits with many more physical qubits than a naive simulator — the factored state representation means only $2^k$ amplitudes are stored, where $k$ is the peak number of simultaneously active (non-Clifford) qubits.
