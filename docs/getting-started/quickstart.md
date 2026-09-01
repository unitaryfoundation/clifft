# Quick Start

This guide walks through compiling and simulating your first quantum circuit
with Clifft. It uses a Stim circuit with Clifft extensions; see
[Circuit Inputs](integrations.md) if your circuit starts in
OpenQASM 2, Qiskit, or Cirq.

## Your First Circuit

Clifft uses [Stim circuit format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md) as input. Here's a Bell state circuit:

```python
import clifft

circuit = """
    H 0
    CNOT 0 1
    M 0 1
"""

# Compile an executable sampling plan
program = clifft.compile(circuit)

# Sample 1000 shots
result = clifft.sample(program, shots=1000)
print(result.measurements[:5])  # First 5 shots
```

The output is an array of measurement bitstrings. For a Bell state, you'll see either `00` or `11` with roughly equal probability.

## Non-Clifford Gates

Clifft extends Stim's gate set with non-Clifford gates like `T` and `T_DAG`:

```python
import clifft

program = clifft.compile("""
    H 0
    T 0
    H 0
    M 0
""")

result = clifft.sample(program, shots=10000)

# Count outcomes
ones = result.measurements[:, 0].sum()
print(f"|1> probability: {ones / len(result.measurements):.3f}")  # ~0.146
```

## Measurement, Detector, and Observable Results

Sampling always returns measurement results. Circuits can also declare
detectors and logical observables, which are returned alongside the
measurements for every shot:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-1]
""")

result = clifft.sample(program, shots=1000, seed=42)
print(result.measurements.shape)  # (1000, 2)
print(result.detectors.shape)     # (1000, 1)
print(result.observables.shape)   # (1000, 1)
```

Measurements are raw circuit outcomes. Detectors are parities of earlier
measurements, commonly used as error syndromes, while observables track declared
logical outcomes.

## Noisy Circuits

Clifft supports Stim's noise channels for error modeling:

```python
import clifft

program = clifft.compile("""
    H 0
    DEPOLARIZE1(0.01) 0
    CNOT 0 1
    DEPOLARIZE2(0.01) 0 1
    M 0 1
""")

result = clifft.sample(program, shots=10000, seed=42)
```

## Next Steps

- [Choose a Workflow](choosing-a-workflow.md) - select the API that matches the result you need
- [Circuit Inputs](integrations.md) - bring a circuit from another format or framework
- [Sampling and Results](../guide/simulation.md) - ordinary shots, detectors, observables, and post-selection
- [Leakage and Loss](../guide/leakage-and-loss.md): noncomputational trajectory sampling
- [Supported Gates](../reference/gates.md) - full gate reference
- [Compiling Circuits](../guide/compilation.md) - inspect or customize the compilation pipeline
