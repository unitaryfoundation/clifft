# Quick Start

This guide walks through compiling and simulating your first quantum circuit with UCC.

## Your First Circuit

UCC uses [Stim circuit format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md) as input. Here's a Bell state circuit:

```python
import ucc

circuit = """
    H 0
    CNOT 0 1
    M 0 1
"""

# Compile to bytecode
program = ucc.compile(circuit)

# Sample 1000 shots
result = ucc.sample(program, shots=1000, seed=42)
print(result.measurements[:5])  # First 5 shots
```

The output is an array of measurement bitstrings. For a Bell state, you'll see either `00` or `11` with roughly equal probability.

## Non-Clifford Gates

UCC extends Stim's gate set with non-Clifford gates like `T` and `T_DAG`:

```python
import ucc

program = ucc.compile("""
    H 0
    T 0
    H 0
    M 0
""")

result = ucc.sample(program, shots=10000, seed=42)

# Count outcomes
ones = result.measurements[:, 0].sum()
print(f"|1> probability: {ones / len(result.measurements):.3f}")  # ~0.146
```

## Statevector Access

For debugging or verification, you can extract the full statevector:

```python
import ucc

# Compile without measurements
program = ucc.compile("""
    H 0
    CNOT 0 1
""")

# Create state, execute, and extract statevector
state = ucc.State(
    peak_rank=program.peak_rank,
    num_measurements=program.num_measurements,
)
ucc.execute(program, state)
sv = ucc.get_statevector(program, state)
print(sv)  # [0.707+0j, 0+0j, 0+0j, 0.707+0j]
```

## Noisy Circuits

UCC supports Stim's noise channels for error modeling:

```python
import ucc

program = ucc.compile("""
    H 0
    DEPOLARIZE1(0.01) 0
    CNOT 0 1
    DEPOLARIZE2(0.01) 0 1
    M 0 1
""")

result = ucc.sample(program, shots=10000, seed=42)
```

## Next Steps

- [Compiling Circuits](../guide/compiling.md) — the compilation pipeline in detail
- [Simulation](../guide/simulation.md) — sampling, statevectors, and detectors
- [Supported Gates](../reference/gates.md) — full gate reference
