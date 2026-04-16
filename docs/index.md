---
hide:
  - navigation
---

# Clifft

<p style="font-size: 1.2em;">
A multi-level compiler and Schrodinger Virtual Machine for quantum circuits.
</p>

[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)

---

## What is Clifft?

Clifft compiles universal quantum circuits (Clifford + T and beyond) into a compact bytecode representation that is executed by a high-performance virtual machine. It solves the exponential memory wall of non-Clifford simulation by *factoring* the quantum state into deterministic coordinate transformations and probabilistic complex amplitudes.

$$|\psi\rangle = \gamma \, U_C \, P \, \Big( |\phi\rangle_A \otimes |0\rangle_D \Big)$$

Only the $2^k$ amplitudes of qubits in active superposition are stored — not $2^n$ for all $n$ qubits. For circuits where non-Clifford entanglement is bounded (e.g., magic state distillation), this yields exponential memory savings.

## Quick Example

```python
import clifft

# Compile a circuit from Stim format
program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")

# Sample measurement outcomes
result = clifft.sample(program, shots=1000, seed=42)
print(result.measurements[:5])  # First 5 shots
```

## Key Features

<div class="grid cards" markdown>

- :material-lightning-bolt: **Multi-Level Compilation**

    Clifford gates are absorbed at compile time. The VM only executes non-Clifford operations and measurements.

- :material-memory: **Factored State**

    Memory scales as $2^k$ where $k$ is the number of active (non-Clifford) qubits, not $2^n$ total qubits.

- :material-speedometer: **RISC Bytecode VM**

    Cache-aligned 32-byte instructions. Single memory allocation. No dynamic resizing in the hot loop.

- :material-format-list-checks: **Stim Compatible**

    Parses Stim circuit format with support for noise channels, detectors, and repeat blocks.

</div>

## Get Started

[Install Clifft](getting-started/installation.md){ .md-button .md-button--primary }
[Try the Playground](playground/){ .md-button }
