---
hide:
  - navigation
---

# Clifft

<p style="font-size: 1.2em;">
A fast exact state vector simulator for near-Clifford quantum circuits.
</p>

[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)

---

## What is Clifft?

Clifft compiles universal quantum circuits (Clifford + T and beyond) into a compact bytecode representation that is executed by a high-performance virtual machine. It is highly performant for near-Clifford circuits, which are mostly Clifford but have some non-Clifford gates. It achieves this performance by *factoring* the quantum state into components that are known ahead of time, versus those that vary shot-by-shot as a result of stochastic noise and measurement.

Clifft's performance scales mostly as $2^k$ vst $2^n$ for all $n$ qubits. The quantity $k$ is the active dimension or rank, and generally grows with non-Clifford gates and shrinks with measurements. For circuits where non-Clifford entanglement is bounded (e.g., magic state distillation), this yields exponential memory savings over standard state vector simulator.

## Quick Example

```python
import clifft

# Compile a circuit in Stim format extended with T gates
program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")

# Sample measurement outcomes
result = clifft.sample(program, shots=1000)
print(result.measurements[:5])  # First 5 shots
```

## Key Features

<div class="grid cards" markdown>

- :material-lightning-bolt: **Multi-Level Compilation**

    Clifford gates are absorbed at compile time. The VM only executes non-Clifford operations and measurements.

- :material-memory: **Factored State**

    Memory scales as $2^k$ where $k$ is the number of active (non-Clifford) qubits, not $2^n$ total qubits.

- :material-speedometer: **Bytecode VM**

    Cache-aligned 32-byte instructions. Single memory allocation. No dynamic resizing in the hot loop.

- :material-format-list-checks: **Stim Compatible**

    Parses Stim circuit format with support for noise channels, detectors, and repeat blocks.

</div>

## Get Started

[Install Clifft](getting-started/installation.md){ .md-button .md-button--primary }
[Try the Playground](playground/){ .md-button }
