---
hide:
  - navigation
---

# Clifft

<p style="font-size: 1.2em;">
A fast exact simulator for near-Clifford quantum circuits.
</p>

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![PyPI version](https://img.shields.io/pypi/v/clifft.svg?color=blue)](https://pypi.org/project/clifft/)
[![Downloads](https://static.pepy.tech/badge/clifft)](https://pepy.tech/project/clifft)
[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=orange&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.foundation)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/unitaryfoundation/clifft/blob/main/CODE_OF_CONDUCT.md)

---

## What is Clifft?

Clifft is an exact simulator for quantum circuits whose dominant structure is Clifford, but whose behavior depends on localized non-Clifford operations. It accepts Stim-compatible circuits, extends them with non-Clifford gates, and compiles them into bytecode executed by a high-performance virtual machine.

Clifft works by factoring the quantum state into an offline Clifford frame, an online Pauli frame, and a dense active state vector. Clifford coordinate transformations are resolved ahead of time, while each shot performs only lightweight frame updates and localized state-vector evolution.

The main cost scales with $2^k$ rather than $2^n$, where $n$ is the total number of qubits and $k$ is the active dimension of the state vector. Non-Clifford operations can increase $k$, while measurements can reduce it. For near-Clifford protocols with frequent measurements, such as magic-state preparation circuits, this can provide large memory and runtime savings over standard dense state-vector simulation.

## Quick Example

Install via `pip install clifft`, then:

```python
import clifft

# Compile a Stim-format circuit extended with T gates.
program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")

# Sample measurement outcomes.
result = clifft.sample(program, shots=1000)
print(result.measurements[:5])  # First 5 shots.
```

## Key Features

<div class="grid cards" markdown>

- :material-format-list-checks: **Stim-Compatible Circuits**

    Parse standard Stim circuits, including noise channels, detectors, observables, and repeat blocks, with extensions for non-Clifford gates.

- :material-api: **Stim-Like Python API**

    Compile circuits once, then sample many shots through a familiar Python interface.

- :material-filter: **Built-In Filtering and Importance Sampling**

    Use detector-based early-exit filters and integrated importance sampling to focus computation on rare events.

- :material-tune: **Optimizing Compiler Pipeline**

    Multi-level optimization passes reduce active state-vector work before execution.

- :material-speedometer: **Near-Clifford Performance**

    For circuits with bounded active dimension, memory and runtime scale with the localized active state rather than the full qubit count.

</div>

## Get Started

[Install Clifft](getting-started/installation.md){ .md-button .md-button--primary }

[Try the Playground](playground/){ .md-button }
