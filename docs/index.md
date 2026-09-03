---
hide:
  - navigation
title: Clifft
---

# Clifft { .visually-hidden }

<p align="center" markdown>
  ![Clifft](assets/logos/clifft-logo-light.png#only-light){ width=420 }
  ![Clifft](assets/logos/clifft-logo-dark.png#only-dark){ width=420 }
</p>

<p style="font-size: 1.2em; text-align: center;">
A fast exact simulator for near-Clifford quantum circuits.
</p>

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![PyPI version](https://img.shields.io/pypi/v/clifft.svg?color=blue)](https://pypi.org/project/clifft/)
[![Downloads](https://static.pepy.tech/badge/clifft)](https://pepy.tech/project/clifft)
[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)
[![arXiv](https://img.shields.io/badge/arXiv-2604.27058-b31b1b.svg)](https://arxiv.org/abs/2604.27058)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=orange&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.foundation)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/unitaryfoundation/clifft/blob/main/CODE_OF_CONDUCT.md)

---

## What is Clifft?

Clifft is an exact simulator for quantum circuits whose dominant structure is Clifford, but whose behavior depends on localized non-Clifford operations. It accepts Stim circuits, extends them with non-Clifford gates, and compiles them into a high-performance symbolic-coordinate sampling plan.

Clifft factors each trajectory into an offline Clifford coordinate map,
branch-dependent Pauli corrections represented by affine Boolean signs, and a
dense active state. Coordinate transformations and symbolic dependencies are
resolved ahead of time; each shot evaluates the prepared signs and active-state
actions without evolving an online tableau or physical-qubit Pauli frame.

The main cost scales with $2^k$ rather than $2^n$, where $n$ is the total
number of qubits and $k$ is the active width. The corresponding active-state
dimension is $2^k$. Non-Clifford operations can increase $k$, while
measurements can reduce it. For near-Clifford protocols with frequent
measurements, such as magic-state preparation circuits, this can provide large
memory and runtime savings over standard dense state-vector simulation.

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

- **Stim Circuits with Non-Clifford Extensions**

    Existing Stim circuits compile directly. Add Clifft's non-Clifford gates
    when needed, then compile once and sample many shots through a familiar
    Python interface. OpenQASM 2, Qiskit, and Cirq inputs are also available.

- **Exact Near-Clifford Simulation**

    Simulate circuits with localized non-Clifford operations exactly, without approximating the quantum state.

- **Leakage and Loss**

    Model state-dependent leakage and loss, including measurement
    classification and back-action on the computational state. See the
    [Leakage and Loss guide](guide/leakage-and-loss.md).

- **Active-Width Scaling**

    For circuits with bounded active width, memory and runtime scale with the localized active state rather than the full qubit count.

</div>

For QEC workflows, Clifft also supports detector-based post-selection, survivor sampling, and stratified importance sampling for rare-event estimation.

## Get Started

[Quick Start](getting-started/quickstart.md){ .md-button .md-button--primary }
[Try the Playground]({{ playground_url }}){ .md-button }

## What's New in 0.10.0

Clifft 0.10.0 adds automatic packed batch sampling for eligible
low-active-width CPU workloads and Apple Silicon NEON kernels for active-state
operations. Advanced callers can use `batch_size` to tune the packed-lane
capacity, while the default cost-aware policy balances throughput and memory.

In calibrated single-core benchmarks, v0.10 is faster than v0.9 on all eight
measured workloads, with a 3.23x median improvement. It also leads calibrated
SymFT on all eight, from 1.05x to 87.7x. See [Performance](guide/performance.md)
for the figures, absolute throughput, dense Quantum Volume results, and
measurement details.

The release also accepts supported unitary OpenQASM 2 circuits without Qiskit
and moves production builds onto Clifft's native Clifford implementation. Stim
remains an independent test oracle.

Read [Packed Sampling in Clifft](updates/packed-sampling.md) for the design,
automatic policy, and v0.9 comparison. See
[Circuit Inputs](guide/circuit-inputs.md) for OpenQASM, Qiskit, and Cirq options,
or [CPU Execution and Tuning](guide/cpu-execution.md) for detailed controls.

### Earlier development updates

Read [Parallel Sampling in Clifft](updates/parallel-sampling.md) for the v0.9.0
threading work and [Symbolic Sampling in Clifft](updates/symbolic-sampling.md)
for the v0.8.0 compiler and sampler redesign.

[Full Changelog](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md){ .md-button }
