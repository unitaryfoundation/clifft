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

Clifft is an exact simulator for quantum circuits whose dominant structure is Clifford, but whose behavior depends on localized non-Clifford operations. It accepts Stim-compatible circuits, extends them with non-Clifford gates, and compiles them into a high-performance symbolic-coordinate sampling plan.

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

- **Stim-Compatible Format and API**

    Parse Stim-format circuits, including noise channels, detectors, observables, and repeat blocks, with extensions for non-Clifford gates. Compile once, then sample many shots through a familiar Python interface.

- **Exact Near-Clifford Simulation**

    Simulate circuits with localized non-Clifford operations exactly, without approximating the quantum state.

- **Optimizing Compiler Pipeline**

    Multi-level optimization passes reduce active state-vector work before execution.

- **Multiple Circuit Inputs**

    Use native Stim-compatible text, unitary OpenQASM 2, or supported circuits
    from Qiskit and Cirq companion packages.

- **Active-Width Scaling**

    For circuits with bounded active width, memory and runtime scale with the localized active state rather than the full qubit count.

- **Leakage and Loss Trajectories (Experimental)**

    Sample five-level leakage and loss models with state-dependent transitions,
    measurement classification, and back-action on the computational state.
    See the [Leakage and Loss guide](guide/leakage-and-loss.md).

</div>

For QEC workflows, Clifft also supports detector-based post-selection, survivor sampling, and stratified importance sampling for rare-event estimation.

## Get Started

[Install Clifft](getting-started/installation.md){ .md-button .md-button--primary }

[Quick Start](getting-started/quickstart.md){ .md-button .md-button--primary }

[Choose a Workflow](getting-started/choosing-a-workflow.md){ .md-button }

[Bring Your Circuit](getting-started/integrations.md){ .md-button }

[Try the Playground]({{ playground_url }}){ .md-button }

## What's New in 0.9.0

Clifft 0.9.0 adds parallel sampling for ordinary,
post-selected, forced-fault, and noncomputational workloads. Pass a total
worker budget with `threads`, and fixed-plan sampling automatically chooses
between running shots concurrently and using OpenMP within a wide shot.
Advanced callers can select an explicit hybrid layout. Fixed seeds produce the
same results across worker layouts, although seeded rows differ from v0.8
because each shot now has its own random stream.

The release also defines `get_statevector()` up to global phase, absorbs
Clifford-valued rotations earlier during compilation, vectorizes additional
active-measurement kernels, and fixes complex-interference cases in
`basis_probabilities()`.

Read [CPU Execution and Tuning](guide/cpu-execution.md) for the
threading model, memory tradeoffs, and expert controls.

## What's New in 0.8.0

Clifft 0.8.0 replaces the original localized-Pauli SVM with a
symbolic-coordinate compiler and sampler. The main `compile()` and sampling
workflows remain, while the public VM bytecode and backend-selection APIs have
been removed. Compiled programs now expose `peak_active_width` and an
`inspect()` view of the sampling plan.

Read [Symbolic Sampling in Clifft](updates/symbolic-sampling.md) for the design
motivation, API migration notes, matched performance results, and deferred
follow-up work.

[Full Changelog](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md){ .md-button }
