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

In its compiled execution modes, Clifft factors each trajectory into an offline
Clifford coordinate map, branch-dependent Pauli corrections represented by
affine Boolean signs, and a dense active state. Coordinate transformations and
symbolic dependencies are resolved ahead of time; each shot evaluates the
prepared signs and active-state actions without evolving an online tableau or
physical-qubit Pauli frame.

The main cost scales with $2^k$ rather than $2^n$, where $n$ is the total
number of qubits and $k$ is the active width. The corresponding active-state
dimension is $2^k$. Non-Clifford operations can increase $k$, while
measurements can reduce it. For near-Clifford protocols with frequent
measurements, such as magic-state preparation circuits, this can provide large
memory and runtime savings over standard dense state-vector simulation.

See [Acknowledgments and references](acknowledgments.md) for the research and
software that shaped Clifft.

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

## What's New in 0.14.0

Clifft 0.14.0 brings two major performance features—inspired by QuEra's
[Tsim](acknowledgments.md#tsim)—into the default compilation and sampling path:
[global sampling reconstruction](guide/compilation.md#global-reconstruction-for-sampling)
builds narrower plans for the outputs you request, and
[shared trajectory execution](guide/cpu-execution.md#shared-trajectory-execution)
reuses active-state evolution across shots that follow the same path while
pruning failed postselected shots early. The release also adds
[selective and bit-packed decoder outputs](guide/simulation.md#select-and-pack-outputs)
(`clifft.sample_detectors()`, `bit_packed=True`), default active-width
scheduling, and runtime AVX2/AVX-512 batch dispatch, delivering an
[11.1x median speedup on one core and 9.83x on eight cores over v0.13](guide/performance.md#changes-in-v014).

Read [Smaller Plans, Shared Trajectories](updates/smaller-plans-shared-trajectories.md)
for the release walkthrough and examples, and the
[release notes](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md)
for changed defaults and compatibility details. Earlier releases are covered
in the [development blog](updates/index.md).

[Full Changelog](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md){ .md-button }
