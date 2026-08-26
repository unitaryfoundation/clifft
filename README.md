<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unitaryfoundation/clifft/main/docs/assets/logos/clifft-logo-dark.png">
    <img src="https://raw.githubusercontent.com/unitaryfoundation/clifft/main/docs/assets/logos/clifft-logo-light.png" alt="Clifft" width="420">
  </picture>
</p>

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://unitaryfoundation.github.io/clifft/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.27058-b31b1b.svg)](https://arxiv.org/abs/2604.27058)
[![License](https://img.shields.io/github/license/unitaryfoundation/clifft.svg)](https://github.com/unitaryfoundation/clifft/blob/main/LICENSE)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=orange&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.foundation)


[![PyPI version](https://img.shields.io/pypi/v/clifft.svg?color=blue)](https://pypi.org/project/clifft/)
[![Downloads](https://static.pepy.tech/badge/clifft)](https://pepy.tech/project/clifft)
[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/unitaryfoundation/clifft/blob/main/CODE_OF_CONDUCT.md)


**Clifft** is a fast exact simulator for near-Clifford quantum circuits.

Built and maintained by the [Unitary Foundation](https://unitary.foundation).

Clifft accepts Stim-format circuits with non-Clifford extensions and a native
unitary OpenQASM 2 subset, then compiles them into symbolic-coordinate sampling
plans. It is designed for circuits whose dominant structure is Clifford, but
whose behavior depends on localized non-Clifford operations.

The dense active state has `2^k` amplitudes, where `k` is its active width. The
main simulation cost therefore scales with `2^k`, rather than directly with the
total number of physical qubits `n`. Non-Clifford operations can increase `k`,
while measurements can reduce it.

Clifft's original design established this factored active-state architecture,
described in the [Clifft paper](https://arxiv.org/abs/2604.27058).
[SymFT](https://arxiv.org/abs/2607.28600), by Wang Fang, Huazhe Lou, and Riling
Li, is the second-generation successor to
[SOFT](https://arxiv.org/abs/2512.23037). Its planner builds on SOFT's
generalized-stabilizer simulation and Clifft's dense active-state
representation. SymFT adds symbolic Clifford-Pauli-frame factorization,
adaptive stabilizer-coordinate planning, and direct multi-coordinate kernels.
Clifft's current sampler adopts these SymFT developments alongside
Clifft-specific compiler, continuation, and API machinery.
See the [theoretical overview](https://unitaryfoundation.github.io/clifft/stable/theory/overview/#method-provenance)
for the fuller lineage and implementation boundaries. The
[symbolic sampling update](https://unitaryfoundation.github.io/clifft/stable/updates/symbolic-sampling/)
explains the migration from the original SVM and reports matched release-target
benchmarks.

## Why Clifft?

- **Native circuit formats**: parse Stim-format circuits with noise, detectors,
  observables, and repeat blocks, plus unitary OpenQASM 2 input and non-Clifford
  extensions.
- **Exact near-Clifford simulation**: simulate localized non-Clifford effects
  without approximating the quantum state.
- **Optimizing compiler pipeline**: resolve Clifford coordinates and symbolic
  dependencies once, then sample many shots from a prepared plan.
- **Active-width scaling**: for low-magic circuits, runtime and memory scale
  with the localized active state rather than the full Hilbert space.

For QEC workflows, Clifft also supports detector-based post-selection, survivor
sampling, and stratified importance sampling for rare-event estimation.

## Installation

<!--pytest.mark.skip-->

```bash
pip install clifft
```

| Platform / CPU family | PyPI wheel |
|---|---|
| Linux `x86_64` with AVX2 | Supported |
| Linux `aarch64` | Supported |
| macOS `arm64` | Supported |
| Windows `amd64` | Supported |

All other platforms and CPU families should build from source. See the
[installation docs](https://unitaryfoundation.github.io/clifft/stable/getting-started/installation/#from-source).

## Quick Start

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")

result = clifft.sample(program, shots=1000, seed=42)
print(result.measurements[:5])
```

For more details and examples, check out the [documentation](https://unitaryfoundation.github.io/clifft) or take Clifft for a spin in the web-based [interactive playground](https://unitaryfoundation.github.io/clifft/playground/).

## Front-End Integrations

Clifft's native API accepts Stim-compatible circuit text. If your workflow
starts in another circuit framework, companion packages make the supported path
discoverable:

- **Qiskit**: [`clifft-qiskit`](https://github.com/unitaryfoundation/clifft-qiskit)
  provides a Qiskit `BackendV2` provider for running supported
  `QuantumCircuit` instances on Clifft.
- **Cirq**: [`clifft-cirq`](https://github.com/unitaryfoundation/clifft-cirq)
  converts parameter-resolved `cirq.Circuit` instances to Clifft text and
  provides a Cirq-style sampler backed by Clifft.

See the [front-end integrations guide](https://unitaryfoundation.github.io/clifft/stable/getting-started/integrations/)
for installation commands, minimal examples, and current limitations.

## Citation

If you use Clifft in your work, please cite the arXiv [preprint](https://arxiv.org/abs/2604.27058) below.
```
@misc{chase2026clifftfastexactsimulation,
      title={Clifft: Fast Exact Simulation of Near-Clifford Quantum Circuits},
      author={Bradley A. Chase and Farrokh Labib},
      year={2026},
      eprint={2604.27058},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.27058},
}
```

## Development

See the [building from source](https://unitaryfoundation.github.io/clifft/stable/development/building/) guide for build
instructions.

## AI Acknowledgement

We used generative AI tools during parts of the research, software-development,
and writing workflow for this project. These tools assisted with code generation
and review, implementation analysis, documentation editing, and checks of
selected derivations or arguments. All substantive design, validation, and
release decisions were made by the human contributors.

## Funding

This work was supported by the U.S. Department of Energy, Office of Science,
Office of Advanced Scientific Computing Research, Accelerated Research in
Quantum Computing under Award Number DE-SC0025336.

This material is also based upon work supported by the U.S. Department of
Energy, Office of Science, National Quantum Information Science Research
Centers, Quantum Science Center.

## License

Apache-2.0
