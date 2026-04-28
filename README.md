<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logos/clifft-logo-dark.png">
    <img src="docs/assets/logos/clifft-logo-light.png" alt="Clifft" width="420">
  </picture>
</p>

[![PyPI version](https://img.shields.io/pypi/v/clifft.svg?color=blue)](https://pypi.org/project/clifft/)
[![Python versions](https://img.shields.io/pypi/pyversions/clifft.svg)](https://pypi.org/project/clifft/)
[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)
[![License](https://img.shields.io/github/license/unitaryfoundation/clifft.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://unitaryfoundation.github.io/clifft/)
[![Discord](https://img.shields.io/badge/Discord-Unitary%20Foundation-orange.svg)](http://discord.unitary.foundation)

**Clifft** is a fast exact simulator for near-Clifford quantum circuits.

Built and maintained by the [Unitary Foundation](https://unitary.foundation).

Clifft accepts Stim-format circuits, extends them with non-Clifford gates, and
compiles them into bytecode executed by a high-performance Schrödinger Virtual
Machine. It is designed for circuits whose dominant structure is Clifford, but
whose behavior depends on localized non-Clifford operations.

The main simulation cost scales with the active dimension `k` of the dense state
vector, rather than directly with the total number of physical qubits `n`.
Non-Clifford operations can increase `k`, while measurements can reduce it.

## Why Clifft?

- **Stim-compatible format and API**: parse Stim-format circuits with noise,
  detectors, observables, and repeat blocks, plus non-Clifford extensions.
- **Exact near-Clifford simulation**: simulate localized non-Clifford effects
  without approximating the quantum state.
- **Optimizing compiler pipeline**: compile once, then sample many shots with
  HIR and bytecode optimization passes.
- **Active-dimension scaling**: for low-magic circuits, runtime and memory scale
  with the localized active state rather than the full Hilbert space.

For QEC workflows, Clifft also supports detector-based post-selection, survivor
sampling, and stratified importance sampling for rare-event estimation.

## Installation

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
[installation docs](docs/getting-started/installation.md#from-source).

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

## Performance

Clifft is designed for near-Clifford circuits where non-Clifford activity
remains localized. However, it is also quite performant as a standard dense statevector simulator, even outside this regime. Full application benchmarks
and citation information will be linked here when the paper is public.

## Citation

Citation information coming soon.

## Development

See the [building from source](docs/development/building.md) guide for build
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

## License

Apache-2.0
