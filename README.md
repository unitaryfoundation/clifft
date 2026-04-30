<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unitaryfoundation/clifft/main/docs/assets/logos/clifft-logo-dark.png">
    <img src="https://raw.githubusercontent.com/unitaryfoundation/clifft/main/docs/assets/logos/clifft-logo-light.png" alt="Clifft" width="420">
  </picture>
</p>

[![PyPI version](https://img.shields.io/pypi/v/clifft.svg?color=blue)](https://pypi.org/project/clifft/)
[![Python versions](https://img.shields.io/pypi/pyversions/clifft.svg)](https://pypi.org/project/clifft/)
[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)
[![License](https://img.shields.io/github/license/unitaryfoundation/clifft.svg)](https://github.com/unitaryfoundation/clifft/blob/main/LICENSE)
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
[installation docs](https://unitaryfoundation.github.io/clifft/getting-started/installation/#from-source).

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

Clifft is designed for near-Clifford circuits where non-Clifford activity remains localized. In this regime, the dominant cost scales with the peak active dimension `k`, not directly with the total number of physical qubits.

<table>
  <thead>
    <tr>
      <th>Regime</th>
      <th>Representative benchmark</th>
      <th>What the results show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pure Clifford QEC</td>
      <td>
        Surface code d=7, r=7
        <a
          href="https://unitaryfoundation.github.io/clifft/playground/?url=https://raw.githubusercontent.com/unitaryfoundation/clifft-paper/main/qec_bench/circuits/surface_d7_r7.stim"
          target="_blank"
          rel="noopener noreferrer"
          title="Open this circuit in the Clifft playground"
          aria-label="Open Surface code d=7, r=7 in the Clifft playground">
          ▶↗
        </a>
      </td>
      <td>Stim remains the right tool; Clifft is roughly 10× slower while preserving the same sampling-oriented workflow.</td>
    </tr>
    <tr>
      <td>Low-magic FT circuits</td>
      <td>
        MSC d=3 cultivation
        <a
          href="https://unitaryfoundation.github.io/clifft/playground/?url=https://raw.githubusercontent.com/unitaryfoundation/clifft-paper/main/qec_bench/circuits/cultivation_d3.stim"
          target="_blank"
          rel="noopener noreferrer"
          title="Open this circuit in the Clifft playground"
          aria-label="Open MSC d=3 cultivation in the Clifft playground">
          ▶↗
        </a>
      </td>
      <td>Clifft reaches 10.4M shots/s, about 370× faster than <a href="https://github.com/QuEraComputing/tsim" target="_blank" rel="noopener noreferrer">Tsim</a> on this benchmark.</td>
    </tr>
    <tr>
      <td>Larger near-Clifford FT circuits</td>
      <td>
        MSC d=5 cultivation
        <a
          href="https://unitaryfoundation.github.io/clifft/playground/?url=https://raw.githubusercontent.com/unitaryfoundation/clifft-paper/main/qec_bench/circuits/cultivation_d5.stim"
          target="_blank"
          rel="noopener noreferrer"
          title="Open this circuit in the Clifft playground"
          aria-label="Open MSC d=5 cultivation in the Clifft playground">
          ▶↗
        </a>
      </td>
      <td>Clifft reaches ~135K shots/s on one CPU core, about 13× faster than <a href="https://github.com/haoliri0/SOFT" target="_blank" rel="noopener noreferrer">SOFT</a> at ~10.6K shots/s on one H800 GPU.</td>
    </tr>
    <tr>
      <td>Dense universal circuits</td>
      <td>Quantum Volume</td>
      <td>In the worst-case dense limit, Clifft remains neck-and-neck with simulators like <a href="https://github.com/Qiskit/qiskit-aer" target="_blank" rel="noopener noreferrer">qiskit-aer</a> and <a href="https://github.com/quantumlib/qsim" target="_blank" rel="noopener noreferrer">qsim</a>.</td>
    </tr>
  </tbody>
</table>

_Throughput numbers above were measured on cloud instances; the links to the in-browser WASM playground will report lower throughput._

For benchmark details, plots, hardware notes, and guidance on when Clifft is a good fit, see the [performance](https://unitaryfoundation.github.io/clifft/performance/) section of the documentation.

The full methodology and scientific results are described in the [Clifft paper](TODO_ARXIV_LINK).

## Citation

Citation information coming soon.

## Development

See the [building from source](https://unitaryfoundation.github.io/clifft/development/building/) guide for build
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
