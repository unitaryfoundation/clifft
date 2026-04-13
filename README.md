# Clifft

[![CI](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/clifft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/unitaryfoundation/clifft/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/clifft)

A multi-level compiler and Schrodinger Virtual Machine (SVM) for quantum circuits.

Clifft supports Clifford + T gates and beyond, with a focus on high-performance simulation.

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
| macOS `x86_64` | Not supported |
| Other CPU families | Not supported |

All other platforms and CPU families should build from source. See
[installation docs](docs/getting-started/installation.md#from-source).

For macOS source builds, install Homebrew `libomp` first if you want
OpenMP-enabled multi-core statevector execution:

<!--pytest.mark.skip-->
```bash
brew install libomp
```

Linux source builds typically find OpenMP automatically with GCC or Clang.

## Quick Start

```python
import clifft

print(clifft.version())
```

## Development

See the [building from source](docs/development/building.md) guide for build instructions.

## Acknowledgements

Clifft was developed with the assistance of AI tools, primarily
[Claude](https://claude.ai) by Anthropic. All AI-generated code has been
reviewed by human contributors. See our
[contributing guide](docs/development/contributing.md) for our AI contribution
policy.

## License

Apache-2.0
