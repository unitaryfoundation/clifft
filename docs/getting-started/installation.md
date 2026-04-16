<!--pytest-codeblocks:skipfile-->

# Installation

## From PyPI

```bash
pip install clifft
```

!!! note "Python 3.12+ required"
    Clifft requires Python 3.12 or later.

## Platform Support

| Platform / CPU family | PyPI wheel |
|---|---|
| Linux `x86_64` with AVX2 | :white_check_mark: Supported |
| Linux `aarch64` | :white_check_mark: Supported |
| macOS `arm64` | :white_check_mark: Supported |
| Windows `amd64` | :white_check_mark: Supported |
| macOS `x86_64` | :x: Not supported |
| Other CPU families | :x: Not supported |

All other platforms and CPU families should build from source. See
[Building from Source](../development/building.md).

## From Source

For development, or if pre-built wheels are not available for your platform or CPU family:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

# Using uv (recommended)
uv venv
uv pip install -e .

# Verify
uv run python -c "import clifft; print(clifft.version())"
```

OpenMP is optional. Linux source builds usually find it automatically with GCC or Clang. On macOS with Apple clang, install Homebrew `libomp` before building if you want multi-core statevector execution:

```bash
brew install libomp
uv pip install -e .
```

If OpenMP is still not detected, pass the Homebrew prefix explicitly:

```bash
SKBUILD_CMAKE_ARGS="-DOpenMP_ROOT=$(brew --prefix libomp)" uv pip install -e .
```

!!! info "First build takes 10-15 minutes"
    The initial build compiles Stim (a dependency) from source, which has many files.
    Subsequent incremental builds are fast.

### Prerequisites (source build)

- **CMake** 3.20+
- **C++ compiler** with C++20 support (GCC 10+, Clang 12+, or Xcode CLT)
- **Python** 3.12+
- **uv** (recommended) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **macOS OpenMP runtime** (optional, enables multi-core statevector kernels) — `brew install libomp`

See [Building from Source](../development/building.md) for the full development setup.
