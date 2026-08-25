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

Source installation builds Clifft's native core directly; Stim is not fetched
or linked into the Python package.

### Prerequisites (source build)

- **CMake** 3.20+
- **C++ compiler** with C++20 support (GCC 10+, Clang 12+, or Xcode CLT)
- **Python** 3.12+
- **uv** (recommended) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **OpenMP runtime** (optional) — detected automatically; Apple Clang users can
  install Homebrew `libomp` for intra-shot parallel sampling

OpenMP-enabled Clifft builds load an OpenMP runtime. Some scientific Python
packages bundle a different runtime, which can cause conflicts, especially on
macOS. Leaving Clifft at its default `threads=1` avoids calling its OpenMP
kernels. If another package has already used its own OpenMP runtime, do not then
request Clifft intra-shot workers in the same macOS process. Build with
`CLIFFT_OPENMP=OFF` or run the packages in separate processes when both need
threaded execution; starting Clifft before the other runtime can also work, but
process isolation is the robust choice.

On POSIX systems, create process workers before using threaded Clifft sampling,
or use the `spawn` or `forkserver` start method. Forking after a threaded sample
and then requesting intra-shot threads in the child can hang in some OpenMP
runtimes.

See [Building from Source](../development/building.md) for the full development setup.
