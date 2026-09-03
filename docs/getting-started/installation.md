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
| Linux `x86_64` with x86-64-v2 support | :white_check_mark: Supported |
| Linux `aarch64` | :white_check_mark: Supported |
| macOS `arm64` | :white_check_mark: Supported |
| Windows `amd64` | :white_check_mark: Supported |

All other platforms and CPU families should build from source. See
[Building from Source](../development/building.md).

Published wheels are CPU-only. The AMD HIP and NVIDIA CUDA backends remain
explicit, source-build-only experiments; see
[HIP Backend](../development/hip-backend.md) and
[CUDA Backend](../development/cuda-backend.md).

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

### Prerequisites (source build)

- **CMake** 3.20+
- **C++ compiler** with C++20 support (GCC 10+, Clang 12+, or Xcode CLT)
- **Python** 3.12+
- **uv** (recommended) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **OpenMP runtime** (optional) — detected automatically; Apple Clang users can
  install Homebrew `libomp` for intra-shot parallel sampling

See [CPU Execution and Tuning](../guide/cpu-execution.md) for automatic
scheduling, explicit layouts, batching, memory tradeoffs, OpenMP runtime
compatibility, and process-start guidance.

See [Building from Source](../development/building.md) for the full development setup.

Next, follow the [Quick Start](quickstart.md) or choose a path from
[Circuit Inputs](../guide/circuit-inputs.md).
