<!--pytest-codeblocks:skipfile-->

# Installation

## From PyPI

```bash
pip install ucc
```

!!! note "Python 3.12+ required"
    UCC requires Python 3.12 or later.

## Platform Support

| Platform | Status |
|----------|--------|
| Linux (x86_64) | :white_check_mark: Supported |
| macOS (Intel & Apple Silicon) | :white_check_mark: Supported |
| Windows | :x: Not yet supported |

## From Source

For development or if pre-built wheels aren't available for your platform:

```bash
git clone https://github.com/unitaryfoundation/ucc-next.git
cd ucc-next

# Using uv (recommended)
uv venv
uv pip install -e .

# Verify
uv run python -c "import ucc; print(ucc.version())"
```

!!! info "First build takes 10-15 minutes"
    The initial build compiles Stim (a dependency) from source, which has many files.
    Subsequent incremental builds are fast.

### Prerequisites (source build)

- **CMake** 3.20+
- **C++ compiler** with C++20 support (GCC 10+, Clang 12+, or Xcode CLT)
- **Python** 3.12+
- **uv** (recommended) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

See [Building from Source](../development/building.md) for the full development setup.
