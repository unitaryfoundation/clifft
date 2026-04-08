<!--pytest-codeblocks:skipfile-->

# Building from Source

## Prerequisites

- **CMake** 3.20+
- **C++ compiler** with C++20 support:
    - Linux: GCC 10+ or Clang 12+
    - macOS: Xcode Command Line Tools (Clang 14+) or `brew install llvm`
- **Python** 3.12+
- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Python Package Build (Recommended)

Builds the `clifft` Python package with C++ extensions:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

uv venv
uv pip install -e .

# Verify
uv run python -c "import clifft; print(clifft.version())"

# Run tests
uv run pytest tests/python/ -v
```

The editable install (`-e .`) means you can re-run `uv pip install -e .` after modifying C++ code to rebuild.

Note that this builds with the maximum number of qubits as set by `CLIFFT_MAX_QUBITS` in the `pyproject.toml`. If you modify, you will need to rebuild. See [CLIFFT_MAX_QUBITS](#clifft_max_qubits) below.

## Standalone C++ Build

For pure C++ development without Python:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Build Types

| Type | Flag | Use Case |
|------|------|----------|
| Debug | `-DCMAKE_BUILD_TYPE=Debug` | Development (default) |
| Release | `-DCMAKE_BUILD_TYPE=Release` | Benchmarking |
| RelWithDebInfo | `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | Profiling |

!!! info "First build takes 10-15 minutes"
    Stim (a dependency) has many source files. Subsequent builds are incremental.
    If you hit memory pressure, reduce parallelism: `cmake --build build -j1`

## CLIFFT_MAX_QUBITS

Clifft's Pauli frame uses a compile-time-sized bitmask. The `CLIFFT_MAX_QUBITS` setting controls the maximum number of qubits the simulator can handle. It must be a multiple of 64.

| Setting | Max Qubits | Use Case |
|---------|-----------|----------|
| 64 (default) | 64 | Development, small circuits |
| 128 | 128 | Distance-7 surface codes |
| 256 | 256 | Large QEC experiments |
| 512 | 512 | Production-scale circuits |

**Python build:** Edit `CLIFFT_MAX_QUBITS` in `pyproject.toml` under `[tool.scikit-build.cmake.define]`, then rebuild:

```bash
uv pip install -e .
```

**C++ build:** Pass the value as a CMake variable:

```bash
cmake -B build -DCLIFFT_MAX_QUBITS=128
cmake --build build -j
```

Or set the environment variable (checked at configure time):

```bash
export CLIFFT_MAX_QUBITS=128
cmake -B build
```

!!! warning "Rebuild required"
    Changing `CLIFFT_MAX_QUBITS` requires a full rebuild. The value is baked into
    struct layouts at compile time.

## WebAssembly Build

For the browser-based [Compiler Explorer](../explorer.md):

```bash
# Requires Docker
just build-wasm
just test-wasm
```

Outputs `explorer/public/clifft_wasm.{js,wasm}`. See the [Explorer](../explorer.md) page.

## IDE Setup

CMake exports `compile_commands.json` for IDE integration:

```bash
ln -sf build/compile_commands.json .
```

This enables clangd, VS Code C++ extension, and CLion to provide accurate diagnostics.

## `just` Shortcuts

The repository includes a `justfile` for common tasks:

```bash
just --list    # Show all recipes
just py        # Full Python workflow (venv + install + test)
just build     # Build C++ targets
just test      # Run C++ tests
just lint      # Run pre-commit checks
```

`just` is optional — all underlying commands are documented above.
