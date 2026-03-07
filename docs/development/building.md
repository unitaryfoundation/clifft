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

Builds the `ucc` Python package with C++ extensions:

```bash
git clone https://github.com/unitaryfoundation/ucc-next.git
cd ucc-next

uv venv
uv pip install -e .

# Verify
uv run python -c "import ucc; print(ucc.version())"

# Run tests
uv run pytest tests/python/ -v
```

The editable install (`-e .`) means you can re-run `uv pip install -e .` after modifying C++ code to rebuild.

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

## WebAssembly Build

For the browser-based [Compiler Explorer](../explorer.md):

```bash
# Requires Docker
just build-wasm
just test-wasm
```

Outputs `explorer/public/ucc_wasm.{js,wasm}`. See the [Explorer](../explorer.md) page.

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
