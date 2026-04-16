<!--pytest-codeblocks:skipfile-->

# Building from Source

## Prerequisites

- **CMake** 3.20+
- **C++ compiler** with C++20 support:
    - Linux: GCC 10+ or Clang 12+
    - macOS: Xcode Command Line Tools (Clang 14+) or `brew install llvm`
- **Python** 3.12+
- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **OpenMP runtime on macOS** (optional, enables multi-core statevector kernels)
    - `brew install libomp`

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

OpenMP support is optional. Linux source builds usually pick it up automatically with GCC or Clang. On macOS with Apple clang, install Homebrew `libomp` before building if you want multi-core statevector execution:

```bash
brew install libomp
uv pip install -e .
```

If OpenMP is not detected automatically, point scikit-build-core at the Homebrew prefix:

```bash
SKBUILD_CMAKE_ARGS="-DOpenMP_ROOT=$(brew --prefix libomp)" uv pip install -e .
```

### Platform and CPU support

| Platform / CPU family | PyPI wheel | Source build | Notes |
|---|---|---|---|
| Linux `x86_64` with AVX2/BMI2/FMA | Supported | Supported | Wheel uses an `x86-64-v3` baseline and can dispatch to the AVX-512 SVM path on capable CPUs. |
| Linux `x86_64` without AVX2 | Not supported | Supported | Use `pip install --no-binary clifft clifft` or build from a checkout. |
| Linux `aarch64` | Supported | Supported | Wheels use a portable ARM baseline; local optimized builds default to native CPU tuning. |
| macOS `arm64` | Supported | Supported | Wheels use a portable Apple Silicon baseline; local optimized builds are supported. |
| Windows `amd64` | Supported | Supported | Wheels use the base SVM path on MSVC; Linux x86 wheels expose the hand-tuned AVX2/AVX-512 paths. |
| macOS `x86_64` | Not supported | Supported | Build from source. |
| Other CPU families | Not supported | Best effort | No wheels are published. |

### CPU baseline policy

- Published wheels use explicit portable baselines chosen in CI.
- Local Python source builds and standalone C++ Release builds default to `CLIFFT_CPU_BASELINE=native`.
- Supported values are `native`, `generic`, and `x86-64-v3`.
- `x86-64-v3` is intended for Linux `x86_64` wheels and requires AVX2, BMI2, and FMA.

Override the default when needed:

```bash
CLIFFT_CPU_BASELINE=generic uv pip install -e .
```

## Standalone C++ Build

For pure C++ development without Python:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

OpenMP is optional for standalone C++ builds too. Linux toolchains usually find it automatically. On macOS with Apple clang, install Homebrew `libomp` first:

```bash
brew install libomp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If `find_package(OpenMP)` still does not resolve, set `OpenMP_ROOT` explicitly:

```bash
cmake -B build -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build -j
```

### Build Types

| Type | Flag | Use Case |
|------|------|----------|
| Debug | `-DCMAKE_BUILD_TYPE=Debug` | Development (default) |
| Release | `-DCMAKE_BUILD_TYPE=Release` | Benchmarking |
| RelWithDebInfo | `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | Profiling |

For optimized source builds, `Release` and `RelWithDebInfo` default to native CPU tuning on the build machine. On x86 GNU/Clang builds, that keeps the AVX2 and AVX-512 SVM specializations available for runtime dispatch.

!!! info "First build takes 10-15 minutes"
    Stim (a dependency) has many source files. Subsequent builds are incremental.
    If you hit memory pressure, reduce parallelism: `cmake --build build -j1`

## CLIFFT_MAX_QUBITS

Clifft's Pauli frame uses a compile-time-sized bitmask. The `CLIFFT_MAX_QUBITS` setting controls the maximum number of qubits the simulator can handle. It must be a multiple of 64.

| Setting | Max Qubits | Use Case |
|---------|-----------|----------|
| 64 | 64 | Best small-circuit and QV performance |
| 128 (default) | 128 | Distance-7 surface codes, 85-qubit distillation |
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

For the browser-based [Playground](../playground/):

```bash
# Requires Docker
just build-wasm
just test-wasm
```

Outputs `playground/public/clifft_wasm.{js,wasm}`. See the [Playground](../playground/) page.

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

## Dependency Groups

Development dependencies are managed via [PEP 735 dependency groups](https://peps.python.org/pep-0735/) in `pyproject.toml`. The `uv` package manager uses these automatically.

- **`dev`** (default) — Installed by `uv sync`. Includes everything needed for building, testing, and linting.
- **`docs`** — Installed with `uv sync --group docs`. Includes the MkDocs toolchain for building the documentation site.

The only runtime dependency is `numpy`. All other packages are development-only.
