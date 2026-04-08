# Clifft Development Guide

This document covers how to build and develop Clifft locally.

## Platform Support

| Platform | Status |
|----------|--------|
| Linux (x86_64) | ✅ Supported |
| macOS (Intel & Apple Silicon) | ✅ Supported |
| Windows | ❌ Not yet supported |

## Prerequisites

- **CMake** 3.20 or later
- **C++ compiler** with C++20 support:
  - Linux: GCC 10+ or Clang 12+
  - macOS: Xcode Command Line Tools (Clang 14+) or `brew install llvm`
- **Python** 3.12+ (for Python bindings and dev tools)
- **uv** (Python package manager) — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Building

Clifft can be built in two ways:
1. **As a Python package** (recommended for most users)
2. **As a standalone C++ library** (for C++ development/testing)

### Python Package Build (Recommended)

This builds the `clifft` Python package with C++ extensions:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

# Create virtual environment and install in editable mode
uv venv
uv pip install -e .

# Verify installation
uv run python -c "import clifft; print(clifft.version())"

# Run Python tests
uv run pytest tests/python/ -v
```

The editable install (`-e .`) rebuilds the C++ extension automatically when you run `uv pip install -e .` again after modifying C++ code.

### Standalone C++ Build

For pure C++ development without Python:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

# Configure (Debug is the default)
cmake -B build

# Build
cmake --build build -j4

# Run C++ tests
ctest --test-dir build --output-on-failure
```

### Build Types

```bash
# Debug (default) - includes debug symbols, no optimization
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release - optimized, no debug symbols
cmake -B build -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo - optimized with debug symbols
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Custom Qubit Width

By default, Clifft supports up to 64 qubits. This limit is a compile-time constant that controls the width of inline Pauli bitmasks (`BitMask<N>`) used in the HIR and VM Pauli frame. The VM `Instruction` struct stays 32 bytes regardless — only compile-time structures (HIR, Pauli frame) grow.

To simulate larger circuits (e.g., distance-7 surface codes require 118 qubits, magic state cultivation needs ~480), rebuild with a higher limit. The value must be a multiple of 64.

At 64 qubits, each `BitMask` is a single `uint64_t` (8 bytes). At 512 qubits, it grows to 8 words (64 bytes). The compiler auto-vectorizes these operations (AVX2/AVX-512 on x86, NEON on ARM), so the overhead is modest for compilation, but the VM hot loop is unaffected since instructions use `uint16_t` axis indices.

#### Python package

The qubit width is configured in `pyproject.toml`. Edit the `CLIFFT_MAX_QUBITS` value under `[tool.scikit-build.cmake.define]`:

```toml
[tool.scikit-build.cmake.define]
CLIFFT_MAX_QUBITS = "512"
```

Then rebuild:

```bash
uv pip install -e .
```

All standard `uv` commands (`uv run`, `uv sync`, etc.) respect this setting automatically. Verify with:

```bash
uv run python -c "import clifft; print(clifft.max_sim_qubits())"
```

#### Standalone C++ build

For pure C++ development, pass the flag directly to CMake:

```bash
cmake -B build -DCLIFFT_MAX_QUBITS=512
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or with `just`:

```bash
just max_qubits=512 configure build test
```

The `max_qubits` variable in the justfile only affects standalone C++ builds. Python builds always read from `pyproject.toml`.

### First Build

The first build takes 10-15 minutes because Stim (a dependency) has many source files. Subsequent incremental builds are fast.

If you encounter memory pressure, reduce parallelism:

```bash
cmake --build build -j1
```

### Expected Warnings

During configuration, you may see warnings like:

```
WARNING: Skipped stim_test target. `GTest` not found.
WARNING: Skipped stim_python_bindings target. `pybind11` not found.
```

These are harmless — they come from Stim's CMakeLists.txt checking for optional dependencies that we don't need.

## Development Workflow

### Code Quality

We use pre-commit hooks to enforce code quality:

```bash
# Install Python dev dependencies
uv sync

# Install pre-commit hooks (runs automatically on git commit)
uv run pre-commit install

# Run all checks manually
uv run pre-commit run --all-files
```

### Running Tests

```bash
# C++ tests (via CTest, requires standalone build)
ctest --test-dir build --output-on-failure

# Python tests (requires Python package build)
uv run pytest tests/python/ -v
```

## IDE Setup

CMake exports `compile_commands.json` to the build directory for IDE integration:

```bash
# For clangd / VS Code / CLion
ln -sf build/compile_commands.json .
```

## Code Coverage

Clifft supports code coverage for both Python and C++ code.

### Prerequisites

For C++ coverage, you need `lcov` and `genhtml`:

```bash
# Ubuntu/Debian
sudo apt install lcov

# macOS
brew install lcov
```

### Running Coverage

Using `just` (recommended):

```bash
# Python coverage only
just py-cov

# C++ coverage only
just cpp-cov

# Both Python and C++ coverage
just cov
```

### Manual Coverage Commands

**Python coverage:**

```bash
uv run pytest tests/python/ -v --cov=clifft --cov-report=term --cov-report=html:coverage/python
```

**C++ coverage:**

```bash
# Build with coverage instrumentation
cmake -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DCLIFFT_COVERAGE=ON
cmake --build build-coverage -j

# Run tests (generates .gcda files)
ctest --test-dir build-coverage --output-on-failure

# Capture and filter coverage data
lcov --capture --directory build-coverage --output-file build-coverage/lcov.info
lcov --remove build-coverage/lcov.info '*/build-coverage/_deps/*' '*/tests/*' '/usr/*' \
     --output-file build-coverage/lcov.info

# Generate HTML report
genhtml build-coverage/lcov.info --output-directory coverage/cpp
```

### Coverage Reports

After running coverage, HTML reports are available at:

- **Python:** `coverage/python/index.html`
- **C++:** `coverage/cpp/index.html`

Open these files in a browser to view detailed line-by-line coverage.

### Notes

- C++ coverage uses a separate build directory (`build-coverage/`) to avoid mixing instrumented and non-instrumented object files.
- Python coverage only measures Python code in `src/python/clifft/`. The C++ extension (`_clifft_core`) is not measured by Python coverage tools—use C++ coverage for that.
- Coverage artifacts are gitignored.


## WebAssembly Build (Compiler Explorer)

Clifft can be compiled to WebAssembly for the browser-based Compiler Explorer. This uses [Emscripten](https://emscripten.org/) via Docker, so no native Emscripten install is needed.

### Prerequisites

- **Docker** must be installed and running. The build uses the `emscripten/emsdk:3.1.74` image.
  - Install Docker: https://docs.docker.com/engine/install/
  - The image is pulled automatically on first build (~1.5 GB download).

### Building

```bash
# Build the Wasm module (outputs to explorer/public/)
just build-wasm

# Run the Node.js smoke tests
just test-wasm
```

The build produces two files:
- `explorer/public/clifft_wasm.js` (~92 KB) - JavaScript loader/glue
- `explorer/public/clifft_wasm.wasm` (~460 KB) - WebAssembly binary

### Manual Build (without `just`)

```bash
# Build
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd):/src" -w /src \
  emscripten/emsdk:3.1.74 \
  bash -c 'emcmake cmake -B build-wasm -S src/wasm \
    -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_QUIET=ON && \
    cmake --build build-wasm -j$(nproc)'

# Copy outputs
mkdir -p explorer/public
cp build-wasm/clifft_wasm.{js,wasm} explorer/public/

# Test
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd):/src" -w /src \
  emscripten/emsdk:3.1.74 \
  node --experimental-vm-modules src/wasm/test_wasm.mjs
```

### Troubleshooting

- **"docker: command not found"** - Install Docker Desktop or Docker Engine for your platform.
- **Permission errors on output files** - The build uses `-u $(id -u):$(id -g)` to match your host user. If files are owned by root, delete `build-wasm/` and rebuild.
- **First build is slow (~5 min)** - Emscripten fetches and compiles Stim + dependencies from source. Subsequent rebuilds are incremental.

# `just` Shortcuts (Optional)

For convenience, this repository includes a `justfile` that wraps common development commands.

## Installing `just`

```bash
# macOS
brew install just

# Ubuntu/Debian
sudo apt install just

# Or via cargo
cargo install just

# Or prebuilt binaries
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

## Usage

Run `just --list` to see all available recipes:

```bash
just --list
```

Recipes evolve over time, so always check `--list` for the current options. Common patterns:

- `just py` — full Python workflow (venv + install + test)
- `just build` — build C++ targets
- `just test` — run C++ tests
- `just lint` — run pre-commit checks

`just` is optional — all underlying CMake and `uv` commands documented above remain the source of truth.
