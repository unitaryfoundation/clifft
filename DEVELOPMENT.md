# UCC Development Guide

This document covers how to build and develop UCC locally.

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

### Clone and Build

```bash
git clone https://github.com/unitaryfoundation/ucc-next.git
cd ucc-next

# Configure (Debug is the default)
cmake -B build

# Build
cmake --build build -j4

# Run tests
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
# C++ tests (via CTest)
ctest --test-dir build --output-on-failure

# Python tests (once bindings are implemented)
uv run pytest tests/python/ -v
```

## IDE Setup

CMake exports `compile_commands.json` to the build directory for IDE integration:

```bash
# For clangd / VS Code / CLion
ln -sf build/compile_commands.json .
```
