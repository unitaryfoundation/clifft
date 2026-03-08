# justfile
set shell := ["bash", "-lc"]

# Default build dir for standalone CMake builds.
build_dir := "build"

# Coverage build dir (separate to avoid mixing instrumented/non-instrumented objects)
cov_build_dir := "build-coverage"

# Show recipes
default:
  @just --list

# -------------------------
# C++ standalone workflow
# -------------------------

configure *args="":
  cmake -B {{build_dir}} {{args}}

build *args="":
  cmake --build {{build_dir}} {{args}}

test *args="":
  ctest --test-dir {{build_dir}} --output-on-failure {{args}}

clean:
  rm -rf {{build_dir}}

# One-shot convenience
rebuild *args="":
  just clean
  just configure
  just build {{args}}

# -------------------------
# Python (scikit-build-core) workflow
# -------------------------

py-venv:
  uv venv

py-install:
  uv pip install -e .

py-test *args="":
  uv run pytest tests/python/ -v {{args}}

# Test that code examples in docs actually execute
py-doctest *args="":
  uv run pytest --codeblocks docs/ README.md -v {{args}}

# Convenience: ensure venv exists, then install, then run python tests
py:
  just py-venv
  just py-install
  just py-test

# -------------------------
# Code Coverage
# -------------------------

# Python coverage: run pytest with coverage and generate HTML report
py-cov:
  uv run pytest tests/python/ -v --cov=ucc --cov-report=term --cov-report=html:coverage/python
  @echo "\n📊 Python coverage report: coverage/python/index.html"

# C++ coverage: build with instrumentation, run tests, generate HTML report
cpp-cov:
  #!/usr/bin/env bash
  set -euo pipefail

  # Use nproc-1 cores (minimum 1) to leave headroom for coverage overhead
  CORES=$(( $(nproc) - 1 ))
  [[ $CORES -lt 1 ]] && CORES=1

  # Clean and build with coverage flags
  rm -rf {{cov_build_dir}}
  cmake -B {{cov_build_dir}} -DCMAKE_BUILD_TYPE=Debug -DUCC_COVERAGE=ON
  cmake --build {{cov_build_dir}} -j${CORES}

  # Run tests to generate .gcda files
  ctest --test-dir {{cov_build_dir}} --output-on-failure

  # Capture coverage data with lcov
  lcov --capture \
       --directory {{cov_build_dir}} \
       --output-file {{cov_build_dir}}/lcov.info \
       --ignore-errors mismatch \
       --rc branch_coverage=1

  # Remove external dependencies and test files from coverage
  lcov --remove {{cov_build_dir}}/lcov.info \
       '*/build-coverage/_deps/*' \
       '*/tests/*' \
       '/usr/*' \
       --output-file {{cov_build_dir}}/lcov.info \
       --ignore-errors unused \
       --rc branch_coverage=1

  # Generate HTML report
  genhtml {{cov_build_dir}}/lcov.info \
          --output-directory coverage/cpp \
          --rc branch_coverage=1

  echo ""
  echo "📊 C++ coverage report: coverage/cpp/index.html"

# Run both Python and C++ coverage
cov: py-cov cpp-cov
  @echo "\n✅ Coverage reports generated:"
  @echo "   Python: coverage/python/index.html"
  @echo "   C++:    coverage/cpp/index.html"

# -------------------------
# Lint / format
# -------------------------

lint:
  uv run pre-commit run --all-files

# -------------------------
# Profiling
# -------------------------

profile_build_dir := "build-profile"

# Build the SVM profiling harness (RelWithDebInfo for perf-friendly symbols)
profile-build:
  cmake -B {{profile_build_dir}} -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUCC_BUILD_PROFILER=ON
  cmake --build {{profile_build_dir}} --target profile_svm -j$(nproc)

# Run the profiler with default settings (50-qubit Clifford, 100k shots)
profile *args="":
  {{profile_build_dir}}/profile_svm {{args}}

# -------------------------
# Benchmarking
# -------------------------

# Run performance benchmarks comparing UCC vs Stim
bench *args="":
  uv run pytest tools/bench/ --benchmark-sort=name --benchmark-columns=Mean,StdDev,Ops {{args}}

# -------------------------
# WebAssembly (Explorer)
# -------------------------

# Build the Wasm module using the emscripten/emsdk Docker image
build-wasm:
  #!/usr/bin/env bash
  set -euo pipefail
  echo "Building UCC Wasm module..."
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "{{justfile_directory()}}:/src" \
    -w /src \
    emscripten/emsdk:3.1.74 \
    bash -c '\
      emcmake cmake -B build-wasm -S src/wasm \
        -DCMAKE_BUILD_TYPE=Release \
        -DFETCHCONTENT_QUIET=ON && \
      cmake --build build-wasm -j$(nproc)'
  mkdir -p explorer/public
  cp build-wasm/ucc_wasm.js explorer/public/ucc_wasm.js
  cp build-wasm/ucc_wasm.wasm explorer/public/ucc_wasm.wasm
  echo "Output: explorer/public/ucc_wasm.{js,wasm}"

# Test the Wasm module with a quick Node.js smoke test
test-wasm:
  #!/usr/bin/env bash
  set -euo pipefail
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "{{justfile_directory()}}:/src" \
    -w /src \
    emscripten/emsdk:3.1.74 \
    node --experimental-vm-modules src/wasm/test_wasm.mjs

# -------------------------
# Documentation
# -------------------------

# Serve docs locally with live reload
docs-serve:
  uv run mkdocs serve -a 0.0.0.0:8000

# Build docs to site/
docs-build:
  uv run mkdocs build --strict

# -------------------------
# Explorer
# -------------------------

# Install explorer dependencies
explorer-install:
  cd explorer && npm ci

# Start the explorer dev server
explorer-dev:
  cd explorer && npx vite --port 8000 --host

# Build the explorer for production
explorer-build:
  cd explorer && npm run build

# -------------------------
# Utility
# -------------------------

# aidigest creates a flat digest of all files in the repo, excluding build artifacts and other ignored files.
# This helps to share with Chat LLMs
aidigest:
    rm -f ai.all.digest.txt
    uvx gitingest . -e "*.pdf" -e explorer -i "design/*.md" -i design/theory.tex -i src -i tests -i README.md -i pyproject.toml -o ai.all.digest.txt

aidesigndigest:
    rm -f aidesign.digest.txt
    uvx gitingest . -i "design/*.md" -o aidesign.digest.txt
