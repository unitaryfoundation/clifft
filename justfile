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
# Utility
# -------------------------

# aidigest creates a flat digest of all files in the repo, excluding build artifacts and other ignored files.
# This helps to share with Chat LLMs
aidigest:
rm -f ai.all.digest.txt
    rm ai.all.digest.txt
    uvx gitingest . -e build -e cmake -e uv.lock -e .git -e .venv -e __pycache__ -e .pytest_cache -e .jj -o ai.all.digest.txt

aidesigndigest:
    rm aidesign.digest.txt
    uvx gitingest . -i "design/*.md" -i "design/theory/*.tex" -o aidesign.digest.txt
