# Clifft AI Agent Instructions

Instructions for AI coding assistants working on this repository.

## Project Overview

Clifft is a multi-level Ahead-Of-Time (AOT) compiler and Schrodinger Virtual
Machine (SVM) for quantum circuits. It supports Clifford + T gates and beyond,
with a focus on high-performance simulation.

## Source of Truth

Use the code under `src/clifft/`, the tests under `tests/`, and the
documentation under `docs/`, `README.md`, and `DEVELOPMENT.md` as the source
of truth. If something appears architecturally ambiguous, inspect the current
implementation and tests first, then ask the human before making a speculative
change.

## Architecture Change Protocol

Do not rewrite architecture-facing docs to justify a code change. If you
discover a technical contradiction or a scenario where the current architecture
cannot support a proposed change:

1. **Stop.** Do not silently implement a workaround.
2. Explain the discrepancy and propose a fix.
3. Wait for confirmation before proceeding.

## Architectural Invariants

These constraints must not be violated:

- **32-Byte Instruction:** The VM `Instruction` union must remain exactly 32
  bytes (`static_assert(sizeof(Instruction) == 32)`).
- **Stim is Immutable:** Fetch Stim via CMake `FetchContent`. Do not fork,
  vendor, or patch Stim source.
- **Single Memory Allocation:** The VM's `ShotState` must allocate its
  coefficient array exactly once based on `peak_rank`.
- **Deterministic RNG:** Do not use `std::uniform_real_distribution` (it is
  implementation-defined). Use `(rng() >> 11) * 0x1.0p-53` for `[0, 1)`.
- **No Global Topology in the VM:** All multi-qubit Pauli interference must
  be compressed into localized operations by the compiler, not evaluated at
  runtime.

## Git Workflow

1. Never commit directly to `main`. Create a feature branch.
2. Make atomic commits with conventional prefixes (`feat:`, `fix:`, `test:`,
   `docs:`).
3. Run `uv run pre-commit run --all-files` before every commit.
4. Include an `Assisted-by:` trailer in commit messages:
   ```
   Assisted-by: Claude (Opus 4.6) <noreply@anthropic.com>
   ```

## Build & Test

```bash
# Python package (recommended)
uv pip install -e .
uv run pytest tests/python/ -v

# C++ standalone
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## C++ Coding Standards

- **Standard:** C++20
- **Namespace:** All code in `namespace clifft { ... }`
- **Comments:** Explain *why*, not *what*. Omit if self-explanatory.
- **ASCII-only source:** No Unicode in source files. Use `pi` not the Greek
  letter, `|0>` not angle brackets, `Schrodinger` not the umlaut form.
- **No plan references in code:** Do not include task numbers, phase
  references, or planning details in comments or test names.

## Testing

- **C++ (Catch2):** Isolate and test components natively. Avoid special
  characters in `TEST_CASE` names (no `[]`, `()`, or `,`).
- **Python (pytest):** Validate against Qiskit-Aer for unitaries and Stim
  for stochastic circuits.
- Code coverage (`just py-cov`, `just cpp-cov`) is available but not
  required on every commit.

## When Stuck

If you cannot resolve an issue after a few attempts, stop and explain the
error clearly. Do not attempt large speculative refactors — ask for guidance.
