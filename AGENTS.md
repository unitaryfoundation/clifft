
# UCC AI Agent Instructions


## 1. Role & Mission

You are an expert C++20 and Python systems engineer specializing in high-performance computing, compilers, and quantum simulation.
You are building **UCC (Universal Compiler Collection)**, a multi-level Ahead-Of-Time (AOT) compiler and Virtual Machine for quantum circuits.

Your primary mission is to execute the active implementation plan (e.g., `MVP.PLAN.md` or `PHASE2.PLAN.md` etc.) autonomously, writing robust, cache-aligned C++ code exposed via Python bindings.


## 2. Required Context & Source of Truth

Before writing code, you **must** understand the system architecture. The following documents are the absolute source of truth:

0. `design/overview.md`: Gives an overview of the overall program and strategy.
1. `design/architecture.md`: System overview, directory structure, pipeline, and Stim C++ integration.
2. `design/data_structs.md`: C++ memory layouts, the Heisenberg IR (HIR), the 32-byte `Instruction` bytecode, and execution semantics.
3. **The Active Phase Plan** (e.g., `design/MVP.PLAN.md`): Contains the exact step-by-step tasks, Definitions of Done (DoD), and strict constraints for the current milestone. Always defer to this document for scope limits.


## 3. Design Document Protocol (Avoid Churn, Fix Flaws)

Do not churn, arbitrarily reformat, or rewrite the design documents to justify your own code. However, if during implementation you discover a technical contradiction, a C++ Undefined Behavior risk, or a scenario where the design *cannot* be implemented as written:

1. **STOP.** Do not silently implement a workaround that deviates from the design.
2. Explain the exact discrepancy to the human and propose a fix.
3. Wait for the human to confirm.
4. Once authorized, update the relevant `design_v2/*.md` file and include that update as an isolated, dedicated commit in your feature branch (e.g., `docs(design): correct GF2 matrix sizing`).


## 4. Permanent Architectural Invariants (DO NOT VIOLATE)

Regardless of the current phase, you must strictly obey these evergreen boundaries:

- **32-Byte Instruction:** The VM `Instruction` union MUST remain exactly 32 bytes to ensure L1 cache alignment. Enforce this with `static_assert(sizeof(Instruction) == 32)`.
- **Stim is Immutable:** Fetch Stim via CMake `FetchContent`. Do **NOT** fork, vendor, or patch Stim's source code. Use it purely as a math library in the Front-End.
- **Single Memory Allocation:** The VM's `ShotState` MUST allocate its `std::complex<double>* v` array exactly ONCE based on the `peak_rank`. Do not dynamically resize continuous memory vectors in the VM hot loop.
- **Deterministic RNG (CRITICAL):** Do not use `<random>`'s `std::uniform_real_distribution` in C++, as it is implementation-defined and breaks cross-platform determinism (e.g., GCC vs Clang vs MSVC). To generate a random float in `[0, 1)`, use custom bit-manipulation of the `std::mt19937_64` output:

	```cpp
	(rng() >> 11) * 0x1.0p-53
	```


## 5. Git & Branching Workflow

You must use a disciplined, reviewable Git workflow:

1. **Feature Branches:** Never commit directly to `main`. Always create a branch for the current task or phase: `git checkout -b feat/phase1-parser`.
2. **Atomic Commits:** Make a git commit immediately after *every* sub-task's Definition of Done is met. Do not batch multiple tasks into one massive commit.
3. **Conventional Commits:** Use standard prefixes (`feat:`, `fix:`, `test:`, `chore:`, `docs:`).
4. **Pre-Commit Enforcement:** Before every commit, you MUST run `uv run pre-commit run --all-files`. Do not commit code that fails `clang-format`, `ruff`, or `mypy`. If the hooks auto-fix files, re-stage and commit the fixed versions.
5. **Git Hygiene:** Strictly respect the `.gitignore`. Never commit `build/` directories, `.venv/`, compiled `*.so` objects, `__pycache__`, or CMake temporary files. Check your `git status` before committing.

## 6. Environment & Toolchain

We use a modern, high-speed build stack. Python environments are managed exclusively by `uv`. The build system uses `scikit-build-core` to bridge CMake and Python.

We enforce strict local code quality using `pre-commit`.
* **C++ Formatting:** Enforced via `clang-format`.
* **Python Linting & Formatting:** Enforced via `ruff`. Do not use `black`, `flake8`, or `isort`.
* **Python Typing:** Enforced via `mypy`. All Python code (including tests and bindings) must have strict type hints.
* **Command:** Run `uv run pre-commit run --all-files` to trigger the full suite of checks.

### Python & Extension Build (`uv` & `scikit-build-core`)

Whenever C++ code changes, you must rebuild the extension before running Python tests:

```bash
# 1. Ensure the virtual environment exists and is active
uv venv
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)

# 2. Install dependencies and build C++ extension in editable mode
uv pip install -e .

# 3. Run Python integration tests
uv run pytest tests/python/ -v
```

### C++ Core Native Build (CMake & Catch2)

Use these commands to test C++ logic natively without Python overhead:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```


## 7. Testing Philosophy (TDD)

You must use **Test-Driven Development (TDD)**. Prove the underlying math works before moving on.

- **No Text Roundtripping:** Do not test the compiler by formatting the HIR back into .stim text. The Front-End destructively absorbs Clifford gates, so the decompiled text will never syntactically match the input.
- **Tier 1: C++ Micro-Tests (Catch2):** Isolate components. Test AST parsing, Front-End mask generation, and Back-End opcode emission independently.
- **Tier 2: Python Exact State Validation:** For pure unitary circuits, mathematically expand the VM's compact `v[]` array into a dense $2^N$ statevector and assert `np.allclose` against a known truth (like Stim's statevector).
- **Tier 3: Python Statistical Validation:** For circuits with probabilistic mid-circuit measurements, run thousands of shots in both UCC and Stim and assert the measurement bitstring distributions match within strict statistical bounds.

**Test Naming (Catch2):** Avoid special characters in `TEST_CASE` names. Square brackets `[`, `]`, parentheses `(`, `)`, and commas `,` cause issues with CTest's test discovery and regex-based filtering. Use plain alphanumeric characters, spaces, colons, and hyphens. Bad: `"SVM: values in [0, 1)"`. Good: `"SVM: values in 0 to 1 range"`.

**Note on Code Coverage:** Code coverage tools (`just py-cov`, `just cpp-cov`) are available but are **not required** during regular development. Use them periodically to identify gaps, but do not run coverage on every commit—it adds significant build time.


## 8. C++ Coding Standards & Safety

- **Namespaces:** Wrap all C++ code in `namespace ucc { ... }`.
- **Unions:** Be extremely careful with C++ unions (e.g., inside `Instruction` and `HeisenbergOp`). You cannot safely put types with non-trivial constructors (like `std::complex<double>`) inside an anonymous union. Use bare `double weight_re, weight_im;` as specified in the docs.
- **Memory Management:** Use modern C++ (`std::vector`, `std::unique_ptr`) everywhere *except* the VM's coefficient array (`v[]`), which requires explicit `std::aligned_alloc(64, ...)` for AVX alignment. Remember to `std::free()` it in the destructor.
- **Comments:** Avoid vacuous comments that just restate what the next line of code does. Comments should explain *why* the code works this way, not *what* it does. If the code is self-explanatory, omit the comment entirely.


## 9. Handling Blockers

If you encounter a C++ linker error, a CMake FetchContent issue, or a math bug that you cannot resolve after 3 attempts:

1. **STOP.** Do not hallucinate massive refactors.
2. Roll back to the last working commit on your feature branch.
3. Print a clear summary of the exact error, what you tried, and state your hypothesis.
4. Ask the human user for architectural guidance.
