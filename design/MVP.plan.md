# UCC MVP Implementation Plan

## Executive Summary & MVP Constraints
You are building the Phase 1 Minimum Viable Product (MVP) of the Unitary Compiler Collection (UCC).

**Strict Constraints for this specific MVP phase (Overrides general guidelines if in conflict):**
1. **64-Qubit Fast-Path ONLY:** Limit all Pauli masks to strictly $\le 64$ qubits using inline `uint64_t`. Do not implement Stim's `simd_bits` template monomorphization yet.
2. **No C++ CLI:** **DO NOT** write a `main.cc` or a command-line executable. The engine is invoked exclusively via the Python API.
3. **Stub the Optimizer:** Pass the Heisenberg IR (HIR) directly from the Front-End to the Back-End. We are skipping Middle-End commutation/fusion for this milestone.
4. **Features:** Noiseless simulation only. Do not implement gap sampling, error models, or complex annotations.
5. **Circuit Scope:** Support basic Cliffords (`H`, `S`, `S_DAG`, `X`, `Y`, `Z`, `CX`, `CY`, `CZ`), $T$ and $T^\dagger$ gates, Measurements (`M`, `MX`, `MPP`), and classical Pauli feedback (`CX rec[-k] q`, `CZ rec[-k] q`).

---

## Phase 1: Project Setup & Build Infrastructure
**Goal:** Establish the CMake and Python build pipeline using `scikit-build-core` for seamless local Python development.
*   **Task 1.0 (Code Quality Scaffolding):**
    *   Create a robust `.gitignore` for C++ (CMake build dirs), Python (`__pycache__`, `.venv`, `*.so`), and IDEs.
    *   Configure `ruff` (lint/format) and `mypy` (strict typing) sections inside `pyproject.toml`. Add `pre-commit` to the dev dependencies.
    *   Create a standard `.clang-format` file (e.g., BasedOnStyle: Google, ColumnLimit: 100).
    *   Create `.pre-commit-config.yaml` hooking into `clang-format`, `ruff`, and `mypy`.
*   **Task 1.1:** Create `CMakeLists.txt` to compile a `ucc_core` library. Use `FetchContent` to pull exactly `v1.15.0` of `https://github.com/quantumlib/Stim.git` (https://github.com/quantumlib/Stim/tree/v1.15.0), and Catch2 (v3).
*   **Task 1.2:** Create `pyproject.toml` configured with `scikit-build-core` and `nanobind`.
*   **Task 1.3:** Create a minimal C++ function, bind it in `src/python/bindings.cc`, and expose it in `src/python/ucc/__init__.py`.
*   **Definition of Done (DoD):** Running `uv pip install -e .` succeeds, and the dummy function can be called from Python without errors. unning `uv run pre-commit run --all-files` passes completely with zero formatting, linting, or typing errors. The `git status` tree is completely free of untracked build artifacts, respecting the `.gitignore`.

## Phase 2: Circuit AST & Parser
**Goal:** Read a stripped `.stim` file into an abstract syntax tree.
*   **Task 2.1:** Define `ucc::Circuit`, `ucc::AstNode`, and the `ucc::GateType` enum. Include fields for targets and measurement record indices (`rec[-k]`).
*   **Task 2.2:** Write a line-by-line parser.
    *   *Resets Trick:* AOT simulators cannot branch the Clifford frame. Mathematically, a reset is a measurement followed by a conditionally applied Pauli. Decompose `R q` into `M q` followed by `CX rec[-1] q` (apply X if measure was 1). Decompose `RX q` into `MX q` followed by `CZ rec[-1] q`. This requires zero new SVM opcodes!
    *   Parse `MPP` into a list of Pauli targets.
    *   Parse `CX rec[-k] q` as a feedback operation.
*   **DoD:** A C++ Catch2 test parses a string containing gates, measurements, and a reset, verifying the AST matches the decomposed structure.

## Phase 3: Front-End (Trace Generation)
**Goal:** Drive Stim's `TableauSimulator` to absorb Cliffords and emit the Heisenberg IR (HIR).
*   **Task 3.1 (HIR Structs):** Define `HeisenbergOp` using `uint64_t` for `destab_mask` and `stab_mask`. Define `HirModule`.
*   **Task 3.2 (Clifford & T Rewinding):** Iterate over the AST. When a Clifford is hit, apply it directly to a `stim::TableauSimulator<64>`. When `T` or `T_DAG` is hit, extract the rewound Pauli Z and X frames from `sim.inv_state.zs[q]` and `sim.inv_state.xs[q]`, and emit a `HeisenbergOp` (with type `T_GATE` or `T_DAG_GATE`).
    *   *(Debug Optional):* At the end of trace generation, extract and save the forward tableau (`sim.inv_state.inverse()`) to `HirModule::final_tableau` for statevector debugging.
*   **Task 3.3 (Measurements & AG Pivots):** When an `M`, `MX`, or `MPP` is hit, rewind the observable. If the measurement commutes with the tableau, record it. If it anti-commutes, compute the Aaronson-Gottesman (AG) pivot matrix using `fwd_after.then(inv_before)`, update the tableau, append the matrix to `HirModule::ag_matrices`, and emit `HeisenbergOp::MEASURE`.
*   **Task 3.4 (Classical Control):** For `CX rec[-k] q` (and the decomposed resets), rewind the target Pauli ($X_q$ or $Z_q$). Emit an `OP_CONDITIONAL_PAULI` operation in the HIR containing the rewound mask and the resolved absolute index of the measurement record.
*   **DoD:** A test feeding `H 0; T 0` verifies the emitted HIR mask corresponds to the $+X$ axis. A test feeding `H 0; M 0` verifies an AG matrix is generated.

## Phase 4: Compiler Back-End (Code Generation)
**Goal:** Lower the HIR into execution-engine bytecode (Skipping the Optimizer for now).
*   **Task 4.1:** Define the `Instruction` struct (union). **Assert that it is exactly 32 bytes.** Define the `ConstantPool`.
*   **Task 4.2 (GF2 Tracking):** Iterate through the HIR. Maintain the active GF(2) basis $V$ (up to 64 `uint64_t` vectors). For `T_GATE`, evaluate the spatial shift $\beta$ against $V$. Emit `OP_BRANCH` (if new dimension), `OP_COLLIDE` (if in basis), or `OP_SCALAR_PHASE` (if $\beta=0$). Compute `x_mask` and `commutation_mask`. Track `peak_rank`. Store the final $V$ basis in `ConstantPool::gf2_basis` if debugging/targetting state vector output.
*   **Task 4.3 (Measurements & Feedback):** Lower `MEASURE` operations into `OP_MEASURE_MERGE`, `OP_MEASURE_FILTER`, or `OP_MEASURE_DETERMINISTIC` by evaluating against $V$. Append `OP_AG_PIVOT` if the Front-End provided an AG matrix index. Lower `OP_CONDITIONAL_PAULI` directly into a bytecode instruction.
*   **DoD:** A test showing 4 independent $T$ gates emits 4 `OP_BRANCH` opcodes and yields a `peak_rank` of 4.

## Phase 5: Schrodinger Virtual Machine Runtime
**Goal:** Execute the bytecode and track the complex coefficient array.
*   **Task 5.1:** Define `SchrodingerState` containing a dynamically allocated `std::complex<double>` array of size $2^{\text{peak\_rank}}$, `uint64_t destab_signs`, `uint64_t stab_signs`, and a `std::vector<uint8_t> meas_record`. Include a deterministic PRNG.
*   **Task 5.2:** Implement the SVM switch statement for all opcodes.
    *   `OP_BRANCH`, `OP_COLLIDE`, `OP_SCALAR_PHASE`.
    *   `OP_MEASURE_MERGE`, `OP_MEASURE_FILTER`, `OP_MEASURE_DETERMINISTIC`, `OP_AG_PIVOT`.
    *   `OP_CONDITIONAL_PAULI` (if `meas_record[idx] == 1`, XOR the masks into `destab_signs`/`stab_signs`).
*   **Task 5.3:** Expose `ucc.compile` and `ucc.sample(program, shots)` via nanobind.
*   **DoD:** Python can successfully call `ucc.sample()` on a compiled program and receive a numpy array of results.

## Phase 6: Validation & Integration Testing
**Goal:** Prove the system produces mathematically correct physics.
*   **Task 6.1 (Statevector Expansion Utility):** Write a Python utility `extract_statevector` that:
    1. Allocates a dense $2^N$ numpy array.
    2. For each GF(2) index $\alpha$ (from 0 to $2^{\text{rank}}-1$), compute the physical basis index by XORing the columns of `gf2_basis` selected by the bits of $\alpha$.
    3. Apply the physical $\pm 1, \pm i$ signs from `destab_signs`/`stab_signs` using the `final_tableau` columns to determine which stabilizer/destabilizer generators contribute.
    4. Place `v[α] * sign * global_weight` at the computed physical index.
*   **Task 6.2 (Statevector Micro-Test):** Use the utility from 6.1 to validate a 4-qubit pure unitary circuit (Cliffords + $T$, no measurements) against a known statevector oracle using `np.allclose`.
*   **Task 6.3 (Statistical Macro-Test):** Load the manually stripped Gidney proxy circuit (S gates only). Run 10,000 shots in `stim` and 10,000 shots in `ucc`. Assert that the probability distributions of the resulting measurement bitstrings match within a strict statistical tolerance (e.g., $< 0.02$ divergence).
*   **Task 6.4 (Statevector Debugging API):** Add a `debug_mode=False` flag to `ucc.compile()`. When true, retain `final_tableau` and `gf2_basis` on the returned `Program`. Expose `ucc.to_statevector(program, schrodinger_state)` that leverages the math from Task 6.1.
*   **DoD:** All tests pass reliably via `pytest`.
