# UCC — Software Architecture & Interfaces

This document describes the concrete software architecture of UCC: directory structure, the Stim integration contract, and the C++ testing strategy.

## 1. Repository Layout

The codebase strictly mirrors the 5-stage pipeline:

*   `src/ucc/circuit/`: Circuit AST, parser, and target encoding.
*   `src/ucc/frontend/`: Drives `stim::TableauSimulator`, absorbs physical Cliffords, emits the Heisenberg IR (HIR).
*   `src/ucc/optimizer/`: Two-level optimization: HIR passes (pure Heisenberg IR manipulation) and bytecode passes (post-lowering instruction fusion).
*   `src/ucc/backend/`: Tracks the virtual frame mapping ($V_{cum}$) and Active/Dormant sets. Synthesizes virtual basis compression. Emits localized RISC bytecode.
*   `src/ucc/svm/`: The runtime Virtual Machine. Executes the RISC bytecode over a dense $2^k$ array and lightweight `ucc::BitMask<kMaxInlineQubits>` Pauli frames.

**Isolation Invariant:** The VM (`svm/`) must never include `stim::Tableau` or evaluate tableau mathematics. It executes purely on basic C++ types and arrays.

## 1b. Two-Level Optimizer

Optimization occurs at two distinct IR levels, each with its own pass manager:

### HIR Passes (Pre-Lowering)
Operate on the Heisenberg IR before bytecode emission. Managed by `PassManager`.

*   **PeepholeFusionPass:** Fuses consecutive T/T-dagger gates on the same qubit (T+T=S, T+T_dag=identity).

### Bytecode Passes (Post-Lowering)
Operate on the finalized RISC bytecode after the Back-End has lowered the HIR. Managed by `BytecodePassManager`. These passes rewrite, reorder, or fuse instructions:

*   **NoiseBlockPass:** Collapses runs of identical `OP_NOISE` instructions into single `OP_NOISE_BLOCK` ops, reducing dispatch overhead.
*   **MultiGatePass:** Fuses contiguous `OP_ARRAY_CNOT` instructions sharing a target axis into `OP_ARRAY_MULTI_CNOT`, and contiguous `OP_ARRAY_CZ` sharing a control axis into `OP_ARRAY_MULTI_CZ`. These star-graph fusions process all controls/targets in one $O(2^k)$ array pass instead of many.
*   **ExpandTPass:** Fuses `OP_EXPAND` immediately followed by `OP_PHASE_T` (or `T_DAG`) into a single `OP_EXPAND_T` (or `OP_EXPAND_T_DAG`) that duplicates and phase-rotates in one loop.
*   **SwapMeasPass:** Fuses `OP_ARRAY_SWAP` immediately followed by `OP_MEAS_ACTIVE_INTERFERE` into `OP_SWAP_MEAS_INTERFERE`, eliminating a redundant array pass.

The default pipeline runs all passes in order: `NoiseBlockPass -> MultiGatePass -> ExpandTPass -> SwapMeasPass`.

## 2. The Stim Integration Contract (CRITICAL)

UCC uses `Stim` exclusively as an AOT mathematical tableau library, **not** as a circuit engine. The runtime Virtual Machine never touches `stim::TableauSimulator`.

Because UCC factors the state into physical and virtual coordinate frames, the compiler must manipulate the stabilizer frame from *both ends* of the circuit. We map this to Stim's APIs as follows:

### Front-End: Physical Lab Frame (Prepending)
The Front-End tracks $U_{phys}^\dagger$. As it steps forward through the circuit, it must mathematically **prepend** physical gates to the inverse tableau.
*   **API:** `sim.inv_state.prepend_H_XZ()`, `sim.inv_state.prepend_ZCX()`.
*   **Extraction:** `sim.inv_state.zs[q]` extracts the physical $Z_q$ operator rewound to $t=0$.

### Back-End: Virtual Coordinate Frame (Appending)
The Back-End synthesizes the virtual compression sequence $V$ that maps the $t=0$ space into the localized array space. Because these virtual gates act on the "input" side of the global state, they must be mathematically **appended** to the tracking frame $V_{cum}$.
*   **API:** We use Stim's `TableauTransposedRaii` wrapper around $V_{cum}$, which transposes the tableau in memory to make row operations (appending) computationally efficient.
*   **Execution:** `transposed_raii.append_ZCX()`, `transposed_raii.append_H_XZ()`.

At the end of compilation, the final offline Clifford frame required for statevector expansion is computed as: $U_C = U_{phys} V_{cum}^\dagger$.

## 3. Testing Strategy (C++ Native)

We test the mathematical transformations layer-by-layer exclusively in C++ using `Catch2`.

1. **Parser Tests (`test_parser.cc`):** Validates lexical conversion of `.stim` text to `ucc::Circuit` AST.
2. **Front-End Tests (`test_frontend.cc`):** Validates that Heisenberg rewinding exactly matches Stim's mathematical definition of $U_{phys}^\dagger P U_{phys}$.
3. **Virtual Compression Tests (`test_backend.cc`):** Feeds random, heavy `stim::PauliString` masks into the Back-End's compressor. Asserts that the resulting $V_{cum}$ successfully compresses the operator to a single virtual qubit.
4. **RISC Math Tests (`test_svm_risc.cc`):** Bypasses the compiler entirely. Manually constructs `Instruction` opcodes and a dummy `SchrodingerState`. Executes them to assert the pure array math and Pauli frame XORs perfectly match theoretical statevector operations.
5. **Statevector Oracle (`test_statevector.cc`):** Expands the VM's factored state representation ($|\psi\rangle = \gamma U_C P |\phi\rangle_A$) into a dense $2^n$ statevector, and verifies it against pure unitary matrix multiplication.

## 4. Python Bindings
UCC uses `nanobind` to expose the C++ core to Python. The Python layer provides `ucc.compile()` and `ucc.sample()`, which serve as the primary interface for statistical integration tests (e.g., comparing UCC measurement distributions against Qiskit-Aer or Stim).
