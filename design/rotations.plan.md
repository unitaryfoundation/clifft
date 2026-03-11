# UCC Implementation Plan: Arbitrary Continuous Rotations

## Executive Summary & Architectural Strategy

This plan extends UCC to natively support arbitrary continuous parameterized rotations ($R_X$, $R_Y$, $R_Z$, $U_3$, and $N$-qubit Pauli rotations) within the Factored State Architecture.

By strictly avoiding the synthesis of these rotations into discrete Clifford+T gates, UCC preserves its bounded active statevector dimension ($k_{\max}$) and its hyper-fast $\mathcal{O}(1)$ branchless array loops.

**Core Architectural Strategies:**
1.  **Clifford Absorption ($U_3$, $R_X$, $R_Y$):** We do *not* add dense $2 \times 2$ matrices to the VM. The Front-End mathematically desugars these gates using standard Cliffords ($H$, $H_{YZ}$) and $R_Z$ rotations. The Cliffords are completely absorbed into the tracking tableau AOT, leaving only localized $Z$-rotations for the VM.
2.  **Unitary Phase Factoring:** The mathematical definition $R_Z(\alpha) = \exp(-i\frac{\alpha\pi}{2} Z)$ factors exactly into a global phase $e^{-i\alpha\pi/2}$ and a relative diagonal phase $\text{diag}(1, e^{i\alpha\pi})$. The Front-End extracts the global phase into `global_weight` (costing zero runtime FLOPs), and the VM only multiplies the relative phase $z = e^{i\alpha\pi}$ onto the $|1\rangle$ array branch.
3.  **The 32-Byte Invariant:** The relative complex phase ($z$) fits perfectly within the existing 24-byte `Instruction::math` payload union, allowing the VM to fetch the rotation angles directly from the L1 cache instruction stream without constant-pool lookups.

Additionally, as you add them, ensure these new gates, HIR nodes and opcodes are added to the documentation and information for the explorer.

---

## Phase 1: Syntax & AST Extension

**Goal:** Parse the new parameterized gates and extract their angular arguments.

*   **Task 1.1 (Gate Definitions):** In `gate_data.h`, add `R_X`, `R_Y`, `R_Z`, `U3` (Arity `SINGLE`), `R_XX`, `R_YY`, `R_ZZ` (Arity `PAIR`), and `R_PAULI` (Arity `MULTI`). Update the `GateTraits` table.
*   **Task 1.2 (Parser Logic):** In `parser.cc`:
    *   Add the new gates to `kGateNames`.
    *   Ensure the existing `args` extraction properly captures `alpha` for `R_X/Y/Z/XX/YY/ZZ/PAULI` (1 arg) and `theta, phi, lambda` for `U3` (3 args).
    *   For `R_PAULI`, adapt the existing `parse_mpp` logic to parse `X0*Y1*Z2` token sequences into Pauli-tagged targets.

## Phase 2: The Heisenberg IR (HIR) & Clifford Absorption

**Goal:** Translate AST nodes into abstract, geometrically-rewound Pauli rotations.

*   **Task 2.1 (HIR Struct):** In `hir.h`, add `OpType::PHASE_ROTATION`. Add `struct { double alpha; } phase_;` to the 12-byte `HeisenbergOp` union payload. Add a factory `make_phase_rotation(destab, stab, sign, alpha)`.
*   **Task 2.2 (Tracing $R_Z$ & Paulis):** In `frontend.cc`:
    *   For $R_Z(\alpha)$: Extract the rewound $Z$ operator. Accumulate the global phase factor $e^{-i\alpha\pi/2}$ into `hir.global_weight`. Emit `PHASE_ROTATION(alpha)` to the HIR.
    *   For $R_{XX}$, $R_{YY}$, $R_{ZZ}$: Similar to above, but the initial rewound Pauli is $X \otimes X$, $Y \otimes Y$, or $Z \otimes Z$.
    *   For `R_PAULI`: Construct the combined Pauli string from the tagged targets, rewind it, and emit `PHASE_ROTATION(alpha)`.
*   **Task 2.3 (Clifford Desugaring):**
    *   For $R_X(\alpha)$: Call `sim.inv_state.prepend_H_XZ()`, trace $R_Z(\alpha)$ logic, call `sim.inv_state.prepend_H_XZ()`.
    *   For $R_Y(\alpha)$: Call `sim.inv_state.prepend_H_YZ()`, trace $R_Z(\alpha)$ logic, call `sim.inv_state.prepend_H_YZ()`.
    *   For $U_3(\theta, \phi, \lambda)$:
        *   Trace $R_Z(\lambda)$ logic.
        *   Call `sim.inv_state.prepend_H_YZ()`, trace $R_Z(\theta)$ logic, call `sim.inv_state.prepend_H_YZ()`.
        *   Trace $R_Z(\phi)$ logic.
        *   *Note:* Ensure you apply the appropriate phase shift scalar to `hir.global_weight` so the final matrix matches the standard Qiskit $U_3$ definition (which factors out an overall phase of $e^{i(\phi+\lambda)\pi/2}$ relative to the raw $R_Z R_Y R_Z$ sequence).

## Phase 3: Middle-End Fusion

**Goal:** Prevent continuous phase sequences from bloating the bytecode stream.

*   **Task 3.1 (Angle Addition):** In `peephole.cc`, update the fusion logic. If two `PHASE_ROTATION` nodes target the exact same `destab_mask` and `stab_mask`, fuse them by adding their angles: $\alpha_{\text{new}} = \alpha_1 + \alpha_2$ (accounting for relative sign).
*   **Task 3.2 (Identity Cancellation):** If the fused angle modulo 4.0 is exactly $0.0$ (within floating point limits, i.e., a full $4\pi/2 = 2\pi$ rotation), delete the node entirely.

## Phase 4: Back-End Compression & RISC Emission

**Goal:** Geometrically fold multi-qubit continuous rotations into localized VM instructions.

*   **Task 4.1 (Opcode Definition):** In `backend.h`, add `OP_PHASE_ROT` and `OP_EXPAND_ROT` to the `Opcode` enum. The `math` union variant already holds `double weight_re` and `weight_im`.
*   **Task 4.2 (Lowering):** In `backend.cc`, when processing a `PHASE_ROTATION` node:
    *   Call `compress_pauli(ctx, p_v)` and `route_to_active_z(ctx, result)`.
    *   If the compression accumulated a negative sign (`result.sign == true`), negate the angle ($\alpha = -\alpha$).
    *   If the node has its `sign()` flag set to true, negate the angle again.
    *   Calculate the relative phase $z = e^{i\alpha\pi} = \cos(\alpha\pi) + i\sin(\alpha\pi)$.
    *   If the target axis is dormant, emit `OP_EXPAND_ROT(result.pivot, z.real(), z.imag())`.
    *   If active, emit `OP_PHASE_ROT(result.pivot, z.real(), z.imag())`.
*   **Task 4.3 (Bytecode passes):** Implement an `ExpandRotPass` to fuse adjacent `OP_EXPAND` and `OP_PHASE_ROT` into `OP_EXPAND_ROT` during the bytecode optimization phase.

## Phase 5: Virtual Machine Execution

**Goal:** Evaluate the continuous complex math dynamically based on the Pauli Frame.

*   **Task 5.1 (Execution Handlers):** In `svm.cc`, implement `exec_phase_rot` and `exec_expand_rot`.
    *   Extract $z = \text{weight\_re} + i \cdot \text{weight\_im}$.
    *   Check the Pauli frame: `bool px = bit_get(state.p_x, axis)`.
    *   If `px == false` ($X$-error absent): Multiply the $|1\rangle_{\text{axis}}$ branch of the array by $z$.
    *   If `px == true` ($X$-error present): Because $X$ anti-commutes with $Z$, the phase direction mathematically inverts. Multiply the array branch by the complex conjugate $z^*$. Multiply the global scalar `gamma` by $z$ to preserve the factored state equivalence.
    *   (For `EXPAND_ROT`, combine the array copy and phase multiplication into a single branchless loop, mirroring `EXPAND_T`).

## Phase 6: Validation & Qiskit Oracles

**Goal:** Mathematically prove that the $N$-qubit continuous array math identically matches dense matrix simulators.

*   **Task 6.1 (Qiskit Translation):** Update `utils_qiskit.py` to parse the new syntax and map it to Qiskit operations (`qc.rx`, `qc.ry`, `qc.rz`, `qc.rzz`, `qc.u`, and `PauliEvolutionGate`).
*   **Task 6.2 (Statevector Fidelity):** In `test_qiskit_aer.py`, add `TestArbitraryRotations`. Generate random circuits containing the new gates and assert that the UCC output statevector maintains $>0.9999$ fidelity with Qiskit-Aer.
