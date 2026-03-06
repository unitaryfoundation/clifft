# **UCC Implementation Plan: Standard Gate Set Expansion**

> **Status: COMPLETED** — Merged in PR #70 (2026-03-06).

## **Executive Summary & Constraints**

This plan expands UCC's Front-End to support the vast majority of Stim's remaining standard library, including all missing single/two-qubit Cliffords, pair measurements, Y-basis resets, and custom multi-parameter noise channels. SPP, SPP_DAG, REPEAT, and stateful noise (CORRELATED_ERROR) are explicitly deferred.

Because UCC utilizes the **Factored State Architecture**, physical Cliffords are mathematically absorbed Ahead-Of-Time (AOT) and measurements are geometrically compiled. Therefore, **this entire expansion requires absolutely zero changes to the Virtual Machine's execution loop or the 32-byte Instruction struct.**

You should see a doc/gates.md file in the stim source dependency tree pulled in as part of the build. Rember to rely on stim native definitions and functions for your implementation.

**Strict Constraints:**

1. **Zero VM Changes:** Do not add any new Opcode types to the VM for Clifford gates, measurements, or noise.
2. **Identity Exclusion:** Do not emit nodes for I, II, I_ERROR, or II_ERROR. They mathematically represent the identity operator and must be silently discarded to prevent polluting the Heisenberg IR (HIR) with zero-cost operations.

## ---

**Phase 1: Aliases, No-Ops, and Classical Padding**

**Goal:** Support alternate Stim syntax and safely ignore explicit identity operations.

* **Task 1.1 (Parser Map Aliases):** In src/ucc/circuit/parser.cc, update kGateNames to map Stim's aliases directly to our existing GateType enums:
  * "H_XZ" $\to$ GateType::H
  * "SQRT_Z" $\to$ GateType::S, "SQRT_Z_DAG" $\to$ GateType::S_DAG
  * "ZCX" $\to$ GateType::CX, "ZCY" $\to$ GateType::CY, "ZCZ" $\to$ GateType::CZ
  * "MZ" $\to$ GateType::M, "MRZ" $\to$ GateType::MR, "RZ" $\to$ GateType::R
* **Task 1.2 (No-Ops):** Add I, II, I_ERROR, and II_ERROR to GateType and kGateNames. In parser.cc's parse_standard_gate, parse their targets to safely validate them and update circuit.num_qubits, but explicitly continue (skip) before emitting an AstNode to the circuit.
* **Task 1.3 (MPAD):** Add MPAD to GateType and kGateNames. MPAD takes 0 or 1 as targets (not qubits). In frontend.cc, for each target value, emit a HeisenbergOp::make_measure with destab_mask=0, stab_mask=0, and sign set to the target value. The Back-End already naturally lowers zero-weight Paulis into OP_MEAS_DORMANT_STATIC with FLAG_IDENTITY, writing to the measurement record deterministically without touching the quantum state.
* **DoD:** The parser successfully ingests I 0; ZCX 0 1; MPAD 1 0 and correctly writes [1, 0] to the measurement record during execution.

## **Phase 2: Syntactic Sugar (Pair Measurements & Y-Resets)**

**Goal:** Support MXX/MYY/MZZ and RY/MRY by mapping them natively to existing parser and Front-End logic.

* **Task 2.1 (Pair Measurements):** Add MXX, MYY, and MZZ to GateType and kGateNames. They are of arity PAIR. In parser.cc, inside parse_standard_gate, intercept these gates. For each pair of targets, emit a single AstNode with gate = GateType::MPP containing two Target::pauli(...) targets corresponding to the requested basis. Preserve any ! inversion flags from the original targets.
* **Task 2.2 (Y-Resets):** Add RY and MRY to GateType and kGateNames. Implement them in frontend.cc by mirroring the existing logic for RX:
  * **Measurement:** Extract the rewound $Y$ observable (sim.inv_state.y_output(qubit)) and emit a MEASURE node (hidden for RY, visible for MRY).
  * **Correction:** Since $X$ anti-commutes with $Y$, applying $X$ flips the $Y$ eigenvalue. Extract the rewound $X$ observable (extract_rewound_x) and emit a CONDITIONAL_PAULI node tied to the measurement.
* **DoD:** MXX 0 1 acts exactly like MPP X0*X1, and RY 0 collapses the state to $|i\rangle$ correctly.

## **Phase 3: The Clifford Expansion**

**Goal:** Support all remaining 1-qubit and 2-qubit Clifford operations natively using a generic Stim adapter.

* **Task 3.1 (Enums):** Add the missing single-qubit (C_NXYZ, C_NZYX, C_XNYZ, C_XYNZ, C_XYZ, C_ZNYX, C_ZYNX, C_ZYX, H_NXY, H_NXZ, H_NYZ, H_XY, H_YZ, SQRT_X, SQRT_X_DAG, SQRT_Y, SQRT_Y_DAG) and two-qubit (CXSWAP, CZSWAP, ISWAP, ISWAP_DAG, SQRT_XX, SQRT_XX_DAG, SQRT_YY, SQRT_YY_DAG, SQRT_ZZ, SQRT_ZZ_DAG, SWAP, SWAPCX, XCX, XCY, XCZ, YCX, YCY, YCZ) gates to GateType and kGateNames. Add "SWAPCZ" as an alias to CZSWAP in kGateNames.
* **Task 3.2 (Generic Front-End Absorption):** Instead of manually implementing 30+ prepend_... switch cases in frontend.cc, write a generic fallback block in apply_single_qubit_clifford and apply_two_qubit_clifford:
  1. Look up the gate's raw string name via gate_name(gate).
  2. Extract its inverse tableau natively from Stim: auto inv_gate = stim::GATE_DATA.at(name).inverse().tableau<kStimWidth>().
  3. Apply it generically to the frame: sim.inv_state.inplace_scatter_prepend(inv_gate, {targets...}).
* **DoD:** A pure Clifford circuit using ISWAP, C_XYZ, and SQRT_XX compiles to zero instructions (fully absorbed AOT) and matches a Qiskit statevector oracle natively in C++.

## **Phase 4: Multi-Parameter Noise**

**Goal:** Support PAULI_CHANNEL_1 and PAULI_CHANNEL_2.

* **Task 4.1 (AST Multi-Arg):** Modify AstNode in circuit.h. Change double arg = 0.0; to std::vector<double> args;. Update the parser's parenthesized argument logic to loop, parse, and push all comma-separated floats into the vector. Update existing noise gates (like X_ERROR) in frontend.cc to use args[0].
* **Task 4.2 (Pauli Channels):** In frontend.cc, implement PAULI_CHANNEL_1 and PAULI_CHANNEL_2.
  * These gates provide exact probabilities for disjoint channels. Instantiate a NoiseSite, iterate over the provided probabilities in node.args, map the corresponding Pauli error (e.g. $X, Y, Z$ for 1Q, or $IX, IY \dots ZZ$ for 2Q) to $t=0$ through sim.inv_state, and push it into site.channels if prob > 0.
  * *Note:* The Back-End and VM already loop over NoiseSite::channels dynamically. By purely populating the AOT struct, the VM supports these gates automatically with zero code changes.
* **DoD:** A Python test proves PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0 evaluates with correct gap-sampled statistics in the VM.

## Phase 5: Native REPEAT Unrolling

**Goal:** Support Stim's `REPEAT N { ... }` control flow via text-level parser unrolling. This executes entirely AOT, requiring zero changes to the Front-End, Back-End, or Virtual Machine, while perfectly preserving `rec[-k]` semantics.

* **Task 5.1 (Safety Limits):** In `src/ucc/util/config.h`, define a new constant `constexpr size_t kMaxUnrolledOps = 10'000'000;`. This protects the compiler from Out-Of-Memory (OOM) crashes on exponentially deep nested loops.
* **Task 5.2 (Block Extraction):** In `parser.cc`, update the parsing loop to detect the `REPEAT` keyword.
1. Parse the repetition integer $N$.
2. Scan forward from the current position in the main text stream to find the opening brace `{`.
3. Continue scanning forward, keeping a `depth` counter (increment on `{`, decrement on `}`), until you find the exact matching closing brace `}`.
4. Extract the `std::string_view` of the block's body (the text between the braces).
5. Advance the outer parser's `remaining` text stream and `line_num` past the block.


* **Task 5.3 (Text-Level Replay):** Loop $N$ times.
* Inside the loop, instantiate a local `std::string_view` of the block's body and process it using the exact same line-by-line parsing logic, passing the main `circuit` by reference.
* *Crucial:* Because the text is re-parsed every iteration against the main `circuit`, `circuit.num_measurements` will naturally increment. This mathematically guarantees that `rec[-k]` relative offsets resolve perfectly, even if they point to measurements *outside* the loop!
* After parsing each line, enforce the safety limit: `if (circuit.nodes.size() > kMaxUnrolledOps) throw ParseError("Circuit exceeds maximum unrolled operations limit");`.


* **DoD:** The parser successfully ingests `M 0; REPEAT 3 { CX rec[-1] 1; M 1 }`. The resulting AST contains 7 nodes, with the `CX` gates pointing to absolute measurement targets `0`, `1`, and `2` respectively. A test attempting to unroll 20 million operations immediately raises a `ParseError`.


## Phase 6: Summarize difference

As a simple final step, create a design/gates.md file that shows a table of gates, organized by type that we implement in UCC. And then have a final section for STIM gates that UCC does not support currentyl.

---

## Open Items (Deferred)

The following Stim gates were explicitly out of scope for this plan and remain unsupported:

| Gate | Reason Deferred |
|------|----------------|
| `SPP` / `SPP_DAG` | Stochastic Pauli products require new Front-End semantics (non-deterministic Clifford frame updates). |
| `CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` | Stateful conditional noise model; requires tracking correlated error chains across multiple sites. |
| `HERALDED_ERASE` | Erasure channel with heralding flag; needs VM support for mid-circuit erasure. |
| `HERALDED_PAULI_CHANNEL_1` | Heralded single-qubit Pauli channel; same erasure infrastructure dependency. |

See `design/gates.md` for the full gate support matrix.
