# UCC Implementation Plan: Dual-Mode Architecture & Global Optimization

## Executive Summary

This plan formalizes the duality between state-aware execution (Simulation) and state-agnostic optimization (Hardware Export). It establishes a modular `PassManager` and implements two critical passes: a linear Peephole sweep and the cubic FastTODD/Heisenberg-TOHPE algorithm (Vandaele 2025 Algorithm 3). By utilizing a fast "Fold-Forward" bitwise update for newly formed Cliffords, it guarantees maximum global clique visibility for FastTODD while eliminating the need for complex CNOT synthesis.

### Strict Constraints

1. **Composition Over Inheritance (HIR):** Use strict C++ struct composition for the HIR types (`StateAwareHir` contains a `HirCore`). Do NOT use virtual inheritance.
2. **Immediate Fold-Forward:** When an optimization pass fuses a $T \cdot T \to S$ (or any other Clifford), the pass MUST immediately mathematically fold that new $S$ gate into the masks of all downstream operations via $\mathcal{O}(m)$ symplectic inner product updates.
3. **Type-Safe Lowering:** `ucc::lower()` must strictly accept only `const StateAwareHir&`. Passing a `StateAgnosticHir` must be caught at compile time (C++) or binding time (nanobind).

---

## Phase 1: Dual-Mode Architecture & Type Safety (C++)

**Goal:** Replace the monolithic `HirModule` with composed, type-safe structures that physically prevent executing state-agnostic routing templates.

* **Task 1.1 (Core Data):** In `src/ucc/frontend/hir.h`, rename the existing `HirModule` to `HirCore`. Remove `ag_matrices` and `final_tableau` from it entirely. It should only contain `std::vector<HeisenbergOp> ops`, `num_qubits`, `num_measurements`, and `global_weight`.
* **Task 1.2 (Type Wrappers):** Define `struct StateAwareHir` containing a `HirCore` core, `ag_matrices`, and `final_tableau`. Define `struct StateAgnosticHir` containing only a `HirCore` core.
* **Task 1.3 (State-Agnostic Trace):** In `frontend.cc`, rename the standard trace to `trace_state_aware`. Implement `trace_state_agnostic(const Circuit& circuit)`.
* Initialize `stim::TableauSimulator` with an Identity Tableau.
* When it hits a `MEASURE` node, **skip AG pivot computation entirely**. Rewind the observable, emit a `MEASURE` op with `ag_matrix_idx = None`, and continue without modifying the unitary frame.


* **Task 1.4 (Back-End Signature):** Update `ucc::lower()` to strictly require `const StateAwareHir& hir`. Update its internals to read from `hir.core.ops`.
* **Task 1.5 (Python API):** In `bindings.cc`, bind both HIR types. Do *not* expose `HirCore` to Python. Bind the two trace functions. Bind `ucc.lower` strictly to `StateAwareHir` (passing agnostic raises `TypeError`). Add `.export_json(filename)` and `.to_qasm()` methods to `StateAgnosticHir`.
* **DoD:** C++ code compiles natively. `sizeof(HeisenbergOp)` remains exactly 32 bytes. Catch2 tests verify `trace_state_agnostic` evaluates anti-commuting measurements without generating AG pivot matrices. Python tests verify `ucc.lower(agnostic_hir)` correctly triggers a `TypeError`.

## Phase 2: PassManager & The "Fold-Forward" Engine

**Goal:** Establish the pass pipeline infrastructure and the fast $\mathcal{O}(m)$ bitwise geometric update utility.

* **Task 2.1 (Pass Interface):** Create `src/ucc/optimizer/pass.h`. Define a pure abstract base class `Pass { virtual void run(HirCore& core) = 0; virtual ~Pass() = default; }`. Define `PassManager` that accepts `unique_ptr<Pass>` and executes them sequentially via composition overloads (`run(StateAwareHir&)` and `run(StateAgnosticHir&)`).
* **Task 2.2 (The Fold-Forward Math):** In `src/ucc/optimizer/pass_utils.cc`, implement `apply_clifford_frame_shift(HirCore& core, size_t start_idx, stim::bitword<64> p_destab, stim::bitword<64> p_stab)`.
* Iterate from `start_idx` to the end of `core.ops`.
* Evaluate the symplectic inner product to check commutation between the downstream operation $Q$ and the formed Clifford axis $P$.
* If they anti-commute, compute the geometric phase shift $Q' \propto -i Q P$: perform a bitwise XOR (`Q.destab ^= P.destab`, `Q.stab ^= P.stab`) and flip the relative sign bit in the payload appropriately.


* **Task 2.3 (Barrier Respect):** Implement a utility `bool can_commute_past(const HirCore& core, size_t op_idx, size_t target_idx)` that verifies an operation can slide left/right without hitting an anti-commuting `MEASURE` or `NOISE` node using the symplectic inner product.
* **DoD:** A Catch2 test manually applies a frame shift to a downstream anti-commuting $X$ mask and verifies it correctly rotates into a $Y$ mask natively in memory, executing in microseconds.

## Phase 3: The Peephole Fusion Pass

**Goal:** Implement fast, adjacent algebraic cancellation.

* **Task 3.1 (Peephole Logic):** Implement `PeepholeFusionPass : public Pass`. Iterate left-to-right. For each non-Clifford gate, scan ahead (respecting barriers) for a gate with identical `destab` and `stab` masks.
* **Task 3.2 (Algebraic Fusion):** If found, apply Dominant Term Factoring algebra: compute $c_{fused} = (c_1 + c_2)/(1 + c_1 c_2)$. Multiply the extracted scalar $(1 + c_1 c_2)$ into `core.global_weight`.
* **Task 3.3 (Cancellation & Fold-Forward):**
* If $c_1 + c_2 = 0$ (e.g., $T \cdot T^\dagger = I$): Delete both nodes.
* If $1 + c_1 c_2 = 0$ (e.g., $T \cdot T = S$): Delete both nodes, and **immediately** call `apply_clifford_frame_shift` on the remainder of the array using the masks of the formed $S$ gate to heal the downstream geometry.


* **DoD:** A Catch2 test feeds `H 0; T 0; T 0; H 0; T 0`. The pass fuses the first two T's into an $S$, instantly updates the third T's masks via fold-forward, resulting in a perfectly clean, Clifford-free `HirCore`.

## Phase 4: The FastTODD Pass (Vandaele 2025)

**Goal:** Implement the global $\mathcal{O}(n^4 m^3)$ 3-STR null-space extraction algorithm (Algorithm 3) natively on the HIR.

* **Task 4.1 (GF2 Utilities):** Add a fast bitwise GF(2) Gaussian elimination utility in `pass_utils.cc` to extract the right null-space of a binary matrix.
* **Task 4.2 (Clique Extraction):** Implement `FastToddPass : public Pass`. Sweep the IR to identify cliques of mutually commuting `T_GATE` and `T_DAG_GATE` operations. The clique terminates when an operation anti-commutes with the active clique or hits a stochastic barrier.
* **Task 4.3 (Matrix L Construction & Null-Space):** For a clique of size $m$, construct the parity table $P$ ($n \times m$). Construct the binary matrix $L$ where $L_{\alpha\beta} = P_\alpha \wedge P_\beta$. Implement the Gaussian elimination step to find vectors $y$ and $z$ satisfying the FastTODD conditions (Theorem 6).
* **Task 4.4 (Substitution & Residual Fold-Forward):** Using Vandaele's objective function (Eq 35/85), select vector $z$ to maximize duplicate/null column creation. Execute the replacement $P' = P \oplus z y^T$.
* Write $P'$ back to the HIR: All-zero columns are deleted (Identity).
* Duplicated columns ($P'_{:,i} == P'_{:,j}$) form an $S$ gate; delete the duplicates and **immediately** call `apply_clifford_frame_shift` on the remainder of the IR.
* Unique columns remain as `T_GATE`s.


* **DoD:** A Catch2 test feeds an unoptimized CCZ decomposition (7 T-gates) into the pass. FastTODD successfully extracts the linear dependencies, reduces it to the optimal 4 T-count, and folds the residuals.

## Phase 5: OpenQASM Interop, Equivalence, & Live Benchmarking

**Goal:** Prove state-of-the-art T-count reduction, exact equivalence, and wall-clock execution time against PyZX using the `op-T-mize` dataset.

* **Task 5.1 (Dataset Translation):** Add `pennylane`, `qiskit`, and `pyzx` to the dev dependencies. Write a lightweight Python utility `tools/bench/qasm2stim.py` to parse OpenQASM circuits and translate them into UCC's `.stim` superset.
* **Task 5.2 (Equivalence Oracle):**
* *Small Circuits ($N \le 12$):* Use `ucc.get_statevector()` to prove the dense statevector of the original circuit exactly matches the optimized circuit (multiplied by `global_weight`).
* *Large Circuits:* Export the optimized `StateAgnosticHir` back to QASM using `.to_qasm()`. Use `pyzx.verify_equality(orig_qasm, optimized_qasm)` to mathematically prove the UCC AOT engine preserved exact unitary logic without OOMing.


* **Task 5.3 (Live Benchmark Script):** Write `tools/bench/test_optmize_bench.py` utilizing `pytest-benchmark`.
* Use `qml.data.load` to fetch `op-T-mize` circuits: `mod5_4`, `hwb6`, `barenco_tof10`, `gf2^5_mult`, `mod_adder_1024`.
* Baseline: Run `pyzx.optimize.t_optimize()`. Record resulting T-count and wall-clock time.
* UCC: Run `ucc.trace_state_agnostic()` followed by `PassManager([Peephole, FastTodd]).run(hir)`. Record resulting T-count and wall-clock time.


* **DoD:** `just bench` executes the suite. PyZX verifies UCC's topological mask manipulations are 100% physically equivalent. The benchmark prints a table showing UCC matches or beats the FastTODD/PyZX T-counts in strictly less wall-clock time.

## Phase 6: Robust FTQC-Safety Integration Testing

**Goal:** Guarantee that aggressive global optimization does not destroy QEC physical error models.

* **Task 6.1 (Noise Barrier Fuzzer):** Create `tests/python/test_optimizer_safety.py`. Construct a fault-tolerant magic state distillation sub-circuit containing dense `T` gates, mid-circuit `M` collapses, `DEPOLARIZE1(0.01)` channels, and `DETECTOR` annotations.
* **Task 6.2 (Statistical Invariance):** Compile the circuit *without* the optimizer and sample 50,000 shots in the SVM to establish baseline detector marginal probabilities.
* **Task 6.3 (Optimization Validation):** Compile the exact same circuit *with* `PassManager([Peephole, FastTodd])`. Assert that the T-count dropped significantly. Run 50,000 shots of the *optimized* program.
* **DoD:** Assert that the SVM's detector marginals match the unoptimized baseline perfectly (within a $5\sigma$ binomial tolerance), proving the compiler successfully blocked illegal commutations and is strictly FTQC-safe.

---

## Appendix: Python API Workflows (Dual-Mode)

These snippets demonstrate how users will interact with the Dual-Mode architecture after this plan is implemented.

### **Workflow A: Executable Simulation (State-Aware)**

End-to-end exact Monte Carlo simulation using the optimized operations.

```python
import ucc
from ucc.passes import PassManager, PeepholeFusionPass, FastToddPass

circuit = ucc.parse("H 0\nT 0\nCX 0 1\nT 1\nM 0")

# 1. Trace from |0> vacuum (computes precise AG pivots for measurements)
hir: ucc.StateAwareHir = ucc.trace_state_aware(circuit)

# 2. Configure optimizer pipeline
pm = PassManager()
pm.add(PeepholeFusionPass())
pm.add(FastToddPass())
pm.run(hir)  # Optimizes the inner HirCore in-place

# 3. Lower to VM bytecode (Strictly requires StateAwareHir)
program = ucc.lower(hir)

# 4. Execute
measurements, detectors, observables = ucc.sample(program, shots=10_000)

```

### **Workflow B: Hardware Template Export (State-Agnostic)**

Optimization for physical QPU routing without tying it to an input state.

```python
import ucc
import pytest
from ucc.passes import PassManager, PeepholeFusionPass, FastToddPass

# A QPU routing subroutine (no known initial state)
subroutine = ucc.parse("CX 0 1\nT 1\nM 1\nCX rec[-1] 0")

# 1. Trace from Identity Tableau (disables AG pivots)
agnostic_hir: ucc.StateAgnosticHir = ucc.trace_state_agnostic(subroutine)

# 2. Optimize (Optimizer works exactly the same!)
pm = PassManager()
pm.add(PeepholeFusionPass())
pm.add(FastToddPass())
pm.run(agnostic_hir)

# 3. Type-Safe Rejection & Export
with pytest.raises(TypeError):
    ucc.lower(agnostic_hir)  # nanobind boundary blocks this! Cannot execute untracked states.

# Export the optimized phase polynomial for PBC routing / lattice surgery
qasm_str = agnostic_hir.to_qasm() # or .export_json("qpu_template.json")

```
