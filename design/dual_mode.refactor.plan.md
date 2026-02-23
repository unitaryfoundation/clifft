# **UCC Implementation Plan: Dual-Mode Compilation & Pass Management**

## **Executive Summary & Constraints**

You are implementing the **State-Aware vs. State-Agnostic Compilation Pipeline** for the Unitary Compiler Collection (UCC).

This update formalizes a core architectural duality: **Execution requires a known input state, but Optimization does not.** You will replace the monolithic HIR with strict, composition-based data structures. This enables an LLVM-style PassManager to apply $\mathcal{O}(1)$ optimizations identically to both executable simulation payloads and state-agnostic hardware templates, while using the C++ and Python type systems to make it physically impossible to execute an untrackable state.

### **Architectural Prerequisites**

This plan relies on the core AOT architecture. Do not begin this plan until:

1. **The Circuit AST & Parser:** Can parse .stim superset syntax.
2. **The Front-End:** Is currently emitting a monolithic HirModule and correctly driving the stim::TableauSimulator.
3. **The Back-End & SVM:** Are successfully lowering and executing the monolithic HirModule.

### **Strict Constraints (Overrides general guidelines if in conflict)**

1. **Composition Over Inheritance:** Do NOT use C++ virtual classes or inheritance for the HIR types. Use strict struct composition (StateAwareHir contains a HirCore).
2. **Stateless Measurement Prohibition:** If trace\_state\_agnostic encounters a MEASURE node, it MUST NOT check for determinism or compute an Aaronson-Gottesman (AG) pivot. It must set ag\_matrix\_idx \= AgMatrixIdx::None.
3. **Type-Safe Lowering:** The Back-End lower() function must strictly accept only const StateAwareHir&. Attempting to pass a StateAgnosticHir to the execution engine must be caught as a static type error at compile time (C++) or binding time (nanobind), not via a runtime check.
4. **Preserve the 32-Byte Invariant:** The size of HeisenbergOp must remain exactly 32 bytes.

## ---

**Part 1: Type System & Composition Refactor (C++)**

**Goal:** Replace the monolithic HirModule with a composed, type-safe architecture.

* **Task 1.1 (Core Data):** In src/ucc/frontend/hir.h, rename the existing HirModule to HirCore. Remove ag\_matrices and final\_tableau from it entirely. It should only contain std::vector\<HeisenbergOp\> ops, num\_qubits, num\_measurements, and global\_weight.
* **Task 1.2 (Type Wrappers):** Define struct StateAwareHir containing a HirCore core, std::vector\<stim::Tableau\<kStimWidth\>\> ag\_matrices, and std::optional\<stim::Tableau\<kStimWidth\>\> final\_tableau. Define struct StateAgnosticHir containing only a HirCore core.
* **Task 1.3 (Frame Shift Op):** In HeisenbergOp, add FRAME\_SHIFT to the OpType enum. Add a static factory make\_frame\_shift().
* **Task 1.4 (Back-End Signature):** Update the Back-End lowering function signature to strictly require the state-aware context: Program lower(const StateAwareHir& hir). Update its internals to read ops from hir.core.ops.
* **Definition of Done (DoD):** The C++ code compiles natively. sizeof(HeisenbergOp) remains exactly 32 bytes.

## **Part 2: Dual-Mode Tracing (The Front-End)**

**Goal:** Branch the tracing logic to handle state-aware vs. state-agnostic input states.

* **Task 2.1 (State-Aware Trace):** Rename ucc::trace() to StateAwareHir trace\_state\_aware(const Circuit& circuit). It continues to initialize at $|0\\rangle^{\\otimes n}$, evaluate commuting measurements, and compute AG pivots exactly as before.
* **Task 2.2 (State-Agnostic Trace):** Implement StateAgnosticHir trace\_state\_agnostic(const Circuit& circuit).
  * Initialize stim::TableauSimulator with an Identity Tableau.
  * When it hits a MEASURE node, it **skips AG pivot computation entirely**. It rewinds the observable, emits a MEASURE op with ag\_matrix\_idx \= None, and continues, leaving the unitary frame completely undisturbed.
* **Task 2.3 (Checkpoints):** Update ucc::parser.cc to recognize a UCC\_CHECKPOINT string. Update trace\_state\_aware so that when it hits a checkpoint, it resets sim.inv\_state back to Identity, but *keeps* its knowledge of the pure stabilizer generators for future measurements. It emits HeisenbergOp::make\_frame\_shift().
* **DoD:** Catch2 tests verify trace\_state\_agnostic produces zero AG matrices for anti-commuting measurements, while trace\_state\_aware computes them accurately.

## **Part 3: The Modular Pass Manager**

**Goal:** Implement a configurable optimization pipeline that operates safely on HirCore.

* **Task 3.1 (Pass Interface):** Create src/ucc/optimizer/pass.h. Define a pure abstract base class class Pass { public: virtual void run(HirCore& core) \= 0; virtual \~Pass() \= default; };.
* **Task 3.2 (PassManager):** Define class PassManager { std::vector\<std::unique\_ptr\<Pass\>\> passes; }.
  * Implement composition overloads: void run(StateAwareHir& hir) (calls pass-\>run(hir.core)).
  * Implement void run(StateAgnosticHir& hir) (calls pass-\>run(hir.core)).
* **Task 3.3 (Peephole Migration):** Migrate existing $\\mathcal{O}(1)$ fusion logic into a class PeepholeFusion : public Pass. Ensure it treats OpType::FRAME\_SHIFT as an impenetrable barrier (no commutations allowed past it).
* **DoD:** A Catch2 test applies PeepholeFusion to a StateAgnosticHir successfully. Another test proves $T$ and $T^\dagger$ separated by a FRAME\_SHIFT do NOT cancel each other.

## **Part 4: Python API & Bindings (nanobind)**

**Goal:** Expose the strict types and pass manager to Python users.

* **Task 4.1 (IR Bindings):** In bindings.cc, bind StateAwareHir and StateAgnosticHir as distinct Python classes. Do *not* expose HirCore to Python directly to enforce the API boundary. Expose .num\_ops via lambda accessors.
* **Task 4.2 (Trace Bindings):** Bind trace\_state\_aware and trace\_state\_agnostic.
* **Task 4.3 (PassManager Bindings):** Bind the PassManager class and its run overloads. Create a submodule ucc.passes and bind PeepholeFusion.
* **Task 4.4 (Lowering Protection):** Bind ucc.lower strictly to StateAwareHir.
* **Task 4.5 (JSON Export):** Add an .export\_json(filename) method to StateAgnosticHir to dump the ops for physical hardware routing.
* **DoD:** Python pytest suite passes. A test explicitly confirms that calling ucc.lower(agnostic\_hir) raises a Python TypeError directly from nanobind before any C++ logic executes.

## ---

**Appendix: Demonstration Python Workflows to Verify**

Once implementation is complete, write Python tests for these exact scenarios to verify the system behavior satisfies the API design:

### **Workflow A: Executable Simulation (State-Aware)**

```python

import ucc
from ucc.passes import PassManager, PeepholeFusion

circuit = ucc.parse("H 0\nT 0\nCX 0 1\nM 0")

# 1. Trace from vacuum (computes AG pivots)
hir: ucc.StateAwareHir = ucc.trace_state_aware(circuit)

# 2. Configure optimizer pipeline
pm = PassManager()
pm.add(PeepholeFusion())
pm.run(hir) # Optimizes the inner HirCore in-place

# 3. Lower to VM bytecode (Strictly requires StateAwareHir)
program = ucc.lower(hir)
results = ucc.sample(program, shots=10_000)
```
### **Workflow B: Hardware Template Export (State-Agnostic)**


```python
import ucc
import pytest
from ucc.passes import PassManager, PeepholeFusion

subroutine = ucc.parse("CX 0 1\nT 1\nM 1\nCX rec[-1] 0")

# 1. Trace from Identity Tableau (disables AG pivots)
agnostic_hir: ucc.StateAgnosticHir = ucc.trace_state_agnostic(subroutine)

# 2. Optimize (Optimizer works exactly the same!)
pm = PassManager()
pm.add(PeepholeFusion())
pm.run(agnostic_hir)

# 3. Type-Safe Rejection & Export
with pytest.raises(TypeError):
    ucc.lower(agnostic_hir)  # nanobind boundary blocks this!

agnostic_hir.export_json("qpu_template.json")
```
