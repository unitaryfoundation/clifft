# **UCC Implementation Plan: Dual-Mode Compilation & Pass Management**

**TODO** -- Have an LLM review this against state of code before starting. And this should probably be a step that is part of the passmanger design work and initial pass implementations.

## **Executive Summary & Constraints**

You are implementing the **State-Aware vs. State-Agnostic Compilation Pipeline** for the Unitary Compiler Collection (UCC).

This update formalizes a core architectural duality: **Execution requires a known input state, but Optimization does not.** You will replace the monolithic HIR with strict, composition-based data structures. This enables an LLVM-style PassManager to apply $\\mathcal{O}(1)$ optimizations identically to both executable simulation payloads and state-agnostic hardware templates, while using the C++ and Python type systems to make it physically impossible to execute an untrackable state.

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

* **Task 1.3 (Frame Shift Op):** In HeisenbergOp, add FRAME\_SHIFT to the OpType enum. Add a static factory make\_frame\_shift().

## **Part 2: Dual-Mode Tracing (The Front-End)**

* **Task 2.1 (State-Aware Trace):** Rename ucc::trace() to StateAwareHir trace\_state\_aware(const Circuit& circuit). It continues to initialize at $|0\\rangle^{\\otimes n}$, evaluate commuting measurements, and compute AG pivots exactly as before.
* **Task 2.2 (State-Agnostic Trace):** Implement StateAgnosticHir trace\_state\_agnostic(const Circuit& circuit).
  * Implement void run(StateAgnosticHir& hir) (calls pass-\>run(hir.core)).

## **Part 4: Python API & Bindings (nanobind)**

* **Task 4.1 (IR Bindings):** In bindings.cc, bind StateAwareHir and StateAgnosticHir as distinct Python classes. Do *not* expose HirCore to Python directly to enforce the API boundary. Expose .num\_ops via lambda accessors.
* **Task 4.2 (Trace Bindings):** Bind trace\_state\_aware and trace\_state\_agnostic.
```python
import ucc
from ucc.passes import PassManager, PeepholeFusion

circuit = ucc.parse("H 0\nT 0\nCX 0 1\nM 0")

# 1. Trace from vacuum (computes AG pivots)
hir: ucc.StateAwareHir = ucc.trace_state_aware(circuit)

# 2. Configure optimizer pipeline
pm = PassManager()
pm.add(PeepholeFusion())
pm.run(hir)  # Optimizes the inner HirCore in-place

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

## ---

**Appendix: Demonstration Python Workflows to Verify**

Once implementation is complete, write Python tests for these exact scenarios to verify the system behavior satisfies the API design:

### **Workflow A: Executable Simulation (State-Aware)**

Python

import ucc
from ucc.passes import PassManager, PeepholeFusion

circuit \= ucc.parse("H 0\\nT 0\\nCX 0 1\\nM 0")

\# 1\. Trace from vacuum (computes AG pivots)
hir: ucc.StateAwareHir \= ucc.trace\_state\_aware(circuit)

\# 2\. Configure optimizer pipeline
pm \= PassManager()
pm.add(PeepholeFusion())
pm.run(hir) \# Optimizes the inner HirCore in-place

\# 3\. Lower to VM bytecode (Strictly requires StateAwareHir)
program \= ucc.lower(hir)
results \= ucc.sample(program, shots=10\_000)

### **Workflow B: Hardware Template Export (State-Agnostic)**

Python

import ucc
import pytest
from ucc.passes import PassManager, PeepholeFusion

subroutine \= ucc.parse("CX 0 1\\nT 1\\nM 1\\nCX rec\[-1\] 0")

\# 1\. Trace from Identity Tableau (disables AG pivots)
agnostic\_hir: ucc.StateAgnosticHir \= ucc.trace\_state\_agnostic(subroutine)

\# 2\. Optimize (Optimizer works exactly the same\!)
pm \= PassManager()
pm.add(PeepholeFusion())
pm.run(agnostic\_hir)

\# 3\. Type-Safe Rejection & Export
with pytest.raises(TypeError):
    ucc.lower(agnostic\_hir)  \# nanobind boundary blocks this\!

agnostic\_hir.export\_json("qpu\_template.json")
