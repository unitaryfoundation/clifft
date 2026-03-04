# **UCC Implementation Plan: Phase 3 (Coherent Noise & Showstoppers)**

## **Executive Summary & Constraints**

This phase upgrades UCC to natively support arbitrary 2-term coherent Pauli rotations. By implementing the generic Linear Combination of Unitaries (LCU) opcodes and upgrading the Middle-End Optimizer to perform algebraic fusion, UCC will exactly simulate coherent interference on scales that crash both State-Vector and Tensor Network simulators.

**Strict Constraints for Phase 3:**

1. **The 32-Byte Invariant:** The HeisenbergOp union already accommodates a weight_re and weight_im payload. However, the 32-byte VM Instruction struct does not have room for raw floats. You MUST extract these weights into a ConstantPool::lcu_pool array, and the bytecode must only store a 32-bit lcu_pool_idx.
2. **Dominant Term Factoring:** The AOT Front-End MUST strictly factor out the Identity term. For $e^{-i\theta P/2}$, it must extract the global scalar $cos(\theta/2)$ into hir.global_weight, leaving the spawned Pauli branch with the relative weight $-i\tan(\theta/2)$.
3. **No External Dependencies in Core:** The benchmark scripts will use qiskit and stim for baseline plotting, but ucc_core must not depend on them to compile.

### ---

**Phase 3.1: Parser Enhancements (Coherent Syntax)**

**Goal:** Extend the Stim-superset parser to accept parameterized coherent 2-qubit rotations.

* **Task 3.1.1 (Gate Definitions):** Update gate_data.h and the parser to recognize a new custom gate: RZZ (Coherent ZZ rotation).
* **Task 3.1.2 (Parsing Logic):** Update the parser loop to accept RZZ(theta) q1 q2. Extract the float theta (in radians) into AstNode::arg and the targets into the targets vector.
* **DoD:** A Catch2 test parses RZZ(0.01) 0 1 successfully, validating the AST contains the correct angle argument and targets.

### **Phase 3.2: Front-End LCU Emission (Dominant Term Math)**

**Goal:** Translate the RZZ AST node into the Heisenberg IR using Dominant Term Factoring algebra: $e^{-i\theta ZZ/2} = \cos(\theta/2) [I - i\tan(\theta/2) ZZ]$.

* **Task 3.2.1 (Heisenberg Rewinding):** When encountering RZZ(theta) in trace():
  * Rewind the $Z_{q_1} \otimes Z_{q_2}$ operator through the current inv_state tableau (by multiplying sim.inv_state.zs[q1] and zs[q2]) to extract its $t=0$ destab_mask and stab_mask.
* **Task 3.2.2 (Factoring & Weight Accumulation):**
  * Multiply the HirModule::global_weight accumulator by $\cos(\theta/2)$.
  * Calculate the relative spawned weight: $c = -i \tan(\theta/2)$.
* **Task 3.2.3 (HIR Emission):** Emit a HeisenbergOp::GATE node. Populate the union payload with gate_.weight_re = 0.0 and gate_.weight_im = -std::tan(theta/2).
* **DoD:** A test feeding RZZ(0.5) 0 1 asserts that global_weight is multiplied by $\cos(0.25)$, and the HIR contains a GATE node with the exact mathematically expected imaginary weight.

### **Phase 3.3: Middle-End Optimizer LCU Fusion**

**Goal:** Ensure the optimizer algebraically fuses generic GATE nodes to prevent memory blowup on deep calibration sequences (crucial for Demo 2).

* **Task 3.3.1 (Fusion Math):** In the PeepholeFusionPass, if two GATE nodes share the exact same destab_mask and stab_mask (and no barriers block them), compute the combined weight using the algebra: $c_{\text{fused}} = \frac{c_1 + c_2}{1 + c_1 c_2}$.
* **Task 3.3.2 (Scalar Extraction):** Multiply the scalar factor $(1 + c_1 c_2)$ into hir.global_weight.
* **Task 3.3.3 (Node Replacement):** Replace the first node with a GATE containing $c_{\text{fused}}$ and delete the second. (If $c_{\text{fused}}$ is sufficiently close to 0, delete both).
* **DoD:** A Catch2 test applies 100 sequential RZZ(0.01) 0 1 gates. The optimizer must reduce this to exactly **one** GATE node with the analytically combined weight, executing in $< 1$ ms.

### **Phase 3.4: Back-End Lowering & SVM Execution**

**Goal:** Lower generic GATE nodes into the Constant Pool and execute the LCU opcodes in the VM.

* **Task 3.4.1 (Constant Pool Population):** Define struct LCUData { double weight_re; double weight_im; };. Add std::vector<LCUData> lcu_pool; to the ConstantPool. In the Back-End, when processing a GATE node, push its weights into the pool and get the lcu_pool_idx.
* **Task 3.4.2 (Bytecode Emission):** Add OP_BRANCH_LCU, OP_COLLIDE_LCU, and OP_SCALAR_PHASE_LCU to the Opcode enum. Evaluate the spatial shift $\beta$ against the active GF(2) basis $V$ (just like T_GATE). Emit the corresponding opcode, assigning lcu_pool_idx into the instruction payload.
* **Task 3.4.3 (SVM Logic):** Implement the three _LCU opcodes in the VM switch statement. The logic is identical to their $T$-gate counterparts (OP_BRANCH, etc.), except they multiply by the complex weight from constant_pool.lcu_pool[inst.lcu.lcu_pool_idx].
* **DoD:** A Python test runs a 4-qubit circuit with Cliffords + RZZ and perfectly matches the amplitudes of a numpy/Qiskit state-vector oracle.

### ---

**Phase 3.5: Demo 1 — The 60-Qubit Scale Showstopper (GHZ + Crosstalk)**

**Goal:** Build the Python script proving UCC scales to memory sizes that crash State-Vector, and entanglement sizes that crash MPS.

* **Task 3.5.1:** Write tools/demos/demo_ghz_crosstalk.py.
* **Task 3.5.2:** Generate a 60-qubit circuit: Prepare a 60-qubit GHZ state, apply a single RZZ(theta) error in the middle, and measure a multi-qubit parity check.
* **Task 3.5.3:** Create a Stim PTA equivalent by replacing RZZ(theta) with a probabilistic Pauli channel E(p) Z q1 Z q2 where $p = \sin^2(\theta/2)$.
* **Task 3.5.4:** Plot small scale (10 qubits) sweeping $\theta$: Show UCC matching Qiskit Statevector (oscillations), while Stim decays flatly.
* **Task 3.5.5:** Run large scale (60 qubits) at $\theta=0.5$ in UCC. Print the execution time ($<50$ ms) and the array size (peak_rank=1, 2 complex numbers). Print a warning that Qiskit is bypassed because it would require 16 Exabytes of RAM.

### **Phase 3.6: Demo 2 — The Infinite Depth Optimizer**

**Goal:** Build the Python script demonstrating that UCC's Middle-End $\mathcal{O}(1)$ fusion produces depth-independent execution times for hardware characterization.

* **Task 3.6.1:** Write tools/demos/demo_infinite_depth.py.
* **Task 3.6.2:** Generate a circuit with an initial $H$ gate, followed by a loop of $N$ repetitions of [CZ 0 1, RZZ(0.01) 0 1], ending with a measurement.
* **Task 3.6.3:** Sweep $N \in [10, 100, 1000, 5000]$. Measure compilation + execution time in UCC.
* **Task 3.6.4:** *Output:* A plot showing the wall-clock execution time of UCC remaining completely flat at $\mathcal{O}(1)$ regardless of $N$. This proves the Middle-End fused the deep coherent accumulation into a single generic instruction before VM execution, bypassing the exponential tracking that destroys other simulators.

### **Phase 3.7: Demo 3 — The Mini-QEC Surface Code Patch**

**Goal:** Prove that coherent noise in a real QEC cycle fundamentally breaks the Pauli-Twirling approximation, proving UCC's utility to the fault-tolerant community.

* **Task 3.7.1:** Write tools/demos/demo_qec_coherent.py.
* **Task 3.7.2:** Use a small 9-qubit rotated surface code patch (or a 5-qubit repetition code). Run 1 round of syndrome extraction.
* **Task 3.7.3:** Inject RZZ(theta) on all data-ancilla links.
* **Task 3.7.4:** Extract the detector event fractions for both UCC (exact coherence) and Stim (PTA).
* **Task 3.7.5:** Plot the difference between the exact coherent detector fraction and the twirled detector fraction as $\theta$ increases, showing that PTA inherently underestimates the logical error footprint.
