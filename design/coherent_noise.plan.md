# UCC Implementation Plan: Coherent Noise & Arbitrary Rotations

## Executive Summary & Constraints

This phase upgrades UCC to natively support arbitrary continuous non-Clifford rotations ($e^{-i\theta P}$). The payload (continuous weights) is stored **inline** directly inside the 24-byte payload union of the 32-byte `Instruction` struct. The VM executes these rotations via a localized `OP_PHASE_LCU(v)` instruction.

**Strict Constraints:**
1. **Inline Math Payloads:** Do NOT extract floats into a Constant Pool. Store `double weight_re; double weight_im;` directly in the `Instruction` union.
2. **Dominant Term Factoring:** The Front-End MUST strictly factor out the Identity term offline to save runtime FLOPs. $e^{-i\theta P} = \cos(\theta) [I - i\tan(\theta) P]$. The global scalar $\cos(\theta)$ is accumulated offline.
3. **RISC Localization:** Multi-qubit LCUs must be compressed AOT by the Back-End. The VM only executes LCU phases on single, isolated array dimensions.

---

## Phase 1: Parser & Front-End Emission

**Goal:** Parse arbitrary continuous rotations and emit them into the HIR with Dominant Term Factoring.

*   **Task 1.1 (Syntax):** Update `gate_data.h` and the parser to recognize gates with continuous arguments (e.g., `RZZ(theta) q1 q2`).
*   **Task 1.2 (Heisenberg Rewinding):** When encountering the gate, rewind the $Z \otimes Z$ generator through the inverse tableau to extract its $t=0$ `destab_mask` and `stab_mask`.
*   **Task 1.3 (Dominant Factoring):** Multiply `HirModule::global_weight` by $\cos(\theta)$. Calculate the relative weight $c = -i \tan(\theta)$.
*   **Task 1.4 (HIR Emission):** Emit an `OpType::GATE` (generic LCU) node to the HIR. Populate its payload with the raw `weight_re` and `weight_im`.
*   **DoD:** The parser accepts `RZZ(0.5) 0 1`, factors out the scalar, and the HIR contains a `GATE` node with the correct Pauli masks and relative complex weight.

## Phase 2: Optimizer LCU Fusion

**Goal:** Ensure the Middle-End algebraically fuses generic `GATE` nodes targeting the same Pauli axis to prevent memory blowup on deep calibration sequences.

*   **Task 2.1 (Fusion Math):** In `PeepholeFusionPass`, if two `GATE` nodes share exact `destab`/`stab` masks, compute $c_{\text{fused}} = \frac{c_1 + c_2}{1 + c_1 c_2}$.
*   **Task 2.2 (Scalar Extraction):** Multiply the scalar $(1 + c_1 c_2)$ into `hir.global_weight`. Replace the first node with $c_{\text{fused}}$ and delete the second.
*   **DoD:** A test feeding 100 sequential `RZZ(0.01)` gates reduces to exactly ONE `GATE` node natively in the HIR.

## Phase 3: Back-End Compression & RISC Emission

**Goal:** Geometrically compress the LCU operation and emit localized RISC opcodes.

*   **Task 3.1 (Opcode Definition):** In `backend.h`, define `OP_PHASE_LCU`. Ensure the `Instruction` payload union has a `math` struct with `double weight_re, weight_im`.
*   **Task 3.2 (Compression Lowering):** In `backend.cc` `lower()`, when processing a `GATE` node:
    1.  Map the node's $t=0$ mask to the current virtual frame: $P_v = V_{cum} P_{t=0} V_{cum}^\dagger$.
    2.  Run the greedy Pauli compressor algorithm to isolate $P_v$ onto a single virtual axis $v$.
    3.  If $v \in D$ (Dormant), emit `OP_EXPAND` to activate it.
    4.  Emit `OP_PHASE_LCU` targeting axis $v$, setting the inline payload to the HIR node's `weight_re` and `weight_im`.
*   **Task 3.3 (Localized VM Execution):** In `svm.cc`, implement the handler for `OP_PHASE_LCU`.
    *   Extract the target axis $v$ (`axis_1`).
    *   Query the Pauli frame tracking bit (`p_x[v]`) to determine if the physical rotation must be algebraically inverted.
    *   Iterate over the active array `v[]`. Apply the complex weight only to the elements where the target axis bit is set (the $|1\rangle_v$ branch).
*   **DoD:** A Python Catch2 test simulates a 4-qubit circuit with Cliffords + `RZZ(0.1)` and perfectly matches the amplitudes of a Qiskit statevector oracle.

## Phase 4: Demonstrations

**Goal:** Prove UCC simulates coherent interference at scales that crash other simulators.

*   **Task 4.1 (Infinite Depth Demo):** Write `demo_infinite_depth.py`. Sweep a circuit consisting of $N \in [10, 1000, 5000]$ repetitions of `[CZ 0 1, RZZ(0.01) 0 1]`. Plot execution time to show it remains completely flat ($\mathcal{O}(1)$).
*   **Task 4.2 (Circuit-Level Coherence vs. PTA Benchmark):** Run a distance-3 rotated surface code patch to prove UCC captures the circuit-level phase accumulations that standard Pauli Twirling Approximations (PTA) fundamentally miss.
    *   **The Theoretical Gap:** Early literature (Bravyi et al. 2018, Beale et al. 2018) established that coherent errors scale worse than stochastic errors at low distances, but concluded that "QEC decoheres noise," validating PTA for surface codes under simplified code-capacity models. However, recent work (*Zhou et al. 2025, Sec II.A*) proves that under realistic **circuit-level noise**, coherent $ZZ(\theta)$ crosstalk during syndrome extraction causes phase interference that PTA mathematically cannot replicate.
    *   **Circuit Construction:** Build a d=3 rotated surface code memory circuit. Inject coherent $R_{ZZ}(\theta)$ gate-based crosstalk immediately following every data-ancilla entangling gate during the syndrome cycles.
    *   **The Baseline (Stim PTA):** Compile a control circuit for Stim, twirling the coherent rotations into stochastic $Z \otimes Z$ errors with probability $p = \sin^2(\theta/2)$. Sample using Stim's detector sampler.
    *   **The Exact Execution (UCC):** Compile the original coherent circuit natively in UCC. The Front-End extracts the massive Identity branches offline, leaving the VM to calculate the exact $-i \tan(\theta/2) Z \otimes Z$ coherent interference over the dense array using the `OP_PHASE_LCU` instruction.
    *   **Metrics & Evaluation:**
        1.  **Detector Fraction Discrepancy:** Plot the exact (UCC) vs. twirled (Stim) detector firing rates. Prove that circuit-level coherent phases alter the physical ancilla outcomes in ways the classical PTA mixture averages out.
        2.  **Logical Failure Rate ($p_L$):** Decode both datasets with PyMatching. Demonstrate that PTA systematically underestimates the logical error rate when coherent noise occurs inside the extraction circuits, highlighting the need for exact simulation.
