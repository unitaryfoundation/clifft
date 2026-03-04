
# UCC — Use Cases & Application Workflows

This document outlines the core use cases enabled by the Unitary Compiler Collection (UCC) framework. Because UCC fundamentally decouples the $\mathcal{O}(n^2)$ topological tracking of a quantum circuit from its $\mathcal{O}(1)$ probabilistic execution via a multi-level Ahead-Of-Time (AOT) compiler, it exposes multiple layers of program representation. This allows UCC to function simultaneously as an ultra-fast simulator, an algebraic optimizer, and a hardware routing compiler.

For each use case, we define the integration point within the pipeline, the minimum functionality required for implementation, relevant references, and our strategy for demonstrating state-of-the-art performance.

---

## 1. Ground-Truth Simulation of Magic State Cultivation (MSC)

### Overview

Generating high-fidelity logical $|T\rangle$ states is a massive bottleneck in fault-tolerant quantum computing. Evaluating the exact non-Clifford cultivation circuit at scale (e.g., code distance $d=5$, involving 42 qubits and 72 $T/T^\dagger$ gates) has historically been computationally intractable.

- **Program Representation Level:** Level 4 (VM Runtime). This exercises the full AOT pipeline, compiling the circuit down to hardware-aligned bytecode and sampling it over millions of shots.
- **Minimum Functionality Needed:**
  - *Phase 1 (MVP) Fast-Path:* Because the $d=5$ MSC circuit uses exactly 42 qubits, it fits entirely within the MVP's $\le 64$ qubit inline uint64\_t execution path.
  - *Phase 2 (Control Flow & Noise):* OP\_POSTSELECT is strictly required. The $d=5$ circuit has a $>99\%$ discard rate under noise. Aborting the SVM loop the moment a parity check fails—rather than evaluating the remaining non-Clifford branches—is essential for tracking scale. Geometric gap sampling (NoiseSchedule) is also required to implement the uniform depolarizing models used in the papers.
- **References:**
  - Gidney, Shutty, Jones (2024): "Magic state cultivation: growing T states as cheap as CNOT gates"
  - Li et al. (2025): "SOFT: a high-performance simulator for universal fault-tolerant quantum circuits"

#### Demonstration & Evaluation Strategy

We will use a two-step benchmarking strategy to prove both correctness and absolute performance:

1. **The S-Gate Baseline (vs. Stim / Gidney):** Gidney evaluated $d=5$ cultivation by replacing $T$ gates with $S$ gates to make it simulatable in Stim, and extrapolated the results. We will simulate this exact $S$-gate proxy circuit through both Stim and UCC.
   - *Correctness:* Because $S$-gates are Cliffords, UCC's Front-End will absorb them entirely. If our logical error rates and discard rates perfectly match Stim, it mathematically proves our trace generation, Heisenberg rewinding, and gap-sampled noise models are flawless before introducing $T$-gate complexity.
   - *Speed Benchmark:* Since the AOT compiler eliminates all tableau math, the SVM simply executes fast integer XORs for noise and measurements. This will benchmark our SVM's raw execution speed against stim::TableauSimulator's dynamic tracking.
2. **The True T-Gate Benchmark (vs. SOFT):** The SOFT simulator recently achieved the first exact simulation of the $d=5$ $T$-gate circuit, but it required 16 NVIDIA H800 GPUs running for 20 days to collect 200 billion shots. We will run the true $T$-gate circuit end-to-end. By leveraging *Dominant Term Factoring* (where the heaviest identity branch requires zero FLOPs and zero memory writes) and OP\_POSTSELECT, we aim to show that UCC running on a conventional CPU cluster can achieve dramatically higher throughput than SOFT's GPU brute-force approach.

---

## 2. Best-in-Class Noise-Aware T-Count Reduction ("Heisenberg-TOHPE")

### Overview

Minimizing non-Clifford gate counts (such as the T-count) is a primary objective in quantum compilation. The 2025 state-of-the-art parity-table algorithms (such as the TOHPE and FastTODD procedures) achieve mathematically optimal reductions by mapping subcircuits to phase polynomials and solving the Third-Order Symmetric Tensor Rank (3-STR) problem.

However, these traditional solvers suffer from two fatal structural bottlenecks:

1. **The Ancilla & Partitioning Penalty:** To gain global visibility on mixed-Clifford circuits, they force "Hadamard Gadgetization," massively bloating the physical ancilla qubit count. Without gadgets, compilers must rely on bulky graph-partitioning heuristics to chop the circuit into Hadamard-free zones.
2. **The Extraction / Synthesis Penalty:** Converting an optimized phase polynomial back into a physical, executable unitary circuit requires synthesizing complex CNOT networks. This destroys hardware routing constraints and inflates circuit depth.

**UCC natively bypasses both bottlenecks.** Because the Front-End's TableauSimulator mathematically *absorbs* all Clifford gates and rewinds everything to the $t=0$ topological basis, the resulting Heisenberg IR (HIR) is natively a global phase polynomial. By embedding the TOHPE algorithm directly into the UCC Middle-End, we can optimize the HIR and export it directly to Pauli-Based Computation (PBC) hardware routers like tqec—or execute it directly in the SVM. **We never have to extract a CNOT unitary circuit**.

- **Program Representation Level:** Level 2 (Optimized Heisenberg IR).
- **Minimum Functionality Needed:** Phase 1 (MVP). Requires Front-End HIR emission and the Middle-End applying the TOHPE GF(2) null-space extraction over the pairwise ANDs of the topological masks within commuting cliques.
- **References:** Vandaele (2025): "Lower T-count with faster algorithms" (arXiv:2407.08695v2); op-T-mize benchmark suite.

### The Algorithm: L1-Resident Heisenberg-TOHPE

Because T-gates generate phase polynomials over $\mathbb{Z}_8$, linear dependencies must be found in the degree-3 Reed-Muller code space. UCC implements the TOHPE algorithm natively using CPU bitwise arithmetic:

1. **Native Slicing via Symplectic Physics:** The Middle-End sweeps the HIR sequentially, grouping T_GATE and T_DAG_GATE nodes into "cliques" using the $\mathcal{O}(1)$ popcount symplectic inner product. Unlike competitors that use manual partitioning algorithms, UCC naturally halts cliques when a rewound mask anti-commutes (e.g., a T-gate rewound through a Hadamard becomes an X-mask and naturally breaks the clique).
2. **Noise-Aware Barrier Enforcement:** The sweep halts strictly at any MEASURE or NOISE node (including coherent noise LCU branches) that anti-commutes with the active clique. This mathematically guarantees that the optimization never conjugates physical error models into new bases, making the compiler strictly FTQC-safe.
3. **Build the Pairwise $L$-Matrix:** For a clique of $m$ T-gates, the spatial stab_masks form a parity table $P$. Following TOHPE, the optimizer constructs the binary matrix $L$ where each row is the bitwise AND of two spatial dimensions: $L_{\alpha\beta} = P_\alpha \wedge P_\beta$.
4. **3-STR Null-Space Extraction:** The optimizer runs fast GF(2) Gaussian elimination to find a vector $y$ in the right null-space of $L$ ($Ly = 0$). This identifies a linear dependence that preserves the third-order tensor signature.
5. **Greedy Substitution & Clifford Fold-Back:** Using the extracted $y$ and a deterministically chosen vector $z$, the optimizer substitutes $P' = P \oplus zy^T$. Redundant T_GATE nodes are physically deleted from the HIR, and residual lower-order phases are converted to Clifford S gates, which are flagged to be mathematically absorbed by the Front-End during the final fold-back pass.

### Computational Scaling & The Arbitrary Qubit ($N$) Rescue

Vandaele reduced the classical complexity of TODD to $\mathcal{O}(n^2 m^3)$ for TOHPE. UCC fundamentally alters how this executes in memory, decoupling optimization time from both Clifford depth and physical qubit count.

- **Scaling with Clifford Gates ($G_C$): $\mathcal{O}(1)$**
  In UCC, the Front-End absorbs all $G_C$ Cliffords dynamically. Cliffords literally do not exist in the HIR, decoupling optimization time from Clifford depth entirely.
- **Scaling with Qubits ($N$): Bounded by $\mathcal{O}(m^2)$ (The Algebraic Compression)**
  Naively, the pairwise matrix $L$ has $\binom{N}{2}$ rows, which would cause a catastrophic memory explosion for large $N$. However, because UCC operates in GF(2), the right null-space of $L$ is a mathematical invariant under row-basis projection. Regardless of whether $N = 64$ or $N = 10,000$, the optimizer extracts the linearly independent row-basis of $P$ (which can never exceed $m$, the number of T-gates). The constructed $L_{compressed}$ matrix has at most $\binom{m}{2} + m$ rows.
- **Scaling with Non-Clifford Gates ($m$): Polynomial $\mathcal{O}(m^3)$**
  Finding the null-space of the bounded $L_{compressed}$ matrix takes strictly $\mathcal{O}(m^3)$ bitwise operations. Combined with the "Chunked Rewinding" extension that strictly limits lightcone smearing, the entire working matrix is permanently trapped at a microscopic size, fitting completely inside the CPU's L1 cache and executing in microseconds.

### Demonstration & Evaluation Strategy (The Dual-Pronged Benchmark)

We will process standard benchmark datasets (such as the 2025 Vandaele suites) through UCC and demonstrate state-of-the-art performance by validating three core claims for the MVP release:

1. **The Table 2 Benchmark (Generic, Ancilla-Free Circuits):** On generic, Hadamard-rich circuits, UCC natively matches the best-in-class ancilla-free T-counts. We will prove that while traditional compilers must run complex partitioning heuristics to isolate Hadamard-free zones and synthesize CNOT boundaries, UCC's AOT Front-End naturally isolates non-commuting cliques via pure symplectic geometry in **microseconds**.
2. **The Table 3 Benchmark (Massive Absolute Reductions):** On naturally Hadamard-free arithmetic circuits (e.g., Galois Field Multipliers, $h=0$), the entire circuit forms a massive commuting clique. We will prove that UCC matches the massive absolute T-count reductions of AlphaTensor / FastTODD, with **zero CNOT synthesis penalty** because we route directly to physical PBC instructions.
3. **FTQC-Safe Noise Preservation:** Purely logical synthesizers destroy physical error models by blindly commuting non-Clifford gates past noise. We will inject a parameterized error model (both stochastic Pauli and coherent over-rotations) into a magic-state distillation factory and mathematically prove via our statevector oracle that UCC uniquely optimizes T-counts while perfectly preserving the syndromic structure and physics of the QEC barriers.

---

## 3. End-to-End Hardware Execution Integrations

### Overview

While UCC acts as an ultra-fast simulator, its Ahead-Of-Time architecture uniquely positions it as a powerful compiler frontend for physical hardware. By mathematically flattening the circuit into an algebraically independent list of multi-qubit Pauli operations (the Optimized HIR), UCC breaks away from rigid DAG-based routing. Because UCC fundamentally flattens a quantum circuit into an algebraically independent list of multi-qubit Pauli rotations, translating this mathematically pure IR back into physical hardware instructions requires architectures that natively support high-weight Pauli parity operations or feature flexible connectivity.

- **Program Representation Level:** Level 2 (Optimized HIR Export).
- **Minimum Functionality Needed:** Phase 4. Export of the HirModule to JSON or text, capturing explicit Pauli masks, non-Clifford complex weights, and the global\_weight accumulator.

---

### 3.1 The NISQ Target: Trapped Ions (Quantinuum H2-2)

Trapped ions utilize a Quantum Charge-Coupled Device (QCCD) architecture with all-to-all connectivity via ion shuttling. They do not care about spatial locality. Furthermore, their native entangling gates (such as the Mølmer–Sørensen gate or parameterized $R_{ZZ}$) map naturally to multi-qubit Pauli exponentials ($e^{-i \theta (Z \otimes Z \dots)}$).

- **Hardware Profile:** Quantinuum H2-2 (56 qubits). This fits perfectly within the UCC MVP's $\le 64$ qubit fast-path, allowing us to leverage the absolute fastest compilation speeds.

#### The "Hero" Benchmark: 56-Qubit Heisenberg-Scrambled Phase Polynomials

To demonstrate UCC definitively outperforming standard industry compilers (like pytket or Qiskit), we target a dense, highly entangling 56-qubit algorithm—such as a Trotterized UCCSD chemistry ansatz or a complex Galois Field arithmetic circuit—interleaved with deep Clifford basis transformations.

- **The standard compiler's failure mode:** Standard compilers use DAG traversal and peephole optimization. They get trapped in local minima trying to commute parameterized rotations and $T$-gates through dense Clifford networks. They are forced to synthesize massive CNOT ladders and inject unnecessary routing layers, destroying 2Q-gate depth.
- **The UCC advantage:** The UCC Front-End mathematically *absorbs 100% of the Clifford routing* in microseconds, revealing the true global phase polynomial. The Middle-End (Heisenberg-TOHPE) globally cancels and fuses redundant Pauli rotations. We then export the Optimized HIR as a minimal list of pure Phase Gadgets. By leveraging the H2-2's all-to-all connectivity, the Quantinuum compiler folds these gadgets directly into native ZZPhase entangling pulses with zero SWAP overhead. A single synthesized final\_tableau block at the end corrects the measurement basis.

#### Implementation Blueprint (pytket $\to$ H2-2)

```python
from pytket import Circuit
import stim

def decode_masks(destab_mask: int, stab_mask: int, n: int) -> list[str]:
    """Convert UCC 64-bit GF(2) masks into Pauli characters per qubit."""
    paulis = []
    for i in range(n):
        x, z = (destab_mask >> i) & 1, (stab_mask >> i) & 1
        if x and z: paulis.append('Y')
        elif x: paulis.append('X')
        elif z: paulis.append('Z')
        else: paulis.append('I')
    return paulis

def compile_hir_to_ion_tket(hir_module) -> Circuit:
    """Compiles UCC Optimized HIR into a pytket circuit tailored for QCCD."""
    n = hir_module.num_qubits
    # +1 Ancilla needed ONLY for multi-Pauli parity measurements
    circ = Circuit(n + 1, hir_module.num_measurements)
    ancilla = n
    meas_idx = 0

    for op in hir_module.ops:
        p_str = decode_masks(op.destab_mask, op.stab_mask, n)
        involved = [q for q, p in enumerate(p_str) if p != 'I']
        if not involved: continue

        # 1. Basis change to Z-basis for all involved qubits
        for q in involved:
            if p_str[q] == 'X': circ.H(q)
            elif p_str[q] == 'Y': circ.Sdg(q); circ.H(q)

        if op.type.name in {"T_GATE", "T_DAG_GATE", "GATE"}:
            # 2. Phase Gadget (Pivot on a data qubit, no ancilla required)
            # T is a pi/4 rotation. pytket Rz is in half-turns.
            if op.type.name == "T_GATE": angle = 0.25
            elif op.type.name == "T_DAG_GATE": angle = -0.25
            else: angle = op.lcu_weight_angle # Conceptual for arbitrary LCU gates

            pivot = involved[-1]

            # The Quantinuum compiler seamlessly compresses this
            # CNOT ladder into native ZZPhase entangling pulses.
            for q in involved[:-1]: circ.CX(q, pivot)
            circ.Rz(angle, pivot)
            for q in reversed(involved[:-1]): circ.CX(q, pivot)

        elif op.type.name == "MEASURE":
            # 3. Measurement Gadget (Requires ancilla to preserve post-measurement state)
            if len(involved) == 1:
                circ.Measure(involved[0], meas_idx)
            else:
                for q in involved: circ.CX(q, ancilla)
                circ.Measure(ancilla, meas_idx)
                circ.Reset(ancilla) # Reset ancilla for reuse
            meas_idx += 1

        # 4. Undo basis changes to return to the active reference frame
        for q in reversed(involved):
            if p_str[q] == 'X': circ.H(q)
            elif p_str[q] == 'Y': circ.H(q); circ.S(q)

    # 5. CRITICAL: Final Frame Correction
    # Because UCC absorbed all Cliffords, the physical qubits are effectively
    # stuck in the t=0 basis. We synthesize the final tableau to map them to the correct output.
    if getattr(hir_module, 'final_tableau', None) is not None:
        final_circ = stim.Circuit.generated_by_clifford_tableau(hir_module.final_tableau)
        # Note: requires a minor utility to convert stim.Circuit -> pytket.Circuit
        # circ.append(convert_stim_to_tket(final_circ))

    return circ
```

---

### 3.2 The FTQC Target: Surface Codes (tqec)

Modern superconducting FTQC architectures execute logic via Pauli-Based Computation (PBC) using lattice surgery. A weight-15 logical Pauli string can be measured in the exact same logical time step as a weight-1 string.

A common anti-pattern in FTQC compilation is to translate intermediate representations into ZX-calculus graphs before lowering to lattice surgery. **This is unnecessary with UCC.** Because the Optimized HIR is *already* a mathematically pure list of multi-qubit Pauli correlation operations, it acts as the native AST for lattice surgery routers like tqec.

By programmatically constructing a tqec.BlockGraph directly from the HIR, we eliminate synthesis layers, preserve $\mathcal{O}(1)$ TOHPE optimizations, and directly emit fault-tolerant spacetime blocks.

#### Implementation Blueprint (tqec Lattice Surgery)

```python
import tqec

def compile_hir_to_tqec(hir_module) -> tqec.BlockGraph:
    """Maps UCC Optimized HIR directly to tqec lattice surgery spacetime blocks."""
    graph = tqec.BlockGraph()

    # 1. Track the spacetime coordinates of logical memory patches
    logical_patches = {
        i: graph.add_memory_patch(f"q_{i}") for i in range(hir_module.num_qubits)
    }

    for op in hir_module.ops:
        p_str = decode_masks(op.destab_mask, op.stab_mask, hir_module.num_qubits)
        involved = [q for q, p in enumerate(p_str) if p != 'I']
        if not involved: continue

        # Advance the BlockGraph time boundary to prevent spatial collisions
        time_slice = graph.add_layer()

        if op.type.name in {"T_GATE", "T_DAG_GATE"}:
            # 1. Inject a magic state patch at a free spatial location
            magic_patch = time_slice.add_magic_state_injection(init_state="T")

            # 2. Define the multi-patch parity measurement (Lattice Surgery merge/split)
            measurement_surface = tqec.CorrelationSurface()
            for q in involved:
                measurement_surface.add_connection(logical_patches[q], basis=p_str[q])

            # The magic state is always measured in Z to trigger the T-rotation
            measurement_surface.add_connection(magic_patch, basis='Z')

            # 3. Add the multi-body parity check to the block graph
            time_slice.add_correlation_surface(measurement_surface)

        elif op.type.name == "MEASURE":
            # Pure lattice surgery measurement without magic states
            measurement_surface = tqec.CorrelationSurface()
            for q in involved:
                measurement_surface.add_connection(logical_patches[q], basis=p_str[q])
            time_slice.add_correlation_surface(measurement_surface)

        elif op.type.name == "DETECTOR":
            # Expose UCC's deterministically compiled parity checks
            # directly to tqec's detector annotations for QEC decoding
            graph.add_detector(op.detector_targets)

    # Note: As with QCCD, a final state injection/correction layer based on
    # hir_module.final_tableau must be appended to the BlockGraph to ensure
    # correct logical readouts.

    return graph
```

---

### 3.3 Future Targets

#### NISQ 2D Superconducting Grids

**Verdict: Poor Target.**

To synthesize the HIR back into a standard CNOT/H/S circuit (`.qasm`) for a near-term superconducting device, a compiler must generate a "Phase Gadget" (a V-shaped ladder of CNOTs) for every HIR operation, plus a final basis correction using the Final Tableau. Because rewinding a $T$-gate through deep Cliffords causes a "Heisenberg Smear" (high-weight Pauli strings), synthesizing it on a rigid 2D grid requires massive CNOT ladders interwoven with dozens of `SWAP` gates, destroying circuit depth and locality. UCC intentionally deletes intermediate routing Cliffords to achieve $\mathcal{O}(1)$ compilation speed, making it inherently hostile to 2D NISQ routing.

#### Neutral Atoms (Shuttling & qLDPC)

**Verdict: Excellent Future Target.**

Neutral atoms use optical tweezers to dynamically shuttle blocks of qubits, making them the premier platform for qLDPC (Quantum Low-Density Parity-Check) codes. qLDPC requires highly non-local, high-weight Pauli parity checks for syndrome extraction. UCC processes a weight-15 parity check exactly as fast as a weight-1 measurement, acting as the perfect logical engine. To preserve shuttling block commands while minimizing the $T$-count (which Eastin-Knill dictates atom arrays still need), the compiler can utilize "Chunked Rewinding" to optimize logic strictly between physical atom movements.

---

## 4. Error Mitigation: Pauli Twirling & Noise Characterization

### Overview

Error mitigation strategies (like Pauli twirling or Probabilistic Error Cancellation) convert coherent errors into stochastic Pauli noise by inserting random Pauli gates into the circuit. Simulating complex twirled noise models typically crushes standard dynamic simulators due to the sheer volume of random branching.

- **Program Representation Level:** Level 3 (Back-End) & Level 4 (VM Runtime).
- **Minimum Functionality Needed:** Phase 2. Multi-Pauli noise channels in the AST and HIR, compiled into the Constant Pool's NoiseSchedule.

#### Demonstration Strategy

Instead of twirling the circuit text (which exponentially bloats the number of circuits to track), the user compiles the circuit *once*. During code generation, the Back-End compiles the twirled Pauli channels into the NoiseSchedule. The SVM then samples these errors via branchless geometric gap sampling. We will demonstrate that executing a heavily twirled circuit in UCC takes effectively the exact same amount of time as simulating the bare circuit, fundamentally solving the simulation bottleneck for error mitigation research.

---

## 5. Variational Quantum Algorithms (VQAs / QAOA) at Scale

### Overview

Variational algorithms (VQE, QAOA) require executing the exact same circuit topology thousands of times, iteratively updating the rotation angles of non-Clifford gates (e.g., $R_Z(\theta)$). Standard simulators must re-compile the circuit or re-traverse the AST for every angle change.

- **Program Representation Level:** Level 4 (VM Constant Pool Mutation).
- **Minimum Functionality Needed:** Phase 2 & 3. Generic GATE OpType in the HIR, LCU fast-math payloads in the Back-End, and the OP\_BRANCH\_LCU opcodes. Template monomorphization (>64 qubit support) is needed for scale.

#### Demonstration Strategy

In UCC, the geometric structure of a VQA circuit never changes—only the complex weights do. The Back-End lowers arbitrary rotations into pre-computed memory shifts (x_mask) and Constant Pool references. We can expose a Python API to directly mutate the floating-point values in the ConstantPool (lcu_pool). This bypasses AOT compilation entirely across training epochs, dropping the per-iteration compilation latency to zero and achieving unprecedented simulation speeds for VQA parameter sweeps.
