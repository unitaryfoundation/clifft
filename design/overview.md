# UCC — System Overview & Theoretical Foundation

**UCC** (Unitary Compiler Collection) is a multi-level Ahead-Of-Time (AOT) compiler and execution engine for universal quantum circuits.

It solves the exponential memory wall of non-Clifford simulation by decoupling the deterministic coordinate transformations of a quantum circuit from its probabilistic complex amplitudes.

## 1. The Factored State Representation

UCC represents the exact physical quantum state at time $t$ via a strict factorization:

$$|\psi^{(t)}\rangle = \gamma^{(t)} U_C^{(t)} P^{(t)} \Big( |\phi^{(t)}\rangle_A \otimes |0\rangle_D \Big)$$

- **$U_C$ (The Clifford Frame):** A static coordinate system mapping virtual qubits to physical qubits. Evaluated entirely AOT.
- **$P$ (The Pauli Frame):** Tracks discrete stochastic bit-flips and phase-flips. Evaluated at runtime via fast $\mathcal{O}(1)$ bitwise XORs.
- **$A$ and $D$ (Active/Dormant Sets):** The virtual qubits are partitioned into $k$ Active qubits (in superposition) and $n-k$ Dormant qubits (in the computational zero state).
- **$|\phi\rangle_A$ (The Active Statevector):** A dense, unnormalized complex array of size $2^k$ tracking non-Clifford interference.
- **$\gamma$ (The Global Scalar):** Tracks the continuous global phase and physical norm.

## 2. The Four-Stage Pipeline

```text
  Circuit Text ──►  1. Front-End (Physical Rewinding)
                         │  Absorbs physical Cliffords. Rewinds non-Cliffords
                         │  and measurements to the t=0 vacuum.
                         ▼
                    Heisenberg IR (HIR)
                         │
                  2. Middle-End (Optimizer)
                         │  O(1) static Pauli commutation checks to fuse
                         │  and cancel redundant operations.
                         ▼
                    Optimized HIR
                         │
                  3. Compiler Back-End (Virtual Compression)
                         │  Maps t=0 Paulis to the active virtual frame.
                         │  Synthesizes greedy O(n) basis compressions.
                         │  Emits localized RISC bytecode.
                         ▼
                    Program (Bytecode + Constant Pool)
                         │
                  4. Virtual Machine (Execution)
                         │  Executes array updates and tracks Pauli frame P.
                         │  Allocates exactly ONE array of size 2^{k_max}.
                         ▼
                    Measurement Results
```

### 2.1 Front-End (Physical Rewinding)

The Front-End mathematically absorbs all physical Clifford operations using a `stim::TableauSimulator`. For every non-Clifford gate, measurement, or noise channel, it extracts the rewound Pauli string at $t=0$ ($P_{t \to 0} = U_{phys}^\dagger P_t U_{phys}$). The output is the Heisenberg IR (HIR): a static list of abstract Pauli operations completely devoid of hardware timing or memory layout.

### 2.2 Compiler Back-End (Virtual Frame Compression)
The Back-End bridges the static HIR to the runtime VM. It maintains a cumulative virtual frame mapping ($V_{cum}$). For every multi-qubit Pauli in the HIR:It maps the $t=0$ Pauli to the current virtual frame: $P_v = V_{cum} P_{t=0} V_{cum}^\dagger$.It executes a greedy $\mathcal{O}(n)$ algorithm to compute a sequence of virtual CNOT/CZ gates ($V$) that compresses $P_v$ onto a single virtual axis.It updates the virtual frame ($V_{cum} \leftarrow V V_{cum}$).It emits localized RISC opcodes corresponding to the compression sequence and the targeted physics.Because multi-qubit measurements are explicitly compressed into single-qubit virtual measurements AOT, Aaronson-Gottesman (AG) pivot matrices are completely eradicated from the VM.

### 2.3 The Virtual Machine (SVM)
The VM executes the RISC bytecode over millions of shots. It tracks the $n$-bit Pauli frame ($P$) and the $2^k$ dense Active Statevector.Zero-Cost Dormant Property: If the Back-End emits a virtual CNOT where the control is in the Dormant set ($D$), the compiler emits an OP_FRAME_CNOT. The VM simply conjugates $P$ via bitwise XORs. Because the control is strictly $|0\rangle$, the complex Active Statevector is completely untouched (Zero FLOPs).Array Compaction: When an Active virtual qubit is measured, it collapses back to $|0\rangle$ or $|1\rangle$ and is demoted to Dormant. To keep the array contiguous in memory, the compiler emits virtual SWAP instructions to route the measured qubit to the highest array axis ($k-1$) just before measurement. The VM then physically halves the array size without strided memory fragmentation.
