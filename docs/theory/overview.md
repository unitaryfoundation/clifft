# Theoretical Overview

Clifft is a multi-level compiler and execution engine for universal quantum circuits. It solves the exponential memory wall of non-Clifford simulation by decoupling the deterministic coordinate transformations of a quantum circuit from its probabilistic complex amplitudes.

## The Factored State Representation

Clifft represents the exact physical quantum state at time $t$ via a strict factorization:

$$|\psi^{(t)}\rangle = \gamma^{(t)} \, U_C^{(t)} \, P^{(t)} \, \Big( |\phi^{(t)}\rangle_A \otimes |0\rangle_D \Big)$$

Each component serves a distinct purpose:

- **$U_C$ (The Clifford Frame):** A static coordinate system mapping virtual qubits to physical qubits. Evaluated entirely at compile time.

- **$P$ (The Pauli Frame):** Tracks discrete stochastic bit-flips and phase-flips from measurements and noise. Updated at runtime via fast $\mathcal{O}(1)$ bitwise XOR operations.

- **$A$ and $D$ (Active/Dormant Sets):** The $n$ virtual qubits are partitioned into $k$ *Active* qubits (in superposition) and $n - k$ *Dormant* qubits (in the computational zero state).

- **$|\phi\rangle_A$ (The Active Statevector):** A dense, unnormalized complex array of size $2^k$ tracking non-Clifford interference. This is the only component that requires exponential memory — but it scales with $k$, not $n$.

- **$\gamma$ (The Global Scalar):** Tracks the continuous global phase and physical norm.

### Why This Matters

For circuits where non-Clifford entanglement is bounded — such as magic state distillation, where syndrome measurements aggressively "cool" superposition — the peak active dimension $k_{\text{max}}$ remains small even as the total qubit count $n$ grows to hundreds. Clifft allocates $2^{k_{\text{max}}}$ complex amplitudes instead of $2^n$, yielding exponential memory savings.

## The Five-Stage Pipeline

```text
Circuit Text  -->  1. Front-End (Physical Rewinding)
                        |  Absorbs physical Cliffords. Rewinds non-Cliffords
                        |  and measurements to the t=0 vacuum.
                        v
                   Heisenberg IR (HIR)
                        |
                   2. Middle-End Optimizer
                        |  O(1) static Pauli commutation checks to fuse
                        |  and cancel redundant operations.
                        v
                   Optimized HIR
                        |
                   3. Back-End (Virtual Compression)
                        |  Maps t=0 Paulis to the active virtual frame.
                        |  Synthesizes greedy O(n) basis compressions.
                        |  Emits localized VM bytecode.
                        v
                   Program (Raw Bytecode + Constant Pool)
                        |
                   4. Bytecode Optimizer
                        |  Fuses sequences of instructions to minimize
                        |  array passes and dispatch overhead.
                        v
                   Optimized Program
                        |
                   5. Virtual Machine (Execution)
                        |  Executes array updates and tracks Pauli frame P.
                        |  Allocates exactly ONE array of size 2^{k_max}.
                        v
                   Measurement Results
```

### Stage 1: Front-End (Physical Rewinding)

The Front-End mathematically absorbs all physical Clifford operations using a stabilizer tableau simulator. For every non-Clifford gate, measurement, or noise channel, it extracts the *rewound* Pauli string at $t = 0$:

$$P_{t \to 0} = U_{\text{phys}}^\dagger \, P_t \, U_{\text{phys}}$$

The output is the **Heisenberg IR (HIR)**: a static list of abstract Pauli operations completely devoid of hardware timing or memory layout.

### Stage 2: Middle-End Optimizer

The optimizer applies $\mathcal{O}(1)$ static Pauli commutation checks to fuse and cancel redundant operations in the HIR. Because the HIR is purely algebraic, the optimizer can reason about gate interactions without simulating the full quantum state.

### Stage 3: Back-End (Virtual Compression)

The Back-End bridges the static HIR to the runtime VM. It maintains a cumulative virtual frame mapping $V_{\text{cum}}$ and, for each multi-qubit Pauli in the HIR:

1. Maps the $t = 0$ Pauli to the current virtual frame: $P_v = V_{\text{cum}} \, P_{t=0} \, V_{\text{cum}}^\dagger$
2. Computes a greedy $\mathcal{O}(n)$ compression sequence of virtual CNOT/CZ gates to localize $P_v$ onto a single virtual axis
3. Emits localized VM opcodes

Because multi-qubit measurements are explicitly compressed into single-qubit virtual measurements at compile time, no tableau mathematics runs at simulation time.

### Stage 4: Bytecode Optimizer

After lowering, a second pass manager applies peephole optimizations directly to the bytecode. These passes fuse sequences of instructions to eliminate redundant passes over the array and reduce dispatch overhead.

### Stage 5: Virtual Machine (SVM)

The VM executes the bytecode over millions of shots. Key properties:

- **Zero-Cost Dormant Operations:** If a virtual CNOT targets a dormant qubit (known to be $|0\rangle$), the VM simply conjugates the Pauli frame via bitwise XOR. The complex array is untouched.

- **Array Compaction:** When an active qubit is measured, it collapses and is demoted to dormant. The compiler emits virtual SWAP instructions to keep the array contiguous in memory, physically halving its size without strided fragmentation.

- **Single Allocation:** The VM allocates its complex amplitude array exactly once based on the peak active rank $k_{\text{max}}$. No dynamic resizing during execution.
