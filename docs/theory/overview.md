# Theoretical Overview

Clifft is a multi-level compiler and execution engine for universal quantum circuits. It reduces the exponential part of exact simulation from the total qubit count to the dynamic active dimension for circuits where non-Clifford effects remain localized.

## The Factored State Representation

Clifft represents the exact physical quantum state at time $t$ via a strict factorization:

$$|\psi^{(t)}\rangle = \gamma^{(t)} \, U_C^{(t)} \, P^{(t)} \, \Big( |\phi^{(t)}\rangle_A \otimes |0\rangle_D \Big)$$

Each component serves a distinct purpose:

- **$U_C$ (The Clifford Frame):** A coordinate system mapping virtual qubits to physical qubits (akin to the Heisenberg picture). Evaluated entirely at compile time.

- **$P$ (The Pauli Frame):** Tracks discrete stochastic bit-flips and phase-flips from measurements and noise. Updated at runtime via fast bitwise XOR operations.

- **$A$ and $D$ (Active/Dormant Sets):** The $n$ virtual qubits are partitioned into $k$ *Active* qubits (in superposition) and $n - k$ *Dormant* qubits (in the computational zero state). $k$ is the **active dimension** (also called the *active rank* in the codebase).

- **$|\phi\rangle_A$ (The Active State Vector):** A dense, unnormalized complex array of size $2^k$ tracking non-Clifford interference. The dense active array is the only component whose size is exponential.

- **$\gamma$ (The Global Scalar):** Tracks the continuous global phase and physical norm.

### Why This Matters

For circuits where non-Clifford entanglement is bounded — such as magic state distillation, where     frequent syndrome measurements can collapse inactive degrees of freedom and keep non-Clifford effects localized — the peak active dimension $k_{\text{max}}$ remains small even as the total qubit count $n$ grows to hundreds. Clifft allocates $2^{k_{\text{max}}}$ complex amplitudes instead of $2^n$, yielding exponential memory savings.

## The Five-Stage Pipeline

```text
Circuit Text  -->  1. Front-End (Heisenberg Mapping)
                        |  Absorbs physical Cliffords. Maps non-Cliffords
                        |  and measurements to the virtual basis.
                        v
                   Heisenberg IR (HIR)
                        |
                   2. Middle-End Optimizer
                        |  O(1) static Pauli commutation checks to fuse
                        |  and cancel redundant operations.
                        v
                   Optimized HIR
                        |
                   3. Back-End (Pauli Localization)
                        |  Synthesizes greedy O(n) virtual Clifford sequences
                        |  that localize each Pauli product to a single axis.
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

### Stage 1: Front-End (Heisenberg Mapping)

The Front-End absorbs all physical Clifford operations into the offline Clifford frame $U_C$ using a stabilizer tableau simulator (Stim's fast implementation in fact!). For every non-Clifford gate, measurement, or noise channel, it extracts the virtual Pauli string via the Heisenberg mapping:

$$P_{\text{virtual}} = U_C^\dagger \, P \, U_C$$

The output is the **Heisenberg IR (HIR)**: an equivalent and more compact representation of the circuit where all Clifford operations are implicitly represented in the mapped Pauli string of the non-Clifford operations.

### Stage 2: Middle-End Optimizer

The optimizer applies fast bitwise Pauli commutation checks to fuse and cancel redundant operations in the HIR, and also push commuting measurements early & non-Cliffords later to limit $k_\text{max}$. Because the HIR is purely algebraic, the optimizer can reason about gate interactions without simulating the full quantum state.

### Stage 3: Back-End Pauli Localization

The back end lowers optimized HIR into SVM bytecode by localizing each relevant Pauli product to a single virtual axis. This compile-time Pauli localization converts multi-qubit rotations and measurements into localized VM operations, so sample-time execution avoids tableau updates and acts directly on the active array.

### Stage 4: Bytecode Optimizer

After lowering, we apply another set of optimizations directly to the bytecode, targetting specific improvements to improve sampling time. These passes fuse sequences of instructions to eliminate redundant passes over the array and reduce dispatch overhead.

### Stage 5: Schrödinger Virtual Machine

The Schrödinger Virtual Machine executes the localized bytecode using the factored state representation. Because Clifford-coordinate updates and Pauli localization have already been performed at compile time, repeated sampling only updates the Pauli frame and the dense active state vector, with exponential cost confined to the current active dimension.
