# UCC — Extensions & Future Directions

## Abstract

The Unitary Compiler Collection (UCC) achieves its unprecedented performance through a strict architectural separation: $\mathcal{O}(n^2)$ topological tracking is resolved by an Ahead-Of-Time (AOT) compiler, reducing the runtime Virtual Machine (VM) to $\mathcal{O}(1)$ probability evaluations and localized RISC operations over a dense, dynamically-sized contiguous array.

While the MVP establishes UCC as an exact, hyper-fast simulator and an algebraic optimizer, the underlying Factored State Architecture naturally supports a much wider array of use cases.

This document outlines forward-looking extensions, system integrations, and architectural upgrades. It details features that improve the toolchain for developers, unlock massively parallel hardware acceleration (SIMD/GPU), and define explicit research directions required to handle massive-scale logical compilation.

---

## 1. Hardware Acceleration: CPU Vectorization & GPUs

Standard quantum simulators are notoriously difficult to port to GPUs or vectorize because of severe thread divergence (random measurements) and fragmented memory access resulting from dynamically sized tree tracking.

UCC completely bypasses this. By guaranteeing **Clifford Frame Determinism**, the AOT compiler eliminates dynamic topological branching and explicitly emits array compaction operations (`OP_FRAME_SWAP` / `OP_ARRAY_SWAP`) prior to measurements. The VM executes a flat, static list of 32-byte RISC opcodes over a strictly contiguous array. This transforms quantum simulation into an ideal SIMD/SIMT workload.

### 1.1 CPU Vectorization (SIMD / AVX-512)
Because the runtime VM operates purely on contiguous complex arrays and 64-bit integer Pauli trackers, it can be highly vectorized in two distinct regimes:

* **Intra-Shot Vectorization (High $k$):** For circuits experiencing deep entanglement ($k \ge 10$), the nested loops over the $v[]$ array during `OP_ARRAY_CNOT` or `OP_PHASE_T` can process 4 to 8 complex amplitudes simultaneously using AVX-512 registers. Because the memory is allocated using AVX boundaries (`std::aligned_alloc(64, ...)`), this yields a near-linear speedup on the math hot-loops.
* **Inter-Shot Batching (Low $k$):** For highly optimized error correction circuits (like Magic State Cultivation), the peak active dimension $k$ is often extremely small ($k \in [2, 5]$). Rather than vectorizing a tiny 4-element array, we vectorize *across* shots. By transitioning the `SchrodingerState` to a Struct-of-Arrays (SoA) layout, a single AVX-512 register can pack 8 independent Monte Carlo shots. Because the RISC bytecode is static, all 8 shots execute the exact same instructions in perfect lockstep. If a measurement outcome diverges between shots, the VM uses SIMD blend instructions (`vblendvpd`) to apply divergent algebraic phases to $\gamma$ without CPU branching.

### 1.2 GPU Acceleration (CUDA / HIP)
UCC's offline coordinate mapping is the "Holy Grail" for unlocking GPU acceleration for trillion-shot trajectories:

* **Massively Parallel Shot-Batching:** We can launch a 1D CUDA Grid where each thread (or warp) handles an independent Monte Carlo shot. If $k_{\max}$ is small, the exact memory required per shot is known AOT. The entire $2^k$ complex array and Pauli bit-trackers can be pinned directly into ultra-fast **GPU Shared Memory** or local registers, bypassing VRAM bottlenecks entirely.
* **Constant Memory for Bytecode:** The compiled program is extremely lightweight. The sequence of 32-byte RISC opcodes is uploaded directly to the GPU's `__constant__` memory, which broadcasts instructions simultaneously to thousands of execution cores with near-zero latency.
* **Device-Side Gap Sampling:** The host-side PRNG can be swapped for a device-native generator (like `cuRAND`), allowing geometric gap sampling for Pauli noise to be evaluated completely asynchronously by the GPU kernels.

---

## 2. Coherent Noise & Approximate Simulation (AOT Pruning)

### 2.1 The Challenge: Coherent Noise Rank Explosion
Simulating stochastic Pauli noise is highly efficient in UCC due to branchless geometric gap sampling. However, **coherent noise** (e.g., a persistent over-rotation $R_{ZZ}(\epsilon)$ on every gate) fundamentally alters the continuous superposition.

UCC mathematically models coherent noise using the Linear Combination of Unitaries (LCU) framework. An over-rotation decomposes as $\cos(\epsilon)I - i\sin(\epsilon)Z$. The Front-End uses Dominant Term Factoring to extract the massive $I$ branch (costing zero runtime FLOPs), leaving the compiler to emit a generic `OP_EXPAND` to evaluate the tiny error branch.

The problem is the **dimension explosion**: if coherent noise is applied to every gate in a deep circuit, the VM's active dimension $k$ increments endlessly. The complex array $v[]$ doubles with every rotation, making exact simulation of dense coherent noise fundamentally intractable and leading to rapid Out-Of-Memory (OOM) errors.

### 2.2 The Extension: AOT Static Pruning
Because the VM's execution relies on a strict static memory allocation based on the `peak_rank` ($k_{\max}$) and exact integer axis-targeting (`axis_1`, `axis_2`), we cannot dynamically abort branches at runtime without destroying the coordinate mapping.

To support deep circuits with coherent noise, UCC can be extended into a **high-performance approximate simulator** via *AOT Static Pruning*.

During the Middle-End optimization phase, the compiler evaluates the fused relative weights ($c$) of LCU operations. If a spawned coherent noise branch falls below a user-defined numerical threshold (e.g., $|c| < 10^{-6}$), the compiler simply deletes the node from the Heisenberg IR (HIR). This occurs *before* the Back-End compresses the virtual basis or calculates $k_{\max}$.

**Impact:** This transforms exponential memory scaling into a constrained polynomial footprint. It allows UCC to simulate massive circuits with weak coherent noise, acting as a highly competitive, bounds-aware alternative to tensor-network approximate simulators, all while executing on the exact, ultra-fast RISC VM.

---

## 3. Arbitrary Qubit Scaling & The HIR Memory Wall

In the VM, UCC natively supports massive qubit scaling (e.g., 65,536 qubits) without bloating memory because the Back-End geometrically compresses all global topology into single-qubit and two-qubit localized `uint16_t` axes.

However, scaling to massive physical topologies introduces a severe memory wall in the **AOT Compiler**.

### 3.1 The "Heisenberg Smear" and L1 Cache Failure
The Front-End and Middle-End operate strictly in the physical topology, utilizing dense `stim::bitword<kStimWidth>` masks. Scaling to 10,000 physical qubits requires 2.5 KB of memory *per operation* just to hold the Pauli masks.

When a localized non-Clifford gate is rewound to the $t=0$ vacuum through deep, highly-entangling Clifford blocks (like surface code syndrome cycles), its operator scrambles and "smears" across the lattice into a massive, high-weight Pauli string. The Middle-End Optimizer relies on $\mathcal{O}(1)$ single-cycle hardware popcounts for commutation sweeps and $\mathcal{O}(n^4m^3)$ matrix reductions in the FastTODD pass. If operations bloat to 2.5 KB, the working set spills out of the CPU's ultra-fast L1 cache, causing compilation times to spike from milliseconds to minutes.

### 3.2 The Solution: Chunked Rewinding (Moving Frames)
To solve the cache wall, we can replace global rewinding with **Chunked Rewinding**.

Instead of rewinding every operation all the way back to the absolute $t=0$ vacuum, the Front-End drops geometric checkpoints (e.g., at the end of every QEC limit-cycle). Operations are rewound only to the boundary of their local temporal chunk.
* This artificially truncates the lightcone, keeping the Pauli strings strictly local and low-weight.
* Because they are low-weight, the HIR can utilize highly-compressed sparse memory layouts (just a few bytes per op), keeping the optimizer fully resident in the L1 cache.
* To connect the chunks during VM execution, the Compiler Back-End will emit a highly optimized `OP_FRAME_SHIFT` instruction. This allows the VM to mathematically realign its 1D Pauli tracking bit-words between chunks, maintaining perfect topological determinism without unrolling the geometric drift.

---

## 4. Advanced Workflows & Alternative Targets

Because UCC compiles circuits into an optimized, algebraically independent list of localized Pauli operations, it can support dynamic workflows far beyond standard simulation.

### 4.1 Real-Time QEC Decoder Streaming
Modern quantum error correction requires decoding syndrome data in real-time. Currently, the UCC VM writes classical parity checks (`OP_DETECTOR`) sequentially to a flat memory array, which is analyzed post-simulation.

Because `OP_DETECTOR` is a trivial classical boolean XOR, UCC can be extended to stream these events directly into a lock-free concurrent queue (e.g., a ring buffer). An asynchronous, highly-optimized C++ QEC decoder (like PyMatching or BPOSD) can run on a parallel CPU thread, consuming detection events and predicting logical frame corrections *while the UCC VM is still executing the quantum simulation*. This mirrors actual physical QPU control-system architectures and enables online decoding workflows.

### 4.2 Ecosystem Integrations & Structured Control Flow
Integration with modern structured languages (like OpenQASM 3) introduces complex control flow—specifically, arbitrary classical logic based on mid-circuit measurements (e.g., `if (meas == 1) { H 0; }`).

UCC formally delineates what is possible under Ahead-Of-Time compilation:
1. **Classically-Controlled Paulis (Supported):** A classically-controlled Pauli (e.g., `if (meas == 1) { X 0; }`) is mathematically safe. It compiles cleanly to an `OP_APPLY_PAULI` instruction. At runtime, the VM evaluates the condition and conditionally XORs the error mask into the Pauli frame $P$. The underlying basis geometry ($U_C$) is undisturbed.
2. **Classically-Controlled Cliffords (Unsupported):** A classically-controlled Clifford gate mathematically branches the global coordinate frame ($U_C$) into divergent geometric universes based on a runtime coin flip. This fundamentally breaks **Clifford Frame Determinism**, making it impossible to synthesize a single virtual compression map ($V_{cum}$) offline.

By enforcing strict topological determinism at the parser level, UCC acts as a rigorously FTQC-safe backend, guaranteeing that the supplied physical architectures and static error models are perfectly preserved throughout compilation and execution.
