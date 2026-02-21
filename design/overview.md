# UCC — System Design & Implementation Plan

**UCC** — the **Unitary Compiler Collection** — is an ultra-fast, multi-level
compiler infrastructure for exact universal quantum circuits.

## 1. System Architecture & Document Map

UCC fundamentally decouples the heavy $\mathcal{O}(n^2)$ geometric structure of a quantum state from its $\mathcal{O}(1)$ probabilistic execution. It achieves this by shifting all Boolean matrix tracking (the stabilizer tableau) into a multi-level Ahead-Of-Time (AOT) compiler pipeline, reducing the runtime Schrodinger Virtual Machine (SVM) to strict, divergence-free array math and bitwise XORs.

This architecture embraces the physics duality: the Front-End compiles the circuit strictly in the **Heisenberg picture** (Heisenberg IR), while the SVM executes the continuous probability amplitudes forward in time in the **Schrodinger picture**.

To support both **state-of-the-art circuit optimization** (e.g., Pauli-Based Computation routing) and **hyper-fast Monte Carlo simulation**, UCC is structured as a strict multi-tier compiler pipeline:

1. **overview.md (This File):** System overview, the 4-stage pipeline, Big-$\mathcal{O}$ performance thesis, algorithmic optimizations, and the roadmap for complex control flow.
2. **data_structs.md:** The Heisenberg IR (HIR), C++ SVM memory model, the 32-byte Instruction bytecode, constant pool, and the runtime execution loops.
3. **architecture.md:** Software engineering blueprint, repository layout, build system, Stim C++ integration, Python bindings, and implementation phasing.

---

## 2. The Four-Stage Pipeline

```
                          ┌─────────────────────────────────────────────────┐
  Circuit Text ──►  1. Front-End (Trace Generation)                        │
  (.stim superset)       │  Parses circuit, tracks Clifford frame via      │
                         │  stim::TableauSimulator, rewinds non-Clifford   │
                         │  ops to t=0 via Heisenberg picture.             │
                         │                                                 │
                         ▼                                                 │
                    Heisenberg IR (HIR)                                     │
                    [flat list of HeisenbergOps]                            │
                         │                                                 │
                  2. Middle-End (Optimizer)                                 │
                         │  O(1) bitwise commutation checks to reorder,    │
                         │  fuse, and cancel operations. Respects          │  Runs ONCE
                         │  stochastic memory barriers (MEASURE, NOISE).   │  per circuit
                         │                                                 │
                         ▼                                                 │
                    Optimized HIR                                           │
                    [may be exported to tqec / PBC routing]                 │
                         │                                                 │
                  3. Compiler Back-End (Code Generation)                   │
                         │  Lowers HIR to hardware-aligned 32-byte         │
                         │  bytecode. Computes x_mask, commutation_mask,       │
                         │  AG pivot matrices. Builds NoiseSchedule.       │
                         │                                                 │
                         ▼                                                 │
                    Program (bytecode + ConstantPool)                         │
                         │                                                 │
                  ───────┼─────────────────────────────────────────────────┘
                         │
                  4. Schrodinger VM (Runtime)          ← Runs N times
                         │  Executes bytecode over millions of shots.
                         │  SoA memory model, L1 cache-resident,
                         │  geometric gap sampling for noise.
                         ▼
                    Measurement Results
```

### 2.1 Front-End (Trace Generation)

The foundation of UCC is **Topological Determinism**. Because stochastic events (noise, random measurement outcomes) merely flip scalar $\pm 1$ signs and do not warp the underlying Boolean matrix grid, the geometric evolution of the frame is 100% deterministic.

The Front-End evaluates this geometry. It mathematically absorbs all Clifford gates (they vanish entirely) and uses Heisenberg rewinding ($P_{t \to 0} = U^\dagger P_t U$) to map every remaining operation to the initial $t=0$ basis. The output is the **Heisenberg IR (HIR)** — a flat, algebraically independent list of operations defined purely by their 64-bit Pauli strings.

### 2.2 Middle-End (Optimizer)

Because the HIR contains no SVM-specific memory routing or active frame constraints, the Middle-End can optimize universal quantum circuits natively at the speed of classical bitwise arithmetic.

The optimizer uses a **Pass-Based Architecture** (similar to LLVM's PassManager or MLIR), running a configurable pipeline of optimization passes over the HIR. The default optimization pipeline is:

```
[PeepholeFusionPass, TohpeGlobalPass, PeepholeFusionPass]
```

- **PeepholeFusionPass:** A fast, $\mathcal{O}(n)$ sweep that checks adjacent operations to cancel trivial redundancies (e.g., $T \cdot T^\dagger \to I$) and fold Cliffords ($T \cdot T \to S$). See `data_structs.md` §2.2 for the fusion algebra.
- **TohpeGlobalPass:** The heavy $\mathcal{O}(m^3)$ Heisenberg-TOHPE algorithm that extracts GF(2) null-spaces to find global topological redundancies across commuting cliques. See `use_cases.md` §2 for the full algorithm description.

Running the peephole pass *before* TOHPE shrinks the non-Clifford gate count ($m$) for the cubic solver, and running it *after* cleans up any newly adjacent redundancies generated by TOHPE's substitutions.

All passes respect **barrier-aware scheduling** where MEASURE and NOISE nodes block reordering of anti-commuting gates (using $\mathcal{O}(1)$ symplectic inner product checks).

At this stage, the optimized HIR can be **exported directly** to physical routing tools like tqec for Pauli-Based Computation on quantum hardware.

### 2.3 Compiler Back-End (Code Generation)

*Note: "Back-End" here refers to the compiler stage, not a QPU hardware backend.*

To simulate the optimized HIR, the Compiler Back-End lowers the abstract Pauli strings into rigid, executable SVM bytecode. It iterates sequentially through the HIR and actively maintains the GF(2) basis $V$ of the shifting superposition. For every operation, it calculates:

- **x_mask:** The exact memory offset into the SVM's coefficient array, determined by evaluating the spatial shift $\beta$ against $V$.
- **commutation_mask:** The pre-computed commutation of the operator against every active basis vector in $V$, packed into a single 32-bit integer for $\mathcal{O}(1)$ phase evaluation.
- **AG Matrices:** The Back-End moves the pre-computed AG matrices from the Front-End's side-table into the `ConstantPool` constant pool.
- **NoiseSchedule:** A side-table mapping bytecode PCs to pre-computed noise bitmasks and probabilities, enabling geometric gap sampling at runtime.

The Back-End also uses the explicit `T_GATE` and `T_DAG_GATE` OpTypes in the HIR to deterministically route to fast-path opcodes (OP_BRANCH, OP_COLLIDE, OP_SCALAR_PHASE) that hardcode the $\tan(\pi/8)$ weight. Generic `GATE` operations are lowered to LCU opcodes with constant pool references.

### 2.4 Schrodinger Virtual Machine (Runtime)

The SVM executes the compiled bytecode over millions of Monte Carlo shots. While the Front-End operates in the Heisenberg picture (rewinding operators to $t=0$), the SVM executes in the Schrodinger picture, propagating the continuous probability amplitudes forward in time.

It uses a Struct-of-Arrays (SoA) memory model and a Constant Pool to ensure perfect CPU L1 cache saturation and GPU SIMT lockstep. Noise is injected via geometric gap sampling over the NoiseSchedule — the bytecode stream itself is free of noise instructions.

---

## 3. The Performance Thesis (Complexity Summary)

By shifting all topological updates into the multi-level AOT pipeline, the UCC Schrodinger execution engine achieves mathematically optimal CPU/GPU throughput.

| Operation | Standard (Dynamic) Simulator | UCC Multi-Level AOT SVM |
| :--- | :--- | :--- |
| **Clifford Gate** | $\mathcal{O}(n/W)$ SIMD row updates | **0 ops** (Absorbed by Front-End) |
| **Pauli Noise** | $\mathcal{O}(n/W)$ SIMD row append | **$\mathcal{O}(1)$** integer XOR (geometric gap sampled) |
| **Measure Pivot (AG)** | $\mathcal{O}(n^2/W)$ SIMD matrix reduction | **$\mathcal{O}(1)$** pre-compiled mask XOR |
| **Dominant LCU Branch** | Standard FLOPs & memory writes | **0 FLOPs, 0 Memory Writes** |
| **Multi-Branch LCU ($K$-term)** | Synthesize deep subcircuits | **$\mathcal{O}(K \cdot 2^k)$** dense array updates |
| **Pauli Commutation (Optimizer)** | $\mathcal{O}(n)$ DAG traversal | **$\mathcal{O}(1)$** bitwise symplectic inner product |
| **Gate Fusion/Cancellation** | $\mathcal{O}(n)$ matrix algebra | **$\mathcal{O}(1)$** scalar multiply or delete |
| **Detectors** | Dynamic track evaluation | **$\mathcal{O}(1)$** classical boolean XOR |
| **Memory Allocations** | $\mathcal{O}(|V|)$ dynamic tree scaling | **Exactly 1** allocation per shot |

---

## 4. Key Algorithmic Optimizations

Beyond the pure math of generalized stabilizers, UCC relies on specific software engineering optimizations to saturate hardware limits.

### 4.1 Dominant Term Factoring (The Tangent Trick)

Evaluating branching superpositions (like a $T$-gate) naively requires writing memory for every single branch, bottlenecking CPU bandwidth. The Front-End elegantly solves this by factoring out the dominant term globally. For a $T$-gate, the exact unitary is:

$$T = e^{i\pi/8} \cos(\pi/8) \left[I - i \tan(\pi/8) Z\right]$$

The Front-End factors out $e^{i\pi/8} \cos(\pi/8)$, leaving the base branch ($I$) with a relative weight of exactly 1.0 and the spawned branch with a weight of $-i \tan(\pi/8)$. The factored-out scalar is accumulated into `HirModule::global_weight` to ensure the exported HIR remains mathematically exact.

**Result:** The heaviest path through the quantum state vector requires zero memory writes and zero FLOPs (it is implicitly carried forward during array doubling). Furthermore, all relative processed weights have a magnitude $\le 1.0$, making IEEE-754 numerical instability mathematically impossible.

For generic non-Clifford gates decomposed via LCU ($U = \sum_k c_k P_k$), the Front-End factors out the dominant Pauli term, stores only the relative complex weight, and accumulates the extracted scalar into `global_weight`. This structurally prevents numerical drift while maintaining exact amplitudes for HIR export.

### 4.2 $\mathcal{O}(1)$ Commutation and Layering (Middle-End)

To check if two HIR operations commute, the optimizer performs a single-cycle symplectic inner product:

```cpp
bool commute = (std::popcount((A.destab_mask & B.stab_mask) ^
                              (A.stab_mask & B.destab_mask)) % 2) == 0;
```

Using this, the Middle-End greedily groups gate nodes into parallel execution layers, minimizing required topological shifts. If two nodes share the exact same Pauli axis (`destab_mask` and `stab_mask`), the optimizer fuses them using the dominant term factoring algebra: for gates parameterized as $I + cP$, the fused weight is $(c_1 + c_2)/(1 + c_1 c_2)$. This cancels entirely when $c_1 + c_2 = 0$ (e.g., $T \cdot T^\dagger = I$), or forms a Clifford when $1 + c_1 c_2 = 0$ (e.g., $T \cdot T = S$). See `data_structs.md` §2.2 for details.

### 4.3 Geometric Gap Sampling (Branchless Noise)

Standard simulators process Pauli noise linearly, checking the RNG at every cycle, which stalls the CPU branch predictor. In UCC, the HIR preserves noise operations with explicit probabilities (required by the optimizer for barrier analysis). The Back-End then lowers these into a **NoiseSchedule** — a side-table in the constant pool mapping bytecode program-counter offsets to pre-computed noise bitmasks.

At runtime, the SVM samples a geometric distance $\Delta$ to the next error event, runs $\Delta$ purely quantum instructions at maximum speed with zero branch checks, pauses only at the sampled PC to XOR the error masks into its sign tracker, and resumes. The bytecode stream itself contains no noise instructions.

### 4.4 Peak Rank Memory Allocation

To prevent Out-Of-Memory errors and fragmentation, the complex coefficient array `v[]` is dynamically allocated **exactly once**. The Back-End tracks the GF(2) basis $V$ during compilation. Every expansion (OP_BRANCH) increases the rank by 1; every measurement merge (OP_MEASURE_MERGE) or filter (OP_MEASURE_FILTER) decreases it by 1. The Back-End outputs a `peak_rank` integer, and the SVM statically allocates a dense array of exactly $2^{\textrm{peak-rank}}$ entries upon initialization.

### 4.5 Deferred Normalization

The SVM hot loop does not track a `log_global_weight` or execute expensive `std::log`/`std::exp` calls. Amplitudes scale naturally during non-Clifford expansions. Because measurement outcomes evaluate the probability ratio $P(0) = S_0 / (S_0 + S_1)$, any missing global scalar mathematically cancels out. The SVM only divides by the $L_2$ norm *after* sampling a measurement outcome, keeping the non-Clifford expansion loops completely free of transcendental math and clearing any floating-point drift.

### 4.6 Strictly FTQC-Safe & Natively Noise-Aware Compilation

Traditional quantum compilers (like standard TODD, Qiskit, or PyZX) are **"FTQC-blind"**—they optimize circuits by blindly commuting non-Clifford gates past physical noise channels without regard to the underlying error model. This conjugates errors (e.g., turning an $X$ error into a $Y$ error via $T X T^\dagger \propto Y$) and destroys the hardware's physical error model. For fault-tolerant quantum computing (FTQC) workflows, this is catastrophic: downstream QEC decoders and phenomenological simulations depend on accurate noise structure.

UCC is **natively noise-aware**. Because UCC parses noise models directly from `.stim` files into concrete `NOISE` nodes in the HIR, the PassManager's $\mathcal{O}(1)$ symplectic inner product naturally blocks any illegal commutations. When a pass attempts to reorder a non-Clifford gate past a noise channel, the inner product check detects anti-commutation and halts the transformation. This guarantees that aggressive global T-count reduction (via TohpeGlobalPass or PeepholeFusionPass) can be performed while mathematically preserving the exact physical (or logical) error models required by downstream decoders.

**Implications by execution regime:**

- **NISQ Circuits (Noise-Aware Simulation):** For noisy intermediate-scale circuits, UCC's noise-aware optimization produces faster *simulation* by reducing the non-Clifford gate count while preserving the error model. However, this does not directly improve *physical execution* on hardware, because the optimized HIR must still be synthesized and routed—a process that may reintroduce overhead.
- **FTQC Circuits (Logical Optimization):** For fault-tolerant logical circuits, UCC's optimizations directly benefit both simulation *and* physical execution. The HIR can be exported to PBC routers (like tqec) without synthesis, and the preserved noise structure ensures QEC decoding remains valid.

---

## 5. The Front-End: AOT Compiler State & Translation Logic

While the SVM executes blind instructions, the Front-End must actively map the shifting topology of the circuit. It does this by driving a `stim::TableauSimulator` for Clifford gate absorption, then extracting Heisenberg-rewound Pauli strings from the inverse tableau for every non-Clifford gate, measurement, and noise channel.

### 5.1 Responsibilities of the Front-End

The Front-End owns the `stim::TableauSimulator` and is responsible for:

1. **Clifford Absorption:** All Clifford gates are mathematically absorbed into the tableau — they vanish from the HIR entirely.

2. **Heisenberg Rewinding:** For every non-Clifford gate, measurement, and noise channel, the Front-End extracts the Heisenberg-rewound Pauli string from the inverse tableau (`sim.inv_state.zs[q]`).

3. **AG Pivot Matrix Computation:** When a measurement collapses the tableau state, the Front-End computes the Aaronson-Gottesman (AG) pivot matrix that encodes the GF(2) change-of-basis. These matrices use `stim::Tableau<kStimWidth>` directly (where `kStimWidth=64` for MVP), providing SIMD-aligned storage and built-in composition via `fwd_after.then(inv_before)`. They are stored in a side-table (`std::vector<stim::Tableau<kStimWidth>> ag_matrices` owned by the `HirModule`), and each `HeisenbergOp::MEASURE` stores an index (`ag_matrix_idx`) into this table along with the reference outcome (`ag_ref_outcome`). This keeps the HIR lean and prevents bloating the optimizer's working set.

4. **Multi-Pauli Measurement Support:** Multi-Pauli measurements (e.g., `MPP X1*X2*Z3`) are natively supported with zero architectural changes. Because the Front-End rewinds any observable into the $t=0$ frame, a multi-Pauli measurement simply becomes a single `HeisenbergOp::MEASURE` with multiple bits set in its Pauli masks. The Middle-End and Back-End process a weight-10 parity check exactly as fast as a weight-1 Pauli measurement.

The Front-End emits raw Pauli strings to the HIR. It does **not** track the GF(2) basis $V$ or compute memory indices — those are Back-End responsibilities.

### 5.2 Responsibilities of the Back-End

The Back-End strictly owns tracking the GF(2) basis $V = [\beta_0, \beta_1, \dots]$ of the shifting superposition. For every HIR operation, the Back-End evaluates the spatial shift $\beta$ against $V$ to classify the operation and emit the appropriate opcode:

* **Expansion (OP_BRANCH):** If $\beta \notin \text{span}(V)$, the operation introduces a new dimension. The SVM's array will double.
* **Collision (OP_COLLIDE):** If $\beta \in \text{span}(V)$, map it to its exact integer representation `x_mask`. The SVM executes an in-place butterfly mix without expanding memory.
* **Scalar/Filter (OP_SCALAR_PHASE):** If $\beta = 0$, the operation is diagonal. The SVM applies phases or zeroes out entries without shifting.

The Back-End also computes `x_mask` (the exact memory offset into the SVM's coefficient array), `commutation_mask` (the pre-computed commutation bitmask), and moves the AG matrices from the Front-End's side-table into the `ConstantPool` constant pool.

---

## 6. Execution Semantics & The Performance Thesis

By splitting the architecture into a multi-level pipeline, UCC functionally solves the limitations of both standard dynamic simulators and strict static transpilers:

1. **Zero Thread Divergence:** In standard simulators, measurements and noise cause parallel execution threads to branch and warp dynamically. Because UCC's Compiler Back-End has pre-calculated all matrix updates and memory indices into static bytecode, 10,000 GPU threads can execute an OP_AG_PIVOT simultaneously in perfect lockstep, regardless of what measurement outcome they individually sampled.

2. **Elimination of the Memory Wall:** By utilizing Dominant Term Factoring in the HIR, the heaviest computational branch of the state vector is implicitly carried forward. It requires zero floating-point operations and zero memory writes at runtime.

3. **Geometric Gap Sampling:** Noise is injected purely on the integer `destab_signs` and `stab_signs` via the NoiseSchedule side-table. Because the SVM array `v[]` is untouched, the SVM jumps from quantum instruction to quantum instruction at maximum CPU clock speed, pausing only to instantly XOR pre-calculated noise bitmasks into the tracker.

4. **Deferred Normalization:** Transcendentals (`std::log`, `std::exp`, division) are entirely stripped from the GATE_LCU hot-loop. The SVM only divides by the $L_2$ norm *after* completing a full OP_MEASURE_MERGE, keeping numerical drift mathematically suppressed and avoiding costly pipeline stalls.

5. **Native Circuit Optimization:** The HIR enables algebraic optimization (fusion, cancellation, reordering) at the speed of bitwise arithmetic — without the overhead of DAG traversal or matrix algebra that traditional compilers require.

---

## 7. Roadmap: Control Flow & Advanced Edge Cases

To function as a drop-in execution backend for modern QEC toolchains, the pipeline must elegantly handle deep logical loops and dynamic feedback.

### 7.1 Deep Circuits and REPEAT Blocks

The Front-End's pre-computed GF(2) masks rely on a geometric reference frame that shifts whenever a Clifford gate is applied. A REPEAT block cannot be trivially mapped to a static SVM loop instruction, because Iteration 1 differs mathematically from Iteration 2. Naively unrolling millions of iterations will cause an Out-Of-Memory (OOM) error for the bytecode.

* **Primary Solution (JIT Streaming):** The Front-End will act as a Generator, yielding bytecode in chunks (e.g., 10,000 instructions). The SVM consumes the chunk, updates its state arrays, and immediately discards the bytecode. Bytecode memory footprint remains strictly $\mathcal{O}(1)$.
* **Advanced Solution (Limit Cycles & FRAME_SHIFT):** QEC circuits often form a "limit cycle" where the geometric frame returns to its exact starting state after one round of syndromes. The Front-End will hash its internal inverse tableau state. If the state matches the start of the loop, it emits a native OP_LOOP. If the frame drifts (e.g., stabilizers pick up minus signs but don't change topology), the compiler will emit an $\mathcal{O}(1)$ OP_FRAME_SHIFT instruction, allowing the SVM to realign its sign trackers mathematically without unrolling the drift.

### 7.2 Mid-Circuit Early Abort (Postselection)

For deep circuits like magic state cultivation factories, a failed classical
mid-circuit parity check means the shot is useless. Continuing to execute
expensive non-Clifford operations (OP_BRANCH expansions) on garbage shots
wastes significant compute.

**OP_POSTSELECT** provides mid-circuit early abort:

1. When the SVM hits this opcode, it evaluates the specified parity check over
   the measurement record.
2. If the parity fails (doesn’t match the expected outcome), the SVM immediately
   exits the bytecode execution loop for that shot.
3. The shot is marked as `discarded = true` in a boolean mask.
4. The SVM resets the memory arrays and proceeds to the next shot.

**N Total Shots:** The simulator always executes exactly $N$ *total* shots.
It does **not** auto-retry to fill a quota of successful shots. The `sample()`
function returns both the measurements and a boolean mask indicating which
shots passed postselection, allowing users to calculate acceptance yield:

```python
results, passed, seeds = ucc.sample(program, shots=10_000)
yield_rate = passed.sum() / len(passed)
```

### 7.3 Active Measurement Feedback & Stateful Noise

Operations that conditionally apply Pauli gates based on runtime measurements (e.g., `CX rec[-1] 1`, `CZ sweep[5] 2`) do *not* branch the Clifford frame; they only conditionally flip relative generator signs.

* **OP_CONDITIONAL_PAULI:** The Front-End pre-calculates the `destab_mask` and `stab_mask` for the target Pauli. The SVM conditionally XORs these masks into the sign tracker based on runtime `meas_record` flags.
* **Heralded Noise:** Handled by a future OP_HERALDED_NOISE instruction that updates sign trackers using pre-computed masks and explicitly appends a 0 or 1 herald flag to the measurement record.
* **Detectors:** Because valid QEC detectors mathematically commute with the logical state, their parity is strictly deterministic even inside massive non-Clifford superpositions. The SVM strictly executes a classical boolean XOR over its historical measurement record. A 1 definitively signals a physical Pauli error, requiring absolutely zero quantum evaluation.

### 7.4 Fundamentally Incompatible Operations

These operations mathematically violate the constraints of AOT stabilizer simulation and will explicitly throw compiler errors:

* **Anti-Hermitian Pauli Products (e.g., `MPP X1*Z1`):** These lack well-defined measurement semantics, break the mathematical closure of the stabilizer group, and would crash the Front-End's topological rewinding.
* **Classically Controlled Cliffords (e.g., `H rec[-1] 0`):** The AOT architecture requires a deterministic topological tableau to pre-compute phase interference. If a geometric Clifford gate is conditioned on a random measurement outcome, the geometric frame splits into multiple possible universes at compile time, making static pre-computation physically impossible. *(Note: Classically controlled Paulis are mathematically fine, see §7.2).*

---

## 8. Dual Use: Simulation & Compilation

A key advantage of the multi-level architecture is that the pipeline has two natural export points:

1. **Optimized HIR → Physical Routing:** The optimized HIR (after the Middle-End) is a flat list of algebraically independent Pauli operations with complex weights. This is exactly the input format required by Pauli-Based Computation (PBC) routing tools like tqec. The HIR can be exported directly without lowering to SVM bytecode, enabling UCC to serve as a **circuit optimization front-end** for physical quantum hardware.

2. **Bytecode → Monte Carlo Simulation:** The full pipeline through the Back-End and SVM enables hyper-fast Monte Carlo sampling for noise threshold estimation, QEC decoder benchmarking, and circuit validation.

This dual-use design means a single codebase serves both the compilation and simulation communities.

### 8.1 Multi-Pauli Measurement Support

Multi-Pauli measurements (e.g., `MPP X1*X2*Z3`) are natively supported with zero architectural changes. Because the Front-End rewinds any observable into the $t=0$ frame, a multi-Pauli measurement simply becomes a single `HeisenbergOp::MEASURE` with multiple bits set in its `destab_mask` and `stab_mask`. The Middle-End and Back-End process a weight-10 parity check exactly as fast as a weight-1 Pauli measurement — the symplectic inner product and memory indexing operations are identical.
