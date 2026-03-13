# UCC Implementation Plan: Cache-Blocked RISC Simulator

## Context & Goal

Simulation of large quantum circuits is ultimately bounded by the "Memory Wall."
UCC will defend against this using a three-tier architectural strategy:

1. **Minimize Memory Traffic (Systems Layer):** When the active rank k exceeds
   the L1 cache capacity (~10 qubits), invert the execution loop to process
   instructions in 16 KB chunks, avoiding L3/RAM thrashing.
2. **Maximize ALU Throughput (Compute Layer):** Once data is pinned in the L1
   cache, fuse RISC instructions into CISC matrices to maximize floating-point
   math per clock cycle.
3. **Minimize Active Support (Math Layer):** Delay and minimize the temporal
   footprint of 2^k using O(1) Symplectic Scheduling.

**Goal:** Overhaul the Virtual Machine and Optimizer to implement all three
tiers, allowing UCC to compete with general-purpose dense simulators on SU(N)
workloads (like Quantum Volume) while strictly retaining its pristine O(N) RISC
architecture for sparse QEC.

**Benchmark Target:** Match or exceed Qiskit-Aer's single-threaded QV-20
performance (~1.7s on the reference machine). The qsim target (~0.2s) requires
SoA layout and explicit SIMD intrinsics, which is out of scope for this plan.

**Supersedes:** `design/dense_sim.plan.md`. That plan proposed a separate
dual-engine architecture with a distinct `DenseSchrodingerState` and SoA memory
layout. This plan takes a fundamentally different approach: we keep the single
RISC VM and its existing `SchrodingerState` (including Pauli frame tracking),
adding `OP_CACHE_BLOCK` as a meta-instruction within the existing `execute()`.
This avoids code duplication and preserves the unified execution model for both
QEC and dense circuits.

---

## Phase 1: Cache-Blocked Execution (Minimizing RAM Traffic)

We introduce a loop-inversion meta-instruction. The VM will pull 16 KB chunks
of the statevector into the L1 cache, applying dozens of RISC instructions
before writing back to RAM.

### 1.1 VM Opcode: OP_CACHE_BLOCK

**File:** `src/ucc/backend/backend.h`

Add to the `Opcode` enum:

```cpp
enum class Opcode : uint8_t {
    // ... existing ...
    OP_CACHE_BLOCK,  // Execute inline micro-program on L1 cache chunks
    OP_ARRAY_U2,     // Fused CISC 2x2 matrix (Phase 2)
};
```

Add the block payload variant to the `Instruction` union:

```cpp
// Variant E: Cache Block header
struct {
    uint32_t num_instructions;  // Offset 8: length of the inline micro-program
    uint8_t _pad_e[20];         // Offset 12
} block;
```

The micro-program is literally the next `num_instructions` entries in the
bytecode vector. This is critical for Instruction Cache (I-cache) spatial
locality -- we do not chase pointers to a separate ConstantPool vector.

### 1.2 VM Execution of OP_CACHE_BLOCK

**File:** `src/ucc/svm/svm.cc`

Update `execute()` to handle `OP_CACHE_BLOCK`. The outer dispatch loop reads
the header, extracts the inline span, then iterates over statevector chunks:

```cpp
case Opcode::OP_CACHE_BLOCK: {
    uint32_t N = pc->block.num_instructions;
    std::span<const Instruction> micro_program(pc + 1, N);

    // L1 cache window = lowest B axes (e.g., bits 0-9 = 1024 complex doubles = 16KB)
    constexpr uint32_t B = 10;
    uint64_t chunk_size = 1ULL << B;
    uint64_t num_chunks = state.v_size() >> B;

    // Process each chunk sequentially. Each chunk gets a local copy of the
    // Pauli frame so that frame-dependent math (OP_ARRAY_U2) resolves
    // correctly. Frame updates are deterministic, so every chunk computes
    // identical frame transitions.
    for (uint64_t c = 0; c < num_chunks; ++c) {
        uint64_t base = c * chunk_size;
        PauliBitMask local_px = state.p_x;
        PauliBitMask local_pz = state.p_z;
        // Execute micro_program on v[base..base+chunk_size)
        // using local_px, local_pz for frame-dependent math
    }

    // After the chunk loop: replay frame updates once (O(N) micro-program
    // scan) to update the global state.p_x, state.p_z, and state.gamma_.

    // Fast-forward past the inline micro-program
    pc += N;
    break;
}
```

The CPU prefetcher streams the bytecode perfectly since the micro-program is
contiguous in memory.

**Threading is deferred.** The single-threaded chunk loop is architecturally
ready for OpenMP parallelism (each chunk is independent), but threading is
out of scope for this plan. See the Future Work section.

### 1.3 Strict Block Boundaries

Stochastic instructions (e.g., `OP_NOISE`) must never appear inside a cache
block. Even in the current single-threaded implementation, the block
execution model replays the micro-program per-chunk with a local Pauli frame
copy, so non-deterministic instructions would corrupt frame consistency
across chunks. When threading is added later, this constraint becomes even
more critical: different threads would roll different random numbers,
desynchronizing the statevector -- a **fatal concurrency violation**.

The `CacheBlockPass` must treat certain instructions as **Basic Block
Terminators**:

**Allowed inside OP_CACHE_BLOCK:**
- `OP_ARRAY_H`, `OP_ARRAY_S`, `OP_ARRAY_S_DAG`, `OP_ARRAY_CZ`
- `OP_PHASE_ROT`, `OP_PHASE_T`, `OP_PHASE_T_DAG`
- `OP_ARRAY_U2` (Phase 2)
- All `OP_FRAME_*` ops
- `OP_ARRAY_CNOT` -- ONLY if its target axis is within the cache window

**Forces Block Closure (Terminators):**
- `OP_NOISE`, `OP_NOISE_BLOCK`, `OP_READOUT_NOISE`
- `OP_MEAS_*` (all measurement variants)
- `OP_EXPAND`, `OP_EXPAND_T`, `OP_EXPAND_T_DAG`, `OP_EXPAND_ROT`
- `OP_APPLY_PAULI`
- `OP_ARRAY_SWAP` (emitted by the Belady router to bring axes into the window)

When the compiler hits a Terminator, it closes the current `OP_CACHE_BLOCK`,
emits the Terminator sequentially into the global bytecode, and opens a new
cache block for subsequent unitary instructions.

**Why this is fine for performance:** Quantum Volume circuits have zero noise
and no mid-circuit measurements, so they form massive, uninterrupted cache
blocks. QEC circuits have noise, but it occurs at the end of the syndrome cycle
after the heavy unitary math is done.

### 1.4 Clairvoyant Oracle Routing (CacheBlockPass)

**File:** `src/ucc/optimizer/cache_block_pass.cc`

This bytecode optimization pass transforms raw RISC bytecode into
cache-blocked form. **It is NOT included in the default pass managers.**
Users must explicitly construct and run it (e.g., via a custom
`BytecodePassManager`). This allows us to validate correctness and
performance incrementally before deciding on automatic application
heuristics.

1. Define the L1 cache window as the lowest B virtual axes (bits 0 through
   B-1, e.g., B=10 for 16 KB).
2. Scan the compiled bytecode. Diagonal operations (`OP_PHASE_ROT`,
   `OP_ARRAY_CZ`, `OP_ARRAY_S`) never force a cache miss. Target axes of
   `OP_ARRAY_H` and `OP_ARRAY_CNOT` (target only) do.
3. When a required axis is outside the window, use **Belady's Optimal
   Algorithm**. Because this is an AOT compiler, scan ahead in the static
   instruction vector to find the currently cached axis that is needed
   furthest in the future.
4. Emit `OP_ARRAY_SWAP` to evict it and bring the needed axis into the
   window.
5. Bundle the continuous safe segment into an `OP_CACHE_BLOCK`.

The cache estimator (`paper/qv_benchmark/cache_estimator.py`) already
demonstrated the feasibility: for QV-20 with cache_size=10, Belady routing
needs only 91 swaps (1% overhead) while reducing strict L1 dependencies from
~8900 full array sweeps to ~3800 L1-only sweeps.

Example Python usage (explicit opt-in):

```python
import ucc

prog = ucc.compile(stim_text,
    hir_passes=ucc.default_hir_pass_manager(),
    bytecode_passes=ucc.default_bytecode_pass_manager())

# Explicit cache-blocking pass (not in defaults)
cache_pass = ucc.CacheBlockPass()
cache_pass.run(prog)

result = ucc.sample(prog, shots=1, seed=42)
```

### 1.5 QV Benchmark Checkpoint

After Phase 1, re-run the QV-20 benchmark to measure the improvement from
cache blocking alone. Expected: significant reduction in wall-clock time due
to elimination of L3/RAM thrashing on the ~8MB statevector.

---

## Phase 2: Single-Axis CISC Fusion (Maximizing ALU Throughput)

Reduce the multi-instruction dispatch penalty inside the L1 cache block by
fusing sequences of single-axis operations into dense 2x2 matrices.

### 2.1 SingleAxisFusionPass

**File:** `src/ucc/optimizer/single_axis_fusion_pass.cc`

Identify sequences of instructions within a cache block that act on the exact
same axis (e.g., U3 decompositions like Rz -> H -> Rz -> H -> Rz).

Because the math dynamically depends on the local Pauli frame bits, the
compiler must pre-calculate the dense 2x2 complex matrices for **all four
possible states** of the Pauli frame at that axis (I, X, Z, Y) at compile
time.

Store the 4 matrices plus frame transition metadata in the ConstantPool.
Emit `OP_ARRAY_U2`. At runtime, the VM inspects its local `p_x`/`p_z` bits,
grabs the correct matrix, and sweeps the 16 KB chunk exactly once.

### 2.2 FusedU2Node Constant Pool Structure

**File:** `src/ucc/backend/backend.h`

```cpp
// A 2-bit state representing the Pauli frame on a single axis:
//   (p_z << 1) | p_x
//   0 = I, 1 = X, 2 = Z, 3 = Y

struct FusedU2Node {
    // 4 matrices (row-major 2x2), indexed by the 2-bit input frame state.
    std::complex<double> matrices[4][4];

    // Global phase accumulated during the fused sequence, per input state.
    std::complex<double> gamma_multipliers[4];

    // Resulting 2-bit (p_z << 1) | p_x state after the sequence finishes.
    uint8_t out_states[4];
};

struct ConstantPool {
    // ... existing fields ...
    std::vector<FusedU2Node> fused_u2_nodes;
};
```

### 2.3 OP_ARRAY_U2 Instruction Encoding

Add to the `Instruction` union:

```cpp
// Variant F: Fused U2 payload
struct {
    uint32_t cp_idx;     // Offset 8 (Index into ConstantPool::fused_u2_nodes)
    uint8_t _pad_f[20];  // Offset 12
} u2;
```

`axis_1` (offset 4) holds the target virtual axis.

### 2.4 Runtime Execution

Inside the chunk execution loop, the thread resolves the matrix in O(1)
without branching on the Pauli frame:

```cpp
// 1. Index the FSM based on the thread-local frame
uint8_t in_state = (local_pz.bit_get(axis) << 1) | local_px.bit_get(axis);

// 2. Grab the exact pre-computed matrix for this frame state
const std::complex<double>* mat = pool.fused_u2_nodes[cp_idx].matrices[in_state];

// 3. Apply mat to the 16KB array chunk (butterfly sweep)

// 4. Fast-forward the thread-local frame
uint8_t out_state = pool.fused_u2_nodes[cp_idx].out_states[in_state];
local_px.bit_set(axis, out_state & 1);
local_pz.bit_set(axis, (out_state >> 1) & 1);
```

After the chunk loop, the frame update pass performs the same O(1) FSM
lookup to update `state.p_x`, `state.p_z`, and multiply `state.gamma_` by
the `gamma_multipliers`. This keeps the VM dispatch loop completely
branchless regarding the Pauli frame and shifts all heavy lifting to the AOT
compiler.

### 2.5 QV Benchmark Checkpoint

After Phase 2, re-run the QV-20 benchmark to measure the cumulative
improvement from cache blocking + CISC fusion. Expected: substantial further
reduction due to eliminating per-instruction dispatch overhead inside the
hot L1 loop.

---

## Phase 3: Symplectic Instruction Scheduling (Time-Travel)

Since Pauli compression is already mathematically optimal, we squeeze the
active dimension k by minimizing the lifetime of active axes.

### 3.1 SchedulerPass

**File:** `src/ucc/optimizer/scheduler_pass.cc`

Create a new HIR optimization pass that uses the O(1) symplectic inner
product (`anti_commute`) to safely reorder instructions across the entire
circuit:

- **Lazy Expansion:** Delay non-Clifford operations that target dormant axes
  (which trigger `OP_EXPAND`) by sliding them as far into the future as
  legally possible.
- **Eager Compaction:** Pull forward active measurements (which halve the
  array via `OP_MEAS_ACTIVE_*`) by sliding them as far into the past as
  legally possible.

This minimizes the integral of memory usage over the simulation's runtime.

**Note:** This phase has the least impact on QV circuits specifically, since QV
ramps to peak_rank=N immediately with no mid-circuit measurements to pull
forward. The primary beneficiaries are structured circuits with interleaved
expansions and measurements (e.g., QEC syndrome extraction rounds).

### 3.2 QV Benchmark Checkpoint

After Phase 3, re-run the QV benchmark suite. For QV circuits, expect minimal
change. For QEC-style circuits, expect significant memory reduction.

---

## Phase 4: Benchmarking & Validation

Validate correctness and demonstrate the elimination of the memory wall.

### 4.1 Benchmark Target: QV-20

A dense 20-qubit Quantum Volume circuit tested via the existing
`paper/qv_benchmark/` suite.

- **Routing Goal:** The CacheBlockPass must reduce full 16 MB array sweeps
  from ~8,900 (current VM) to fewer than ~200 (a >96% reduction in RAM
  traffic).
- **Runtime Goal:** Execution runtime must drop from ~6.0 seconds to match
  or exceed Qiskit-Aer's performance (~1.7s) for equivalent statevector
  dimensions on a single thread.
- **Correctness Goal:** The final statevector must remain mathematically
  exact. `OP_NOISE` distributions must be verified to continue functioning
  properly under the block-terminator discipline (noise ops never appear
  inside cache blocks).

### 4.2 Regression Testing

All existing C++ Catch2 tests and Python integration tests must continue to
pass. The cache-blocked execution path must produce bit-identical results
to the existing sequential VM for deterministic (noiseless) circuits.

For stochastic circuits, the PRNG stream must remain identical (cache
blocking does not change the instruction ordering of noise/measurement ops,
only the execution strategy of the deterministic unitary segments between
them).

### 4.3 New Tests

- **C++ Micro-Tests:** Handcraft small cache blocks (cache_bits=2, 4-qubit
  array) and verify chunk boundaries, frame consistency, and exact
  statevector output.
- **C++ FusedU2 Tests:** Verify the 4-matrix FSM produces correct results
  for all Pauli frame input states.
- **Python Cross-Validation:** Compare UCC with and without cache blocking
  against Qiskit-Aer statevectors on QV-4 through QV-10.

---

## Architectural Invariants (Reiterated)

All existing invariants from `AGENTS.md` remain in force:

- **32-Byte Instruction:** `static_assert(sizeof(Instruction) == 32)`
- **Single Memory Allocation:** `SchrodingerState::v_` allocated once at
  `peak_rank`; no dynamic resizing in the VM hot loop.
- **Stim is Immutable:** No forking or patching Stim source.
- **Deterministic RNG:** `(rng() >> 11) * 0x1.0p-53`, no
  `std::uniform_real_distribution`.
- **No Global Topology in the VM:** All global compression remains AOT.
- **Contiguous Array Compaction:** SWAP-to-top before measurement.

Additionally:

- **No OP_NOISE inside OP_CACHE_BLOCK:** Stochastic instructions are strict
  block terminators to prevent Pauli frame corruption across chunks (and
  future thread divergence).
- **Inline Micro-Programs:** The cache block's instruction sequence lives
  contiguously in the bytecode vector for I-cache locality. No pointer
  chasing to ConstantPool.
- **CacheBlockPass is opt-in:** Not included in default pass managers.
  Must be explicitly constructed and run by the user.

---

## Future Work (Out of Scope)

- **OpenMP Threading:** The single-threaded chunk loop in `OP_CACHE_BLOCK`
  is architecturally ready for `#pragma omp parallel for schedule(static)`.
  Each chunk is independent (local Pauli frame copy, no shared mutable
  state). Adding threading is a standalone follow-up once single-threaded
  correctness and performance are validated.
- **Automatic CacheBlockPass Heuristics:** Decide when to automatically
  apply cache blocking (e.g., when `peak_rank > L1_bits` and the circuit
  has no mid-circuit noise). Add to default pass manager once the
  cost/benefit threshold is understood.
- **SoA Layout + SIMD:** Achieving qsim-level performance (~0.2s for QV-20)
  likely requires Structure-of-Arrays memory layout and explicit AVX2/AVX-512
  intrinsics. This is a separate architectural decision.
