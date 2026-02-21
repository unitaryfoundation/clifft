# UCC — Data Structures & Execution Semantics

## Abstract

UCC's multi-level pipeline produces two distinct data representations:

1. **The Heisenberg IR (HIR)** — an abstract, order-independent list of Pauli-string
   operations emitted by the Front-End and consumed by the Middle-End Optimizer.
2. **The Execution Bytecode** — a hardware-aligned, cache-optimized instruction
   stream emitted by the Compiler Back-End and executed by the execution engine.

This document defines the C++ data structures for both representations, the
constant pool, the per-shot execution state, and the runtime execution semantics.
All structures target the initial ≤64-qubit path using inline `uint64_t` masks.

---

## 1. The Heisenberg IR (HIR)

The HIR is the output of the Front-End and the input to the Middle-End Optimizer.
It is a flat, algebraically independent list of operations defined purely by their
64-bit Pauli strings. The HIR contains **no SVM-specific memory routing** (no
`x_mask`, no `commutation_mask`, no array indices) — it is a pure mathematical
representation of the circuit's non-Clifford and stochastic content.

### 1.1 HeisenbergOp

```cpp
#include <complex>
#include <cstdint>
#include <vector>

enum class OpType : uint8_t {
    T_GATE,       // T gate (π/8 phase)
    T_DAG_GATE,   // T† gate (inverse π/8 phase)
    GATE,         // Generic non-Clifford gate (CCZ, arbitrary rotations via LCU)
    MEASURE,      // Destructive state collapse
    NOISE,        // Stochastic Pauli error channel
    DETECTOR      // Classical QEC parity check
};

// The Abstract Heisenberg IR — no memory routing, no SVM semantics.
struct HeisenbergOp {
    OpType type;

    // The Rewound Pauli String (the topological geometry at t=0)
    // Using stim::simd_bits<W> enables scaling to ~400 qubits (AVX-512).
    // Design note: This puts masks on the heap and introduces minor AOT
    // allocation overhead, which may slightly impact optimizer cache locality
    // compared to inline uint64_t structs. However, it trivially enables
    // scaling to AVX-512 widths for the paper prototype. SVM runtime
    // performance is unaffected because the SVM bytecode still uses the
    // 32-bit index template approach (see §7) to preserve 32-byte L1 cache
    // alignment.
    stim::simd_bits<W> destab_mask;   // Pauli X-bits
    stim::simd_bits<W> stab_mask;     // Pauli Z-bits

    // Payload (context-dependent based on OpType)
    union {
        // T_GATE / T_DAG_GATE: no payload needed (weight is implicit)
        // The is_dagger flag is encoded in the OpType itself.

        // GATE (generic LCU): complex weight for the spawned branch
        // (dominant term already factored out). Stored as bare doubles
        // to avoid std::complex<double> in a union (non-trivial ctor
        // is UB-adjacent).
        struct {
            double weight_re;   // Real part of relative branch weight
            double weight_im;   // Imaginary part
        } gate;

        // NOISE: per-channel error probability
        double noise_prob;

        // MEASURE: AG matrix index + reference outcome + classical target
        struct {
            uint32_t ag_matrix_idx;     // Index into HirModule::ag_matrices
            uint8_t  ag_ref_outcome;    // Reference outcome (0 or 1)
            uint32_t classical_target;  // Measurement record index
        } measure;

        // DETECTOR: classical target index
        uint32_t detector_target;
    };
};

// Pre-computed GF(2) transformation matrices for Aaronson-Gottesman pivots.
// Stored in HirModule during Front-End tracing, then moved to ConstantPool
// by the Back-End. Keeping these in a side-table prevents bloating the HIR
// and slowing down the optimizer.
struct GF2Matrix;

// The output of the Front-End / input to the Middle-End
struct HirModule {
    std::vector<HeisenbergOp> ops;
    std::vector<GF2Matrix> ag_matrices;  // Side-table for AG pivot matrices
    uint32_t num_qubits;

    // Global weight accumulator for HIR export to physical routing tools.
    // Initialized by the Front-End, which accumulates the dominant terms
    // factored out of each gate (e.g., exp(iπ/8)·cos(π/8) for T-gates).
    // Updated by the optimizer during fusion (see §2.2). The SVM simulator
    // ignores this (deferred normalization), but it is essential for correct
    // amplitude tracking in PBC export.
    std::complex<double> global_weight = {1.0, 0.0};
};
```

### 1.2 Design Rationale

**Explicit T-Gate OpTypes:** The HIR uses explicit `T_GATE` and `T_DAG_GATE`
OpTypes instead of relying on floating-point weight comparison. This eliminates
the brittle logic of checking if weights exactly equal $\pm i \cdot \tan(\pi/8)$.
The Back-End uses these explicit enums to safely and deterministically route to
fast-path opcodes (OP_BRANCH, OP_COLLIDE, OP_SCALAR_PHASE) that hardcode the
weight. Generic `GATE` operations (CCZ, arbitrary rotations) go through the LCU
opcodes with constant pool references.

Importantly, the **optimizer remains gate-agnostic**: commutation, fusion, and
cancellation logic depends only on `destab_mask`/`stab_mask`. The optimizer
never needs to distinguish T from other gates — it treats all gate OpTypes
identically based on their Pauli masks.

**Bare doubles, not std::complex:** The `gate` payload uses two bare `double`
fields instead of `std::complex<double>`. The C++ standard does not guarantee
that `std::complex<double>` is trivially constructible, and placing a type with
a potentially non-trivial constructor in a raw `union` is UB-adjacent. Bare
doubles are safe, trivial, and identical in memory layout.

**Dominant Term Factoring:** For non-Clifford gates (like $T \propto I - i(\sqrt{2}-1)Z$),
the Front-End automatically factors out the dominant $I$ term. The $I$ path gets
a relative weight of 1.0 (zero runtime FLOPs), and the payload stores only the
relative complex scalar for the spawned branch. This structurally prevents
IEEE-754 numerical instability by keeping all relative weights $\le 1.0$.
For T: `weight_re = 0.0, weight_im = TAN_PI_8` (or `-TAN_PI_8` for T†).

**AG Matrix Side-Table:** The Front-End computes AG pivot matrices whenever a
measurement collapses the tableau state. To prevent bloating the HIR and slowing
down the optimizer, these matrices are stored in a side-table (`ag_matrices`)
owned by the `HirModule`. Each `HeisenbergOp::MEASURE` stores only a
`uint32_t ag_matrix_idx` that points into this array. The Back-End later moves
the entire side-table into the `ConstantPool` constant pool.

**Noise as HIR Citizen:** Noise operations live in the HIR (not just in the SVM)
because the Middle-End optimizer needs them as **barriers** — a gate cannot
commute past an anti-commuting noise channel without conjugating the error into
a coherent rotation, altering the physics. Gates that *commute* with a noise
channel's Pauli string may freely pass it (see §2.3).

**simd_bits for 400-Qubit Scaling:** The Pauli masks use `stim::simd_bits<W>`
to support the ~400-qubit stretch goal for magic state cultivation simulation
using AVX-512. While this moves masks to the heap and introduces minor AOT
allocation overhead (potentially impacting optimizer cache locality), it
trivially enables scaling beyond 64 qubits. The SVM runtime performance remains
completely unaffected because bytecode uses the 32-bit index template approach.

---

## 2. The Middle-End Optimizer

The optimizer operates entirely on the HIR using $\mathcal{O}(1)$ bitwise
arithmetic. No DAG traversal, no matrix algebra.

### 2.1 Commutation Check

```cpp
bool commutes(const HeisenbergOp& A, const HeisenbergOp& B) {
    return (std::popcount((A.destab_mask & B.stab_mask) ^
                          (A.stab_mask & B.destab_mask)) % 2) == 0;
}
```

### 2.2 Fusion and Cancellation

If two gate nodes (`T_GATE`, `T_DAG_GATE`, or `GATE`) share the exact same
`destab_mask` and `stab_mask`, they target the exact same topological axis
and can be algebraically combined.

**Dominant Term Factoring Algebra:** Because UCC uses dominant term factoring,
every non-Clifford gate is parameterized as $I + cP$, where the Identity branch
has an implicit relative weight of $1.0$ and $c$ is the complex relative weight
of the Pauli branch stored in the `gate` payload.

When multiplying two such gates on the exact same Pauli axis $P$:

$$(I + c_1 P)(I + c_2 P) = I + c_1 P + c_2 P + c_1 c_2 P^2$$

Since $P^2 = I$ for all Pauli strings, this simplifies to:

$$(1 + c_1 c_2)I + (c_1 + c_2)P$$

To maintain the framework's strict invariant that the Identity branch has a
relative weight of exactly $1.0$, we factor out $(1 + c_1 c_2)$:

$$(1 + c_1 c_2)\left[I + \frac{c_1 + c_2}{1 + c_1 c_2}P\right]$$

The **new relative weight** is therefore:

$$c_{\text{fused}} = \frac{c_1 + c_2}{1 + c_1 c_2}$$

The factored-out coefficient $(1 + c_1 c_2)$ becomes a **global scalar**. While
the SVM simulator defers normalization and can ignore this, the compiler tracks
it in a `global_weight` accumulator so the HIR can be safely exported to physical
routing tools (like tqec or PBC).

**Fusion Outcomes:** The optimizer classifies the result into three cases:

- **Cancellation** ($c_1 + c_2 = 0$): The Pauli branch vanishes entirely
  (e.g., $T \cdot T^\dagger = I$). Both nodes are deleted from the HIR.
  The global weight accumulates $(1 + c_1 c_2)$.

- **Clifford Formation** ($1 + c_1 c_2 = 0$): The Identity branch vanishes,
  leaving a pure Pauli/Clifford rotation (e.g., $T \cdot T = S$). The node is
  temporarily flagged for fold-back. The global weight accumulates $(c_1 + c_2)$.

- **Standard Fusion** (otherwise): Replace the pair with a single `GATE` node
  containing the new relative weight $c_{\text{fused}}$. The global weight
  accumulates $(1 + c_1 c_2)$.

### 2.3 The Optimizer Sweep Algorithm

The optimizer performs a DAG-free $\mathcal{O}(n \cdot k)$ sweep over the flat
HIR vector, where $n$ is the number of operations and $k$ is the average
look-ahead distance:

1. **Iterate:** Loop sequentially through `std::vector<HeisenbergOp>`.

2. **Look Ahead & Commute:** For each gate, scan forward looking for another
   gate with identical `destab_mask` and `stab_mask`. Check if the gate can
   commute past every intermediate operation using the $\mathcal{O}(1)$ bitwise
   symplectic inner product.

3. **Respect Barriers:** Halt the look-ahead if it encounters an anti-commuting
   gate, or a stochastic barrier (`MEASURE` or `NOISE` node) that the gate
   anti-commutes with.

4. **Fuse & Track:** If a match is found and the path is clear, mathematically
   combine them using the algebra above and multiply the extracted scalar into
   the running `global_weight`.

5. **Compact the Vector:**
   - *Cancellation:* Delete both nodes.
   - *Clifford Formation:* Flag `needs_fold_back = true`, mark the node as a
     pure Pauli, but leave it in the IR. Continue the sweep to exhaustively
     fuse as much of the circuit as possible before the heavy Front-End re-run.
   - *Standard Fusion:* Replace the pair with a single node containing the new
     relative weight.

6. **Fold-Back:** Only after the entire vector has been exhaustively swept, if
   `needs_fold_back` was flagged, feed the optimized HIR back into the Front-End
   to absorb the newly formed Cliffords into the topological tableau and emit a
   fresh, fully-reduced HIR.

### 2.4 Fusion Implementation Sketch

```cpp
#include <complex>
#include <cmath>

enum class FuseResult {
    FUSED,             // Standard fusion, new relative weight computed
    CANCELLED,         // Cancelled to Identity (e.g., T · T†)
    FORMED_CLIFFORD    // Denominator vanished, formed a pure Pauli (e.g., T · T = S)
};

// Helper to extract the complex relative weight from an op
std::complex<double> get_weight(const HeisenbergOp& op) {
    if (op.type == OpType::T_GATE) return {0.0, TAN_PI_8};
    if (op.type == OpType::T_DAG_GATE) return {0.0, -TAN_PI_8};
    if (op.type == OpType::GATE) return {op.gate.weight_re, op.gate.weight_im};
    return {0.0, 0.0};
}

// Fuses op2 into op1, assuming they share the exact same Pauli masks.
FuseResult fuse_ops(HeisenbergOp& op1,
                    const HeisenbergOp& op2,
                    std::complex<double>& global_weight)
{
    std::complex<double> c1 = get_weight(op1);
    std::complex<double> c2 = get_weight(op2);

    // Expand (I + c1*P)(I + c2*P) = (1 + c1*c2)I + (c1 + c2)P
    std::complex<double> i_coeff = 1.0 + c1 * c2;
    std::complex<double> p_coeff = c1 + c2;

    const double EPSILON = 1e-12;

    // Case 1: Cancellation (e.g., T · T† = I)
    // The Pauli branch vanishes.
    if (std::abs(p_coeff) < EPSILON) {
        global_weight *= i_coeff;
        return FuseResult::CANCELLED;
    }

    // Case 2: Pure Pauli / Clifford formed (e.g., T · T = S)
    // The Identity branch vanishes.
    if (std::abs(i_coeff) < EPSILON) {
        global_weight *= p_coeff;
        return FuseResult::FORMED_CLIFFORD;
    }

    // Case 3: Standard Fusion
    // Normalize so the Identity branch has a relative weight of exactly 1.0
    std::complex<double> new_weight = p_coeff / i_coeff;
    global_weight *= i_coeff;

    // The operation is now a generic non-Clifford with the new weight
    op1.type = OpType::GATE;
    op1.gate.weight_re = new_weight.real();
    op1.gate.weight_im = new_weight.imag();

    return FuseResult::FUSED;
}
```

### 2.5 Barriers

While gate nodes can be freely reordered by commutation, the optimizer respects
barrier semantics for `MEASURE` and `NOISE` nodes:

- **MEASURE** nodes destroy superposition. A gate cannot commute past a
  measurement unless their Pauli strings perfectly commute (sharing an eigenbasis).
- **NOISE** nodes define the physical error model. A gate cannot commute past
  a noise channel whose Pauli string **anti-commutes** with the gate's Pauli
  string — doing so would conjugate the error into a coherent rotation, altering
  the physics. Gates that *commute* with the noise channel's Pauli string may
  freely pass it.

Both barrier types use the same symplectic inner product check (§2.1) to
determine whether a specific gate is blocked.

---

## 3. The Constant Pool (ConstantPool)

To ensure the Instruction struct stays at exactly 32 bytes, all heavy,
variable-length, or rarely accessed data is extracted by the Back-End into a
global constant pool. Bytecode instructions store 32-bit indices into this pool.

```cpp
// Pre-computed GF(2) transformation matrices for Aaronson-Gottesman pivots.
// Each matrix encodes the exact sign-tracker update for a measurement that
// required Gaussian elimination at compile time.
//
// These matrices are computed by the Front-End (which owns the TableauSimulator)
// and initially stored in HirModule::ag_matrices. The Back-End moves them
// into ConstantPool::ag_matrices during code generation.
struct GF2Matrix {
    uint64_t destab_cols_x[64];
    uint64_t destab_cols_z[64];
    uint64_t stab_cols_x[64];
    uint64_t stab_cols_z[64];
};

// Extracted floating-point components for generic LCU decompositions.
// The Back-End pre-computes these from HeisenbergOp::lcu_weight.
struct LCUData {
    double weight;              // |c_branch / c_dominant|
    std::complex<double> c_dom; // Phase of the dominant term
};

// A single Pauli error channel within a noise site.
// Multi-Pauli channels (e.g., DEPOLARIZE1 with X/Y/Z errors) are represented
// as multiple NoiseChannel entries at the same program counter.
struct NoiseChannel {
    double probability;         // Per-channel error probability
    uint64_t destab_mask;       // Pre-computed XOR mask for destab_signs
    uint64_t stab_mask;         // Pre-computed XOR mask for stab_signs
};

// A noise site groups all error channels that apply at a single bytecode PC.
// DEPOLARIZE1 → 3 channels (X, Y, Z), DEPOLARIZE2 → 15 channels, etc.
struct NoiseSite {
    uint32_t pc;                           // Bytecode program counter
    std::vector<NoiseChannel> channels;    // One or more error channels
    double total_probability;              // Sum of channel probabilities
                                           // (pre-computed for gap sampling)
};

// Global read-only payloads for the simulation, allocated once by the Back-End.
struct ConstantPool {
    std::vector<GF2Matrix>              ag_matrices;
    std::vector<std::vector<uint32_t>>  detector_targets;
    std::vector<LCUData>                lcu_pool;
    std::vector<NoiseSite>              noise_schedule;  // Sorted by pc
};
```

### 3.1 NoiseSchedule Design

The `noise_schedule` is sorted by program counter. At runtime, the SVM uses
**geometric gap sampling** over this schedule:

1. Using `total_probability` across all remaining sites, sample a geometric
   distance $\Delta$ to the next error event.
2. Execute $\Delta$ purely quantum bytecode instructions at full speed (zero
   branch checks, zero RNG calls).
3. At the sampled site, select *which* channel fired (weighted by individual
   `probability` within the site). XOR the selected channel's
   `destab_mask`/`stab_mask` into the sign tracker.
4. Resume from step 1.

For single-Pauli noise (X_ERROR, Z_ERROR), the site has exactly one channel
and step 3 is trivial. For multi-Pauli noise (DEPOLARIZE1 with 3 channels,
DEPOLARIZE2 with 15), step 3 requires a single weighted random selection
among the channels.

This keeps noise entirely out of the bytecode hot-loop. The SVM's instruction
stream is 100% quantum operations.

---

## 4. The 32-Byte Instruction Bytecode

The core SVM bytecode is a C-style union restricted to **exactly 32 bytes**.
This guarantees that exactly two instructions fit perfectly into a single
64-byte L1 cache line, maximizing hardware prefetching and preventing GPU
Constant Memory bloat.

```cpp
enum class Opcode : uint8_t {
    // T-Gate Fast Paths (hardcoded tan(π/8) weight)
    OP_BRANCH,          // New dimension: array doubles
    OP_COLLIDE,         // Existing dimension: in-place butterfly
    OP_SCALAR_PHASE,    // β=0: diagonal phase application

    // Generic LCU (arbitrary non-Clifford via constant pool)
    OP_BRANCH_LCU,
    OP_COLLIDE_LCU,
    OP_SCALAR_PHASE_LCU,

    // Measurement & State Collapse
    OP_MEASURE_MERGE,         // Anti-commuting measurement: sample + shrink array
    OP_MEASURE_FILTER,        // Commuting measurement: sample + zero half the array
    OP_MEASURE_DETERMINISTIC, // Deterministic outcome: no array change, sign-track only

    // Topology & Classical Logic
    OP_AG_PIVOT,        // Aaronson-Gottesman sign-tracker update
    OP_INDEX_CNOT,      // Basis relabeling within the GF(2) space
    OP_DETECTOR,        // Classical parity check over measurement record

    // Control Flow
    OP_POSTSELECT       // Mid-circuit early abort (see §6.13)
};

struct alignas(32) Instruction {

    // --- Offset 0: Common Header (Exactly 8 Bytes) ---
    Opcode opcode;            // Offset 0 (1 byte)  — union discriminant
    uint8_t base_phase_idx;   // Offset 1 (1 byte)  — maps to {1, i, -1, -i}
    bool is_dagger;           // Offset 2 (1 byte)
    uint8_t ag_ref_outcome;   // Offset 3 (1 byte)
    uint32_t commutation_mask;    // Offset 4 (4 bytes) — pre-computed interference

    // --- Offset 8: Mutually Exclusive Payload (Exactly 24 Bytes) ---
    union {
        // Variant A: T-Gate Fast Path (BRANCH / COLLIDE / SCALAR_PHASE)
        struct {
            uint64_t destab_mask;   // Offset 8  (8 bytes)
            uint64_t stab_mask;     // Offset 16 (8 bytes)
            uint32_t x_mask;        // Offset 24 (4 bytes)
            // 4 bytes padding
        } branch;

        // Variant B: Generic LCU (BRANCH_LCU / COLLIDE_LCU / SCALAR_PHASE_LCU)
        struct {
            uint64_t destab_mask;   // Offset 8
            uint64_t stab_mask;     // Offset 16
            uint32_t x_mask;        // Offset 24
            uint32_t lcu_pool_idx;  // Offset 28 — index into ConstantPool::lcu_pool
        } lcu;

        // Variant C: Measurement / Array Resizing
        struct {
            uint64_t destab_mask;   // Offset 8
            uint64_t stab_mask;     // Offset 16
            uint32_t x_mask;        // Offset 24
            uint32_t bit_index;     // Offset 28 — dimension index for shrinking
        } meas;

        // Variant D: Meta / Classical / AG Pivot
        struct {
            uint32_t payload_idx;   // Offset 8  — index into ag_matrices or detector_targets
            uint32_t ag_stab_slot;  // Offset 12
        } meta;
    };
};

static_assert(sizeof(Instruction) == 32, "Instruction must be exactly 32 bytes");
```

### 4.1 Opcode Classification

| Opcode | HIR Source | Array Effect | Constant Pool |
|--------|-----------|-------------|---------------|
| `OP_BRANCH` | `T_GATE` / `T_DAG_GATE`, $\beta \notin \text{span}(V)$ | Doubles | None (hardcoded `tan(π/8)`) |
| `OP_COLLIDE` | `T_GATE` / `T_DAG_GATE`, $\beta \in \text{span}(V)$ | Butterfly mix | None |
| `OP_SCALAR_PHASE` | `T_GATE` / `T_DAG_GATE`, $\beta = 0$ | Diagonal phase | None |
| `OP_BRANCH_LCU` | `GATE` (generic LCU), $\beta \notin \text{span}(V)$ | Doubles | `lcu_pool[idx]` |
| `OP_COLLIDE_LCU` | `GATE` (generic LCU), $\beta \in \text{span}(V)$ | Butterfly mix | `lcu_pool[idx]` |
| `OP_SCALAR_PHASE_LCU` | `GATE` (generic LCU), $\beta = 0$ | Diagonal phase | `lcu_pool[idx]` |
| `OP_MEASURE_MERGE` | `MEASURE`, anti-commuting | Halves | None |
| `OP_MEASURE_FILTER` | `MEASURE`, commuting ($\beta \in \text{span}(V)$) | Zeros half | None |
| `OP_MEASURE_DETERMINISTIC` | `MEASURE`, deterministic ($\beta = 0$, eigenstate) | None | None |
| `OP_AG_PIVOT` | `MEASURE` (random, requires basis change) | None (signs only) | `ag_matrices[idx]` |
| `OP_INDEX_CNOT` | Internal basis mgmt | None | None |
| `OP_DETECTOR` | `DETECTOR` | None | `detector_targets[idx]` |
| `OP_POSTSELECT` | `POSTSELECT` / `ASSERT_DETECTOR` | None (early exit) | `detector_targets[idx]` |

The Back-End uses the explicit `T_GATE` and `T_DAG_GATE` OpTypes to
deterministically route to fast-path opcodes (OP_BRANCH, OP_COLLIDE,
OP_SCALAR_PHASE). Generic `GATE` operations are lowered to LCU opcodes
with constant pool references.

---

## 5. The Execution State (SchrodingerState)

The SVM isolates the discrete topological tracking (the physical Pauli frame)
from the continuous probabilistic state vector. The complex array `v[]` is
dynamically allocated **exactly once** upon SVM initialization based on the
`peak_rank` emitted by the Back-End.

```cpp
#include <complex>
#include <vector>
#include <random>
#include <bit>

constexpr double TAN_PI_8 = 0.4142135623730950;
constexpr std::complex<double> I_UNIT(0, 1);
constexpr std::complex<double> PHASES[4] = {
    {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}
};

struct alignas(64) SchrodingerState {
    // 1. The Stochastic Sign Tracker (the physical Pauli frame)
    //    For ≤64 qubits, tracked inline as uint64_t.
    uint64_t destab_signs = 0;
    uint64_t stab_signs = 0;

    // 2. Dense Coefficient Array
    //    Allocated ONCE to exactly 2^(peak_rank) entries.
    //    64-byte aligned for AVX2/AVX-512 vectorization.
    std::complex<double>* v = nullptr;
    uint32_t v_size = 1;

    // 3. Classical Memory
    std::vector<uint8_t> meas_record;
    std::mt19937_64 rng;

    // 4. Per-Shot Seed (for deterministic replay)
    //    Stored explicitly so failed shots can be replayed with identical RNG state.
    uint64_t seed;

    // 5. Postselection Status
    //    Set to true if OP_POSTSELECT discards this shot.
    bool discarded = false;

    SchrodingerState(uint32_t peak_rank, uint64_t seed_) : rng(seed_), seed(seed_) {
        size_t capacity = 1ULL << peak_rank;
        v = (std::complex<double>*)
            std::aligned_alloc(64, capacity * sizeof(std::complex<double>));
        std::fill(v, v + capacity, std::complex<double>(0.0, 0.0));
        v[0] = {1.0, 0.0};  // Vacuum state
    }

    ~SchrodingerState() { std::free(v); }
};

// 🚨 CRITICAL RNG INVARIANT 🚨
//
// The SVM MUST NOT use C++ <random>'s std::uniform_real_distribution or any
// standard continuous distributions. These are IMPLEMENTATION-DEFINED and
// will break cross-platform determinism (e.g., generating different floats
// on GCC vs. Clang vs. MSVC for the exact same seed).
//
// All probability checks and gap sampling MUST use a custom, deterministic
// bitwise mapping from uint64_t to double in [0, 1):
//
//     inline double uniform_double(std::mt19937_64& rng) {
//         return (rng() >> 11) * 0x1.0p-53;  // Exact 53-bit mantissa
//     }
//
// This guarantees bit-identical results across all platforms and compilers.
```

### 5.1 Future: SoA Layout for SIMD Multi-Shot Batching

For SIMD-batched execution (multiple shots per vector register), the scalar
`std::complex<double>*` layout is replaced by a Struct-of-Arrays layout:

```cpp
template <size_t W>
struct alignas(64) VectorBatch {
    float real[W];
    float imag[W];
};
```

Under this model, `W` shots share the same bytecode instruction stream (enabled
by Topological Determinism) and execute in perfect SIMD lockstep. The scalar
path (`W=1`, `std::complex<double>`) is the MVP; the batched path is a
performance optimization layered on top.

---

## 6. Execution Semantics (SVM Runtime Loop)

The SVM executes instructions against a `SchrodingerState` in a tight switch loop.
Noise is injected via geometric gap sampling over the `NoiseSchedule`, not
via bytecode instructions.

```cpp
void execute(SchrodingerState& state,
             const std::vector<Instruction>& bytecode,
             const ConstantPool& constant_pool) {

    // --- Noise schedule state (geometric gap sampling) ---
    uint32_t next_noise_idx = 0;
    uint32_t next_noise_pc = constant_pool.noise_schedule.empty()
        ? UINT32_MAX
        : sample_next_noise_pc(state.rng, constant_pool.noise_schedule, 0);

    for (uint32_t pc = 0; pc < bytecode.size(); pc++) {

        // --- Inject noise if we've reached the sampled PC ---
        while (pc == next_noise_pc) {
            const auto& site = constant_pool.noise_schedule[next_noise_idx];
            state.destab_signs ^= site.destab_mask;
            state.stab_signs   ^= site.stab_mask;
            next_noise_pc = sample_next_noise_pc(
                state.rng, constant_pool.noise_schedule, next_noise_idx);
        }

        const auto& inst = bytecode[pc];
        switch (inst.opcode) {
```

### 6.1 T-Gate Fast Path: OP_BRANCH

Expands the coefficient array into a new dimension. The dominant ($I$) term is
implicitly carried forward — zero FLOPs, zero memory writes for the base path.

```cpp
            case Opcode::OP_BRANCH: {
                int frame_parity =
                    (std::popcount(state.destab_signs & inst.branch.destab_mask) ^
                     std::popcount(state.stab_signs & inst.branch.stab_mask)) & 1;

                int phys_idx = (inst.base_phase_idx + 2 * frame_parity) & 3;
                std::complex<double> p_phase = PHASES[phys_idx];
                std::complex<double> rel_weight =
                    (inst.is_dagger ? -I_UNIT : I_UNIT) * TAN_PI_8;

                uint32_t old_size = state.v_size;
                state.v_size *= 2;

                for (uint32_t idx = 0; idx < old_size; idx++) {
                    uint32_t new_idx = idx | inst.branch.x_mask;
                    int branch_parity =
                        std::popcount(new_idx & inst.commutation_mask) & 1;
                    std::complex<double> xi =
                        p_phase * (branch_parity ? -1.0 : 1.0);

                    // Spawn relative branch; base branch is identity.
                    state.v[new_idx] = state.v[idx] * (rel_weight * xi);
                }
                break;
            }
```

### 6.2 T-Gate Fast Path: OP_COLLIDE

In-place butterfly mix when the shift vector already exists in the GF(2) basis.

```cpp
            case Opcode::OP_COLLIDE: {
                int frame_parity =
                    (std::popcount(state.destab_signs & inst.branch.destab_mask) ^
                     std::popcount(state.stab_signs & inst.branch.stab_mask)) & 1;

                std::complex<double> p_phase =
                    PHASES[(inst.base_phase_idx + 2 * frame_parity) & 3];
                std::complex<double> rel_weight =
                    (inst.is_dagger ? -I_UNIT : I_UNIT) * TAN_PI_8;

                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    uint32_t peer_idx = idx ^ inst.branch.x_mask;
                    if (idx < peer_idx) {
                        int p_idx  = std::popcount(idx & inst.commutation_mask) & 1;
                        int p_peer = std::popcount(peer_idx & inst.commutation_mask) & 1;

                        std::complex<double> xi_idx =
                            p_phase * (p_idx ? -1.0 : 1.0);
                        std::complex<double> xi_peer =
                            p_phase * (p_peer ? -1.0 : 1.0);

                        auto old_idx  = state.v[idx];
                        auto old_peer = state.v[peer_idx];

                        state.v[idx]      = old_idx  + old_peer * (rel_weight * xi_peer);
                        state.v[peer_idx] = old_peer + old_idx  * (rel_weight * xi_idx);
                    }
                }
                break;
            }
```

### 6.3 Generic LCU: OP_BRANCH_LCU

Same structure as OP_BRANCH but fetches weight and phase from the constant pool.

```cpp
            case Opcode::OP_BRANCH_LCU: {
                const auto& lcu = constant_pool.lcu_pool[inst.lcu.lcu_pool_idx];

                int frame_parity =
                    (std::popcount(state.destab_signs & inst.lcu.destab_mask) ^
                     std::popcount(state.stab_signs & inst.lcu.stab_mask)) & 1;

                int phys_idx = (inst.base_phase_idx + 2 * frame_parity) & 3;
                std::complex<double> p_phase = PHASES[phys_idx];

                uint32_t old_size = state.v_size;
                state.v_size *= 2;

                for (uint32_t idx = 0; idx < old_size; idx++) {
                    uint32_t new_idx = idx | inst.lcu.x_mask;
                    int branch_parity =
                        std::popcount(new_idx & inst.commutation_mask) & 1;
                    std::complex<double> xi =
                        p_phase * (branch_parity ? -1.0 : 1.0);

                    auto old_val = state.v[idx];
                    state.v[new_idx] = old_val * (lcu.c_dom * lcu.weight * xi);
                    state.v[idx]     = old_val * lcu.c_dom;
                }
                break;
            }
```

### 6.4 Measurement: OP_MEASURE_MERGE

Samples a measurement outcome, merges paired amplitudes, shrinks the array,
and renormalizes (deferred normalization clears IEEE-754 drift).

```cpp
            case Opcode::OP_MEASURE_MERGE: {
                double prob_0 = 0.0, prob_1 = 0.0;

                // Pass 1: L2 norm gather
                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    uint32_t peer = idx ^ inst.meas.x_mask;
                    if (idx < peer) {
                        prob_0 += std::norm(state.v[idx] + state.v[peer]);
                        prob_1 += std::norm(state.v[idx] - state.v[peer]);
                    }
                }

                // Pass 2: Sample outcome
                std::uniform_real_distribution<double> dist(0.0, prob_0 + prob_1);
                uint8_t outcome = (dist(state.rng) < prob_1) ? 1 : 0;
                state.meas_record.push_back(outcome);

                // Pass 3: Merge + renormalize
                double norm = 1.0 / std::sqrt(outcome ? prob_1 : prob_0);
                uint32_t write = 0;
                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    uint32_t peer = idx ^ inst.meas.x_mask;
                    if (idx < peer) {
                        auto merged = outcome
                            ? (state.v[idx] - state.v[peer])
                            : (state.v[idx] + state.v[peer]);
                        state.v[write++] = merged * norm;
                    }
                }
                state.v_size /= 2;
                break;
            }
```

### 6.5 Sign Tracker Update: OP_AG_PIVOT

Applies a pre-computed GF(2) matrix to the sign tracker. This is the
$\mathcal{O}(n^2)$ Gaussian elimination that the Front-End pre-computed and
evicted to the constant pool — the SVM just executes a fast matrix-vector multiply.

```cpp
            case Opcode::OP_AG_PIVOT: {
                uint8_t outcome = state.rng() % 2;
                state.meas_record.push_back(outcome);

                const auto& mat = constant_pool.ag_matrices[inst.meta.payload_idx];

                uint64_t new_destab = 0, new_stab = 0;
                for (size_t i = 0; i < 64; i++) {
                    int d = (std::popcount(state.destab_signs & mat.destab_cols_x[i]) ^
                             std::popcount(state.stab_signs   & mat.destab_cols_z[i])) & 1;
                    int s = (std::popcount(state.destab_signs & mat.stab_cols_x[i]) ^
                             std::popcount(state.stab_signs   & mat.stab_cols_z[i])) & 1;
                    if (d) new_destab |= (1ULL << i);
                    if (s) new_stab   |= (1ULL << i);
                }

                if (outcome != inst.ag_ref_outcome) {
                    new_stab ^= (1ULL << inst.meta.ag_stab_slot);
                }

                state.destab_signs = new_destab;
                state.stab_signs   = new_stab;
                break;
            }
```

### 6.6 T-Gate Fast Path: OP_SCALAR_PHASE

Diagonal phase application when $\beta = 0$ (the operator commutes with all
basis vectors). No array expansion or butterfly — just multiply each element
by a phase-dependent scalar.

```cpp
            case Opcode::OP_SCALAR_PHASE: {
                int frame_parity =
                    (std::popcount(state.destab_signs & inst.branch.destab_mask) ^
                     std::popcount(state.stab_signs & inst.branch.stab_mask)) & 1;

                std::complex<double> p_phase =
                    PHASES[(inst.base_phase_idx + 2 * frame_parity) & 3];
                std::complex<double> rel_weight =
                    (inst.is_dagger ? -I_UNIT : I_UNIT) * TAN_PI_8;
                std::complex<double> scalar = 1.0 + rel_weight * p_phase;

                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    int branch_parity =
                        std::popcount(idx & inst.commutation_mask) & 1;
                    std::complex<double> s =
                        branch_parity ? (1.0 - rel_weight * p_phase) : scalar;
                    state.v[idx] *= s;
                }
                break;
            }
```

### 6.7 Generic LCU: OP_COLLIDE_LCU

In-place butterfly mix using weights from the constant pool.

```cpp
            case Opcode::OP_COLLIDE_LCU: {
                const auto& lcu = constant_pool.lcu_pool[inst.lcu.lcu_pool_idx];

                int frame_parity =
                    (std::popcount(state.destab_signs & inst.lcu.destab_mask) ^
                     std::popcount(state.stab_signs & inst.lcu.stab_mask)) & 1;

                std::complex<double> p_phase =
                    PHASES[(inst.base_phase_idx + 2 * frame_parity) & 3];

                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    uint32_t peer_idx = idx ^ inst.lcu.x_mask;
                    if (idx < peer_idx) {
                        int p_idx  = std::popcount(idx & inst.commutation_mask) & 1;
                        int p_peer = std::popcount(peer_idx & inst.commutation_mask) & 1;

                        std::complex<double> xi_idx =
                            p_phase * (p_idx ? -1.0 : 1.0);
                        std::complex<double> xi_peer =
                            p_phase * (p_peer ? -1.0 : 1.0);

                        auto old_idx  = state.v[idx];
                        auto old_peer = state.v[peer_idx];

                        state.v[idx]      = old_idx  * lcu.c_dom
                            + old_peer * (lcu.c_dom * lcu.weight * xi_peer);
                        state.v[peer_idx] = old_peer * lcu.c_dom
                            + old_idx  * (lcu.c_dom * lcu.weight * xi_idx);
                    }
                }
                break;
            }
```

### 6.8 Generic LCU: OP_SCALAR_PHASE_LCU

Diagonal phase application using weights from the constant pool.

```cpp
            case Opcode::OP_SCALAR_PHASE_LCU: {
                const auto& lcu = constant_pool.lcu_pool[inst.lcu.lcu_pool_idx];

                int frame_parity =
                    (std::popcount(state.destab_signs & inst.lcu.destab_mask) ^
                     std::popcount(state.stab_signs & inst.lcu.stab_mask)) & 1;

                std::complex<double> p_phase =
                    PHASES[(inst.base_phase_idx + 2 * frame_parity) & 3];

                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    int branch_parity =
                        std::popcount(idx & inst.commutation_mask) & 1;
                    std::complex<double> xi =
                        p_phase * (branch_parity ? -1.0 : 1.0);
                    state.v[idx] *= lcu.c_dom * (1.0 + lcu.weight * xi);
                }
                break;
            }
```

### 6.9 Measurement: OP_MEASURE_FILTER

Commuting measurement where $\beta \in \text{span}(V)$: the measurement
anti-commutes with some basis vectors but does not require array shrinking.
Instead, it zeroes out the half of the array corresponding to the rejected
outcome.

**Note on v_size:** Currently this opcode zeroes out half of the array but does
not actually halve `v_size`. The MVP relies on tracking the maximum size via
`peak_rank` during code generation and allocating exactly once. See the "Future
Extensions: Memory Compaction" section for details on potential future
enhancements.

```cpp
            case Opcode::OP_MEASURE_FILTER: {
                double prob_0 = 0.0, prob_1 = 0.0;

                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    int parity = std::popcount(idx & inst.commutation_mask) & 1;
                    if (parity)
                        prob_1 += std::norm(state.v[idx]);
                    else
                        prob_0 += std::norm(state.v[idx]);
                }

                std::uniform_real_distribution<double> dist(0.0, prob_0 + prob_1);
                uint8_t outcome = (dist(state.rng) < prob_1) ? 1 : 0;
                state.meas_record.push_back(outcome);

                double norm = 1.0 / std::sqrt(outcome ? prob_1 : prob_0);
                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    int parity = std::popcount(idx & inst.commutation_mask) & 1;
                    if (parity == outcome)
                        state.v[idx] *= norm;
                    else
                        state.v[idx] = {0.0, 0.0};
                }
                break;
            }
```

### 6.10 Measurement: OP_MEASURE_DETERMINISTIC

The measured qubit is in a known eigenstate — the outcome is determined entirely
by the sign tracker, with no array manipulation. This handles the case where
$\beta = 0$ for a measurement (the measurement commutes with all active basis
vectors in $V$).

```cpp
            case Opcode::OP_MEASURE_DETERMINISTIC: {
                int frame_parity =
                    (std::popcount(state.destab_signs & inst.meas.destab_mask) ^
                     std::popcount(state.stab_signs & inst.meas.stab_mask)) & 1;

                uint8_t outcome = frame_parity ^ inst.ag_ref_outcome;
                state.meas_record.push_back(outcome);
                // No array manipulation — state is already in an eigenstate.
                break;
            }
```

### 6.11 Basis Management: OP_INDEX_CNOT

Relabels dimensions within the GF(2) basis without changing the state. This
is emitted by the Back-End when it needs to adjust the active basis mapping
(e.g., after a measurement changes the basis structure). Swaps paired
amplitudes in the coefficient array.

```cpp
            case Opcode::OP_INDEX_CNOT: {
                // XOR-fold one basis dimension onto another.
                // x_mask identifies the source bit; commutation_mask identifies the target.
                for (uint32_t idx = 0; idx < state.v_size; idx++) {
                    if (idx & inst.branch.x_mask) {
                        uint32_t peer = idx ^ inst.commutation_mask;
                        if (idx < peer)
                            std::swap(state.v[idx], state.v[peer]);
                    }
                }
                break;
            }
```

### 6.12 Classical Logic: OP_DETECTOR

```cpp
            case Opcode::OP_DETECTOR: {
                const auto& targets =
                    constant_pool.detector_targets[inst.meta.payload_idx];
                uint8_t triggered = 0;
                for (uint32_t t : targets) {
                    triggered ^= state.meas_record[t];
                }
                // Record detection event to output sink
                break;
            }

        } // switch
    } // for
}
```

### 6.13 Mid-Circuit Early Abort: OP_POSTSELECT

For deep circuits like magic state cultivation factories, a failed classical
mid-circuit parity check means the shot is useless — continuing to execute
expensive non-Clifford operations (OP_BRANCH expansions) wastes compute.

**Execution Semantics:**

```cpp
            case Opcode::OP_POSTSELECT: {
                const auto& targets =
                    constant_pool.detector_targets[inst.meta.payload_idx];
                uint8_t parity = 0;
                for (uint32_t t : targets) {
                    parity ^= state.meas_record[t];
                }

                // Check against expected outcome (typically 0 for "pass")
                if (parity != inst.ag_ref_outcome) {
                    // Shot failed postselection — early exit
                    state.discarded = true;
                    return;  // Exit the bytecode execution loop entirely
                }
                break;
            }
```

**Key Design Decisions:**

1. **No Auto-Retry:** The simulator always executes exactly $N$ *total* shots.
   It does **not** auto-retry to fill a quota of successful shots. This gives
   users precise control and enables accurate yield calculation.

2. **Return Value:** The `sample()` function returns both the measurement
   results and a boolean mask indicating which shots passed postselection:

   ```cpp
   struct SampleResult {
       std::vector<std::vector<uint8_t>> measurements;  // [N][num_meas]
       std::vector<bool> passed;                        // [N] postselection mask
       std::vector<uint64_t> seeds;                     // [N] per-shot seeds
   };
   ```

3. **Yield Calculation:** Users calculate acceptance yield as:
   `yield = count(passed) / N`

This prevents wasting deep non-Clifford FLOPs on garbage shots while giving
users full visibility into postselection statistics.
```

---

## 7. Scaling Beyond 64 Qubits

The inline `uint64_t` tracker is capped at 64 qubits. Scaling uses **template
monomorphization** — the Instruction struct and SVM loop are parameterized by
`MaskType`.

| Qubit Count | `MaskType` | Mask Storage | Sign Tracker |
|-------------|-----------|-------------|-------------|
| $n \le 64$ | `uint64_t` | Inline in Instruction | Inline `uint64_t` pair |
| $n > 64$ | `uint32_t` | 32-bit index into `ConstantPool::mask_arena` | `stim::simd_bits<W>` |

Under the arbitrary-qubit path, the `destab_mask` and `stab_mask` fields change
semantics: they become indices into a densely packed
`std::vector<stim::simd_bits<W>> mask_arena` in the constant pool.

```cpp
template <typename MaskType>
inline int get_parity(const MaskType& destab_mask, const MaskType& stab_mask,
                      const SchrodingerState<MaskType>& state,
                      const ConstantPool& constant_pool) {
    if constexpr (std::is_same_v<MaskType, uint64_t>) {
        // FAST PATH: single-cycle hardware POPCNT
        return (std::popcount(state.destab_signs & destab_mask) ^
                std::popcount(state.stab_signs & stab_mask)) & 1;
    } else {
        // ARBITRARY PATH: resolve index via mask arena
        const auto& d = constant_pool.mask_arena[destab_mask];
        const auto& s = constant_pool.mask_arena[stab_mask];
        return ((state.destab_signs & d).popcnt() ^
                (state.stab_signs & s).popcnt()) & 1;
    }
}
```

The C++ compiler evaluates `if constexpr` at compile time, emitting two
separate optimized binaries with identical bytecode layout. The Instruction
struct actually *shrinks* for $n > 64$ (two 4-byte indices replace two 8-byte
masks), maintaining cache alignment.

---

## Appendix A: Constants

```cpp
constexpr double TAN_PI_8 = 0.4142135623730950;   // tan(π/8)
constexpr std::complex<double> I_UNIT(0, 1);
constexpr std::complex<double> PHASES[4] = {
    {1.0, 0.0},   // phase index 0: +1
    {0.0, 1.0},   // phase index 1: +i
    {-1.0, 0.0},  // phase index 2: -1
    {0.0, -1.0}   // phase index 3: -i
};
```

---

## Appendix B: Future Extensions: Memory Compaction

For infinitely deep magic state cultivation circuits, constantly injecting and
measuring states without reclaiming memory would cause `peak_rank` to grow
indefinitely, eventually leading to an Out-Of-Memory (OOM) error.

### The Problem

Currently, `OP_MEASURE_FILTER` zeroes out half of the coefficient array but
does not actually reduce `v_size`. The MVP allocates exactly $2^{\textrm{peak-rank}}$
entries once and relies on the Back-End tracking the maximum rank during code
generation. This works for bounded circuits but fails for unbounded cultivation
loops.

### Future Solution: Active Memory Reclamation

A future extension could enable actual memory reclamation:

1. **OP_INDEX_CNOT for Alignment:** Use `OP_INDEX_CNOT` to mathematically align
   measured parities to the highest dimension of the active GF(2) basis $V$.
   This reorders the basis so that the zeroed-out entries are contiguous at
   the end of the array.

2. **OP_MEASURE_FILTER with Compaction:** After alignment, `OP_MEASURE_FILTER`
   can actually halve `v_size` rather than just zeroing entries. The Back-End
   would emit this variant when it detects that the measurement is aligned to
   the highest basis dimension.

3. **Rank Tracking in Code Generation:** The Back-End would track not just
   `peak_rank` but also the current "active rank" that shrinks after compacting
   measurements. This enables reuse of the freed memory for subsequent
   non-Clifford expansions.

### Paper Prototype Assessment

For the paper prototype, we will assess whether the specific magic state
cultivation circuits under study require this memory compaction capability.
If the circuits have bounded rank (the number of active non-Clifford dimensions
never exceeds a threshold), the MVP's single-allocation approach suffices.
If the circuits exhibit unbounded rank growth, we will implement the compaction
extension as part of the prototype work.
