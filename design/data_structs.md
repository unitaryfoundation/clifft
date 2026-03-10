# UCC — Data Structures & Memory Layouts

This document defines the strict C++ memory layouts for the Heisenberg IR (HIR), the RISC VM bytecode, and the runtime state.

**Architectural Invariant:** The VM `Instruction` struct MUST remain exactly 32 bytes to ensure two instructions fit perfectly into a 64-byte L1 CPU cache line.

## 1. The Heisenberg IR (HIR)

The HIR represents physical operations mapped to the $t=0$ reference frame. It contains no VM execution logic.

```cpp
constexpr size_t kStimWidth = 64; // Limits MVP to 64 qubits

struct HeisenbergOp {
    // Re-wound static Pauli masks
    stim::bitword<kStimWidth> destab_mask_; // 8 bytes: X-bits at t=0
    stim::bitword<kStimWidth> stab_mask_;   // 8 bytes: Z-bits at t=0

    // Payload union (up to 12 bytes allowed to maintain 32-byte total size)
    // Payload union (exactly 12 bytes to maintain 32-byte total size)
    union {
        struct { uint32_t meas_record_idx; } measure_;
        struct { uint32_t controlling_meas; } conditional_;
        struct { uint32_t site_idx; } noise_;
        struct { uint32_t entry_idx; } readout_;
        struct { uint32_t target_list_idx; } detector_;
        struct { uint32_t obs_idx; uint32_t target_list_idx; } observable_;
    };

    OpType type_;
    bool sign_;
    uint8_t flags_;
    // exactly 32 bytes
};
static_assert(sizeof(HeisenbergOp) == 32);
```

## VM Execution State (SchrodingerState)
The VM State maps exactly to the Factored State Representation.

```c++
struct alignas(64) SchrodingerState {
    // 1. The Pauli Frame (P)
    // Tracks physical errors & measurement parities.
    stim::bitword<kStimWidth> p_x = 0;
    stim::bitword<kStimWidth> p_z = 0;

    // 2. Global Scalar (gamma)
    // Tracks continuous global phase and deferred normalization.
    std::complex<double> gamma = {1.0, 0.0};

    // 3. Active Statevector (|phi>_A)
    // Allocated ONCE to 2^{peak_rank} elements.
    std::complex<double>* v = nullptr;
    uint64_t v_size = 1;               // Current active size (2^k)
    uint32_t active_k = 0;             // Current active dimension k

    // Classical Memory & QEC Tracking
    std::vector<uint8_t> meas_record;
    std::vector<uint8_t> det_record;
    std::vector<uint8_t> obs_record;

    // Deterministic PRNG
    Xoshiro256PlusPlus rng_;
};
};
```
## The 32-Byte RISC Instruction Bytecode
Because the Back-End compresses all multi-qubit global topology into localized virtual operations AOT, the VM uses a RISC-like instruction set targeting specific virtual axes.We use uint16_t for axis indices, liberating the struct from massive 64-bit mask requirements and making it natively ready for 512-qubit scaling.

```c++
enum class Opcode : uint8_t {
    // Frame Opcodes (Zero-cost dormant updates. Update p_x, p_z only)
    OP_FRAME_CNOT, OP_FRAME_CZ, OP_FRAME_H, OP_FRAME_S, OP_FRAME_S_DAG, OP_FRAME_SWAP,

    // Array Opcodes (Update p_x, p_z AND loop over v[] to swap/mix)
    OP_ARRAY_CNOT, OP_ARRAY_CZ, OP_ARRAY_SWAP,
    OP_ARRAY_MULTI_CNOT,  // Fused star-graph: multiple controls, one target
    OP_ARRAY_MULTI_CZ,    // Fused star-graph: one control, multiple targets
    OP_ARRAY_H,           // Hadamard on active axis (butterfly + frame swap)
    OP_ARRAY_S,           // Phase S on active axis (diag(1, i) + frame update)
    OP_ARRAY_S_DAG,       // Phase S-dagger on active axis (diag(1, -i) + frame update)

    // Local Math & Expansion
    OP_EXPAND,        // Virtual H_v on dormant: k -> k+1, gamma /= sqrt(2)
    OP_PHASE_T,       // Active diagonal T phase
    OP_PHASE_T_DAG,   // Active diagonal T-dagger phase
    OP_EXPAND_T,      // Fused EXPAND + PHASE_T in one array pass
    OP_EXPAND_T_DAG,  // Fused EXPAND + PHASE_T_DAG in one array pass

    // Measurement
    OP_MEAS_DORMANT_STATIC,    // Deterministic outcome from p_x
    OP_MEAS_DORMANT_RANDOM,    // Random pivot, algebraic phase to gamma
    OP_MEAS_ACTIVE_DIAGONAL,   // Z-basis filter, halves array (k -> k-1)
    OP_MEAS_ACTIVE_INTERFERE,  // X-basis fold, halves array (k -> k-1)
    OP_SWAP_MEAS_INTERFERE,    // Fused ARRAY_SWAP + MEAS_ACTIVE_INTERFERE

    // Classical / Errors
    OP_APPLY_PAULI,    // XORs a full N-bit mask from ConstantPool into P
    OP_NOISE,          // Stochastic Pauli channel (rolls RNG, may apply Pauli)
    OP_NOISE_BLOCK,    // Contiguous block of noise sites [start, start+count)
    OP_READOUT_NOISE,  // Classical bit-flip on measurement result
    OP_DETECTOR,       // Parity check over measurement records
    OP_POSTSELECT,     // Post-selection check: abort shot if parity != 0
    OP_OBSERVABLE,     // Logical observable accumulator
    NUM_OPCODES        // Sentinel: must remain last for binding completeness checks
};

struct alignas(32) Instruction {
    Opcode opcode;      // Offset 0
    uint8_t _reserved;  // Offset 1
    uint8_t flags;      // Offset 2
    uint8_t _pad;       // Offset 3
    uint16_t axis_1;    // Offset 4
    uint16_t axis_2;    // Offset 6

    // 24 bytes remaining for payload (Offsets 8..31)
    union {
        struct {
            double weight_re;  // Offset 8
            double weight_im;  // Offset 16
        } math;

        struct {
            uint32_t classical_idx;  // Offset 8
            uint32_t expected_val;   // Offset 12
        } classical;

        struct {
            uint32_t cp_mask_idx;    // Offset 8
            uint32_t condition_idx;  // Offset 12
        } pauli;

        // For OP_ARRAY_MULTI_CNOT / OP_ARRAY_MULTI_CZ star-graph fusions
        struct {
            uint64_t mask;  // Offset 8: ctrl_mask (MULTI_CNOT) or target_mask (MULTI_CZ)
        } multi_gate;

        uint8_t raw[24];
    };
};
static_assert(sizeof(Instruction) == 32, "Must be exactly 32 bytes");
```
## Execution Semantics Example

* **Virtual Array Layout:** Virtual axes 0 to active_k - 1 map perfectly to the little-endian bit indices of the complex array v[]. Virtual axes $\ge$ active_k are strictly in the $|0\rangle_D$ dormant state.
* **Frame Propagation:** When OP_FRAME_CNOT(c, t) executes, the VM simply updates the error frame: `p_x[t] ^= p_x[c] and p_z[c] ^= p_z[t]`. The complex array v is untouched.
* **T-Gate Phase Resolution:** When applying OP_PHASE_T(v), the VM checks the error parity p_x[v]. If 1, it algebraically commutes past the rotation, effectively applying $T^\dagger$ to the array, whilst updating $\gamma$ accordingly.
* **Array Compaction:** When an Active axis is measured (OP_MEAS_ACTIVE_*), it transitions to Dormant, and the complex array v[] must physically halve in size ($k \to k-1$). To ensure the array folds contiguously without strided indexing, the Back-End explicitly emits `OP_FRAME_SWAP` and `OP_ARRAY_SWAP` to route the target virtual qubit to the highest active memory axis ($k-1$) immediately before measurement.
