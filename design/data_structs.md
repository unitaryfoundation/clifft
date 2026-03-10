# UCC — Data Structures & Memory Layouts

This document defines the strict C++ memory layouts for the Heisenberg IR (HIR), the RISC VM bytecode, and the runtime state.

**Architectural Invariant:** The VM `Instruction` struct MUST remain exactly 32 bytes to ensure two instructions fit perfectly into a 64-byte L1 CPU cache line.

## 1. The Heisenberg IR (HIR)

The HIR represents physical operations mapped to the $t=0$ reference frame. It contains no VM execution logic.

```cpp
constexpr uint32_t kMaxInlineQubits = UCC_MAX_QUBITS;

struct HeisenbergOp {
    // Re-wound static Pauli masks
    ucc::BitMask<kMaxInlineQubits> destab_mask_;
    ucc::BitMask<kMaxInlineQubits> stab_mask_;

    // Payload union (exactly 12 bytes to maintain 32-byte total size at 64 qubits)
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
};

#if UCC_MAX_QUBITS == 64
static_assert(sizeof(HeisenbergOp) == 32, "HeisenbergOp must be exactly 32 bytes at 64-qubit width");
#endif
```

## VM Execution State (SchrodingerState)
The VM State maps exactly to the Factored State Representation.

```c++
class SchrodingerState {
public:
    // 1. The Pauli Frame (P)
    // Tracks physical errors & measurement parities.
    ucc::BitMask<kMaxInlineQubits> p_x = 0;
    ucc::BitMask<kMaxInlineQubits> p_z = 0;

    // 2. Global Scalar (gamma)
    // Tracks continuous global phase and deferred normalization.
    std::complex<double> gamma = {1.0, 0.0};

    // 3. Active Statevector (|phi>_A)
    // Allocated ONCE, accessed via v()
    uint32_t active_k = 0;

    // Classical Memory
    std::vector<uint8_t> meas_record;
    std::vector<uint8_t> det_record;
    std::vector<uint8_t> obs_record;

    // Gap-based noise sampling
    uint32_t next_noise_idx = 0;

private:
    std::complex<double>* v_ = nullptr;  // 64-byte aligned
    uint64_t array_size_ = 0;            // 2^peak_rank (allocated capacity)
    Xoshiro256PlusPlus rng_;
};
```
## The 32-Byte RISC Instruction Bytecode
Because the Back-End compresses all multi-qubit global topology into localized virtual operations at compile time, the VM uses a RISC-like instruction set targeting specific virtual axes. We use `uint16_t` for axis indices, liberating the struct from massive 64-bit mask requirements and making it natively ready for 512-qubit scaling.

```c++
enum class Opcode : uint8_t {
    // Frame Opcodes (Zero-cost dormant updates. Update p_x, p_z only)
    OP_FRAME_CNOT,
    OP_FRAME_CZ,
    OP_FRAME_H,
    OP_FRAME_S,
    OP_FRAME_S_DAG,
    OP_FRAME_SWAP,

    // Array Opcodes (Update p_x, p_z AND loop over v[] to swap/mix)
    OP_ARRAY_CNOT,
    OP_ARRAY_CZ,
    OP_ARRAY_SWAP,
    OP_ARRAY_MULTI_CNOT,
    OP_ARRAY_MULTI_CZ,
    OP_ARRAY_H,
    OP_ARRAY_S,
    OP_ARRAY_S_DAG,

    // Local Math & Expansion
    OP_EXPAND,
    OP_PHASE_T,
    OP_PHASE_T_DAG,
    OP_EXPAND_T,
    OP_EXPAND_T_DAG,

    // Measurement
    OP_MEAS_DORMANT_STATIC,
    OP_MEAS_DORMANT_RANDOM,
    OP_MEAS_ACTIVE_DIAGONAL,
    OP_MEAS_ACTIVE_INTERFERE,
    OP_SWAP_MEAS_INTERFERE,

    // Classical / Errors
    OP_APPLY_PAULI,
    OP_NOISE,
    OP_NOISE_BLOCK,
    OP_READOUT_NOISE,
    OP_DETECTOR,
    OP_POSTSELECT,
    OP_OBSERVABLE,
    NUM_OPCODES
};

struct alignas(32) Instruction {
    // Flag bits for measurement instructions
    static constexpr uint8_t FLAG_SIGN = 1 << 0;
    static constexpr uint8_t FLAG_HIDDEN = 1 << 1;
    static constexpr uint8_t FLAG_IDENTITY = 1 << 2;

    Opcode opcode;      // Offset 0
    uint8_t _reserved;  // Offset 1
    uint8_t flags;      // Offset 2
    uint8_t _pad;       // Offset 3
    uint16_t axis_1;    // Offset 4
    uint16_t axis_2;    // Offset 6

    // 24 bytes remaining for payload (Offsets 8..31)
    union {
        // Variant A: Local Math Payloads
        struct {
            double weight_re;   // Offset 8
            double weight_im;   // Offset 16
            uint8_t _pad_a[8];  // Offset 24
        } math;

        // Variant B: Classical targets (Measurements)
        struct {
            uint32_t classical_idx;  // Offset 8
            uint32_t expected_val;   // Offset 12
            uint8_t _pad_b[16];      // Offset 16
        } classical;

        // Variant C: Full Pauli injection (Errors/Conditionals)
        struct {
            uint32_t cp_mask_idx;    // Offset 8
            uint32_t condition_idx;  // Offset 12
            uint8_t _pad_c[16];      // Offset 16
        } pauli;

        // Variant D: Multi-gate bitmask (MULTI_CNOT/MULTI_CZ)
        struct {
            uint64_t mask;       // Offset 8 (64-bit control/target bitmask)
            uint8_t _pad_d[16];  // Offset 16
        } multi_gate;

        uint8_t raw[24];
    };
};
static_assert(sizeof(Instruction) == 32, "Instruction must be exactly 32 bytes");
```
## Execution Semantics Example

* **Virtual Array Layout:** Virtual axes 0 to active_k - 1 map perfectly to the little-endian bit indices of the complex array v[]. Virtual axes $\ge$ active_k are strictly in the $|0\rangle_D$ dormant state.
* **Frame Propagation:** When OP_FRAME_CNOT(c, t) executes, the VM simply updates the error frame: `p_x[t] ^= p_x[c] and p_z[c] ^= p_z[t]`. The complex array v is untouched.
* **T-Gate Phase Resolution:** When applying OP_PHASE_T(v), the VM checks the error parity p_x[v]. If 1, it algebraically commutes past the rotation, effectively applying $T^\dagger$ to the array, whilst updating $\gamma$ accordingly.
* **Array Compaction:** When an Active axis is measured (OP_MEAS_ACTIVE_*), it transitions to Dormant, and the complex array v[] must physically halve in size ($k \to k-1$). To ensure the array folds contiguously without strided indexing, the Back-End explicitly emits `OP_FRAME_SWAP` and `OP_ARRAY_SWAP` to route the target virtual qubit to the highest active memory axis ($k-1$) immediately before measurement.
