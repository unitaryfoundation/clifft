#pragma once

#include "ucc/backend/source_map.h"
#include "ucc/frontend/hir.h"

#include "stim.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ucc {

// =============================================================================
// RISC Execution Bytecode Opcodes
// =============================================================================
//
// The VM uses a localized RISC instruction set. All multi-qubit global topology
// is compressed into local 1-qubit and 2-qubit virtual axis operations by the
// Back-End AOT compiler. The VM never evaluates basis spans or commutations.

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
    OP_PHASE_ROT,     // Continuous Z-rotation on active axis (arbitrary angle)
    OP_EXPAND_ROT,    // Fused EXPAND + PHASE_ROT in one array pass

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

// =============================================================================
// 32-Byte RISC Instruction Bytecode
// =============================================================================
//
// Exactly 32 bytes ensures 2 instructions per 64-byte L1 cache line.
// Uses uint16_t for axis indices, enabling 512-qubit scaling without
// architectural changes.

struct alignas(32) Instruction {
    // Flag bits for measurement instructions
    static constexpr uint8_t FLAG_SIGN = 1 << 0;      // Measurement sign (XOR with outcome)
    static constexpr uint8_t FLAG_HIDDEN = 1 << 1;    // Hidden measurement (not in visible record)
    static constexpr uint8_t FLAG_IDENTITY = 1 << 2;  // Identity measurement (no frame interaction)
    // Flag bit for detector/postselect: noiseless reference parity is 1
    static constexpr uint8_t FLAG_EXPECTED_ONE = 1 << 3;

    Opcode opcode;      // Offset 0
    uint8_t _reserved;  // Offset 1 (padding for 32-byte alignment)
    uint8_t flags;      // Offset 2
    uint8_t _pad;       // Offset 3
    uint16_t axis_1;    // Offset 4 (Virtual axis target/control)
    uint16_t axis_2;    // Offset 6 (Virtual axis target 2)

    // 24 bytes remaining for payload (offsets 8..31)
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
            uint32_t cp_mask_idx;    // Offset 8 (Index into ConstantPool)
            uint32_t condition_idx;  // Offset 12
            uint8_t _pad_c[16];      // Offset 16
        } pauli;

        // Variant D: Multi-gate bitmask (MULTI_CNOT/MULTI_CZ)
        struct {
            uint64_t mask;       // Offset 8 (64-bit control/target bitmask)
            uint8_t _pad_d[16];  // Offset 16
        } multi_gate;

        uint8_t raw[24];  // Full payload access
    };
};

static_assert(sizeof(Instruction) == 32, "Instruction must be exactly 32 bytes");

// =============================================================================
// Instruction Factories
// =============================================================================
//
// Static factory methods that construct fully-initialized Instructions.
// Used by both the Back-End compiler and test code.

[[nodiscard]] Instruction make_frame_cnot(uint16_t ctrl, uint16_t tgt);
[[nodiscard]] Instruction make_frame_cz(uint16_t a, uint16_t b);
[[nodiscard]] Instruction make_frame_h(uint16_t v);
[[nodiscard]] Instruction make_frame_s(uint16_t v);
[[nodiscard]] Instruction make_frame_s_dag(uint16_t v);
[[nodiscard]] Instruction make_frame_swap(uint16_t a, uint16_t b);
[[nodiscard]] Instruction make_array_cnot(uint16_t ctrl_axis, uint16_t tgt_axis);
[[nodiscard]] Instruction make_array_cz(uint16_t a_axis, uint16_t b_axis);
[[nodiscard]] Instruction make_array_swap(uint16_t a, uint16_t b);
[[nodiscard]] Instruction make_array_multi_cnot(uint16_t target, uint64_t ctrl_mask);
[[nodiscard]] Instruction make_array_multi_cz(uint16_t control, uint64_t target_mask);
[[nodiscard]] Instruction make_array_h(uint16_t axis);
[[nodiscard]] Instruction make_array_s(uint16_t axis);
[[nodiscard]] Instruction make_array_s_dag(uint16_t axis);
[[nodiscard]] Instruction make_expand(uint16_t axis);
[[nodiscard]] Instruction make_phase_t(uint16_t axis);
[[nodiscard]] Instruction make_phase_t_dag(uint16_t axis);
[[nodiscard]] Instruction make_expand_t(uint16_t axis);
[[nodiscard]] Instruction make_expand_t_dag(uint16_t axis);
[[nodiscard]] Instruction make_phase_rot(uint16_t axis, double re, double im);
[[nodiscard]] Instruction make_expand_rot(uint16_t axis, double re, double im);
[[nodiscard]] Instruction make_swap_meas_interfere(uint16_t swap_from, uint16_t swap_to,
                                                   uint32_t classical_idx, bool sign);
[[nodiscard]] Instruction make_meas(Opcode meas_opcode, uint16_t axis, uint32_t classical_idx,
                                    bool sign);
[[nodiscard]] Instruction make_apply_pauli(uint32_t cp_mask_idx, uint32_t condition_idx);
[[nodiscard]] Instruction make_noise(uint32_t site_idx);
[[nodiscard]] Instruction make_noise_block(uint32_t start_site, uint32_t count);
[[nodiscard]] Instruction make_readout_noise(uint32_t entry_idx);
/// Strongly-typed flag for detector/postselect expected noiseless parity.
enum class ExpectedParity : uint8_t { Zero = 0, One = 1 };

[[nodiscard]] Instruction make_detector(uint32_t det_list_idx, uint32_t classical_idx,
                                        ExpectedParity expected);
[[nodiscard]] Instruction make_postselect(uint32_t det_list_idx, uint32_t classical_idx,
                                          ExpectedParity expected);
[[nodiscard]] Instruction make_observable(uint32_t target_list_idx, uint32_t obs_idx);

// =============================================================================
// Constant Pool
// =============================================================================
//
// Heavy data referenced by index from Instructions. Kept separate to maintain
// the 32-byte Instruction size constraint.

/// Full Pauli mask stored in the ConstantPool for OP_APPLY_PAULI.
/// Uses BitMask<kMaxInlineQubits> instead of stim::PauliString to avoid
/// heap allocations on the VM execution hot path.
struct PauliMask {
    PauliBitMask x;
    PauliBitMask z;
    bool sign = false;
};

struct ConstantPool {
    // Forward Clifford tableau at circuit end (for statevector expansion).
    // Computed as U_C = U_phys * V_cum^dag at end of compilation.
    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    // Global scalar gamma accumulated during compilation
    std::complex<double> global_weight = {1.0, 0.0};

    // Full N-bit Pauli masks for OP_APPLY_PAULI (indexed by cp_mask_idx)
    std::vector<PauliMask> pauli_masks;

    // Noise sites for OP_NOISE (virtual-frame-mapped channels)
    std::vector<NoiseSite> noise_sites;

    // Readout noise entries for OP_READOUT_NOISE
    std::vector<ReadoutNoiseEntry> readout_noise;

    // Target lists for detector parity checks
    std::vector<std::vector<uint32_t>> detector_targets;

    // Target lists for observable parity checks
    std::vector<std::vector<uint32_t>> observable_targets;

    // Cumulative hazard table for gap-based noise sampling.
    // Entry i = sum of -log(1 - prob_sum_j) for noise sites j=0..i.
    // Allows O(1) skip of silent noise sites via exponential gap sampling.
    std::vector<double> noise_hazards;
};

// =============================================================================
// Compiled Module
// =============================================================================
//
// Complete output of the Back-End: bytecode + constant pool + metadata.

struct CompiledModule {
    std::vector<Instruction> bytecode;
    ConstantPool constant_pool;
    uint32_t num_qubits = 0;        // Total physical qubits n
    uint32_t peak_rank = 0;         // Maximum active dimension k reached
    uint32_t num_measurements = 0;  // Visible measurements (user-facing count)
    uint32_t total_meas_slots = 0;  // Visible + hidden measurements (VM allocation)
    uint32_t num_detectors = 0;     // Total detectors
    uint32_t num_observables = 0;   // Total observables

    // Expected noiseless observable parities for syndrome normalization.
    // When non-empty, sample()/sample_survivors() XOR obs_record[i] with
    // expected_observables[i] before writing output.
    std::vector<uint8_t> expected_observables;

    // Source line mapping and per-instruction active_k history.
    // Empty if the HIR had no source map.
    SourceMap source_map;
};

// =============================================================================
// Back-End API
// =============================================================================

/// Lower HIR to executable RISC bytecode.
/// Tracks virtual frame V_cum, compresses multi-qubit Paulis to local ops.
/// If postselection_mask is non-empty, detectors at indices where
/// mask[det_idx] != 0 are emitted as OP_POSTSELECT instead of OP_DETECTOR.
/// expected_detectors/expected_observables carry noiseless reference parities
/// for error syndrome normalization.
[[nodiscard]] CompiledModule lower(const HirModule& hir,
                                   std::span<const uint8_t> postselection_mask = {},
                                   std::span<const uint8_t> expected_detectors = {},
                                   std::span<const uint8_t> expected_observables = {});

}  // namespace ucc
