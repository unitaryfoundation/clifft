#pragma once

#include "clifft/backend/source_map.h"
#include "clifft/frontend/hir.h"

#include "stim.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft {

// =============================================================================
// VM Execution Bytecode Opcodes
// =============================================================================
//
// The VM uses a localized instruction set. All multi-qubit global topology
// is localized into 1-qubit and 2-qubit virtual axis operations by the
// Back-End AOT compiler. The VM never evaluates basis spans or commutations.

enum class Opcode : uint8_t {
    // Frame Opcodes (Zero-cost dormant updates. Update p_x, p_z only)
    OP_FRAME_CNOT,
    OP_FRAME_CZ,
    OP_FRAME_H,
    OP_FRAME_S,
    OP_FRAME_S_DAG,
    OP_FRAME_SWAP,

    // Array Opcodes (Fixed k. Update p_x, p_z AND loop over v[] to swap/mix)
    OP_ARRAY_CNOT,
    OP_ARRAY_CZ,
    OP_ARRAY_SWAP,
    OP_ARRAY_MULTI_CNOT,  // Fused star-graph: multiple controls, one target
    OP_ARRAY_MULTI_CZ,    // Fused star-graph: one control, multiple targets
    OP_ARRAY_H,           // Hadamard on active axis (butterfly + frame swap)
    OP_ARRAY_S,           // Phase S on active axis (diag(1, i) + frame update)
    OP_ARRAY_S_DAG,       // Phase S-dagger on active axis (diag(1, -i) + frame update)
    OP_ARRAY_T,           // Active diagonal T phase (non-Clifford)
    OP_ARRAY_T_DAG,       // Active diagonal T-dagger phase (non-Clifford)
    OP_ARRAY_ROT,         // Continuous Z-rotation on active axis (arbitrary angle)
    OP_ARRAY_U2,          // Fused single-axis 2x2 unitary (ConstantPool lookup)
    OP_ARRAY_U4,          // Fused 2-axis 4x4 unitary (ConstantPool lookup)

    // Subspace Expansion (k -> k+1; non-Clifford rotations may be fused in)
    OP_EXPAND,        // Virtual H_v on dormant: k -> k+1, gamma /= sqrt(2)
    OP_EXPAND_T,      // Fused EXPAND + ARRAY_T in one array pass
    OP_EXPAND_T_DAG,  // Fused EXPAND + ARRAY_T_DAG in one array pass
    OP_EXPAND_ROT,    // Fused EXPAND + ARRAY_ROT in one array pass

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
    OP_EXP_VAL,        // Read-only expectation value probe (full virtual Pauli mask)
    NUM_OPCODES        // Sentinel: must remain last for binding completeness checks
};

// =============================================================================
// 32-Byte VM Instruction Bytecode
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

        // Variant E: Fused single-axis 2x2 unitary (OP_ARRAY_U2)
        struct {
            uint32_t cp_idx;     // Offset 8 (Index into ConstantPool::fused_u2_nodes)
            uint8_t _pad_e[20];  // Offset 12
        } u2;

        // Variant F: Fused 2-axis 4x4 unitary (OP_ARRAY_U4)
        struct {
            uint32_t cp_idx;     // Offset 8 (Index into ConstantPool::fused_u4_nodes)
            uint8_t _pad_f[20];  // Offset 12
        } u4;

        // Variant G: Expectation value probe (OP_EXP_VAL)
        // Same layout as pauli variant but with semantically meaningful names.
        struct {
            uint32_t cp_exp_val_idx;  // Offset 8 (Index into ConstantPool::exp_val_masks)
            uint32_t exp_val_idx;     // Offset 12 (Index into state.exp_vals)
            uint8_t _pad_g[16];       // Offset 16
        } exp_val;

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
[[nodiscard]] Instruction make_array_t(uint16_t axis);
[[nodiscard]] Instruction make_array_t_dag(uint16_t axis);
[[nodiscard]] Instruction make_array_rot(uint16_t axis, double re, double im);
[[nodiscard]] Instruction make_array_u2(uint16_t axis, uint32_t cp_idx);
[[nodiscard]] Instruction make_array_u4(uint16_t axis_lo, uint16_t axis_hi, uint32_t cp_idx);
[[nodiscard]] Instruction make_expand(uint16_t axis);
[[nodiscard]] Instruction make_expand_t(uint16_t axis);
[[nodiscard]] Instruction make_expand_t_dag(uint16_t axis);
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
[[nodiscard]] Instruction make_exp_val(uint32_t cp_exp_val_idx, uint32_t exp_val_idx);

// =============================================================================
// Constant Pool
// =============================================================================
//
// Heavy data referenced by index from Instructions. Kept separate to maintain
// the 32-byte Instruction size constraint.

// Pre-computed 2x2 unitary for OP_ARRAY_U2 (single-axis CISC fusion).
// For each of 4 possible incoming Pauli frame states on the target axis
// ((p_z << 1) | p_x: 0=I, 1=X, 2=Z, 3=Y), stores the fused matrix,
// accumulated global phase, and resulting frame state.
struct FusedU2Node {
    // Row-major 2x2 matrices indexed by 2-bit input frame state.
    // matrices[s][0..3] = {a, b, c, d} where U|0>=a|0>+c|1>, U|1>=b|0>+d|1>
    std::complex<double> matrices[4][4];

    // Global phase multiplier accumulated during the fused sequence.
    std::complex<double> gamma_multipliers[4];

    // Resulting 2-bit (p_z << 1) | p_x frame state after the sequence.
    uint8_t out_states[4];
};

// Pre-computed 4x4 unitary for OP_ARRAY_U4 (2-axis tile fusion).
// For each of 16 possible incoming Pauli frame states on {axis_lo, axis_hi}
// ((pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo), stores the fused
// 4x4 matrix, accumulated global phase, and resulting frame state.
struct FusedU4Node {
    struct Entry {
        // Row-major 4x4 matrix: U|j> = sum_i matrix[i][j] |i>
        // Basis ordering: |00>, |01>, |10>, |11> with lo=LSB, hi=MSB.
        std::complex<double> matrix[4][4];

        // Global phase multiplier accumulated during the fused sequence.
        std::complex<double> gamma_multiplier;

        // Resulting 4-bit frame state: (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo
        uint8_t out_state;
    };
    Entry entries[16];  // Indexed by 4-bit incoming frame state
};

struct ConstantPool {
    // Forward Clifford tableau at circuit end (for statevector expansion).
    // Computed as U_C = U_phys * V_cum^dag at end of compilation.
    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    // Global scalar gamma accumulated during compilation
    std::complex<double> global_weight = {1.0, 0.0};

    // Full N-bit Pauli masks for OP_APPLY_PAULI. Bytecode references slots
    // by cp_mask_idx, treated as a PauliMaskHandle into this arena.
    PauliMaskArena pauli_masks;

    // Full N-bit Pauli masks for OP_EXP_VAL. Separate from pauli_masks to
    // avoid index collision between frame mutation (APPLY_PAULI) and
    // read-only probing (EXP_VAL).
    PauliMaskArena exp_val_masks;

    // Noise sites for OP_NOISE (virtual-frame-mapped channels). Each
    // NoiseChannel inside a NoiseSite carries a handle into
    // noise_channel_masks below.
    std::vector<NoiseSite> noise_sites;

    // Pauli mask arena for the (virtual-frame-mapped) noise channel
    // masks referenced from noise_sites.
    PauliMaskArena noise_channel_masks;

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

    // Fused single-axis 2x2 unitary nodes for OP_ARRAY_U2.
    std::vector<FusedU2Node> fused_u2_nodes;

    // Fused 2-axis 4x4 unitary nodes for OP_ARRAY_U4.
    std::vector<FusedU4Node> fused_u4_nodes;
};

// =============================================================================
// Compiled Module
// =============================================================================
//
// Complete output of the Back-End: bytecode + constant pool + metadata.

struct CompiledModule {
    std::vector<Instruction> bytecode;
    ConstantPool constant_pool;
    uint32_t num_qubits = 0;         // Total physical qubits n
    uint32_t peak_rank = 0;          // Maximum active dimension k reached
    uint32_t num_measurements = 0;   // Visible measurements (user-facing count)
    uint32_t total_meas_slots = 0;   // Visible + hidden measurements (VM allocation)
    uint32_t num_detectors = 0;      // Total detectors
    uint32_t num_observables = 0;    // Total observables
    uint32_t num_exp_vals = 0;       // Total expectation value probes
    bool has_postselection = false;  // True if any detector uses OP_POSTSELECT

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

/// Lower HIR to executable VM bytecode.
/// Tracks virtual frame V_cum, localizes multi-qubit Paulis to local ops.
/// If postselection_mask is non-empty, detectors at indices where
/// mask[det_idx] != 0 are emitted as OP_POSTSELECT instead of OP_DETECTOR.
/// expected_detectors/expected_observables carry noiseless reference parities
/// for error syndrome normalization.
[[nodiscard]] CompiledModule lower(const HirModule& hir,
                                   std::span<const uint8_t> postselection_mask = {},
                                   std::span<const uint8_t> expected_detectors = {},
                                   std::span<const uint8_t> expected_observables = {});

}  // namespace clifft
