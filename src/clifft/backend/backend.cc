#include "clifft/backend/backend.h"

#include "clifft/backend/compiler_context.h"
#include "clifft/util/mask_view.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <climits>
#include <cmath>
#include <numbers>
#include <span>

namespace clifft {

using internal::CompilerContext;
using internal::LocalizationResult;
using internal::LocalizedBasis;

constexpr double kInvSqrt2 = std::numbers::sqrt2 / 2.0;

// =========================================================================
// Instruction factories
// =========================================================================

Instruction make_frame_cnot(uint16_t ctrl, uint16_t tgt) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_CNOT;
    i.axis_1 = ctrl;
    i.axis_2 = tgt;
    return i;
}

Instruction make_frame_cz(uint16_t a, uint16_t b) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_CZ;
    i.axis_1 = a;
    i.axis_2 = b;
    return i;
}

Instruction make_frame_h(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_H;
    i.axis_1 = v;
    return i;
}

Instruction make_frame_s(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_S;
    i.axis_1 = v;
    return i;
}

Instruction make_frame_s_dag(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_S_DAG;
    i.axis_1 = v;
    return i;
}

Instruction make_frame_swap(uint16_t a, uint16_t b) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_SWAP;
    i.axis_1 = a;
    i.axis_2 = b;
    return i;
}

Instruction make_array_cnot(uint16_t ctrl_axis, uint16_t tgt_axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_CNOT;
    i.axis_1 = ctrl_axis;
    i.axis_2 = tgt_axis;
    return i;
}

Instruction make_array_cz(uint16_t a_axis, uint16_t b_axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_CZ;
    i.axis_1 = a_axis;
    i.axis_2 = b_axis;
    return i;
}

Instruction make_array_swap(uint16_t a, uint16_t b) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_SWAP;
    i.axis_1 = a;
    i.axis_2 = b;
    return i;
}

Instruction make_array_multi_cnot(uint16_t target, uint64_t ctrl_mask) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_MULTI_CNOT;
    i.axis_1 = target;
    i.multi_gate.mask = ctrl_mask;
    return i;
}

Instruction make_array_multi_cz(uint16_t control, uint64_t target_mask) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_MULTI_CZ;
    i.axis_1 = control;
    i.multi_gate.mask = target_mask;
    return i;
}

Instruction make_array_h(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_H;
    i.axis_1 = axis;
    return i;
}

Instruction make_array_s(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_S;
    i.axis_1 = axis;
    return i;
}

Instruction make_array_s_dag(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_S_DAG;
    i.axis_1 = axis;
    return i;
}

Instruction make_expand(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_EXPAND;
    i.axis_1 = axis;
    return i;
}

Instruction make_array_t(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_T;
    i.axis_1 = axis;
    return i;
}

Instruction make_array_t_dag(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_T_DAG;
    i.axis_1 = axis;
    return i;
}

Instruction make_expand_t(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_EXPAND_T;
    i.axis_1 = axis;
    return i;
}

Instruction make_expand_t_dag(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_EXPAND_T_DAG;
    i.axis_1 = axis;
    return i;
}

Instruction make_array_rot(uint16_t axis, double re, double im) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_ROT;
    i.axis_1 = axis;
    i.math.weight_re = re;
    i.math.weight_im = im;
    return i;
}

Instruction make_expand_rot(uint16_t axis, double re, double im) {
    Instruction i{};
    i.opcode = Opcode::OP_EXPAND_ROT;
    i.axis_1 = axis;
    i.math.weight_re = re;
    i.math.weight_im = im;
    return i;
}

Instruction make_array_u2(uint16_t axis, uint32_t cp_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_U2;
    i.axis_1 = axis;
    i.u2.cp_idx = cp_idx;
    return i;
}

Instruction make_array_u4(uint16_t axis_lo, uint16_t axis_hi, uint32_t cp_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_U4;
    i.axis_1 = axis_lo;
    i.axis_2 = axis_hi;
    i.u4.cp_idx = cp_idx;
    return i;
}

Instruction make_swap_meas_interfere(uint16_t swap_from, uint16_t swap_to, uint32_t classical_idx,
                                     bool sign) {
    Instruction i{};
    i.opcode = Opcode::OP_SWAP_MEAS_INTERFERE;
    i.axis_1 = swap_from;
    i.axis_2 = swap_to;
    i.classical.classical_idx = classical_idx;
    if (sign) {
        i.flags |= Instruction::FLAG_SIGN;
    }
    return i;
}

Instruction make_meas(Opcode meas_opcode, uint16_t axis, uint32_t classical_idx, bool sign) {
    Instruction i{};
    i.opcode = meas_opcode;
    i.axis_1 = axis;
    i.classical.classical_idx = classical_idx;
    if (sign) {
        i.flags |= Instruction::FLAG_SIGN;
    }
    return i;
}

Instruction make_apply_pauli(uint32_t cp_mask_idx, uint32_t condition_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_APPLY_PAULI;
    i.pauli.cp_mask_idx = cp_mask_idx;
    i.pauli.condition_idx = condition_idx;
    return i;
}

Instruction make_noise(uint32_t site_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_NOISE;
    i.pauli.cp_mask_idx = site_idx;
    return i;
}

Instruction make_noise_block(uint32_t start_site, uint32_t count) {
    Instruction i{};
    i.opcode = Opcode::OP_NOISE_BLOCK;
    i.pauli.cp_mask_idx = start_site;
    i.pauli.condition_idx = count;
    return i;
}

Instruction make_readout_noise(uint32_t entry_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_READOUT_NOISE;
    i.pauli.cp_mask_idx = entry_idx;
    return i;
}

Instruction make_detector(uint32_t det_list_idx, uint32_t classical_idx, ExpectedParity expected) {
    Instruction i{};
    i.opcode = Opcode::OP_DETECTOR;
    i.pauli.cp_mask_idx = det_list_idx;
    i.pauli.condition_idx = classical_idx;
    if (expected == ExpectedParity::One) {
        i.flags |= Instruction::FLAG_EXPECTED_ONE;
    }
    return i;
}

Instruction make_postselect(uint32_t det_list_idx, uint32_t classical_idx,
                            ExpectedParity expected) {
    Instruction i{};
    i.opcode = Opcode::OP_POSTSELECT;
    i.pauli.cp_mask_idx = det_list_idx;
    i.pauli.condition_idx = classical_idx;
    if (expected == ExpectedParity::One) {
        i.flags |= Instruction::FLAG_EXPECTED_ONE;
    }
    return i;
}

Instruction make_observable(uint32_t target_list_idx, uint32_t obs_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_OBSERVABLE;
    i.pauli.cp_mask_idx = target_list_idx;
    i.pauli.condition_idx = obs_idx;
    return i;
}

Instruction make_exp_val(uint32_t cp_exp_val_idx, uint32_t exp_val_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_EXP_VAL;
    i.exp_val.cp_exp_val_idx = cp_exp_val_idx;
    i.exp_val.exp_val_idx = exp_val_idx;
    return i;
}

namespace {

// =========================================================================
// Emit helpers: push gate to pending queue and emit VM opcode
// =========================================================================

void emit_cnot(CompilerContext& ctx, uint16_t ctrl, uint16_t tgt) {
    ctx.virtual_frame.append_gate({internal::PendingGateType::CNOT, ctrl, tgt});

    uint32_t k = ctx.reg_manager.active_k();
    if (ctrl >= k) {
        ctx.emit(make_frame_cnot(ctrl, tgt));
    } else {
        assert(tgt < k && "CNOT with active control but dormant target should never occur");
        ctx.emit(make_array_cnot(ctrl, tgt));
    }
}

void emit_cz(CompilerContext& ctx, uint16_t a, uint16_t b) {
    ctx.virtual_frame.append_gate({internal::PendingGateType::CZ, a, b});

    uint32_t k = ctx.reg_manager.active_k();
    if (a >= k || b >= k) {
        ctx.emit(make_frame_cz(a, b));
    } else {
        ctx.emit(make_array_cz(a, b));
    }
}

void emit_s(CompilerContext& ctx, uint16_t v) {
    ctx.virtual_frame.append_gate({internal::PendingGateType::S, v, 0});

    if (v < ctx.reg_manager.active_k()) {
        ctx.emit(make_array_s(v));
    } else {
        ctx.emit(make_frame_s(v));
    }
}

void emit_swap(CompilerContext& ctx, uint16_t a, uint16_t b) {
    if (a == b)
        return;

    uint32_t k = ctx.reg_manager.active_k();
    assert((a < k) == (b < k) && "Cannot swap between active and dormant partitions");

    ctx.virtual_frame.append_gate({internal::PendingGateType::SWAP, a, b});

    if (a < k && b < k) {
        ctx.emit(make_array_swap(a, b));
    } else {
        ctx.emit(make_frame_swap(a, b));
    }
}

// Map an HIR Pauli (at t=0) into the current virtual frame.
stim::PauliString<kStimWidth> map_to_virtual(CompilerContext& ctx, MaskView destab_mask,
                                             MaskView stab_mask, bool sign, uint32_t n) {
    return ctx.virtual_frame.map_pauli(destab_mask, stab_mask, sign, n);
}

// Route a localized Pauli to the Z basis on an active axis.
// Handles three cases:
//   - Dormant Z: no-op (already diagonal in dormant frame)
//   - Dormant X: swap to next active slot, H into Z, expand
//   - Active X: H into Z on the array
// Mutates result.pivot if a swap is needed.
void route_to_active_z(CompilerContext& ctx, LocalizationResult& result) {
    bool is_dormant = result.pivot >= ctx.reg_manager.active_k();

    if (is_dormant && result.basis == LocalizedBasis::Z_BASIS) {
        return;
    }

    if (is_dormant) {
        uint16_t next_axis = static_cast<uint16_t>(ctx.reg_manager.active_k());

        if (result.pivot != next_axis) {
            emit_swap(ctx, result.pivot, next_axis);
            result.pivot = next_axis;
        }

        ctx.virtual_frame.append_gate({internal::PendingGateType::H, result.pivot, 0});

        ctx.emit(make_frame_h(result.pivot));
        ctx.emit(make_expand(result.pivot));
        ctx.reg_manager.activate();
        return;
    }

    if (result.basis == LocalizedBasis::X_BASIS) {
        ctx.virtual_frame.append_gate({internal::PendingGateType::H, result.pivot, 0});
        ctx.emit(make_array_h(result.pivot));
    }
}

}  // namespace

namespace internal {

// =========================================================================
// localize_pauli: Greedy O(n) Pauli reduction
// =========================================================================
//
// Implements the constructive proof from Lemma (Pauli Localization).
// Given a non-identity PauliString P = X^x Z^z, computes virtual Clifford V
// such that V P V^dag = (+/-)P_v where P_v in {X_v, Z_v}.

// Pick a pivot from the dormant region (axes >= k), falling back to the
// lowest set bit overall if every set bit is in the active region. Caller
// asserts `bits` is non-empty so the fallback always finds a bit.
static uint32_t pick_dormant_pivot(MaskView bits, uint32_t k) {
    uint32_t r = lowest_bit_at_or_above(bits, k);
    return (r < bits.num_words() * 64) ? r : bits.lowest_bit();
}

// Pick a pivot from the active region (axes < k), falling back to the
// lowest set bit overall.
static uint32_t pick_active_pivot(MaskView bits, uint32_t k) {
    uint32_t r = lowest_bit_below(bits, k);
    return (r < bits.num_words() * 64) ? r : bits.lowest_bit();
}

LocalizationResult localize_pauli(CompilerContext& ctx,
                                  const stim::PauliString<kStimWidth>& pauli) {
    const uint32_t n = ctx.reg_manager.num_qubits();
    const uint32_t words = (n + 63) / 64;

    // Runtime-width scratch storage. PauliBitMask (BitMask<128>) would
    // truncate the input for circuits with n > kMaxInlineQubits, hiding
    // bits in qubits 128+ and producing a spuriously-empty Pauli that
    // tripped the identity-Pauli assertion below.
    std::vector<uint64_t> x_bits(words, 0);
    std::vector<uint64_t> z_bits(words, 0);
    MutableMaskView x_view{std::span<uint64_t>(x_bits)};
    MutableMaskView z_view{std::span<uint64_t>(z_bits)};
    for (uint32_t w = 0; w < words; ++w) {
        x_view.words[w] = pauli.xs.u64[w];
        z_view.words[w] = pauli.zs.u64[w];
    }

    assert((!x_view.is_zero() || !z_view.is_zero()) && "Cannot localize identity Pauli");

    uint32_t pivot;
    LocalizedBasis basis;
    bool sign = pauli.sign;

    // Gates are pushed to the pending queue (no transpose needed).

    uint32_t k = ctx.reg_manager.active_k();

    if (!x_view.is_zero()) {
        // =============================================================
        // Case 1: X-support is non-empty.
        // =============================================================

        // Pick pivot from X-support, preferring dormant axes (>= k).
        pivot = pick_dormant_pivot(x_view, k);
        bool z_pivot = z_view.bit_get(pivot);

        // X-localization: CNOT(pivot -> q) annihilates X_q.
        // Z propagation: Z_t -> Z_c Z_t, so z_pivot ^= z_q.
        // Sign rule: CNOT(c,t) flips sign iff x_c & z_t & (x_t ^ z_c ^ 1).
        // Here x_pivot=1, x_q=1, so: sign ^= z_q & z_pivot.
        std::vector<uint64_t> to_clear_x_buf(x_bits);
        MutableMaskView to_clear_x{std::span<uint64_t>(to_clear_x_buf)};
        to_clear_x.bit_set(pivot, false);
        while (!to_clear_x.is_zero()) {
            uint32_t q = to_clear_x.lowest_bit();
            emit_cnot(ctx, static_cast<uint16_t>(pivot), static_cast<uint16_t>(q));
            if (z_view.bit_get(q)) {
                if (z_pivot)
                    sign ^= true;
                z_pivot ^= true;
            }
            to_clear_x.clear_lowest_bit();
        }

        // Z-localization: CZ(pivot, q) clears residual Z_q.
        // Sign rule: CZ(a,b) flips sign iff x_a & x_b & (z_a ^ z_b).
        // Here x_q=0 (already cleared), so sign NEVER flips.
        std::vector<uint64_t> to_clear_z_buf(z_bits);
        MutableMaskView to_clear_z{std::span<uint64_t>(to_clear_z_buf)};
        to_clear_z.bit_set(pivot, false);
        while (!to_clear_z.is_zero()) {
            uint32_t q = to_clear_z.lowest_bit();
            emit_cz(ctx, static_cast<uint16_t>(pivot), static_cast<uint16_t>(q));
            to_clear_z.clear_lowest_bit();
        }

        // Y_pivot -> S Y S^dag = -X. Emit S to resolve.
        // Sign rule: S(v) flips sign iff x_v & z_v. Both are 1 here.
        if (z_pivot) {
            emit_s(ctx, static_cast<uint16_t>(pivot));
            sign ^= true;
        }

        basis = LocalizedBasis::X_BASIS;

    } else {
        // =============================================================
        // Case 2: Pure Z-string (x_bits == 0, z_bits != 0).
        // =============================================================

        // Pick pivot from Z-support, preferring active axes (< k).
        pivot = pick_active_pivot(z_view, k);

        // Z-localization: CNOT(q -> pivot) folds Z_q onto pivot.
        // Sign rule: CNOT(c,t) flips sign iff x_c & z_t & (x_t ^ z_c ^ 1).
        // Here x_c=x_q=0, so sign NEVER flips.
        std::vector<uint64_t> to_clear_z_buf(z_bits);
        MutableMaskView to_clear_z{std::span<uint64_t>(to_clear_z_buf)};
        to_clear_z.bit_set(pivot, false);
        while (!to_clear_z.is_zero()) {
            uint32_t q = to_clear_z.lowest_bit();
            emit_cnot(ctx, static_cast<uint16_t>(q), static_cast<uint16_t>(pivot));
            to_clear_z.clear_lowest_bit();
        }

        basis = LocalizedBasis::Z_BASIS;
    }

    return {static_cast<uint16_t>(pivot), basis, sign};
}

}  // namespace internal

// =========================================================================
// lower(): Full pipeline wiring
// =========================================================================

CompiledModule lower(const HirModule& hir, std::span<const uint8_t> postselection_mask,
                     std::span<const uint8_t> expected_detectors,
                     std::span<const uint8_t> expected_observables) {
    using internal::CompilerContext;
    using internal::localize_pauli;
    using internal::LocalizedBasis;

    const uint32_t n = hir.num_qubits;
    // Two ceilings:
    //   - kMaxInlineQubits: SVM frame storage (state.p_x / p_z) is still a
    //     fixed-width BitMask<kMaxInlineQubits>, so any conditional Pauli
    //     or noise channel touching higher qubits would be silently dropped
    //     at execution time. Lifting this requires runtime-width SVM frame
    //     storage (planned for the next migration PR).
    //   - 65536: bytecode axis operands are uint16_t. trace() enforces
    //     this; lower() repeats the check defensively.
    if (n > kMaxInlineQubits) {
        throw std::runtime_error(
            "Circuit num_qubits (" + std::to_string(n) + ") exceeds the SVM frame width (" +
            std::to_string(kMaxInlineQubits) +
            "). The HIR supports wider circuits but the runtime frame does not yet; "
            "lifting this gate is the subject of a follow-up migration PR.");
    }
    if (n > 65536) {
        throw std::runtime_error("Circuit exceeds 65536-qubit VM axis limit: " + std::to_string(n) +
                                 " qubits");
    }
    CompilerContext ctx(n);
    uint32_t det_emit_idx = 0;
    uint32_t total_meas_slots = hir.num_measurements + hir.num_hidden_measurements;

    // Pre-size ConstantPool arenas. CONDITIONAL_PAULI emits exactly one
    // OP_APPLY_PAULI per op; EXP_VAL emits exactly one OP_EXP_VAL. Each
    // HIR noise channel maps into one ConstantPool channel slot.
    size_t num_apply_paulis = 0;
    size_t num_exp_val_masks = 0;
    for (const auto& op : hir.ops) {
        if (op.op_type() == OpType::CONDITIONAL_PAULI)
            ++num_apply_paulis;
        else if (op.op_type() == OpType::EXP_VAL)
            ++num_exp_val_masks;
    }
    size_t num_noise_channels = 0;
    for (const auto& site : hir.noise_sites)
        num_noise_channels += site.channels.size();
    ctx.constant_pool.pauli_masks = PauliMaskArena(n, num_apply_paulis);
    ctx.constant_pool.exp_val_masks = PauliMaskArena(n, num_exp_val_masks);
    ctx.constant_pool.noise_channel_masks = PauliMaskArena(n, num_noise_channels);
    size_t next_cp_pauli = 0;
    size_t next_cp_exp_val = 0;
    size_t next_cp_noise = 0;

    bool has_source_map = hir.source_map.size() == hir.ops.size();
    bool has_postselection = false;

    for (size_t op_idx = 0; op_idx < hir.ops.size(); ++op_idx) {
        const auto& op = hir.ops[op_idx];
        size_t bc_before = ctx.bytecode.size();

        switch (op.op_type()) {
            case OpType::T_GATE: {
                auto p_v =
                    map_to_virtual(ctx, hir.destab_mask(op), hir.stab_mask(op), hir.sign(op), n);
                auto result = localize_pauli(ctx, p_v);

                route_to_active_z(ctx, result);

                // Global phase correction when localization inverted the Pauli (-Z).
                // T on -Z = e^{i*pi/4} T^dag; T^dag on -Z = e^{-i*pi/4} T.
                if (result.sign) {
                    if (!op.is_dagger()) {
                        ctx.constant_pool.global_weight *=
                            std::complex<double>(kInvSqrt2, kInvSqrt2);
                    } else {
                        ctx.constant_pool.global_weight *=
                            std::complex<double>(kInvSqrt2, -kInvSqrt2);
                    }
                }

                bool phase_flip = result.sign ^ op.is_dagger();
                if (phase_flip) {
                    ctx.emit(make_array_t_dag(result.pivot));
                } else {
                    ctx.emit(make_array_t(result.pivot));
                }

                break;
            }

            case OpType::PHASE_ROTATION: {
                double alpha = op.alpha();
                auto p_v =
                    map_to_virtual(ctx, hir.destab_mask(op), hir.stab_mask(op), hir.sign(op), n);
                auto result = localize_pauli(ctx, p_v);

                route_to_active_z(ctx, result);

                // The Front-End unconditionally extracted the physical global phase.
                // If localize_pauli dynamically flipped the geometric sign
                // (result.sign != hir.sign(op)), correct the global phase artifact
                // left behind by the VM's diagonal factorization.
                if (result.sign != hir.sign(op)) {
                    double corr = op.alpha() * std::numbers::pi * (hir.sign(op) ? -1.0 : 1.0);
                    ctx.constant_pool.global_weight *=
                        std::complex<double>(std::cos(corr), std::sin(corr));
                }

                // result.sign absorbs both the original hir.sign(op) and any
                // localization flips.
                if (result.sign)
                    alpha = -alpha;

                // Compute relative phase z = e^{i*alpha*pi}
                double angle = alpha * std::numbers::pi;
                double z_re = std::cos(angle);
                double z_im = std::sin(angle);

                ctx.emit(make_array_rot(result.pivot, z_re, z_im));

                break;
            }

            case OpType::MEASURE: {
                uint32_t classical_idx = static_cast<uint32_t>(op.meas_record_idx());

                auto p_v =
                    map_to_virtual(ctx, hir.destab_mask(op), hir.stab_mask(op), hir.sign(op), n);

                // Identity Pauli check: walk the full mapped width directly,
                // not through a fixed-width PauliBitMask intermediate (which
                // would silently treat any high-qubit-only support as zero
                // and emit a deterministic measurement).
                const uint32_t pv_words = (n + 63) / 64;
                bool is_identity = true;
                for (uint32_t w = 0; w < pv_words; ++w) {
                    if (p_v.xs.u64[w] != 0 || p_v.zs.u64[w] != 0) {
                        is_identity = false;
                        break;
                    }
                }
                if (is_identity) {
                    // Identity Pauli: outcome is determined solely by the sign.
                    Instruction id_meas =
                        make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, classical_idx, p_v.sign);
                    id_meas.flags |= Instruction::FLAG_IDENTITY;
                    ctx.emit(id_meas);
                    break;
                }

                auto result = localize_pauli(ctx, p_v);
                bool is_active = result.pivot < ctx.reg_manager.active_k();

                if (!is_active) {
                    // Dormant pivot
                    if (result.basis == LocalizedBasis::Z_BASIS) {
                        ctx.emit(make_meas(Opcode::OP_MEAS_DORMANT_STATIC, result.pivot,
                                           classical_idx, result.sign));
                    } else {
                        ctx.emit(make_meas(Opcode::OP_MEAS_DORMANT_RANDOM, result.pivot,
                                           classical_idx, result.sign));
                    }
                    // X_BASIS post-measurement: append virtual H to align coordinate system
                    if (result.basis == LocalizedBasis::X_BASIS) {
                        ctx.virtual_frame.append_gate(
                            {internal::PendingGateType::H, result.pivot, 0});
                    }
                } else {
                    // Active pivot: compact to k-1 before measurement
                    uint16_t top = static_cast<uint16_t>(ctx.reg_manager.active_k() - 1);

                    if (result.pivot != top) {
                        emit_swap(ctx, result.pivot, top);
                        result.pivot = top;
                    }

                    if (result.basis == LocalizedBasis::Z_BASIS) {
                        ctx.emit(make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL, result.pivot,
                                           classical_idx, result.sign));
                    } else {
                        ctx.emit(make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE, result.pivot,
                                           classical_idx, result.sign));
                    }

                    if (result.basis == LocalizedBasis::X_BASIS) {
                        ctx.virtual_frame.append_gate(
                            {internal::PendingGateType::H, result.pivot, 0});
                    }

                    ctx.reg_manager.deactivate();
                }

                break;
            }

            case OpType::CONDITIONAL_PAULI: {
                auto p_v =
                    map_to_virtual(ctx, hir.destab_mask(op), hir.stab_mask(op), hir.sign(op), n);

                auto cp_handle = static_cast<PauliMaskHandle>(next_cp_pauli++);
                auto slot = ctx.constant_pool.pauli_masks.mut_at(cp_handle);
                stim_to_mask_view(p_v.xs, n, slot.x());
                stim_to_mask_view(p_v.zs, n, slot.z());
                slot.set_sign(p_v.sign);

                uint32_t cond_idx = static_cast<uint32_t>(op.controlling_meas());

                ctx.emit(make_apply_pauli(static_cast<uint32_t>(cp_handle), cond_idx));
                break;
            }

            case OpType::NOISE: {
                auto site_idx = static_cast<uint32_t>(op.noise_site_idx());
                assert(site_idx < hir.noise_sites.size());
                const auto& hir_site = hir.noise_sites[site_idx];

                NoiseSite mapped_site;
                for (const auto& ch : hir_site.channels) {
                    auto in_view = hir.noise_channel_masks.at(ch.mask);
                    auto out_handle = static_cast<PauliMaskHandle>(next_cp_noise++);
                    auto out_slot = ctx.constant_pool.noise_channel_masks.mut_at(out_handle);
                    ctx.virtual_frame.map_noise_channel(in_view.x(), in_view.z(), out_slot.x(),
                                                        out_slot.z(), n);
                    mapped_site.channels.push_back(NoiseChannel{out_handle, ch.prob});
                }

                // Compute total channel probability for gap sampling
                double prob_sum = 0.0;
                for (const auto& ch : mapped_site.channels) {
                    prob_sum += ch.prob;
                }

                uint32_t cp_idx = static_cast<uint32_t>(ctx.constant_pool.noise_sites.size());
                ctx.constant_pool.noise_sites.push_back(std::move(mapped_site));
                ctx.emit(make_noise(cp_idx));

                // Accumulate hazard for exponential gap sampling.
                // log1p(-x) avoids catastrophic cancellation when prob_sum is near zero,
                // where log(1 - x) would lose significant digits.
                // Clamp to 1 - 2^-53 (one ULP below 1.0 in double precision).
                // This matches the maximum value of random_double() = (rng() >> 11) * 2^-53,
                // preventing log1p(-1) = -inf when prob_sum rounds to exactly 1.0.
                ctx.noise_hazards_accum += -std::log1p(-std::min(prob_sum, 1.0 - 0x1.0p-53));
                ctx.constant_pool.noise_hazards.push_back(ctx.noise_hazards_accum);
                break;
            }

            case OpType::READOUT_NOISE: {
                auto entry_idx = static_cast<uint32_t>(op.readout_noise_idx());
                assert(entry_idx < hir.readout_noise.size());
                const auto& entry = hir.readout_noise[entry_idx];

                uint32_t cp_idx = static_cast<uint32_t>(ctx.constant_pool.readout_noise.size());
                ctx.constant_pool.readout_noise.push_back(entry);
                ctx.emit(make_readout_noise(cp_idx));
                break;
            }

            case OpType::DETECTOR: {
                auto det_idx = static_cast<uint32_t>(op.detector_idx());
                assert(det_idx < hir.detector_targets.size());
                const auto& targets = hir.detector_targets[det_idx];

                uint32_t cp_idx = static_cast<uint32_t>(ctx.constant_pool.detector_targets.size());
                ctx.constant_pool.detector_targets.push_back(targets);

                bool is_postselected = det_emit_idx < postselection_mask.size() &&
                                       postselection_mask[det_emit_idx] != 0;
                ExpectedParity exp = (det_emit_idx < expected_detectors.size() &&
                                      expected_detectors[det_emit_idx] != 0)
                                         ? ExpectedParity::One
                                         : ExpectedParity::Zero;
                if (is_postselected) {
                    ctx.emit(make_postselect(cp_idx, det_emit_idx, exp));
                    has_postselection = true;
                } else {
                    ctx.emit(make_detector(cp_idx, det_emit_idx, exp));
                }
                ++det_emit_idx;
                break;
            }

            case OpType::OBSERVABLE: {
                auto target_list_idx = op.observable_target_list_idx();
                assert(target_list_idx < hir.observable_targets.size());
                const auto& targets = hir.observable_targets[target_list_idx];

                uint32_t cp_idx =
                    static_cast<uint32_t>(ctx.constant_pool.observable_targets.size());
                ctx.constant_pool.observable_targets.push_back(targets);

                auto obs_idx = static_cast<uint32_t>(op.observable_idx());
                ctx.emit(make_observable(cp_idx, obs_idx));
                break;
            }

            case OpType::EXP_VAL: {
                auto p_v =
                    map_to_virtual(ctx, hir.destab_mask(op), hir.stab_mask(op), hir.sign(op), n);

                auto cp_handle = static_cast<PauliMaskHandle>(next_cp_exp_val++);
                auto slot = ctx.constant_pool.exp_val_masks.mut_at(cp_handle);
                stim_to_mask_view(p_v.xs, n, slot.x());
                stim_to_mask_view(p_v.zs, n, slot.z());
                slot.set_sign(p_v.sign);

                ctx.emit(make_exp_val(static_cast<uint32_t>(cp_handle),
                                      static_cast<uint32_t>(op.exp_val_idx())));
                break;
            }

            case OpType::NUM_OP_TYPES:
#if defined(__GNUC__) || defined(__clang__)
                __builtin_unreachable();
#elif defined(_MSC_VER)
                __assume(0);
#endif
        }

        // Tag all instructions emitted by this HIR op with source lines and k
        size_t bc_after = ctx.bytecode.size();
        size_t emitted = bc_after - bc_before;
        if (has_source_map) {
            const auto& lines = hir.source_map[op_idx];
            for (size_t e = 0; e < emitted; ++e) {
                ctx.source_map.append(lines, ctx.emit_k_history[bc_before + e]);
            }
        }
    }

    // Compute final tableau U_C = U_phys * V_cum^{-1}.
    // A.then(B) evaluates to B * A in matrix multiplication.
    // We want U_phys * V_cum^{-1}, so: v_cum_inv.then(U_phys) = U_phys * V_cum^{-1}.
    auto& final_v_cum = ctx.virtual_frame.mutable_materialized_tableau();
    if (hir.final_tableau.has_value()) {
        stim::Tableau<kStimWidth> v_cum_inv = final_v_cum.inverse();
        ctx.constant_pool.final_tableau = v_cum_inv.then(*hir.final_tableau);
    }

    ctx.constant_pool.global_weight *= hir.global_weight;

    uint16_t peak = ctx.reg_manager.peak_k();
    if (peak >= 63) {
        throw std::runtime_error("peak active rank (" + std::to_string(peak) +
                                 ") >= 63: would cause undefined behavior in SVM 1ULL << k shifts");
    }

    CompiledModule result;
    result.bytecode = std::move(ctx.bytecode);
    result.constant_pool = std::move(ctx.constant_pool);
    result.source_map = std::move(ctx.source_map);
    result.num_qubits = hir.num_qubits;
    result.peak_rank = peak;
    result.num_measurements = hir.num_measurements;
    result.total_meas_slots = total_meas_slots;
    result.num_detectors = hir.num_detectors;
    result.num_observables = hir.num_observables;
    result.num_exp_vals = hir.num_exp_vals;
    result.has_postselection = has_postselection;
    result.expected_observables.assign(expected_observables.begin(), expected_observables.end());

    return result;
}

}  // namespace clifft
