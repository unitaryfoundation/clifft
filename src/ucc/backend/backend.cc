#include "ucc/backend/backend.h"

#include "ucc/backend/compiler_context.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>

namespace ucc {

using internal::CompilerContext;
using internal::CompressedBasis;
using internal::CompressionResult;

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

Instruction make_phase_t(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_PHASE_T;
    i.axis_1 = axis;
    return i;
}

Instruction make_phase_t_dag(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_PHASE_T_DAG;
    i.axis_1 = axis;
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

Instruction make_readout_noise(uint32_t entry_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_READOUT_NOISE;
    i.pauli.cp_mask_idx = entry_idx;
    return i;
}

Instruction make_detector(uint32_t det_list_idx, uint32_t classical_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_DETECTOR;
    i.pauli.cp_mask_idx = det_list_idx;
    i.pauli.condition_idx = classical_idx;
    return i;
}

Instruction make_postselect(uint32_t det_list_idx, uint32_t classical_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_POSTSELECT;
    i.pauli.cp_mask_idx = det_list_idx;
    i.pauli.condition_idx = classical_idx;
    return i;
}

Instruction make_observable(uint32_t target_list_idx, uint32_t obs_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_OBSERVABLE;
    i.pauli.cp_mask_idx = target_list_idx;
    i.pauli.condition_idx = obs_idx;
    return i;
}

namespace {

// =========================================================================
// Emit helpers: append gate to V_cum and emit RISC opcode
// =========================================================================

// Emit helpers accept pre-transposed tableau wrappers by reference so that
// the O(n^2) transpose happens once per compress_pauli call, not per gate.

void emit_cnot(CompilerContext& ctx, stim::TableauTransposedRaii<kStimWidth>& trans_cum,
               stim::TableauTransposedRaii<kStimWidth>& trans_local, uint16_t ctrl, uint16_t tgt) {
    trans_cum.append_ZCX(ctrl, tgt);
    trans_local.append_ZCX(ctrl, tgt);

    uint32_t k = ctx.reg_manager.active_k();
    if (ctrl >= k) {
        // Dormant control: CNOT is identity on the array.
        ctx.bytecode.push_back(make_frame_cnot(ctrl, tgt));
    } else {
        assert(tgt < k && "CNOT with active control but dormant target should never occur");
        ctx.bytecode.push_back(make_array_cnot(ctrl, tgt));
    }
}

void emit_cz(CompilerContext& ctx, stim::TableauTransposedRaii<kStimWidth>& trans_cum,
             stim::TableauTransposedRaii<kStimWidth>& trans_local, uint16_t a, uint16_t b) {
    trans_cum.append_ZCZ(a, b);
    trans_local.append_ZCZ(a, b);

    uint32_t k = ctx.reg_manager.active_k();
    if (a >= k || b >= k) {
        // Either operand dormant: CZ is identity on the array.
        ctx.bytecode.push_back(make_frame_cz(a, b));
    } else {
        ctx.bytecode.push_back(make_array_cz(a, b));
    }
}

void emit_s(CompilerContext& ctx, stim::TableauTransposedRaii<kStimWidth>& trans_cum,
            stim::TableauTransposedRaii<kStimWidth>& trans_local, uint16_t v) {
    trans_cum.append_S(v);
    trans_local.append_S(v);

    if (v < ctx.reg_manager.active_k()) {
        ctx.bytecode.push_back(make_array_s(v));
    } else {
        ctx.bytecode.push_back(make_frame_s(v));
    }
}

// Emit a logical SWAP using a caller-provided transposed tableau scope,
// avoiding a redundant O(n^2) transpose when the caller already holds one.
void emit_swap(CompilerContext& ctx, stim::TableauTransposedRaii<kStimWidth>& trans_cum, uint16_t a,
               uint16_t b) {
    if (a == b)
        return;

    uint32_t k = ctx.reg_manager.active_k();
    assert((a < k) == (b < k) && "Cannot swap between active and dormant partitions");

    trans_cum.append_ZCX(a, b);
    trans_cum.append_ZCX(b, a);
    trans_cum.append_ZCX(a, b);

    if (a < k && b < k) {
        ctx.bytecode.push_back(make_array_swap(a, b));
    } else {
        ctx.bytecode.push_back(make_frame_swap(a, b));
    }
}

// Map an HIR Pauli (at t=0) into the current virtual frame: P_v = V_cum(P_t0).
stim::PauliString<kStimWidth> map_to_virtual(const CompilerContext& ctx, uint64_t destab_mask,
                                             uint64_t stab_mask, bool sign, uint32_t n) {
    stim::PauliString<kStimWidth> p(n);
    p.xs.u64[0] = destab_mask;
    p.zs.u64[0] = stab_mask;
    p.sign = sign;
    return ctx.v_cum(p);
}

}  // namespace

namespace internal {

// =========================================================================
// compress_pauli: Greedy O(n) Pauli reduction
// =========================================================================
//
// Implements the constructive proof from Lemma (Pauli Compression).
// Given a non-identity PauliString P = X^x Z^z, computes virtual Clifford V
// such that V P V^dag = (+/-)P_v where P_v in {X_v, Z_v}.

CompressionResult compress_pauli(CompilerContext& ctx, const stim::PauliString<kStimWidth>& pauli) {
    const uint32_t n = ctx.reg_manager.num_qubits();

    static_assert(kStimWidth == 64, "compress_pauli assumes 64-bit width");
    uint64_t x_bits = pauli.xs.u64[0];
    uint64_t z_bits = pauli.zs.u64[0];

    assert((x_bits | z_bits) != 0 && "Cannot compress identity Pauli");

    // Reset reusable scratch tableau to identity for this compression.
    ctx.v_local = ctx.v_local_identity;

    uint16_t pivot;
    CompressedBasis basis;

    // Hoist the O(n^2) transpose into a single scope. All O(n) gate
    // emissions use the pre-transposed wrappers, keeping the total
    // complexity at O(n^2) rather than O(n^3).
    {
        stim::TableauTransposedRaii<kStimWidth> trans_cum(ctx.v_cum);
        stim::TableauTransposedRaii<kStimWidth> trans_local(ctx.v_local);

        uint32_t k = ctx.reg_manager.active_k();
        uint64_t active_mask = (k == 0) ? 0ULL : ((1ULL << k) - 1);

        if (x_bits != 0) {
            // =============================================================
            // Case 1: X-support is non-empty.
            // =============================================================

            // Pick pivot from X-support, preferring dormant axes (>= k).
            uint64_t x_dormant = x_bits & ~active_mask;
            if (x_dormant != 0) {
                pivot = static_cast<uint16_t>(std::countr_zero(x_dormant));
            } else {
                pivot = static_cast<uint16_t>(std::countr_zero(x_bits));
            }

            // X-compression: CNOT(pivot -> q) annihilates X_q.
            // Z propagation: Z_t -> Z_c Z_t, so z_pivot ^= z_q.
            uint64_t to_clear_x = x_bits & ~(1ULL << pivot);
            while (to_clear_x != 0) {
                uint16_t q = static_cast<uint16_t>(std::countr_zero(to_clear_x));
                emit_cnot(ctx, trans_cum, trans_local, pivot, q);
                if (z_bits & (1ULL << q)) {
                    z_bits ^= (1ULL << pivot);
                }
                to_clear_x &= to_clear_x - 1;
            }

            // Z-compression: CZ(pivot, q) clears residual Z_q.
            uint64_t to_clear_z = z_bits & ~(1ULL << pivot);
            while (to_clear_z != 0) {
                uint16_t q = static_cast<uint16_t>(std::countr_zero(to_clear_z));
                emit_cz(ctx, trans_cum, trans_local, pivot, q);
                to_clear_z &= to_clear_z - 1;
            }

            // Y_pivot -> S Y S^dag = -X. Emit S to resolve.
            if (z_bits & (1ULL << pivot)) {
                emit_s(ctx, trans_cum, trans_local, pivot);
            }

            basis = CompressedBasis::X_BASIS;

        } else {
            // =============================================================
            // Case 2: Pure Z-string (x_bits == 0, z_bits != 0).
            // =============================================================

            // Pick pivot from Z-support, preferring active axes (< k).
            uint64_t z_active = z_bits & active_mask;
            if (z_active != 0) {
                pivot = static_cast<uint16_t>(std::countr_zero(z_active));
            } else {
                pivot = static_cast<uint16_t>(std::countr_zero(z_bits));
            }

            // Z-compression: CNOT(q -> pivot) folds Z_q onto pivot.
            uint64_t to_clear_z = z_bits & ~(1ULL << pivot);
            while (to_clear_z != 0) {
                uint16_t q = static_cast<uint16_t>(std::countr_zero(to_clear_z));
                emit_cnot(ctx, trans_cum, trans_local, q, pivot);
                to_clear_z &= to_clear_z - 1;
            }

            basis = CompressedBasis::Z_BASIS;
        }
    }  // RAII destructors un-transpose both tableaux here.

    stim::PauliString<kStimWidth> compressed = ctx.v_local(pauli);
    bool sign = compressed.sign;

    return {pivot, basis, sign};
}

}  // namespace internal

// =========================================================================
// lower(): Full pipeline wiring
// =========================================================================

CompiledModule lower(const HirModule& hir, const std::vector<uint8_t>& postselection_mask) {
    using internal::CompilerContext;
    using internal::compress_pauli;
    using internal::CompressedBasis;

    const uint32_t n = hir.num_qubits;
    CompilerContext ctx(n);

    // Track detector/observable emission indices
    uint32_t det_emit_idx = 0;

    // Track the last measurement outcome index for use_last_outcome resolution
    uint32_t last_meas_idx = 0;

    // Count hidden measurements separately (they have their own index space)
    uint32_t hidden_meas_count = 0;
    // Total measurement slots: visible + hidden
    uint32_t total_meas_slots = hir.num_measurements;
    for (const auto& op : hir.ops) {
        if (op.op_type() == OpType::MEASURE && op.is_hidden()) {
            ++hidden_meas_count;
        }
    }
    total_meas_slots += hidden_meas_count;

    // Hidden measurements use indices starting after visible measurements
    uint32_t hidden_meas_emit_idx = hir.num_measurements;

    bool has_source_map = hir.source_map.size() == hir.ops.size();

    for (size_t op_idx = 0; op_idx < hir.ops.size(); ++op_idx) {
        const auto& op = hir.ops[op_idx];
        size_t bc_before = ctx.bytecode.size();

        switch (op.op_type()) {
            case OpType::T_GATE: {
                auto p_v = map_to_virtual(ctx, op.destab_mask(), op.stab_mask(), op.sign(), n);
                auto result = compress_pauli(ctx, p_v);

                bool is_dormant = result.pivot >= ctx.reg_manager.active_k();

                if (is_dormant && result.basis == CompressedBasis::Z_BASIS) {
                    // Dormant Z_v: fall through to PHASE_T emission.
                    // The VM loop naturally skips array elements when v >= active_k.
                } else if (is_dormant) {
                    // Dormant X_v: route to next active axis
                    uint16_t next_axis = static_cast<uint16_t>(ctx.reg_manager.active_k());

                    {
                        stim::TableauTransposedRaii<kStimWidth> trans_cum(ctx.v_cum);
                        if (result.pivot != next_axis) {
                            emit_swap(ctx, trans_cum, result.pivot, next_axis);
                            result.pivot = next_axis;
                        }

                        trans_cum.append_H_XZ(result.pivot);
                    }

                    ctx.bytecode.push_back(make_frame_h(result.pivot));

                    ctx.bytecode.push_back(make_expand(result.pivot));
                    ctx.reg_manager.activate();
                } else if (result.basis == CompressedBasis::X_BASIS) {
                    // Active X_v. Map to Z_v in V_cum and array+frame.
                    {
                        stim::TableauTransposedRaii<kStimWidth> trans(ctx.v_cum);
                        trans.append_H_XZ(result.pivot);
                    }
                    ctx.bytecode.push_back(make_array_h(result.pivot));
                }

                // Emit diagonal phase gate
                bool phase_flip = result.sign ^ op.is_dagger();
                if (phase_flip) {
                    ctx.bytecode.push_back(make_phase_t_dag(result.pivot));
                } else {
                    ctx.bytecode.push_back(make_phase_t(result.pivot));
                }

                break;
            }

            case OpType::MEASURE: {
                // Determine classical output index: visible measurements
                // use the front-end's pre-computed record index; hidden
                // measurements get indices in the separate hidden range.
                uint32_t classical_idx;
                if (op.is_hidden()) {
                    classical_idx = hidden_meas_emit_idx++;
                } else {
                    classical_idx = static_cast<uint32_t>(op.meas_record_idx());
                }

                // Map the t=0 Pauli to virtual frame, then compress
                auto p_v = map_to_virtual(ctx, op.destab_mask(), op.stab_mask(), op.sign(), n);

                // Check for identity Pauli (deterministic measurement = 0)
                uint64_t x_bits = p_v.xs.u64[0];
                uint64_t z_bits = p_v.zs.u64[0];
                if ((x_bits | z_bits) == 0) {
                    // Identity Pauli: outcome is determined solely by the sign
                    Instruction id_meas =
                        make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, classical_idx, p_v.sign);
                    id_meas.flags |= Instruction::FLAG_IDENTITY;
                    ctx.bytecode.push_back(id_meas);
                    last_meas_idx = classical_idx;
                    break;
                }

                auto result = compress_pauli(ctx, p_v);
                bool is_active = result.pivot < ctx.reg_manager.active_k();

                // Classify and emit the measurement opcode
                if (!is_active) {
                    // Dormant pivot
                    if (result.basis == CompressedBasis::Z_BASIS) {
                        ctx.bytecode.push_back(make_meas(Opcode::OP_MEAS_DORMANT_STATIC,
                                                         result.pivot, classical_idx, result.sign));
                    } else {
                        ctx.bytecode.push_back(make_meas(Opcode::OP_MEAS_DORMANT_RANDOM,
                                                         result.pivot, classical_idx, result.sign));
                    }
                    // X_BASIS post-measurement: append virtual H to align coordinate system
                    if (result.basis == CompressedBasis::X_BASIS) {
                        stim::TableauTransposedRaii<kStimWidth> trans(ctx.v_cum);
                        trans.append_H_XZ(result.pivot);
                    }
                } else {
                    // Active pivot: compact to k-1 before measurement
                    uint16_t top = static_cast<uint16_t>(ctx.reg_manager.active_k() - 1);

                    {
                        stim::TableauTransposedRaii<kStimWidth> trans_cum(ctx.v_cum);
                        if (result.pivot != top) {
                            emit_swap(ctx, trans_cum, result.pivot, top);
                            result.pivot = top;
                        }

                        if (result.basis == CompressedBasis::Z_BASIS) {
                            ctx.bytecode.push_back(make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL,
                                                             result.pivot, classical_idx,
                                                             result.sign));
                        } else {
                            ctx.bytecode.push_back(make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE,
                                                             result.pivot, classical_idx,
                                                             result.sign));
                        }

                        if (result.basis == CompressedBasis::X_BASIS) {
                            trans_cum.append_H_XZ(result.pivot);
                        }
                    }

                    ctx.reg_manager.deactivate();
                }

                last_meas_idx = classical_idx;
                break;
            }

            case OpType::CONDITIONAL_PAULI: {
                // Map the t=0 Pauli to virtual frame
                auto p_v = map_to_virtual(ctx, op.destab_mask(), op.stab_mask(), op.sign(), n);

                // Store in constant pool
                uint32_t cp_idx = static_cast<uint32_t>(ctx.constant_pool.pauli_masks.size());
                ctx.constant_pool.pauli_masks.push_back(std::move(p_v));

                // Resolve the controlling measurement index
                uint32_t cond_idx;
                if (op.use_last_outcome()) {
                    cond_idx = last_meas_idx;
                } else {
                    cond_idx = static_cast<uint32_t>(op.controlling_meas());
                }

                ctx.bytecode.push_back(make_apply_pauli(cp_idx, cond_idx));
                break;
            }

            case OpType::NOISE: {
                // Map each noise channel's Pauli through V_cum
                auto site_idx = static_cast<uint32_t>(op.noise_site_idx());
                assert(site_idx < hir.noise_sites.size());
                const auto& hir_site = hir.noise_sites[site_idx];

                NoiseSite mapped_site;
                for (const auto& ch : hir_site.channels) {
                    stim::PauliString<kStimWidth> p(n);
                    p.xs.u64[0] = ch.destab_mask;
                    p.zs.u64[0] = ch.stab_mask;
                    stim::PauliString<kStimWidth> mapped = ctx.v_cum(p);
                    mapped_site.channels.push_back({mapped.xs.u64[0], mapped.zs.u64[0], ch.prob});
                }

                // Compute total channel probability for gap sampling
                double prob_sum = 0.0;
                for (const auto& ch : mapped_site.channels) {
                    prob_sum += ch.prob;
                }

                uint32_t cp_idx = static_cast<uint32_t>(ctx.constant_pool.noise_sites.size());
                ctx.constant_pool.noise_sites.push_back(std::move(mapped_site));
                ctx.bytecode.push_back(make_noise(cp_idx));

                // Accumulate hazard for exponential gap sampling
                ctx.noise_hazards_accum += -std::log(1.0 - std::min(prob_sum, 1.0 - 1e-15));
                ctx.constant_pool.noise_hazards.push_back(ctx.noise_hazards_accum);
                break;
            }

            case OpType::READOUT_NOISE: {
                auto entry_idx = static_cast<uint32_t>(op.readout_noise_idx());
                assert(entry_idx < hir.readout_noise.size());
                const auto& entry = hir.readout_noise[entry_idx];

                uint32_t cp_idx = static_cast<uint32_t>(ctx.constant_pool.readout_noise.size());
                ctx.constant_pool.readout_noise.push_back(entry);
                ctx.bytecode.push_back(make_readout_noise(cp_idx));
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
                if (is_postselected) {
                    ctx.bytecode.push_back(make_postselect(cp_idx, det_emit_idx));
                } else {
                    ctx.bytecode.push_back(make_detector(cp_idx, det_emit_idx));
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
                ctx.bytecode.push_back(make_observable(cp_idx, obs_idx));
                break;
            }

            case OpType::CLIFFORD_PHASE: {
                auto p_v = map_to_virtual(ctx, op.destab_mask(), op.stab_mask(), op.sign(), n);
                auto result = compress_pauli(ctx, p_v);

                bool is_dormant = result.pivot >= ctx.reg_manager.active_k();

                if (is_dormant && result.basis == CompressedBasis::Z_BASIS) {
                    // Dormant Z: frame-only S/S_dag, no expansion needed.
                } else if (is_dormant) {
                    // Dormant X: S|+> = |+i>, which breaks the |0>_D invariant.
                    // Must expand into the active array.
                    uint16_t next_axis = static_cast<uint16_t>(ctx.reg_manager.active_k());

                    {
                        stim::TableauTransposedRaii<kStimWidth> trans_cum(ctx.v_cum);
                        if (result.pivot != next_axis) {
                            emit_swap(ctx, trans_cum, result.pivot, next_axis);
                            result.pivot = next_axis;
                        }

                        trans_cum.append_H_XZ(result.pivot);
                    }

                    ctx.bytecode.push_back(make_frame_h(result.pivot));

                    ctx.bytecode.push_back(make_expand(result.pivot));
                    ctx.reg_manager.activate();
                } else if (result.basis == CompressedBasis::X_BASIS) {
                    // Active X: emit H to rotate to Z basis.
                    {
                        stim::TableauTransposedRaii<kStimWidth> trans(ctx.v_cum);
                        trans.append_H_XZ(result.pivot);
                    }
                    ctx.bytecode.push_back(make_array_h(result.pivot));
                }

                // Re-evaluate active status: the array may have just expanded.
                bool is_active_now = result.pivot < ctx.reg_manager.active_k();

                bool phase_flip = result.sign ^ op.is_dagger();

                if (!is_active_now) {
                    if (phase_flip) {
                        ctx.bytecode.push_back(make_frame_s_dag(result.pivot));
                    } else {
                        ctx.bytecode.push_back(make_frame_s(result.pivot));
                    }
                } else {
                    if (phase_flip) {
                        ctx.bytecode.push_back(make_array_s_dag(result.pivot));
                    } else {
                        ctx.bytecode.push_back(make_array_s(result.pivot));
                    }
                }

                // Do NOT update V_cum here. The runtime opcode (OP_ARRAY_S /
                // OP_FRAME_S) applies the gate dynamically. Absorbing it into
                // V_cum would double-apply it.

                break;
            }
        }

        // Batch source-map and k-history: tag all instructions emitted by this HIR op
        size_t bc_after = ctx.bytecode.size();
        size_t emitted = bc_after - bc_before;
        uint32_t k_now = ctx.reg_manager.active_k();
        for (size_t e = 0; e < emitted; ++e) {
            if (has_source_map) {
                ctx.source_map.push_back(hir.source_map[op_idx]);
            }
            ctx.active_k_history.push_back(k_now);
        }
    }

    // Compute final tableau U_C = U_phys * V_cum^{-1}.
    // A.then(B) evaluates to B * A in matrix multiplication.
    // We want U_phys * V_cum^{-1}, so: v_cum_inv.then(U_phys) = U_phys * V_cum^{-1}.
    if (hir.final_tableau.has_value()) {
        stim::Tableau<kStimWidth> v_cum_inv = ctx.v_cum.inverse();
        ctx.constant_pool.final_tableau = v_cum_inv.then(*hir.final_tableau);
    }

    ctx.constant_pool.global_weight = hir.global_weight;

    uint16_t peak = ctx.reg_manager.peak_k();
    assert(peak < 64 && "peak_rank >= 64 would cause undefined behavior in 1ULL << k shifts");

    CompiledModule result;
    result.bytecode = std::move(ctx.bytecode);
    result.constant_pool = std::move(ctx.constant_pool);
    result.source_map = std::move(ctx.source_map);
    result.active_k_history = std::move(ctx.active_k_history);
    result.num_qubits = hir.num_qubits;
    result.peak_rank = peak;
    result.num_measurements = hir.num_measurements;
    result.total_meas_slots = total_meas_slots;
    result.num_detectors = hir.num_detectors;
    result.num_observables = hir.num_observables;
    return result;
}

}  // namespace ucc
