#include "ucc/backend/backend.h"

#include "ucc/backend/compiler_context.h"

#include <cassert>

namespace ucc {

using internal::CompilerContext;
using internal::CompressedBasis;
using internal::CompressionResult;

namespace {

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

Instruction make_frame_s(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_S;
    i.axis_1 = v;
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
    ctx.bytecode.push_back(make_frame_s(v));
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

    // Local tableau tracks ONLY this compression's gates.
    // Used at the end to derive the sign without interference from
    // previously accumulated gates in ctx.v_cum.
    stim::Tableau<kStimWidth> v_local(n);

    uint16_t pivot;
    CompressedBasis basis;

    // Hoist the O(n^2) transpose into a single scope. All O(n) gate
    // emissions use the pre-transposed wrappers, keeping the total
    // complexity at O(n^2) rather than O(n^3).
    {
        stim::TableauTransposedRaii<kStimWidth> trans_cum(ctx.v_cum);
        stim::TableauTransposedRaii<kStimWidth> trans_local(v_local);

        uint32_t k = ctx.reg_manager.active_k();
        uint64_t active_mask = (k == 0) ? 0ULL : ((1ULL << k) - 1);

        if (x_bits != 0) {
            // =============================================================
            // Case 1: X-support is non-empty.
            // =============================================================

            // Pick pivot from X-support, preferring dormant axes (>= k).
            uint64_t x_dormant = x_bits & ~active_mask;
            if (x_dormant != 0) {
                pivot = static_cast<uint16_t>(__builtin_ctzll(x_dormant));
            } else {
                pivot = static_cast<uint16_t>(__builtin_ctzll(x_bits));
            }

            // X-compression: CNOT(pivot -> q) annihilates X_q.
            // Z propagation: Z_t -> Z_c Z_t, so z_pivot ^= z_q.
            uint64_t to_clear_x = x_bits & ~(1ULL << pivot);
            while (to_clear_x != 0) {
                uint16_t q = static_cast<uint16_t>(__builtin_ctzll(to_clear_x));
                emit_cnot(ctx, trans_cum, trans_local, pivot, q);
                if (z_bits & (1ULL << q)) {
                    z_bits ^= (1ULL << pivot);
                }
                to_clear_x &= to_clear_x - 1;
            }

            // Z-compression: CZ(pivot, q) clears residual Z_q.
            uint64_t to_clear_z = z_bits & ~(1ULL << pivot);
            while (to_clear_z != 0) {
                uint16_t q = static_cast<uint16_t>(__builtin_ctzll(to_clear_z));
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
                pivot = static_cast<uint16_t>(__builtin_ctzll(z_active));
            } else {
                pivot = static_cast<uint16_t>(__builtin_ctzll(z_bits));
            }

            // Z-compression: CNOT(q -> pivot) folds Z_q onto pivot.
            uint64_t to_clear_z = z_bits & ~(1ULL << pivot);
            while (to_clear_z != 0) {
                uint16_t q = static_cast<uint16_t>(__builtin_ctzll(to_clear_z));
                emit_cnot(ctx, trans_cum, trans_local, q, pivot);
                to_clear_z &= to_clear_z - 1;
            }

            basis = CompressedBasis::Z_BASIS;
        }
    }  // RAII destructors un-transpose both tableaux here.

    // v_local is now in standard layout; evaluate to get the sign.
    stim::PauliString<kStimWidth> compressed = v_local(pauli);
    bool sign = compressed.sign;

    return {pivot, basis, sign};
}

}  // namespace internal

// =========================================================================
// lower(): Stub for Phase 4 pipeline wiring
// =========================================================================

CompiledModule lower(const HirModule& hir) {
    CompiledModule result;
    result.num_qubits = hir.num_qubits;
    result.num_measurements = hir.num_measurements;
    result.num_detectors = hir.num_detectors;
    result.num_observables = hir.num_observables;

    // TODO: Implement pipeline wiring (Phase 4)
    // - Iterate HIR ops
    // - Map t=0 Paulis to virtual frame via V_cum
    // - Run compress_pauli
    // - Emit OP_EXPAND, OP_PHASE_T, measurement opcodes
    // - Compute final_tableau (U_C = U_phys * V_cum^dag)

    return result;
}

}  // namespace ucc
