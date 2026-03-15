#include "ucc/optimizer/single_axis_fusion_pass.h"

#include "ucc/util/constants.h"

#include <array>
#include <complex>
#include <utility>

namespace ucc {
namespace {

using Mat2 = std::array<std::complex<double>, 4>;  // row-major 2x2

// Opcodes that can be fused into a single-axis U2 node.
// Includes both array-touching and frame-only single-axis ops.
bool is_fusible(Opcode op) {
    switch (op) {
        case Opcode::OP_ARRAY_H:
        case Opcode::OP_ARRAY_S:
        case Opcode::OP_ARRAY_S_DAG:
        case Opcode::OP_PHASE_T:
        case Opcode::OP_PHASE_T_DAG:
        case Opcode::OP_PHASE_ROT:
        case Opcode::OP_FRAME_H:
        case Opcode::OP_FRAME_S:
        case Opcode::OP_FRAME_S_DAG:
            return true;
        default:
            return false;
    }
}

// Does this opcode touch the amplitude array (vs frame-only)?
bool is_array_op(Opcode op) {
    switch (op) {
        case Opcode::OP_ARRAY_H:
        case Opcode::OP_ARRAY_S:
        case Opcode::OP_ARRAY_S_DAG:
        case Opcode::OP_PHASE_T:
        case Opcode::OP_PHASE_T_DAG:
        case Opcode::OP_PHASE_ROT:
            return true;
        default:
            return false;
    }
}

// Get the axis of a fusible instruction.
uint16_t fusible_axis(const Instruction& inst) {
    return inst.axis_1;
}

// M = U * M  (left-multiply)
inline void mat_mul_left(Mat2& M, const Mat2& U) {
    Mat2 out;
    out[0] = U[0] * M[0] + U[1] * M[2];
    out[1] = U[0] * M[1] + U[1] * M[3];
    out[2] = U[2] * M[0] + U[3] * M[2];
    out[3] = U[2] * M[1] + U[3] * M[3];
    M = out;
}

// Simulate one instruction on a given frame state, accumulating into M and gamma.
// px, pz are the current Pauli frame bits (modified in place).
void simulate_instruction(const Instruction& inst, uint8_t& px, uint8_t& pz, Mat2& M,
                          std::complex<double>& gamma) {
    Mat2 U = {1.0, 0.0, 0.0, 1.0};  // identity

    switch (inst.opcode) {
        case Opcode::OP_ARRAY_H:
            U = {kInvSqrt2, kInvSqrt2, kInvSqrt2, -kInvSqrt2};
            if (px && pz)
                gamma *= std::complex<double>(-1.0, 0.0);
            std::swap(px, pz);
            break;

        case Opcode::OP_ARRAY_S:
            U = {1.0, 0.0, 0.0, kI};
            if (px)
                gamma *= kI;
            pz ^= px;
            break;

        case Opcode::OP_ARRAY_S_DAG:
            U = {1.0, 0.0, 0.0, kMinusI};
            if (px)
                gamma *= kMinusI;
            pz ^= px;
            break;

        case Opcode::OP_PHASE_T:
            if (px) {
                U[3] = kExpMinusIPiOver4;
                gamma *= kExpIPiOver4;
            } else {
                U[3] = kExpIPiOver4;
            }
            break;

        case Opcode::OP_PHASE_T_DAG:
            if (px) {
                U[3] = kExpIPiOver4;
                gamma *= kExpMinusIPiOver4;
            } else {
                U[3] = kExpMinusIPiOver4;
            }
            break;

        case Opcode::OP_PHASE_ROT: {
            std::complex<double> z(inst.math.weight_re, inst.math.weight_im);
            if (px) {
                U[3] = std::conj(z);
                gamma *= z;
            } else {
                U[3] = z;
            }
            break;
        }

        // Frame-only ops: no array matrix, just update frame and gamma
        case Opcode::OP_FRAME_H:
            if (px && pz)
                gamma *= std::complex<double>(-1.0, 0.0);
            std::swap(px, pz);
            return;  // no matrix to multiply

        case Opcode::OP_FRAME_S:
            if (px)
                gamma *= kI;
            pz ^= px;
            return;

        case Opcode::OP_FRAME_S_DAG:
            if (px)
                gamma *= kMinusI;
            pz ^= px;
            return;

        default:
            return;
    }

    mat_mul_left(M, U);
}

// Build a FusedU2Node by simulating the instruction sequence for all 4 frame states.
FusedU2Node build_fused_node(const std::vector<Instruction>& bytecode, size_t start, size_t end) {
    FusedU2Node node{};

    for (uint8_t in_state = 0; in_state < 4; ++in_state) {
        Mat2 M = {1.0, 0.0, 0.0, 1.0};
        std::complex<double> gamma = {1.0, 0.0};
        uint8_t px = in_state & 1;
        uint8_t pz = (in_state >> 1) & 1;

        for (size_t i = start; i < end; ++i) {
            simulate_instruction(bytecode[i], px, pz, M, gamma);
        }

        for (int j = 0; j < 4; ++j)
            node.matrices[in_state][j] = M[j];
        node.gamma_multipliers[in_state] = gamma;
        node.out_states[in_state] = static_cast<uint8_t>((pz << 1) | px);
    }

    return node;
}

}  // namespace

void SingleAxisFusionPass::run(CompiledModule& module) {
    auto& old_bc = module.bytecode;
    auto& old_sm = module.source_map;
    bool has_sm = !old_sm.empty();

    std::vector<Instruction> new_bc;
    SourceMap new_sm;
    new_bc.reserve(old_bc.size());
    if (has_sm)
        new_sm.reserve(old_bc.size(), old_sm.data().size());

    size_t i = 0;
    while (i < old_bc.size()) {
        if (!is_fusible(old_bc[i].opcode)) {
            new_bc.push_back(old_bc[i]);
            if (has_sm)
                new_sm.copy_entry(old_sm, i);
            ++i;
            continue;
        }

        // Start of a potential fusible run on this axis
        uint16_t axis = fusible_axis(old_bc[i]);
        size_t run_start = i;
        size_t run_end = i + 1;
        int array_op_count = is_array_op(old_bc[i].opcode) ? 1 : 0;
        bool has_rot = (old_bc[i].opcode == Opcode::OP_PHASE_ROT);

        while (run_end < old_bc.size() && is_fusible(old_bc[run_end].opcode) &&
               fusible_axis(old_bc[run_end]) == axis) {
            if (is_array_op(old_bc[run_end].opcode))
                ++array_op_count;
            if (old_bc[run_end].opcode == Opcode::OP_PHASE_ROT)
                has_rot = true;
            ++run_end;
        }

        // Only fuse if we save at least two array passes, OR if the run contains
        // a continuous rotation. Isolated length-2 sequences of lightweight ops
        // (e.g. H+T) are cheaper unfused than a dense 2x2 matrix sweep.
        if (array_op_count >= 3 || (array_op_count >= 2 && has_rot)) {
            FusedU2Node node = build_fused_node(old_bc, run_start, run_end);
            uint32_t cp_idx = static_cast<uint32_t>(module.constant_pool.fused_u2_nodes.size());
            module.constant_pool.fused_u2_nodes.push_back(node);
            new_bc.push_back(make_array_u2(axis, cp_idx));
            if (has_sm)
                new_sm.merge_entries(old_sm, run_start, run_end);
            i = run_end;
        } else {
            // Emit the run unfused
            for (size_t j = run_start; j < run_end; ++j) {
                new_bc.push_back(old_bc[j]);
                if (has_sm)
                    new_sm.copy_entry(old_sm, j);
            }
            i = run_end;
        }
    }

    old_bc = std::move(new_bc);
    if (has_sm)
        old_sm = std::move(new_sm);
}

}  // namespace ucc
