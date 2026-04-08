#include "clifft/optimizer/tile_axis_fusion_pass.h"

#include "clifft/util/constants.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <utility>

namespace clifft {
namespace {

using Mat4 = std::array<std::complex<double>, 16>;  // row-major 4x4

constexpr Mat4 kIdentity4 = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

// Whether this opcode is a single-axis array op that can participate in a tile.
bool is_1q_array_op(Opcode op) {
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

// Whether this opcode is a 2-qubit array op that can participate in a tile.
bool is_2q_array_op(Opcode op) {
    switch (op) {
        case Opcode::OP_ARRAY_CNOT:
        case Opcode::OP_ARRAY_CZ:
        case Opcode::OP_ARRAY_SWAP:
            return true;
        default:
            return false;
    }
}

// Whether this opcode is a single-axis frame-only op.
bool is_1q_frame_op(Opcode op) {
    switch (op) {
        case Opcode::OP_FRAME_H:
        case Opcode::OP_FRAME_S:
        case Opcode::OP_FRAME_S_DAG:
            return true;
        default:
            return false;
    }
}

// Whether this opcode is a 2-qubit frame-only op.
bool is_2q_frame_op(Opcode op) {
    switch (op) {
        case Opcode::OP_FRAME_CNOT:
        case Opcode::OP_FRAME_CZ:
        case Opcode::OP_FRAME_SWAP:
            return true;
        default:
            return false;
    }
}

// Returns the set of axes an instruction touches (max 2).
// Returns {axis, axis} for single-axis ops, {lo, hi} for two-axis ops.
// Returns {UINT16_MAX, UINT16_MAX} for ops that don't participate in tiles.
std::pair<uint16_t, uint16_t> get_axes(const Instruction& inst) {
    if (is_1q_array_op(inst.opcode) || is_1q_frame_op(inst.opcode)) {
        return {inst.axis_1, inst.axis_1};
    }
    if (is_2q_array_op(inst.opcode) || is_2q_frame_op(inst.opcode)) {
        uint16_t lo = std::min(inst.axis_1, inst.axis_2);
        uint16_t hi = std::max(inst.axis_1, inst.axis_2);
        return {lo, hi};
    }
    return {UINT16_MAX, UINT16_MAX};
}

// Whether an instruction can be part of a tile on axes {a, b}.
// Returns true if the instruction operates exclusively on {a, b}.
bool fits_tile(const Instruction& inst, uint16_t a, uint16_t b) {
    auto [lo, hi] = get_axes(inst);
    if (lo == UINT16_MAX)
        return false;
    // Single-axis op: must target a or b
    if (lo == hi)
        return (lo == a || lo == b);
    // Two-axis op: must target exactly {a, b}
    return (lo == a && hi == b);
}

// M = U * M  (left-multiply, 4x4)
inline void mat4_mul_left(Mat4& M, const Mat4& U) {
    Mat4 out;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            std::complex<double> sum{0.0, 0.0};
            for (int k = 0; k < 4; ++k) {
                sum += U[r * 4 + k] * M[k * 4 + c];
            }
            out[r * 4 + c] = sum;
        }
    }
    M = out;
}

// Build a 4x4 matrix from a 1Q gate on axis_lo (LSB) of the tile.
// Kronecker product: I_2 (x) U_2  (U acts on the low axis)
Mat4 kron_lo(const std::array<std::complex<double>, 4>& u) {
    Mat4 m = {};
    // |b_hi, b_lo>: index = (b_hi << 1) | b_lo
    // U acts on b_lo, I on b_hi
    for (int bh = 0; bh < 2; ++bh) {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                m[(bh * 2 + r) * 4 + (bh * 2 + c)] = u[r * 2 + c];
            }
        }
    }
    return m;
}

// Build a 4x4 matrix from a 1Q gate on axis_hi (MSB) of the tile.
// Kronecker product: U_2 (x) I_2  (U acts on the high axis)
Mat4 kron_hi(const std::array<std::complex<double>, 4>& u) {
    Mat4 m = {};
    // U acts on b_hi, I on b_lo
    for (int bl = 0; bl < 2; ++bl) {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                m[(r * 2 + bl) * 4 + (c * 2 + bl)] = u[r * 2 + c];
            }
        }
    }
    return m;
}

// Build a 4x4 Kronecker product from gate on specified tile axis.
Mat4 kron_axis(const std::array<std::complex<double>, 4>& u, bool on_hi) {
    return on_hi ? kron_hi(u) : kron_lo(u);
}

using Mat2 = std::array<std::complex<double>, 4>;

// Standard 2x2 gate matrices
constexpr Mat2 kHadamard2 = {kInvSqrt2, kInvSqrt2, kInvSqrt2, -kInvSqrt2};
constexpr Mat2 kS2 = {{1.0, 0.0, 0.0, kI}};
constexpr Mat2 kSdag2 = {{1.0, 0.0, 0.0, kMinusI}};

// CNOT matrix in the 4x4 basis {|00>, |01>, |10>, |11>}
// where control is hi and target is lo:
//   |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
Mat4 make_cnot_4x4(uint16_t ctrl, uint16_t tgt, uint16_t tile_lo, uint16_t tile_hi) {
    // Determine which tile axis is control and which is target
    bool ctrl_is_hi = (ctrl == tile_hi);
    (void)tgt;
    (void)tile_lo;

    // Matrix convention: mat[row][col] = <row|U|col>, applied as new = mat * old.
    // CNOT is self-inverse so the matrix is symmetric (row/col are interchangeable).
    Mat4 m = {};
    if (ctrl_is_hi) {
        // Control = hi (MSB), Target = lo (LSB)
        // U|00>=|00>, U|01>=|01>, U|10>=|11>, U|11>=|10>
        m[0 * 4 + 0] = 1;  // <00|U|00> = 1
        m[1 * 4 + 1] = 1;  // <01|U|01> = 1
        m[2 * 4 + 3] = 1;  // <10|U|11> = 1
        m[3 * 4 + 2] = 1;  // <11|U|10> = 1
    } else {
        // Control = lo (LSB), Target = hi (MSB)
        // U|00>=|00>, U|01>=|11>, U|10>=|10>, U|11>=|01>
        m[0 * 4 + 0] = 1;  // <00|U|00> = 1
        m[1 * 4 + 3] = 1;  // <01|U|11> = 1
        m[2 * 4 + 2] = 1;  // <10|U|10> = 1
        m[3 * 4 + 1] = 1;  // <11|U|01> = 1
    }
    return m;
}

// CZ matrix: diag(1, 1, 1, -1) regardless of axis ordering.
Mat4 make_cz_4x4() {
    Mat4 m = {};
    m[0 * 4 + 0] = 1;
    m[1 * 4 + 1] = 1;
    m[2 * 4 + 2] = 1;
    m[3 * 4 + 3] = -1;
    return m;
}

// SWAP matrix: |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>
Mat4 make_swap_4x4() {
    Mat4 m = {};
    m[0 * 4 + 0] = 1;
    m[1 * 4 + 2] = 1;
    m[2 * 4 + 1] = 1;
    m[3 * 4 + 3] = 1;
    return m;
}

// Simulate one instruction on a 2-axis tile, accumulating into M and gamma.
// px_lo, pz_lo, px_hi, pz_hi are the current Pauli frame bits (modified in place).
// tile_lo and tile_hi are the tile's fixed axis pair.
void simulate_tile_instruction(const Instruction& inst, uint8_t& px_lo, uint8_t& pz_lo,
                               uint8_t& px_hi, uint8_t& pz_hi, Mat4& M, std::complex<double>& gamma,
                               uint16_t tile_lo, uint16_t tile_hi) {
    // Determine which tile axis this instruction operates on
    bool on_hi_axis = false;
    if (is_1q_array_op(inst.opcode) || is_1q_frame_op(inst.opcode)) {
        on_hi_axis = (inst.axis_1 == tile_hi);
    }

    // Get the relevant frame bits for this operation
    uint8_t& px = on_hi_axis ? px_hi : px_lo;
    uint8_t& pz = on_hi_axis ? pz_hi : pz_lo;

    switch (inst.opcode) {
        // 1Q array ops: build 2x2, Kronecker into 4x4, left-multiply M
        case Opcode::OP_ARRAY_H: {
            Mat4 U = kron_axis(kHadamard2, on_hi_axis);
            if (px && pz)
                gamma *= std::complex<double>(-1.0, 0.0);
            std::swap(px, pz);
            mat4_mul_left(M, U);
            break;
        }
        case Opcode::OP_ARRAY_S: {
            Mat4 U = kron_axis(kS2, on_hi_axis);
            if (px)
                gamma *= kI;
            pz ^= px;
            mat4_mul_left(M, U);
            break;
        }
        case Opcode::OP_ARRAY_S_DAG: {
            Mat4 U = kron_axis(kSdag2, on_hi_axis);
            if (px)
                gamma *= kMinusI;
            pz ^= px;
            mat4_mul_left(M, U);
            break;
        }
        case Opcode::OP_PHASE_T: {
            Mat2 u = {1.0, 0.0, 0.0, 1.0};
            if (px) {
                u[3] = kExpMinusIPiOver4;
                gamma *= kExpIPiOver4;
            } else {
                u[3] = kExpIPiOver4;
            }
            mat4_mul_left(M, kron_axis(u, on_hi_axis));
            break;
        }
        case Opcode::OP_PHASE_T_DAG: {
            Mat2 u = {1.0, 0.0, 0.0, 1.0};
            if (px) {
                u[3] = kExpIPiOver4;
                gamma *= kExpMinusIPiOver4;
            } else {
                u[3] = kExpMinusIPiOver4;
            }
            mat4_mul_left(M, kron_axis(u, on_hi_axis));
            break;
        }
        case Opcode::OP_PHASE_ROT: {
            std::complex<double> z(inst.math.weight_re, inst.math.weight_im);
            Mat2 u = {1.0, 0.0, 0.0, 1.0};
            if (px) {
                u[3] = std::conj(z);
                gamma *= z;
            } else {
                u[3] = z;
            }
            mat4_mul_left(M, kron_axis(u, on_hi_axis));
            break;
        }

        // 1Q frame-only ops: update frame and gamma, no matrix
        case Opcode::OP_FRAME_H:
            if (px && pz)
                gamma *= std::complex<double>(-1.0, 0.0);
            std::swap(px, pz);
            break;
        case Opcode::OP_FRAME_S:
            if (px)
                gamma *= kI;
            pz ^= px;
            break;
        case Opcode::OP_FRAME_S_DAG:
            if (px)
                gamma *= kMinusI;
            pz ^= px;
            break;

        // 2Q array ops: build 4x4 directly
        case Opcode::OP_ARRAY_CNOT: {
            Mat4 U = make_cnot_4x4(inst.axis_1, inst.axis_2, tile_lo, tile_hi);
            // Frame update: CNOT(ctrl, tgt)
            // p_x[tgt] ^= p_x[ctrl], p_z[ctrl] ^= p_z[tgt]
            bool ctrl_is_hi = (inst.axis_1 == tile_hi);
            if (ctrl_is_hi) {
                px_lo ^= px_hi;
                pz_hi ^= pz_lo;
            } else {
                px_hi ^= px_lo;
                pz_lo ^= pz_hi;
            }
            mat4_mul_left(M, U);
            break;
        }
        case Opcode::OP_ARRAY_CZ: {
            Mat4 U = make_cz_4x4();
            // Frame update: CZ(c, t)
            if (px_lo && px_hi)
                gamma *= std::complex<double>(-1.0, 0.0);
            pz_hi ^= px_lo;
            pz_lo ^= px_hi;
            mat4_mul_left(M, U);
            break;
        }
        case Opcode::OP_ARRAY_SWAP: {
            Mat4 U = make_swap_4x4();
            std::swap(px_lo, px_hi);
            std::swap(pz_lo, pz_hi);
            mat4_mul_left(M, U);
            break;
        }

        // 2Q frame-only ops
        case Opcode::OP_FRAME_CNOT: {
            bool ctrl_is_hi = (inst.axis_1 == tile_hi);
            if (ctrl_is_hi) {
                px_lo ^= px_hi;
                pz_hi ^= pz_lo;
            } else {
                px_hi ^= px_lo;
                pz_lo ^= pz_hi;
            }
            break;
        }
        case Opcode::OP_FRAME_CZ:
            if (px_lo && px_hi)
                gamma *= std::complex<double>(-1.0, 0.0);
            pz_hi ^= px_lo;
            pz_lo ^= px_hi;
            break;
        case Opcode::OP_FRAME_SWAP:
            std::swap(px_lo, px_hi);
            std::swap(pz_lo, pz_hi);
            break;

        default:
            assert(false && "Unexpected opcode in tile simulation");
            break;
    }
}

// Build a FusedU4Node by simulating the instruction sequence for all 16 frame states.
FusedU4Node build_fused_u4_node(const std::vector<Instruction>& bytecode, size_t start, size_t end,
                                uint16_t tile_lo, uint16_t tile_hi) {
    FusedU4Node node{};

    for (uint8_t in_state = 0; in_state < 16; ++in_state) {
        Mat4 M = kIdentity4;
        std::complex<double> gamma = {1.0, 0.0};
        uint8_t px_lo = in_state & 1;
        uint8_t pz_lo = (in_state >> 1) & 1;
        uint8_t px_hi = (in_state >> 2) & 1;
        uint8_t pz_hi = (in_state >> 3) & 1;

        for (size_t i = start; i < end; ++i) {
            simulate_tile_instruction(bytecode[i], px_lo, pz_lo, px_hi, pz_hi, M, gamma, tile_lo,
                                      tile_hi);
        }

        auto& entry = node.entries[in_state];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                entry.matrix[r][c] = M[r * 4 + c];
        entry.gamma_multiplier = gamma;
        entry.out_state = static_cast<uint8_t>((pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo);
    }

    return node;
}

}  // namespace

void TileAxisFusionPass::run(CompiledModule& module) {
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
        const auto& inst = old_bc[i];
        auto [lo, hi] = get_axes(inst);

        // Only start a tile from a 2Q array op (ensures we have a tile pair)
        if (!is_2q_array_op(inst.opcode)) {
            new_bc.push_back(inst);
            if (has_sm)
                new_sm.copy_entry(old_sm, i);
            ++i;
            continue;
        }

        // Found a 2Q array op on axes {lo, hi}. Scan forward for the tile run.
        uint16_t tile_lo = lo;
        uint16_t tile_hi = hi;
        size_t run_start = i;
        size_t run_end = i + 1;
        int array_op_count = 1;  // The 2Q op we just found

        while (run_end < old_bc.size()) {
            const auto& next = old_bc[run_end];

            if (fits_tile(next, tile_lo, tile_hi)) {
                if (is_1q_array_op(next.opcode) || is_2q_array_op(next.opcode))
                    ++array_op_count;
                ++run_end;
                continue;
            }

            // Does not fit the tile. Stop scanning.
            break;
        }

        // Only fuse if we save at least 2 array sweeps (min 3 array ops)
        if (array_op_count >= 3) {
            FusedU4Node node = build_fused_u4_node(old_bc, run_start, run_end, tile_lo, tile_hi);
            uint32_t cp_idx = static_cast<uint32_t>(module.constant_pool.fused_u4_nodes.size());
            module.constant_pool.fused_u4_nodes.push_back(node);
            new_bc.push_back(make_array_u4(tile_lo, tile_hi, cp_idx));
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

}  // namespace clifft
