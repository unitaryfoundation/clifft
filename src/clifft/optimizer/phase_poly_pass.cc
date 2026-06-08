#include "clifft/optimizer/phase_poly_pass.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/mcr_tcount.h"
#include "clifft/optimizer/t_fusion.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace clifft {

namespace {

// Maximum number of T gates per commuting block that this pass will attempt
// to optimize. Blocks larger than this are skipped to bound compile time.
// m <= 48 keeps each row of the augmented matrix within a single uint64_t
// (48 data bits + 1 RHS bit = 49 bits).
static constexpr int kMaxBlockSize = 48;

// =============================================================================
// GF(2) linear algebra
// =============================================================================

// Row-reduce the system over GF(2) to reduced row echelon form (RREF).
// Each row is a uint64_t encoding m data bits (columns 0..m-1) and one RHS
// bit at position m. Returns false when the system is inconsistent (a row
// with zero data and RHS=1 survives elimination).
static bool gf2_rref(std::vector<uint64_t>& rows, int m) {
    int cur = 0;
    for (int col = 0; col < m && cur < static_cast<int>(rows.size()); ++col) {
        int pivot = -1;
        for (int r = cur; r < static_cast<int>(rows.size()); ++r) {
            if ((rows[r] >> col) & 1ULL) {
                pivot = r;
                break;
            }
        }
        if (pivot == -1)
            continue;

        std::swap(rows[cur], rows[pivot]);

        // Full RREF: eliminate the pivot column from every other row.
        for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
            if (r != cur && ((rows[r] >> col) & 1ULL))
                rows[r] ^= rows[cur];
        }
        ++cur;
    }

    uint64_t data_mask = (m < 63) ? ((1ULL << m) - 1ULL) : ~0ULL;
    for (auto& row : rows) {
        if ((row & data_mask) == 0 && ((row >> m) & 1ULL))
            return false;
    }
    return true;
}

// Check whether A_aug * y = 0 has a solution with y[a] XOR y[b] = 1.
// When a solution exists, write it into *sol and return true.
//
// The pair constraint is represented as an extra row (e_a XOR e_b) with
// RHS = 1 appended to A_aug. Solving the augmented system by RREF then
// reading off the pivot values (free variables set to 0) yields y.
static bool tohpe_try_pair(const std::vector<uint64_t>& a_aug_base, int m, int a, int b,
                            uint64_t& sol) {
    auto rows = a_aug_base;
    // Pair row: (e_a XOR e_b) * y = 1
    rows.push_back(((1ULL << a) ^ (1ULL << b)) | (1ULL << m));

    if (!gf2_rref(rows, m))
        return false;

    // In RREF each non-zero data row has exactly one set data bit (the pivot).
    // With all free variables set to 0, pivot variables equal their RHS bits.
    uint64_t data_mask = (m < 63) ? ((1ULL << m) - 1ULL) : ~0ULL;
    sol = 0;
    for (auto& row : rows) {
        uint64_t data = row & data_mask;
        if (data == 0)
            continue;
        int pivot_col = std::countr_zero(data);
        if ((row >> m) & 1ULL)
            sol |= (1ULL << pivot_col);
    }
    return true;
}

// =============================================================================
// Parity table helpers
// =============================================================================

// Encode a T gate's Pauli as a single uint64_t column suitable for the
// parity table. Bits [0, nq-1] hold the X (destabilizer) mask and bits
// [nq, 2*nq-1] hold the Z (stabilizer) mask. Caller guarantees nq <= 32.
static uint64_t encode_pauli(const HirModule& hir, const HeisenbergOp& op, uint32_t nq) {
    MaskView xv = hir.destab_mask(op);
    MaskView zv = hir.stab_mask(op);
    uint64_t x = (xv.num_words() > 0) ? xv.words[0] : 0ULL;
    uint64_t z = (zv.num_words() > 0) ? zv.words[0] : 0ULL;
    uint64_t mask = (nq < 64) ? ((1ULL << nq) - 1ULL) : ~0ULL;
    return (x & mask) | ((z & mask) << nq);
}

// Build the TOHPE augmented constraint matrix A_aug for m columns over the
// n_bits = 2*nq-dimensional symplectic row index space.
//
// Rows encode three TOHPE conditions (Vandaele 2024, Theorem 1):
//   C1: Hamming weight of y is even   -> one all-ones row
//   C2: Parity table P times y = 0   -> n_bits rows (one per Pauli bit)
//   C3': pairwise AND of P rows       -> C(n_bits, 2) rows
//
// Only non-zero rows are kept; this prunes the matrix significantly for
// circuits where many Pauli bits are unused.
static std::vector<uint64_t> build_a_aug(const std::vector<uint64_t>& cols, int m, int n_bits) {
    std::vector<uint64_t> a;

    // C1
    uint64_t ones = (m < 64) ? ((1ULL << m) - 1ULL) : ~0ULL;
    if (ones != 0)
        a.push_back(ones);

    // C2: one row per Pauli bit position
    for (int alpha = 0; alpha < n_bits; ++alpha) {
        uint64_t row = 0;
        for (int j = 0; j < m; ++j) {
            if ((cols[j] >> alpha) & 1ULL)
                row |= (1ULL << j);
        }
        if (row != 0)
            a.push_back(row);
    }

    // C3': AND of every row pair
    for (int alpha = 0; alpha < n_bits; ++alpha) {
        for (int beta = alpha + 1; beta < n_bits; ++beta) {
            uint64_t row = 0;
            for (int j = 0; j < m; ++j) {
                if (((cols[j] >> alpha) & 1ULL) && ((cols[j] >> beta) & 1ULL))
                    row |= (1ULL << j);
            }
            if (row != 0)
                a.push_back(row);
        }
    }

    return a;
}

// =============================================================================
// Block identification
// =============================================================================

struct CommutingBlock {
    std::vector<size_t> t_pos;  // indices of T_GATE ops in hir.ops
};

// Greedy left-to-right scan: extend the current block with each T gate that
// commutes with all members already in the block. Anti-commutation forces a
// block boundary and starts a new block.
static std::vector<CommutingBlock> find_commuting_blocks(const HirModule& hir,
                                                         const std::vector<uint8_t>& deleted) {
    std::vector<CommutingBlock> blocks;
    std::vector<size_t> cur;

    auto flush = [&]() {
        if (!cur.empty()) {
            blocks.push_back({std::move(cur)});
            cur.clear();
        }
    };

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (deleted[i])
            continue;
        const auto& op = hir.ops[i];
        if (op.op_type() != OpType::T_GATE)
            continue;

        bool fits = true;
        for (size_t j : cur) {
            if (anti_commute(hir.destab_mask(op), hir.stab_mask(op),
                             hir.destab_mask(hir.ops[j]), hir.stab_mask(hir.ops[j]))) {
                fits = false;
                break;
            }
        }

        if (fits) {
            cur.push_back(i);
        } else {
            flush();
            cur.push_back(i);
        }
    }
    flush();
    return blocks;
}

}  // namespace

// =============================================================================
// PhasePolynomialPass::run
// =============================================================================

void PhasePolynomialPass::run(HirModule& hir) {
    t_reductions_ = 0;
    blocks_optimized_ = 0;
    mcr_stats_ = McrTcountStats{};
    t_gates_before_ = hir.num_t_gates();

    run_mcr_tcount(hir, mcr_stats_);

    // Single 64-bit column requires 2*nq <= 64.
    if (hir.num_qubits > 32) {
        t_gates_after_ = hir.num_t_gates();
        return;
    }

    const int n_bits = static_cast<int>(2 * hir.num_qubits);
    const uint32_t nq = hir.num_qubits;
    const uint64_t nq_mask = (nq < 64) ? ((1ULL << nq) - 1ULL) : ~0ULL;

    bool changed = true;
    while (changed) {
        changed = false;
        size_t n = hir.ops.size();
        std::vector<uint8_t> deleted(n, 0);

        auto blocks = find_commuting_blocks(hir, deleted);

        for (auto& blk : blocks) {
            int m = static_cast<int>(blk.t_pos.size());
            if (m < 2 || m > kMaxBlockSize)
                continue;

            for (size_t pos : blk.t_pos)
                normalize_t_sign(hir, hir.ops[pos]);

            // Extract initial column vectors.
            std::vector<uint64_t> cols(m);
            for (int j = 0; j < m; ++j)
                cols[j] = encode_pauli(hir, hir.ops[blk.t_pos[j]], nq);

            // active[j] is false once the j-th column has been destroyed.
            std::vector<bool> active(m, true);
            std::vector<std::pair<int, int>> destroyed;

            // Iterate TOHPE reductions to fixpoint.
            bool found_pair = true;
            while (found_pair) {
                found_pair = false;

                std::vector<int> ai;
                ai.reserve(m);
                for (int j = 0; j < m; ++j) {
                    if (active[j])
                        ai.push_back(j);
                }
                int ma = static_cast<int>(ai.size());
                if (ma < 2)
                    break;

                std::vector<uint64_t> acols(ma);
                for (int k = 0; k < ma; ++k)
                    acols[k] = cols[ai[k]];

                auto a_aug = build_a_aug(acols, ma, n_bits);

                for (int ka = 0; ka < ma && !found_pair; ++ka) {
                    for (int kb = ka + 1; kb < ma && !found_pair; ++kb) {
                        uint64_t y_active = 0;
                        if (!tohpe_try_pair(a_aug, ma, ka, kb, y_active))
                            continue;

                        // Apply the transformation: for each active column k
                        // with y_active[k]=1, XOR it with z = acol[ka] ^ acol[kb].
                        uint64_t z_col = acols[ka] ^ acols[kb];
                        for (int k = 0; k < ma; ++k) {
                            if ((y_active >> k) & 1ULL)
                                acols[k] ^= z_col;
                        }

                        // Write transformed columns back.
                        for (int k = 0; k < ma; ++k)
                            cols[ai[k]] = acols[k];

                        // Destroy the pair (they are now equal).
                        active[ai[ka]] = false;
                        active[ai[kb]] = false;
                        destroyed.push_back({ai[ka], ai[kb]});
                        found_pair = true;
                    }
                }
            }

            if (destroyed.empty())
                continue;
            changed = true;
            ++blocks_optimized_;

            // Apply S-gate residuals for each destroyed pair and mark deletions.
            for (auto [pa, pb] : destroyed) {
                size_t pos_a = blk.t_pos[pa];
                size_t pos_b = blk.t_pos[pb];

                bool dag_a = hir.ops[pos_a].is_dagger();
                bool dag_b = hir.ops[pos_b].is_dagger();

                deleted[pos_a] = true;
                deleted[pos_b] = true;
                t_reductions_ += 2;

                // T+T_dag or T_dag+T cancel to identity: no Clifford residual.
                int dir = (dag_a ? -1 : 1) + (dag_b ? -1 : 1);
                if (dir == 0)
                    continue;

                bool s_is_dagger = (dir == -2);

                // cols[pa] == cols[pb] after transformation; use either.
                uint64_t pauli_bits = cols[pa];
                uint64_t xw = pauli_bits & nq_mask;
                uint64_t zw = (pauli_bits >> nq) & nq_mask;

                // MaskViews reference local storage; lifetime covers the call.
                MaskView xv{std::span<const uint64_t>(&xw, 1)};
                MaskView zv{std::span<const uint64_t>(&zw, 1)};

                size_t start = std::max(pos_a, pos_b) + 1;
                apply_virtual_s_downstream(hir, start, xv, zv, false, s_is_dagger, deleted);
            }

            // Update surviving T gate Pauli masks in-place for those whose
            // columns were modified by the TOHPE transformation.
            for (int j = 0; j < m; ++j) {
                if (!active[j])
                    continue;
                uint64_t orig = encode_pauli(hir, hir.ops[blk.t_pos[j]], nq);
                if (cols[j] == orig)
                    continue;
                auto mut = hir.mask_at(hir.ops[blk.t_pos[j]]);
                mut.x().words[0] = cols[j] & nq_mask;
                mut.z().words[0] = (cols[j] >> nq) & nq_mask;
                // Sign bit: TOHPE modifies the Pauli axis but the rotation
                // direction (dagger flag) and global phase remain unchanged.
                // Reset sign to false to match the normalized convention used
                // throughout the optimizer.
                mut.set_sign(false);
            }
        }

        if (changed)
            compact_deleted_ops(hir, deleted);
    }

    t_gates_after_ = hir.num_t_gates();
}

}  // namespace clifft
