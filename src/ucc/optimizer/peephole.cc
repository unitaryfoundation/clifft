#include "ucc/optimizer/peephole.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ucc {

namespace {

/// Symplectic inner product over bitword<64> masks.
/// Returns true if the two Pauli strings anti-commute.
inline bool anti_commute(stim::bitword<64> x1, stim::bitword<64> z1, stim::bitword<64> x2,
                         stim::bitword<64> z2) {
    return ((x1 & z2) ^ (z1 & x2)).popcount() % 2 != 0;
}

/// Symplectic inner product over raw uint64_t masks.
/// Used for NoiseChannel which stores raw integers.
inline bool anti_commute_raw(uint64_t x1, uint64_t z1, uint64_t x2, uint64_t z2) {
    return std::popcount((x1 & z2) ^ (z1 & x2)) % 2 != 0;
}

/// Compute the effective rotation direction for a T gate.
/// T -> +1, T_dag -> -1, negated if the Pauli sign is negative.
inline int effective_angle(const HeisenbergOp& op) {
    return (op.is_dagger() ? -1 : 1) * (op.sign() ? -1 : 1);
}

/// Check whether op_j blocks op_i from commuting past it.
/// Returns true if op_i is blocked (anti-commutes with op_j).
inline bool is_blocked(const HeisenbergOp& op_i, const HeisenbergOp& op_j, const HirModule& hir) {
    switch (op_j.op_type()) {
        case OpType::T_GATE:
        case OpType::CLIFFORD_PHASE:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            return anti_commute(op_i.destab_mask(), op_i.stab_mask(), op_j.destab_mask(),
                                op_j.stab_mask());

        case OpType::NOISE: {
            auto site_idx = static_cast<uint32_t>(op_j.noise_site_idx());
            const auto& channels = hir.noise_sites[site_idx].channels;
            uint64_t xi = static_cast<uint64_t>(op_i.destab_mask());
            uint64_t zi = static_cast<uint64_t>(op_i.stab_mask());
            for (const auto& ch : channels) {
                if (anti_commute_raw(xi, zi, ch.destab_mask, ch.stab_mask)) {
                    return true;
                }
            }
            return false;
        }

        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::READOUT_NOISE:
            return false;

        default:
            return true;
    }
}

}  // namespace

void PeepholeFusionPass::run(HirModule& hir) {
    cancellations_ = 0;
    fusions_ = 0;

    bool changed = true;
    while (changed) {
        changed = false;
        size_t n = hir.ops.size();
        std::vector<bool> deleted(n, false);

        for (size_t i = 0; i < n; ++i) {
            if (deleted[i])
                continue;
            if (hir.ops[i].op_type() != OpType::T_GATE)
                continue;

            const auto& op_i = hir.ops[i];
            auto destab_i = op_i.destab_mask();
            auto stab_i = op_i.stab_mask();

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                const auto& op_j = hir.ops[j];

                // Check if op_j is a matching T gate on the same axis
                if (op_j.op_type() == OpType::T_GATE && op_j.destab_mask() == destab_i &&
                    op_j.stab_mask() == stab_i) {
                    int total = effective_angle(op_i) + effective_angle(op_j);

                    if (total == 0) {
                        // T + T_dag = identity cancellation
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else {
                        // |total| == 2: fuse into S or S_dag
                        hir.ops[i] = HeisenbergOp::make_clifford_phase(
                            destab_i, stab_i, /*sign=*/false, /*dagger=*/(total == -2));
                        deleted[j] = true;
                        ++fusions_;
                    }

                    changed = true;
                    break;
                }

                // Not a match -- check if op_j blocks commutation
                if (is_blocked(op_i, op_j, hir)) {
                    break;
                }
            }
        }

        // Compact: remove deleted ops via erase-remove idiom
        if (changed) {
            size_t write = 0;
            for (size_t read = 0; read < n; ++read) {
                if (!deleted[read]) {
                    if (write != read) {
                        hir.ops[write] = hir.ops[read];
                    }
                    ++write;
                }
            }
            hir.ops.erase(hir.ops.begin() + static_cast<ptrdiff_t>(write), hir.ops.end());
        }
    }
}

}  // namespace ucc
