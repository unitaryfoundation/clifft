#include "clifft/optimizer/peephole.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/util/constants.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <utility>
#include <vector>

namespace clifft {

namespace {

// =========================================================================
// Symplectic conjugation: absorb a virtual S gate into downstream ops
// =========================================================================

/// Conjugate Pauli Q by S_P in place.
///
/// For S gate (is_dagger=false), c_phase=3 computes S^dag Q S.
/// For S_dag gate (is_dagger=true), c_phase=1 computes S Q S^dag.
///
/// The symplectic product of two single-qubit Paulis A_i, B_i contributes
/// a phase i^{+1} when A*B advances cyclically (X->Y->Z->X) and i^{-1}
/// when it retreats. The total phase from the commutator [P, Q] is
/// i^{sum of per-qubit contributions}.
inline void conjugate_pauli_by_S(MaskView x_p, MaskView z_p, bool sign_p, MutableMaskView x_q,
                                 MutableMaskView z_q, bool& sign_q, bool is_dagger) {
    if (!anti_commute(x_p, z_p, x_q, z_q))
        return;

    // Compute per-qubit phase contributions from the Pauli product P*Q.
    // Each qubit pair (A_i, B_i) contributes +1 or -1 to the total phase
    // exponent based on the cyclic ordering X->Y->Z->X.
    int phase = 0;
    for (uint32_t w = 0; w < x_p.num_words(); ++w) {
        uint64_t X1 = x_p.words[w];
        uint64_t Z1 = z_p.words[w];
        uint64_t X2 = x_q.words[w];
        uint64_t Z2 = z_q.words[w];

        // +1 phase contributions: XY->Z, YZ->X, ZX->Y
        uint64_t mask_plus = (X1 & ~Z1 & X2 & Z2) | (X1 & Z1 & ~X2 & Z2) | (~X1 & Z1 & X2 & ~Z2);
        // -1 phase contributions: YX->Z, ZY->X, XZ->Y
        uint64_t mask_minus = (X1 & ~Z1 & ~X2 & Z2) | (X1 & Z1 & X2 & ~Z2) | (~X1 & Z1 & X2 & Z2);

        phase += std::popcount(mask_plus);
        phase -= std::popcount(mask_minus);
    }

    int p_mod4 = ((phase % 4) + 4) % 4;
    // S (is_dagger=false): downstream we want S^dag Q S = +i P Q -> c_phase=1
    // S_dag (is_dagger=true): downstream we want S Q S^dag = -i P Q -> c_phase=3
    int c_phase = is_dagger ? 3 : 1;

    int total_phase = (p_mod4 + c_phase) % 4;
    if (sign_p)
        total_phase = (total_phase + 2) % 4;
    if (sign_q)
        total_phase = (total_phase + 2) % 4;

    sign_q = (total_phase == 2);
    x_q.xor_with(x_p);
    z_q.xor_with(z_p);
}

// =========================================================================
// Canonical tableau phase tracking
// =========================================================================
//
// A stim::Tableau fixes its Clifford only up to global phase;
// to_flat_unitary_matrix() resolves the ambiguity by scaling the row-major
// flat matrix so its first nonzero entry is real positive. Replacing the
// physical fused rotation R = Pi_+ + (+-i) Pi_- (projector form, including
// its intrinsic e^{+-i*pi/4} relative to the SU(2) rotation) with the
// tableau update U_C' = U_C * S_P therefore shifts the API-visible global
// phase: canonical(U_C') = canonical(U_C) * R holds only up to an eighth
// root of unity. The helpers below compute that root so the pass can fold
// it into hir.global_weight and keep get_statevector() exact.
//
// The flat matrix of a tableau is the Choi state of the Clifford: a
// 2n-qubit stabilizer state whose flat index packs (row << n) | col with
// qubit q <-> bit q, stabilized by X_k (x) U(X_k) and Z_k (x) U(Z_k) on
// the (input, output) halves. The first nonzero flat-matrix entry sits at
// the minimum-integer support index of that state, and entries relative
// to it follow by walking support generators, so the whole computation is
// polynomial in n.

using ChoiRow = stim::PauliString<kStimWidth>;
using ChoiIndex = std::vector<uint64_t>;

/// Row-reduced description of a tableau's Choi state: support generators
/// in descending-pivot order plus the minimum-integer support index, which
/// stim's matrix canonicalization anchors at a positive real amplitude.
struct ChoiSupport {
    std::vector<ChoiRow> x_rows;
    std::vector<uint32_t> pivots;
    std::vector<ChoiIndex> x_masks;
    ChoiIndex anchor;
};

std::vector<ChoiRow> choi_generators(const stim::Tableau<kStimWidth>& tab) {
    const size_t n = tab.num_qubits;
    std::vector<ChoiRow> gens;
    gens.reserve(2 * n);
    for (size_t k = 0; k < n; ++k) {
        for (bool z_input : {false, true}) {
            const auto image = z_input ? tab.zs[k] : tab.xs[k];
            ChoiRow g(2 * n);
            if (z_input) {
                g.zs[k] = true;
            } else {
                g.xs[k] = true;
            }
            for (size_t q = 0; q < n; ++q) {
                g.xs[n + q] = image.xs[q];
                g.zs[n + q] = image.zs[q];
            }
            g.sign = image.sign;
            gens.push_back(std::move(g));
        }
    }
    return gens;
}

[[nodiscard]] ChoiIndex choi_x_mask(const ChoiRow& row, uint32_t words) {
    ChoiIndex mask(words, 0);
    for (uint32_t w = 0; w < words; ++w) {
        mask[w] = row.xs.u64[w];
    }
    return mask;
}

[[nodiscard]] bool index_bit(const ChoiIndex& idx, uint32_t bit) {
    return ((idx[bit / 64] >> (bit % 64)) & 1U) != 0;
}

void index_xor(ChoiIndex& dst, const ChoiIndex& src) {
    for (size_t w = 0; w < dst.size(); ++w) {
        dst[w] ^= src[w];
    }
}

constexpr std::complex<double> kIPow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

/// Phase of P|j> = i^{2*sign + #Y + 2*parity(j & z)} |j ^ x| for a
/// Hermitian signed Pauli P = (-1)^sign X^x Z^z (Y contributing i each).
[[nodiscard]] std::complex<double> pauli_ket_phase(const ChoiRow& p, const ChoiIndex& j) {
    uint32_t phase_idx = p.sign ? 2U : 0U;
    for (size_t w = 0; w < j.size(); ++w) {
        phase_idx += static_cast<uint32_t>(std::popcount(p.xs.u64[w] & p.zs.u64[w]));
        phase_idx += 2U * (static_cast<uint32_t>(std::popcount(p.zs.u64[w] & j[w])) & 1U);
    }
    return kIPow[phase_idx & 3U];
}

/// Gauss-Jordan reduce the Choi stabilizer group and locate the
/// minimum-integer support index. Scanning columns from the most
/// significant bit down guarantees each pivot row's leading X bit is its
/// pivot, so clearing pivot bits greedily minimizes over the support coset.
[[nodiscard]] ChoiSupport build_choi_support(std::vector<ChoiRow> rows, size_t num_choi_qubits) {
    const auto m = static_cast<uint32_t>(num_choi_qubits);
    const uint32_t words = (m + 63U) / 64U;

    ChoiSupport s;

    size_t rank_x = 0;
    for (uint32_t col = m; col-- > 0;) {
        size_t pivot = rows.size();
        for (size_t r = rank_x; r < rows.size(); ++r) {
            if (rows[r].xs[col]) {
                pivot = r;
                break;
            }
        }
        if (pivot == rows.size()) {
            continue;
        }
        std::swap(rows[rank_x], rows[pivot]);
        for (size_t r = 0; r < rows.size(); ++r) {
            if (r != rank_x && rows[r].xs[col]) {
                rows[r].ref() *= rows[rank_x];
            }
        }
        s.pivots.push_back(col);
        ++rank_x;
    }

    // The remaining rows are pure-Z constraints. After Gauss-Jordan over
    // their Z parts, each pivot equation fixes one bit of a particular
    // support element (zero on all free columns). Signs must be read only
    // after the reduction finishes: clearing a later column can multiply an
    // earlier pivot row and flip its sign.
    std::vector<uint32_t> z_pivots;
    size_t rank_z = rank_x;
    for (uint32_t col = m; col-- > 0;) {
        size_t pivot = rows.size();
        for (size_t r = rank_z; r < rows.size(); ++r) {
            if (rows[r].zs[col]) {
                pivot = r;
                break;
            }
        }
        if (pivot == rows.size()) {
            continue;
        }
        std::swap(rows[rank_z], rows[pivot]);
        for (size_t r = rank_x; r < rows.size(); ++r) {
            if (r != rank_z && rows[r].zs[col]) {
                rows[r].ref() *= rows[rank_z];
            }
        }
        z_pivots.push_back(col);
        ++rank_z;
    }
    ChoiIndex base(words, 0);
    for (size_t i = 0; i < z_pivots.size(); ++i) {
        if (rows[rank_x + i].sign) {
            base[z_pivots[i] / 64] |= uint64_t{1} << (z_pivots[i] % 64);
        }
    }
    // A full-rank tableau group leaves no inconsistent identity rows.
    for (size_t r = rank_z; r < rows.size(); ++r) {
        assert(!rows[r].sign && "inconsistent Choi stabilizer group");
    }

    s.x_rows.assign(std::make_move_iterator(rows.begin()),
                    std::make_move_iterator(rows.begin() + static_cast<ptrdiff_t>(rank_x)));
    s.x_masks.reserve(rank_x);
    for (const auto& row : s.x_rows) {
        s.x_masks.push_back(choi_x_mask(row, words));
    }

    for (size_t i = 0; i < s.x_rows.size(); ++i) {
        if (index_bit(base, s.pivots[i])) {
            index_xor(base, s.x_masks[i]);
        }
    }
    s.anchor = std::move(base);
    return s;
}

/// Amplitude of |index> in the Choi state, in units where the anchor
/// (stim's positive-real first nonzero matrix entry) has amplitude 1.
/// Returns 0 when the index lies outside the support.
[[nodiscard]] std::complex<double> choi_amplitude(const ChoiSupport& s, const ChoiIndex& index) {
    ChoiIndex residual = index;
    index_xor(residual, s.anchor);
    ChoiIndex current = s.anchor;
    std::complex<double> amp{1.0, 0.0};

    for (size_t i = 0; i < s.x_rows.size(); ++i) {
        if (!index_bit(residual, s.pivots[i])) {
            continue;
        }
        amp *= pauli_ket_phase(s.x_rows[i], current);
        index_xor(current, s.x_masks[i]);
        index_xor(residual, s.x_masks[i]);
    }

    for (uint64_t word : residual) {
        if (word != 0) {
            return {0.0, 0.0};
        }
    }
    return amp;
}

}  // namespace

namespace internal {

void apply_s_to_tableau(stim::Tableau<kStimWidth>& tab, MaskView x_v, MaskView z_v, bool sign_v,
                        bool is_dagger) {
    // Map P_virt forward through U_C to get P_phys in O(n^2), then
    // conjugate all physical rows by P_phys in O(n^2).
    const size_t words = std::min<size_t>((tab.num_qubits + 63) / 64, x_v.num_words());

    stim::PauliString<kStimWidth> p_virt(tab.num_qubits);
    for (size_t w = 0; w < words; ++w) {
        p_virt.xs.u64[w] = x_v.words[w];
        p_virt.zs.u64[w] = z_v.words[w];
    }
    p_virt.sign = sign_v;

    stim::PauliString<kStimWidth> p_phys = tab(p_virt);

    std::vector<uint64_t> px_phys(words, 0);
    std::vector<uint64_t> pz_phys(words, 0);
    for (size_t w = 0; w < words; ++w) {
        px_phys[w] = p_phys.xs.u64[w];
        pz_phys[w] = p_phys.zs.u64[w];
    }
    bool psign_phys = p_phys.sign;
    MaskView px_view{std::span<const uint64_t>(px_phys)};
    MaskView pz_view{std::span<const uint64_t>(pz_phys)};

    std::vector<uint64_t> q_x(words, 0);
    std::vector<uint64_t> q_z(words, 0);
    MutableMaskView qx_view{std::span<uint64_t>(q_x)};
    MutableMaskView qz_view{std::span<uint64_t>(q_z)};

    // Tableau generators: S_P X_q S_P^dag, so pass !is_dagger
    for (size_t q = 0; q < tab.num_qubits; ++q) {
        auto apply_to_ps = [&](stim::PauliStringRef<kStimWidth> row) {
            for (size_t w = 0; w < words; ++w) {
                q_x[w] = row.xs.u64[w];
                q_z[w] = row.zs.u64[w];
            }
            bool q_sign = row.sign;

            conjugate_pauli_by_S(px_view, pz_view, psign_phys, qx_view, qz_view, q_sign,
                                 !is_dagger);

            for (size_t w = 0; w < words; ++w) {
                row.xs.u64[w] = q_x[w];
                row.zs.u64[w] = q_z[w];
            }
            row.sign = q_sign;
        };

        apply_to_ps(tab.xs[q]);
        apply_to_ps(tab.zs[q]);
    }
}

std::complex<double> s_absorption_phase(const stim::Tableau<kStimWidth>& original,
                                        const stim::Tableau<kStimWidth>& updated, MaskView x_v,
                                        MaskView z_v, bool sign_v, bool is_dagger) {
    const size_t n = original.num_qubits;
    const ChoiSupport old_support = build_choi_support(choi_generators(original), 2 * n);
    const ChoiSupport new_support = build_choi_support(choi_generators(updated), 2 * n);

    // R = a*I + b*P in matrix form, so column c of R has entries a at row c
    // and b * phase(P|c>) at row c ^ x. The new canonical anchor packs
    // (row << n) | col, with col in the low half, so the partner flat index
    // is anchor ^ x_v.
    const std::complex<double> a =
        is_dagger ? std::complex<double>{0.5, -0.5} : std::complex<double>{0.5, 0.5};
    const std::complex<double> b =
        is_dagger ? std::complex<double>{0.5, 0.5} : std::complex<double>{0.5, -0.5};

    const ChoiIndex& anchor = new_support.anchor;
    const uint32_t mask_words =
        std::min<uint32_t>(x_v.num_words(), static_cast<uint32_t>(anchor.size()));

    // phase(P|c>) for c = low half of the anchor. Mask words beyond the
    // input half never overlap z_v, whose bits above n are zero.
    uint32_t alpha_idx = sign_v ? 2U : 0U;
    bool x_is_zero = true;
    for (uint32_t w = 0; w < mask_words; ++w) {
        alpha_idx += static_cast<uint32_t>(std::popcount(x_v.words[w] & z_v.words[w]));
        alpha_idx += 2U * (static_cast<uint32_t>(std::popcount(z_v.words[w] & anchor[w])) & 1U);
        x_is_zero &= (x_v.words[w] == 0);
    }
    const std::complex<double> alpha = kIPow[alpha_idx & 3U];

    std::complex<double> w_entry;
    if (x_is_zero) {
        w_entry = (a + b * alpha) * choi_amplitude(old_support, anchor);
    } else {
        ChoiIndex partner = anchor;
        for (uint32_t w = 0; w < mask_words; ++w) {
            partner[w] ^= x_v.words[w];
        }
        w_entry = a * choi_amplitude(old_support, anchor) +
                  b * alpha * choi_amplitude(old_support, partner);
    }

    const double mag = std::abs(w_entry);
    assert(mag > 0.25 && "canonical anchor outside expected support");
    return w_entry / mag;
}

}  // namespace internal

namespace {

/// Absorb a virtual S gate on Pauli generator (x_v, z_v) into all
/// downstream HIR operations and the final tableau.
void apply_virtual_s_downstream(HirModule& hir, size_t start_idx, MaskView x_v, MaskView z_v,
                                bool sign_v, bool is_dagger, const std::vector<uint8_t>& deleted) {
    // 1. Conjugate all downstream ops
    for (size_t k = start_idx; k < hir.ops.size(); ++k) {
        if (deleted[k])
            continue;
        auto& op = hir.ops[k];

        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI:
            case OpType::EXP_VAL: {
                auto m = hir.mask_at(op);
                bool sign_i = m.sign();
                conjugate_pauli_by_S(x_v, z_v, sign_v, m.x(), m.z(), sign_i, is_dagger);
                m.set_sign(sign_i);
                break;
            }

            case OpType::PHASE_ROTATION: {
                auto m = hir.mask_at(op);
                bool sign_before = m.sign();
                bool sign_i = sign_before;
                conjugate_pauli_by_S(x_v, z_v, sign_v, m.x(), m.z(), sign_i, is_dagger);

                // If S-conjugation flipped the Pauli axis sign, do NOT
                // negate alpha. The physical SU(2) rotation direction is
                // preserved natively by the backend's own sign handling.
                // However, the front-end extracted the U(1) global phase
                // using the OLD sign, and the backend expects global_weight
                // to match the NEW sign. Patch the difference.
                if (sign_i != sign_before) {
                    double corr = op.alpha() * std::numbers::pi * (sign_before ? -1.0 : 1.0);
                    hir.global_weight *= std::complex<double>(std::cos(corr), std::sin(corr));
                }

                m.set_sign(sign_i);
                break;
            }

            case OpType::NOISE: {
                auto site_idx = static_cast<uint32_t>(op.noise_site_idx());
                for (auto& ch : hir.noise_sites[site_idx].channels) {
                    auto m = hir.noise_channel_masks.mut_at(ch.mask);
                    bool dummy_sign = false;
                    conjugate_pauli_by_S(x_v, z_v, sign_v, m.x(), m.z(), dummy_sign, is_dagger);
                }
                break;
            }

            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::NUM_OP_TYPES:
                break;
        }
    }

    // 2. Final Tableau: U_C' = U_C S (requires inverted dagger flag)
    if (hir.final_tableau.has_value()) {
        const stim::Tableau<kStimWidth> original = *hir.final_tableau;
        internal::apply_s_to_tableau(*hir.final_tableau, x_v, z_v, sign_v, is_dagger);

        // The op stream lost the exact rotation R but the tableau only
        // gained its symplectic action; restore the dropped global phase.
        hir.global_weight *=
            internal::s_absorption_phase(original, *hir.final_tableau, x_v, z_v, sign_v, is_dagger);
    }
}

// =========================================================================
// Peephole helpers
// =========================================================================

/// Normalize a T gate to a positive Pauli sign, absorbing the phase
/// difference into global_weight. The identity is:
///   T(-P) = exp(+i*pi/4) * T_dag(+P)
///   T_dag(-P) = exp(-i*pi/4) * T(+P)
/// After normalization, all T gates have sign=false and the effective
/// rotation direction is determined solely by the dagger flag.
inline void normalize_t_sign(HirModule& hir, HeisenbergOp& op,
                             std::complex<double>& global_weight) {
    if (op.op_type() == OpType::T_GATE && hir.sign(op)) {
        global_weight *= op.is_dagger() ? kExpMinusIPiOver4 : kExpIPiOver4;
        op.set_dagger(!op.is_dagger());
        hir.set_sign(op, false);
    }
}

/// Check whether op_j blocks op_i from commuting past it.
/// Returns true if op_i is blocked (anti-commutes with op_j).
inline bool is_blocked(const HeisenbergOp& op_i, const HeisenbergOp& op_j, const HirModule& hir) {
    switch (op_j.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            return anti_commute(hir.destab_mask(op_i), hir.stab_mask(op_i), hir.destab_mask(op_j),
                                hir.stab_mask(op_j));

        case OpType::NOISE: {
            auto site_idx = static_cast<uint32_t>(op_j.noise_site_idx());
            const auto& channels = hir.noise_sites[site_idx].channels;
            for (const auto& ch : channels) {
                auto cv = hir.noise_channel_masks.at(ch.mask);
                if (anti_commute(hir.destab_mask(op_i), hir.stab_mask(op_i), cv.x(), cv.z())) {
                    return true;
                }
            }
            return false;
        }

        case OpType::EXP_VAL:
            return true;

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

    bool has_source_map = hir.source_map.size() == hir.ops.size();

    bool changed = true;
    while (changed) {
        changed = false;
        size_t n = hir.ops.size();
        std::vector<uint8_t> deleted(n, 0);

        for (size_t i = 0; i < n; ++i) {
            if (deleted[i])
                continue;
            if (hir.ops[i].op_type() != OpType::T_GATE)
                continue;

            normalize_t_sign(hir, hir.ops[i], hir.global_weight);
            // After normalization the mask handle is unchanged; capture the
            // arena-resident views once so we can compare against later ops.
            auto destab_i = hir.destab_mask(hir.ops[i]);
            auto stab_i = hir.stab_mask(hir.ops[i]);

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                // Normalize candidate T gates before comparison so that
                // sign-induced dagger flips are accounted for in effective_angle.
                if (hir.ops[j].op_type() == OpType::T_GATE)
                    normalize_t_sign(hir, hir.ops[j], hir.global_weight);

                // Re-fetch op_i in case anything mutated it (it didn't, but
                // be explicit since views must remain valid).
                const auto& op_i = hir.ops[i];
                const auto& op_j = hir.ops[j];

                // Check if op_j is a matching T gate on the same axis.
                // After normalization both ops have sign=false, so
                // effective_angle depends only on the dagger flag.
                if (op_j.op_type() == OpType::T_GATE && hir.destab_mask(op_j) == destab_i &&
                    hir.stab_mask(op_j) == stab_i) {
                    int dir_i = op_i.is_dagger() ? -1 : 1;
                    int dir_j = op_j.is_dagger() ? -1 : 1;
                    int total = dir_i + dir_j;

                    if (total == 0) {
                        // T + T_dag = identity cancellation
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else {
                        // |total| == 2: fuse into S, absorb downstream
                        bool s_is_dagger = (total == -2);
                        deleted[i] = true;
                        deleted[j] = true;

                        apply_virtual_s_downstream(hir, j + 1, destab_i, stab_i, false, s_is_dagger,
                                                   deleted);
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

        // PHASE_ROTATION fusion: merge same-axis rotations by adding angles.
        for (size_t i = 0; i < n; ++i) {
            if (deleted[i])
                continue;
            if (hir.ops[i].op_type() != OpType::PHASE_ROTATION)
                continue;

            auto destab_i = hir.destab_mask(hir.ops[i]);
            auto stab_i = hir.stab_mask(hir.ops[i]);

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                const auto& op_i = hir.ops[i];
                const auto& op_j = hir.ops[j];

                if (op_j.op_type() == OpType::PHASE_ROTATION && hir.destab_mask(op_j) == destab_i &&
                    hir.stab_mask(op_j) == stab_i) {
                    // Compute fused angle accounting for sign bits.
                    double alpha_i = op_i.alpha() * (hir.sign(op_i) ? -1.0 : 1.0);
                    double alpha_j = op_j.alpha() * (hir.sign(op_j) ? -1.0 : 1.0);
                    double fused = alpha_i + alpha_j;

                    // Normalize to [0, 2) relative phase range
                    fused = fused - 2.0 * std::floor(fused / 2.0);

                    constexpr double kDemoteEps = 1e-12;
                    if (std::abs(fused) < kDemoteEps || std::abs(fused - 2.0) < kDemoteEps) {
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else if (std::abs(fused - 0.5) < kDemoteEps) {
                        // S gate: absorb downstream (phase already in global_weight)
                        deleted[i] = true;
                        deleted[j] = true;
                        apply_virtual_s_downstream(hir, j + 1, destab_i, stab_i, false, false,
                                                   deleted);
                        ++fusions_;
                    } else if (std::abs(fused - 1.5) < kDemoteEps) {
                        // S_dag gate: absorb downstream (phase already in global_weight)
                        deleted[i] = true;
                        deleted[j] = true;
                        apply_virtual_s_downstream(hir, j + 1, destab_i, stab_i, false, true,
                                                   deleted);
                        ++fusions_;
                    } else if (std::abs(fused - 0.25) < kDemoteEps) {
                        hir.demote_to_tgate(hir.ops[i], false);
                        if (has_source_map) {
                            auto& dst = hir.source_map[i];
                            auto& src = hir.source_map[j];
                            dst.insert(dst.end(), src.begin(), src.end());
                        }
                        deleted[j] = true;
                        ++fusions_;
                    } else if (std::abs(fused - 1.75) < kDemoteEps) {
                        hir.demote_to_tgate(hir.ops[i], /*dagger=*/true);
                        if (has_source_map) {
                            auto& dst = hir.source_map[i];
                            auto& src = hir.source_map[j];
                            dst.insert(dst.end(), src.begin(), src.end());
                        }
                        deleted[j] = true;
                        ++fusions_;
                    } else {
                        hir.demote_to_phase_rotation(hir.ops[i], fused);
                        if (has_source_map) {
                            auto& dst = hir.source_map[i];
                            auto& src = hir.source_map[j];
                            dst.insert(dst.end(), src.begin(), src.end());
                        }
                        deleted[j] = true;
                        ++fusions_;
                    }

                    changed = true;
                    break;
                }

                if (is_blocked(op_i, op_j, hir)) {
                    break;
                }
            }
        }

        // Demote standalone PHASE_ROTATIONs at Clifford/T angles.
        // S and S_dag angles are absorbed downstream; T and T_dag are
        // demoted to cheaper T_GATE ops; identities are deleted.
        for (size_t i = 0; i < n; ++i) {
            if (deleted[i] || hir.ops[i].op_type() != OpType::PHASE_ROTATION)
                continue;

            double alpha = hir.ops[i].alpha() * (hir.sign(hir.ops[i]) ? -1.0 : 1.0);
            double a_mod2 = alpha - 2.0 * std::floor(alpha / 2.0);

            constexpr double kDemoteEps = 1e-12;
            if (std::abs(a_mod2) < kDemoteEps || std::abs(a_mod2 - 2.0) < kDemoteEps) {
                deleted[i] = true;
                ++cancellations_;
                changed = true;
            } else if (std::abs(a_mod2 - 0.5) < kDemoteEps) {
                // S: absorb downstream (phase already in global_weight)
                apply_virtual_s_downstream(hir, i + 1, hir.destab_mask(hir.ops[i]),
                                           hir.stab_mask(hir.ops[i]), false, false, deleted);
                deleted[i] = true;
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 1.5) < kDemoteEps) {
                // S_dag: absorb downstream (phase already in global_weight)
                apply_virtual_s_downstream(hir, i + 1, hir.destab_mask(hir.ops[i]),
                                           hir.stab_mask(hir.ops[i]), false, true, deleted);
                deleted[i] = true;
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 0.25) < kDemoteEps) {
                hir.demote_to_tgate(hir.ops[i], false);
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 1.75) < kDemoteEps) {
                hir.demote_to_tgate(hir.ops[i], /*dagger=*/true);
                ++fusions_;
                changed = true;
            }
        }

        // Compact: remove deleted ops via erase-remove idiom
        if (changed) {
            size_t write = 0;
            for (size_t read = 0; read < n; ++read) {
                if (!deleted[read]) {
                    if (write != read) {
                        hir.ops[write] = hir.ops[read];
                        if (has_source_map) {
                            hir.source_map[write] = std::move(hir.source_map[read]);
                        }
                    }
                    ++write;
                }
            }
            hir.ops.erase(hir.ops.begin() + static_cast<ptrdiff_t>(write), hir.ops.end());
            if (has_source_map) {
                hir.source_map.resize(write);
            }
        }
    }
}

}  // namespace clifft
