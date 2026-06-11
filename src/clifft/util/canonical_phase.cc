#include "clifft/util/canonical_phase.h"

#include <bit>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>

namespace clifft::internal {

namespace {

using ChoiRow = stim::PauliString<kStimWidth>;

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

void index_xor(ChoiIndex& dst, const ChoiIndex& src) {
    for (size_t w = 0; w < dst.size(); ++w) {
        dst[w] ^= src[w];
    }
}

/// Phase of P|j> = i^{2*sign + #Y + 2*parity(j & z)} |j ^ x| for a
/// Hermitian signed Pauli P = (-1)^sign X^x Z^z (Y contributing i each).
[[nodiscard]] std::complex<double> pauli_ket_phase(const ChoiRow& p, const ChoiIndex& j) {
    uint32_t phase_idx = p.sign ? 2U : 0U;
    for (size_t w = 0; w < j.size(); ++w) {
        phase_idx += static_cast<uint32_t>(std::popcount(p.xs.u64[w] & p.zs.u64[w]));
        phase_idx += 2U * (static_cast<uint32_t>(std::popcount(p.zs.u64[w] & j[w])) & 1U);
    }
    return kImagPow[phase_idx & 3U];
}

}  // namespace

// Scanning columns from the most significant bit down guarantees each
// pivot row's leading X bit is its pivot, so clearing pivot bits greedily
// minimizes over the support coset.
ChoiSupport build_choi_support(const stim::Tableau<kStimWidth>& tab) {
    std::vector<ChoiRow> rows = choi_generators(tab);
    const auto m = static_cast<uint32_t>(2 * tab.num_qubits);
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
        if (choi_index_bit(base, s.pivots[i])) {
            index_xor(base, s.x_masks[i]);
        }
    }
    s.anchor = std::move(base);
    return s;
}

std::complex<double> choi_amplitude(const ChoiSupport& s, const ChoiIndex& index) {
    ChoiIndex residual = index;
    index_xor(residual, s.anchor);
    ChoiIndex current = s.anchor;
    std::complex<double> amp{1.0, 0.0};

    for (size_t i = 0; i < s.x_rows.size(); ++i) {
        if (!choi_index_bit(residual, s.pivots[i])) {
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

}  // namespace clifft::internal
