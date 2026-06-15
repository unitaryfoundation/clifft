#include "clifft/optimizer/exact_phase_polynomial_t_count_pass.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {

namespace {

constexpr uint8_t kMaxSupportedRank = 4;

struct Axis {
    std::vector<uint64_t> x;
    std::vector<uint64_t> z;
    bool sign = false;
};

struct BasisRow {
    std::vector<uint64_t> bits;
    uint32_t coord = 0;
    uint32_t pivot = 0;
};

struct BlockModel {
    uint8_t rank = 0;
    size_t original_t_count = 0;
    int constant_phase = 0;
    std::vector<Axis> generators;
    std::array<uint8_t, 1u << kMaxSupportedRank> coeff{};
};

struct ModelTerm {
    Axis axis;
    uint32_t coord = 0;
    int coeff = 0;
};

struct Emission {
    uint32_t coord = 0;
    uint8_t coeff = 0;
};

struct Candidate {
    bool found = false;
    uint8_t residual_constant = 0;
    std::vector<Emission> odd_terms;
    std::vector<Emission> residual_terms;
};

[[nodiscard]] uint8_t mod8(int value) {
    return static_cast<uint8_t>(value & 7);
}

[[nodiscard]] std::complex<double> eighth_root(int exponent) {
    const double angle = static_cast<double>(exponent & 7) * std::numbers::pi / 4.0;
    return {std::cos(angle), std::sin(angle)};
}

[[nodiscard]] Axis axis_from_op(const HirModule& hir, const HeisenbergOp& op) {
    Axis axis;
    auto x = hir.destab_mask(op);
    auto z = hir.stab_mask(op);
    axis.x.assign(x.words.begin(), x.words.end());
    axis.z.assign(z.words.begin(), z.words.end());
    axis.sign = hir.sign(op);
    return axis;
}

[[nodiscard]] bool axis_same_unsigned(const Axis& a, const Axis& b) {
    return a.x == b.x && a.z == b.z;
}

[[nodiscard]] std::vector<uint64_t> combined_bits(const Axis& axis) {
    std::vector<uint64_t> bits;
    bits.reserve(axis.x.size() + axis.z.size());
    bits.insert(bits.end(), axis.x.begin(), axis.x.end());
    bits.insert(bits.end(), axis.z.begin(), axis.z.end());
    return bits;
}

[[nodiscard]] bool is_zero_bits(const std::vector<uint64_t>& bits) {
    return std::all_of(bits.begin(), bits.end(), [](uint64_t word) { return word == 0; });
}

[[nodiscard]] uint32_t first_set_bit(const std::vector<uint64_t>& bits) {
    for (uint32_t w = 0; w < bits.size(); ++w) {
        if (bits[w] != 0) {
            return 64U * w + static_cast<uint32_t>(std::countr_zero(bits[w]));
        }
    }
    throw std::logic_error("first_set_bit called on zero vector");
}

void xor_bits(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src) {
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] ^= src[i];
    }
}

[[nodiscard]] bool has_bit(const std::vector<uint64_t>& bits, uint32_t bit) {
    return ((bits[bit / 64] >> (bit % 64)) & 1ULL) != 0;
}

[[nodiscard]] uint32_t reduce_to_coord(std::vector<BasisRow>& basis, const Axis& axis,
                                       uint8_t& rank, uint8_t max_rank, bool& ok) {
    std::vector<uint64_t> bits = combined_bits(axis);
    uint32_t coord = 0;

    for (const auto& row : basis) {
        if (has_bit(bits, row.pivot)) {
            xor_bits(bits, row.bits);
            coord ^= row.coord;
        }
    }

    if (is_zero_bits(bits)) {
        ok = true;
        return coord;
    }

    if (rank >= max_rank) {
        ok = false;
        return 0;
    }

    const uint32_t pivot = first_set_bit(bits);
    const uint32_t row_coord = coord ^ (1u << rank);
    basis.push_back(BasisRow{std::move(bits), row_coord, pivot});
    std::sort(basis.begin(), basis.end(),
              [](const BasisRow& a, const BasisRow& b) { return a.pivot < b.pivot; });
    ++rank;

    ok = true;
    return 1u << (rank - 1);
}

[[nodiscard]] int pauli_product_phase_mod4(const Axis& left, const Axis& right) {
    int phase = 0;
    for (size_t w = 0; w < left.x.size(); ++w) {
        const uint64_t x1 = left.x[w];
        const uint64_t z1 = left.z[w];
        const uint64_t x2 = right.x[w];
        const uint64_t z2 = right.z[w];
        const uint64_t x3 = x1 ^ x2;
        const uint64_t z3 = z1 ^ z2;

        phase += static_cast<int>(std::popcount(x1 & z1));
        phase += static_cast<int>(std::popcount(x2 & z2));
        phase += 2 * static_cast<int>(std::popcount(x2 & z1));
        phase -= static_cast<int>(std::popcount(x3 & z3));
    }
    if (left.sign)
        phase += 2;
    if (right.sign)
        phase += 2;
    return phase & 3;
}

[[nodiscard]] Axis multiply_axes(const Axis& left, const Axis& right) {
    Axis out;
    out.x.resize(left.x.size());
    out.z.resize(left.z.size());
    for (size_t w = 0; w < left.x.size(); ++w) {
        out.x[w] = left.x[w] ^ right.x[w];
        out.z[w] = left.z[w] ^ right.z[w];
    }

    const int phase = pauli_product_phase_mod4(left, right);
    // The pass only multiplies commuting Paulis, so the product is Hermitian.
    if ((phase & 1) != 0) {
        throw std::logic_error("commuting Pauli product had non-Hermitian phase");
    }
    out.sign = (phase == 2);
    return out;
}

[[nodiscard]] Axis product_axis(const std::vector<Axis>& generators, uint32_t coord) {
    Axis out;
    out.x.assign(generators.front().x.size(), 0);
    out.z.assign(generators.front().z.size(), 0);
    out.sign = false;

    for (uint8_t i = 0; i < generators.size(); ++i) {
        if (((coord >> i) & 1u) != 0) {
            out = multiply_axes(out, generators[i]);
        }
    }
    return out;
}

[[nodiscard]] bool is_t_gate_block_op(const HeisenbergOp& op) {
    return op.op_type() == OpType::T_GATE;
}

[[nodiscard]] bool block_is_pairwise_commuting(const HirModule& hir, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        for (size_t j = i + 1; j < end; ++j) {
            if (anti_commute(hir.destab_mask(hir.ops[i]), hir.stab_mask(hir.ops[i]),
                             hir.destab_mask(hir.ops[j]), hir.stab_mask(hir.ops[j]))) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool build_block_model(const HirModule& hir, size_t begin, size_t end,
                                     uint8_t max_rank, BlockModel& model) {
    model = BlockModel{};
    model.original_t_count = end - begin;

    std::vector<BasisRow> basis;
    std::vector<ModelTerm> terms;
    terms.reserve(end - begin);

    for (size_t i = begin; i < end; ++i) {
        Axis axis = axis_from_op(hir, hir.ops[i]);
        const int base_coeff = hir.ops[i].is_dagger() ? 7 : 1;

        bool ok = true;
        const uint32_t coord = reduce_to_coord(basis, axis, model.rank, max_rank, ok);
        if (!ok || coord == 0) {
            return false;
        }

        bool known_generator = false;
        for (const auto& gen : model.generators) {
            if (axis_same_unsigned(axis, gen)) {
                known_generator = true;
                break;
            }
        }
        if (!known_generator && std::popcount(coord) == 1) {
            axis.sign = false;
            model.generators.push_back(std::move(axis));
        }

        terms.push_back(ModelTerm{axis_from_op(hir, hir.ops[i]), coord, base_coeff});
    }

    if (model.generators.size() != model.rank) {
        return false;
    }

    for (const auto& term : terms) {
        const Axis coord_axis = product_axis(model.generators, term.coord);
        if (!axis_same_unsigned(coord_axis, term.axis)) {
            return false;
        }

        int coeff = term.coeff;
        if (coord_axis.sign != term.axis.sign) {
            model.constant_phase = (model.constant_phase + term.coeff) & 7;
            coeff = -coeff;
        }
        model.coeff[term.coord] = mod8(model.coeff[term.coord] + coeff);
    }

    return true;
}

[[nodiscard]] std::array<uint8_t, 1u << kMaxSupportedRank> truth_table(
    const std::array<uint8_t, 1u << kMaxSupportedRank>& coeff, uint8_t rank) {
    std::array<uint8_t, 1u << kMaxSupportedRank> values{};
    const uint32_t limit = 1u << rank;
    for (uint32_t parity = 1; parity < limit; ++parity) {
        if (coeff[parity] == 0)
            continue;
        for (uint32_t assignment = 0; assignment < limit; ++assignment) {
            if ((std::popcount(parity & assignment) & 1u) != 0) {
                values[assignment] = mod8(values[assignment] + coeff[parity]);
            }
        }
    }
    return values;
}

[[nodiscard]] uint8_t degree(uint32_t monomial) {
    return static_cast<uint8_t>(std::popcount(monomial));
}

[[nodiscard]] bool clifford_residual_from_truth(
    const std::array<uint8_t, 1u << kMaxSupportedRank>& residual_truth, uint8_t rank,
    uint8_t& residual_constant, std::vector<Emission>& residual_terms) {
    std::array<uint8_t, 1u << kMaxSupportedRank> anf = residual_truth;
    const uint32_t limit = 1u << rank;

    for (uint8_t bit = 0; bit < rank; ++bit) {
        const uint32_t step = 1u << bit;
        for (uint32_t mask = 0; mask < limit; ++mask) {
            if ((mask & step) != 0) {
                anf[mask] = mod8(static_cast<int>(anf[mask]) - static_cast<int>(anf[mask ^ step]));
            }
        }
    }

    residual_constant = anf[0];

    std::array<uint8_t, kMaxSupportedRank> linear{};
    std::array<std::array<uint8_t, kMaxSupportedRank>, kMaxSupportedRank> quadratic{};

    for (uint32_t mask = 1; mask < limit; ++mask) {
        const uint8_t d = degree(mask);
        const uint8_t c = anf[mask];
        if (d >= 3) {
            if (c != 0)
                return false;
        } else if (d == 2) {
            if (c != 0 && c != 4)
                return false;
            if (c == 4) {
                const uint32_t i = std::countr_zero(mask);
                const uint32_t j = std::countr_zero(mask ^ (1u << i));
                quadratic[i][j] = 4;
            }
        } else {
            if ((c & 1u) != 0)
                return false;
            linear[std::countr_zero(mask)] = c;
        }
    }

    std::array<uint8_t, 1u << kMaxSupportedRank> coeff{};
    for (uint8_t i = 0; i < rank; ++i) {
        for (uint8_t j = i + 1; j < rank; ++j) {
            if (quadratic[i][j] == 4) {
                const uint32_t coord = (1u << i) | (1u << j);
                coeff[coord] = 2;
                linear[i] = mod8(static_cast<int>(linear[i]) - 2);
                linear[j] = mod8(static_cast<int>(linear[j]) - 2);
            }
        }
    }

    for (uint8_t i = 0; i < rank; ++i) {
        if ((linear[i] & 1u) != 0) {
            return false;
        }
        coeff[1u << i] = linear[i];
    }

    residual_terms.clear();
    for (uint32_t coord = 1; coord < limit; ++coord) {
        if (coeff[coord] != 0) {
            residual_terms.push_back(Emission{coord, coeff[coord]});
        }
    }
    return true;
}

[[nodiscard]] bool candidate_is_exact(
    const std::array<uint8_t, 1u << kMaxSupportedRank>& original_truth,
    const std::vector<Emission>& odd_terms, uint8_t rank, uint8_t& residual_constant,
    std::vector<Emission>& residual_terms) {
    std::array<uint8_t, 1u << kMaxSupportedRank> candidate_coeff{};
    for (const auto& term : odd_terms) {
        candidate_coeff[term.coord] = term.coeff;
    }
    const auto candidate_truth = truth_table(candidate_coeff, rank);

    std::array<uint8_t, 1u << kMaxSupportedRank> residual_truth{};
    const uint32_t limit = 1u << rank;
    for (uint32_t assignment = 0; assignment < limit; ++assignment) {
        residual_truth[assignment] = mod8(static_cast<int>(original_truth[assignment]) -
                                          static_cast<int>(candidate_truth[assignment]));
    }

    return clifford_residual_from_truth(residual_truth, rank, residual_constant, residual_terms);
}

void search_signs_for_combination(const std::vector<uint32_t>& coords, size_t idx,
                                  std::vector<Emission>& odd_terms,
                                  const std::array<uint8_t, 1u << kMaxSupportedRank>& original,
                                  uint8_t rank, size_t block_len, Candidate& best) {
    if (idx == coords.size()) {
        uint8_t residual_constant = 0;
        std::vector<Emission> residual_terms;
        if (!candidate_is_exact(original, odd_terms, rank, residual_constant, residual_terms)) {
            return;
        }
        const size_t total_terms = odd_terms.size() + residual_terms.size();
        if (total_terms > block_len) {
            return;
        }
        if (!best.found || total_terms < best.odd_terms.size() + best.residual_terms.size()) {
            best.found = true;
            best.residual_constant = residual_constant;
            best.odd_terms = odd_terms;
            best.residual_terms = std::move(residual_terms);
        }
        return;
    }

    odd_terms.push_back(Emission{coords[idx], 1});
    search_signs_for_combination(coords, idx + 1, odd_terms, original, rank, block_len, best);
    odd_terms.back().coeff = 7;
    search_signs_for_combination(coords, idx + 1, odd_terms, original, rank, block_len, best);
    odd_terms.pop_back();
}

void search_combinations(uint32_t next_coord, uint32_t limit, size_t remaining,
                         std::vector<uint32_t>& coords,
                         const std::array<uint8_t, 1u << kMaxSupportedRank>& original,
                         uint8_t rank, size_t block_len, Candidate& best) {
    if (remaining == 0) {
        std::vector<Emission> odd_terms;
        search_signs_for_combination(coords, 0, odd_terms, original, rank, block_len, best);
        return;
    }
    if (limit - next_coord < remaining) {
        return;
    }

    for (uint32_t coord = next_coord; coord <= limit - remaining; ++coord) {
        coords.push_back(coord);
        search_combinations(coord + 1, limit, remaining - 1, coords, original, rank, block_len,
                            best);
        coords.pop_back();
    }
}

[[nodiscard]] Candidate find_best_candidate(const BlockModel& model, size_t block_len) {
    Candidate best;
    const auto original_truth = truth_table(model.coeff, model.rank);
    const uint32_t limit = 1u << model.rank;

    for (size_t odd_count = 0; odd_count < model.original_t_count; ++odd_count) {
        std::vector<uint32_t> coords;
        search_combinations(1, limit, odd_count, coords, original_truth, model.rank, block_len,
                            best);
        if (best.found) {
            return best;
        }
    }
    return best;
}

void set_op_axis(HirModule& hir, HeisenbergOp& op, const Axis& axis) {
    auto mask = hir.mask_at(op);
    for (size_t w = 0; w < axis.x.size(); ++w) {
        mask.x().words[w] = axis.x[w];
        mask.z().words[w] = axis.z[w];
    }
    mask.set_sign(false);
}

void rewrite_block(HirModule& hir, size_t begin, size_t end, const BlockModel& model,
                   const Candidate& candidate) {
    std::vector<Emission> emissions;
    emissions.reserve(candidate.odd_terms.size() + candidate.residual_terms.size());
    emissions.insert(emissions.end(), candidate.odd_terms.begin(), candidate.odd_terms.end());
    emissions.insert(emissions.end(), candidate.residual_terms.begin(),
                     candidate.residual_terms.end());

    int emitted_constant = 0;
    size_t write = begin;
    for (const auto& emission : emissions) {
        Axis axis = product_axis(model.generators, emission.coord);
        int coeff = emission.coeff;
        if (axis.sign) {
            emitted_constant = (emitted_constant + coeff) & 7;
            coeff = -coeff;
            axis.sign = false;
        }
        coeff &= 7;
        if (coeff == 0)
            continue;

        auto& op = hir.ops[write];
        set_op_axis(hir, op, axis);
        if ((coeff & 1) != 0) {
            hir.demote_to_tgate(op, coeff == 7);
        } else {
            hir.demote_to_phase_rotation(op, static_cast<double>(coeff) / 4.0);
        }
        ++write;
    }

    const int global_delta =
        mod8(model.constant_phase + candidate.residual_constant + emitted_constant);
    hir.global_weight *= eighth_root(global_delta);

    const bool has_source_map = hir.source_map.size() == hir.ops.size();
    if (has_source_map) {
        for (size_t i = begin; i < write; ++i) {
            hir.source_map[i].clear();
        }
    }

    const size_t removed = end - write;
    if (removed == 0)
        return;

    for (size_t read = end; read < hir.ops.size(); ++read) {
        hir.ops[write] = hir.ops[read];
        if (has_source_map) {
            hir.source_map[write] = std::move(hir.source_map[read]);
        }
        ++write;
    }
    hir.ops.erase(hir.ops.end() - static_cast<ptrdiff_t>(removed), hir.ops.end());
    if (has_source_map) {
        hir.source_map.resize(hir.ops.size());
    }
}

}  // namespace

ExactPhasePolynomialTCountPass::ExactPhasePolynomialTCountPass(uint8_t max_rank)
    : max_rank_(max_rank) {
    if (max_rank_ > kMaxSupportedRank) {
        max_rank_ = kMaxSupportedRank;
    }
}

void ExactPhasePolynomialTCountPass::run(HirModule& hir) {
    blocks_considered_ = 0;
    blocks_optimized_ = 0;
    t_removed_ = 0;

    size_t i = 0;
    while (i < hir.ops.size()) {
        if (!is_t_gate_block_op(hir.ops[i])) {
            ++i;
            continue;
        }

        const size_t begin = i;
        while (i < hir.ops.size() && is_t_gate_block_op(hir.ops[i])) {
            ++i;
        }
        const size_t end = i;
        const size_t block_len = end - begin;
        if (block_len < 2) {
            continue;
        }

        ++blocks_considered_;
        if (!block_is_pairwise_commuting(hir, begin, end)) {
            continue;
        }

        BlockModel model;
        if (!build_block_model(hir, begin, end, max_rank_, model)) {
            continue;
        }

        const Candidate candidate = find_best_candidate(model, block_len);
        if (!candidate.found || candidate.odd_terms.size() >= block_len) {
            continue;
        }

        rewrite_block(hir, begin, end, model, candidate);
        ++blocks_optimized_;
        t_removed_ += block_len - candidate.odd_terms.size();
        i = begin + candidate.odd_terms.size() + candidate.residual_terms.size();
    }
}

}  // namespace clifft
