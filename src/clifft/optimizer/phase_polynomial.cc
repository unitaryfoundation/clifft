#include "clifft/optimizer/phase_polynomial.h"

#include "clifft/optimizer/tohpe.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace clifft::phase_detail {
namespace {

template <size_t N>
BitMask<N> bit(uint32_t i) {
    BitMask<N> result;
    result.bit_set(i, true);
    return result;
}

template <class Table, class Mask, class Work>
void add_term(Table& table, Mask mask, int coefficient, Work& work) {
    work.charge();
    auto it = table.lower_bound(mask);
    if (it == table.end() || it->first != mask) {
        if (coefficient & 7) {
            work.check_terms(table.size() + 1);
            table.emplace_hint(it, mask, coefficient & 7);
        }
    } else {
        it->second = (it->second + coefficient) & 7;
        if (!it->second)
            table.erase(it);
    }
}

}  // namespace

bool WideBudget::try_charge(size_t amount) {
    if (work_exhausted || amount > max_updates - updates)
        return false;
    updates += amount;
    return true;
}

void WideBudget::charge(size_t amount) {
    if (!try_charge(amount)) {
        work_exhausted = true;
        throw SearchLimit{};
    }
}

void WideBudget::check_terms(size_t size) {
    if (size > max_terms) {
        term_limit_reached = true;
        throw SearchLimit{};
    }
}

template <size_t N>
void PhasePolynomial<N>::Signature::add(Mask parity, int coefficient, Work& work) {
    for (auto bits = parity; !bits.is_zero();) {
        const auto i = bits.lowest_bit();
        bits.clear_lowest_bit();
        work.charge();
        auto& weight = linear.at(i);
        weight = (weight + coefficient) & 7;
        if (!(coefficient & 3))
            continue;
        work.charge(bits.popcount());
        for (auto rest = bits; !rest.is_zero(); rest.clear_lowest_bit()) {
            const auto it = pairs.try_emplace(std::pair{i, rest.lowest_bit()}).first;
            auto& pair = it->second;
            pair.weight = (pair.weight + coefficient) & 3;
            if (coefficient & 1)
                pair.odd_columns ^= parity;
            if (!pair.weight && pair.odd_columns.is_zero())
                pairs.erase(it);
        }
    }
}

template <size_t N>
void PhasePolynomial<N>::add(Mask parity, int coefficient, Work& work) {
    add_term(terms_, parity, coefficient, work);
    if (signature_)
        signature_->add(parity, coefficient, work);
}

template <size_t N>
const typename PhasePolynomial<N>::Signature& PhasePolynomial<N>::signature(Work& work) {
    if (!signature_) {
        signature_.emplace();
        for (const auto& [mask, coefficient] : terms_)
            signature_->add(mask, coefficient, work);
    }
    return *signature_;
}

template <size_t N>
size_t PhasePolynomial<N>::odd_terms() const {
    return std::ranges::count_if(terms_, [](const auto& term) { return term.second & 1; });
}

template <size_t N>
std::optional<PauliDerivative<N>> PhasePolynomial<N>::Signature::derivative(Mask flip,
                                                                            Work& work) const {
    // Changing a parity a contributes c*(1 - 2*a(x)) when a(flip) is odd.
    // Contract the stored single, pair and triple moments with the flip to
    // obtain the constant p(flip) and the weights of those changing parities.
    PauliDerivative<N> result;
    std::array<uint8_t, N> weights{};
    work.charge(N + pairs.size());
    for (auto bits = flip; !bits.is_zero(); bits.clear_lowest_bit()) {
        const auto i = bits.lowest_bit();
        weights.at(i) = linear.at(i);
        result.constant += weights.at(i);
    }
    Mask triples;
    for (const auto& [indices, pair] : pairs) {
        if ((pair.odd_columns & flip).popcount() & 1)
            return std::nullopt;  // A quadratic term survives in the derivative.
        const auto [i, j] = indices;
        const bool fi = flip.bit_get(i), fj = flip.bit_get(j);
        if (fi)
            weights[j] += pair.weight;
        if (fj)
            weights[i] += pair.weight;
        if (fi && fj) {
            result.constant -= 2 * pair.weight;
            triples ^= pair.odd_columns;
        }
    }
    // The cubic contribution to p(flip) vanishes: every pair row above
    // has even overlap with flip.
    result.constant &= 7;
    if (result.constant & 1)
        return std::nullopt;
    result.parity = triples;
    for (uint32_t i = 0; i < N; ++i) {
        // The linear slope is -2*(weights[i] - 2*triples[i]) modulo eight.
        if (weights[i] & 1)
            return std::nullopt;
        if (weights[i] & 2)
            result.parity.bit_xor(i);
    }
    return result;
}

template <size_t N>
std::optional<PauliDerivative<N>> PhasePolynomial<N>::derivative(Mask flip, Work& work) {
    PauliDerivative<N> result;
    if (flip.is_zero())
        return result;
    const auto& phase = signature(work);
    if (phase.pairs.empty()) {
        // Flipping x changes c*x by c - 2*c*x. Identity regions and
        // single-axis phases need only the coefficients of the flipped axes.
        work.charge(flip.popcount());
        for (; !flip.is_zero(); flip.clear_lowest_bit()) {
            const auto i = flip.lowest_bit();
            const auto c = phase.linear.at(i);
            if (c & 1)
                return std::nullopt;
            result.constant = (result.constant + c) & 7;
            result.parity.bit_set(i, c & 2);
        }
        return result;
    }
    if (phase.pairs.size() < terms_.size())
        return phase.derivative(flip, work);
    // For every parity p that changes, the difference is c*(1-2*p).
    // Its quadratic part is binary and higher degrees vanish modulo eight.
    std::array<uint8_t, N> linear{};
    std::array<Mask, N> quadratic{};
    work.charge(terms_.size());
    for (const auto& [parity, coefficient] : terms_) {
        if (!((parity & flip).popcount() & 1))
            continue;
        result.constant = (result.constant + coefficient) & 7;
        work.charge(parity.popcount());
        for (auto bits = parity; !bits.is_zero();) {
            const auto i = bits.lowest_bit();
            bits.clear_lowest_bit();
            auto& weight = linear.at(i);
            weight = (weight - 2 * coefficient) & 7;
            if (coefficient & 1)
                quadratic.at(i) ^= bits;
        }
    }
    if (result.constant & 1)
        return std::nullopt;
    for (uint32_t i = 0; i < N; ++i) {
        if (!quadratic[i].is_zero() || (linear[i] & 3))
            return std::nullopt;
        result.parity.bit_set(i, linear[i] == 4);
    }
    return result;
}

template <size_t N>
std::vector<BitMask<N>> PhasePolynomial<N>::core_axes(uint32_t width, Work& work) {
    const auto& phase = signature(work);
    assert(width <= N);
    // Translations with Pauli derivatives form the null space of the
    // pair-versus-single signature, including its repeated-index rows.
    std::array<Mask, N> rows{};
    const auto insert = [&](Mask row) {
        while (!row.is_zero()) {
            auto& pivot = rows.at(row.lowest_bit());
            if (pivot.is_zero()) {
                pivot = row;
                return;
            }
            row ^= pivot;
        }
    };
    std::vector<Mask> diagonal(width);
    Mask constant;
    for (uint32_t i = 0; i < width; ++i) {
        if (phase.linear[i] & 1) {
            constant |= bit<N>(i);
            diagonal[i] = bit<N>(i);
        }
    }
    insert(constant);
    for (const auto& [indices, pair] : phase.pairs) {
        const auto [i, j] = indices;
        if (pair.weight & 1) {
            diagonal[i] |= bit<N>(j);
            diagonal[j] |= bit<N>(i);
        }
        work.charge(width + 1);
        insert(pair.odd_columns);
    }
    for (auto row : diagonal) {
        work.charge(width + 1);
        insert(row);
    }
    // Reduced rows are the dual coordinates: each pivot column is a unit
    // vector. Setting nonpivot input bits to zero selects a core assignment.
    work.charge(width * width);
    for (uint32_t j = width; j-- > 0;) {
        if (rows[j].is_zero())
            continue;
        for (uint32_t i = 0; i < j; ++i) {
            if (rows[i].bit_get(j))
                rows[i] ^= rows[j];
        }
    }
    std::vector<Mask> axes;
    for (uint32_t i = 0; i < width; ++i) {
        if (!rows[i].is_zero())
            axes.push_back(rows[i]);
    }
    return axes;
}

template <size_t N>
std::optional<PhaseReduction<N>> PhasePolynomial<N>::linear_reduction(uint32_t width, Work& work) {
    const auto& phase = signature(work);
    if (!phase.pairs.empty())
        return std::nullopt;
    PhaseReduction<N> result;
    work.charge(width);
    for (uint32_t i = 0; i < width; ++i) {
        const auto c = phase.linear[i];
        if (c) {
            result.phases.emplace_back(bit<N>(i), c);
            result.t_count += c & 1;
        }
    }
    result.core_width = static_cast<uint32_t>(result.t_count);
    return result;
}

template <size_t N>
PhaseReduction<N> PhasePolynomial<N>::synthesize(uint32_t width, Work& work) {
    if (auto linear = linear_reduction(width, work))
        return std::move(*linear);
    return synthesize(width, core_axes(width, work), work);
}

template <size_t N>
PhaseReduction<N> PhasePolynomial<N>::synthesize(uint32_t width, std::span<const Mask> axes,
                                                 Work& work) {
    const auto core_width = static_cast<uint32_t>(axes.size());

    // Replace each pivot coordinate x_j with the parity of its reduced row.
    // The other bits stay fixed. This change of coordinates is its own inverse:
    // applying it twice cancels all the nonpivot bits added to x_j.
    const auto change_basis = [&](Mask parity) {
        work.charge(core_width);
        auto converted = parity;
        for (auto row : axes) {
            const auto j = row.lowest_bit();
            if (parity.bit_get(j))
                converted ^= row ^ bit<N>(j);
        }
        return converted;
    };
    Signature remainder;
    std::set<Mask> projected;
    for (const auto& [parity, coefficient] : terms_) {
        remainder.add(change_basis(parity), coefficient, work);
        if (!(coefficient & 1))
            continue;
        work.charge(core_width);
        Mask core;
        for (uint32_t j = 0; j < core_width; ++j) {
            if (parity.bit_get(axes[j].lowest_bit()))
                core.bit_set(j, true);
        }
        // Each distinct odd column participates modulo two.
        if (!core.is_zero()) {
            const auto [it, inserted] = projected.insert(core);
            if (!inserted)
                projected.erase(it);
        }
    }
    std::vector<Mask> odd(projected.begin(), projected.end());
    if (odd.size() > core_width && odd.size() <= kMaxTohpeColumns) {
        const auto n = odd.size();
        const auto words = (core_width * (core_width + 1) / 2 + 63) / 64;
        // The projected columns already give an exact synthesis. If optimizing
        // them costs more than the remaining allowance, leave it for later regions.
        if (work.try_charge(n * n * n * (words + 1)))
            odd = tohpe(std::move(odd), core_width);
    }
    Terms out;
    for (auto column : odd) {
        Mask parity;
        work.charge(column.popcount() + 1);
        for (; !column.is_zero(); column.clear_lowest_bit())
            parity.bit_set(axes[column.lowest_bit()].lowest_bit(), true);
        add_term(out, parity, 1, work);
        remainder.add(parity, -1, work);
    }
    // A matching non-Clifford signature leaves only S, Z and CZ phases.
    for (uint32_t i = 0; i < width; ++i) {
        const auto c = remainder.linear[i];
        if (c & 1)
            throw std::logic_error("phase synthesis left a non-Clifford rotation");
        if (c)
            add_term(out, bit<N>(i), c, work);
    }
    for (const auto& [indices, pair] : remainder.pairs) {
        if (pair.weight != 2 || !pair.odd_columns.is_zero())
            throw std::logic_error("phase synthesis changed the non-Clifford signature");
        const auto [i, j] = indices;
        add_term(out, bit<N>(i), 2, work);
        add_term(out, bit<N>(j), 2, work);
        add_term(out, bit<N>(i) | bit<N>(j), -2, work);
    }
    PhaseReduction<N> result{core_width, 0, {}};
    for (const auto& [parity, coefficient] : out) {
        result.phases.emplace_back(change_basis(parity), coefficient);
        result.t_count += coefficient & 1;
    }
    return result;
}

template class PhasePolynomial<64>;
template class PhasePolynomial<128>;

}  // namespace clifft::phase_detail
