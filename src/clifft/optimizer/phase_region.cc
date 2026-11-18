#include "clifft/optimizer/phase_region.h"

#include "clifft/optimizer/pauli_axis.h"

#include <algorithm>

namespace clifft::phase_detail {
using optimizer_detail::body_bit;
using optimizer_detail::CliffordAbsorption;
using optimizer_detail::pivot_of;

template <size_t N>
BitMask<N> PhaseRegion<N>::coordinates(PauliString& body) const {
    Mask result;
    for (const auto& [pivot, row] : rows) {
        if (body_bit(body, pivot)) {
            body.mut_x().xor_with(row.axis.x());
            body.mut_z().xor_with(row.axis.z());
            result ^= row.coordinates;
        }
    }
    return result;
}

template <size_t N>
bool PhaseRegion<N>::extend(const HirModule& hir, const KnownStabilizers& fixed,
                            const CliffordAbsorption& frame, const KnownStabilizers* postselected,
                            uint32_t variable_limit, Work& work) {
    const size_t words = (hir.num_qubits + 63) / 64;
    const auto& op = hir.ops[end];
    const auto type = op.op_type();
    if (type == OpType::T_GATE) {
        work.charge((hir.num_qubits + 3 * axes.size() + 1) * words);
        auto axis = frame.read(hir.mask_view(op), hir.num_qubits);
        auto reduced = fixed.reduce_body(axis);
        const bool entry_stabilizer = reduced.x().is_zero() && reduced.z().is_zero();
        if ((postselected && !postselected->commutes(axis.view())) ||
            std::ranges::any_of(axes,
                                [&](const auto& g) { return !axis.view().commutes(g.view()); }) ||
            (!entry_stabilizer && !required.commutes(axis.view()))) {
            return false;
        }
        auto mask = coordinates(reduced);
        if (!reduced.x().is_zero() || !reduced.z().is_zero()) {
            if (axes.size() == variable_limit) {
                capped = true;
                return false;
            }
            Mask next;
            next.bit_set(axes.size(), true);
            const auto pivot = pivot_of(reduced);
            rows.emplace(pivot, Row{std::move(reduced), mask ^ next});
            mask = next;
            auto generator = axis;
            generator.set_sign(false);
            axes.push_back(std::move(generator));
        }
        auto residual = axis_product(axes, mask, hir.num_qubits);
        auto unsigned_axis = axis;
        unsigned_axis.set_sign(false);
        residual.right_multiply(unsigned_axis.view());
        const auto negative = available.eigenvalue(residual);
        if (!negative) {
            return false;
        }
        if (!residual.x().is_zero() || !residual.z().is_zero()) {
            residual.set_sign(residual.sign() ^ *negative);
            required.insert(std::move(residual));
        }
        int coefficient = op.is_dagger() != axis.sign() ? -1 : 1;
        if (*negative) {
            coefficient = -coefficient;
        }
        polynomial.add(mask, coefficient, work);
        rotations.push_back(end);
    } else if (type == OpType::NOISE) {
        const auto& channels = hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
        for (const auto& channel : channels) {
            work.charge((hir.num_qubits + axes.size() + 1) * words);
            auto axis = frame.read(hir.noise_channel_masks.at(channel.mask), hir.num_qubits);
            if (channel.prob > 0 && (!required.commutes(axis.view()) ||
                                     std::ranges::any_of(axes, [&](const PauliString& g) {
                                         return !axis.view().commutes(g.view());
                                     }))) {
                return false;
            }
            if (channel.prob > 0) {
                available.intersect(axis.view());
            }
        }
    } else if (type == OpType::MEASURE || type == OpType::CONDITIONAL_PAULI ||
               type == OpType::EXP_VAL) {
        work.charge((hir.num_qubits + 2 * axes.size() + 1) * words);
        const auto axis = frame.read(hir.mask_view(op), hir.num_qubits);
        if (!required.commutes(axis.view())) {
            return false;
        }
        Mask flip;
        for (size_t j = 0; j < axes.size(); ++j) {
            if (!axis.view().commutes(axes[j].view())) {
                flip.bit_set(static_cast<uint32_t>(j), true);
            }
        }
        const auto difference = polynomial.derivative(flip, work);
        if (!difference) {
            return false;
        }
        auto pulled = axis;
        work.charge((difference->parity.popcount() + 1) * words);
        const auto correction = axis_product(axes, difference->parity, hir.num_qubits);
        pulled.right_multiply(correction.view());
        // U^dag M U carries the negative phase difference.
        pulled.add_phase(static_cast<uint8_t>((-int(difference->constant) / 2) & 3));
        assert(pulled.is_hermitian());
        observers.push_back({end, std::move(pulled)});
        available.intersect(axis.view());
    } else if (type != OpType::DETECTOR && type != OpType::OBSERVABLE &&
               type != OpType::READOUT_NOISE) {
        return false;
    }
    ++end;
    return true;
}

template struct PhaseRegion<64>;
template struct PhaseRegion<128>;

namespace {
using Mask = BitMask<128>;

WideRewrite materialize(const PhaseReduction<128>& reduction, const std::vector<PauliString>& axes,
                        size_t end, uint32_t width, WideBudget& budget) {
    WideRewrite result{end, reduction.core_width, reduction.t_count, {}, {}, {}, false};
    for (const auto& [mask, c] : reduction.phases) {
        budget.charge((mask.popcount() + 1) * ((width + 63) / 64));
        result.phases.emplace_back(axis_product(axes, mask, width), c);
    }
    return result;
}

std::optional<WideRewrite> synthesize_aligned(const PhaseRegion<128>& region, const HirModule& hir,
                                              size_t end, const KnownStabilizers& fixed,
                                              const CliffordAbsorption& frame, WideBudget& budget) {
    const size_t words = (hir.num_qubits + 63) / 64;
    if (!hir.final_tableau) {
        return std::nullopt;
    }
    std::vector<PauliString> preferred;
    struct Change {
        Mask old_coordinates, new_coordinates;
    };
    std::map<uint32_t, Change> conversion;
    for (uint32_t q = 0; q < hir.num_qubits && preferred.size() < region.axes.size(); ++q) {
        budget.charge((hir.num_qubits + 4 * region.axes.size() + 1) * words);
        budget.charge(hir.num_qubits);
        auto axis = frame.read(hir.final_tableau->unsigned_z_preimage(q));
        axis.set_sign(false);
        if (!region.required.commutes(axis.view()) ||
            std::ranges::any_of(region.axes,
                                [&](const auto& g) { return !axis.view().commutes(g.view()); }) ||
            std::ranges::any_of(preferred,
                                [&](const auto& g) { return !axis.view().commutes(g.view()); }))
            continue;
        auto reduced = fixed.reduce_body(axis);
        auto old = region.coordinates(reduced);
        if (!reduced.x().is_zero() || !reduced.z().is_zero() || old.is_zero())
            continue;
        Mask next;
        next.bit_set(preferred.size(), true);
        for (const auto& [pivot, row] : conversion) {
            if (old.bit_get(pivot)) {
                old ^= row.old_coordinates;
                next ^= row.new_coordinates;
            }
        }
        if (!old.is_zero()) {
            conversion.emplace(old.lowest_bit(), Change{old, next});
            preferred.push_back(std::move(axis));
        }
    }
    if (preferred.size() != region.axes.size()) {
        return std::nullopt;
    }
    PhasePolynomial<128> aligned;
    for (const auto& [mask, coefficient] : region.polynomial.terms()) {
        budget.charge((hir.num_qubits + 3 * region.axes.size() + 1) * words);
        auto remaining = mask;
        Mask converted;
        for (const auto& [pivot, row] : conversion) {
            if (remaining.bit_get(pivot)) {
                remaining ^= row.old_coordinates;
                converted ^= row.new_coordinates;
            }
        }
        assert(remaining.is_zero());
        auto residual = axis_product(region.axes, mask, hir.num_qubits);
        residual.right_multiply(axis_product(preferred, converted, hir.num_qubits).view());
        const auto negative = region.available.eigenvalue(residual);
        if (!negative) {
            return std::nullopt;
        }
        const int signed_coefficient = *negative ? -coefficient : coefficient;
        aligned.add(converted, signed_coefficient, budget);
    }
    auto reduction = aligned.synthesize(preferred.size(), budget);
    return materialize(reduction, preferred, end, hir.num_qubits, budget);
}

}  // namespace

std::optional<WideRewrite> try_wide_region(const HirModule& hir, size_t start,
                                           const KnownStabilizers& fixed,
                                           const optimizer_detail::CliffordAbsorption& frame,
                                           uint32_t variable_limit, WideBudget& budget,
                                           const KnownStabilizers* postselected,
                                           std::optional<size_t> final_observer_end) {
    std::optional<WideRewrite> best;
    size_t best_rotation_count = 0;
    PhaseRegion<128> region(start, fixed);
    auto& rotations = region.rotations;
    auto& observers = region.observers;
    try {
        auto consider = [&](WideRewrite candidate) {
            // Count the untouched rotations after the incumbent over the same input range.
            size_t t_count = rotations.size();
            size_t core_width = region.axes.size();
            if (best) {
                t_count -= best_rotation_count - best->t_count;
                core_width = best->core_width;
            }
            if (candidate.t_count < t_count ||
                (candidate.t_count == t_count && candidate.core_width < core_width)) {
                best_rotation_count = rotations.size();
                best = std::move(candidate);
            }
        };
        auto synthesize_prefix = [&] {
            const size_t end = rotations.back() + 1;
            auto reduction = region.polynomial.synthesize(region.axes.size(), budget);
            consider(materialize(reduction, region.axes, end, hir.num_qubits, budget));
            // Core rank is a lower bound on T count in every equivalent basis.
            if (reduction.t_count == reduction.core_width)
                return;
            if (auto aligned = synthesize_aligned(region, hir, end, fixed, frame, budget)) {
                consider(std::move(*aligned));
            }
        };
        size_t first_search_rotations = 0;
        while (region.end < hir.ops.size()) {
            const auto type = hir.ops[region.end].op_type();
            // Search the first wide prefix and the complete region. Intervening observers
            // can use a linear polynomial directly without another basis search.
            if (type != OpType::T_GATE && region.axes.size() > 64 && first_search_rotations == 0) {
                first_search_rotations = rotations.size();
                synthesize_prefix();
            }
            if ((type == OpType::MEASURE || type == OpType::CONDITIONAL_PAULI ||
                 type == OpType::EXP_VAL) &&
                first_search_rotations && rotations.size() > first_search_rotations &&
                (observers.empty() || observers.back().index < rotations.back())) {
                if (auto linear = region.polynomial.linear_reduction(region.axes.size(), budget)) {
                    consider(materialize(*linear, region.axes, rotations.back() + 1, hir.num_qubits,
                                         budget));
                }
            }
            if (!region.extend(hir, fixed, frame, postselected, variable_limit, budget))
                break;
        }
        if (region.axes.size() > 64 && rotations.size() != first_search_rotations) {
            synthesize_prefix();
        }
        if (region.axes.size() > 64 && final_observer_end && region.end >= *final_observer_end &&
            !observers.empty()) {
            if ((best && best_rotation_count == rotations.size()) ||
                region.polynomial.odd_terms() < rotations.size() ||
                region.polynomial.core_axes(region.axes.size(), budget).size() <
                    region.axes.size()) {
                return WideRewrite{region.end,           0,   0, {}, std::move(rotations),
                                   std::move(observers), true};
            }
        }
    } catch (const SearchLimit&) {
        // A completed prefix is still usable when an extension exhausts the budget.
    }
    if (best) {
        // Both lists grow in source order. Keep only the entries covered by this rewrite.
        rotations.resize(best_rotation_count);
        while (!observers.empty() && observers.back().index >= best->end) {
            observers.pop_back();
        }
        best->rotations = std::move(rotations);
        best->observers = std::move(observers);
    }
    return best;
}
}  // namespace clifft::phase_detail
