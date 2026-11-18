#include "clifft/optimizer/wide_phase_region.h"

#include "clifft/optimizer/pauli_axis.h"

#include <algorithm>
#include <map>

namespace clifft::phase_detail::wide {
namespace {

using optimizer_detail::body_bit;
using optimizer_detail::pivot_of;

PauliString product(const std::vector<PauliString>& axes, Mask mask, uint32_t width) {
    PauliString result(width);
    for (; !mask.is_zero(); mask.clear_lowest_bit()) {
        result.right_multiply(axes[mask.lowest_bit()].view());
    }
    assert(result.is_hermitian());
    return result;
}

struct Row {
    PauliString axis;
    Mask coordinates;
};

Mask coordinates(PauliString& body, const std::map<uint32_t, Row>& rows) {
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

std::optional<Rewrite> synthesize_region(const Polynomial& terms, const PhaseSignature& signature,
                                         const std::vector<PauliString>& axes, size_t end,
                                         uint32_t width, Budget& budget) {
    const auto basis = reduce_core(signature, axes.size(), budget);
    if (!basis)
        return std::nullopt;
    const auto synthesis = synthesize(terms, *basis, budget);
    Rewrite result{end, basis->core_width, 0, {}, {}, {}, false};
    std::vector<PauliString> transformed;
    for (auto parity : basis->parities) {
        budget.charge((parity.popcount() + 1) * ((width + 63) / 64));
        transformed.push_back(product(axes, parity, width));
    }
    for (const auto& [parity, coefficient] : synthesis) {
        budget.charge((parity.popcount() + 1) * ((width + 63) / 64));
        result.phases.emplace_back(product(transformed, parity, width), coefficient);
        result.t_count += coefficient & 1;
    }
    return result;
}

// A linear phase polynomial already gives the rotations in the collected basis.
std::optional<Rewrite> linear_rewrite(const PhaseSignature& signature,
                                      const std::vector<PauliString>& axes, size_t end,
                                      size_t words, Budget& budget) {
    if (!signature.pairs.empty())
        return std::nullopt;
    budget.charge(axes.size() * words);
    Rewrite result{end, 0, 0, {}, {}, {}, false};
    for (size_t i = 0; i < axes.size(); ++i) {
        const auto coefficient = signature.linear[i];
        if (!coefficient)
            continue;
        result.phases.emplace_back(axes[i], coefficient);
        result.t_count += coefficient & 1;
    }
    result.core_width = static_cast<uint32_t>(result.t_count);
    return result;
}

// Commuting rotation axes, their coordinates modulo fixed Paulis, and their phase polynomial.
struct Region {
    std::map<uint32_t, Row> rows;
    std::vector<PauliString> axes;
    KnownStabilizers required;
    Polynomial terms;
    std::optional<PhaseSignature> signature;

    const PhaseSignature& phase_signature(Budget& budget) {
        if (!signature) {
            signature.emplace();
            for (const auto& [parity, coefficient] : terms) {
                signature->add(parity, coefficient, budget);
            }
        }
        return *signature;
    }

    std::optional<Rewrite> synthesize_aligned(const HirModule& hir, size_t end,
                                              const KnownStabilizers& fixed,
                                              const KnownStabilizers& available,
                                              const optimizer_detail::CliffordAbsorption& frame,
                                              Budget& budget);
};

std::optional<Rewrite> Region::synthesize_aligned(const HirModule& hir, size_t end,
                                                  const KnownStabilizers& fixed,
                                                  const KnownStabilizers& available,
                                                  const optimizer_detail::CliffordAbsorption& frame,
                                                  Budget& budget) {
    const size_t words = (hir.num_qubits + 63) / 64;
    if (!hir.final_tableau) {
        return std::nullopt;
    }
    std::vector<PauliString> preferred;
    struct Change {
        Mask old_coordinates, new_coordinates;
    };
    std::map<uint32_t, Change> conversion;
    for (uint32_t q = 0; q < hir.num_qubits && preferred.size() < axes.size(); ++q) {
        budget.charge((hir.num_qubits + 4 * axes.size() + 1) * words);
        budget.charge(hir.num_qubits);
        auto axis = frame.read(hir.final_tableau->unsigned_z_preimage(q));
        axis.set_sign(false);
        if (!required.commutes(axis.view()) ||
            std::ranges::any_of(axes,
                                [&](const auto& g) { return !axis.view().commutes(g.view()); }) ||
            std::ranges::any_of(preferred,
                                [&](const auto& g) { return !axis.view().commutes(g.view()); }))
            continue;
        auto reduced = fixed.reduce_body(axis);
        auto old = coordinates(reduced, rows);
        if (!reduced.x().is_zero() || !reduced.z().is_zero() || old.is_zero())
            continue;
        auto next = bit(preferred.size());
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
    if (preferred.size() != axes.size()) {
        return std::nullopt;
    }
    Polynomial aligned_terms;
    PhaseSignature aligned_signature;
    for (const auto& [mask, coefficient] : terms) {
        budget.charge((hir.num_qubits + 3 * axes.size() + 1) * words);
        auto remaining = mask;
        Mask converted;
        for (const auto& [pivot, row] : conversion) {
            if (remaining.bit_get(pivot)) {
                remaining ^= row.old_coordinates;
                converted ^= row.new_coordinates;
            }
        }
        assert(remaining.is_zero());
        auto residual = product(axes, mask, hir.num_qubits);
        residual.right_multiply(product(preferred, converted, hir.num_qubits).view());
        const auto negative = available.eigenvalue(residual);
        if (!negative) {
            return std::nullopt;
        }
        const int signed_coefficient = *negative ? -coefficient : coefficient;
        budget.add(aligned_terms, converted, signed_coefficient);
        aligned_signature.add(converted, signed_coefficient, budget);
    }
    return synthesize_region(aligned_terms, aligned_signature, preferred, end, hir.num_qubits,
                             budget);
}

}  // namespace

std::optional<Rewrite> try_region(const HirModule& hir, size_t start, const KnownStabilizers& fixed,
                                  const optimizer_detail::CliffordAbsorption& frame,
                                  uint32_t variable_limit, Budget& budget,
                                  const KnownStabilizers* postselected,
                                  std::optional<size_t> final_observer_end) {
    std::optional<Rewrite> best;
    size_t best_rotation_count = 0;
    std::vector<size_t> rotations;
    std::vector<Observer> observers;
    try {
        KnownStabilizers available = fixed;
        Region region;
        auto consider = [&](Rewrite candidate) {
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
            auto candidate = synthesize_region(region.terms, region.phase_signature(budget),
                                               region.axes, end, hir.num_qubits, budget);
            if (!candidate)
                return;
            consider(std::move(*candidate));
            if (auto aligned =
                    region.synthesize_aligned(hir, end, fixed, available, frame, budget)) {
                consider(std::move(*aligned));
            }
        };
        size_t first_search_rotations = 0;
        size_t end = start;
        const size_t words = (hir.num_qubits + 63) / 64;
        for (size_t i = start; i < hir.ops.size(); ++i) {
            const auto& op = hir.ops[i];
            const OpType type = op.op_type();
            // Search the first wide prefix and the complete region. Intervening observers
            // can use a linear polynomial directly without another basis search.
            if (type != OpType::T_GATE && region.axes.size() > 64 && first_search_rotations == 0) {
                first_search_rotations = rotations.size();
                synthesize_prefix();
            }
            if (type == OpType::T_GATE) {
                budget.charge((hir.num_qubits + 3 * region.axes.size() + 1) * words);
                auto axis = frame.read(hir.mask_view(op), hir.num_qubits);
                auto reduced = fixed.reduce_body(axis);
                const bool entry_stabilizer = reduced.x().is_zero() && reduced.z().is_zero();
                if ((postselected && !postselected->commutes(axis.view())) ||
                    std::ranges::any_of(
                        region.axes,
                        [&](const auto& g) { return !axis.view().commutes(g.view()); }) ||
                    (!entry_stabilizer && !region.required.commutes(axis.view()))) {
                    break;
                }
                auto mask = coordinates(reduced, region.rows);
                if (!reduced.x().is_zero() || !reduced.z().is_zero()) {
                    if (region.axes.size() == variable_limit) {
                        break;
                    }
                    const auto next = bit(region.axes.size());
                    const auto pivot = pivot_of(reduced);
                    region.rows.emplace(pivot, Row{std::move(reduced), mask ^ next});
                    mask = next;
                    auto generator = axis;
                    generator.set_sign(false);
                    region.axes.push_back(std::move(generator));
                }
                auto residual = product(region.axes, mask, hir.num_qubits);
                auto unsigned_axis = axis;
                unsigned_axis.set_sign(false);
                residual.right_multiply(unsigned_axis.view());
                const auto negative = available.eigenvalue(residual);
                if (!negative) {
                    break;
                }
                if (!residual.x().is_zero() || !residual.z().is_zero()) {
                    residual.set_sign(residual.sign() ^ *negative);
                    region.required.insert(std::move(residual));
                }
                int coefficient = op.is_dagger() != axis.sign() ? -1 : 1;
                if (*negative) {
                    coefficient = -coefficient;
                }
                budget.add(region.terms, mask, coefficient);
                if (region.signature) {
                    region.signature->add(mask, coefficient, budget);
                }
                rotations.push_back(i);
            } else if (type == OpType::NOISE) {
                const auto& channels =
                    hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
                bool blocked = false;
                for (const auto& channel : channels) {
                    budget.charge((hir.num_qubits + region.axes.size() + 1) * words);
                    auto axis =
                        frame.read(hir.noise_channel_masks.at(channel.mask), hir.num_qubits);
                    if (channel.prob > 0 &&
                        (!region.required.commutes(axis.view()) ||
                         std::ranges::any_of(region.axes, [&](const PauliString& g) {
                             return !axis.view().commutes(g.view());
                         }))) {
                        blocked = true;
                        break;
                    }
                    if (channel.prob > 0) {
                        available.intersect(axis.view());
                    }
                }
                if (blocked) {
                    break;
                }
            } else if (type == OpType::MEASURE || type == OpType::CONDITIONAL_PAULI ||
                       type == OpType::EXP_VAL) {
                budget.charge((hir.num_qubits + 2 * region.axes.size() + 1) * words);
                const auto axis = frame.read(hir.mask_view(op), hir.num_qubits);
                if (!region.required.commutes(axis.view())) {
                    break;
                }
                Mask flip;
                for (size_t j = 0; j < region.axes.size(); ++j) {
                    if (!axis.view().commutes(region.axes[j].view())) {
                        flip.bit_set(static_cast<uint32_t>(j), true);
                    }
                }
                if (first_search_rotations != 0 && rotations.size() > first_search_rotations &&
                    (observers.empty() || observers.back().index < rotations.back())) {
                    if (auto linear = linear_rewrite(region.phase_signature(budget), region.axes,
                                                     rotations.back() + 1, words, budget)) {
                        consider(std::move(*linear));
                    }
                }
                const auto difference = pauli_derivative(region.terms, flip, budget);
                if (!difference) {
                    break;
                }
                auto pulled = axis;
                budget.charge((difference->parity.popcount() + 1) * words);
                const auto correction = product(region.axes, difference->parity, hir.num_qubits);
                pulled.right_multiply(correction.view());
                pulled.add_phase(static_cast<uint8_t>((-int(difference->constant) / 2) & 3));
                assert(pulled.is_hermitian());
                observers.push_back({i, std::move(pulled)});
                available.intersect(axis.view());
            } else if (type != OpType::DETECTOR && type != OpType::OBSERVABLE &&
                       type != OpType::READOUT_NOISE) {
                break;
            }
            end = i + 1;
        }
        if (region.axes.size() > 64 && rotations.size() != first_search_rotations) {
            synthesize_prefix();
        }
        if (region.axes.size() > 64 && final_observer_end && end >= *final_observer_end &&
            !observers.empty()) {
            const size_t odd_terms = static_cast<size_t>(std::ranges::count_if(
                region.terms, [](const auto& term) { return (term.second & 1) != 0; }));
            if ((best && best_rotation_count == rotations.size()) || odd_terms < rotations.size() ||
                !region.phase_signature(budget)
                     .clifford_kernel(region.axes.size(), budget)
                     .empty()) {
                return Rewrite{end, 0, 0, {}, std::move(rotations), std::move(observers), true};
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
}  // namespace clifft::phase_detail::wide
