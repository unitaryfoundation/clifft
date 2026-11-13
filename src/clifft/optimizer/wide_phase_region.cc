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

bool is_beneficial(const Rewrite& candidate, size_t input_t_count, size_t support_width) {
    return candidate.t_count < input_t_count ||
           (candidate.t_count == input_t_count && candidate.core_width < support_width);
}

bool is_better(const Rewrite& candidate, const Rewrite& incumbent) {
    return candidate.t_count < incumbent.t_count ||
           (candidate.t_count == incumbent.t_count && candidate.core_width < incumbent.core_width);
}

Rewrite synthesize_region(const Polynomial& terms, Polynomial polynomial,
                          const std::vector<PauliString>& axes, size_t end, uint32_t width,
                          Budget& budget) {
    auto basis = reduce_core(polynomial, axes.size(), budget);
    const auto synthesis = synthesize(terms, polynomial, basis, budget);
    Rewrite result{end, basis.core_width, 0, {}};
    std::vector<PauliString> transformed;
    for (auto parity : basis.parities) {
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

// The commuting prefix of a run of T rotations, expressed as parity terms over
// its independent axes modulo the fixed stabilizers.
struct Collection {
    std::map<uint32_t, Row> rows;
    std::vector<PauliString> axes;
    KnownStabilizers required;
    Polynomial terms;
    size_t end = 0;
};

// next(index) returns rotation `index` or no value at the end of the run. The
// prefix stops before a rotation that anticommutes with an earlier axis or a
// stabilizer relation already used, or whose relation to the earlier axes is
// not fixed. No value means the prefix needs more than variable_limit axes.
template <typename Next>
std::optional<Collection> collect(uint32_t num_qubits, size_t start, Next&& next,
                                  const KnownStabilizers& fixed, uint32_t variable_limit,
                                  Budget& budget) {
    Collection out;
    out.end = start;
    const size_t words = (num_qubits + 63) / 64;
    for (;; ++out.end) {
        const std::optional<AxisRotation> rotation = next(out.end);
        if (!rotation) {
            break;
        }
        budget.charge((num_qubits + 3 * out.axes.size() + 1) * words);
        const PauliString& axis = rotation->axis;
        if (!out.required.commutes(axis.view()) ||
            std::ranges::any_of(out.axes,
                                [&](const auto& g) { return !axis.view().commutes(g.view()); })) {
            break;
        }
        auto reduced = fixed.reduce_body(axis);
        auto mask = coordinates(reduced, out.rows);
        if (!reduced.x().is_zero() || !reduced.z().is_zero()) {
            if (out.axes.size() == variable_limit) {
                return std::nullopt;
            }
            const auto next_bit = bit(out.axes.size());
            const auto pivot = pivot_of(reduced);
            out.rows.emplace(pivot, Row{std::move(reduced), mask ^ next_bit});
            mask = next_bit;
            auto generator = axis;
            generator.set_sign(false);
            out.axes.push_back(std::move(generator));
        }
        auto residual = product(out.axes, mask, num_qubits);
        auto unsigned_axis = axis;
        unsigned_axis.set_sign(false);
        residual.right_multiply(unsigned_axis.view());
        const auto negative = fixed.eigenvalue(residual);
        if (!negative) {
            break;
        }
        residual.set_sign(residual.sign() ^ *negative);
        out.required.insert(std::move(residual));
        int coefficient = rotation->dagger != axis.sign() ? -1 : 1;
        if (*negative)
            coefficient = -coefficient;
        budget.add(out.terms, mask, coefficient);
    }
    return out;
}

}  // namespace

std::optional<Rewrite> reduce_rotations(uint32_t num_qubits,
                                        std::span<const AxisRotation> rotations,
                                        const KnownStabilizers& fixed, uint32_t variable_limit,
                                        Budget& budget) {
    try {
        const auto region = collect(
            num_qubits, 0,
            [&](size_t index) -> std::optional<AxisRotation> {
                return index < rotations.size() ? std::optional(rotations[index]) : std::nullopt;
            },
            fixed, variable_limit, budget);
        if (!region || !core_fits(region->terms, region->axes.size(), budget)) {
            return std::nullopt;
        }
        Polynomial polynomial;
        for (const auto& [mask, coefficient] : region->terms) {
            add_parity(polynomial, mask, coefficient, budget);
        }
        return synthesize_region(region->terms, std::move(polynomial), region->axes, region->end,
                                 num_qubits, budget);
    } catch (const SearchLimit&) {
        return std::nullopt;
    }
}

std::optional<Rewrite> try_region(const HirModule& hir, size_t start, const KnownStabilizers& fixed,
                                  const optimizer_detail::CliffordAbsorption& frame,
                                  uint32_t variable_limit, Budget& budget) {
    std::optional<Rewrite> best;
    try {
        auto region = collect(
            hir.num_qubits, start,
            [&](size_t index) -> std::optional<AxisRotation> {
                if (index >= hir.ops.size() || hir.ops[index].op_type() != OpType::T_GATE) {
                    return std::nullopt;
                }
                return AxisRotation{frame.read(hir.mask_view(hir.ops[index]), hir.num_qubits),
                                    hir.ops[index].is_dagger()};
            },
            fixed, variable_limit, budget);
        if (!region) {
            return std::nullopt;
        }
        auto& [rows, axes, required, terms, end] = *region;
        Polynomial polynomial;
        const size_t words = (hir.num_qubits + 63) / 64;
        // A narrow prefix must retain the ordinary collector's ability to
        // pull observers back from later in the region.
        if (axes.size() <= 64)
            return std::nullopt;
        if (!core_fits(terms, axes.size(), budget))
            return std::nullopt;
        for (const auto& [mask, coefficient] : terms) {
            add_parity(polynomial, mask, coefficient, budget);
        }
        best = synthesize_region(terms, polynomial, axes, end, hir.num_qubits, budget);
        if (!is_beneficial(*best, end - start, axes.size())) {
            best.reset();
        }
        if (!hir.final_tableau)
            return best;

        // Preserve the collected basis as a valid candidate. The alternative
        // uses output coordinates only when they span the same fixed subspace.
        budget.charge(2 * size_t(hir.num_qubits) * hir.num_qubits * words);
        const auto final_inverse = hir.final_tableau->inverse();
        std::vector<PauliString> preferred;
        struct Change {
            Mask old_coordinates, new_coordinates;
        };
        std::map<uint32_t, Change> conversion;
        for (uint32_t q = 0; q < hir.num_qubits && preferred.size() < axes.size(); ++q) {
            budget.charge((hir.num_qubits + 4 * axes.size() + 1) * words);
            PauliString z(hir.num_qubits);
            z.set_pauli(q, false, true);
            auto axis = frame.read(final_inverse.apply(z.view()));
            axis.set_sign(false);
            if (!required.commutes(axis.view()) ||
                std::ranges::any_of(
                    axes, [&](const auto& g) { return !axis.view().commutes(g.view()); }) ||
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
        if (preferred.size() != axes.size())
            return best;
        Polynomial aligned_terms, aligned_polynomial;
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
            const auto negative = fixed.eigenvalue(residual);
            if (!negative)
                return best;
            const int signed_coefficient = *negative ? -coefficient : coefficient;
            budget.add(aligned_terms, converted, signed_coefficient);
            add_parity(aligned_polynomial, converted, signed_coefficient, budget);
        }
        auto aligned = synthesize_region(aligned_terms, std::move(aligned_polynomial), preferred,
                                         end, hir.num_qubits, budget);
        if (is_beneficial(aligned, end - start, axes.size()) &&
            (!best || is_better(aligned, *best))) {
            best = std::move(aligned);
        }
    } catch (const SearchLimit&) {
        // The caller can still use a completed candidate or the ordinary path.
    }
    return best;
}
}  // namespace clifft::phase_detail::wide
