#include "clifft/optimizer/active_width_analysis.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <utility>

namespace clifft {

namespace detail {

PauliString pauli_body(const HirModule& hir, const HeisenbergOp& op) {
    PauliString result(hir.num_qubits);
    result.mut_x().xor_with(hir.destab_mask(op));
    result.mut_z().xor_with(hir.stab_mask(op));
    return result;
}

double dense_work_contribution(WidthEffect effect, uint32_t before, uint32_t after) {
    switch (effect) {
        case WidthEffect::RotationNeutral:
        case WidthEffect::RotationPromote:
        case WidthEffect::InstrumentActive:
        case WidthEffect::InstrumentActivate:
            return std::ldexp(1.0, static_cast<int>(after));
        case WidthEffect::MeasureActive:
            return std::ldexp(1.0, static_cast<int>(before));
        default:
            return 0.0;
    }
}

}  // namespace detail

namespace {

// The (x, z) mask pair is treated as one combined GF(2) vector of length
// 2 * domain, with x occupying [0, domain) and z occupying [domain, 2 *
// domain), domain = words_per_row * 64. Bits at or beyond num_qubits within
// each half are always zero -- PauliString and the generator storage both
// keep that padding clear -- so they are inert dead space at the top of
// each half and are never selected as a pivot.
uint32_t combined_domain(uint32_t words_per_row) {
    return words_per_row * 64;
}

bool combined_bit_get(MaskView x, MaskView z, uint32_t domain, uint32_t bit) {
    return bit < domain ? x.bit_get(bit) : z.bit_get(bit - domain);
}

uint32_t combined_lowest_bit(MaskView x, MaskView z, uint32_t domain) {
    const uint32_t x_bit = x.lowest_bit();
    if (x_bit < domain) {
        return x_bit;
    }
    return domain + z.lowest_bit();
}

}  // namespace

DormantSubspace::DormantSubspace(uint32_t num_qubits)
    : num_qubits_(num_qubits),
      words_per_row_((num_qubits + 63) / 64),
      dimension_(num_qubits),
      gen_x_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      gen_z_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      echelon_x_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      echelon_z_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      echelon_pivot_(num_qubits, 0),
      scratch_x_(words_per_row_, 0),
      scratch_z_(words_per_row_, 0) {
    // S starts as the span of Z on every qubit: generator q is Z_q.
    for (uint32_t q = 0; q < num_qubits; ++q) {
        row_z(q).bit_set(q, true);
    }
}

MaskView DormantSubspace::row_x(uint32_t index) const {
    return MaskView{std::span<const uint64_t>(
        gen_x_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
}

MaskView DormantSubspace::row_z(uint32_t index) const {
    return MaskView{std::span<const uint64_t>(
        gen_z_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
}

MutableMaskView DormantSubspace::row_x(uint32_t index) {
    return MutableMaskView{std::span<uint64_t>(
        gen_x_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
}

MutableMaskView DormantSubspace::row_z(uint32_t index) {
    return MutableMaskView{std::span<uint64_t>(
        gen_z_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
}

MutableMaskView DormantSubspace::echelon_row_x(uint32_t index) const {
    return MutableMaskView{std::span<uint64_t>(
        echelon_x_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
}

MutableMaskView DormantSubspace::echelon_row_z(uint32_t index) const {
    return MutableMaskView{std::span<uint64_t>(
        echelon_z_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
}

std::optional<uint32_t> DormantSubspace::find_anticommuting_generator(MaskView x,
                                                                      MaskView z) const {
    assert(x.num_words() == words_per_row_ && z.num_words() == words_per_row_ &&
           "Pauli body must share the subspace's word width");
    for (uint32_t i = 0; i < dimension_; ++i) {
        if (anti_commute(row_x(i), row_z(i), x, z)) {
            return i;
        }
    }
    return std::nullopt;
}

bool DormantSubspace::commutes_with_all(MaskView x, MaskView z) const {
    return !find_anticommuting_generator(x, z).has_value();
}

void DormantSubspace::intersect_with_pivot(MaskView x, MaskView z, uint32_t pivot) {
    assert(pivot < dimension_ && "pivot must be a live generator index");
    for (uint32_t i = 0; i < dimension_; ++i) {
        if (i == pivot) {
            continue;
        }
        if (anti_commute(row_x(i), row_z(i), x, z)) {
            row_x(i).xor_with(row_x(pivot));
            row_z(i).xor_with(row_z(pivot));
        }
    }
    const uint32_t last = dimension_ - 1;
    if (pivot != last) {
        std::ranges::copy(row_x(last).words, row_x(pivot).words.begin());
        std::ranges::copy(row_z(last).words, row_z(pivot).words.begin());
    }
    --dimension_;
    echelon_dirty_ = true;
}

void DormantSubspace::append_generator(MaskView x, MaskView z) {
    assert(dimension_ < num_qubits_ && "S is already Lagrangian; nothing can extend it");
    std::ranges::copy(x.words, row_x(dimension_).words.begin());
    std::ranges::copy(z.words, row_z(dimension_).words.begin());
    ++dimension_;
    echelon_dirty_ = true;
}

bool DormantSubspace::apply_rotation(MaskView x, MaskView z) {
    const std::optional<uint32_t> pivot = find_anticommuting_generator(x, z);
    if (!pivot.has_value()) {
        return false;
    }
    intersect_with_pivot(x, z, *pivot);
    return true;
}

DormantSubspace::MeasurementEffect DormantSubspace::apply_measurement(MaskView x, MaskView z) {
    const std::optional<uint32_t> pivot = find_anticommuting_generator(x, z);
    if (pivot.has_value()) {
        intersect_with_pivot(x, z, *pivot);
        append_generator(x, z);
        return MeasurementEffect::DormantRandom;
    }
    if (contains(x, z)) {
        return MeasurementEffect::Classical;
    }
    append_generator(x, z);
    return MeasurementEffect::Active;
}

std::vector<PauliString> DormantSubspace::generators() const {
    std::vector<PauliString> result;
    result.reserve(dimension_);
    for (uint32_t i = 0; i < dimension_; ++i) {
        PauliString generator(num_qubits_);
        generator.mut_x().xor_with(row_x(i));
        generator.mut_z().xor_with(row_z(i));
        result.push_back(std::move(generator));
    }
    return result;
}

std::optional<uint32_t> DormantSubspace::reduce_against_membership_cache(
    MutableMaskView work_x, MutableMaskView work_z) const {
    const uint32_t domain = combined_domain(words_per_row_);
    for (uint32_t row = 0; row < echelon_dimension_; ++row) {
        const uint32_t pivot = echelon_pivot_[row];
        if (combined_bit_get(work_x, work_z, domain, pivot)) {
            work_x.xor_with(echelon_row_x(row));
            work_z.xor_with(echelon_row_z(row));
        }
    }
    const uint32_t remainder = combined_lowest_bit(work_x, work_z, domain);
    if (remainder >= 2 * domain) {
        return std::nullopt;
    }
    return remainder;
}

void DormantSubspace::rebuild_membership_cache_if_dirty() const {
    if (!echelon_dirty_) {
        return;
    }
    echelon_dimension_ = 0;
    for (uint32_t i = 0; i < dimension_; ++i) {
        std::ranges::copy(row_x(i).words, echelon_row_x(echelon_dimension_).words.begin());
        std::ranges::copy(row_z(i).words, echelon_row_z(echelon_dimension_).words.begin());
        const std::optional<uint32_t> pivot = reduce_against_membership_cache(
            echelon_row_x(echelon_dimension_), echelon_row_z(echelon_dimension_));
        // The generator list is maintained as a basis by intersect_with_pivot
        // and append_generator, so inserting an already-independent vector
        // into the echelon form can never reduce it to zero.
        assert(pivot.has_value() && "generator list was not linearly independent");
        echelon_pivot_[echelon_dimension_] = pivot.value_or(0);
        ++echelon_dimension_;
    }
    echelon_dirty_ = false;
}

bool DormantSubspace::contains(MaskView x, MaskView z) const {
    assert(x.num_words() == words_per_row_ && z.num_words() == words_per_row_ &&
           "Pauli body must share the subspace's word width");
    rebuild_membership_cache_if_dirty();
    // Reduce a scratch copy; the cached echelon rows stay put for reuse by
    // later contains() calls.
    std::ranges::copy(x.words, scratch_x_.begin());
    std::ranges::copy(z.words, scratch_z_.begin());
    const std::optional<uint32_t> remainder =
        reduce_against_membership_cache(MutableMaskView{scratch_x_}, MutableMaskView{scratch_z_});
    return !remainder.has_value();
}

WidthTransition classify_and_apply(const HirModule& hir, const HeisenbergOp& op,
                                   DormantSubspace& subspace) {
    const uint32_t before = subspace.active_width();
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            if (subspace.apply_rotation(x, z)) {
                return WidthTransition{before, subspace.active_width(),
                                       WidthEffect::RotationPromote};
            }
            if (subspace.contains(x, z)) {
                return WidthTransition{before, before, WidthEffect::RotationStabilizer};
            }
            return WidthTransition{before, before, WidthEffect::RotationNeutral};
        }
        case OpType::MEASURE: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            switch (subspace.apply_measurement(x, z)) {
                case DormantSubspace::MeasurementEffect::DormantRandom:
                    return WidthTransition{before, before, WidthEffect::MeasureDormantRandom};
                case DormantSubspace::MeasurementEffect::Classical:
                    return WidthTransition{before, before, WidthEffect::MeasureClassical};
                case DormantSubspace::MeasurementEffect::Active:
                    return WidthTransition{before, subspace.active_width(),
                                           WidthEffect::MeasureActive};
            }
            assert(false && "unreachable DormantSubspace::MeasurementEffect");
            return WidthTransition{before, before, WidthEffect::None};
        }
        case OpType::INSTRUMENT: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            const InstrumentSite& site =
                hir.instrument_sites.at(static_cast<uint32_t>(op.instrument_site_idx()));

            if (!subspace.commutes_with_all(x, z)) {
                const bool traps = hir.neglect_instrument_damping ||
                                   site.probabilities.p_fire[0] == site.probabilities.p_fire[1];
                if (traps) {
                    return WidthTransition{before, before, WidthEffect::InstrumentDormantTrap};
                }
                [[maybe_unused]] const bool promoted = subspace.apply_rotation(x, z);
                assert(promoted && "instrument body must anticommute with S here");
                return WidthTransition{before, subspace.active_width(),
                                       WidthEffect::InstrumentActivate};
            }
            if (subspace.contains(x, z)) {
                return WidthTransition{before, before, WidthEffect::InstrumentClassical};
            }
            return WidthTransition{before, before, WidthEffect::InstrumentActive};
        }
        case OpType::CONDITIONAL_PAULI:
        case OpType::NOISE:
        case OpType::READOUT_NOISE:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::EXP_VAL:
        case OpType::NUM_OP_TYPES:
            return WidthTransition{before, before, WidthEffect::None};
    }
    assert(false && "unreachable OpType");
    return WidthTransition{before, before, WidthEffect::None};
}

ActiveWidthTrace analyze_active_width(const HirModule& hir) {
    ActiveWidthTrace trace;
    DormantSubspace subspace(hir.num_qubits);
    trace.initial_width = subspace.active_width();
    trace.peak_width = trace.initial_width;
    trace.transitions.reserve(hir.ops.size());

    for (const HeisenbergOp& op : hir.ops) {
        const WidthTransition transition = classify_and_apply(hir, op, subspace);
        trace.transitions.push_back(transition);
        trace.peak_width = std::max(trace.peak_width, transition.after);
    }

    trace.final_width =
        trace.transitions.empty() ? trace.initial_width : trace.transitions.back().after;
    return trace;
}

double estimate_dense_work(const ActiveWidthTrace& trace) {
    double work = 0.0;
    for (const WidthTransition& transition : trace.transitions) {
        work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
    return work;
}

}  // namespace clifft
