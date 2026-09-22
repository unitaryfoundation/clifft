#include "clifft/optimizer/active_width_analysis.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <optional>
#include <utility>

namespace clifft {

namespace detail {

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

// Combined vectors place X bits before Z bits, with each half word-aligned.
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
      rows_x_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      rows_z_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      pivot_(num_qubits, 0),
      anticommute_flags_(num_qubits, 0),
      scratch_x_(words_per_row_, 0),
      scratch_z_(words_per_row_, 0) {
    // Z on each qubit gives the initial reduced basis for the all-zero state.
    const uint32_t domain = combined_domain(words_per_row_);
    for (uint32_t q = 0; q < num_qubits; ++q) {
        row_z(q).bit_set(q, true);
        pivot_[q] = domain + q;
    }
}

bool DormantSubspace::commutes_with_all(MaskView x, MaskView z) const {
    assert(x.num_words() == words_per_row_ && z.num_words() == words_per_row_ &&
           "Pauli body must share the subspace's word width");
    // Hoist the single-word case out of the row scan to avoid rebuilding
    // mask views and entering a generic word loop for every row.
    if (words_per_row_ == 1) {
        for (uint32_t i = 0; i < dimension_; ++i) {
            if ((std::popcount((rows_x_[i] & z.words[0]) ^ (rows_z_[i] & x.words[0])) & 1) != 0) {
                return false;
            }
        }
        return true;
    }
    for (uint32_t i = 0; i < dimension_; ++i) {
        if (anti_commute(row_x(i), row_z(i), x, z)) {
            return false;
        }
    }
    return true;
}

void DormantSubspace::reduce_into_scratch(MaskView x, MaskView z) const {
    std::ranges::copy(x.words, scratch_x_.begin());
    std::ranges::copy(z.words, scratch_z_.begin());
    const uint32_t domain = combined_domain(words_per_row_);
    MutableMaskView work_x{scratch_x_};
    MutableMaskView work_z{scratch_z_};
    // Each row is zero at the other pivots, so reducing in any order cannot
    // reintroduce a pivot bit that was already cleared.
    for (uint32_t i = 0; i < dimension_; ++i) {
        if (combined_bit_get(work_x, work_z, domain, pivot_[i])) {
            work_x.xor_with(row_x(i));
            work_z.xor_with(row_z(i));
        }
    }
}

bool DormantSubspace::intersect(MaskView x, MaskView z) {
    std::optional<uint32_t> pivot_row;
    // Separate scans keep the word-width branch out of the per-row loop.
    if (words_per_row_ == 1) {
        for (uint32_t i = 0; i < dimension_; ++i) {
            const bool anticommutes =
                (std::popcount((rows_x_[i] & z.words[0]) ^ (rows_z_[i] & x.words[0])) & 1) != 0;
            anticommute_flags_[i] = anticommutes ? 1 : 0;
            if (anticommutes && (!pivot_row.has_value() || pivot_[i] > pivot_[*pivot_row])) {
                pivot_row = i;
            }
        }
    } else {
        for (uint32_t i = 0; i < dimension_; ++i) {
            const bool anticommutes = anti_commute(row_x(i), row_z(i), x, z);
            anticommute_flags_[i] = anticommutes ? 1 : 0;
            if (anticommutes && (!pivot_row.has_value() || pivot_[i] > pivot_[*pivot_row])) {
                pivot_row = i;
            }
        }
    }
    if (!pivot_row.has_value()) {
        return false;
    }
    // Using the largest pivot preserves the leading bits of all remaining
    // rows when we cancel their anticommutation with (x, z).
    const uint32_t k = *pivot_row;
    for (uint32_t i = 0; i < dimension_; ++i) {
        if (i != k && anticommute_flags_[i] != 0) {
            row_x(i).xor_with(row_x(k));
            row_z(i).xor_with(row_z(k));
        }
    }
    const uint32_t last = dimension_ - 1;
    if (k != last) {
        std::ranges::copy(row_x(last).words, row_x(k).words.begin());
        std::ranges::copy(row_z(last).words, row_z(k).words.begin());
        pivot_[k] = pivot_[last];
    }
    --dimension_;
    return true;
}

void DormantSubspace::insert_reduced(MaskView r_x, MaskView r_z) {
    assert(dimension_ < num_qubits_ && "S is already Lagrangian; nothing can extend it");
    const uint32_t domain = combined_domain(words_per_row_);
    const uint32_t b = combined_lowest_bit(r_x, r_z, domain);
    assert(b < 2 * domain && "insert_reduced requires a nonzero vector");
    for (uint32_t i = 0; i < dimension_; ++i) {
        if (combined_bit_get(row_x(i), row_z(i), domain, b)) {
            row_x(i).xor_with(r_x);
            row_z(i).xor_with(r_z);
        }
    }
    std::ranges::copy(r_x.words, row_x(dimension_).words.begin());
    std::ranges::copy(r_z.words, row_z(dimension_).words.begin());
    pivot_[dimension_] = b;
    ++dimension_;
}

bool DormantSubspace::apply_rotation(MaskView x, MaskView z) {
    // intersect() also checks commutation, avoiding a separate scan.
    return intersect(x, z);
}

DormantSubspace::MeasurementEffect DormantSubspace::apply_measurement(MaskView x, MaskView z) {
    if (intersect(x, z)) {
        // The body anticommuted with the old S, so it is independent of the
        // remaining generators and its reduced remainder is nonzero.
        reduce_into_scratch(x, z);
        insert_reduced(MaskView{scratch_x_}, MaskView{scratch_z_});
        return MeasurementEffect::DormantRandom;
    }
    reduce_into_scratch(x, z);
    if (MaskView{scratch_x_}.is_zero() && MaskView{scratch_z_}.is_zero()) {
        return MeasurementEffect::Classical;
    }
    insert_reduced(MaskView{scratch_x_}, MaskView{scratch_z_});
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

bool DormantSubspace::contains(MaskView x, MaskView z) const {
    assert(x.num_words() == words_per_row_ && z.num_words() == words_per_row_ &&
           "Pauli body must share the subspace's word width");
    reduce_into_scratch(x, z);
    return MaskView{scratch_x_}.is_zero() && MaskView{scratch_z_}.is_zero();
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
