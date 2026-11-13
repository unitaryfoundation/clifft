#pragma once

#include "clifft/sampling/branch_choice.h"
#include "clifft/sampling/measurement_branch.h"
#include "clifft/sampling/online_frame.h"
#include "clifft/util/xoshiro.h"

#include <optional>

namespace clifft::noncomp_detail {

using sampling::copy_pauli;
using sampling::LocalClifford;
using sampling::LocalPauli;
using sampling::OnlineFrame;

// Prepared rotation data lets OnlineState update the frame for Clifford angles
// and use dense kernels otherwise, without recomputing coefficients per shot.
struct OnlineRotation {
    sampling::PreparedPromotion coefficients;
    std::optional<CliffordRotation> clifford;
};

// Combines OnlineFrame with dense amplitudes for the active coordinates; all
// remaining coordinates represent |0>. OnlineWorker supplies physical operations
// and handles classical statuses, while this class updates the quantum state.
// A pending activation is a resource boundary, not a compilation boundary.
// Dispatch has already sampled any instrument outcome. It suspends before
// changing the frame or coefficients, retaining that outcome across growth.
class OnlineState {
  public:
    OnlineFrame frame;
    sampling::State state;
    uint32_t peak = 0;

    explicit OnlineState(uint32_t n) : frame(n), state(0), pending_axis_(n) {}

    void reset() noexcept {
        assert(!pending());
        frame.reset();
        state.reset();
        peak = 0;
    }

    [[nodiscard]] bool pending() const noexcept { return pending_kind_ != PendingKind::None; }

    // The caller has left ordinary dispatch and grown the coefficient buffer.
    void finish_activation() noexcept {
        assert(pending() && state.max_active_width() > state.active_width());
        const auto kind = pending_kind_;
        pending_kind_ = PendingKind::None;
        if (kind == PendingKind::Rotation)
            rotate(pending_axis_.view(), pending_rotation_);
        else
            no_fire(pending_axis_.view(), pending_zero_, pending_one_, pending_probability_);
        assert(!pending());
    }

    std::optional<uint32_t> dormant_x(PauliStringView p) const noexcept {
        return OnlineFrame::dormant_x(p, state.active_width());
    }

    sampling::ActivePauli active(PauliStringView p) const noexcept {
        return OnlineFrame::active(p, state.active_width());
    }

    sampling::MeasurementProbabilities populations(PauliStringView p) const noexcept {
        if (dormant_x(p))
            return {0.5, 0.5};
        const auto a = active(p);
        if (a.is_identity())
            return p.sign() ? sampling::MeasurementProbabilities{0, 1}
                            : sampling::MeasurementProbabilities{1, 0};
        auto values = sampling::measurement_probabilities(
            state, sampling::prepare_measurement_unchecked(a, state.active_width(), pivot(a)));
        if (p.sign())
            std::swap(values.zero, values.one);
        return values;
    }

    void rotate(PauliStringView p, OnlineRotation rotation) noexcept {
        const bool sign = p.sign();
        if (rotation.clifford) {
            frame.rotate_clifford(p, *rotation.clifford);
            return;
        }
        if (auto dormant = dormant_x(p)) {
            if (state.active_width() == state.max_active_width()) {
                copy_pauli(p, pending_axis_);
                pending_kind_ = PendingKind::Rotation;
                pending_rotation_ = rotation;
                return;
            }
            promote(p, *dormant);
            sampling::apply_promotion(state, rotation.coefficients, sign);
        } else {
            const auto a = active(p);
            if (!a.is_identity())
                sampling::apply_rotation(
                    state,
                    {sampling::prepare_pauli_unchecked(a, state.active_width()),
                     rotation.coefficients.cosine, rotation.coefficients.sine},
                    sign);
        }
    }

    void no_fire(PauliStringView p, double g, double e, double probability) noexcept {
        const bool sign = p.sign();
        if (g == e)
            return;
        if (auto dormant = dormant_x(p)) {
            if (state.active_width() == state.max_active_width()) {
                copy_pauli(p, pending_axis_);
                pending_kind_ = PendingKind::Filter;
                pending_zero_ = g;
                pending_one_ = e;
                pending_probability_ = probability;
                return;
            }
            promote(p, *dormant);
            if (sign)
                std::swap(g, e);
            sampling::apply_new_x_instrument_no_fire(state, g, e, probability);
        } else {
            if (sign)
                std::swap(g, e);
            const auto a = active(p);
            if (!a.is_identity())
                sampling::apply_instrument_no_fire(
                    state, sampling::prepare_pauli_unchecked(a, state.active_width()), g, e,
                    probability);
        }
    }

    bool measure(PauliStringView p, Xoshiro256PlusPlus& rng,
                 std::optional<bool> forced = {}) noexcept {
        const bool sign = p.sign();
        if (auto dormant = dormant_x(p)) {
            const bool branch =
                forced ? (*forced ^ sign)
                       : detail::choose_branch(
                             detail::BranchChoice::Measurement, std::array{0.5, 0.5},
                             [&](const auto& masses) noexcept {
                                 return static_cast<unsigned>(rng.next_double() < masses[1]);
                             }) != 0;
            frame.measure_dormant(p, *dormant, branch);
            return branch ^ sign;
        }
        const auto a = active(p);
        if (a.is_identity()) {
            assert(!forced || *forced == sign);
            return sign;
        }
        const uint32_t width = state.active_width();
        const uint32_t removed = pivot(a);
        const auto measurement = sampling::prepare_measurement_unchecked(a, width, removed);
        const auto probabilities = sampling::measurement_probabilities(state, measurement);
        const double total = probabilities.total();
        assert(total > 0 && is_finite_robust(total));
        bool branch;
        if (forced) {
            branch = *forced ^ sign;
        } else {
            const auto classification = sampling::classify_measurement_branch(probabilities);
            if (classification.kind == sampling::MeasurementBranchKind::Random) {
                branch = detail::choose_branch(detail::BranchChoice::Measurement,
                                               std::array{probabilities.zero, probabilities.one},
                                               [&](const auto& masses) noexcept {
                                                   return static_cast<unsigned>(
                                                       rng.next_double() * total < masses[1]);
                                               }) != 0;
            } else {
                branch = classification.kind == sampling::MeasurementBranchKind::DeterministicOne;
            }
        }
        assert(probabilities.for_branch(branch) > 0);
        sampling::collapse_measurement(state, measurement, branch,
                                       probabilities.for_branch(branch));

        frame.finish_active_measurement(a, removed, width, branch);
        return branch ^ sign;
    }

  private:
    enum class PendingKind { None, Rotation, Filter };
    PauliString pending_axis_;
    PendingKind pending_kind_ = PendingKind::None;
    OnlineRotation pending_rotation_;
    double pending_zero_ = 0;
    double pending_one_ = 0;
    double pending_probability_ = 0;

    static uint32_t pivot(sampling::ActivePauli p) noexcept {
        return OnlineFrame::measurement_pivot(p);
    }

    void promote(PauliStringView p, uint32_t dormant) noexcept {
        assert(state.active_width() < state.max_active_width());
        frame.promote(p, dormant, state.active_width());
        peak = std::max(peak, state.active_width() + 1);
    }
};

}  // namespace clifft::noncomp_detail
