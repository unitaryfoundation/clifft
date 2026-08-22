#include "clifft/sampling/hip/executable.h"

#include "clifft/sampling/kernels.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace clifft::sampling::hip {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

void require_uint32_size(size_t size, const char* storage) {
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error(std::string("HIP executable ") + storage + " exceeds uint32 range");
    }
}

void flatten_pauli(detail::Action& action, const PreparedPauli& pauli) {
    action.phase_real = static_cast<int8_t>(pauli.even_phase.real());
    action.phase_imag = static_cast<int8_t>(pauli.even_phase.imag());
    action.x = pauli.x;
    action.z = pauli.z;
}

}  // namespace

Executable::Executable(const SamplingPlan& plan)
    : initial_active_width_(plan.initial_active_width),
      peak_active_width_(plan.peak_active_width),
      num_symbols_(static_cast<uint32_t>(plan.symbols.size())),
      num_visible_records_(plan.num_visible_records),
      num_hidden_records_(plan.num_hidden_records),
      num_detectors_(plan.num_detectors),
      num_observables_(plan.num_observables),
      num_exp_vals_(plan.num_exp_vals),
      has_postselection_(plan.has_postselection) {
    plan.validate();
    require_uint32_size(plan.symbols.size(), "symbol storage");
    require_uint32_size(plan.actions.size(), "action storage");
    if (plan.peak_active_width > kThreadPerShotMaxActiveWidth) {
        throw std::invalid_argument(
            "HIP thread-per-shot execution supports peak active width at "
            "most " +
            std::to_string(kThreadPerShotMaxActiveWidth));
    }
    if (plan.num_instrument_sites != 0 || !plan.instrument_distributions.empty()) {
        throw std::invalid_argument("HIP execution does not support transition instruments");
    }

    std::vector<bool> bound_presampled(plan.symbols.size(), false);
    noise_sites_.reserve(plan.presampled_noise_sites.size());
    for (const PresampledNoiseSite& site : plan.presampled_noise_sites) {
        require_uint32_size(noise_outcomes_.size(), "noise outcome storage");
        const uint32_t begin = static_cast<uint32_t>(noise_outcomes_.size());
        double cumulative_probability = 0.0;
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            cumulative_probability += outcome.probability;
            noise_outcomes_.push_back({index(outcome.symbol), 0, cumulative_probability});
            bound_presampled[index(outcome.symbol)] = true;
        }
        require_uint32_size(noise_outcomes_.size() - begin, "noise site outcome storage");
        noise_sites_.push_back(
            {begin, static_cast<uint32_t>(noise_outcomes_.size()) - begin, cumulative_probability});
    }
    for (size_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
        if (plan.symbols[symbol].kind == SymbolKind::Presampled && !bound_presampled[symbol]) {
            throw std::invalid_argument(
                "HIP execution requires a distribution for every presampled symbol");
        }
    }

    actions_.reserve(plan.actions.size());
    expressions_.reserve(plan.actions.size());
    for (const PlannedAction& action : plan.actions) {
        actions_.push_back(lower_action(action));
    }
}

uint32_t Executable::append_expression(const AffineBool& expression) {
    require_uint32_size(expressions_.size(), "expression storage");
    require_uint32_size(expression_terms_.size(), "expression term storage");
    if (expression.terms().size() >
        std::numeric_limits<uint32_t>::max() - expression_terms_.size()) {
        throw std::length_error("HIP executable expression term storage exceeds uint32 range");
    }
    const uint32_t expression_index = static_cast<uint32_t>(expressions_.size());
    const uint32_t term_begin = static_cast<uint32_t>(expression_terms_.size());
    for (SymbolId term : expression.terms()) {
        expression_terms_.push_back(index(term));
    }
    expressions_.push_back({term_begin,
                            static_cast<uint32_t>(expression.terms().size()),
                            static_cast<uint8_t>(expression.constant()),
                            {}});
    return expression_index;
}

detail::Action Executable::lower_action(const PlannedAction& planned) {
    return std::visit(
        [&](const auto& typed) -> detail::Action {
            using T = std::decay_t<decltype(typed)>;
            detail::Action action;
            action.active_before = planned.active_before;
            if constexpr (std::is_same_v<T, RotateActivePauli>) {
                const PreparedRotation rotation =
                    prepare_rotation(typed.pauli, planned.active_before, typed.half_turns);
                action.tag = detail::ActionTag::RotateActivePauli;
                action.expression = append_expression(typed.sign);
                flatten_pauli(action, rotation.pauli);
                action.pair_stride_or_z_without_pivot = rotation.pauli.pairing_bit;
                action.value0 = rotation.cosine;
                action.value1 = rotation.sine;
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                const PreparedPromotion promotion = prepare_promotion(typed.half_turns);
                action.tag = detail::ActionTag::PromoteDormantRotation;
                action.expression = append_expression(typed.sign);
                action.value0 = promotion.cosine;
                action.value1 = promotion.sine;
            } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                const PreparedMeasurement measurement =
                    prepare_measurement(typed.pauli, planned.active_before, typed.active_pivot);
                action.tag = detail::ActionTag::MeasureActivePauli;
                action.expression = append_expression(typed.outcome);
                flatten_pauli(action, measurement.pauli);
                action.index0 = index(typed.branch);
                action.index1 = index(typed.record);
                action.index2 = measurement.pivot;
                action.pair_stride_or_z_without_pivot = measurement.z_without_pivot;
            } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                action.tag = detail::ActionTag::MeasureDormantRandom;
                action.expression = append_expression(typed.outcome);
                action.index0 = index(typed.branch);
                action.index1 = index(typed.record);
            } else if constexpr (std::is_same_v<T, RecordClassical>) {
                action.tag = detail::ActionTag::RecordClassical;
                action.expression = append_expression(typed.outcome);
                action.index0 = index(typed.record);
            } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                action.tag = detail::ActionTag::DefineSymbol;
                action.expression = append_expression(typed.value);
                action.index0 = index(typed.symbol);
            } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                action.tag = detail::ActionTag::ApplyReadoutNoise;
                action.expression = append_expression(typed.source);
                action.index0 = index(typed.flip);
                action.index1 = index(typed.record);
                action.value0 = typed.prob_zero_to_one;
                action.value1 = typed.prob_one_to_zero;
            } else if constexpr (std::is_same_v<T, WriteDetector>) {
                action.tag = detail::ActionTag::WriteDetector;
                action.flags = typed.postselected ? detail::kPostselected : 0;
                action.expression = append_expression(typed.outcome);
                action.index0 = index(typed.detector);
            } else if constexpr (std::is_same_v<T, WriteObservable>) {
                action.tag = detail::ActionTag::WriteObservable;
                action.expression = append_expression(typed.outcome);
                action.index0 = index(typed.observable);
            } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                action.tag = detail::ActionTag::WriteExpectationValue;
                action.expression = append_expression(typed.sign);
                action.index0 = index(typed.exp_val);
                if (!typed.active_projection.has_value()) {
                    action.flags = detail::kAbsentActiveProjection;
                } else {
                    flatten_pauli(action,
                                  prepare_pauli(*typed.active_projection, planned.active_before));
                }
            } else if constexpr (std::is_same_v<T, ApplyInstrument> ||
                                 std::is_same_v<T, InstrumentBoundary>) {
                throw std::invalid_argument(
                    "HIP execution does not support transition instruments");
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
            }
            return action;
        },
        planned.action);
}

}  // namespace clifft::sampling::hip
