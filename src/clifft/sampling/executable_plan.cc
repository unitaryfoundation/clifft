#include "clifft/sampling/executable_plan.h"

#include "clifft/util/noise_sampling.h"
#include "clifft/util/runtime_isa.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace clifft::sampling {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

}  // namespace

ExecutablePlan::ExecutablePlan(const SamplingPlan& plan)
    : num_qubits_(plan.num_qubits),
      initial_active_width_(plan.initial_active_width),
      max_active_width_(plan.max_active_width),
      num_visible_records_(plan.num_visible_records),
      num_hidden_records_(plan.num_hidden_records),
      num_detectors_(plan.num_detectors),
      num_observables_(plan.num_observables),
      num_exp_vals_(plan.num_exp_vals),
      has_postselection_(plan.has_postselection),
      global_weight_(plan.global_weight),
      final_tableau_(plan.final_tableau),
      instrument_distributions_(plan.instrument_distributions) {
    plan.validate();
    const internal::RuntimeIsa runtime_isa = internal::runtime_isa();
    internal::validate_runtime_isa(runtime_isa);
    if (plan.symbols.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable symbol count exceeds uint32 range");
    }
    num_symbols_ = static_cast<uint32_t>(plan.symbols.size());

    size_t num_expression_terms = 0;
    for (const PlannedAction& planned : plan.actions) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli> ||
                              std::is_same_v<T, PromoteDormantRotation>) {
                    num_expression_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                                     std::is_same_v<T, MeasureDormantRandom>) {
                    // The current measurement branch is stored separately so
                    // replay can solve for it from the requested record.
                    num_expression_terms += typed.outcome.terms().size() - 1;
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    num_expression_terms += typed.outcome.terms().size();
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    num_expression_terms += typed.value.terms().size();
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    num_expression_terms += typed.source.terms().size();
                } else if constexpr (std::is_same_v<T, WriteDetector> ||
                                     std::is_same_v<T, WriteObservable>) {
                    num_expression_terms += typed.outcome.terms().size();
                } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                    num_expression_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                    num_expression_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    // Boundaries have no affine payload.
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);
    }
    if (num_expression_terms > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression storage exceeds uint32 range");
    }
    // Retain action-order terms only long enough to build the reverse dependency map.
    std::vector<uint32_t> expression_terms;
    expression_terms.reserve(num_expression_terms);
    std::vector<uint32_t> expression_term_begins;
    expression_term_begins.reserve(plan.actions.size());
    expression_register_constants_.reserve(plan.actions.size());

    auto prepare_expression = [&](const AffineBool& expression) {
        if (expression_register_constants_.size() >= std::numeric_limits<uint32_t>::max()) {
            throw std::length_error("sampling executable expression count exceeds uint32 range");
        }
        const uint32_t register_id = static_cast<uint32_t>(expression_register_constants_.size());
        expression_term_begins.push_back(static_cast<uint32_t>(expression_terms.size()));
        expression_register_constants_.push_back(static_cast<uint8_t>(expression.constant()));
        for (SymbolId term : expression.terms()) {
            expression_terms.push_back(index(term));
        }
        return PreparedExpression{register_id};
    };
    auto prepare_measurement_correction = [&](const AffineBool& outcome, uint32_t branch) {
        if (expression_register_constants_.size() >= std::numeric_limits<uint32_t>::max()) {
            throw std::length_error("sampling executable expression count exceeds uint32 range");
        }
        const uint32_t register_id = static_cast<uint32_t>(expression_register_constants_.size());
        const uint32_t begin = static_cast<uint32_t>(expression_terms.size());
        expression_term_begins.push_back(begin);
        expression_register_constants_.push_back(static_cast<uint8_t>(outcome.constant()));
        for (SymbolId term : outcome.terms()) {
            if (index(term) != branch) {
                expression_terms.push_back(index(term));
            }
        }
        assert(expression_terms.size() == static_cast<size_t>(begin) + outcome.terms().size() - 1 &&
               "validated measurement outcome must contain its branch exactly once");
        return PreparedExpression{register_id};
    };

    actions_.reserve(plan.actions.size());
    instrument_resume_offsets_.assign(plan.num_instrument_sites,
                                      std::numeric_limits<uint32_t>::max());
    presampled_symbols_.reserve(plan.symbols.size());
    std::vector<bool> bound_presampled(plan.symbols.size(), false);
    noise_sites_.reserve(plan.presampled_noise_sites.size());
    noise_hazards_.reserve(plan.presampled_noise_sites.size());
    double cumulative_hazard = 0.0;
    for (const PresampledNoiseSite& site : plan.presampled_noise_sites) {
        const uint32_t begin = static_cast<uint32_t>(noise_outcomes_.size());
        double cumulative_probability = 0.0;
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            cumulative_probability += outcome.probability;
            noise_outcomes_.push_back({index(outcome.symbol), cumulative_probability});
            bound_presampled[index(outcome.symbol)] = true;
        }
        noise_sites_.push_back(
            {begin, static_cast<uint32_t>(noise_outcomes_.size()) - begin, cumulative_probability});
        cumulative_hazard += bernoulli_hazard(cumulative_probability);
        noise_hazards_.push_back(cumulative_hazard);
    }
    for (uint32_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
        if (plan.symbols[symbol].kind == SymbolKind::Presampled) {
            presampled_symbols_.push_back(symbol);
            if (!bound_presampled[symbol]) {
                unbound_presampled_symbols_.push_back(symbol);
            }
        }
    }

    std::vector<uint32_t> boundary_noise_starts;
    boundary_noise_starts.reserve(plan.num_instrument_sites);
    for (const PlannedAction& planned : plan.actions) {
        if (const auto* boundary = std::get_if<InstrumentBoundary>(&planned.action)) {
            boundary_noise_starts.push_back(boundary->next_noise_site);
        }
    }
    initial_noise_end_ =
        boundary_noise_starts.empty() ? plan.num_noise_sites : boundary_noise_starts.front();

    size_t boundary_index = 0;
    auto lower_action = [&](const PlannedAction& planned) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli>) {
                    actions_.emplace_back(ExecuteRotation{
                        prepare_rotation(typed.pauli, planned.active_before, typed.half_turns),
                        prepare_expression(typed.sign)});
                } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                    actions_.emplace_back(ExecutePromotion{prepare_promotion(typed.half_turns),
                                                           prepare_expression(typed.sign)});
                } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                    actions_.emplace_back(ExecuteActiveMeasurement{
                        prepare_measurement(typed.pauli, planned.active_before, typed.active_pivot),
                        prepare_measurement_correction(typed.outcome, index(typed.branch)),
                        index(typed.branch), index(typed.record)});
                } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                    actions_.emplace_back(ExecuteDormantMeasurement{
                        prepare_measurement_correction(typed.outcome, index(typed.branch)),
                        index(typed.branch), index(typed.record)});
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    actions_.emplace_back(ExecuteClassicalRecord{prepare_expression(typed.outcome),
                                                                 index(typed.record)});
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    actions_.emplace_back(ExecuteSymbolDefinition{prepare_expression(typed.value),
                                                                  index(typed.symbol)});
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    has_readout_noise_ = true;
                    actions_.emplace_back(
                        ExecuteReadoutNoise{prepare_expression(typed.source), index(typed.flip),
                                            index(typed.record), num_readout_noise_sites_++,
                                            typed.prob_zero_to_one, typed.prob_one_to_zero});
                } else if constexpr (std::is_same_v<T, WriteDetector>) {
                    actions_.emplace_back(ExecuteDetector{prepare_expression(typed.outcome),
                                                          index(typed.detector),
                                                          typed.postselected});
                } else if constexpr (std::is_same_v<T, WriteObservable>) {
                    actions_.emplace_back(ExecuteObservable{prepare_expression(typed.outcome),
                                                            index(typed.observable)});
                } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                    std::optional<PreparedPauli> active_projection;
                    if (typed.active_projection.has_value()) {
                        active_projection =
                            prepare_pauli(*typed.active_projection, planned.active_before);
                    }
                    actions_.emplace_back(ExecuteExpectation{std::move(active_projection),
                                                             prepare_expression(typed.sign),
                                                             index(typed.exp_val)});
                } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                    has_instruments_ = true;
                    std::optional<PreparedMeasurement> measurement;
                    if (typed.mode == InstrumentMode::Active ||
                        typed.mode == InstrumentMode::Activate) {
                        const uint32_t width = typed.mode == InstrumentMode::Activate
                                                   ? planned.active_after
                                                   : planned.active_before;
                        const uint64_t support =
                            typed.source.x != 0 ? typed.source.x : typed.source.z;
                        const uint32_t pivot = static_cast<uint32_t>(std::countr_zero(support));
                        measurement = prepare_measurement(typed.source, width, pivot);
                    }
                    actions_.emplace_back(ExecuteInstrument{
                        typed.mode, prepare_expression(typed.sign), std::move(measurement),
                        index(typed.site),
                        typed.destination_flip.has_value()
                            ? std::optional<uint32_t>{index(*typed.destination_flip)}
                            : std::nullopt});
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    const uint32_t noise_end = boundary_index + 1 < boundary_noise_starts.size()
                                                   ? boundary_noise_starts[boundary_index + 1]
                                                   : plan.num_noise_sites;
                    instrument_resume_offsets_[index(typed.site)] =
                        static_cast<uint32_t>(actions_.size());
                    actions_.emplace_back(ExecuteBoundary{index(typed.site), planned.active_before,
                                                          boundary_noise_starts[boundary_index],
                                                          noise_end, typed.symbol_prefix_size});
                    ++boundary_index;
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);
    };

    size_t planned_index = 0;
    while (planned_index < plan.actions.size()) {
        FusedRotationRun run = prepare_fused_rotation_run(
            std::span<const PlannedAction>(plan.actions).subspan(planned_index));
        if (run.rotation.has_value()) {
            const uint32_t fused_index = static_cast<uint32_t>(fused_rotations_.size());
            fused_rotations_.emplace_back(std::move(*run.rotation), runtime_isa);
            actions_.emplace_back(ExecuteFusedRotation{fused_index});
            planned_index += run.action_count;
            continue;
        }
        const size_t unfused_count = std::max<size_t>(run.action_count, 1);
        const size_t run_end = planned_index + unfused_count;
        for (; planned_index < run_end; ++planned_index) {
            lower_action(plan.actions[planned_index]);
        }
    }

    // Transpose expression-major terms into the symbol-major dependency tape.
    expression_dependency_offsets_.assign(static_cast<size_t>(num_symbols_) + 1, 0);
    for (uint32_t symbol : expression_terms) {
        ++expression_dependency_offsets_[static_cast<size_t>(symbol) + 1];
    }
    for (size_t i = 1; i < expression_dependency_offsets_.size(); ++i) {
        expression_dependency_offsets_[i] += expression_dependency_offsets_[i - 1];
    }
    expression_dependency_targets_.resize(expression_terms.size());
    std::vector<uint32_t> next_dependency = expression_dependency_offsets_;
    for (size_t expression = 0; expression < expression_term_begins.size(); ++expression) {
        const uint32_t register_id = static_cast<uint32_t>(expression);
        const uint32_t begin = expression_term_begins[expression];
        const uint32_t end = expression + 1 < expression_term_begins.size()
                                 ? expression_term_begins[expression + 1]
                                 : static_cast<uint32_t>(expression_terms.size());
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t symbol = expression_terms[i];
            expression_dependency_targets_[next_dependency[symbol]++] = register_id;
        }
    }
}

std::vector<double> ExecutablePlan::noise_site_probabilities() const {
    std::vector<double> probabilities;
    probabilities.reserve(noise_sites_.size() + num_readout_noise_sites_);
    for (const PreparedNoiseSite& site : noise_sites_) {
        probabilities.push_back(site.total_probability);
    }
    for (const Action& action : actions_) {
        const auto* readout = std::get_if<ExecuteReadoutNoise>(&action);
        if (readout == nullptr) {
            continue;
        }
        if (readout->prob_zero_to_one != readout->prob_one_to_zero) {
            throw std::invalid_argument(
                "k-fault conditioning does not support asymmetric readout noise; "
                "measurement record index " +
                std::to_string(readout->record) + " has probabilities (" +
                std::to_string(readout->prob_zero_to_one) + ", " +
                std::to_string(readout->prob_one_to_zero) + ")");
        }
        probabilities.push_back(readout->prob_zero_to_one);
    }
    return probabilities;
}

}  // namespace clifft::sampling
