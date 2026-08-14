#include "clifft/sampling/executable_plan_builder.h"

#include "clifft/util/noise_sampling.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace clifft::sampling {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

// Moving these small lowering helpers out of the constructor stopped non-LTO
// Release builds from inlining them and measurably regressed small-plan
// preparation, so preserve that previously implicit optimization here.
#if defined(_MSC_VER)
#define CLIFFT_BUILDER_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define CLIFFT_BUILDER_FORCE_INLINE inline __attribute__((always_inline))
#else
#define CLIFFT_BUILDER_FORCE_INLINE inline
#endif

bool activates_new_x(const ApplyInstrument& instrument, uint32_t active_after) {
    return instrument.mode == InstrumentMode::Activate && active_after > 0 &&
           instrument.source.z == 0 && instrument.source.x == (uint64_t{1} << (active_after - 1));
}

}  // namespace

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::ExpressionDependencies
ExecutablePlan::ExpressionDependencies::build(uint32_t num_symbols,
                                              std::span<const uint32_t> expression_terms,
                                              std::span<const uint32_t> expression_term_begins) {
    ExpressionDependencies result;
    result.offsets_.assign(static_cast<size_t>(num_symbols) + 1, 0);
    for (uint32_t symbol : expression_terms) {
        assert(symbol < num_symbols && "expression term must refer to a plan symbol");
        ++result.offsets_[static_cast<size_t>(symbol) + 1];
    }
    for (size_t i = 1; i < result.offsets_.size(); ++i) {
        result.offsets_[i] += result.offsets_[i - 1];
    }
    result.targets_.resize(expression_terms.size());
    std::vector<uint32_t> next_dependency = result.offsets_;
    for (size_t expression = 0; expression < expression_term_begins.size(); ++expression) {
        const uint32_t register_id = static_cast<uint32_t>(expression);
        const uint32_t begin = expression_term_begins[expression];
        const uint32_t end = expression + 1 < expression_term_begins.size()
                                 ? expression_term_begins[expression + 1]
                                 : static_cast<uint32_t>(expression_terms.size());
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t symbol = expression_terms[i];
            result.targets_[next_dependency[symbol]++] = register_id;
        }
    }
    return result;
}

void ExecutablePlanBuilder::build(ExecutablePlan& output, const SamplingPlan& source) {
    ExecutablePlanBuilder builder(output, source);
    builder.compile();
}

ExecutablePlanBuilder::ExecutablePlanBuilder(ExecutablePlan& output, const SamplingPlan& source)
    : output_(output), source_(source) {}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::compile() {
    source_.validate();
    backend_ = resolve_executor_backend(clifft::internal::runtime_isa());
    output_.backend_ = backend_;
    initialize_program();
    prepare_noise_and_boundaries();
    lower_action_stream();
    build_expression_dependencies();
    validate_executable_plan();
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::initialize_program() {
    if (source_.symbols.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable symbol count exceeds uint32 range");
    }

    output_.num_symbols_ = static_cast<uint32_t>(source_.symbols.size());

    output_.actions_.reserve(source_.actions.size());
    output_.expression_register_constants_.reserve(source_.actions.size());
    // The reserve prepass pays for itself on expression-heavy plans by
    // avoiding repeated growth of the temporary term tape.
    expression_terms_.reserve(estimate_expression_terms());
    expression_term_begins_.reserve(source_.actions.size());
    output_.instrument_resume_offsets_.assign(source_.num_instrument_sites,
                                              std::numeric_limits<uint32_t>::max());
}

CLIFFT_BUILDER_FORCE_INLINE size_t ExecutablePlanBuilder::estimate_expression_terms() const {
    size_t num_terms = 0;
    for (const PlannedAction& planned : source_.actions) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, RotateActivePauli> ||
                              std::is_same_v<T, PromoteDormantRotation>) {
                    num_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                                     std::is_same_v<T, MeasureDormantRandom>) {
                    // The current branch is stored separately so replay can
                    // solve for it from the requested record.
                    num_terms += typed.outcome.terms().size() - 1;
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    num_terms += typed.outcome.terms().size();
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    num_terms += typed.value.terms().size();
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    num_terms += typed.source.terms().size();
                } else if constexpr (std::is_same_v<T, WriteDetector> ||
                                     std::is_same_v<T, WriteObservable>) {
                    num_terms += typed.outcome.terms().size();
                } else if constexpr (std::is_same_v<T, WriteExpectationValue> ||
                                     std::is_same_v<T, ApplyInstrument>) {
                    num_terms += typed.sign.terms().size();
                } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                    // Boundaries have no affine payload.
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
                }
            },
            planned.action);
    }
    if (num_terms > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression storage exceeds uint32 range");
    }
    return num_terms;
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::prepare_noise_and_boundaries() {
    output_.presampled_symbols_.reserve(source_.symbols.size());
    std::vector<bool> bound_presampled(source_.symbols.size(), false);
    output_.noise_sites_.reserve(source_.presampled_noise_sites.size());
    output_.noise_hazards_.reserve(source_.presampled_noise_sites.size());

    double cumulative_hazard = 0.0;
    for (const PresampledNoiseSite& site : source_.presampled_noise_sites) {
        const uint32_t begin = static_cast<uint32_t>(output_.noise_outcomes_.size());
        double cumulative_probability = 0.0;
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            cumulative_probability += outcome.probability;
            output_.noise_outcomes_.push_back({index(outcome.symbol), cumulative_probability});
            bound_presampled[index(outcome.symbol)] = true;
        }
        output_.noise_sites_.push_back(
            {begin, static_cast<uint32_t>(output_.noise_outcomes_.size()) - begin,
             site.total_probability});
        cumulative_hazard += bernoulli_hazard(cumulative_probability);
        output_.noise_hazards_.push_back(cumulative_hazard);
    }
    for (uint32_t symbol = 0; symbol < source_.symbols.size(); ++symbol) {
        if (source_.symbols[symbol].kind != SymbolKind::Presampled) {
            continue;
        }
        output_.presampled_symbols_.push_back(symbol);
        if (!bound_presampled[symbol]) {
            output_.unbound_presampled_symbols_.push_back(symbol);
        }
    }

    boundary_noise_starts_.reserve(source_.num_instrument_sites);
    for (const PlannedAction& planned : source_.actions) {
        if (const auto* boundary = std::get_if<InstrumentBoundary>(&planned.action)) {
            boundary_noise_starts_.push_back(boundary->next_noise_site);
        }
    }
    output_.initial_noise_end_ =
        boundary_noise_starts_.empty() ? source_.num_noise_sites : boundary_noise_starts_.front();
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::ensure_expression_term_capacity(
    size_t additional_terms) const {
    constexpr size_t kMaxExpressionTerms = std::numeric_limits<uint32_t>::max();
    // Dynamic fusion can replace source signs with denser basis expressions,
    // so the source-plan count is only a reserve estimate after lowering.
    if (additional_terms > kMaxExpressionTerms - expression_terms_.size()) {
        throw std::length_error("sampling executable expression storage exceeds uint32 range");
    }
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedExpression
ExecutablePlanBuilder::prepare_expression(const AffineBool& expression) {
    if (output_.expression_register_constants_.size() >= std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression count exceeds uint32 range");
    }
    ensure_expression_term_capacity(expression.terms().size());
    const uint32_t register_id =
        static_cast<uint32_t>(output_.expression_register_constants_.size());
    expression_term_begins_.push_back(static_cast<uint32_t>(expression_terms_.size()));
    output_.expression_register_constants_.push_back(static_cast<uint8_t>(expression.constant()));
    for (SymbolId term : expression.terms()) {
        expression_terms_.push_back(index(term));
    }
    return {register_id};
}

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::PreparedExpression
ExecutablePlanBuilder::prepare_measurement_correction(const AffineBool& outcome, uint32_t branch) {
    if (output_.expression_register_constants_.size() >= std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable expression count exceeds uint32 range");
    }
    ensure_expression_term_capacity(outcome.terms().size() - 1);
    const uint32_t register_id =
        static_cast<uint32_t>(output_.expression_register_constants_.size());
    const uint32_t begin = static_cast<uint32_t>(expression_terms_.size());
    expression_term_begins_.push_back(begin);
    output_.expression_register_constants_.push_back(static_cast<uint8_t>(outcome.constant()));
    for (SymbolId term : outcome.terms()) {
        if (index(term) != branch) {
            expression_terms_.push_back(index(term));
        }
    }
    assert(expression_terms_.size() == static_cast<size_t>(begin) + outcome.terms().size() - 1 &&
           "validated measurement outcome must contain its branch exactly once");
    return {register_id};
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::lower_action(const PlannedAction& planned,
                                                                     size_t& boundary_index) {
    std::visit(
        [&](const auto& typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, RotateActivePauli>) {
                PreparedRotation rotation =
                    prepare_rotation(typed.pauli, planned.active_before, typed.half_turns);
                const DirectRotationKernel kernel =
                    resolve_direct_rotation_kernel(rotation, backend_);
                output_.actions_.emplace_back(ExecutablePlan::ExecuteRotation{
                    std::move(rotation), prepare_expression(typed.sign), kernel});
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecutePromotion{
                    prepare_promotion(typed.half_turns), prepare_expression(typed.sign)});
            } else if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                PreparedMeasurement measurement =
                    prepare_measurement(typed.pauli, planned.active_before, typed.active_pivot);
                const ActiveMeasurementKernel kernel =
                    resolve_active_measurement_kernel(measurement, backend_);
                output_.actions_.emplace_back(ExecutablePlan::ExecuteActiveMeasurement{
                    std::move(measurement),
                    prepare_measurement_correction(typed.outcome, index(typed.branch)),
                    index(typed.branch), index(typed.record), kernel});
            } else if constexpr (std::is_same_v<T, MeasureDormantRandom>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteDormantMeasurement{
                    prepare_measurement_correction(typed.outcome, index(typed.branch)),
                    index(typed.branch), index(typed.record)});
            } else if constexpr (std::is_same_v<T, RecordClassical>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteClassicalRecord{
                    prepare_expression(typed.outcome), index(typed.record)});
            } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteSymbolDefinition{
                    prepare_expression(typed.value), index(typed.symbol)});
            } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                output_.has_readout_noise_ = true;
                output_.actions_.emplace_back(ExecutablePlan::ExecuteReadoutNoise{
                    prepare_expression(typed.source), index(typed.flip), index(typed.record),
                    output_.num_readout_noise_sites_++, typed.prob_zero_to_one,
                    typed.prob_one_to_zero});
            } else if constexpr (std::is_same_v<T, WriteDetector>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteDetector{
                    prepare_expression(typed.outcome), index(typed.detector), typed.postselected});
            } else if constexpr (std::is_same_v<T, WriteObservable>) {
                output_.actions_.emplace_back(ExecutablePlan::ExecuteObservable{
                    prepare_expression(typed.outcome), index(typed.observable)});
            } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                std::optional<PreparedPauli> active_projection;
                if (typed.active_projection.has_value()) {
                    active_projection =
                        prepare_pauli(*typed.active_projection, planned.active_before);
                }
                output_.actions_.emplace_back(ExecutablePlan::ExecuteExpectation{
                    std::move(active_projection), prepare_expression(typed.sign),
                    index(typed.exp_val)});
            } else if constexpr (std::is_same_v<T, ApplyInstrument>) {
                output_.has_instruments_ = true;
                const uint32_t site = index(typed.site);
                switch (typed.mode) {
                    case InstrumentMode::DormantTrap:
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteDormantInstrumentTrap{site}});
                        return;
                    case InstrumentMode::Classical: {
                        assert(typed.destination_flip.has_value() &&
                               "validated in-line instrument must define a destination flip");
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteClassicalInstrument{
                                prepare_expression(typed.sign), site,
                                index(*typed.destination_flip)}});
                        return;
                    }
                    case InstrumentMode::Active: {
                        assert(typed.destination_flip.has_value() &&
                               "validated in-line instrument must define a destination flip");
                        const uint64_t support =
                            typed.source.x != 0 ? typed.source.x : typed.source.z;
                        const uint32_t pivot = static_cast<uint32_t>(std::countr_zero(support));
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteActiveInstrument{
                                prepare_measurement(typed.source, planned.active_before, pivot),
                                prepare_expression(typed.sign), site,
                                index(*typed.destination_flip)}});
                        return;
                    }
                    case InstrumentMode::Activate: {
                        assert(typed.destination_flip.has_value() &&
                               "validated in-line instrument must define a destination flip");
                        const uint32_t destination_flip = index(*typed.destination_flip);
                        if (activates_new_x(typed, planned.active_after)) {
                            output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                                ExecutablePlan::ExecuteNewXInstrumentActivation{
                                    prepare_expression(typed.sign), site, destination_flip,
                                    resolve_new_x_instrument_kernel(planned.active_before,
                                                                    backend_)}});
                            return;
                        }
                        const uint64_t support =
                            typed.source.x != 0 ? typed.source.x : typed.source.z;
                        const uint32_t pivot = static_cast<uint32_t>(std::countr_zero(support));
                        output_.actions_.emplace_back(ExecutablePlan::ExecuteInstrument{
                            ExecutablePlan::ExecuteMeasuredInstrumentActivation{
                                prepare_measurement(typed.source, planned.active_after, pivot),
                                prepare_expression(typed.sign), site, destination_flip}});
                        return;
                    }
                }
                throw std::logic_error("validated instrument mode has no executable lowering");
            } else if constexpr (std::is_same_v<T, InstrumentBoundary>) {
                const uint32_t noise_end = boundary_index + 1 < boundary_noise_starts_.size()
                                               ? boundary_noise_starts_[boundary_index + 1]
                                               : source_.num_noise_sites;
                output_.instrument_resume_offsets_[index(typed.site)] =
                    static_cast<uint32_t>(output_.actions_.size());
                output_.actions_.emplace_back(ExecutablePlan::ExecuteBoundary{
                    index(typed.site), planned.active_before,
                    boundary_noise_starts_[boundary_index], noise_end, typed.symbol_prefix_size});
                ++boundary_index;
            } else {
                static_assert(kAlwaysFalse<T>, "Unhandled SamplingAction alternative");
            }
        },
        planned.action);
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::lower_action_stream() {
    size_t planned_index = 0;
    size_t boundary_index = 0;
    while (planned_index < source_.actions.size()) {
        DynamicFusedRotationRun dynamic_run;
        // AVX2 dynamic fusion regressed large active states despite helping
        // narrower ones, so only the consistently profitable AVX-512 path lowers it.
        if (backend_ == ExecutorBackend::Avx512) {
            dynamic_run = prepare_dynamic_fused_rotation_run(
                std::span<const PlannedAction>(source_.actions).subspan(planned_index));
        }
        if (dynamic_run.rotation.has_value()) {
            PreparedDynamicFusedRotation prepared = std::move(*dynamic_run.rotation);
            ExecutablePlan::PreparedDynamicFusedRotationExecution execution;
            execution.sign_basis.reserve(prepared.sign_basis.size());
            for (const AffineBool& sign : prepared.sign_basis) {
                execution.sign_basis.push_back(prepare_expression(sign));
            }
            execution.variants.reserve(prepared.variants.size());
            for (PreparedFusedRotation& variant : prepared.variants) {
                execution.variants.emplace_back(std::move(variant), backend_);
            }
            const uint32_t fused_index =
                static_cast<uint32_t>(output_.dynamic_fused_rotations_.size());
            output_.dynamic_fused_rotations_.push_back(std::move(execution));
            output_.actions_.emplace_back(ExecutablePlan::ExecuteDynamicFusedRotation{fused_index});
            planned_index += dynamic_run.action_count;
            continue;
        }

        FusedRotationRun run = prepare_fused_rotation_run(
            std::span<const PlannedAction>(source_.actions).subspan(planned_index));
        if (run.rotation.has_value()) {
            const uint32_t fused_index = static_cast<uint32_t>(output_.fused_rotations_.size());
            output_.fused_rotations_.emplace_back(std::move(*run.rotation), backend_);
            output_.actions_.emplace_back(ExecutablePlan::ExecuteFusedRotation{fused_index});
            planned_index += run.action_count;
            continue;
        }
        const size_t unfused_count = std::max<size_t>(run.action_count, 1);
        const size_t run_end = planned_index + unfused_count;
        for (; planned_index < run_end; ++planned_index) {
            lower_action(source_.actions[planned_index], boundary_index);
        }
    }
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::build_expression_dependencies() {
    output_.expression_dependencies_ = ExecutablePlan::ExpressionDependencies::build(
        output_.num_symbols_, expression_terms_, expression_term_begins_);
}

void ExecutablePlan::ExpressionDependencies::validate(uint32_t num_symbols,
                                                      size_t num_registers) const noexcept {
#ifndef NDEBUG
    assert(offsets_.size() == static_cast<size_t>(num_symbols) + 1 &&
           "expression dependency offsets have the wrong size");
    assert(!offsets_.empty() && offsets_.front() == 0 && offsets_.back() == targets_.size() &&
           "expression dependency ranges are inconsistent");
    for (size_t i = 1; i < offsets_.size(); ++i) {
        assert(offsets_[i] >= offsets_[i - 1] && "expression dependency offsets are not ordered");
    }
    for (uint32_t target : targets_) {
        assert(target < num_registers && "expression dependency target is out of range");
    }
#else
    static_cast<void>(num_symbols);
    static_cast<void>(num_registers);
#endif
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::validate_executable_plan() const {
#ifndef NDEBUG
    assert(expression_term_begins_.size() == output_.expression_register_constants_.size() &&
           "expression register storage is inconsistent");
    output_.expression_dependencies_.validate(output_.num_symbols_,
                                              output_.expression_register_constants_.size());

    auto validate_expression = [&](ExecutablePlan::PreparedExpression expression) {
        assert(expression.register_id < output_.expression_register_constants_.size() &&
               "action expression is out of range");
    };
    for (const ExecutablePlan::Action& action : output_.actions_) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation>) {
                    validate_expression(typed.sign);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFusedRotation>) {
                    assert(typed.rotation_index < output_.fused_rotations_.size() &&
                           "fused rotation is out of range");
                } else if constexpr (std::is_same_v<T,
                                                    ExecutablePlan::ExecuteDynamicFusedRotation>) {
                    assert(typed.rotation_index < output_.dynamic_fused_rotations_.size() &&
                           "dynamic fused rotation is out of range");
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecutePromotion>) {
                    validate_expression(typed.sign);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                    validate_expression(typed.correction);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDormantMeasurement>) {
                    validate_expression(typed.correction);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteClassicalRecord>) {
                    validate_expression(typed.outcome);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteSymbolDefinition>) {
                    validate_expression(typed.value);
                    assert(typed.symbol < output_.num_symbols_ && "defined symbol is out of range");
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise>) {
                    validate_expression(typed.source);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDetector>) {
                    validate_expression(typed.outcome);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteObservable>) {
                    validate_expression(typed.outcome);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteExpectation>) {
                    validate_expression(typed.sign);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteInstrument>) {
                    std::visit(
                        [&](const auto& instrument) {
                            using Instrument = std::decay_t<decltype(instrument)>;
                            assert(instrument.site < output_.instrument_distributions_.size() &&
                                   "instrument site is out of range");
                            if constexpr (!std::is_same_v<
                                              Instrument,
                                              ExecutablePlan::ExecuteDormantInstrumentTrap>) {
                                validate_expression(instrument.sign);
                                assert(instrument.destination_flip < output_.num_symbols_ &&
                                       "instrument destination flip is out of range");
                            }
                        },
                        typed.form);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteBoundary>) {
                    assert(typed.site < output_.instrument_resume_offsets_.size() &&
                           "instrument boundary site is out of range");
                } else {
                    static_assert(kAlwaysFalse<T>, "Unhandled executable action alternative");
                }
            },
            action);
    }
    for (const ExecutablePlan::PreparedDynamicFusedRotationExecution& rotation :
         output_.dynamic_fused_rotations_) {
        for (ExecutablePlan::PreparedExpression sign : rotation.sign_basis) {
            validate_expression(sign);
        }
    }
#endif
}

#undef CLIFFT_BUILDER_FORCE_INLINE

}  // namespace clifft::sampling
