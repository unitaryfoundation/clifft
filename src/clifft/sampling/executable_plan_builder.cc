#include "clifft/sampling/executable_plan_builder.h"

#include "clifft/util/noise_sampling.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
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

struct PresampledExpressionBlock {
    bool constant = false;
    std::vector<uint32_t> terms;
    std::vector<uint32_t> registers;
    uint32_t parent = std::numeric_limits<uint32_t>::max();
    bool invert_parent = false;
    uint32_t depth = 0;
    std::vector<uint32_t> delta_terms;
};

inline constexpr uint64_t kMinBatchExpressionTerms = 1024;
inline constexpr uint64_t kBatchExpressionCostNumerator = 3;
inline constexpr uint64_t kBatchExpressionCostDenominator = 4;

std::vector<uint32_t> symmetric_difference(std::span<const uint32_t> left,
                                           std::span<const uint32_t> right) {
    std::vector<uint32_t> result;
    result.reserve(left.size() + right.size());
    std::ranges::set_symmetric_difference(left, right, std::back_inserter(result));
    return result;
}

uint64_t expression_hash(bool constant, std::span<const uint32_t> terms) noexcept {
    uint64_t hash = constant ? 0x9e3779b97f4a7c15ULL : 0xcbf29ce484222325ULL;
    for (uint32_t term : terms) {
        hash ^= term;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

}  // namespace

CLIFFT_BUILDER_FORCE_INLINE ExecutablePlan::ExpressionDependencies
ExecutablePlan::ExpressionDependencies::build(uint32_t num_symbols,
                                              std::span<const uint32_t> expression_terms,
                                              std::span<const uint32_t> expression_term_begins) {
    assert((expression_term_begins.empty() || expression_term_begins.front() == 0) &&
           "the first expression must begin at the start of the term tape");
    assert(std::ranges::is_sorted(expression_term_begins) &&
           "expression term ranges must be ordered");
    assert((expression_term_begins.empty() ||
            expression_term_begins.back() <= expression_terms.size()) &&
           "expression term ranges must stay inside the term tape");

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
    prepare_batch_rotation_runs();
    build_expression_dependencies();
    prepare_batch_expression_initialization();
    validate_executable_plan();
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::initialize_program() {
    if (source_.symbols.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable symbol count exceeds uint32 range");
    }
    if (source_.actions.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("sampling executable action count exceeds uint32 range");
    }

    output_.num_symbols_ = static_cast<uint32_t>(source_.symbols.size());

    output_.actions_.reserve(source_.actions.size());
    if (source_.source_map.has_value()) {
        output_.action_plan_ranges_.reserve(source_.actions.size());
    }
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
            record_action_origin(static_cast<uint32_t>(planned_index),
                                 static_cast<uint32_t>(planned_index + dynamic_run.action_count));
            planned_index += dynamic_run.action_count;
            continue;
        }

        FusedRotationRun run = prepare_fused_rotation_run(
            std::span<const PlannedAction>(source_.actions).subspan(planned_index));
        if (run.rotation.has_value()) {
            const uint32_t fused_index = static_cast<uint32_t>(output_.fused_rotations_.size());
            output_.fused_rotations_.emplace_back(std::move(*run.rotation), backend_);
            output_.actions_.emplace_back(ExecutablePlan::ExecuteFusedRotation{fused_index});
            record_action_origin(static_cast<uint32_t>(planned_index),
                                 static_cast<uint32_t>(planned_index + run.action_count));
            planned_index += run.action_count;
            continue;
        }
        const size_t unfused_count = std::max<size_t>(run.action_count, 1);
        const size_t run_end = planned_index + unfused_count;
        for (; planned_index < run_end; ++planned_index) {
            lower_action(source_.actions[planned_index], boundary_index);
            record_action_origin(static_cast<uint32_t>(planned_index),
                                 static_cast<uint32_t>(planned_index + 1));
        }
    }
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::record_action_origin(uint32_t plan_begin,
                                                                             uint32_t plan_end) {
    if (!source_.source_map.has_value()) {
        return;
    }
    assert(plan_begin < plan_end && plan_end <= source_.actions.size() &&
           "executable action must name a nonempty plan range");
    assert(output_.action_plan_ranges_.size() + 1 == output_.actions_.size() &&
           "each executable action must receive exactly one plan range");
    output_.action_plan_ranges_.push_back({plan_begin, plan_end});
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::prepare_batch_rotation_runs() {
    output_.batch_rotation_run_lengths_.assign(output_.actions_.size(), 0);
    bool prepared_run = false;
    size_t action_index = 0;
    while (action_index < output_.actions_.size()) {
        if (!std::holds_alternative<ExecutablePlan::ExecuteRotation>(
                output_.actions_[action_index])) {
            ++action_index;
            continue;
        }
        size_t run_end = action_index + 1;
        while (run_end < output_.actions_.size() &&
               std::holds_alternative<ExecutablePlan::ExecuteRotation>(output_.actions_[run_end])) {
            ++run_end;
        }
        while (run_end - action_index >= ExecutablePlan::kMinBatchRotationRunLength) {
            const size_t run_length = std::min<size_t>(run_end - action_index,
                                                       ExecutablePlan::kMaxBatchRotationRunLength);
            output_.batch_rotation_run_lengths_[action_index] = static_cast<uint8_t>(run_length);
            prepared_run = true;
            action_index += run_length;
        }
        action_index = run_end;
    }
    if (!prepared_run) {
        output_.batch_rotation_run_lengths_.clear();
    }
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::build_expression_dependencies() {
    output_.expression_dependencies_ = ExecutablePlan::ExpressionDependencies::build(
        output_.num_symbols_, expression_terms_, expression_term_begins_);
}

void ExecutablePlanBuilder::prepare_batch_expression_initialization() {
    if (output_.has_instruments_) {
        return;
    }
    std::unordered_multimap<uint64_t, uint32_t> interned;
    std::vector<PresampledExpressionBlock> blocks;
    blocks.reserve(expression_term_begins_.size());
    interned.reserve(expression_term_begins_.size());
    uint64_t original_presampled_terms = 0;
    for (size_t expression = 0; expression < expression_term_begins_.size(); ++expression) {
        const uint32_t begin = expression_term_begins_[expression];
        const uint32_t end = expression + 1 < expression_term_begins_.size()
                                 ? expression_term_begins_[expression + 1]
                                 : static_cast<uint32_t>(expression_terms_.size());
        std::vector<uint32_t> terms;
        for (uint32_t term = begin; term < end; ++term) {
            const uint32_t symbol = expression_terms_[term];
            if (source_.symbols[symbol].kind == SymbolKind::Presampled) {
                terms.push_back(symbol);
                ++original_presampled_terms;
            }
        }
        const bool constant = output_.expression_register_constants_[expression] != 0;
        const uint64_t hash = expression_hash(constant, terms);
        uint32_t block_index = std::numeric_limits<uint32_t>::max();
        const auto [first, last] = interned.equal_range(hash);
        for (auto position = first; position != last; ++position) {
            const PresampledExpressionBlock& candidate = blocks[position->second];
            if (candidate.constant == constant && candidate.terms == terms) {
                block_index = position->second;
                break;
            }
        }
        if (block_index == std::numeric_limits<uint32_t>::max()) {
            block_index = static_cast<uint32_t>(blocks.size());
            PresampledExpressionBlock block;
            block.constant = constant;
            block.terms = std::move(terms);
            blocks.push_back(std::move(block));
            interned.emplace(hash, block_index);
        }
        blocks[block_index].registers.push_back(static_cast<uint32_t>(expression));
    }
    if (original_presampled_terms == 0) {
        return;
    }

    std::vector<std::vector<uint32_t>> blocks_by_symbol(output_.num_symbols_);
    std::vector<uint32_t> intersection_counts(blocks.size(), 0);
    std::vector<uint32_t> candidate_parents;
    uint32_t max_depth = 0;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        PresampledExpressionBlock& block = blocks[block_index];
        block.delta_terms = block.terms;
        uint64_t best_cost = block.delta_terms.size();
        // A parent can beat direct initialization only when it shares a term.
        // Accumulating intersections through the symbol index scores every
        // viable earlier block exactly without scanning every term pair.
        candidate_parents.clear();
        for (uint32_t symbol : block.terms) {
            for (uint32_t parent_index : blocks_by_symbol[symbol]) {
                if (intersection_counts[parent_index]++ == 0) {
                    candidate_parents.push_back(parent_index);
                }
            }
        }
        std::ranges::sort(candidate_parents);
        for (uint32_t parent_index : candidate_parents) {
            const bool invert_parent = block.constant != blocks[parent_index].constant;
            const uint64_t cost = static_cast<uint64_t>(block.terms.size()) +
                                  blocks[parent_index].terms.size() -
                                  2 * static_cast<uint64_t>(intersection_counts[parent_index]) +
                                  static_cast<uint64_t>(invert_parent);
            if (cost < best_cost) {
                best_cost = cost;
                block.parent = parent_index;
                block.invert_parent = invert_parent;
            }
        }
        for (uint32_t parent_index : candidate_parents) {
            intersection_counts[parent_index] = 0;
        }
        if (block.parent != std::numeric_limits<uint32_t>::max()) {
            block.delta_terms = symmetric_difference(block.terms, blocks[block.parent].terms);
            block.depth = blocks[block.parent].depth + 1;
        }
        max_depth = std::max(max_depth, block.depth);
        for (uint32_t symbol : block.terms) {
            blocks_by_symbol[symbol].push_back(static_cast<uint32_t>(block_index));
        }
    }

    uint64_t prepared_operations = 0;
    for (const PresampledExpressionBlock& block : blocks) {
        prepared_operations += block.delta_terms.size();
        prepared_operations += static_cast<uint64_t>(block.invert_parent);
        prepared_operations +=
            static_cast<uint64_t>(block.parent != std::numeric_limits<uint32_t>::max());
        prepared_operations += block.registers.size() - 1;
    }
    // The staged program trades each retained edge for extra parent and copy
    // passes. Require a substantial execution-work reduction so small or
    // weakly related expression sets stay on the simpler dependency path.
    if (original_presampled_terms < kMinBatchExpressionTerms ||
        prepared_operations * kBatchExpressionCostDenominator >
            original_presampled_terms * kBatchExpressionCostNumerator) {
        return;
    }

    std::vector<std::vector<ExecutablePlan::PresampledExpressionInitialization>>
        initializations_by_level(static_cast<size_t>(max_depth) + 1);
    std::vector<std::vector<ExecutablePlan::PresampledExpressionDelta>> deltas_by_level(
        static_cast<size_t>(max_depth) + 1);
    for (const PresampledExpressionBlock& block : blocks) {
        assert(!block.registers.empty() && "interned expression block must have a destination");
        const uint32_t destination = block.registers.front();
        const uint32_t parent = block.parent == std::numeric_limits<uint32_t>::max()
                                    ? std::numeric_limits<uint32_t>::max()
                                    : blocks[block.parent].registers.front();
        initializations_by_level[block.depth].push_back({destination, parent, block.invert_parent});
        for (uint32_t symbol : block.delta_terms) {
            deltas_by_level[block.depth].push_back({symbol, destination});
        }
        for (size_t register_index = 1; register_index < block.registers.size(); ++register_index) {
            output_.presampled_copies_.push_back({destination, block.registers[register_index]});
        }
    }

    output_.presampled_initialization_level_offsets_.push_back(0);
    output_.presampled_delta_level_offsets_.push_back(0);
    for (size_t level = 0; level < initializations_by_level.size(); ++level) {
        auto& deltas = deltas_by_level[level];
        std::ranges::sort(deltas, {}, &ExecutablePlan::PresampledExpressionDelta::symbol);
        output_.presampled_initializations_.insert(output_.presampled_initializations_.end(),
                                                   initializations_by_level[level].begin(),
                                                   initializations_by_level[level].end());
        output_.presampled_deltas_.insert(output_.presampled_deltas_.end(), deltas.begin(),
                                          deltas.end());
        output_.presampled_initialization_level_offsets_.push_back(
            static_cast<uint32_t>(output_.presampled_initializations_.size()));
        output_.presampled_delta_level_offsets_.push_back(
            static_cast<uint32_t>(output_.presampled_deltas_.size()));
    }
}

CLIFFT_BUILDER_FORCE_INLINE void ExecutablePlanBuilder::validate_executable_plan() const {
#ifndef NDEBUG
    assert(expression_term_begins_.size() == output_.expression_register_constants_.size() &&
           "expression register storage is inconsistent");
    output_.expression_dependencies_.validate(output_.num_symbols_,
                                              output_.expression_register_constants_.size());
    if (!output_.presampled_initialization_level_offsets_.empty()) {
        assert(output_.presampled_initialization_level_offsets_.size() ==
                   output_.presampled_delta_level_offsets_.size() &&
               output_.presampled_initialization_level_offsets_.front() == 0 &&
               output_.presampled_delta_level_offsets_.front() == 0 &&
               output_.presampled_initialization_level_offsets_.back() ==
                   output_.presampled_initializations_.size() &&
               output_.presampled_delta_level_offsets_.back() ==
                   output_.presampled_deltas_.size() &&
               "presampled expression levels must cover their operation tapes");
        for (const ExecutablePlan::PresampledExpressionInitialization& initialization :
             output_.presampled_initializations_) {
            assert(initialization.destination < output_.expression_register_constants_.size() &&
                   (initialization.parent == std::numeric_limits<uint32_t>::max() ||
                    initialization.parent < output_.expression_register_constants_.size()) &&
                   "presampled expression initialization must name valid registers");
        }
        for (const ExecutablePlan::PresampledExpressionDelta& delta : output_.presampled_deltas_) {
            assert(delta.symbol < source_.symbols.size() &&
                   source_.symbols[delta.symbol].kind == SymbolKind::Presampled &&
                   delta.destination < output_.expression_register_constants_.size() &&
                   "presampled expression delta must name valid storage");
        }
        for (const ExecutablePlan::PresampledExpressionCopy& copy : output_.presampled_copies_) {
            assert(copy.source < output_.expression_register_constants_.size() &&
                   copy.destination < output_.expression_register_constants_.size() &&
                   "presampled expression copy must name valid registers");
        }
    }
    assert((output_.batch_rotation_run_lengths_.empty() ||
            output_.batch_rotation_run_lengths_.size() == output_.actions_.size()) &&
           "batch rotation metadata must be empty or parallel to the action stream");
    if (!output_.batch_rotation_run_lengths_.empty()) {
        for (size_t action_index = 0; action_index < output_.actions_.size(); ++action_index) {
            const uint32_t run_length = output_.batch_rotation_run_lengths_[action_index];
            if (run_length == 0) {
                continue;
            }
            assert(run_length >= ExecutablePlan::kMinBatchRotationRunLength &&
                   run_length <= ExecutablePlan::kMaxBatchRotationRunLength &&
                   run_length <= output_.actions_.size() - action_index &&
                   "batch rotation run length must be valid");
            for (size_t offset = 0; offset < run_length; ++offset) {
                assert(std::holds_alternative<ExecutablePlan::ExecuteRotation>(
                           output_.actions_[action_index + offset]) &&
                       "batch rotation run must contain only direct rotations");
                assert((offset == 0 ||
                        output_.batch_rotation_run_lengths_[action_index + offset] == 0) &&
                       "batch rotation runs must not overlap");
            }
            action_index += run_length - 1;
        }
    }
    if (source_.source_map.has_value()) {
        assert(output_.action_plan_ranges_.size() == output_.actions_.size() &&
               "executable provenance must remain parallel to the action stream");
        uint32_t expected_begin = 0;
        for (const ExecutablePlan::PlanActionRange& range : output_.action_plan_ranges_) {
            assert(range.begin == expected_begin && range.begin < range.end &&
                   range.end <= source_.actions.size() &&
                   "executable provenance must partition the plan action stream");
            expected_begin = range.end;
        }
        assert(expected_begin == source_.actions.size() &&
               "executable provenance must cover every plan action");
    } else {
        assert(output_.action_plan_ranges_.empty() &&
               "ordinary lowering must not retain debug provenance");
    }

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
