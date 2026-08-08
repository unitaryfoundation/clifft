#include "clifft/sampling/executor.h"

#include "clifft/util/fault_sampling.h"
#include "clifft/util/noise_sampling.h"
#include "clifft/util/numeric.h"
#include "clifft/util/runtime_isa.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace clifft::sampling {

namespace {

template <typename>
inline constexpr bool kAlwaysFalse = false;

constexpr double kLogHalf = -std::numbers::ln2;

}  // namespace

MeasurementBranchClassification classify_measurement_branch(
    MeasurementProbabilities probabilities) noexcept {
    const double total = probabilities.total();
    assert(is_finite_robust(probabilities.zero) && probabilities.zero >= 0.0 &&
           is_finite_robust(probabilities.one) && probabilities.one >= 0.0 &&
           is_finite_robust(total) && total > 0.0 &&
           "measurement probabilities must be finite, nonnegative, and nonzero");
    const double epsilon = kMeasurementDustEpsilon * total;
    if (probabilities.one <= epsilon) {
        return {.kind = MeasurementBranchKind::DeterministicZero,
                .clamped_dust = probabilities.one > 0.0};
    }
    if (probabilities.zero <= epsilon) {
        return {.kind = MeasurementBranchKind::DeterministicOne,
                .clamped_dust = probabilities.zero > 0.0};
    }
    return {.kind = MeasurementBranchKind::Random};
}

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
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    const bool prepare_avx512_sidecars = runtime_isa == internal::RuntimeIsa::Avx512;
#endif
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
            fused_rotations_.push_back(std::move(*run.rotation));
            FusedRotationSidecar sidecar;
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
            if (prepare_avx512_sidecars) {
                sidecar = prepare_fused_rotation_avx512_sidecar(fused_rotations_.back());
            }
#endif
            fused_rotation_sidecars_.push_back(std::move(sidecar));
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

Executor::Executor(const ExecutablePlan& plan, uint64_t seed)
    : root_plan_(&plan),
      plan_(&plan),
      state_(plan.max_active_width_, plan.initial_active_width_, plan.global_weight_),
      symbols_(plan.num_symbols_, 0),
      expression_registers_(plan.expression_register_constants_),
      records_(static_cast<size_t>(plan.num_visible_records_) + plan.num_hidden_records_, 0),
      detectors_(plan.num_detectors_, 0),
      observables_(plan.num_observables_, 0),
      exp_vals_(plan.num_exp_vals_, 0.0),
      forced_record_mask_(records_.size(), 0),
      forced_record_values_(records_.size(), 0),
      rng_(seed) {
    previous_presampled_ones_.reserve(plan.presampled_symbols_.size());
}

void Executor::run_shot() noexcept {
    plan_ = root_plan_;
    assert(plan_->unbound_presampled_symbols_.empty() &&
           "automatic execution requires every presampled symbol to have a distribution");
    reset_shot();
    sample_presampled_noise(0, plan_->initial_noise_end_);
    (void)execute_actions<false, true, false>({});
}

void Executor::run_shot(std::span<const uint8_t> presampled_values) noexcept {
    plan_ = root_plan_;
    reset_shot();
    assign_presampled_values(presampled_values);
    (void)execute_actions<false, false, false>({});
}

void Executor::run_shot(KFaultSampler& fault_sampler) noexcept {
    plan_ = root_plan_;
    assert(fault_sampler.num_sites() ==
               plan_->noise_sites_.size() + plan_->num_readout_noise_sites_ &&
           "conditioned sampler must cover every quantum and readout fault site");
    reset_shot();
    forced_fault_sites_ = fault_sampler.sample([&]() noexcept { return rng_.next_double(); });
    forced_fault_cursor_ = 0;
    assign_forced_quantum_faults();
    (void)execute_actions<false, false, true>({});
    forced_fault_sites_ = {};
    forced_fault_cursor_ = 0;
}

void Executor::resume(const ExecutablePlan& continuation,
                      std::optional<ForcedTraceOut> forced_trace_out) {
    if (!pending_trap_.has_value()) {
        throw std::invalid_argument("sampling executor resume requires a pending instrument trap");
    }
    const uint32_t site = index(pending_trap_->site);
    if (site >= continuation.instrument_resume_offsets_.size() ||
        site >= plan_->instrument_resume_offsets_.size()) {
        throw std::invalid_argument("sampling continuation omits the trapped instrument site");
    }
    const uint32_t offset = continuation.instrument_resume_offsets_[site];
    if (offset >= continuation.actions_.size()) {
        throw std::invalid_argument(
            "sampling continuation has no resume boundary for the trapped site");
    }
    const auto* boundary =
        std::get_if<ExecutablePlan::ExecuteBoundary>(&continuation.actions_[offset]);
    const uint32_t old_offset = plan_->instrument_resume_offsets_[site];
    const auto* old_boundary =
        old_offset < plan_->actions_.size()
            ? std::get_if<ExecutablePlan::ExecuteBoundary>(&plan_->actions_[old_offset])
            : nullptr;
    if (boundary == nullptr || old_boundary == nullptr || boundary->site != site ||
        boundary->active_width != state_.active_width()) {
        throw std::invalid_argument(
            "sampling continuation boundary is incompatible with the live active state");
    }
    if (boundary->symbol_prefix_size != old_boundary->symbol_prefix_size) {
        throw std::invalid_argument(
            "sampling continuation changes symbol identities before the trapped site");
    }
    if (continuation.num_qubits_ != plan_->num_qubits_ ||
        continuation.num_visible_records_ != plan_->num_visible_records_ ||
        continuation.num_detectors_ != plan_->num_detectors_ ||
        continuation.num_observables_ != plan_->num_observables_ ||
        continuation.num_exp_vals_ != plan_->num_exp_vals_) {
        throw std::invalid_argument(
            "sampling continuation changes externally visible plan dimensions");
    }
    if (!continuation.unbound_presampled_symbols_.empty()) {
        throw std::invalid_argument(
            "sampling continuation requires distributions for every presampled symbol");
    }

    state_.ensure_capacity(continuation.max_active_width_);
    symbols_.resize(std::max(symbols_.size(), static_cast<size_t>(continuation.num_symbols_)), 0);
    std::fill(symbols_.begin() + boundary->symbol_prefix_size, symbols_.end(), uint8_t{0});
    expression_registers_.resize(
        std::max(expression_registers_.size(), continuation.expression_register_constants_.size()),
        0);
    initialize_expression_registers(continuation, boundary->symbol_prefix_size);
    const size_t record_count =
        static_cast<size_t>(continuation.num_visible_records_) + continuation.num_hidden_records_;
    records_.resize(std::max(records_.size(), record_count), 0);
    forced_record_mask_.resize(records_.size(), 0);
    forced_record_values_.resize(records_.size(), 0);
    previous_presampled_ones_.reserve(continuation.presampled_symbols_.size());

    if (forced_trace_out.has_value()) {
        if (forced_trace_out->source > 1 || index(forced_trace_out->record) >= record_count) {
            throw std::invalid_argument("sampling continuation forced record is out of range");
        }
        const uint32_t record = index(forced_trace_out->record);
        forced_record_mask_[record] = 1;
        forced_record_values_[record] = forced_trace_out->source;
    }

    plan_ = &continuation;
    pending_trap_.reset();
    (void)execute_actions<false, true, false>({}, offset);
    if (forced_trace_out.has_value() && forced_record_mask_[index(forced_trace_out->record)] != 0) {
        throw std::logic_error("sampling continuation did not consume its forced trace-out record");
    }
}

void Executor::return_to_root_plan() noexcept {
    assert(!pending_trap_.has_value() &&
           "a trapped shot must resume before releasing its continuation plan");
    plan_ = root_plan_;
}

ReplayResult Executor::replay_shot(std::span<const uint8_t> forced_records,
                                   std::span<const uint8_t> presampled_values) noexcept {
    plan_ = root_plan_;
    assert(forced_records.size() ==
               static_cast<size_t>(plan_->num_visible_records_) + plan_->num_hidden_records_ &&
           "one forced value is required for every plan record");
    assert(std::ranges::all_of(forced_records, [](uint8_t value) { return value <= 1; }) &&
           "forced records must be Boolean");
    reset_shot();
    assign_presampled_values(presampled_values);
    return execute_actions<true, false, false>(forced_records);
}

void Executor::reset_shot() noexcept {
    state_.reset();
    // Validation guarantees that every mutable symbol and output is overwritten
    // before use on a completed shot. Only nonfiring noise symbols need restoring.
    for (uint32_t symbol : previous_presampled_ones_) {
        assert(symbol < symbols_.size() && symbols_[symbol] == 1 &&
               "tracked presampled symbols must be set");
        symbols_[symbol] = 0;
    }
    previous_presampled_ones_.clear();
    initialize_expression_registers(*plan_, 0);
    std::ranges::fill(forced_record_mask_, uint8_t{0});
    discarded_ = false;
    pending_trap_.reset();
}

void Executor::initialize_expression_registers(const ExecutablePlan& plan,
                                               uint32_t symbol_prefix_size) noexcept {
    assert(expression_registers_.size() >= plan.expression_register_constants_.size() &&
           "expression register storage must cover the active plan");
    assert(symbol_prefix_size <= plan.num_symbols_ &&
           "expression reconstruction prefix must belong to the active plan");
    std::ranges::copy(plan.expression_register_constants_, expression_registers_.begin());
    for (uint32_t symbol = 0; symbol < symbol_prefix_size; ++symbol) {
        if (symbols_[symbol] != 0) {
            propagate_true_symbol(plan, symbol);
        }
    }
}

void Executor::propagate_true_symbol(const ExecutablePlan& plan, uint32_t symbol) noexcept {
    assert(static_cast<size_t>(symbol) + 1 < plan.expression_dependency_offsets_.size() &&
           "assigned symbol must have an expression dependency range");
    const uint32_t begin = plan.expression_dependency_offsets_[symbol];
    const uint32_t end = plan.expression_dependency_offsets_[symbol + 1];
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t register_id = plan.expression_dependency_targets_[i];
        assert(register_id < expression_registers_.size() &&
               "expression dependency must refer to a preallocated register");
        expression_registers_[register_id] ^= uint8_t{1};
    }
}

void Executor::assign_symbol(uint32_t symbol, bool value) noexcept {
    // Expression registers begin at their constant values and receive each
    // symbol's contribution exactly once when that symbol is assigned true.
    // False is the unpropagated baseline, so assigning false updates symbols_
    // without modifying registers. symbols_ remains authoritative because resume
    // reconstructs continuation registers by replaying the live true prefix.
    assert(symbol < symbols_.size() && "assigned symbol must belong to the active plan");
    symbols_[symbol] = static_cast<uint8_t>(value);
    if (value) {
        propagate_true_symbol(*plan_, symbol);
    }
}

void Executor::assign_presampled_values(std::span<const uint8_t> presampled_values) noexcept {
    assert(presampled_values.size() == plan_->presampled_symbols_.size() &&
           "one value is required for every presampled symbol");
    for (size_t i = 0; i < presampled_values.size(); ++i) {
        assert(presampled_values[i] <= 1 && "presampled symbols must be Boolean");
        const uint32_t symbol = plan_->presampled_symbols_[i];
        assign_symbol(symbol, presampled_values[i] != 0);
        if (presampled_values[i] != 0) {
            previous_presampled_ones_.push_back(symbol);
        }
    }
}

void Executor::sample_presampled_noise(uint32_t begin, uint32_t end) noexcept {
    assert(begin <= end && end <= plan_->noise_sites_.size() &&
           "noise segment must stay inside prepared sites");
    uint32_t first_candidate = begin;
    while (first_candidate < end) {
        const double current_hazard =
            first_candidate == 0 ? 0.0 : plan_->noise_hazards_[first_candidate - 1];
        if (current_hazard >= plan_->noise_hazards_[end - 1]) {
            return;
        }
        const uint32_t site_index =
            sample_next_noise_site(plan_->noise_hazards_, first_candidate, rng_.next_double());
        if (site_index == kNoNoiseSite || site_index >= end) {
            return;
        }
        activate_noise_site(site_index);
        first_candidate = site_index + 1;
    }
}

void Executor::activate_noise_site(uint32_t site_index) noexcept {
    assert(site_index < plan_->noise_sites_.size() && "noise site must be prepared");
    const ExecutablePlan::PreparedNoiseSite& site = plan_->noise_sites_[site_index];
    assert(site.outcome_count > 0 && site.total_probability > 0.0 &&
           "a firing noise site must contain positive-probability outcomes");
    uint32_t outcome_index = site.outcome_begin;
    if (site.outcome_count > 1) {
        const double channel_draw = rng_.next_double() * site.total_probability;
        while (channel_draw >= plan_->noise_outcomes_[outcome_index].cumulative_probability) {
            ++outcome_index;
            assert(outcome_index < site.outcome_begin + site.outcome_count &&
                   "channel draw must select one prepared outcome");
        }
    }
    const uint32_t symbol = plan_->noise_outcomes_[outcome_index].symbol;
    assert(symbol < symbols_.size() && symbols_[symbol] == 0 &&
           "a noise site may define only one fresh symbol per shot");
    assign_symbol(symbol, true);
    previous_presampled_ones_.push_back(symbol);
}

void Executor::assign_forced_quantum_faults() noexcept {
    const uint32_t num_quantum_sites = static_cast<uint32_t>(plan_->noise_sites_.size());
    while (forced_fault_cursor_ < forced_fault_sites_.size() &&
           forced_fault_sites_[forced_fault_cursor_] < num_quantum_sites) {
        activate_noise_site(forced_fault_sites_[forced_fault_cursor_]);
        ++forced_fault_cursor_;
    }
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteRotation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    apply_rotation(state_, action.rotation, evaluate(action.sign));
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    assert(action.rotation_index < plan_->fused_rotations_.size() &&
           "fused rotation action must reference a prepared descriptor");
    assert(action.rotation_index < plan_->fused_rotation_sidecars_.size() &&
           "fused rotation action must reference a prepared sidecar slot");
    const PreparedFusedRotation& rotation = plan_->fused_rotations_[action.rotation_index];
    const FusedRotationSidecar& sidecar =
        plan_->fused_rotation_sidecars_[action.rotation_index];
    if (sidecar) {
        sidecar.kernel(state_, rotation, sidecar.storage.get());
    } else {
        apply_fused_rotation(state_, rotation);
    }
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecutePromotion& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    apply_promotion(state_, action.promotion, evaluate(action.sign));
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    const MeasurementProbabilities probabilities =
        measurement_probabilities(state_, action.measurement);
    const bool correction = evaluate(action.correction);
    bool branch = false;
    if constexpr (ForceRecords) {
        branch = (forced_records[action.record] != 0) ^ correction;
        const std::optional<double> log_increment = force_active_branch(probabilities, branch);
        if (!log_increment.has_value()) {
            result.reachable = false;
            return;
        }
        result.log_probability += *log_increment;
    } else {
        if (forced_record_mask_[action.record] != 0) {
            branch = (forced_record_values_[action.record] != 0) ^ correction;
            const std::optional<double> forced = force_active_branch(probabilities, branch);
            assert(forced.has_value() &&
                   "forced continuation measurement branch must be reachable");
            forced_record_mask_[action.record] = 0;
        } else {
            branch = sample_active_branch(probabilities);
        }
    }
    assign_symbol(action.branch, branch);
    collapse_measurement(state_, action.measurement, branch, probabilities.for_branch(branch));
    records_[action.record] = static_cast<uint8_t>(branch ^ correction);
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    const bool correction = evaluate(action.correction);
    bool branch = false;
    if constexpr (ForceRecords) {
        branch = (forced_records[action.record] != 0) ^ correction;
        result.log_probability += kLogHalf;
    } else {
        if (forced_record_mask_[action.record] != 0) {
            branch = (forced_record_values_[action.record] != 0) ^ correction;
            forced_record_mask_[action.record] = 0;
        } else {
            branch = sample_dormant_branch();
        }
    }
    assign_symbol(action.branch, branch);
    records_[action.record] = static_cast<uint8_t>(branch ^ correction);
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    records_[action.record] = static_cast<uint8_t>(evaluate(action.outcome));
    if constexpr (ForceRecords) {
        if (records_[action.record] != forced_records[action.record]) {
            result.reachable = false;
        }
    } else if (forced_record_mask_[action.record] != 0) {
        assert(records_[action.record] == forced_record_values_[action.record] &&
               "forced continuation classical record must match its deterministic value");
        forced_record_mask_[action.record] = 0;
    }
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    assign_symbol(action.symbol, evaluate(action.value));
}

template <bool ForceRecords, bool ForceFaults>
void Executor::execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                              std::span<const uint8_t>, ReplayResult& result) noexcept {
    if constexpr (ForceRecords) {
        result.reachable = false;
    } else {
        const bool source = evaluate(action.source);
        assert(records_[action.record] == static_cast<uint8_t>(source) &&
               "readout source must match the current record value");
        bool flip = false;
        if constexpr (ForceFaults) {
            const uint32_t fault_site =
                static_cast<uint32_t>(plan_->noise_sites_.size()) + action.site;
            assert((forced_fault_cursor_ >= forced_fault_sites_.size() ||
                    forced_fault_sites_[forced_fault_cursor_] >= fault_site) &&
                   "conditioned readout sites must be consumed in circuit order");
            if (forced_fault_cursor_ < forced_fault_sites_.size() &&
                forced_fault_sites_[forced_fault_cursor_] == fault_site) {
                flip = true;
                ++forced_fault_cursor_;
            }
        } else {
            const double probability = source ? action.prob_one_to_zero : action.prob_zero_to_one;
            flip = probability >= 1.0 || (probability > 0.0 && rng_.next_double() < probability);
        }
        assign_symbol(action.flip, flip);
        records_[action.record] ^= static_cast<uint8_t>(flip);
    }
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteDetector& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    const bool outcome = evaluate(action.outcome);
    detectors_[action.detector] = static_cast<uint8_t>(outcome);
    discarded_ |= action.postselected && outcome;
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteObservable& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    observables_[action.observable] = static_cast<uint8_t>(evaluate(action.outcome));
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteExpectation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    assert(action.exp_val < exp_vals_.size() && "expectation slot must be preallocated");
    if (!action.active_projection.has_value()) {
        // Outputs are overwritten instead of cleared at each shot. This store
        // also prevents stale values when a continuation changes the planner's
        // classification of the same probe from active to exact zero.
        exp_vals_[action.exp_val] = 0.0;
        return;
    }
    const double value = expectation_value(state_, *action.active_projection);
    exp_vals_[action.exp_val] = evaluate(action.sign) ? -value : value;
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteInstrument& action,
                              std::span<const uint8_t>, ReplayResult& result) noexcept {
    if constexpr (ForceRecords) {
        result.reachable = false;
        return;
    }

    assert(action.site < plan_->instrument_distributions_.size() &&
           "instrument action must reference a prepared distribution");
    const InstrumentDistribution& distribution = plan_->instrument_distributions_[action.site];
    if (action.destination_flip.has_value()) {
        assign_symbol(*action.destination_flip, false);
    }

    auto trap = [&](uint8_t source, bool destination_pending) {
        pending_trap_ = InstrumentTrap{InstrumentSiteId{action.site}, source, destination_pending};
    };
    auto finish_fire = [&](uint8_t source) {
        assert(distribution.p_fire[source] > 0.0 &&
               "a fired instrument source must have positive mass");
        const double draw = rng_.next_double() * distribution.p_fire[source];
        int destination = -1;
        if (draw < distribution.p_computational_dest[source][0]) {
            destination = 0;
        } else if (draw < distribution.p_computational_dest[source][0] +
                              distribution.p_computational_dest[source][1]) {
            destination = 1;
        }
        if (destination < 0) {
            trap(source, false);
        } else if (destination != source) {
            assert(action.destination_flip.has_value() &&
                   "in-line instrument requires a destination-flip symbol");
            assign_symbol(*action.destination_flip, true);
        }
    };

    if (action.mode == InstrumentMode::Classical) {
        const uint8_t source = static_cast<uint8_t>(evaluate(action.sign));
        if (rng_.next_double() < distribution.p_fire[source]) {
            finish_fire(source);
        }
        return;
    }

    if (action.mode == InstrumentMode::DormantTrap) {
        const double mass = distribution.p_fire[0] + distribution.p_fire[1];
        if (rng_.next_double() * 2.0 >= mass) {
            return;
        }
        const uint8_t source =
            rng_.next_double() * mass < distribution.p_fire[0] ? uint8_t{0} : uint8_t{1};
        trap(source, true);
        return;
    }

    assert(action.measurement.has_value() &&
           "active instrument requires a prepared source measurement");
    if (action.mode == InstrumentMode::Activate) {
        activate_zero_coordinate(state_);
    }
    const bool sign = evaluate(action.sign);
    const MeasurementProbabilities eigen_populations =
        measurement_probabilities(state_, *action.measurement);
    const double total = eigen_populations.total();
    const double epsilon = kMeasurementDustEpsilon * total;
    const double physical_zero = eigen_populations.for_branch(sign);
    const double physical_one = eigen_populations.for_branch(!sign);
    const double fire_zero =
        physical_zero <= epsilon ? 0.0 : distribution.p_fire[0] * physical_zero;
    const double fire_one = physical_one <= epsilon ? 0.0 : distribution.p_fire[1] * physical_one;
    const double draw = rng_.next_double() * total;
    std::optional<uint8_t> fired_source;
    if (draw < fire_zero) {
        fired_source = 0;
    } else if (draw < fire_zero + fire_one) {
        fired_source = 1;
    }

    if (!fired_source.has_value()) {
        const double factor_zero = std::sqrt(1.0 - distribution.p_fire[static_cast<uint8_t>(sign)]);
        const double factor_one = std::sqrt(1.0 - distribution.p_fire[static_cast<uint8_t>(!sign)]);
        const double no_fire_probability = factor_zero * factor_zero * eigen_populations.zero +
                                           factor_one * factor_one * eigen_populations.one;
        assert(no_fire_probability > 0.0 &&
               "a selected no-fire branch must have positive probability");
        apply_instrument_no_fire(state_, action.measurement->pauli, factor_zero, factor_one,
                                 no_fire_probability);
        return;
    }

    const bool eigen_branch = (*fired_source != 0) ^ sign;
    collapse_instrument_source(state_, action.measurement->pauli, eigen_branch,
                               eigen_populations.for_branch(eigen_branch));
    finish_fire(*fired_source);
}

template <bool SampleNoise>
void Executor::execute_action(const ExecutablePlan::ExecuteBoundary& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    if constexpr (SampleNoise) {
        sample_presampled_noise(action.noise_begin, action.noise_end);
    }
}

template <bool ForceRecords, bool SampleNoise, bool ForceFaults>
ReplayResult Executor::execute_actions(std::span<const uint8_t> forced_records,
                                       uint32_t begin) noexcept {
    ReplayResult result;
    assert(begin <= plan_->actions_.size() && "execution offset must be inside the action stream");
    for (size_t action_index = begin; action_index < plan_->actions_.size(); ++action_index) {
        const ExecutablePlan::Action& action = plan_->actions_[action_index];
        std::visit(
            [&](const auto& typed) noexcept {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteBoundary>) {
                    execute_action<SampleNoise>(typed, forced_records, result);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise>) {
                    execute_action<ForceRecords, ForceFaults>(typed, forced_records, result);
                } else {
                    execute_action<ForceRecords>(typed, forced_records, result);
                }
            },
            action);
        if constexpr (ForceRecords) {
            if (!result.reachable) {
                return result;
            }
        }
        if (discarded_) {
            return result;
        }
        if (pending_trap_.has_value()) {
            return result;
        }
    }
    return result;
}

bool Executor::evaluate(ExecutablePlan::PreparedExpression expression) const noexcept {
    assert(expression.register_id < plan_->expression_register_constants_.size() &&
           expression.register_id < expression_registers_.size() &&
           "prepared affine expression must refer to the active register file");
    return expression_registers_[expression.register_id] != 0;
}

bool Executor::sample_active_branch(MeasurementProbabilities probabilities) noexcept {
    const MeasurementBranchClassification classification =
        classify_measurement_branch(probabilities);
    switch (classification.kind) {
        case MeasurementBranchKind::Random:
            return rng_.next_double() * probabilities.total() >= probabilities.zero;
        case MeasurementBranchKind::DeterministicZero:
            dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
            return false;
        case MeasurementBranchKind::DeterministicOne:
            dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
            return true;
    }
    assert(false && "unhandled measurement branch classification");
    return false;
}

std::optional<double> Executor::force_active_branch(MeasurementProbabilities probabilities,
                                                    bool branch) noexcept {
    const MeasurementBranchClassification classification =
        classify_measurement_branch(probabilities);
    dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
    switch (classification.kind) {
        case MeasurementBranchKind::Random:
            return std::log(probabilities.for_branch(branch) / probabilities.total());
        case MeasurementBranchKind::DeterministicZero:
            return branch ? std::nullopt : std::optional<double>{0.0};
        case MeasurementBranchKind::DeterministicOne:
            return branch ? std::optional<double>{0.0} : std::nullopt;
    }
    assert(false && "unhandled measurement branch classification");
    return std::nullopt;
}

bool Executor::sample_dormant_branch() noexcept {
    return rng_.next_double() >= 0.5;
}

namespace {

template <typename RunShot>
SamplingResult sample_fixed_rows(const ExecutablePlan& plan, uint32_t shots,
                                 std::optional<uint64_t> seed, RunShot&& run_shot) {
    auto checked_size = [shots](size_t stride) {
        if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
            throw std::length_error("sampling output size exceeds size_t range");
        }
        return static_cast<size_t>(shots) * stride;
    };

    SamplingResult result;
    result.measurements.resize(checked_size(plan.num_visible_records()));
    result.detectors.resize(checked_size(plan.num_detectors()));
    result.observables.resize(checked_size(plan.num_observables()));
    result.exp_vals.resize(checked_size(plan.num_exp_vals()));
    if (shots == 0) {
        return result;
    }

    auto run = [&](Executor& executor) {
        for (uint32_t shot = 0; shot < shots; ++shot) {
            run_shot(executor);
            std::ranges::copy(executor.visible_records(),
                              result.measurements.begin() +
                                  static_cast<size_t>(shot) * plan.num_visible_records());
            std::ranges::copy(
                executor.detectors(),
                result.detectors.begin() + static_cast<size_t>(shot) * plan.num_detectors());
            std::ranges::copy(
                executor.observables(),
                result.observables.begin() + static_cast<size_t>(shot) * plan.num_observables());
            std::ranges::copy(
                executor.exp_vals(),
                result.exp_vals.begin() + static_cast<size_t>(shot) * plan.num_exp_vals());
        }
    };
    if (seed.has_value()) {
        Executor executor(plan, *seed);
        run(executor);
    } else {
        Executor executor(plan);
        executor.reseed_from_entropy();
        run(executor);
    }
    return result;
}

template <typename RunShot>
SamplingSurvivorResult sample_surviving_rows(const ExecutablePlan& plan, uint32_t shots,
                                             std::optional<uint64_t> seed, bool keep_records,
                                             RunShot&& run_shot) {
    SamplingSurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(plan.num_observables(), 0);
    if (keep_records) {
        auto checked_reserve = [shots](size_t stride) {
            if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
                throw std::length_error("survivor output size exceeds size_t range");
            }
            return static_cast<size_t>(shots) * stride;
        };
        result.measurements.reserve(checked_reserve(plan.num_visible_records()));
        result.detectors.reserve(checked_reserve(plan.num_detectors()));
        result.observables.reserve(checked_reserve(plan.num_observables()));
        result.exp_vals.reserve(checked_reserve(plan.num_exp_vals()));
    }
    auto run = [&](Executor& executor) {
        for (uint32_t shot = 0; shot < shots; ++shot) {
            run_shot(executor);
            if (executor.discarded()) {
                continue;
            }
            ++result.passed_shots;
            bool logical_error = false;
            for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
                const bool value = executor.observables()[observable] != 0;
                result.observable_ones[observable] += static_cast<uint64_t>(value);
                logical_error |= value;
            }
            result.logical_errors += static_cast<uint32_t>(logical_error);
            if (keep_records) {
                result.measurements.insert(result.measurements.end(),
                                           executor.visible_records().begin(),
                                           executor.visible_records().end());
                result.detectors.insert(result.detectors.end(), executor.detectors().begin(),
                                        executor.detectors().end());
                result.observables.insert(result.observables.end(), executor.observables().begin(),
                                          executor.observables().end());
                result.exp_vals.insert(result.exp_vals.end(), executor.exp_vals().begin(),
                                       executor.exp_vals().end());
            }
        }
    };
    if (seed.has_value()) {
        Executor executor(plan, *seed);
        run(executor);
    } else {
        Executor executor(plan);
        executor.reseed_from_entropy();
        run(executor);
    }
    return result;
}

}  // namespace

SamplingResult sample(const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "fixed-plan sampling does not support instrument traps; use the trajectory driver");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "batch sampling requires a distribution for every presampled symbol");
    }
    if (plan.has_postselection()) {
        throw std::invalid_argument(
            "fixed-row sampling does not support postselection; use sample_survivors");
    }

    return sample_fixed_rows(plan, shots, seed,
                             [](Executor& executor) noexcept { executor.run_shot(); });
}

std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed) {
    return sample(plan, shots, seed).measurements;
}

SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                        std::optional<uint64_t> seed, bool keep_records) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "survivor sampling does not support instrument traps; use the trajectory driver");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "survivor sampling requires a distribution for every presampled symbol");
    }

    return sample_surviving_rows(plan, shots, seed, keep_records,
                                 [](Executor& executor) noexcept { executor.run_shot(); });
}

SamplingResult sample_k(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                        std::optional<uint64_t> seed) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "forced-fault sampling does not support instrument traps or trajectory drivers");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "forced-fault sampling requires a distribution for every presampled symbol");
    }
    if (plan.has_postselection()) {
        throw std::invalid_argument(
            "fixed-row forced-fault sampling does not support postselection; use "
            "sample_k_survivors");
    }
    if (shots == 0) {
        return sample_fixed_rows(plan, shots, seed,
                                 [](Executor& executor) noexcept { executor.run_shot(); });
    }
    KFaultSampler fault_sampler(plan.noise_site_probabilities(), k);
    return sample_fixed_rows(
        plan, shots, seed, [&](Executor& executor) noexcept { executor.run_shot(fault_sampler); });
}

SamplingSurvivorResult sample_k_survivors(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                                          std::optional<uint64_t> seed, bool keep_records) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "forced-fault survivor sampling does not support instrument traps or trajectory "
            "drivers");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "forced-fault survivor sampling requires a distribution for every presampled symbol");
    }
    if (shots == 0) {
        return sample_surviving_rows(plan, shots, seed, keep_records,
                                     [](Executor& executor) noexcept { executor.run_shot(); });
    }
    KFaultSampler fault_sampler(plan.noise_site_probabilities(), k);
    return sample_surviving_rows(plan, shots, seed, keep_records, [&](Executor& executor) noexcept {
        executor.run_shot(fault_sampler);
    });
}

std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                             std::span<const uint8_t> forced_records,
                                             size_t num_records) {
    if (plan.has_instruments()) {
        throw std::invalid_argument("record probabilities do not yet support instruments");
    }
    if (plan.num_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "record probabilities do not yet support plans with presampled symbols");
    }
    if (plan.has_readout_noise()) {
        throw std::invalid_argument("record probabilities do not yet support readout noise");
    }
    if (plan.num_hidden_records() != 0) {
        throw std::invalid_argument(
            "record probabilities do not yet support plans with hidden records");
    }
    const size_t stride = plan.num_visible_records();
    if (stride == 0) {
        throw std::invalid_argument(
            "record probabilities require a plan with at least one visible record");
    }
    if (num_records > std::numeric_limits<size_t>::max() / stride ||
        forced_records.size() != num_records * stride) {
        throw std::invalid_argument(
            "record buffer length must equal num_records times visible records");
    }
    if (!std::ranges::all_of(forced_records, [](uint8_t value) { return value <= 1; })) {
        throw std::invalid_argument("record bytes must be Boolean");
    }

    std::vector<double> log_probabilities(num_records);
    Executor executor(plan);
    for (size_t record = 0; record < num_records; ++record) {
        const ReplayResult replay =
            executor.replay_shot(forced_records.subspan(record * stride, stride));
        log_probabilities[record] =
            replay.reachable ? replay.log_probability : std::numeric_limits<double>::lowest();
    }
    return log_probabilities;
}

}  // namespace clifft::sampling
