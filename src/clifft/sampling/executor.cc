#include "clifft/sampling/executor.h"

#include "clifft/util/fault_sampling.h"
#include "clifft/util/noise_sampling.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>

namespace clifft::sampling {

namespace {

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
    assert(site.outcome_count > 0 &&
           "a firing noise site must contain positive-probability outcomes");
    const uint32_t outcome_end = site.outcome_begin + site.outcome_count;
    assert(outcome_end <= plan_->noise_outcomes_.size() &&
           plan_->noise_outcomes_[outcome_end - 1].cumulative_probability > 0.0 &&
           "a firing noise site must contain positive-probability outcomes");
    uint32_t outcome_index = site.outcome_begin;
    if (site.outcome_count > 1) {
        const double execution_probability =
            plan_->noise_outcomes_[outcome_end - 1].cumulative_probability;
        const double channel_draw = rng_.next_double() * execution_probability;
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
    action.apply(state_, evaluate(action.sign));
}

template <bool ForceRecords>
void Executor::execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    assert(action.rotation_index < plan_->fused_rotations_.size() &&
           "fused rotation action must reference prepared execution");
    plan_->fused_rotations_[action.rotation_index].apply(state_);
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

}  // namespace clifft::sampling
