#include "clifft/sampling/executor.h"

#include "clifft/util/fault_sampling.h"
#include "clifft/util/intra_shot_parallel.h"
#include "clifft/util/noise_sampling.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace clifft::sampling {

namespace {

constexpr double kLogHalf = -std::numbers::ln2;

// Zero and one identify the selected Pauli eigenvalue branch before affine
// corrections turn that branch into a physical measurement record.
enum class MeasurementBranchKind : uint8_t {
    Random,
    DeterministicZero,
    DeterministicOne,
};

struct MeasurementBranchClassification {
    MeasurementBranchKind kind = MeasurementBranchKind::Random;
    bool clamped_dust = false;
};

uint32_t resolve_intra_shot_workers(const ExecutablePlan& plan, uint32_t requested_workers,
                                    uint32_t min_active_width) {
    if (requested_workers == 0 ||
        requested_workers > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("sampling executor intra-shot worker count is invalid");
    }
    return should_parallelize_intra_shot(plan.peak_active_width(), requested_workers,
                                         min_active_width)
               ? requested_workers
               : 1;
}

[[nodiscard]] MeasurementBranchClassification classify_measurement_branch(
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

}  // namespace

Executor::Executor(const ExecutablePlan& plan, uint64_t seed, uint32_t intra_shot_workers,
                   uint32_t intra_shot_min_active_width)
    : root_plan_(&plan),
      plan_(&plan),
      state_(plan.peak_active_width_, plan.initial_active_width_,
             resolve_intra_shot_workers(plan, intra_shot_workers, intra_shot_min_active_width),
             intra_shot_min_active_width),
      symbols_(plan.num_symbols_, 0),
      expression_registers_(plan.expression_register_constants_),
      records_(static_cast<size_t>(plan.num_visible_records_) + plan.num_hidden_records_, 0),
      detectors_(plan.num_detectors_, 0),
      observables_(plan.num_observables_, 0),
      exp_vals_(plan.num_exp_vals_, 0.0),
      forced_record_mask_(records_.size(), 0),
      forced_record_values_(records_.size(), 0),
      rng_(seed),
      backend_(plan.backend_),
      intra_shot_workers_(
          resolve_intra_shot_workers(plan, intra_shot_workers, intra_shot_min_active_width)),
      intra_shot_min_active_width_(intra_shot_min_active_width) {
    previous_presampled_ones_.reserve(plan.presampled_symbols_.size());
}

void Executor::run_shot() noexcept {
    plan_ = root_plan_;
    assert(plan_->unbound_presampled_symbols_.empty() &&
           "automatic execution requires every presampled symbol to have a distribution");
    reset_shot();
    sample_presampled_noise(0, plan_->initial_noise_end_);
    (void)execute_actions_for_backend<ShotMode::SampleNoise>({});
}

void Executor::run_shot(std::span<const uint8_t> presampled_values) noexcept {
    plan_ = root_plan_;
    reset_shot();
    assign_presampled_values(presampled_values);
    (void)execute_actions_for_backend<ShotMode::UsePresampledNoise>({});
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
    (void)execute_actions_for_backend<ShotMode::FixedFaultCount>({});
    forced_fault_sites_ = {};
    forced_fault_cursor_ = 0;
}

void Executor::resume(const ExecutablePlan& continuation,
                      std::optional<ForcedTraceOut> forced_trace_out) {
    if (!pending_trap_.has_value()) {
        throw std::invalid_argument("sampling executor resume requires a pending instrument trap");
    }
    if (continuation.backend_ != backend_) {
        throw std::invalid_argument(
            "sampling continuation uses a different executor backend than the root plan");
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

    state_.ensure_capacity(continuation.peak_active_width_);
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
    (void)execute_actions_for_backend<ShotMode::SampleNoise>({}, offset);
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
    return execute_actions_for_backend<ShotMode::ReplayRecords>(forced_records);
}

void Executor::reset_shot() noexcept {
    state_.reset_parallel(intra_shot_workers_, intra_shot_min_active_width_);
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
    for (uint32_t register_id : plan.expression_dependencies_.dependent_registers(symbol)) {
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

template <ExecutorBackend Backend, Executor::IntraShotMode IntraShot>
void Executor::execute_action(const ExecutablePlan::ExecuteRotation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    const bool sign = evaluate(action.sign);
    if (action.kernel == DirectRotationKernel::Scalar) {
        if constexpr (IntraShot == IntraShotMode::OpenMP) {
            apply_rotation_parallel(state_, action.rotation, sign, intra_shot_workers_,
                                    intra_shot_min_active_width_);
        } else {
            apply_rotation(state_, action.rotation, sign);
        }
        return;
    }
    if constexpr (Backend == ExecutorBackend::Avx2) {
        if constexpr (IntraShot == IntraShotMode::OpenMP) {
            apply_direct_rotation_avx2_parallel(state_, action.rotation, action.kernel, sign,
                                                intra_shot_workers_, intra_shot_min_active_width_);
        } else {
            apply_direct_rotation_avx2(state_, action.rotation, action.kernel, sign);
        }
    } else if constexpr (Backend == ExecutorBackend::Avx512) {
        if constexpr (IntraShot == IntraShotMode::OpenMP) {
            apply_direct_rotation_avx512_parallel(state_, action.rotation, action.kernel, sign,
                                                  intra_shot_workers_,
                                                  intra_shot_min_active_width_);
        } else {
            apply_direct_rotation_avx512(state_, action.rotation, action.kernel, sign);
        }
    } else {
        assert(false && "scalar executor requires scalar direct-rotation actions");
        apply_rotation(state_, action.rotation, sign);
    }
}

template <Executor::IntraShotMode IntraShot>
void Executor::execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    assert(action.rotation_index < plan_->fused_rotations_.size() &&
           "fused rotation action must reference prepared execution");
    if constexpr (IntraShot == IntraShotMode::OpenMP) {
        plan_->fused_rotations_[action.rotation_index].apply_parallel(state_, intra_shot_workers_,
                                                                      intra_shot_min_active_width_);
    } else {
        plan_->fused_rotations_[action.rotation_index].apply(state_);
    }
}

template <Executor::IntraShotMode IntraShot>
void Executor::execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    assert(action.rotation_index < plan_->dynamic_fused_rotations_.size() &&
           "dynamic fused rotation action must reference prepared execution");
    const auto& rotation = plan_->dynamic_fused_rotations_[action.rotation_index];
    uint32_t variant = 0;
    for (size_t i = 0; i < rotation.sign_basis.size(); ++i) {
        variant |= static_cast<uint32_t>(evaluate(rotation.sign_basis[i])) << i;
    }
    assert(variant < rotation.variants.size() &&
           "dynamic fused sign value must select a prepared variant");
    if constexpr (IntraShot == IntraShotMode::OpenMP) {
        rotation.variants[variant].apply_parallel(state_, intra_shot_workers_,
                                                  intra_shot_min_active_width_);
    } else {
        rotation.variants[variant].apply(state_);
    }
}

template <Executor::IntraShotMode IntraShot>
void Executor::execute_action(const ExecutablePlan::ExecutePromotion& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    if constexpr (IntraShot == IntraShotMode::OpenMP) {
        apply_promotion_parallel(state_, action.promotion, evaluate(action.sign),
                                 intra_shot_workers_, intra_shot_min_active_width_);
    } else {
        apply_promotion(state_, action.promotion, evaluate(action.sign));
    }
}

template <ExecutorBackend Backend, Executor::ShotMode Mode>
void Executor::execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    const MeasurementProbabilities probabilities = [&]() noexcept {
        if (action.kernel == ActiveMeasurementKernel::Scalar) {
            return measurement_probabilities(state_, action.measurement);
        }
        if constexpr (Backend == ExecutorBackend::Avx2) {
            return active_measurement_probabilities_avx2(state_, action.measurement, action.kernel);
        } else if constexpr (Backend == ExecutorBackend::Avx512) {
            return active_measurement_probabilities_avx512(state_, action.measurement,
                                                           action.kernel);
        } else {
            assert(false && "scalar executor requires scalar active-measurement actions");
            return measurement_probabilities(state_, action.measurement);
        }
    }();
    const bool correction = evaluate(action.correction);
    bool branch = false;
    if constexpr (Mode == ShotMode::ReplayRecords) {
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
    const double branch_probability = probabilities.for_branch(branch);
    if (action.kernel == ActiveMeasurementKernel::Scalar) {
        collapse_measurement(state_, action.measurement, branch, branch_probability);
    } else if constexpr (Backend == ExecutorBackend::Avx2) {
        collapse_active_measurement_avx2(state_, action.measurement, action.kernel, branch,
                                         branch_probability);
    } else if constexpr (Backend == ExecutorBackend::Avx512) {
        collapse_active_measurement_avx512(state_, action.measurement, action.kernel, branch,
                                           branch_probability);
    } else {
        assert(false && "scalar executor requires scalar active-measurement actions");
        collapse_measurement(state_, action.measurement, branch, branch_probability);
    }
    records_[action.record] = static_cast<uint8_t>(branch ^ correction);
}

template <Executor::ShotMode Mode>
void Executor::execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    const bool correction = evaluate(action.correction);
    bool branch = false;
    if constexpr (Mode == ShotMode::ReplayRecords) {
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

template <Executor::ShotMode Mode>
void Executor::execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                              std::span<const uint8_t> forced_records,
                              ReplayResult& result) noexcept {
    records_[action.record] = static_cast<uint8_t>(evaluate(action.outcome));
    if constexpr (Mode == ShotMode::ReplayRecords) {
        if (records_[action.record] != forced_records[action.record]) {
            result.reachable = false;
        }
    } else if (forced_record_mask_[action.record] != 0) {
        assert(records_[action.record] == forced_record_values_[action.record] &&
               "forced continuation classical record must match its deterministic value");
        forced_record_mask_[action.record] = 0;
    }
}

void Executor::execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    assign_symbol(action.symbol, evaluate(action.value));
}

template <Executor::ShotMode Mode>
void Executor::execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                              std::span<const uint8_t>, ReplayResult& result) noexcept {
    if constexpr (Mode == ShotMode::ReplayRecords) {
        result.reachable = false;
    } else {
        const bool source = evaluate(action.source);
        assert(records_[action.record] == static_cast<uint8_t>(source) &&
               "readout source must match the current record value");
        bool flip = false;
        if constexpr (Mode == ShotMode::FixedFaultCount) {
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

void Executor::execute_action(const ExecutablePlan::ExecuteDetector& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    const bool outcome = evaluate_syndrome(action.outcome);
    detectors_[action.detector] = static_cast<uint8_t>(outcome);
    discarded_ |= action.postselected && outcome;
}

void Executor::execute_action(const ExecutablePlan::ExecuteObservable& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    observables_[action.observable] = static_cast<uint8_t>(evaluate_syndrome(action.outcome));
}

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

void Executor::trap_instrument(uint32_t site, uint8_t source, bool destination_pending) noexcept {
    pending_trap_ = InstrumentTrap{InstrumentSiteId{site}, source, destination_pending};
}

void Executor::finish_instrument_fire(uint32_t site, uint32_t destination_flip, uint8_t source,
                                      const InstrumentDistribution& distribution) noexcept {
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
        trap_instrument(site, source, false);
    } else if (destination != source) {
        assign_symbol(destination_flip, true);
    }
}

void Executor::execute_instrument(
    const ExecutablePlan::ExecuteClassicalInstrument& action) noexcept {
    assert(action.site < plan_->instrument_distributions_.size() &&
           "instrument action must reference a prepared distribution");
    const InstrumentDistribution& distribution = plan_->instrument_distributions_[action.site];
    assign_symbol(action.destination_flip, false);
    const uint8_t source = static_cast<uint8_t>(evaluate(action.sign));
    if (rng_.next_double() < distribution.p_fire[source]) {
        finish_instrument_fire(action.site, action.destination_flip, source, distribution);
    }
}

void Executor::execute_instrument(
    const ExecutablePlan::ExecuteDormantInstrumentTrap& action) noexcept {
    assert(action.site < plan_->instrument_distributions_.size() &&
           "instrument action must reference a prepared distribution");
    const InstrumentDistribution& distribution = plan_->instrument_distributions_[action.site];
    const double mass = distribution.p_fire[0] + distribution.p_fire[1];
    if (rng_.next_double() * 2.0 >= mass) {
        return;
    }
    const uint8_t source =
        rng_.next_double() * mass < distribution.p_fire[0] ? uint8_t{0} : uint8_t{1};
    trap_instrument(action.site, source, true);
}

template <ExecutorBackend Backend, typename Action>
void Executor::execute_quantum_instrument(const Action& action) noexcept {
    constexpr bool kActivatesCoordinate =
        std::is_same_v<Action, ExecutablePlan::ExecuteMeasuredInstrumentActivation>;
    constexpr bool kActivatesNewX =
        std::is_same_v<Action, ExecutablePlan::ExecuteNewXInstrumentActivation>;
    static_assert(std::is_same_v<Action, ExecutablePlan::ExecuteActiveInstrument> ||
                  kActivatesCoordinate || kActivatesNewX);

    assert(action.site < plan_->instrument_distributions_.size() &&
           "instrument action must reference a prepared distribution");
    const InstrumentDistribution& distribution = plan_->instrument_distributions_[action.site];
    assign_symbol(action.destination_flip, false);
    if constexpr (kActivatesCoordinate) {
        activate_zero_coordinate(state_);
    }
    const bool sign = evaluate(action.sign);
    const MeasurementProbabilities eigen_populations = [&]() noexcept {
        if constexpr (kActivatesNewX) {
            // Under the State normalization invariant, the clean |0> coordinate has
            // exact half populations in the X eigenbasis. Using those semantic values
            // retains any residual floating-point norm drift instead of reducing and
            // renormalizing it inside this activation.
            return MeasurementProbabilities{0.5, 0.5};
        } else {
            return measurement_probabilities(state_, action.measurement);
        }
    }();
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
        if constexpr (kActivatesNewX) {
            if (action.kernel == NewXInstrumentKernel::Scalar) {
                apply_new_x_instrument_no_fire(state_, factor_zero, factor_one,
                                               no_fire_probability);
            } else if constexpr (Backend == ExecutorBackend::Avx2 ||
                                 Backend == ExecutorBackend::Avx512) {
                apply_new_x_instrument_no_fire_avx2(state_, factor_zero, factor_one,
                                                    no_fire_probability);
            } else {
                assert(false && "scalar executor requires scalar new-X instrument actions");
                apply_new_x_instrument_no_fire(state_, factor_zero, factor_one,
                                               no_fire_probability);
            }
        } else {
            apply_instrument_no_fire(state_, action.measurement.pauli, factor_zero, factor_one,
                                     no_fire_probability);
        }
        return;
    }

    const bool eigen_branch = (*fired_source != 0) ^ sign;
    if constexpr (kActivatesNewX) {
        collapse_new_x_instrument_source(state_, eigen_branch,
                                         eigen_populations.for_branch(eigen_branch));
    } else {
        collapse_instrument_source(state_, action.measurement.pauli, eigen_branch,
                                   eigen_populations.for_branch(eigen_branch));
    }
    finish_instrument_fire(action.site, action.destination_flip, *fired_source, distribution);
}

template <ExecutorBackend Backend>
void Executor::execute_instrument(const ExecutablePlan::ExecuteActiveInstrument& action) noexcept {
    execute_quantum_instrument<Backend>(action);
}

template <ExecutorBackend Backend>
void Executor::execute_instrument(
    const ExecutablePlan::ExecuteMeasuredInstrumentActivation& action) noexcept {
    execute_quantum_instrument<Backend>(action);
}

template <ExecutorBackend Backend>
void Executor::execute_instrument(
    const ExecutablePlan::ExecuteNewXInstrumentActivation& action) noexcept {
    execute_quantum_instrument<Backend>(action);
}

template <ExecutorBackend Backend, Executor::ShotMode Mode>
void Executor::execute_action(const ExecutablePlan::ExecuteInstrument& action,
                              std::span<const uint8_t>, ReplayResult& result) noexcept {
    if constexpr (Mode == ShotMode::ReplayRecords) {
        result.reachable = false;
    } else {
        std::visit(
            [&](const auto& instrument) noexcept {
                using T = std::decay_t<decltype(instrument)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteClassicalInstrument> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteDormantInstrumentTrap>) {
                    execute_instrument(instrument);
                } else {
                    execute_instrument<Backend>(instrument);
                }
            },
            action.form);
    }
}

template <Executor::ShotMode Mode>
void Executor::execute_action(const ExecutablePlan::ExecuteBoundary& action,
                              std::span<const uint8_t>, ReplayResult&) noexcept {
    if constexpr (Mode == ShotMode::SampleNoise) {
        sample_presampled_noise(action.noise_begin, action.noise_end);
    }
}

template <ExecutorBackend Backend, Executor::ShotMode Mode, Executor::IntraShotMode IntraShot>
ReplayResult Executor::execute_actions(std::span<const uint8_t> forced_records,
                                       uint32_t begin) noexcept {
    ReplayResult result;
    assert(begin <= plan_->actions_.size() && "execution offset must be inside the action stream");
    for (size_t action_index = begin; action_index < plan_->actions_.size(); ++action_index) {
        const ExecutablePlan::Action& action = plan_->actions_[action_index];
        std::visit(
            [&](const auto& typed) noexcept {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteBoundary> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteDormantMeasurement> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteClassicalRecord>) {
                    execute_action<Mode>(typed, forced_records, result);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation>) {
                    execute_action<Backend, IntraShot>(typed, forced_records, result);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFusedRotation> ||
                                     std::is_same_v<T,
                                                    ExecutablePlan::ExecuteDynamicFusedRotation> ||
                                     std::is_same_v<T, ExecutablePlan::ExecutePromotion>) {
                    execute_action<IntraShot>(typed, forced_records, result);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement> ||
                                     std::is_same_v<T, ExecutablePlan::ExecuteInstrument>) {
                    execute_action<Backend, Mode>(typed, forced_records, result);
                } else {
                    execute_action(typed, forced_records, result);
                }
            },
            action);
        if constexpr (Mode == ShotMode::ReplayRecords) {
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

template <Executor::ShotMode Mode>
ReplayResult Executor::execute_actions_for_backend(std::span<const uint8_t> forced_records,
                                                   uint32_t begin) noexcept {
    if (intra_shot_workers_ > 1) {
        switch (backend_) {
            case ExecutorBackend::Scalar:
                return execute_actions<ExecutorBackend::Scalar, Mode, IntraShotMode::OpenMP>(
                    forced_records, begin);
            case ExecutorBackend::Avx2:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
                return execute_actions<ExecutorBackend::Avx2, Mode, IntraShotMode::OpenMP>(
                    forced_records, begin);
#else
                break;
#endif
            case ExecutorBackend::Avx512:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
                return execute_actions<ExecutorBackend::Avx512, Mode, IntraShotMode::OpenMP>(
                    forced_records, begin);
#else
                break;
#endif
        }
        assert(false && "unhandled sampling executor backend");
        return {};
    }
    switch (backend_) {
        case ExecutorBackend::Scalar:
            return execute_actions<ExecutorBackend::Scalar, Mode, IntraShotMode::Serial>(
                forced_records, begin);
        case ExecutorBackend::Avx2:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
            return execute_actions<ExecutorBackend::Avx2, Mode, IntraShotMode::Serial>(
                forced_records, begin);
#else
            break;
#endif
        case ExecutorBackend::Avx512:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
            return execute_actions<ExecutorBackend::Avx512, Mode, IntraShotMode::Serial>(
                forced_records, begin);
#else
            break;
#endif
    }
    assert(false && "unhandled sampling executor backend");
    return {};
}

bool Executor::evaluate(ExecutablePlan::PreparedExpression expression) const noexcept {
    assert(expression.register_id < plan_->expression_register_constants_.size() &&
           expression.register_id < expression_registers_.size() &&
           "prepared affine expression must refer to the active register file");
    return expression_registers_[expression.register_id] != 0;
}

bool Executor::evaluate_syndrome(
    const ExecutablePlan::PreparedSyndromeValue& value) const noexcept {
    if (const auto* expression = std::get_if<ExecutablePlan::PreparedExpression>(&value)) {
        return evaluate(*expression);
    }
    const ExecutablePlan::PreparedRecordParity parity =
        std::get<ExecutablePlan::PreparedRecordParity>(value);
    const size_t end = static_cast<size_t>(parity.begin) + parity.count;
    assert(end <= plan_->record_parity_terms_.size() &&
           "prepared record parity must stay in its term tape");
    bool outcome = parity.constant;
    for (size_t term = parity.begin; term < end; ++term) {
        outcome ^= records_[plan_->record_parity_terms_[term]] != 0;
    }
    return outcome;
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
