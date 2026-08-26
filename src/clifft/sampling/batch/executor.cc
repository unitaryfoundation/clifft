#include "clifft/sampling/batch/executor.h"

#include "clifft/util/fault_sampling.h"
#include "clifft/util/noise_sampling.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace clifft::sampling {

namespace {

enum class MeasurementBranchKind : uint8_t {
    Random,
    DeterministicZero,
    DeterministicOne,
};

[[nodiscard]] MeasurementBranchKind classify_measurement_branch(
    MeasurementProbabilities probabilities) noexcept {
    const double total = probabilities.total();
    assert(is_finite_robust(probabilities.zero) && probabilities.zero >= 0.0 &&
           is_finite_robust(probabilities.one) && probabilities.one >= 0.0 &&
           is_finite_robust(total) && total > 0.0 &&
           "measurement probabilities must be finite, nonnegative, and nonzero");
    const double epsilon = kMeasurementDustEpsilon * total;
    if (probabilities.one <= epsilon) {
        return MeasurementBranchKind::DeterministicZero;
    }
    if (probabilities.zero <= epsilon) {
        return MeasurementBranchKind::DeterministicOne;
    }
    return MeasurementBranchKind::Random;
}

[[nodiscard]] const ExecutablePlan* validate_batch_plan(const ExecutablePlan& plan) {
    if (plan.has_instruments()) {
        throw std::invalid_argument("packed sampling does not support instrument continuations");
    }
    return &plan;
}

[[nodiscard]] size_t checked_size(uint64_t entries, const char* description) {
    if (entries > std::numeric_limits<size_t>::max()) {
        throw std::length_error(std::string("packed batch ") + description +
                                " allocation exceeds size_t range");
    }
    return static_cast<size_t>(entries);
}

}  // namespace

BatchExecutor::BatchExecutor(const ExecutablePlan& plan, uint32_t lane_capacity,
                             BatchOutputMode output_mode, BatchSamplingMode sampling_mode)
    : BatchExecutor(plan, output_mode, sampling_mode,
                    batch_detail::batch_worker_storage_layout(plan, lane_capacity, output_mode,
                                                              sampling_mode)) {}

BatchExecutor::BatchExecutor(const ExecutablePlan& plan, BatchOutputMode output_mode,
                             BatchSamplingMode sampling_mode,
                             const batch_detail::BatchWorkerStorageLayout& storage)
    : plan_(validate_batch_plan(plan)),
      output_mode_(output_mode),
      sampling_mode_(sampling_mode),
      lane_capacity_(storage.lane_capacity),
      word_capacity_(storage.word_capacity),
      state_(storage.peak_active_width, storage.initial_active_width, storage.lane_capacity),
      shot_indices_(storage.shot_index_entries),
      symbols_(storage.symbol_columns, storage.lane_capacity),
      batch_noise_carriers_(storage.noise_carrier_columns, storage.lane_capacity),
      expression_registers_(storage.expression_register_columns, storage.lane_capacity),
      records_(storage.record_columns, storage.lane_capacity),
      detectors_(storage.detector_columns, storage.lane_capacity),
      observables_(storage.observable_columns, storage.lane_capacity),
      forced_readout_(storage.forced_readout_columns, storage.lane_capacity),
      exp_vals_(checked_size(storage.exp_value_entries, "expectation"), 0.0),
      live_words_(storage.live_word_entries, 0),
      scratch_words_(storage.scratch_word_entries, 0),
      compaction_sources_(storage.compaction_source_entries),
      lane_bytes_(storage.lane_byte_entries, 0),
      signed_sines_(storage.signed_sine_entries, 0.0),
      probability_zero_(storage.probability_zero_entries, 0.0),
      probability_one_(storage.probability_one_entries, 0.0),
      lane_values_(storage.lane_value_entries, 0.0) {}

void BatchExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept {
    assert(sampling_mode_ == BatchSamplingMode::Ordinary &&
           "ordinary execution requires an ordinary packed worker");
    reset_batch(root, first_shot, shots);
    sample_presampled_noise();
    execute_actions();
    if (output_mode_ == BatchOutputMode::Rows) {
        finalize_live_lanes();
    }
}

void BatchExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                              KFaultSampler& fault_sampler) noexcept {
    assert(sampling_mode_ == BatchSamplingMode::FixedFaults &&
           "conditioned execution requires a fixed-fault packed worker");
    reset_batch(root, first_shot, shots);
    assign_forced_faults(fault_sampler);
    execute_actions();
    if (output_mode_ == BatchOutputMode::Rows) {
        finalize_live_lanes();
    }
}

void BatchExecutor::reset_batch(const SeedRoot& root, uint32_t first_shot,
                                uint32_t shots) noexcept {
    assert(shots <= lane_capacity_ && "packed batch must fit retained capacity");
    live_count_ = shots;
    fill_low_lane_mask(live_words_, shots);
    if (plan_->batch_presampled_program_.has_value()) {
        batch_noise_carriers_.clear();
    } else {
        symbols_.clear();
    }
    expression_registers_.clear();
    records_.clear();
    detectors_.clear();
    observables_.clear();
    if (sampling_mode_ == BatchSamplingMode::FixedFaults) {
        forced_readout_.clear();
    }
    std::ranges::fill(exp_vals_, 0.0);
    state_.reset(shots);
    const std::array<uint64_t, 4> rng_words =
        derive_state(root, first_shot, kBatchSamplingExecutorDomain);
    rng_.seed_full(rng_words[0], rng_words[1], rng_words[2], rng_words[3]);
    for (uint32_t lane = 0; lane < shots; ++lane) {
        shot_indices_[lane] = first_shot + lane;
    }
    initialize_expression_registers();
}

void BatchExecutor::fill_random_half_bits() noexcept {
    for (size_t word = 0; word < word_capacity_; ++word) {
        scratch_words_[word] = rng_() & live_words_[word];
    }
}

void BatchExecutor::initialize_expression_registers() noexcept {
    for (size_t expression = 0; expression < plan_->expression_register_constants_.size();
         ++expression) {
        if (plan_->expression_register_constants_[expression] != 0) {
            expression_registers_.assign(expression, live_words_, live_words_);
        }
    }
}

void BatchExecutor::initialize_presampled_expressions() noexcept {
    assert(plan_->batch_presampled_program_.has_value() &&
           "presampled expression program must contain matching levels");
    const BatchPresampledProgram& program = *plan_->batch_presampled_program_;
    const size_t levels = program.initialization_level_offsets_.size() - 1;
    for (size_t level = 0; level < levels; ++level) {
        const uint32_t initialization_begin = program.initialization_level_offsets_[level];
        const uint32_t initialization_end = program.initialization_level_offsets_[level + 1];
        for (uint32_t index = initialization_begin; index < initialization_end; ++index) {
            const BatchPresampledProgram::InitializeExpression& initialization =
                program.initializations_[index];
            expression_registers_.copy(initialization.destination, initialization.parent);
            if (initialization.invert_parent) {
                expression_registers_.xor_into(initialization.destination, live_words_);
            }
        }
        const uint32_t carrier_xor_begin = program.carrier_xor_level_offsets_[level];
        const uint32_t carrier_xor_end = program.carrier_xor_level_offsets_[level + 1];
        for (uint32_t index = carrier_xor_begin; index < carrier_xor_end; ++index) {
            const BatchPresampledProgram::XorCarrierIntoExpression& carrier_xor =
                program.carrier_xors_[index];
            expression_registers_.xor_into(carrier_xor.destination,
                                           batch_noise_carriers_.column(carrier_xor.carrier));
        }
    }
    for (const BatchPresampledProgram::CopyExpression& copy : program.copies_) {
        expression_registers_.copy(copy.destination, copy.source);
    }
}

void BatchExecutor::finalize_presampled_symbols() noexcept {
    if (!plan_->batch_presampled_program_.has_value()) {
        for (uint32_t symbol : plan_->presampled_symbols_) {
            propagate_symbol(symbol, symbols_.column(symbol));
        }
        return;
    }
    initialize_presampled_expressions();
}

void BatchExecutor::sample_presampled_noise() noexcept {
    const uint32_t end = static_cast<uint32_t>(plan_->noise_sites_.size());
    if (plan_->uniform_noise_inverse_hazard_.has_value()) {
        const ExecutablePlan::PreparedNoiseSite& first_site = plan_->noise_sites_.front();
        const uint32_t first_outcome_end = first_site.outcome_begin + first_site.outcome_count;
        if (plan_->noise_outcomes_[first_outcome_end - 1].cumulative_probability >= 1.0) {
            for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
                for (uint32_t site = 0; site < end; ++site) {
                    activate_noise_site(lane, site);
                }
            }
            finalize_presampled_symbols();
            return;
        }
        const double inverse_hazard = *plan_->uniform_noise_inverse_hazard_;
        const uint64_t total_draws = static_cast<uint64_t>(active_lanes()) * end;
        uint64_t draw_index = 0;
        while (draw_index < total_draws) {
            const double gap = -std::log(1.0 - rng_.next_double()) * inverse_hazard;
            if (gap >= static_cast<double>(total_draws - draw_index)) {
                break;
            }
            draw_index += static_cast<uint64_t>(gap);
            const uint32_t lane = static_cast<uint32_t>(draw_index / end);
            const uint32_t site = static_cast<uint32_t>(draw_index % end);
            activate_noise_site(lane, site);
            ++draw_index;
        }
        finalize_presampled_symbols();
        return;
    }
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        uint32_t first_candidate = 0;
        while (first_candidate < end) {
            const double current_hazard =
                first_candidate == 0 ? 0.0 : plan_->noise_hazards_[first_candidate - 1];
            if (current_hazard >= plan_->noise_hazards_[end - 1]) {
                break;
            }
            const uint32_t site =
                sample_next_noise_site(plan_->noise_hazards_, first_candidate, rng_.next_double());
            if (site == kNoNoiseSite || site >= end) {
                break;
            }
            activate_noise_site(lane, site);
            first_candidate = site + 1;
        }
    }
    finalize_presampled_symbols();
}

void BatchExecutor::assign_forced_faults(KFaultSampler& fault_sampler) noexcept {
    assert(fault_sampler.num_sites() ==
               plan_->noise_sites_.size() + plan_->num_readout_noise_sites_ &&
           "conditioned batch sampler must cover every fault site");
    const uint32_t quantum_sites = static_cast<uint32_t>(plan_->noise_sites_.size());
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        const std::span<const uint32_t> selected =
            fault_sampler.sample([&]() noexcept { return rng_.next_double(); });
        for (uint32_t site : selected) {
            if (site < quantum_sites) {
                activate_noise_site(lane, site);
            } else {
                forced_readout_.set_bit(site - quantum_sites, lane);
            }
        }
    }
    finalize_presampled_symbols();
}

void BatchExecutor::activate_noise_site(uint32_t lane, uint32_t site_index) noexcept {
    assert(site_index < plan_->noise_sites_.size() && "noise site must be prepared");
    const ExecutablePlan::PreparedNoiseSite& site = plan_->noise_sites_[site_index];
    const uint32_t outcome_end = site.outcome_begin + site.outcome_count;
    assert(site.outcome_count > 0 && outcome_end <= plan_->noise_outcomes_.size() &&
           "firing noise site must contain prepared outcomes");
    uint32_t outcome_index = site.outcome_begin;
    if (site.outcome_count > 1) {
        const double execution_probability =
            plan_->noise_outcomes_[outcome_end - 1].cumulative_probability;
        const double draw = rng_.next_double() * execution_probability;
        while (draw >= plan_->noise_outcomes_[outcome_index].cumulative_probability) {
            ++outcome_index;
            assert(outcome_index < outcome_end && "channel draw must select a prepared outcome");
        }
    }
    if (!plan_->batch_presampled_program_.has_value()) {
        const uint32_t symbol = plan_->noise_outcomes_[outcome_index].symbol;
        assert(!symbols_.bit(symbol, lane) && "noise site must define a fresh lane symbol");
        symbols_.set_bit(symbol, lane);
        return;
    }

    const BatchPresampledProgram& program = *plan_->batch_presampled_program_;
    const BatchPresampledProgram::OutcomeAssignments& batch_outcome =
        program.outcome_assignments_[outcome_index];
    const uint32_t assignment_end = batch_outcome.begin + batch_outcome.count;
    assert(assignment_end <= program.assigned_carriers_.size() &&
           "batch noise assignment must stay in its prepared tape");
    for (uint32_t assignment = batch_outcome.begin; assignment < assignment_end; ++assignment) {
        const uint32_t carrier = program.assigned_carriers_[assignment];
        assert(!batch_noise_carriers_.bit(carrier, lane) &&
               "batch noise carrier must be assigned once per site");
        batch_noise_carriers_.set_bit(carrier, lane);
    }
}

void BatchExecutor::propagate_symbol(uint32_t symbol, std::span<const uint64_t> values) noexcept {
    for (uint32_t register_id : plan_->expression_dependencies_.dependent_registers(symbol)) {
        expression_registers_.xor_into(register_id, values);
    }
}

void BatchExecutor::execute_actions() noexcept {
    for (size_t action_index = 0; action_index < plan_->actions_.size(); ++action_index) {
        const ExecutablePlan::Action& action = plan_->actions_[action_index];
        std::visit([&](const auto& typed) noexcept { execute_action(typed, action_index); },
                   action);
        if (live_count_ == 0) {
            return;
        }
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteRotation& action, size_t) noexcept {
    const std::span<const uint64_t> signs = evaluate(action.sign);
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        lane_bytes_[lane] = static_cast<uint8_t>(lane_bit(signs, lane));
    }
    prepare_interleaved_rotation_sines(signed_sines_, action.rotation.sine,
                                       std::span<const uint8_t>(lane_bytes_).first(active_lanes()));
    apply_interleaved_rotation(state_, action.rotation, signed_sines_);
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                                   size_t) noexcept {
    assert(action.rotation_index < plan_->fused_rotations_.size() &&
           "fused rotation action must reference prepared execution");
    apply_interleaved_fused_rotation(state_,
                                     plan_->fused_rotations_[action.rotation_index].rotation());
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action,
                                   size_t) noexcept {
    assert(action.rotation_index < plan_->dynamic_fused_rotations_.size() &&
           "dynamic fused rotation action must reference prepared execution");
    const auto& rotation = plan_->dynamic_fused_rotations_[action.rotation_index];
    std::array<const PreparedFusedRotation*, 4> variants{};
    assert(rotation.variants.size() <= variants.size() &&
           "dynamic fused rotation variant count must fit prepared scratch");
    for (size_t variant = 0; variant < rotation.variants.size(); ++variant) {
        variants[variant] = &rotation.variants[variant].rotation();
    }
    std::array<std::span<const uint64_t>, 2> sign_bits{};
    assert(rotation.sign_basis.size() <= sign_bits.size() &&
           "dynamic fused rotation sign basis must fit prepared scratch");
    for (size_t basis = 0; basis < rotation.sign_basis.size(); ++basis) {
        sign_bits[basis] = evaluate(rotation.sign_basis[basis]);
    }
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        uint32_t variant = 0;
        for (size_t basis = 0; basis < rotation.sign_basis.size(); ++basis) {
            variant |= static_cast<uint32_t>(lane_bit(sign_bits[basis], lane)) << basis;
        }
        assert(variant < rotation.variants.size() && "dynamic sign must select a prepared variant");
        lane_bytes_[lane] = static_cast<uint8_t>(variant);
    }
    apply_interleaved_dynamic_fused_rotation(
        state_,
        std::span<const PreparedFusedRotation* const>(variants).first(rotation.variants.size()),
        std::span<const uint8_t>(lane_bytes_).first(active_lanes()));
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecutePromotion& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> signs = evaluate(action.sign);
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        lane_bytes_[lane] = static_cast<uint8_t>(lane_bit(signs, lane));
    }
    prepare_interleaved_rotation_sines(signed_sines_, action.promotion.sine,
                                       std::span<const uint8_t>(lane_bytes_).first(active_lanes()));
    apply_interleaved_promotion(state_, action.promotion, signed_sines_);
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    interleaved_measurement_probabilities(state_, action.measurement, probability_zero_,
                                          probability_one_);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        const MeasurementProbabilities probabilities{probability_zero_[lane],
                                                     probability_one_[lane]};
        const MeasurementBranchKind classification = classify_measurement_branch(probabilities);
        bool branch = classification == MeasurementBranchKind::DeterministicOne;
        if (is_live(lane)) {
            switch (classification) {
                case MeasurementBranchKind::Random:
                    branch = rng_.next_double() * probabilities.total() >= probabilities.zero;
                    break;
                case MeasurementBranchKind::DeterministicZero:
                    branch = false;
                    break;
                case MeasurementBranchKind::DeterministicOne:
                    branch = true;
                    break;
            }
        }
        lane_bytes_[lane] = static_cast<uint8_t>(branch);
        if (branch) {
            scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
        }
        lane_values_[lane] = probabilities.for_branch(branch);
    }
    collapse_interleaved_measurement(state_, action.measurement,
                                     std::span<const uint8_t>(lane_bytes_).first(active_lanes()),
                                     std::span<const double>(lane_values_).first(active_lanes()));
    propagate_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, corrections, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    fill_random_half_bits();
    propagate_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, corrections, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                                   size_t) noexcept {
    if (records_.num_columns() != 0) {
        records_.assign(action.record, evaluate(action.outcome), live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action,
                                   size_t) noexcept {
    propagate_symbol(action.symbol, evaluate(action.value));
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> sources = evaluate(action.source);
    std::ranges::fill(scratch_words_, uint64_t{0});
    bool sampled_packed = false;
    if (sampling_mode_ == BatchSamplingMode::FixedFaults) {
        const std::span<const uint64_t> forced = forced_readout_.column(action.site);
        for (size_t word = 0; word < word_capacity_; ++word) {
            scratch_words_[word] = forced[word] & live_words_[word];
        }
        sampled_packed = true;
    } else if (action.prob_zero_to_one == action.prob_one_to_zero) {
        const double probability = action.prob_zero_to_one;
        if (probability <= 0.0) {
            sampled_packed = true;
        } else if (probability >= 1.0) {
            std::ranges::copy(live_words_, scratch_words_.begin());
            sampled_packed = true;
        } else if (probability == 0.5) {
            fill_random_half_bits();
            sampled_packed = true;
        } else if (action.batch_symmetric_inverse_hazard > 0.0) {
            uint64_t lane = 0;
            while (lane < active_lanes()) {
                const double gap =
                    -std::log(1.0 - rng_.next_double()) * action.batch_symmetric_inverse_hazard;
                if (gap >= static_cast<double>(active_lanes() - lane)) {
                    break;
                }
                lane += static_cast<uint64_t>(gap);
                scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
                ++lane;
            }
            sampled_packed = true;
        }
    }
    if (!sampled_packed) {
        for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
            if (!is_live(lane)) {
                continue;
            }
            const bool source = lane_bit(sources, lane);
            const double probability = source ? action.prob_one_to_zero : action.prob_zero_to_one;
            if (probability >= 1.0 || (probability > 0.0 && rng_.next_double() < probability)) {
                scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
            }
        }
    }
    propagate_symbol(action.flip, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, sources, scratch_words_, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDetector& action,
                                   size_t action_index) noexcept {
    const std::span<const uint64_t> outcomes = evaluate_record_parity(action.outcome);
    if (output_mode_ == BatchOutputMode::Rows) {
        detectors_.assign(action.detector, outcomes, live_words_);
    }
    if (!action.postselected) {
        return;
    }
    uint32_t rejected = 0;
    for (size_t word = 0; word < word_capacity_; ++word) {
        const uint64_t dead = outcomes[word] & live_words_[word];
        rejected += static_cast<uint32_t>(std::popcount(dead));
        live_words_[word] &= ~dead;
    }
    live_count_ -= rejected;
    if (should_compact(action_index)) {
        compact_live_lanes();
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteObservable& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> outcomes = evaluate_observable(action.outcome);
    observables_.assign(action.observable, outcomes, live_words_);
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteExpectation& action,
                                   size_t) noexcept {
    if (output_mode_ == BatchOutputMode::AggregateSurvivors) {
        return;
    }
    assert(action.exp_val < plan_->num_exp_vals_ &&
           "expectation action must reference preallocated storage");
    double* output = exp_vals_.data() + static_cast<size_t>(action.exp_val) * lane_capacity_;
    if (!action.active.has_value()) {
        for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
            if (is_live(lane)) {
                output[lane] = 0.0;
            }
        }
        return;
    }
    const std::span<const uint64_t> signs = evaluate(action.active->sign);
    interleaved_expectation_values(state_, action.active->projection, lane_values_);
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        if (is_live(lane)) {
            output[lane] = lane_bit(signs, lane) ? -lane_values_[lane] : lane_values_[lane];
        }
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteInstrument&, size_t) noexcept {
    assert(false && "instrument actions must remain on the scalar trajectory executor");
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteBoundary&, size_t) noexcept {
    assert(false && "continuation boundaries must remain on the scalar trajectory executor");
}

std::span<const uint64_t> BatchExecutor::evaluate(
    ExecutablePlan::PreparedExpression expression) const noexcept {
    assert(expression.register_id < expression_registers_.num_columns() &&
           "prepared expression must reference a packed register");
    return expression_registers_.column(expression.register_id);
}

std::span<const uint64_t> BatchExecutor::evaluate_observable(
    const ExecutablePlan::PreparedObservableValue& value) noexcept {
    if (const auto* expression = std::get_if<ExecutablePlan::PreparedExpression>(&value)) {
        return evaluate(*expression);
    }
    return evaluate_record_parity(std::get<ExecutablePlan::PreparedRecordParity>(value));
}

std::span<const uint64_t> BatchExecutor::evaluate_record_parity(
    ExecutablePlan::PreparedRecordParity parity) noexcept {
    const uint32_t end = parity.begin + parity.count;
    assert(end <= plan_->record_parity_terms_.size() &&
           "prepared record parity must stay in its term tape");
    const size_t words = packed_word_count(active_lanes());
    if (parity.constant) {
        std::ranges::copy(std::span<const uint64_t>(live_words_).first(words),
                          scratch_words_.begin());
    } else {
        std::ranges::fill(std::span<uint64_t>(scratch_words_).first(words), uint64_t{0});
    }
    for (uint32_t term = parity.begin; term < end; ++term) {
        const std::span<const uint64_t> record = records_.column(plan_->record_parity_terms_[term]);
        for (size_t word = 0; word < words; ++word) {
            scratch_words_[word] ^= record[word];
        }
    }
    return scratch_words_;
}

bool BatchExecutor::lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept {
    assert(bits.size() >= word_capacity_ && lane < active_lanes() &&
           "packed expression lookup must reference an active lane");
    return ((bits[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

bool BatchExecutor::is_live(uint32_t lane) const noexcept {
    assert(lane < active_lanes() && "live lookup must reference the current lane span");
    return ((live_words_[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

bool BatchExecutor::should_compact(size_t action_index) const noexcept {
    if (live_count_ == 0 || live_count_ == active_lanes()) {
        return false;
    }
    const uint64_t remaining_actions = plan_->actions_.size() - action_index - 1;
    if (remaining_actions == 0) {
        return false;
    }
    const uint64_t old_words = packed_word_count(active_lanes());
    const uint64_t new_words = packed_word_count(live_count_);
    const uint64_t dead_lanes = active_lanes() - live_count_;
    const uint64_t bit_columns = expression_registers_.num_columns() + records_.num_columns() +
                                 detectors_.num_columns() + observables_.num_columns() +
                                 forced_readout_.num_columns();
    const uint64_t carry_cost =
        dead_lanes * remaining_actions + (old_words - new_words) * remaining_actions * 8;
    const uint64_t compact_cost = bit_columns * old_words +
                                  static_cast<uint64_t>(plan_->num_exp_vals_) * live_count_ +
                                  static_cast<uint64_t>(live_count_) * 3;
    return carry_cost > compact_cost;
}

void BatchExecutor::compact_live_lanes() noexcept {
    if (live_count_ == active_lanes()) {
        return;
    }
    if (live_count_ == 0) {
        state_.compact_lanes({});
        std::ranges::fill(live_words_, uint64_t{0});
        return;
    }
    const uint32_t old_lanes = active_lanes();
    uint32_t destination = 0;
    for (uint32_t source = 0; source < old_lanes; ++source) {
        if (is_live(source)) {
            compaction_sources_[destination++] = source;
        }
    }
    assert(destination == live_count_ && "lane compaction must retain every live context");
    const std::span<const uint32_t> sources(compaction_sources_.data(), live_count_);
    expression_registers_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    records_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    detectors_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    observables_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    forced_readout_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    state_.compact_lanes(sources);

    for (destination = 0; destination < live_count_; ++destination) {
        const uint32_t source = sources[destination];
        if (destination != source) {
            shot_indices_[destination] = shot_indices_[source];
        }
    }
    if (output_mode_ == BatchOutputMode::Rows) {
        for (uint32_t exp_val = 0; exp_val < plan_->num_exp_vals_; ++exp_val) {
            double* values = exp_vals_.data() + static_cast<size_t>(exp_val) * lane_capacity_;
            for (destination = 0; destination < live_count_; ++destination) {
                const uint32_t source = sources[destination];
                if (destination != source) {
                    values[destination] = values[source];
                }
            }
        }
    }
    fill_low_lane_mask(live_words_, active_lanes());
}

uint32_t BatchExecutor::accumulate_survivor_counts(
    std::span<uint64_t> observable_ones) const noexcept {
    assert(output_mode_ == BatchOutputMode::AggregateSurvivors &&
           observable_ones.size() == plan_->num_observables_ &&
           "aggregate survivor outputs must match the prepared observable count");
    uint32_t logical_errors = 0;
    const size_t words = packed_word_count(active_lanes());
    for (size_t word = 0; word < words; ++word) {
        const uint64_t live = live_words_[word];
        uint64_t any_observable = 0;
        for (uint32_t observable = 0; observable < plan_->num_observables_; ++observable) {
            const uint64_t ones = observables_.column(observable)[word] & live;
            observable_ones[observable] += static_cast<uint64_t>(std::popcount(ones));
            any_observable |= ones;
        }
        logical_errors += static_cast<uint32_t>(std::popcount(any_observable));
    }
    return logical_errors;
}

void BatchExecutor::finalize_live_lanes() noexcept {
    if (live_count_ != active_lanes()) {
        compact_live_lanes();
    }
}

uint32_t BatchExecutor::shot_index(uint32_t lane) const noexcept {
    assert(lane < live_count_ && active_lanes() == live_count_ &&
           "batch outputs require finalized live lanes");
    return shot_indices_[lane];
}

bool BatchExecutor::measurement(uint32_t lane, uint32_t record) const noexcept {
    assert(lane < live_count_ && record < plan_->num_visible_records_ &&
           "measurement output must be visible and live");
    return records_.bit(record, lane);
}

bool BatchExecutor::detector(uint32_t lane, uint32_t detector_index) const noexcept {
    assert(lane < live_count_ && detector_index < plan_->num_detectors_ &&
           "detector output must be live and in range");
    return detectors_.bit(detector_index, lane);
}

bool BatchExecutor::observable(uint32_t lane, uint32_t observable_index) const noexcept {
    assert(lane < live_count_ && observable_index < plan_->num_observables_ &&
           "observable output must be live and in range");
    return observables_.bit(observable_index, lane);
}

double BatchExecutor::exp_val(uint32_t lane, uint32_t exp_val_index) const noexcept {
    assert(lane < live_count_ && exp_val_index < plan_->num_exp_vals_ &&
           "expectation output must be live and in range");
    return exp_vals_[static_cast<size_t>(exp_val_index) * lane_capacity_ + lane];
}

}  // namespace clifft::sampling
