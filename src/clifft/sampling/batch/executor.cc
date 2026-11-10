#include "clifft/sampling/batch/executor.h"

#include "clifft/sampling/measurement_branch.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>

namespace clifft::sampling {

namespace {

[[nodiscard]] size_t checked_size(uint64_t entries, const char* description) {
    if (entries > std::numeric_limits<size_t>::max()) {
        throw std::length_error(std::string("packed batch ") + description +
                                " allocation exceeds size_t range");
    }
    return static_cast<size_t>(entries);
}

}  // namespace

BatchExecutor::BatchExecutor(const ExecutablePlan& plan, uint32_t lane_capacity,
                             BatchOutputMode output_mode, BatchSamplingMode sampling_mode,
                             const PreparedBatchNoise* noise)
    : BatchExecutor(plan, output_mode, sampling_mode,
                    batch_detail::batch_worker_storage_layout(plan, lane_capacity, output_mode,
                                                              sampling_mode, noise),
                    noise) {}

BatchExecutor::BatchExecutor(const ExecutablePlan& plan, BatchOutputMode output_mode,
                             BatchSamplingMode sampling_mode,
                             const batch_detail::BatchWorkerStorageLayout& storage,
                             const PreparedBatchNoise* noise)
    : BatchClassicalState(plan, output_mode, sampling_mode, storage, noise),
      state_(storage.peak_active_width, storage.initial_active_width, storage.lane_capacity),
      exp_vals_(checked_size(storage.exp_value_entries, "expectation"), 0.0),
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

void BatchExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                              const BatchExternalInputs& inputs) noexcept {
    assert(sampling_mode_ == BatchSamplingMode::ExternalInputs &&
           "external execution requires an external-input packed worker");
    assert(inputs.plan_ == plan_ && inputs.lane_capacity_ == lane_capacity_ &&
           "external inputs must match the worker's plan and retained capacity");
    reset_batch(root, first_shot, shots);
    assign_external_inputs(inputs);
    execute_actions();
    if (output_mode_ == BatchOutputMode::Rows) {
        finalize_live_lanes();
    }
}

void BatchExecutor::reset_batch(const SeedRoot& root, uint32_t first_shot,
                                uint32_t shots) noexcept {
    reset_classical(root, first_shot, shots);
    compacted_during_execution_ = false;
    std::ranges::fill(exp_vals_, 0.0);
    state_.reset(shots);
}

void BatchExecutor::execute_actions() noexcept {
    for (const ExecutablePlan::Action& action : plan_->actions_) {
        std::visit([&](const auto& typed) noexcept { execute_action(typed); }, action);
        if (live_count_ == 0) {
            return;
        }
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteRotation& action) noexcept {
    const std::span<const uint64_t> signs = evaluate(action.sign);
    prepare_interleaved_rotation_sines(std::span<double>(signed_sines_).first(active_lanes()),
                                       action.rotation.sine, signs);
    apply_interleaved_rotation(state_, action.rotation, signed_sines_);
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteFusedRotation& action) noexcept {
    assert(action.rotation_index < plan_->fused_rotations_.size() &&
           "fused rotation action must reference prepared execution");
    apply_interleaved_fused_rotation(state_,
                                     plan_->fused_rotations_[action.rotation_index].rotation());
}

void BatchExecutor::execute_action(
    const ExecutablePlan::ExecuteDynamicFusedRotation& action) noexcept {
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

void BatchExecutor::execute_action(const ExecutablePlan::ExecutePromotion& action) noexcept {
    const std::span<const uint64_t> signs = evaluate(action.sign);
    prepare_interleaved_rotation_sines(std::span<double>(signed_sines_).first(active_lanes()),
                                       action.promotion.sine, signs);
    apply_interleaved_promotion(state_, action.promotion, signed_sines_);
}

void BatchExecutor::execute_action(
    const ExecutablePlan::ExecuteActiveMeasurement& action) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    interleaved_measurement_probabilities(state_, action.measurement, probability_zero_,
                                          probability_one_);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t lane = 0; lane < active_lanes(); ++lane) {
        const MeasurementProbabilities probabilities{probability_zero_[lane],
                                                     probability_one_[lane]};
        const MeasurementBranchKind classification =
            classify_measurement_branch(probabilities).kind;
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

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDetector& action) noexcept {
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
    if (should_compact(action)) {
        compact_live_lanes(CompactionMode::ContinueExecution);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteExpectation& action) noexcept {
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

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteInstrument&) noexcept {
    assert(false && "instrument actions must remain on the scalar trajectory executor");
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteBoundary&) noexcept {
    assert(false && "continuation boundaries must remain on the scalar trajectory executor");
}

bool BatchExecutor::should_compact(const ExecutablePlan::ExecuteDetector& detector) const noexcept {
    const uint64_t bit_columns = expression_registers_.num_columns() + records_.num_columns() +
                                 detectors_.num_columns() + observables_.num_columns() +
                                 forced_readout_.num_columns();
    return batch_detail::should_compact_batch_lanes(
        {.remaining_lane_work = detector.remaining_batch_lane_work,
         .state_size = state_.size(),
         .bit_columns = bit_columns,
         .row_output_entries = plan_->num_exp_vals_,
         .word_capacity = word_capacity_,
         .peak_active_width = plan_->peak_active_width(),
         .active_lanes = active_lanes(),
         .live_lanes = live_count_},
        output_mode_);
}

void BatchExecutor::compact_live_lanes(CompactionMode mode) noexcept {
    if (live_count_ == active_lanes()) {
        return;
    }
    if (live_count_ == 0) {
        state_.compact_lanes({});
        active_lanes_ = 0;
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
    // Output access only reads visible records and output sidecars. The skipped
    // expression, forced-readout, and hidden record columns are cleared or
    // overwritten before execution of the next batch.
    if (mode == CompactionMode::ContinueExecution) {
        compacted_during_execution_ = true;
        expression_registers_.compact(live_words_, old_lanes, live_count_, scratch_words_);
        forced_readout_.compact(live_words_, old_lanes, live_count_, scratch_words_);
        records_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    } else {
        records_.compact_prefix(plan_->num_visible_records_, live_words_, old_lanes, live_count_,
                                scratch_words_);
    }
    detectors_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    observables_.compact(live_words_, old_lanes, live_count_, scratch_words_);
    state_.compact_lanes(sources);
    active_lanes_ = live_count_;

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

void BatchExecutor::finalize_live_lanes() noexcept {
    if (live_count_ != active_lanes()) {
        compact_live_lanes(CompactionMode::FinalizeOutputs);
    }
}

double BatchExecutor::exp_val(uint32_t lane, uint32_t exp_val_index) const noexcept {
    assert(lane < live_count_ && exp_val_index < plan_->num_exp_vals_ &&
           "expectation output must be live and in range");
    return exp_vals_[static_cast<size_t>(exp_val_index) * lane_capacity_ + lane];
}

}  // namespace clifft::sampling
