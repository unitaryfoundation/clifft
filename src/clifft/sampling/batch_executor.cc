#include "clifft/sampling/batch_executor.h"

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

struct MeasurementBranchClassification {
    MeasurementBranchKind kind = MeasurementBranchKind::Random;
    bool clamped_dust = false;
};

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

[[nodiscard]] const ExecutablePlan* validate_batch_plan(const ExecutablePlan& plan) {
    if (plan.has_instruments()) {
        throw std::invalid_argument("packed sampling does not support instrument continuations");
    }
    return &plan;
}

[[nodiscard]] size_t checked_product(size_t left, size_t right, const char* description) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::length_error(std::string("packed batch ") + description +
                                " allocation exceeds size_t range");
    }
    return left * right;
}

[[nodiscard]] uint64_t interleaved_state_bytes_per_lane(const ExecutablePlan& plan) noexcept {
    const uint64_t coefficient_capacity = uint64_t{1} << plan.peak_active_width();
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (plan.peak_active_width() != 0 && coefficient_capacity > kMax / 3) {
        return kMax;
    }
    const uint64_t entries = plan.peak_active_width() == 0 ? 4 : 3 * coefficient_capacity;
    if (entries > kMax / sizeof(double)) {
        return kMax;
    }
    return sizeof(double) * entries;
}

}  // namespace

#if !defined(__EMSCRIPTEN__)
uint32_t resolve_batch_capacity(const ExecutablePlan& plan, uint32_t shots,
                                uint32_t intra_shot_workers,
                                std::optional<uint32_t> requested_batch_size) {
    if (requested_batch_size.has_value() && *requested_batch_size == 0) {
        throw std::invalid_argument("batch_size must be a positive integer or 'auto'");
    }
    if (shots == 0 || plan.has_instruments()) {
        return 1;
    }
    if (intra_shot_workers > 1) {
        if (requested_batch_size.has_value() && *requested_batch_size > 1) {
            throw std::invalid_argument(
                "packed batch_size is incompatible with intra-shot workers");
        }
        return 1;
    }
    if (requested_batch_size.has_value()) {
        const uint32_t capacity =
            std::max(uint32_t{1}, std::min({*requested_batch_size, shots, kMaxExplicitBatchShots}));
        const uint64_t state_bytes_per_lane = interleaved_state_bytes_per_lane(plan);
        const uint64_t lane_pitch = (static_cast<uint64_t>(capacity) + 7) & ~uint64_t{7};
        if (capacity > 1 && state_bytes_per_lane > kMaxExplicitBatchStateBudget / lane_pitch) {
            throw std::invalid_argument(
                "explicit batch_size exceeds the 64 MiB packed-state limit; request a smaller "
                "batch_size");
        }
        return capacity;
    }
    if (shots < kDefaultMinAutoBatchShots) {
        return 1;
    }
    if (plan.peak_active_width() > 5) {
        return 1;
    }
    const uint64_t state_bytes_per_lane = interleaved_state_bytes_per_lane(plan);
    const size_t footprint_capacity =
        std::max<uint64_t>(1, kDefaultBatchStateBudget / state_bytes_per_lane);
    return static_cast<uint32_t>(
        std::min<size_t>({shots, kDefaultMaxAutoBatchShots, footprint_capacity}));
}
#endif

BatchExecutor::BatchExecutor(const ExecutablePlan& plan, uint32_t lane_capacity,
                             BatchOutputMode output_mode)
    : plan_(validate_batch_plan(plan)),
      output_mode_(output_mode),
      lane_capacity_(lane_capacity),
      word_capacity_(packed_word_count(lane_capacity)),
      state_(plan.peak_active_width_, plan.initial_active_width_, lane_capacity),
      shot_indices_(lane_capacity),
      symbols_(plan.num_symbols_, lane_capacity),
      expression_registers_(plan.expression_register_constants_.size(), lane_capacity),
      records_(output_mode == BatchOutputMode::Rows || !plan.batch_record_parities_.empty()
                   ? static_cast<size_t>(plan.num_visible_records_) + plan.num_hidden_records_
                   : 0,
               lane_capacity),
      detectors_(output_mode == BatchOutputMode::Rows ? plan.num_detectors_ : 0, lane_capacity),
      observables_(plan.num_observables_, lane_capacity),
      forced_readout_(plan.num_readout_noise_sites_, lane_capacity),
      exp_vals_(output_mode == BatchOutputMode::Rows
                    ? checked_product(plan.num_exp_vals_, lane_capacity, "expectation")
                    : 0,
                0.0),
      live_words_(word_capacity_, 0),
      scratch_words_(word_capacity_, 0),
      compaction_scratch_(word_capacity_, 0),
      compaction_sources_(lane_capacity),
      lane_bytes_(lane_capacity, 0),
      signed_sines_(lane_capacity, 0.0),
      probability_zero_(lane_capacity, 0.0),
      probability_one_(lane_capacity, 0.0),
      lane_values_(lane_capacity, 0.0) {}

void BatchExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept {
    fixed_fault_mode_ = false;
    reset_batch(root, first_shot, shots);
    sample_presampled_noise();
    execute_actions();
    if (output_mode_ == BatchOutputMode::Rows) {
        finalize_live_lanes();
    }
}

void BatchExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                              KFaultSampler& fault_sampler) noexcept {
    fixed_fault_mode_ = true;
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
    attempted_shots_ = shots;
    active_lanes_ = shots;
    live_count_ = shots;
    fill_low_lane_mask(live_words_, shots);
    symbols_.clear();
    expression_registers_.clear();
    records_.clear();
    detectors_.clear();
    observables_.clear();
    forced_readout_.clear();
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
    assert(!plan_->presampled_initialization_level_offsets_.empty() &&
           plan_->presampled_initialization_level_offsets_.size() ==
               plan_->presampled_delta_level_offsets_.size() &&
           "presampled expression program must contain matching levels");
    const size_t levels = plan_->presampled_initialization_level_offsets_.size() - 1;
    for (size_t level = 0; level < levels; ++level) {
        const uint32_t initialization_begin =
            plan_->presampled_initialization_level_offsets_[level];
        const uint32_t initialization_end =
            plan_->presampled_initialization_level_offsets_[level + 1];
        for (uint32_t index = initialization_begin; index < initialization_end; ++index) {
            const ExecutablePlan::PresampledExpressionInitialization& initialization =
                plan_->presampled_initializations_[index];
            if (initialization.parent == std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            expression_registers_.copy(initialization.destination, initialization.parent);
            if (initialization.invert_parent) {
                expression_registers_.xor_into(initialization.destination, live_words_);
            }
        }
        const uint32_t delta_begin = plan_->presampled_delta_level_offsets_[level];
        const uint32_t delta_end = plan_->presampled_delta_level_offsets_[level + 1];
        for (uint32_t index = delta_begin; index < delta_end; ++index) {
            const ExecutablePlan::PresampledExpressionDelta& delta =
                plan_->presampled_deltas_[index];
            expression_registers_.xor_into(delta.destination, symbols_.column(delta.symbol));
        }
    }
    for (const ExecutablePlan::PresampledExpressionCopy& copy : plan_->presampled_copies_) {
        expression_registers_.copy(copy.destination, copy.source);
    }
}

void BatchExecutor::sample_presampled_noise() noexcept {
    const uint32_t end = static_cast<uint32_t>(plan_->noise_sites_.size());
    if (plan_->uniform_noise_inverse_hazard_.has_value()) {
        const ExecutablePlan::PreparedNoiseSite& first_site = plan_->noise_sites_.front();
        const uint32_t first_outcome_end = first_site.outcome_begin + first_site.outcome_count;
        if (plan_->noise_outcomes_[first_outcome_end - 1].cumulative_probability >= 1.0) {
            for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
                for (uint32_t site = 0; site < end; ++site) {
                    activate_noise_site(lane, site);
                }
            }
            if (plan_->presampled_initialization_level_offsets_.empty()) {
                for (uint32_t symbol : plan_->presampled_symbols_) {
                    propagate_symbol(symbol);
                }
            } else {
                initialize_presampled_expressions();
            }
            return;
        }
        const double inverse_hazard = *plan_->uniform_noise_inverse_hazard_;
        const uint64_t total_draws = static_cast<uint64_t>(active_lanes_) * end;
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
        if (plan_->presampled_initialization_level_offsets_.empty()) {
            for (uint32_t symbol : plan_->presampled_symbols_) {
                propagate_symbol(symbol);
            }
        } else {
            initialize_presampled_expressions();
        }
        return;
    }
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
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
    if (plan_->presampled_initialization_level_offsets_.empty()) {
        for (uint32_t symbol : plan_->presampled_symbols_) {
            propagate_symbol(symbol);
        }
    } else {
        initialize_presampled_expressions();
    }
}

void BatchExecutor::assign_forced_faults(KFaultSampler& fault_sampler) noexcept {
    assert(fault_sampler.num_sites() ==
               plan_->noise_sites_.size() + plan_->num_readout_noise_sites_ &&
           "conditioned batch sampler must cover every fault site");
    const uint32_t quantum_sites = static_cast<uint32_t>(plan_->noise_sites_.size());
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
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
    if (plan_->presampled_initialization_level_offsets_.empty()) {
        for (uint32_t symbol : plan_->presampled_symbols_) {
            propagate_symbol(symbol);
        }
    } else {
        initialize_presampled_expressions();
    }
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
    if (plan_->batch_noise_outcomes_.empty()) {
        const uint32_t symbol = plan_->noise_outcomes_[outcome_index].symbol;
        assert(!symbols_.bit(symbol, lane) && "noise site must define a fresh lane symbol");
        symbols_.set_bit(symbol, lane);
        return;
    }

    const ExecutablePlan::PreparedBatchNoiseOutcome& batch_outcome =
        plan_->batch_noise_outcomes_[outcome_index];
    const uint32_t assignment_end = batch_outcome.assignment_begin + batch_outcome.assignment_count;
    assert(assignment_end <= plan_->batch_noise_assignments_.size() &&
           "batch noise assignment must stay in its prepared tape");
    for (uint32_t assignment = batch_outcome.assignment_begin; assignment < assignment_end;
         ++assignment) {
        const uint32_t symbol = plan_->batch_noise_assignments_[assignment];
        assert(!symbols_.bit(symbol, lane) && "batch noise carrier must be assigned once per site");
        symbols_.set_bit(symbol, lane);
    }
}

void BatchExecutor::propagate_symbol(uint32_t symbol) noexcept {
    const std::span<const uint64_t> values = symbols_.column(symbol);
    for (uint32_t register_id : plan_->expression_dependencies_.dependent_registers(symbol)) {
        expression_registers_.xor_into(register_id, values);
    }
}

void BatchExecutor::assign_symbol(uint32_t symbol, std::span<const uint64_t> values) noexcept {
    symbols_.assign(symbol, values, live_words_);
    propagate_symbol(symbol);
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
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        lane_bytes_[lane] = static_cast<uint8_t>(lane_bit(signs, lane));
    }
    prepare_interleaved_rotation_sines(signed_sines_, action.rotation.sine,
                                       std::span<const uint8_t>(lane_bytes_).first(active_lanes_));
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
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
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
        std::span<const uint8_t>(lane_bytes_).first(active_lanes_));
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecutePromotion& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> signs = evaluate(action.sign);
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        lane_bytes_[lane] = static_cast<uint8_t>(lane_bit(signs, lane));
    }
    prepare_interleaved_rotation_sines(signed_sines_, action.promotion.sine,
                                       std::span<const uint8_t>(lane_bytes_).first(active_lanes_));
    apply_interleaved_promotion(state_, action.promotion, signed_sines_);
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    interleaved_measurement_probabilities(state_, action.measurement, probability_zero_,
                                          probability_one_);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        const MeasurementProbabilities probabilities{probability_zero_[lane],
                                                     probability_one_[lane]};
        const MeasurementBranchClassification classification =
            classify_measurement_branch(probabilities);
        bool branch = classification.kind == MeasurementBranchKind::DeterministicOne;
        if (is_live(lane)) {
            switch (classification.kind) {
                case MeasurementBranchKind::Random:
                    branch = rng_.next_double() * probabilities.total() >= probabilities.zero;
                    break;
                case MeasurementBranchKind::DeterministicZero:
                    dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
                    branch = false;
                    break;
                case MeasurementBranchKind::DeterministicOne:
                    dust_clamps_ += static_cast<uint64_t>(classification.clamped_dust);
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
                                     std::span<const uint8_t>(lane_bytes_).first(active_lanes_),
                                     std::span<const double>(lane_values_).first(active_lanes_));
    assign_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, corrections, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    fill_random_half_bits();
    assign_symbol(action.branch, scratch_words_);
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
    assign_symbol(action.symbol, evaluate(action.value));
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> sources = evaluate(action.source);
    std::ranges::fill(scratch_words_, uint64_t{0});
    bool sampled_packed = false;
    if (fixed_fault_mode_) {
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
            while (lane < active_lanes_) {
                const double gap =
                    -std::log(1.0 - rng_.next_double()) * action.batch_symmetric_inverse_hazard;
                if (gap >= static_cast<double>(active_lanes_ - lane)) {
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
        for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
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
    assign_symbol(action.flip, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, sources, scratch_words_, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDetector& action,
                                   size_t action_index) noexcept {
    const std::span<const uint64_t> outcomes =
        action.record_parity == std::numeric_limits<uint32_t>::max()
            ? evaluate(action.outcome)
            : evaluate_record_parity(action.record_parity);
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
    const std::span<const uint64_t> outcomes =
        action.record_parity == std::numeric_limits<uint32_t>::max()
            ? evaluate(action.outcome)
            : evaluate_record_parity(action.record_parity);
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
    const std::span<const uint64_t> signs = evaluate(action.sign);
    if (!action.active_projection.has_value()) {
        for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
            if (is_live(lane)) {
                output[lane] = 0.0;
            }
        }
        return;
    }
    interleaved_expectation_values(state_, *action.active_projection, lane_values_);
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
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

std::span<const uint64_t> BatchExecutor::evaluate_record_parity(uint32_t parity_index) noexcept {
    assert(parity_index < plan_->batch_record_parities_.size() &&
           "prepared record parity must be in range");
    const ExecutablePlan::PreparedRecordParity& parity =
        plan_->batch_record_parities_[parity_index];
    const uint32_t end = parity.begin + parity.count;
    assert(end <= plan_->batch_record_parity_terms_.size() &&
           "prepared record parity must stay in its term tape");
    const size_t words = packed_word_count(active_lanes_);
    if (parity.constant) {
        std::ranges::copy(std::span<const uint64_t>(live_words_).first(words),
                          scratch_words_.begin());
    } else {
        std::ranges::fill(std::span<uint64_t>(scratch_words_).first(words), uint64_t{0});
    }
    for (uint32_t term = parity.begin; term < end; ++term) {
        const std::span<const uint64_t> record =
            records_.column(plan_->batch_record_parity_terms_[term]);
        for (size_t word = 0; word < words; ++word) {
            scratch_words_[word] ^= record[word];
        }
    }
    return scratch_words_;
}

bool BatchExecutor::lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept {
    assert(bits.size() >= word_capacity_ && lane < active_lanes_ &&
           "packed expression lookup must reference an active lane");
    return ((bits[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

bool BatchExecutor::is_live(uint32_t lane) const noexcept {
    assert(lane < active_lanes_ && "live lookup must reference the current lane span");
    return ((live_words_[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

bool BatchExecutor::should_compact(size_t action_index) const noexcept {
    if (live_count_ == 0 || live_count_ == active_lanes_) {
        return false;
    }
    const uint64_t remaining_actions = plan_->actions_.size() - action_index - 1;
    if (remaining_actions == 0) {
        return false;
    }
    const uint64_t old_words = packed_word_count(active_lanes_);
    const uint64_t new_words = packed_word_count(live_count_);
    const uint64_t dead_lanes = active_lanes_ - live_count_;
    const uint64_t bit_columns = symbols_.num_columns() + expression_registers_.num_columns() +
                                 records_.num_columns() + detectors_.num_columns() +
                                 observables_.num_columns() + forced_readout_.num_columns();
    const uint64_t carry_cost =
        dead_lanes * remaining_actions + (old_words - new_words) * remaining_actions * 8;
    const uint64_t compact_cost = bit_columns * old_words +
                                  static_cast<uint64_t>(plan_->num_exp_vals_) * live_count_ +
                                  static_cast<uint64_t>(live_count_) * 3;
    return carry_cost > compact_cost;
}

void BatchExecutor::compact_live_lanes() noexcept {
    if (live_count_ == active_lanes_) {
        return;
    }
    if (live_count_ == 0) {
        state_.compact_lanes({});
        active_lanes_ = 0;
        std::ranges::fill(live_words_, uint64_t{0});
        ++compactions_;
        return;
    }
    const uint32_t old_lanes = active_lanes_;
    uint32_t destination = 0;
    for (uint32_t source = 0; source < old_lanes; ++source) {
        if (is_live(source)) {
            compaction_sources_[destination++] = source;
        }
    }
    assert(destination == live_count_ && "lane compaction must retain every live context");
    const std::span<const uint32_t> sources(compaction_sources_.data(), live_count_);
    symbols_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    expression_registers_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    records_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    detectors_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    observables_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    forced_readout_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
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
    active_lanes_ = live_count_;
    fill_low_lane_mask(live_words_, active_lanes_);
    ++compactions_;
}

uint32_t BatchExecutor::accumulate_survivor_counts(
    std::span<uint64_t> observable_ones) const noexcept {
    assert(output_mode_ == BatchOutputMode::AggregateSurvivors &&
           observable_ones.size() == plan_->num_observables_ &&
           "aggregate survivor outputs must match the prepared observable count");
    uint32_t logical_errors = 0;
    const size_t words = packed_word_count(active_lanes_);
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
    if (live_count_ != active_lanes_) {
        compact_live_lanes();
    }
}

uint32_t BatchExecutor::shot_index(uint32_t lane) const noexcept {
    assert(lane < live_count_ && active_lanes_ == live_count_ &&
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
