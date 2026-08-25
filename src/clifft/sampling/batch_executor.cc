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
#include <type_traits>

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

}  // namespace

uint32_t resolve_batch_capacity(const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers,
                                uint32_t intra_shot_workers,
                                std::optional<uint32_t> requested_batch_size) {
    if (requested_batch_size.has_value() && *requested_batch_size == 0) {
        throw std::invalid_argument("batch_size must be a positive integer or 'auto'");
    }
    if (shots == 0 || shot_workers == 0 || plan.has_instruments()) {
        return 1;
    }
    if (intra_shot_workers > 1) {
        if (requested_batch_size.has_value() && *requested_batch_size > 1) {
            throw std::invalid_argument(
                "packed batch_size is incompatible with intra-shot workers");
        }
        return 1;
    }
#if defined(__EMSCRIPTEN__)
    if (requested_batch_size.has_value() && *requested_batch_size > 1) {
        throw std::invalid_argument("packed batch_size is unavailable in WebAssembly builds");
    }
    return 1;
#else
    const uint32_t worker_shots =
        static_cast<uint32_t>((static_cast<uint64_t>(shots) + shot_workers - 1) / shot_workers);
    if (requested_batch_size.has_value()) {
        return std::max(uint32_t{1},
                        std::min({*requested_batch_size, worker_shots, kMaxExplicitBatchShots}));
    }
    if (worker_shots < kDefaultMinAutoBatchShots) {
        return 1;
    }
    // Dense action-major lane traversal and per-lane fault presampling do not
    // yet beat the shot-major executor reliably. Keep those plans scalar until
    // a benchmark-supported crossover is available; explicit capacities remain
    // useful for profiling and correctness coverage of the complete executor.
    if (plan.peak_active_width() != 0 || plan.num_presampled_symbols() != 0 ||
        plan.has_readout_noise()) {
        return 1;
    }
    const size_t state_bytes = State::allocation_bytes_for(plan.peak_active_width());
    const size_t footprint_capacity = std::max<size_t>(1, kDefaultBatchStateBudget / state_bytes);
    return static_cast<uint32_t>(
        std::min<size_t>({worker_shots, kDefaultMaxAutoBatchShots, footprint_capacity}));
#endif
}

BatchExecutor::BatchExecutor(const ExecutablePlan& plan, uint32_t lane_capacity,
                             BatchOutputMode output_mode)
    : plan_(validate_batch_plan(plan)),
      output_mode_(output_mode),
      lane_capacity_(lane_capacity),
      word_capacity_(packed_word_count(lane_capacity)),
      state_bytes_per_lane_(State::allocation_bytes_for(plan.peak_active_width_)),
      state_storage_(checked_product(state_bytes_per_lane_, lane_capacity, "state")),
      state_slots_(lane_capacity),
      rngs_(lane_capacity),
      shot_indices_(lane_capacity),
      symbols_(plan.num_symbols_, lane_capacity),
      expression_registers_(plan.expression_register_constants_.size(), lane_capacity),
      records_(output_mode == BatchOutputMode::Rows
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
      compaction_scratch_(word_capacity_, 0) {
    if (lane_capacity_ == 0) {
        throw std::invalid_argument("packed sampling lane capacity must be positive");
    }
    states_.reserve(lane_capacity_);
    auto* storage = static_cast<std::byte*>(state_storage_.data());
    for (uint32_t lane = 0; lane < lane_capacity_; ++lane) {
        states_.push_back(State::from_borrowed_storage(
            plan.peak_active_width_, plan.initial_active_width_,
            storage + static_cast<size_t>(lane) * state_bytes_per_lane_, state_bytes_per_lane_));
    }
}

void BatchExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept {
    fixed_fault_mode_ = false;
    reset_batch(root, first_shot, shots);
    sample_presampled_noise();
    [[maybe_unused]] bool executed = false;
    switch (plan_->backend_) {
        case ExecutorBackend::Scalar:
            execute_actions<ExecutorBackend::Scalar>();
            executed = true;
            break;
        case ExecutorBackend::Avx2:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
            execute_actions<ExecutorBackend::Avx2>();
            executed = true;
            break;
#else
            break;
#endif
        case ExecutorBackend::Avx512:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
            execute_actions<ExecutorBackend::Avx512>();
            executed = true;
            break;
#else
            break;
#endif
    }
    assert(executed && "unhandled packed sampling executor backend");
    if (output_mode_ == BatchOutputMode::Rows) {
        finalize_live_lanes();
    }
}

void BatchExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                              KFaultSampler& fault_sampler) noexcept {
    fixed_fault_mode_ = true;
    reset_batch(root, first_shot, shots);
    assign_forced_faults(fault_sampler);
    [[maybe_unused]] bool executed = false;
    switch (plan_->backend_) {
        case ExecutorBackend::Scalar:
            execute_actions<ExecutorBackend::Scalar>();
            executed = true;
            break;
        case ExecutorBackend::Avx2:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
            execute_actions<ExecutorBackend::Avx2>();
            executed = true;
            break;
#else
            break;
#endif
        case ExecutorBackend::Avx512:
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
            execute_actions<ExecutorBackend::Avx512>();
            executed = true;
            break;
#else
            break;
#endif
    }
    assert(executed && "unhandled packed sampling executor backend");
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
    for (uint32_t lane = 0; lane < shots; ++lane) {
        state_slots_[lane] = lane;
        states_[lane].reset();
        shot_indices_[lane] = first_shot + lane;
        const std::array<uint64_t, 4> words =
            derive_state(root, shot_indices_[lane], kSamplingExecutorDomain);
        rngs_[lane].seed_full(words[0], words[1], words[2], words[3]);
    }
    initialize_expression_registers();
}

void BatchExecutor::initialize_expression_registers() noexcept {
    for (size_t expression = 0; expression < plan_->expression_register_constants_.size();
         ++expression) {
        if (plan_->expression_register_constants_[expression] != 0) {
            expression_registers_.assign(expression, live_words_, live_words_);
        }
    }
}

void BatchExecutor::sample_presampled_noise() noexcept {
    const uint32_t end = static_cast<uint32_t>(plan_->noise_sites_.size());
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        uint32_t first_candidate = 0;
        while (first_candidate < end) {
            const double current_hazard =
                first_candidate == 0 ? 0.0 : plan_->noise_hazards_[first_candidate - 1];
            if (current_hazard >= plan_->noise_hazards_[end - 1]) {
                break;
            }
            const uint32_t site = sample_next_noise_site(plan_->noise_hazards_, first_candidate,
                                                         rngs_[lane].next_double());
            if (site == kNoNoiseSite || site >= end) {
                break;
            }
            activate_noise_site(lane, site);
            first_candidate = site + 1;
        }
    }
    for (uint32_t symbol : plan_->presampled_symbols_) {
        propagate_symbol(symbol);
    }
}

void BatchExecutor::assign_forced_faults(KFaultSampler& fault_sampler) noexcept {
    assert(fault_sampler.num_sites() ==
               plan_->noise_sites_.size() + plan_->num_readout_noise_sites_ &&
           "conditioned batch sampler must cover every fault site");
    const uint32_t quantum_sites = static_cast<uint32_t>(plan_->noise_sites_.size());
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        const std::span<const uint32_t> selected =
            fault_sampler.sample([&]() noexcept { return rngs_[lane].next_double(); });
        for (uint32_t site : selected) {
            if (site < quantum_sites) {
                activate_noise_site(lane, site);
            } else {
                forced_readout_.set_bit(site - quantum_sites, lane);
            }
        }
    }
    for (uint32_t symbol : plan_->presampled_symbols_) {
        propagate_symbol(symbol);
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
        const double draw = rngs_[lane].next_double() * execution_probability;
        while (draw >= plan_->noise_outcomes_[outcome_index].cumulative_probability) {
            ++outcome_index;
            assert(outcome_index < outcome_end && "channel draw must select a prepared outcome");
        }
    }
    const uint32_t symbol = plan_->noise_outcomes_[outcome_index].symbol;
    assert(!symbols_.bit(symbol, lane) && "noise site must define a fresh lane symbol");
    symbols_.set_bit(symbol, lane);
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

template <ExecutorBackend Backend>
void BatchExecutor::execute_actions() noexcept {
    for (size_t action_index = 0; action_index < plan_->actions_.size(); ++action_index) {
        const ExecutablePlan::Action& action = plan_->actions_[action_index];
        std::visit(
            [&](const auto& typed) noexcept {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                    execute_action<Backend>(typed, action_index);
                } else {
                    execute_action(typed, action_index);
                }
            },
            action);
        if (live_count_ == 0) {
            return;
        }
    }
}

template <ExecutorBackend Backend>
void BatchExecutor::execute_action(const ExecutablePlan::ExecuteRotation& action, size_t) noexcept {
    const std::span<const uint64_t> signs = evaluate(action.sign);
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (!is_live(lane)) {
            continue;
        }
        const bool sign = lane_bit(signs, lane);
        if (action.kernel == DirectRotationKernel::Scalar) {
            apply_rotation(state(lane), action.rotation, sign);
        } else if constexpr (Backend == ExecutorBackend::Avx2) {
            apply_direct_rotation_avx2(state(lane), action.rotation, action.kernel, sign);
        } else if constexpr (Backend == ExecutorBackend::Avx512) {
            apply_direct_rotation_avx512(state(lane), action.rotation, action.kernel, sign);
        } else {
            assert(false && "scalar batch executor requires scalar rotation actions");
        }
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                                   size_t) noexcept {
    assert(action.rotation_index < plan_->fused_rotations_.size() &&
           "fused rotation action must reference prepared execution");
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (is_live(lane)) {
            plan_->fused_rotations_[action.rotation_index].apply(state(lane));
        }
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action,
                                   size_t) noexcept {
    assert(action.rotation_index < plan_->dynamic_fused_rotations_.size() &&
           "dynamic fused rotation action must reference prepared execution");
    const auto& rotation = plan_->dynamic_fused_rotations_[action.rotation_index];
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (!is_live(lane)) {
            continue;
        }
        uint32_t variant = 0;
        for (size_t basis = 0; basis < rotation.sign_basis.size(); ++basis) {
            variant |= static_cast<uint32_t>(lane_bit(evaluate(rotation.sign_basis[basis]), lane))
                       << basis;
        }
        assert(variant < rotation.variants.size() && "dynamic sign must select a prepared variant");
        rotation.variants[variant].apply(state(lane));
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecutePromotion& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> signs = evaluate(action.sign);
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (is_live(lane)) {
            apply_promotion(state(lane), action.promotion, lane_bit(signs, lane));
        }
    }
}

template <ExecutorBackend Backend>
void BatchExecutor::execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (!is_live(lane)) {
            continue;
        }
        const MeasurementProbabilities probabilities = [&]() noexcept {
            if (action.kernel == ActiveMeasurementKernel::Scalar) {
                return measurement_probabilities(state(lane), action.measurement);
            }
            if constexpr (Backend == ExecutorBackend::Avx2) {
                return active_measurement_probabilities_avx2(state(lane), action.measurement,
                                                             action.kernel);
            } else if constexpr (Backend == ExecutorBackend::Avx512) {
                return active_measurement_probabilities_avx512(state(lane), action.measurement,
                                                               action.kernel);
            } else {
                assert(false && "scalar batch executor requires scalar measurement actions");
                return measurement_probabilities(state(lane), action.measurement);
            }
        }();
        const bool branch = sample_active_branch(lane, probabilities);
        if (branch) {
            scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
        }
        const double branch_probability = probabilities.for_branch(branch);
        if (action.kernel == ActiveMeasurementKernel::Scalar) {
            collapse_measurement(state(lane), action.measurement, branch, branch_probability);
        } else if constexpr (Backend == ExecutorBackend::Avx2) {
            collapse_active_measurement_avx2(state(lane), action.measurement, action.kernel, branch,
                                             branch_probability);
        } else if constexpr (Backend == ExecutorBackend::Avx512) {
            collapse_active_measurement_avx512(state(lane), action.measurement, action.kernel,
                                               branch, branch_probability);
        }
    }
    assign_symbol(action.branch, scratch_words_);
    if (output_mode_ == BatchOutputMode::Rows) {
        records_.assign_xor(action.record, scratch_words_, corrections, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                                   size_t) noexcept {
    const std::span<const uint64_t> corrections = evaluate(action.correction);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (is_live(lane) && rngs_[lane].next_double() >= 0.5) {
            scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
        }
    }
    assign_symbol(action.branch, scratch_words_);
    if (output_mode_ == BatchOutputMode::Rows) {
        records_.assign_xor(action.record, scratch_words_, corrections, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                                   size_t) noexcept {
    if (output_mode_ == BatchOutputMode::Rows) {
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
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (!is_live(lane)) {
            continue;
        }
        const bool source = lane_bit(sources, lane);
        const bool flip =
            fixed_fault_mode_ ? forced_readout_.bit(action.site, lane) : [&]() noexcept {
                const double probability =
                    source ? action.prob_one_to_zero : action.prob_zero_to_one;
                return probability >= 1.0 ||
                       (probability > 0.0 && rngs_[lane].next_double() < probability);
            }();
        if (flip) {
            scratch_words_[lane >> 6] |= uint64_t{1} << (lane & 63);
        }
    }
    assign_symbol(action.flip, scratch_words_);
    if (output_mode_ == BatchOutputMode::Rows) {
        records_.assign_xor(action.record, sources, scratch_words_, live_words_);
    }
}

void BatchExecutor::execute_action(const ExecutablePlan::ExecuteDetector& action,
                                   size_t action_index) noexcept {
    const std::span<const uint64_t> outcomes = evaluate(action.outcome);
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
    observables_.assign(action.observable, evaluate(action.outcome), live_words_);
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
    for (uint32_t lane = 0; lane < active_lanes_; ++lane) {
        if (!is_live(lane)) {
            continue;
        }
        if (!action.active_projection.has_value()) {
            output[lane] = 0.0;
        } else {
            const double value = expectation_value(state(lane), *action.active_projection);
            output[lane] = lane_bit(signs, lane) ? -value : value;
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

bool BatchExecutor::lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept {
    assert(bits.size() >= word_capacity_ && lane < active_lanes_ &&
           "packed expression lookup must reference an active lane");
    return ((bits[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

bool BatchExecutor::is_live(uint32_t lane) const noexcept {
    assert(lane < active_lanes_ && "live lookup must reference the current lane span");
    return ((live_words_[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

State& BatchExecutor::state(uint32_t lane) noexcept {
    assert(lane < active_lanes_ && state_slots_[lane] < states_.size() &&
           "batch lane must map to retained state storage");
    return states_[state_slots_[lane]];
}

const State& BatchExecutor::state(uint32_t lane) const noexcept {
    assert(lane < active_lanes_ && state_slots_[lane] < states_.size() &&
           "batch lane must map to retained state storage");
    return states_[state_slots_[lane]];
}

bool BatchExecutor::sample_active_branch(uint32_t lane,
                                         MeasurementProbabilities probabilities) noexcept {
    const MeasurementBranchClassification classification =
        classify_measurement_branch(probabilities);
    switch (classification.kind) {
        case MeasurementBranchKind::Random:
            return rngs_[lane].next_double() * probabilities.total() >= probabilities.zero;
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
        active_lanes_ = 0;
        std::ranges::fill(live_words_, uint64_t{0});
        ++compactions_;
        return;
    }
    const uint32_t old_lanes = active_lanes_;
    symbols_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    expression_registers_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    records_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    detectors_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    observables_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);
    forced_readout_.compact(live_words_, old_lanes, live_count_, compaction_scratch_);

    uint32_t destination = 0;
    for (uint32_t source = 0; source < old_lanes; ++source) {
        if (!is_live(source)) {
            continue;
        }
        if (destination != source) {
            state_slots_[destination] = state_slots_[source];
            rngs_[destination] = rngs_[source];
            shot_indices_[destination] = shot_indices_[source];
            for (uint32_t exp_val = 0; exp_val < exp_vals_.size() / lane_capacity_; ++exp_val) {
                exp_vals_[static_cast<size_t>(exp_val) * lane_capacity_ + destination] =
                    exp_vals_[static_cast<size_t>(exp_val) * lane_capacity_ + source];
            }
        }
        ++destination;
    }
    assert(destination == live_count_ && "lane compaction must retain every live context");
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
