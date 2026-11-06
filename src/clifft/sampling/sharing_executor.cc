#include "clifft/sampling/sharing_executor.h"

#include "clifft/sampling/measurement_branch.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <stdexcept>
#include <type_traits>

namespace clifft::sampling {

SharingExecutor::SharingExecutor(const ExecutablePlan& plan, const PreparedBatchNoise& noise,
                                 const EarlyPostselection& checks, uint32_t lane_capacity,
                                 uint32_t initial_states, bool share)
    : BatchClassicalState(plan, BatchOutputMode::AggregateSurvivors, BatchSamplingMode::Ordinary,
                          batch_detail::batch_worker_storage_layout(
                              plan, lane_capacity, BatchOutputMode::AggregateSurvivors,
                              BatchSamplingMode::Ordinary, noise.mode()),
                          &noise),
      checks_(&checks),
      share_(share),
      shot_rng_(lane_capacity),
      heads_(lane_capacity, kNoShot),
      next_(lane_capacity, kNoShot),
      groups_(lane_capacity),
      free_(lane_capacity),
      choices_(lane_capacity) {
    if (checks.plan_ != &plan) {
        throw std::invalid_argument("entrance checks must belong to the sharing plan");
    }
    if (lane_capacity == 0 || initial_states == 0) {
        throw std::invalid_argument("sharing requires positive lane and initial state capacities");
    }
    states_.reserve(std::min(initial_states, lane_capacity));
    grow_state_pool(std::min(initial_states, lane_capacity));
    split_factors_.reserve(plan.actions_.size());
    for (const auto& action : plan.actions_) {
        split_factors_.push_back(std::visit(
            [&](const auto& typed) -> uint32_t {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation> ||
                              std::is_same_v<T, ExecutablePlan::ExecutePromotion> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                    return 2;
                } else if constexpr (std::is_same_v<T,
                                                    ExecutablePlan::ExecuteDynamicFusedRotation>) {
                    return uint32_t{1}
                           << plan.dynamic_fused_rotations_[typed.rotation_index].sign_basis.size();
                } else {
                    return 1;
                }
            },
            action));
    }
}

void SharingExecutor::grow_state_pool(uint32_t required) {
    assert(required <= lane_capacity_ && "shared states cannot outnumber classical lanes");
    // This is an explicit suspension boundary. Live coefficients and sampled
    // outcomes remain intact while storage grows, then the same action resumes.
    while (states_.size() < required) {
        const uint32_t slot = static_cast<uint32_t>(states_.size());
        states_.emplace_back(plan_->peak_active_width(), plan_->initial_active_width());
        free_[free_count_++] = slot;
    }
}

void SharingExecutor::run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) {
    reset_classical(root, first_shot, shots);
    sample_presampled_noise();
    apply_early_postselection(*checks_);
    entrance_rejected_ = shots - live_count_;
    storage_boundaries_ = 0;
    num_splits_ = 0;
    peak_groups_ = 0;
    group_count_ = 0;
    prune_pending_ = false;
    action_index_ = 0;
    if (live_count_ == 0) {
        return;
    }
    const uint32_t initial_groups = share_ ? 1 : live_count_;
    if (initial_groups > states_.size()) {
        ++storage_boundaries_;
        grow_state_pool(initial_groups);
    }
    group_count_ = initial_groups;
    peak_groups_ = group_count_;
    free_count_ = static_cast<uint32_t>(states_.size()) - group_count_;
    for (uint32_t index = 0; index < free_count_; ++index) {
        free_[index] = static_cast<uint32_t>(states_.size()) - 1 - index;
    }
    for (uint32_t group = 0; group < group_count_; ++group) {
        groups_[group] = group;
        heads_[group] = kNoShot;
        states_[group].reset();
    }
    uint32_t independent_group = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (!is_live(shot)) {
            continue;
        }
        const uint32_t group = share_ ? 0 : independent_group++;
        next_[shot] = heads_[group];
        heads_[group] = shot;
        const auto words =
            derive_state(root, uint64_t{first_shot} + shot, kSharedSamplingExecutorDomain);
        shot_rng_[shot].seed_full(words[0], words[1], words[2], words[3]);
    }
    while (const uint32_t required = dispatch_until_storage_boundary()) {
        ++storage_boundaries_;
        grow_state_pool(required);
    }
}

uint32_t SharingExecutor::dispatch_until_storage_boundary() noexcept {
    while (action_index_ < plan_->actions_.size() && live_count_ != 0) {
        const auto& action = plan_->actions_[action_index_];
        std::visit(
            [&](const auto& typed) noexcept {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (!std::is_same_v<T, ExecutablePlan::ExecuteDetector> &&
                              !std::is_same_v<T, ExecutablePlan::ExecuteClassicalRecord> &&
                              !std::is_same_v<T, ExecutablePlan::ExecuteSymbolDefinition> &&
                              !std::is_same_v<T, ExecutablePlan::ExecuteObservable>) {
                    if (prune_pending_) {
                        prune();
                        prune_pending_ = false;
                    }
                }
            },
            action);
        const uint32_t required = static_cast<uint32_t>(std::min<uint64_t>(
            live_count_, uint64_t{group_count_} * split_factors_[action_index_]));
        if (required > states_.size()) {
            return required;
        }
        std::visit([&](const auto& typed) noexcept { execute_action(typed); }, action);
        ++action_index_;
    }
    return 0;
}

void SharingExecutor::copy_state(uint32_t to, uint32_t from) noexcept {
    states_[to].set_active_width(states_[from].active_width());
    std::ranges::copy(states_[from].real(), states_[to].real().begin());
    std::ranges::copy(states_[from].imag(), states_[to].imag().begin());
}

uint32_t SharingExecutor::split(uint32_t group) noexcept {
    uint32_t heads[2] = {kNoShot, kNoShot}, count[2] = {0, 0};
    for (uint32_t shot = heads_[group]; shot != kNoShot;) {
        const uint32_t next = next_[shot], value = choices_[shot];
        next_[shot] = heads[value];
        heads[value] = shot;
        ++count[value];
        shot = next;
    }
    if (count[0] == 0 || count[1] == 0) {
        heads_[group] = heads[count[1] != 0];
        return kNoShot;
    }
    assert(free_count_ != 0 && "state split must fit the preflight capacity");
    const uint32_t other = free_[--free_count_];
    copy_state(other, group);
    heads_[group] = heads[0];
    heads_[other] = heads[1];
    groups_[group_count_++] = other;
    ++num_splits_;
    peak_groups_ = std::max(peak_groups_, group_count_);
    return other;
}

void SharingExecutor::prune() noexcept {
    uint32_t kept = 0;
    for (uint32_t index = 0; index < group_count_; ++index) {
        const auto group = groups_[index];
        uint32_t head = kNoShot;
        for (uint32_t shot = heads_[group]; shot != kNoShot;) {
            const uint32_t next = next_[shot];
            if (lane_bit(live_words_, shot)) {
                next_[shot] = head;
                head = shot;
            }
            shot = next;
        }
        heads_[group] = head;
        if (head != kNoShot) {
            groups_[kept++] = group;
        } else {
            free_[free_count_++] = group;
        }
    }
    group_count_ = kept;
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteRotation& action) noexcept {
    const auto values = evaluate(action.sign);
    sign_action(values, [&](uint32_t group, bool sign) {
        apply_rotation(states_[group], action.rotation, sign);
    });
}

template <class F>
void SharingExecutor::sign_action(std::span<const uint64_t> values, F&& apply) noexcept {
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        for (uint32_t shot = heads_[group]; shot != kNoShot; shot = next_[shot]) {
            choices_[shot] = lane_bit(values, shot);
        }
        const auto other = split(group);
        apply(group, choices_[heads_[group]]);
        if (other != kNoShot) {
            apply(other, true);
        }
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecutePromotion& action) noexcept {
    sign_action(evaluate(action.sign), [&](uint32_t group, bool sign) {
        apply_promotion(states_[group], action.promotion, sign);
    });
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteFusedRotation& action) noexcept {
    for (uint32_t index = 0; index < group_count_; ++index) {
        auto group = groups_[index];
        plan_->fused_rotations_[action.rotation_index].apply(states_[group]);
    }
}

void SharingExecutor::execute_action(
    const ExecutablePlan::ExecuteDynamicFusedRotation& action) noexcept {
    const auto& rotation = plan_->dynamic_fused_rotations_[action.rotation_index];
    for (const auto& sign : rotation.sign_basis) {
        auto values = evaluate(sign);
        const uint32_t original_group_count = group_count_;
        for (uint32_t index = 0; index < original_group_count; ++index) {
            auto group = groups_[index];
            for (uint32_t shot = heads_[group]; shot != kNoShot; shot = next_[shot]) {
                choices_[shot] = lane_bit(values, shot);
            }
            (void)split(group);
        }
    }
    for (uint32_t index = 0; index < group_count_; ++index) {
        auto group = groups_[index];
        uint32_t value = 0;
        for (size_t sign_index = 0; sign_index < rotation.sign_basis.size(); ++sign_index) {
            value |= uint32_t(lane_bit(evaluate(rotation.sign_basis[sign_index]), heads_[group]))
                     << sign_index;
        }
        rotation.variants[value].apply(states_[group]);
    }
}

void SharingExecutor::execute_action(
    const ExecutablePlan::ExecuteActiveMeasurement& action) noexcept {
    std::ranges::fill(scratch_words_, uint64_t{0});
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        const auto probabilities = measurement_probabilities(states_[group], action.measurement);
        const auto kind = classify_measurement_branch(probabilities).kind;
        for (uint32_t shot = heads_[group]; shot != kNoShot; shot = next_[shot]) {
            const bool branch =
                kind == MeasurementBranchKind::Random
                    ? shot_rng_[shot].next_double() * probabilities.total() >= probabilities.zero
                    : kind == MeasurementBranchKind::DeterministicOne;
            choices_[shot] = branch;
            if (branch) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
        }
        const auto other = split(group);
        const bool branch = choices_[heads_[group]];
        collapse_measurement(states_[group], action.measurement, branch,
                             probabilities.for_branch(branch));
        if (other != kNoShot) {
            collapse_measurement(states_[other], action.measurement, true, probabilities.one);
        }
    }
    propagate_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, evaluate(action.correction),
                            live_words_);
    }
}

void SharingExecutor::execute_action(
    const ExecutablePlan::ExecuteDormantMeasurement& action) noexcept {
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t index = 0; index < group_count_; ++index) {
        for (uint32_t shot = heads_[groups_[index]]; shot != kNoShot; shot = next_[shot]) {
            if (shot_rng_[shot].next_double() < 0.5) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
        }
    }
    propagate_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, evaluate(action.correction),
                            live_words_);
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteDetector& action) noexcept {
    const auto values = evaluate_record_parity(action.outcome);
    if (action.postselected) {
        for (size_t word = 0; word < values.size(); ++word) {
            const auto dead = values[word] & live_words_[word];
            live_count_ -= std::popcount(dead);
            live_words_[word] &= ~dead;
        }
        prune_pending_ = true;
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteReadoutNoise& action) noexcept {
    const auto values = evaluate(action.source);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t index = 0; index < group_count_; ++index) {
        for (uint32_t shot = heads_[groups_[index]]; shot != kNoShot; shot = next_[shot]) {
            const double probability =
                lane_bit(values, shot) ? action.prob_one_to_zero : action.prob_zero_to_one;
            if (probability >= 1 ||
                (probability > 0 && shot_rng_[shot].next_double() < probability)) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
        }
    }
    propagate_symbol(action.flip, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.xor_into(action.record, scratch_words_);
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteExpectation&) noexcept {}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteInstrument&) noexcept {
    assert(false && "instrument actions are excluded during sharing setup");
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteBoundary&) noexcept {
    assert(false && "continuation boundaries are excluded during sharing setup");
}

}  // namespace clifft::sampling
