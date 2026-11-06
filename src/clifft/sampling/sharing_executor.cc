#include "clifft/sampling/sharing_executor.h"

#include "clifft/sampling/measurement_branch.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <map>
#include <stdexcept>
#include <type_traits>

namespace clifft::sampling {

SharingKey::SharingKey(const ExecutablePlan& plan) : plan_(&plan) {
    std::vector<uint8_t> required(plan.num_expression_registers());
    for (const auto& action : plan.actions_) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation> ||
                              std::is_same_v<T, ExecutablePlan::ExecutePromotion>) {
                    required[typed.sign.register_id] = 1;
                } else if constexpr (std::is_same_v<T,
                                                    ExecutablePlan::ExecuteDynamicFusedRotation>) {
                    for (const auto& sign :
                         plan.dynamic_fused_rotations_[typed.rotation_index].sign_basis) {
                        required[sign.register_id] = 1;
                    }
                }
            },
            action);
    }
    // A derived sign input is uniform only when its defining expression is
    // uniform. Definitions are ordered, so one reverse pass closes this set.
    for (auto it = plan.actions_.rbegin(); it != plan.actions_.rend(); ++it) {
        if (const auto* definition = std::get_if<ExecutablePlan::ExecuteSymbolDefinition>(&*it)) {
            for (auto reg : plan.expression_dependencies_.dependent_registers(definition->symbol)) {
                if (required[reg]) {
                    required[definition->value.register_id] = 1;
                    break;
                }
            }
        }
    }
    std::vector<AffineBool> rows(required.size());
    for (uint32_t symbol : plan.presampled_symbols_) {
        for (uint32_t reg : plan.expression_dependencies_.dependent_registers(symbol)) {
            if (required[reg])
                rows[reg] ^= AffineBool::symbol(SymbolId{symbol});
        }
    }
    std::map<uint32_t, AffineBool> basis;
    for (uint32_t reg = 0; reg < rows.size(); ++reg) {
        auto row = std::move(rows[reg]);
        while (!row.terms().empty()) {
            const auto pivot = index(row.terms().back());
            auto found = basis.find(pivot);
            if (found == basis.end()) {
                basis.emplace(pivot, std::move(row));
                registers_.push_back(reg);
                break;
            }
            row ^= found->second;
        }
    }
    split_symbols_.resize(plan.num_symbols_);
    for (uint32_t symbol = 0; symbol < plan.num_symbols_; ++symbol) {
        for (uint32_t reg : plan.expression_dependencies_.dependent_registers(symbol)) {
            split_symbols_[symbol] |= required[reg];
        }
    }
}

SharingExecutor::SharingExecutor(const ExecutablePlan& plan, const PreparedBatchNoise& noise,
                                 const EarlyPostselection& checks, uint32_t lane_capacity,
                                 uint32_t initial_states, bool share, const SharingKey* key,
                                 uint64_t max_state_bytes)
    : BatchClassicalState(plan, BatchOutputMode::AggregateSurvivors, BatchSamplingMode::Ordinary,
                          batch_detail::batch_worker_storage_layout(
                              plan, lane_capacity, BatchOutputMode::AggregateSurvivors,
                              BatchSamplingMode::Ordinary, noise.mode()),
                          &noise),
      checks_(&checks),
      share_(share),
      key_(share ? key : nullptr),
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
    if (key_ && key_->plan_ != &plan) {
        throw std::invalid_argument("sharing key must belong to the executor plan");
    }
    if (key_) {
        key_stride_ = (key_->registers_.size() + 63) / 64;
        if (key_stride_ != 0) {
            if (lane_capacity > key_rows_.max_size() / key_stride_ ||
                size_t{lane_capacity} > key_hash_.max_size() / 4) {
                throw std::length_error("sharing keys exceed addressable storage");
            }
            key_rows_.resize(size_t{lane_capacity} * key_stride_);
            key_hash_.resize(std::bit_ceil(size_t{lane_capacity} * 2));
        }
    }
    max_states_ = static_cast<uint32_t>(std::min<uint64_t>(
        lane_capacity, max_state_bytes / State::storage_bytes(plan.peak_active_width())));
    if (max_states_ == 0) {
        throw std::invalid_argument("max_state_bytes cannot hold one shared state per worker");
    }
    initial_states = std::min(initial_states, max_states_);
    states_.reserve(initial_states);
    grow_state_pool(initial_states);
    split_factors_.reserve(plan.actions_.size());
    for (const auto& action : plan.actions_) {
        split_factors_.push_back(std::visit(
            [&](const auto& typed) -> uint32_t {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation> ||
                              std::is_same_v<T, ExecutablePlan::ExecutePromotion>) {
                    return key_ ? 1 : 2;
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                    return 2;
                } else if constexpr (std::is_same_v<T,
                                                    ExecutablePlan::ExecuteDynamicFusedRotation>) {
                    return key_ ? 1
                                : uint32_t{1} << plan.dynamic_fused_rotations_[typed.rotation_index]
                                                     .sign_basis.size();
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDormantMeasurement>) {
                    return key_ && key_->split_symbols_[typed.branch] ? 2 : 1;
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise>) {
                    return key_ && key_->split_symbols_[typed.flip] ? 2 : 1;
                } else {
                    return 1;
                }
            },
            action));
    }
}

void SharingExecutor::grow_state_pool(uint32_t required) {
    assert(required <= lane_capacity_ && "shared states cannot outnumber classical lanes");
    if (required > max_states_) {
        throw std::runtime_error(
            "shared state pool exceeds max_state_bytes; increase the budget or reduce batch_size "
            "or threads");
    }
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
    initialize_groups(root, first_shot, shots);
    const auto run = [&]<ExecutorBackend Backend>() {
        while (const uint32_t required = dispatch_until_storage_boundary<Backend>()) {
            ++storage_boundaries_;
            grow_state_pool(required);
        }
    };
    switch (plan_->backend_) {
        case ExecutorBackend::Scalar:
            return run.template operator()<ExecutorBackend::Scalar>();
        case ExecutorBackend::Neon:
#if defined(CLIFFT_ENABLE_APPLE_NEON)
            return run.template operator()<ExecutorBackend::Neon>();
#else
            break;
#endif
        case ExecutorBackend::Avx2:
#if defined(CLIFFT_ENABLE_X86_RUNTIME_DISPATCH)
            return run.template operator()<ExecutorBackend::Avx2>();
#else
            break;
#endif
        case ExecutorBackend::Avx512:
#if defined(CLIFFT_ENABLE_X86_RUNTIME_DISPATCH)
            return run.template operator()<ExecutorBackend::Avx512>();
#else
            break;
#endif
    }
    assert(false && "unhandled sharing executor backend");
}

void SharingExecutor::initialize_groups(const SeedRoot& root, uint32_t first_shot, uint32_t shots) {
    if (key_stride_ != 0) {
        std::ranges::fill(key_rows_, uint64_t{0});
        std::ranges::fill(key_hash_, kNoShot);
        // Collect only surviving set bits. At low noise rates most key bits
        // vanish after entrance rejection, avoiding a full per-shot transpose.
        for (size_t k = 0; k < key_->registers_.size(); ++k) {
            const auto reg = key_->registers_[k];
            const auto column = evaluate({reg});
            for (size_t word = 0; word < word_capacity_; ++word) {
                const auto constant =
                    plan_->expression_register_constants_[reg] ? live_words_[word] : 0;
                uint64_t bits = (column[word] ^ constant) & live_words_[word];
                while (bits) {
                    const size_t shot = word * 64 + std::countr_zero(bits);
                    key_rows_[shot * key_stride_ + k / 64] |= uint64_t{1} << (k & 63);
                    bits &= bits - 1;
                }
            }
        }
    }
    if (share_ && key_stride_ == 0) {
        group_count_ = 1;
        heads_[0] = kNoShot;
    }
    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (!is_live(shot))
            continue;
        uint32_t group = 0;
        if (key_stride_ != 0) {
            const auto* key = key_rows_.data() + size_t{shot} * key_stride_;
            uint64_t hash = 0;
            for (size_t word = 0; word < key_stride_; ++word) {
                auto value = key[word];
                hash ^= splitmix64(value);
            }
            size_t bucket = hash & (key_hash_.size() - 1);
            while (true) {
                group = key_hash_[bucket];
                if (group == kNoShot) {
                    group = group_count_++;
                    key_hash_[bucket] = group;
                    heads_[group] = kNoShot;
                    break;
                }
                const auto* other = key_rows_.data() + size_t{heads_[group]} * key_stride_;
                if (std::equal(key, key + key_stride_, other))
                    break;
                bucket = (bucket + 1) & (key_hash_.size() - 1);
            }
        } else if (!share_) {
            group = group_count_++;
            heads_[group] = kNoShot;
        }
        next_[shot] = heads_[group];
        heads_[group] = shot;
        const auto words =
            derive_state(root, uint64_t{first_shot} + shot, kSharedSamplingExecutorDomain);
        shot_rng_[shot].seed_full(words[0], words[1], words[2], words[3]);
    }
    free_count_ = 0;
    if (group_count_ > states_.size()) {
        ++storage_boundaries_;
        grow_state_pool(group_count_);
    }
    peak_groups_ = group_count_;
    free_count_ = static_cast<uint32_t>(states_.size()) - group_count_;
    for (uint32_t index = 0; index < free_count_; ++index) {
        free_[index] = static_cast<uint32_t>(states_.size()) - 1 - index;
    }
    for (uint32_t group = 0; group < group_count_; ++group) {
        groups_[group] = group;
        states_[group].reset();
    }
}

template <ExecutorBackend Backend>
uint32_t SharingExecutor::dispatch_until_storage_boundary() noexcept {
    while (action_index_ < plan_->actions_.size() && live_count_ != 0) {
        const auto plan_index = checks_->action_order()[action_index_];
        const auto& action = plan_->actions_[plan_index];
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
        const uint32_t required = static_cast<uint32_t>(
            std::min<uint64_t>(live_count_, uint64_t{group_count_} * split_factors_[plan_index]));
        if (required > states_.size()) {
            return required;
        }
        std::visit(
            [&](const auto& typed) noexcept {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteRotation> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement>) {
                    execute_action<Backend>(typed);
                } else {
                    execute_action(typed);
                }
            },
            action);
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

uint32_t SharingExecutor::split_by_sign(uint32_t group, std::span<const uint64_t> values) noexcept {
    const auto bit = [&](uint32_t shot) { return ((values[shot >> 6] >> (shot & 63)) & 1) != 0; };
    if (key_) {
#ifndef NDEBUG
        const bool first = bit(heads_[group]);
        for (auto shot = next_[heads_[group]]; shot != kNoShot; shot = next_[shot]) {
            assert(bit(shot) == first && "prepared sharing signs must be uniform within a group");
        }
#endif
        return kNoShot;
    }
    const bool first = bit(heads_[group]);
    uint32_t shot = next_[heads_[group]];
    while (shot != kNoShot && bit(shot) == first) {
        shot = next_[shot];
    }
    // Most signs agree throughout a group. Preserve its list without writing
    // choices or relinking members unless the sign actually separates them.
    if (shot == kNoShot) {
        return kNoShot;
    }
    for (shot = heads_[group]; shot != kNoShot; shot = next_[shot]) {
        choices_[shot] = bit(shot);
    }
    return split(group);
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

template <ExecutorBackend Backend>
void SharingExecutor::execute_action(const ExecutablePlan::ExecuteRotation& action) noexcept {
    const auto values = evaluate(action.sign);
    sign_action(values, [&](uint32_t group, bool sign) {
        apply_direct_rotation<Backend>(states_[group], action.rotation, action.kernel, sign);
    });
}

template <class F>
void SharingExecutor::sign_action(std::span<const uint64_t> values, F&& apply) noexcept {
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        const auto other = split_by_sign(group, values);
        apply(group, lane_bit(values, heads_[group]));
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
            (void)split_by_sign(group, values);
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

template <ExecutorBackend Backend>
void SharingExecutor::execute_action(
    const ExecutablePlan::ExecuteActiveMeasurement& action) noexcept {
    std::ranges::fill(scratch_words_, uint64_t{0});
    const auto collapse = [&](uint32_t group, bool branch, double probability) {
        collapse_active_measurement<Backend>(states_[group], action.measurement, action.kernel,
                                             branch, probability);
    };
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        const auto probabilities = active_measurement_probabilities<Backend>(
            states_[group], action.measurement, action.kernel);
        const auto kind = classify_measurement_branch(probabilities).kind;
        if (kind != MeasurementBranchKind::Random) {
            const bool branch = kind == MeasurementBranchKind::DeterministicOne;
            if (branch) {
                for (uint32_t shot = heads_[group]; shot != kNoShot; shot = next_[shot]) {
                    scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
                }
            }
            collapse(group, branch, probabilities.for_branch(branch));
            continue;
        }
        for (uint32_t shot = heads_[group]; shot != kNoShot; shot = next_[shot]) {
            const bool branch =
                shot_rng_[shot].next_double() * probabilities.total() >= probabilities.zero;
            choices_[shot] = branch;
            if (branch) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
        }
        const auto other = split(group);
        const bool branch = choices_[heads_[group]];
        collapse(group, branch, probabilities.for_branch(branch));
        if (other != kNoShot) {
            collapse(other, true, probabilities.one);
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
            const bool branch = shot_rng_[shot].next_double() < 0.5;
            choices_[shot] = branch;
            if (branch) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
        }
    }
    if (key_ && key_->split_symbols_[action.branch]) {
        const auto original_groups = group_count_;
        for (uint32_t i = 0; i < original_groups; ++i)
            (void)split(groups_[i]);
    }
    propagate_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, evaluate(action.correction),
                            live_words_);
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteDetector& action) noexcept {
    if (action.postselected) {
        const auto values = evaluate_record_parity(action.outcome);
        const auto previous_live_count = live_count_;
        for (size_t word = 0; word < values.size(); ++word) {
            const auto dead = values[word] & live_words_[word];
            live_count_ -= std::popcount(dead);
            live_words_[word] &= ~dead;
        }
        prune_pending_ |= live_count_ != previous_live_count;
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteReadoutNoise& action) noexcept {
    const auto values = evaluate(action.source);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t index = 0; index < group_count_; ++index) {
        for (uint32_t shot = heads_[groups_[index]]; shot != kNoShot; shot = next_[shot]) {
            const double probability =
                lane_bit(values, shot) ? action.prob_one_to_zero : action.prob_zero_to_one;
            const bool flip = probability >= 1 ||
                              (probability > 0 && shot_rng_[shot].next_double() < probability);
            choices_[shot] = flip;
            if (flip) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
        }
    }
    if (key_ && key_->split_symbols_[action.flip]) {
        const auto original_groups = group_count_;
        for (uint32_t i = 0; i < original_groups; ++i)
            (void)split(groups_[i]);
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
