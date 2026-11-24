#include "clifft/sampling/sharing_executor.h"

#include "clifft/sampling/measurement_branch.h"
#include "clifft/sampling/pauli_preparation.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace clifft::sampling {

SharingKey::SharingKey(const ExecutablePlan& plan) : plan_(&plan) {
    const auto required = plan.rotation_sign_registers();
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
                                 uint64_t max_worker_state_bytes, BatchOutputMode output_mode)
    : BatchClassicalState(
          plan, output_mode, BatchSamplingMode::Ordinary,
          batch_detail::batch_worker_storage_layout(plan, lane_capacity, output_mode,
                                                    BatchSamplingMode::Ordinary, &noise),
          &noise),
      checks_(&checks),
      share_(share),
      key_(share ? key : nullptr),
      shot_rng_(lane_capacity),
      order_(lane_capacity),
      position_(lane_capacity),
      group_of_(lane_capacity),
      begin_(lane_capacity),
      end_(lane_capacity),
      departing_(lane_capacity),
      groups_(lane_capacity),
      free_(lane_capacity),
      choices_(lane_capacity),
      state_width_(plan.initial_capacity_width()),
      max_worker_state_bytes_(max_worker_state_bytes) {
    if (output_mode == BatchOutputMode::Rows && plan.num_exp_vals() != 0) {
        throw std::invalid_argument("shared row execution does not support expectation probes");
    }
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
        lane_capacity, max_worker_state_bytes / group_storage_bytes(plan, state_width_)));
    if (max_states_ == 0) {
        throw std::invalid_argument("max_worker_state_bytes cannot hold one shared state");
    }
    if (plan.uses_runtime_frame()) {
        sign_stride_ = size_t{2} * plan.frame_words();
        if (lane_capacity > shot_signs_.max_size() / sign_stride_) {
            throw std::length_error("frame sign bits exceed addressable storage");
        }
        shot_signs_.resize(size_t{lane_capacity} * sign_stride_);
        frame_random_.resize(word_capacity_);
        frame_of_.resize(lane_capacity);
        free_frames_.resize(lane_capacity);
        fired_groups_.resize(lane_capacity);
        changed_rows_.resize(sign_stride_);
        // Find the runs of quarter turns the frame fixes, in dispatch order.
        // They commute, so a shot's tableau depends only on the set it fires.
        const auto order = checks.action_order();
        const auto in_run = [&](size_t position) {
            const auto* clifford =
                std::get_if<ExecutablePlan::ExecuteFrameClifford>(&plan.actions_[order[position]]);
            return clifford != nullptr && clifford->fixed_by_frame &&
                   (clifford->rotation == CliffordRotation::SQRT ||
                    clifford->rotation == CliffordRotation::SQRT_DAG);
        };
        uint32_t longest = 0;
        for (size_t begin = 0; begin < order.size();) {
            size_t end = begin;
            while (end < order.size() && in_run(end)) {
                ++end;
            }
            if (end - begin >= 2) {
                clifford_runs_.resize(order.size());
                clifford_runs_[begin] = static_cast<uint32_t>(end - begin);
                longest = std::max(longest, clifford_runs_[begin]);
            }
            begin = std::max(end, begin + 1);
        }
        if (longest != 0) {
            const size_t stride = (size_t{longest} + 63) / 64;
            if (lane_capacity > run_bits_.max_size() / stride) {
                throw std::length_error("frame run bits exceed addressable storage");
            }
            run_bits_.resize(size_t{lane_capacity} * stride);
            run_table_.resize(std::bit_ceil(size_t{lane_capacity} * 2));
            for (auto* table :
                 {&run_class_, &class_source_, &class_frame_, &class_seen_by_, &class_owner_,
                  &class_size_, &class_group_, &run_stay_, &run_lanes_, &run_leaving_}) {
                table->resize(lane_capacity);
            }
        }
    }
    initial_states = std::min(initial_states, max_states_);
    grow_state_pool({.states = initial_states, .width = state_width_});
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
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFrameClifford>) {
                    // Only a quarter turn changes the tableau of the shots it fires on.
                    return typed.rotation == CliffordRotation::SQRT ||
                                   typed.rotation == CliffordRotation::SQRT_DAG
                               ? 2
                               : 1;
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteFrameRotation> ||
                                     std::is_same_v<T, ExecutablePlan::ExecuteFrameMeasurement>) {
                    return 2;
                } else {
                    return 1;
                }
            },
            action));
    }
}

uint64_t SharingExecutor::frame_storage_bytes(const ExecutablePlan& plan) {
    if (!plan.uses_runtime_frame()) {
        return 0;
    }
    // Two rows per frame coordinate, the frame's six scratch Paulis and the
    // mapped Pauli, each with separately allocated X and Z words, and the row
    // mask.
    constexpr uint64_t kScratchPaulis = 7;
    constexpr uint64_t kAllocationOverhead = 16;
    const uint64_t words_bytes = uint64_t{plan.frame_words()} * sizeof(uint64_t);
    const uint64_t paulis = 2 * uint64_t{plan.frame_width_} + kScratchPaulis;
    const uint64_t pauli_bytes = sizeof(PauliString) + 2 * (words_bytes + kAllocationOverhead);
    return saturating_add_u64(saturating_multiply_u64(paulis, pauli_bytes),
                              sizeof(SharedFrame) + 2 * words_bytes + kAllocationOverhead);
}

uint64_t SharingExecutor::group_storage_bytes(const ExecutablePlan& plan, uint32_t width) {
    return saturating_add_u64(State::storage_bytes(width), frame_storage_bytes(plan));
}

uint64_t SharingExecutor::classical_storage_bytes(const ExecutablePlan& plan,
                                                  const PreparedBatchNoise& noise,
                                                  const SharingKey* key, uint32_t lane_capacity,
                                                  BatchOutputMode output_mode) {
    // Per-lane RNG streams, group ranges, and branch choices.
    constexpr uint64_t kLaneBytes =
        sizeof(Xoshiro256PlusPlus) + 8 * sizeof(uint32_t) + sizeof(uint8_t);
    uint64_t bytes = saturating_add_u64(
        batch_detail::batch_worker_classical_bytes(plan, lane_capacity, output_mode,
                                                   BatchSamplingMode::Ordinary, &noise),
        saturating_multiply_u64(lane_capacity, kLaneBytes));
    bytes =
        saturating_add_u64(bytes, saturating_multiply_u64(plan.num_actions(), sizeof(uint32_t)));
    if (const uint64_t stride = key ? (key->num_bits() + 63) / 64 : 0; stride != 0) {
        bytes = saturating_add_u64(
            bytes, saturating_multiply_u64(lane_capacity, stride * sizeof(uint64_t)));
        bytes = saturating_add_u64(
            bytes,
            saturating_multiply_u64(std::bit_ceil(uint64_t{lane_capacity} * 2), sizeof(uint32_t)));
    }
    if (plan.uses_runtime_frame()) {
        // One sign bit per tableau row for every lane, the packed draws of
        // random frame measurements, and three group tables.
        const uint64_t sign_words = 2 * uint64_t{plan.frame_words()};
        bytes = saturating_add_u64(
            bytes, saturating_multiply_u64(lane_capacity, sign_words * sizeof(uint64_t)));
        bytes = saturating_add_u64(
            bytes, saturating_multiply_u64(packed_word_count(lane_capacity), sizeof(uint64_t)));
        bytes =
            saturating_add_u64(bytes, saturating_multiply_u64(lane_capacity, 3 * sizeof(uint32_t)));
        // Regrouping a run of conditional quarter turns: one bit per turn for
        // every lane, bounded here by all of the plan's conditional Cliffords,
        // the class table, and ten entries per lane.
        uint64_t cliffords = 0;
        for (const auto& action : plan.actions_) {
            cliffords += std::holds_alternative<ExecutablePlan::ExecuteFrameClifford>(action);
        }
        bytes = saturating_add_u64(
            bytes,
            saturating_multiply_u64(lane_capacity, (cliffords + 63) / 64 * sizeof(uint64_t)));
        bytes = saturating_add_u64(
            bytes, saturating_multiply_u64(
                       std::bit_ceil(uint64_t{lane_capacity} * 2) + 10 * uint64_t{lane_capacity},
                       sizeof(uint32_t)));
    }
    return bytes;
}

uint32_t max_safe_shared_capacity(const ExecutablePlan& plan, const PreparedBatchNoise& noise,
                                  const SharingKey* key, uint64_t max_worker_state_bytes,
                                  BatchOutputMode output_mode) {
    // Width-zero plans keep no coefficient states, unless a runtime Clifford
    // frame gives every group a tableau.
    const bool grouped = shares_groups(plan);
    uint32_t capacity =
        !grouped ? kMaxAutoSharedBatchShots
                 : static_cast<uint32_t>(std::min<uint64_t>(
                       kMaxAutoSharedBatchShots,
                       max_worker_state_bytes / SharingExecutor::group_storage_bytes(
                                                    plan, plan.initial_capacity_width())));
    // The sampler retains one observable counter per output alongside the executor.
    const uint64_t counter_bytes =
        saturating_multiply_u64(plan.num_observables(), sizeof(uint64_t));
    // A worker that groups nothing is a packed executor without the sharing arrays.
    const auto worker_bytes = [&](uint32_t lanes) {
        return grouped
                   ? SharingExecutor::classical_storage_bytes(plan, noise, key, lanes, output_mode)
                   : batch_detail::batch_worker_storage_bytes(plan, lanes, output_mode,
                                                              BatchSamplingMode::Ordinary, &noise);
    };
    while (capacity > 1 && saturating_add_u64(worker_bytes(capacity), counter_bytes) >
                               kMaxAutoSharedWorkerClassicalBytes) {
        capacity /= 2;
    }
    return capacity;
}

BatchExecutionPolicy resolve_shared_execution_policy(uint32_t safe_capacity, uint32_t shots,
                                                     uint32_t workers) noexcept {
    assert(safe_capacity != 0 && workers != 0 && "shared policy needs a capacity and a worker");
    const uint32_t min_capacity = std::min(safe_capacity, uint32_t{2});
    if (shots == 0) {
        return {.lane_capacity = min_capacity, .worker_count = 0};
    }
    // An even share gives every requested worker a batch. Below the lane floor,
    // fewer workers with fuller batches share more states per coefficient update.
    const uint64_t share = (uint64_t{shots} + workers - 1) / workers;
    const uint32_t capacity = static_cast<uint32_t>(
        std::min<uint64_t>({safe_capacity, std::max(shots, min_capacity),
                            std::max<uint64_t>(share, kMinAutoSharedLanesPerWorker)}));
    const uint64_t batches = (uint64_t{shots} + capacity - 1) / capacity;
    return {.lane_capacity = capacity,
            .worker_count = static_cast<uint32_t>(std::min<uint64_t>(workers, batches))};
}

std::vector<batch_detail::BatchTuningCandidate> shared_tuning_candidates(
    uint32_t safe_capacity, uint32_t shots, uint32_t workers, uint64_t output_row_bytes) {
    std::vector<batch_detail::BatchTuningCandidate> candidates;
    if (shots == 0) {
        return candidates;
    }
    // Cheap probes grow toward this target. A full batch per production worker
    // may need more shots; retained output bytes impose a separate hard limit.
    constexpr uint32_t kTargetProbeShots = 65536;
    const uint32_t min_capacity = std::min(safe_capacity, uint32_t{2});
    const uint32_t max_capacity = std::min(safe_capacity, std::max(shots, min_capacity));
    const uint32_t output_limit =
        output_row_bytes == 0 ? shots
                              : batch_detail::max_batch_probe_shots(shots, output_row_bytes);
    const auto append = [&](uint32_t requested) {
        const uint32_t capacity = std::min(requested, max_capacity);
        if (std::ranges::any_of(candidates, [&](const auto& candidate) {
                return candidate.policy.lane_capacity == capacity;
            })) {
            return;
        }
        const uint64_t batches = (uint64_t{shots} + capacity - 1) / capacity;
        const uint32_t used = static_cast<uint32_t>(std::min<uint64_t>(workers, batches));
        // One full batch per worker measures the candidate at its production
        // fill; the floor keeps single-lane probes long enough to time.
        constexpr uint64_t kMinProbeShots = 256;
        const uint32_t probe_shots = static_cast<uint32_t>(
            std::min<uint64_t>(shots, std::max(kMinProbeShots, uint64_t{capacity} * used)));
        if (probe_shots <= output_limit) {
            const uint32_t maximum =
                std::min(output_limit, std::max(kTargetProbeShots, probe_shots));
            candidates.push_back({{capacity, used, true}, probe_shots, maximum});
        }
    };
    append(resolve_shared_execution_policy(safe_capacity, shots, workers).lane_capacity);
    if (candidates.empty()) {
        return candidates;
    }
    for (const uint32_t capacity : {64u, 256u, 1024u, 4096u, kMaxAutoSharedBatchShots}) {
        append(capacity);
    }
    return candidates;
}

uint64_t SharingExecutor::state_pool_bytes() const noexcept {
    uint64_t bytes = saturating_add_u64(
        saturating_multiply_u64(states_.capacity(), sizeof(State)),
        saturating_multiply_u64(state_chunks_.capacity(), sizeof(PageAlignedAllocation)));
    for (const auto& chunk : state_chunks_) {
        bytes = saturating_add_u64(bytes, chunk.size());
    }
    return saturating_add_u64(bytes,
                              saturating_multiply_u64(frames_.size(), frame_storage_bytes(*plan_)));
}

namespace {

// One slab of state slots, each sized for the given width.
[[nodiscard]] PageAlignedAllocation allocate_state_slab(size_t slots, uint32_t width) {
    const size_t slot_bytes = State::slot_bytes(width);
    if (slots > std::numeric_limits<size_t>::max() / slot_bytes) {
        throw std::length_error("shared state slab exceeds addressable memory");
    }
    const auto alignment = slot_bytes >= PageAlignedAllocation::kHugePageSize
                               ? PageAlignedAllocation::Alignment::HugePageEligible
                               : PageAlignedAllocation::Alignment::BasePage;
    return PageAlignedAllocation(slots * slot_bytes, alignment);
}

[[nodiscard]] double* slab_slot(PageAlignedAllocation& slab, size_t slot, uint32_t width) {
    return reinterpret_cast<double*>(static_cast<std::byte*>(slab.data()) +
                                     slot * State::slot_bytes(width));
}

}  // namespace

void SharingExecutor::grow_state_pool(StorageRequest request) {
    assert(request.states <= lane_capacity_ && "shared states cannot outnumber classical lanes");
    // This is an explicit suspension boundary. Live coefficients and sampled
    // outcomes remain intact while storage grows, then the same action resumes.
    if (request.width > state_width_) {
        if (request.width >= kDenseActiveWidthLimit) {
            throw std::overflow_error(
                "shared sampling active width would reach " + std::to_string(request.width) +
                ", but the dense-state limit is " + std::to_string(kDenseActiveWidthLimit));
        }
        const uint32_t max_states = static_cast<uint32_t>(std::min<uint64_t>(
            lane_capacity_, max_worker_state_bytes_ / group_storage_bytes(*plan_, request.width)));
        if (states_.size() > max_states) {
            throw std::runtime_error(
                "shared states of active width " + std::to_string(request.width) +
                " exceed max_worker_state_bytes; increase the budget or reduce batch_size");
        }
        // A slab holds slots of one width, so every state moves to a new slab.
        std::vector<State> widened;
        widened.reserve(states_.capacity());
        PageAlignedAllocation slab = allocate_state_slab(states_.size(), request.width);
        for (size_t slot = 0; slot < states_.size(); ++slot) {
            const State& narrow = states_[slot];
            State& wide =
                widened.emplace_back(State::borrowed(slab_slot(slab, slot, request.width),
                                                     request.width, narrow.initial_active_width()));
            wide.set_active_width(narrow.active_width());
            std::ranges::copy(narrow.real(), wide.real().begin());
            std::ranges::copy(narrow.imag(), wide.imag().begin());
        }
        states_ = std::move(widened);
        state_chunks_.clear();
        state_chunks_.push_back(std::move(slab));
        state_width_ = request.width;
        max_states_ = max_states;
    }
    const uint32_t required = request.states;
    if (required > max_states_) {
        throw std::runtime_error(
            "shared state pool exceeds max_worker_state_bytes; increase the budget or reduce "
            "batch_size");
    }
    const uint32_t old_count = static_cast<uint32_t>(states_.size());
    if (required <= old_count) {
        return;
    }
    // Keep initial_states exact on construction, then grow geometrically so
    // splitting batches allocate O(log n) contiguous slabs instead of tiny chunks.
    const uint32_t target =
        old_count == 0
            ? required
            : static_cast<uint32_t>(std::min<uint64_t>(
                  max_states_,
                  std::max<uint64_t>(required,
                                     uint64_t{old_count} + std::max<uint64_t>(old_count / 2, 1))));
    const uint32_t add = target - old_count;
    states_.reserve(target);
    state_chunks_.reserve(state_chunks_.size() + 1);
    if (sign_stride_ != 0) {
        frames_.reserve(target);
    }
    state_chunks_.push_back(allocate_state_slab(add, state_width_));
    for (uint32_t i = 0; i < add; ++i) {
        const uint32_t slot = old_count + i;
        if (sign_stride_ != 0) {
            frames_.emplace_back(plan_->frame_width_, sign_stride_);
            free_frames_[free_frame_count_++] = slot;
        }
        states_.push_back(State::borrowed(slab_slot(state_chunks_.back(), i, state_width_),
                                          state_width_, plan_->initial_active_width()));
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
    num_merges_ = 0;
    peak_groups_ = 0;
    group_count_ = 0;
    prune_pending_ = false;
    action_index_ = 0;
    if (live_count_ == 0) {
        if (output_mode_ == BatchOutputMode::Rows) {
            active_lanes_ = 0;
        }
        return;
    }
    initialize_groups(root, first_shot, shots);
    std::fill_n(shot_signs_.begin(), size_t{shots} * sign_stride_, uint64_t{0});
    const auto run = [&]<ExecutorBackend Backend>() {
        while (const StorageRequest request = dispatch_until_storage_boundary<Backend>()) {
            ++storage_boundaries_;
            grow_state_pool(request);
        }
#ifndef NDEBUG
        // Every listed group holds one reference to its tableau, and nothing
        // else does.
        uint64_t frame_users = 0;
        for (const SharedFrame& frame : frames_) {
            frame_users += frame.users;
        }
        assert((sign_stride_ == 0 || frame_users == group_count_) &&
               "tableau references match the groups");
#endif
        // Only the classical output columns need compaction after dispatch.
        // Coefficient groups and hidden state are reset before the next batch.
        if (output_mode_ == BatchOutputMode::Rows && live_count_ != active_lanes()) {
            uint32_t destination = 0;
            for (uint32_t source = 0; source < active_lanes(); ++source) {
                if (is_live(source)) {
                    shot_indices_[destination++] = shot_indices_[source];
                }
            }
            records_.compact_prefix(plan_->num_visible_records_, live_words_, active_lanes(),
                                    live_count_, scratch_words_);
            detectors_.compact(live_words_, active_lanes(), live_count_, scratch_words_);
            observables_.compact(live_words_, active_lanes(), live_count_, scratch_words_);
            active_lanes_ = live_count_;
            fill_low_lane_mask(live_words_, active_lanes());
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
    // Assign every live lane a group. Until the ranges are laid out, end_
    // counts a group's lanes and begin_ holds the lane whose key represents it.
    if (share_ && key_stride_ == 0) {
        group_count_ = 1;
        end_[0] = 0;
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
                    end_[group] = 0;
                    begin_[group] = shot;
                    break;
                }
                const auto* other = key_rows_.data() + size_t{begin_[group]} * key_stride_;
                if (std::equal(key, key + key_stride_, other))
                    break;
                bucket = (bucket + 1) & (key_hash_.size() - 1);
            }
        } else if (!share_) {
            group = group_count_++;
            end_[group] = 0;
        }
        group_of_[shot] = group;
        ++end_[group];
        const auto words =
            derive_state(root, uint64_t{first_shot} + shot, kSharedSamplingExecutorDomain);
        shot_rng_[shot].seed_full(words[0], words[1], words[2], words[3]);
    }
    // Lay the groups out as consecutive ranges and fill them in lane order.
    uint32_t placed = 0;
    for (uint32_t group = 0; group < group_count_; ++group) {
        begin_[group] = placed;
        placed += end_[group];
        end_[group] = begin_[group];
    }
    for_each_set_bit(MaskView{std::span<const uint64_t>(live_words_)}, [&](uint32_t shot) {
        const uint32_t group = group_of_[shot];
        position_[shot] = end_[group];
        order_[end_[group]++] = shot;
    });
    free_count_ = 0;
    if (group_count_ > states_.size()) {
        ++storage_boundaries_;
        grow_state_pool({.states = group_count_, .width = state_width_});
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
    if (sign_stride_ == 0) {
        return;
    }
    frame_stamp_ = 0;
    free_frame_count_ = 0;
    for (uint32_t frame = static_cast<uint32_t>(frames_.size()); frame-- > 0;) {
        frames_[frame].users = 0;
        frames_[frame].stamp = 0;
        free_frames_[free_frame_count_++] = frame;
    }
    // Every shot starts with the identity tableau, which the groups share.
    // Without sharing, each group keeps its own throughout.
    uint32_t frame = kNoShot;
    for (uint32_t group = 0; group < group_count_; ++group) {
        if (frame == kNoShot || !share_) {
            frame = acquire_frame(nullptr);
        }
        frame_of_[group] = frame;
        ++frames_[frame].users;
    }
}

template <ExecutorBackend Backend>
SharingExecutor::StorageRequest SharingExecutor::dispatch_until_storage_boundary() noexcept {
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
        ++frame_stamp_;
        if (const uint32_t run = clifford_runs_.empty() ? 0 : clifford_runs_[action_index_];
            run != 0) {
            const uint32_t groups = group_count_ + prepare_clifford_run(run);
            if (groups > states_.size()) {
                return {.states = groups, .width = state_width_};
            }
            execute_clifford_run(run);
            action_index_ += run;
            continue;
        }
        const uint32_t required = static_cast<uint32_t>(
            std::min<uint64_t>(live_count_, uint64_t{group_count_} * split_factors_[plan_index]));
        if (required > states_.size()) {
            return {.states = required, .width = state_width_};
        }
        if (const auto* rotation = std::get_if<ExecutablePlan::ExecuteFrameRotation>(&action);
            rotation != nullptr && frame_rotation_needs_width(*rotation)) {
            return {.states = static_cast<uint32_t>(states_.size()), .width = state_width_ + 1};
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
    return {};
}

void SharingExecutor::copy_state(uint32_t to, uint32_t from) noexcept {
    states_[to].set_active_width(states_[from].active_width());
    std::ranges::copy(states_[from].real(), states_[to].real().begin());
    std::ranges::copy(states_[from].imag(), states_[to].imag().begin());
}

void SharingExecutor::swap_lanes(uint32_t first, uint32_t second) noexcept {
    std::swap(order_[first], order_[second]);
    position_[order_[first]] = first;
    position_[order_[second]] = second;
}

uint32_t SharingExecutor::detach_tail(uint32_t group, uint32_t tail) noexcept {
    const uint32_t end = end_[group];
    if (tail == begin_[group] || tail == end) {
        return kNoShot;
    }
    assert(free_count_ != 0 && "state split must fit the preflight capacity");
    const uint32_t other = free_[--free_count_];
    copy_state(other, group);
    if (sign_stride_ != 0) {
        // The copy refers to the same frame.
        frame_of_[other] = frame_of_[group];
        ++frames_[frame_of_[group]].users;
    }
    end_[group] = tail;
    begin_[other] = tail;
    end_[other] = end;
    for (uint32_t shot : lanes(other)) {
        group_of_[shot] = other;
    }
    groups_[group_count_++] = other;
    ++num_splits_;
    peak_groups_ = std::max(peak_groups_, group_count_);
    return other;
}

uint32_t SharingExecutor::split(uint32_t group) noexcept {
    uint32_t tail = end_[group];
    for (uint32_t index = begin_[group]; index < tail;) {
        if (choices_[order_[index]] != 0) {
            swap_lanes(index, --tail);
        } else {
            ++index;
        }
    }
    return detach_tail(group, tail);
}

void SharingExecutor::split_by_sign(std::span<const uint64_t> values) noexcept {
    if (key_) {
#ifndef NDEBUG
        for (uint32_t index = 0; index < group_count_; ++index) {
            const auto members = lanes(groups_[index]);
            for (uint32_t shot : members) {
                assert(lane_bit(values, shot) == lane_bit(values, members.front()) &&
                       "prepared sharing signs must be uniform within a group");
            }
        }
#endif
        return;
    }
    if (group_count_ == live_count_) {
        // Every group holds one lane.
        return;
    }
    if (live_count_ < live_words_.size()) {
        // Few lanes remain, so walking them costs less than scanning the sign
        // words of the whole batch.
        const uint32_t original_group_count = group_count_;
        for (uint32_t index = 0; index < original_group_count; ++index) {
            const uint32_t group = groups_[index];
            for (uint32_t shot : lanes(group)) {
                choices_[shot] = lane_bit(values, shot);
            }
            (void)split(group);
        }
        return;
    }
    // A group separates exactly when some, but not all, of its lanes carry the
    // less common sign. Visiting only those lanes keeps the work proportional
    // to the lanes that disagree with the batch, which is few when most shots
    // share a trajectory.
    uint32_t ones = 0;
    for (size_t word = 0; word < live_words_.size(); ++word) {
        ones += static_cast<uint32_t>(std::popcount(values[word] & live_words_[word]));
    }
    const uint64_t majority = 2 * uint64_t{ones} > live_count_ ? ~uint64_t{0} : 0;
    for (size_t word = 0; word < live_words_.size(); ++word) {
        uint64_t bits = (values[word] ^ majority) & live_words_[word];
        while (bits) {
            const uint32_t shot = static_cast<uint32_t>(word * 64 + std::countr_zero(bits));
            bits &= bits - 1;
            const uint32_t group = group_of_[shot];
            swap_lanes(position_[shot], end_[group] - ++departing_[group]);
        }
    }
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        if (departing_[group] != 0) {
            const uint32_t tail = end_[group] - departing_[group];
            departing_[group] = 0;
            detach_tail(group, tail);
        }
    }
}

void SharingExecutor::prune() noexcept {
    uint32_t kept = 0;
    for (uint32_t index = 0; index < group_count_; ++index) {
        const auto group = groups_[index];
        // Compact the live lanes toward the front. Dropped lanes belong to no
        // group afterwards, so their slots need not be preserved.
        uint32_t tail = begin_[group];
        for (uint32_t position = begin_[group]; position < end_[group]; ++position) {
            const uint32_t shot = order_[position];
            if (lane_bit(live_words_, shot)) {
                order_[tail] = shot;
                position_[shot] = tail++;
            }
        }
        end_[group] = tail;
        if (tail != begin_[group]) {
            groups_[kept++] = group;
        } else {
            free_[free_count_++] = group;
            if (sign_stride_ != 0) {
                release_frame(frame_of_[group]);
            }
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
    split_by_sign(values);
    for (uint32_t index = 0; index < group_count_; ++index) {
        const uint32_t group = groups_[index];
        apply(group, lane_bit(values, lanes(group).front()));
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
        split_by_sign(evaluate(sign));
    }
    for (uint32_t index = 0; index < group_count_; ++index) {
        auto group = groups_[index];
        uint32_t value = 0;
        for (size_t sign_index = 0; sign_index < rotation.sign_basis.size(); ++sign_index) {
            value |=
                uint32_t(lane_bit(evaluate(rotation.sign_basis[sign_index]), lanes(group).front()))
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
                for (uint32_t shot : lanes(group)) {
                    scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
                }
            }
            collapse(group, branch, probabilities.for_branch(branch));
            continue;
        }
        for (uint32_t shot : lanes(group)) {
            const bool branch =
                shot_rng_[shot].next_double() * probabilities.total() >= probabilities.zero;
            choices_[shot] = branch;
            if (branch) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
        }
        const auto other = split(group);
        const bool branch = choices_[lanes(group).front()];
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
    BatchClassicalState::execute_action(action);
    split_on_symbol(action.branch);
}

void SharingExecutor::split_on_symbol(uint32_t symbol) noexcept {
    if (!key_ || !key_->split_symbols_[symbol]) {
        return;
    }
    // Packed outcomes split only the compiler-prepared future sign dependencies.
    const auto original_groups = group_count_;
    for (uint32_t i = 0; i < original_groups; ++i) {
        const auto group = groups_[i];
        for (uint32_t shot : lanes(group)) {
            choices_[shot] = lane_bit(scratch_words_, shot);
        }
        (void)split(group);
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteDetector& action) noexcept {
    if (!action.postselected && output_mode_ == BatchOutputMode::AggregateSurvivors) {
        return;
    }
    const auto values = evaluate_record_parity(action.outcome);
    if (output_mode_ == BatchOutputMode::Rows) {
        detectors_.assign(action.detector, values, live_words_);
    }
    if (action.postselected) {
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
    if (action.batch_symmetric_inverse_hazard > 0.0) {
        // Rare symmetric flips are independent of both the source bit and group.
        BatchClassicalState::execute_action(action);
        split_on_symbol(action.flip);
        return;
    }
    const auto values = evaluate(action.source);
    std::ranges::fill(scratch_words_, uint64_t{0});
    for (uint32_t index = 0; index < group_count_; ++index) {
        for (uint32_t shot : lanes(groups_[index])) {
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

std::span<const uint64_t> SharingExecutor::frame_pauli_rows(uint32_t pauli) const noexcept {
    // The plan stores a frame Pauli as its X words then its Z words, which is
    // the row layout: X_q selects the image of X_q and Z_q the image of Z_q.
    return std::span<const uint64_t>(plan_->frame_paulis_)
        .subspan(size_t{pauli} * sign_stride_, sign_stride_);
}

bool SharingExecutor::shot_flips(uint32_t shot, std::span<const uint64_t> rows) const noexcept {
    const uint64_t* signs = shot_signs_.data() + size_t{shot} * sign_stride_;
    uint64_t parity = 0;
    for (size_t word = 0; word < sign_stride_; ++word) {
        parity ^= signs[word] & rows[word];
    }
    return (std::popcount(parity) & 1) != 0;
}

void SharingExecutor::flip_shot_signs(uint32_t shot, std::span<const uint64_t> rows) noexcept {
    uint64_t* signs = shot_signs_.data() + size_t{shot} * sign_stride_;
    for (size_t word = 0; word < sign_stride_; ++word) {
        signs[word] ^= rows[word];
    }
}

uint32_t SharingExecutor::acquire_frame(const SharedFrame* source) noexcept {
    assert(free_frame_count_ != 0 && "the pool holds a tableau for every group");
    const uint32_t frame = free_frames_[--free_frame_count_];
    SharedFrame& slot = frames_[frame];
    if (source != nullptr && source->changed) {
        for (size_t row = 0; row < source->frame.rows.size(); ++row) {
            copy_pauli(source->frame.rows[row].view(), slot.frame.rows[row]);
        }
        slot.changed = true;
    } else if (slot.changed) {
        slot.frame.reset();
        slot.changed = false;
    }
    return frame;
}

void SharingExecutor::release_frame(uint32_t frame) noexcept {
    assert(frames_[frame].users != 0 && "a released tableau had a group");
    if (--frames_[frame].users == 0) {
        free_frames_[free_frame_count_++] = frame;
    }
}

void SharingExecutor::collect_merged_groups() noexcept {
    // A merge leaves the taken lanes where they were and only renames their
    // group, so the ranges are rebuilt from group_of_. Every lane in a range
    // is live here: the action that merges runs after any pending prune.
    uint32_t kept = 0;
    for (uint32_t index = 0; index < group_count_; ++index) {
        const uint32_t group = groups_[index];
        if (begin_[group] != end_[group]) {
            groups_[kept++] = group;
            end_[group] = 0;
        } else {
            free_[free_count_++] = group;
        }
    }
    group_count_ = kept;
    const MaskView live{std::span<const uint64_t>(live_words_)};
    for_each_set_bit(live, [&](uint32_t shot) { ++end_[group_of_[shot]]; });
    uint32_t placed = 0;
    for (uint32_t index = 0; index < group_count_; ++index) {
        const uint32_t group = groups_[index];
        begin_[group] = placed;
        placed += end_[group];
        end_[group] = begin_[group];
    }
    for_each_set_bit(live, [&](uint32_t shot) {
        const uint32_t group = group_of_[shot];
        position_[shot] = end_[group];
        order_[end_[group]++] = shot;
    });
}

uint32_t SharingExecutor::prepare_clifford_run(uint32_t length) noexcept {
    const auto order = checks_->action_order();
    const size_t stride = (size_t{length} + 63) / 64;
    std::fill_n(run_bits_.begin(), size_t{active_lanes()} * stride, uint64_t{0});
    // Collect only the set bits: most shots fire few of the run's turns.
    for (uint32_t k = 0; k < length; ++k) {
        const auto& action = std::get<ExecutablePlan::ExecuteFrameClifford>(
            plan_->actions_[order[action_index_ + k]]);
        const auto fires = evaluate(action.condition);
        for (size_t word = 0; word < live_words_.size(); ++word) {
            for (uint64_t bits = fires[word] & live_words_[word]; bits != 0; bits &= bits - 1) {
                const size_t shot = word * 64 + static_cast<size_t>(std::countr_zero(bits));
                run_bits_[shot * stride + k / 64] |= uint64_t{1} << (k & 63);
            }
        }
    }
    std::ranges::fill(run_table_, kNoShot);
    uint32_t added = 0;
    for (uint32_t index = 0; index < group_count_; ++index) {
        const uint32_t group = groups_[index];
        const uint32_t source = frame_of_[group];
        uint32_t classes = 0;
        bool quiet = false;
        run_stay_[group] = kNoShot;
        for (uint32_t shot : lanes(group)) {
            const uint64_t* fired = run_bits_.data() + size_t{shot} * stride;
            uint64_t hash = source;
            bool any = false;
            for (size_t word = 0; word < stride; ++word) {
                any |= fired[word] != 0;
                uint64_t mixed = hash ^ fired[word];
                hash = splitmix64(mixed);
            }
            if (!any) {
                run_class_[shot] = kNoShot;
                quiet = true;
                continue;
            }
            // Find the class of this tableau and set, or start it with this shot.
            size_t bucket = hash & (run_table_.size() - 1);
            uint32_t found = run_table_[bucket];
            while (found != kNoShot && (class_source_[found] != source ||
                                        !std::equal(fired, fired + stride,
                                                    run_bits_.data() + size_t{found} * stride))) {
                bucket = (bucket + 1) & (run_table_.size() - 1);
                found = run_table_[bucket];
            }
            if (found == kNoShot) {
                found = shot;
                run_table_[bucket] = shot;
                class_source_[shot] = source;
                class_frame_[shot] = kNoShot;
                class_seen_by_[shot] = kNoShot;
                class_owner_[shot] = kNoShot;
            }
            run_class_[shot] = found;
            if (class_seen_by_[found] != group) {
                class_seen_by_[found] = group;
                ++classes;
                if (run_stay_[group] == kNoShot) {
                    run_stay_[group] = found;
                }
            }
        }
        // The shots that fire nothing keep the group. When every shot fires,
        // the first class does, so a group of one class needs no new state.
        if (quiet) {
            run_stay_[group] = kNoShot;
            added += classes;
        } else if (classes != 0) {
            added += classes - 1;
        }
    }
    return added;
}

void SharingExecutor::execute_clifford_run(uint32_t length) noexcept {
    const auto order = checks_->action_order();
    const size_t stride = (size_t{length} + 63) / 64;
    // Applies every turn of the class's set; the turns commute, so their
    // order does not matter.
    const auto rotate = [&](uint32_t frame, uint32_t klass) noexcept {
        SharedFrame& slot = frames_[frame];
        const uint64_t* fired = run_bits_.data() + size_t{klass} * stride;
        for (size_t word = 0; word < stride; ++word) {
            for (uint64_t bits = fired[word]; bits != 0; bits &= bits - 1) {
                const size_t k = word * 64 + static_cast<size_t>(std::countr_zero(bits));
                const auto& action = std::get<ExecutablePlan::ExecuteFrameClifford>(
                    plan_->actions_[order[action_index_ + k]]);
                slot.frame.rotate_fixed_clifford(plan_->frame_pauli(action.pauli), action.rotation);
            }
        }
        slot.changed = true;
    };
    // Points a group at its class's tableau, making that tableau on first use.
    const auto take_class_frame = [&](uint32_t group, uint32_t klass) noexcept {
        const uint32_t source = frame_of_[group];
        if (class_frame_[klass] == kNoShot) {
            if (frames_[source].users == 1) {
                // No other group has the source tableau: rotate it in place.
                rotate(source, klass);
                class_frame_[klass] = source;
                return;
            }
            class_frame_[klass] = acquire_frame(&frames_[source]);
            rotate(class_frame_[klass], klass);
        }
        frame_of_[group] = class_frame_[klass];
        ++frames_[class_frame_[klass]].users;
        release_frame(source);
    };
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        const uint32_t stay = run_stay_[group];
        const uint32_t begin = begin_[group];
        const uint32_t end = end_[group];
        // Count the group's shots in every class that leaves it.
        uint32_t leaving_classes = 0;
        uint32_t leaving_shots = 0;
        for (uint32_t position = begin; position < end; ++position) {
            const uint32_t shot = order_[position];
            run_lanes_[position] = shot;
            const uint32_t klass = run_class_[shot];
            if (klass == stay) {
                continue;
            }
            if (class_owner_[klass] != group) {
                class_owner_[klass] = group;
                class_size_[klass] = 0;
                run_leaving_[leaving_classes++] = klass;
            }
            ++class_size_[klass];
            ++leaving_shots;
        }
        if (leaving_classes != 0) {
            // Every leaving class takes a range behind the shots that stay, a
            // copy of the group's coefficients, and its own tableau.
            uint32_t cursor = end - leaving_shots;
            for (uint32_t i = 0; i < leaving_classes; ++i) {
                const uint32_t klass = run_leaving_[i];
                assert(free_count_ != 0 && "the run's groups fit the prepared capacity");
                const uint32_t other = free_[--free_count_];
                copy_state(other, group);
                frame_of_[other] = frame_of_[group];
                ++frames_[frame_of_[group]].users;
                take_class_frame(other, klass);
                begin_[other] = end_[other] = cursor;
                cursor += class_size_[klass];
                class_group_[klass] = other;
                groups_[group_count_++] = other;
                ++num_splits_;
            }
            end_[group] = begin;
            for (uint32_t position = begin; position < end; ++position) {
                const uint32_t shot = run_lanes_[position];
                const uint32_t klass = run_class_[shot];
                const uint32_t target = klass == stay ? group : class_group_[klass];
                group_of_[shot] = target;
                position_[shot] = end_[target];
                order_[end_[target]++] = shot;
            }
        }
        // The source tableau was copied for the other classes above, so the
        // group's own class can now rotate it in place.
        if (stay != kNoShot) {
            take_class_frame(group, stay);
        }
    }
    peak_groups_ = std::max(peak_groups_, group_count_);
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteFrameClifford& action) noexcept {
    if (action.rotation == CliffordRotation::IDENTITY) {
        return;
    }
    const auto fires = evaluate(action.condition);
    const auto rows = frame_pauli_rows(action.pauli);
    // A frame maps Paulis by conjugation, which preserves commutation: a row
    // anticommutes with the image of the action's Pauli exactly when its
    // generator anticommutes with the Pauli. The rotation therefore changes
    // the same rows in every tableau: the X rows where the Pauli has Z and
    // the Z rows where it has X.
    const size_t half = sign_stride_ / 2;
    std::copy_n(rows.begin() + half, half, changed_rows_.begin());
    std::copy_n(rows.begin(), half, changed_rows_.begin() + half);
    // Only the lanes the condition selects are visited: most shots fire few
    // of a plan's conditional Cliffords.
    const auto for_each_fired = [&](auto&& visit) noexcept {
        for (size_t word = 0; word < live_words_.size(); ++word) {
            for (uint64_t bits = fires[word] & live_words_[word]; bits != 0; bits &= bits - 1) {
                visit(static_cast<uint32_t>(word * 64 + std::countr_zero(bits)));
            }
        }
    };
    if (action.rotation == CliffordRotation::PAULI) {
        // A Pauli rotation negates the changed rows: it flips the sign bits
        // of the shots it fires on and leaves every tableau alone.
        for_each_fired([&](uint32_t shot) noexcept { flip_shot_signs(shot, changed_rows_); });
        return;
    }
    const auto rotate = [&](SharedFrame& slot) noexcept {
        if (action.fixed_by_frame) {
            slot.frame.rotate_fixed_clifford(plan_->frame_pauli(action.pauli), action.rotation);
        } else {
            slot.frame.map(plan_->frame_pauli_x(action.pauli), plan_->frame_pauli_z(action.pauli),
                           slot.mapped);
            slot.frame.rotate_clifford(slot.mapped.view(), action.rotation);
        }
        slot.changed = true;
    };
    // A quarter turn changes the tableau, so the shots it fires on move to the
    // rotated one. They gather at the tail of their group, which then leaves
    // with them unless the whole group fires.
    uint32_t fired_count = 0;
    for_each_fired([&](uint32_t shot) noexcept {
        const uint32_t group = group_of_[shot];
        if (departing_[group]++ == 0) {
            fired_groups_[fired_count++] = group;
        }
        swap_lanes(position_[shot], end_[group] - departing_[group]);
    });
    for (uint32_t index = 0; index < fired_count; ++index) {
        const uint32_t group = fired_groups_[index];
        const uint32_t tail = end_[group] - departing_[group];
        departing_[group] = 0;
        const uint32_t fired = tail == begin_[group] ? group : detach_tail(group, tail);
        const uint32_t source = frame_of_[fired];
        SharedFrame& slot = frames_[source];
        if (slot.stamp != frame_stamp_) {
            slot.stamp = frame_stamp_;
            slot.link = kNoShot;
        }
        if (slot.link != kNoShot || slot.users != 1) {
            if (slot.link == kNoShot) {
                // Other groups keep this tableau: rotate a copy.
                slot.link = acquire_frame(&slot);
                rotate(frames_[slot.link]);
            }
            frame_of_[fired] = slot.link;
            ++frames_[slot.link].users;
            release_frame(source);
        } else {
            rotate(slot);
        }
        if (action.fixed_by_frame) {
#ifndef NDEBUG
            for (uint32_t shot : lanes(fired)) {
                assert(!shot_flips(shot, rows) &&
                       "a Pauli fixed by the frame keeps its sign in every shot");
            }
#endif
            continue;
        }
        // Each changed row was multiplied by the shot's image of the Pauli,
        // so it also takes that image's sign.
        for (uint32_t shot : lanes(fired)) {
            if (shot_flips(shot, rows)) {
                flip_shot_signs(shot, changed_rows_);
            }
        }
    }
}

SharingExecutor::SharedFrame& SharingExecutor::prepare_frame_rotation(
    uint32_t group, const ExecutablePlan::ExecuteFrameRotation& action) noexcept {
    SharedFrame& slot = frames_[frame_of_[group]];
    if (slot.stamp != frame_stamp_) {
        slot.stamp = frame_stamp_;
        slot.frame.map(plan_->frame_pauli_x(action.pauli), plan_->frame_pauli_z(action.pauli),
                       slot.mapped);
        const PauliStringView axis = slot.mapped.view();
        // Groups that share a tableau have the same active width.
        const uint32_t width = states_[group].active_width();
        const auto dormant = OnlineFrame::dormant_x(axis, width);
        slot.active = OnlineFrame::active(axis, width);
        slot.axis = dormant                     ? SharedFrame::Axis::Dormant
                    : slot.active.is_identity() ? SharedFrame::Axis::Identity
                                                : SharedFrame::Axis::Active;
        slot.pivot = dormant.value_or(0);
        slot.axis_sign = axis.sign();
        slot.applied = false;
    }
    return slot;
}

bool SharingExecutor::frame_rotation_needs_width(
    const ExecutablePlan::ExecuteFrameRotation& action) noexcept {
    for (uint32_t index = 0; index < group_count_; ++index) {
        const uint32_t group = groups_[index];
        if (prepare_frame_rotation(group, action).axis == SharedFrame::Axis::Dormant &&
            states_[group].active_width() == states_[group].max_active_width()) {
            return true;
        }
    }
    return false;
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteFrameRotation& action) noexcept {
    const auto signs = evaluate(action.sign);
    const auto rows = frame_pauli_rows(action.pauli);
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        SharedFrame& slot = prepare_frame_rotation(group, action);
        if (slot.axis == SharedFrame::Axis::Identity) {
            // Only dormant Z factors remain: the rotation is a global phase.
            continue;
        }
        const uint32_t width = states_[group].active_width();
        for (uint32_t shot : lanes(group)) {
            choices_[shot] = lane_bit(signs, shot) ^ slot.axis_sign ^ shot_flips(shot, rows);
        }
        const bool promotes = slot.axis == SharedFrame::Axis::Dormant;
        if (promotes && !slot.applied) {
            assert(width < states_[group].max_active_width() &&
                   "promotion width is checked before dispatch");
            // The promotion's coordinate change does not depend on the sign,
            // so it runs once for every group on this tableau.
            slot.frame.promote(slot.mapped.view(), slot.pivot, width);
            slot.applied = true;
            slot.changed = true;
        }
        const auto apply = [&](uint32_t target, bool sign) noexcept {
            if (promotes) {
                apply_promotion(states_[target], action.rotation, sign);
            } else {
                apply_rotation(states_[target],
                               {prepare_pauli_unchecked(slot.active, width), action.rotation.cosine,
                                action.rotation.sine},
                               sign);
            }
        };
        const uint32_t other = split(group);
        apply(group, choices_[lanes(group).front()] != 0);
        if (other != kNoShot) {
            apply(other, true);
        }
    }
}

void SharingExecutor::execute_action(
    const ExecutablePlan::ExecuteFrameMeasurement& action) noexcept {
    const auto rows = frame_pauli_rows(action.pauli);
    const size_t half = sign_stride_ / 2;
    // Random outcomes of measurements that leave the coefficients alone come
    // from packed draws, one bit per lane whether or not its group uses it, so
    // a seeded batch does not depend on how its shots are grouped.
    for (size_t word = 0; word < word_capacity_; ++word) {
        frame_random_[word] = rng_() & live_words_[word];
    }
    std::ranges::fill(scratch_words_, uint64_t{0});
    bool merged = false;
    const uint32_t original_group_count = group_count_;
    for (uint32_t index = 0; index < original_group_count; ++index) {
        const uint32_t group = groups_[index];
        SharedFrame& slot = frames_[frame_of_[group]];
        // Groups that share a tableau have the same active width.
        const uint32_t width = states_[group].active_width();
        if (slot.stamp != frame_stamp_) {
            // Map the Pauli and update the tableau once for every group on
            // it. A branch refers to the unsigned image and only negates the
            // rows collected in the mask, which each shot applies to its bits.
            slot.stamp = frame_stamp_;
            slot.link = kNoShot;
            OnlineFrame& frame = slot.frame;
            frame.map(plan_->frame_pauli_x(action.pauli), plan_->frame_pauli_z(action.pauli),
                      slot.mapped);
            const PauliStringView axis = slot.mapped.view();
            slot.axis_sign = axis.sign();
            const auto collect_rows = [&](bool z_bits, uint32_t coordinate) noexcept {
                std::ranges::fill(slot.rows, uint64_t{0});
                for (uint32_t q = 0; q < frame.n; ++q) {
                    const PauliString& x_image = frame.rows[q];
                    const PauliString& z_image = frame.rows[frame.n + q];
                    if (z_bits ? x_image.z().bit_get(coordinate)
                               : x_image.x().bit_get(coordinate)) {
                        slot.rows[q >> 6] |= uint64_t{1} << (q & 63);
                    }
                    if (z_bits ? z_image.z().bit_get(coordinate)
                               : z_image.x().bit_get(coordinate)) {
                        slot.rows[half + (q >> 6)] |= uint64_t{1} << (q & 63);
                    }
                }
            };
            if (const auto dormant = OnlineFrame::dormant_x(axis, width)) {
                // A fair coin that negates the rows with X on the pivot.
                slot.axis = SharedFrame::Axis::Dormant;
                collect_rows(false, *dormant);
                frame.measure_dormant(axis, *dormant, false);
                slot.changed = true;
            } else if (slot.active = OnlineFrame::active(axis, width); slot.active.is_identity()) {
                // Only dormant Z factors remain, so the unsigned image has eigenvalue +1.
                slot.axis = SharedFrame::Axis::Identity;
            } else {
                // A collapse that negates the rows with Z on the measured coordinate.
                slot.axis = SharedFrame::Axis::Active;
                slot.pivot = OnlineFrame::measurement_pivot(slot.active);
                frame.localize_measured(slot.active, slot.pivot, width);
                collect_rows(true, slot.pivot);
                frame.compact(slot.pivot, width);
                slot.changed = true;
            }
        }
        // The symbol and record refer to the plan's Pauli, so a shot's value
        // also takes the sign of its image.
        const auto record = [&](uint32_t shot, bool branch) noexcept {
            if (branch ^ slot.axis_sign ^ shot_flips(shot, rows)) {
                scratch_words_[shot >> 6] |= uint64_t{1} << (shot & 63);
            }
            if (branch) {
                flip_shot_signs(shot, slot.rows);
            }
        };
        if (slot.axis != SharedFrame::Axis::Active) {
            const bool random = slot.axis == SharedFrame::Axis::Dormant;
            for (uint32_t shot : lanes(group)) {
                record(shot, random && lane_bit(frame_random_, shot));
            }
            continue;
        }
        const PreparedMeasurement measurement =
            prepare_measurement_unchecked(slot.active, width, slot.pivot);
        const MeasurementProbabilities probabilities =
            measurement_probabilities(states_[group], measurement);
        const auto kind = classify_measurement_branch(probabilities).kind;
        for (uint32_t shot : lanes(group)) {
            const bool branch =
                kind == MeasurementBranchKind::Random
                    ? shot_rng_[shot].next_double() * probabilities.total() >= probabilities.zero
                    : kind == MeasurementBranchKind::DeterministicOne;
            choices_[shot] = branch;
            record(shot, branch);
        }
        if (width == 1) {
            // Either outcome leaves no coefficients. Dropping the scalar's
            // phase and rounding lets the groups on this tableau merge.
            State& state = states_[group];
            state.set_active_width(0);
            state.real()[0] = 1.0;
            state.imag()[0] = 0.0;
            // Groups prepared by a sharing key keep the uniform rotation
            // signs that static actions after a retirement rely on.
            if (!share_ || key_ != nullptr) {
                continue;
            }
            if (slot.link == kNoShot) {
                slot.link = group;
                continue;
            }
            for (uint32_t shot : lanes(group)) {
                group_of_[shot] = slot.link;
            }
            // The emptied group stays listed, and its state reserved, until
            // the action ends, so a later split cannot take the state.
            end_[group] = begin_[group];
            release_frame(frame_of_[group]);
            ++num_merges_;
            merged = true;
            continue;
        }
        const auto collapse = [&](uint32_t target, bool branch) noexcept {
            collapse_measurement(states_[target], measurement, branch,
                                 probabilities.for_branch(branch));
        };
        const uint32_t other = kind == MeasurementBranchKind::Random ? split(group) : kNoShot;
        collapse(group, choices_[lanes(group).front()] != 0);
        if (other != kNoShot) {
            collapse(other, true);
        }
    }
    propagate_symbol(action.branch, scratch_words_);
    if (records_.num_columns() != 0) {
        records_.assign_xor(action.record, scratch_words_, evaluate(action.correction),
                            live_words_);
    }
    if (merged) {
        collect_merged_groups();
    }
}

void SharingExecutor::execute_action(const ExecutablePlan::ExecuteFrameRetirement&) noexcept {
    // Every shot's frame is the identity again, so one tableau serves them
    // all. Each stabilizer the plan relied on came from a measurement that
    // left its coordinate dormant, so no group keeps coefficients.
    std::fill_n(shot_signs_.begin(), size_t{active_lanes()} * sign_stride_, uint64_t{0});
    for (uint32_t index = 0; index < group_count_; ++index) {
        const uint32_t group = groups_[index];
        assert(states_[group].active_width() == 0 && "a retired frame has no active coordinates");
        release_frame(frame_of_[group]);
    }
    // Groups prepared by a sharing key keep the uniform rotation signs that
    // later static actions rely on, so they stay apart.
    const bool merge = share_ && key_ == nullptr;
    uint32_t frame = kNoShot;
    for (uint32_t index = 0; index < group_count_; ++index) {
        const uint32_t group = groups_[index];
        if (merge && index != 0) {
            for (uint32_t shot : lanes(group)) {
                group_of_[shot] = groups_[0];
            }
            end_[group] = begin_[group];
            ++num_merges_;
            continue;
        }
        if (frame == kNoShot || !share_) {
            frame = acquire_frame(nullptr);
        }
        frame_of_[group] = frame;
        ++frames_[frame].users;
    }
    if (merge && group_count_ > 1) {
        collect_merged_groups();
    }
}

}  // namespace clifft::sampling
