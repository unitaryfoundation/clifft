#include "clifft/sampling/batch/tuning.h"

#include "clifft/sampling/executable_plan.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace clifft::sampling::batch_detail {

std::vector<BatchTuningCandidate> batch_tuning_candidates(
    const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers, uint32_t intra_shot_workers,
    BatchOutputMode output_mode, BatchSamplingMode sampling_mode, uint64_t additional_worker_bytes,
    BatchExecutionPolicy baseline) {
    std::vector<BatchTuningCandidate> candidates;
    if (shots == 0) {
        return candidates;
    }
    // Calibration must not allocate production-sized output arrays just to
    // compare a handful of policies. Worker storage has its own budgets below.
    constexpr uint64_t kProbeOutputBudget = 8 * 1024 * 1024;
    const uint64_t row_bytes =
        output_mode == BatchOutputMode::Rows
            ? static_cast<uint64_t>(plan.num_visible_records()) + plan.num_detectors() +
                  plan.num_observables() +
                  static_cast<uint64_t>(plan.num_exp_vals()) * sizeof(double) + 1
            : 0;
    const uint32_t max_probe_shots = static_cast<uint32_t>(
        std::min<uint64_t>(shots, kProbeOutputBudget / std::max<uint64_t>(1, row_bytes)));
    const auto append = [&](BatchExecutionPolicy policy) {
        const uint64_t minimum = std::min<uint64_t>(
            shots, static_cast<uint64_t>(policy.lane_capacity) * policy.worker_count);
        if (minimum > max_probe_shots ||
            std::ranges::any_of(candidates, [&](const auto& candidate) {
                return candidate.policy.lane_capacity == policy.lane_capacity &&
                       candidate.policy.worker_count == policy.worker_count;
            })) {
            return;
        }
        candidates.push_back({policy, static_cast<uint32_t>(minimum), max_probe_shots});
    };
    append(baseline);
    if (candidates.empty() || !kPackedBatchExecutionAvailable || intra_shot_workers > 1) {
        return candidates;
    }
    for (const uint32_t size : std::array<uint32_t, 5>{1, 64, 256, 1024, 2048}) {
        const uint32_t capacity = std::min(size, shots);
        const uint32_t workers = static_cast<uint32_t>(std::min<uint64_t>(
            shot_workers, (static_cast<uint64_t>(shots) + capacity - 1) / capacity));
        if (capacity > 1) {
            const uint64_t bytes = saturating_add_u64(
                batch_worker_storage_bytes(plan, capacity, output_mode, sampling_mode),
                additional_worker_bytes);
            if (bytes > kDefaultBatchWorkerBudget ||
                saturating_multiply_u64(bytes, workers) > kDefaultBatchTotalWorkerBudget) {
                continue;
            }
        }
        append({capacity, workers});
    }
    return candidates;
}

}  // namespace clifft::sampling::batch_detail
