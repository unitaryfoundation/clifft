#pragma once

#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/results.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling::batch_detail {

struct BatchTuningCandidate {
    BatchExecutionPolicy policy;
    uint32_t min_probe_shots;
    uint32_t max_probe_shots;
};

[[nodiscard]] std::vector<BatchTuningCandidate> batch_tuning_candidates(
    const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers, uint32_t intra_shot_workers,
    BatchOutputMode output_mode, BatchSamplingMode sampling_mode, uint64_t additional_worker_bytes,
    BatchExecutionPolicy baseline);

// The clock and probe factory are injected so budget and selection behavior can
// be tested without depending on machine load or the speed of a particular CPU.
template <typename Now, typename MakeProbe>
BatchTuningReport sweep_batch_candidates(std::span<const BatchTuningCandidate> candidates,
                                         BatchExecutionPolicy baseline, uint32_t intra_shot_workers,
                                         double budget_seconds, const SeedRoot& calibration_root,
                                         Now&& now, MakeProbe&& make_probe) {
    const double start = now();
    BatchTuningReport report;
    report.batch_size = baseline.lane_capacity;
    report.baseline_batch_size = baseline.lane_capacity;
    report.shot_workers = baseline.worker_count;
    report.intra_shot_workers = intra_shot_workers;
    report.stop_reason = "completed";
    report.trials.reserve(candidates.size());
    double best_rate = 0;
    size_t measured_candidates = 0;
    uint64_t probe_index = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const double candidate_start = now();
        const double remaining = budget_seconds - (candidate_start - start);
        if (remaining <= 0) {
            report.stop_reason = "budget_exhausted";
            break;
        }
        const auto& candidate = candidates[i];
        const double slice = remaining / static_cast<double>(candidates.size() - i);
        BatchTuningTrial trial;
        trial.batch_size = candidate.policy.lane_capacity;
        trial.shot_workers = candidate.policy.worker_count;
        {
            auto probe = make_probe(candidate.policy);
            auto run = [&](uint32_t shots) {
                const auto words =
                    derive_state(calibration_root, probe_index++, kBatchCalibrationDomain);
                probe(shots, words[0]);
                report.trial_shots += shots;
            };
            if (now() - candidate_start < slice && now() - start < budget_seconds) {
                run(candidate.min_probe_shots);
                trial.warmup_shots = candidate.min_probe_shots;
            }
            trial.setup_seconds = now() - candidate_start;
            uint32_t shots = candidate.min_probe_shots;
            while (now() - candidate_start < slice && now() - start < budget_seconds) {
                const double before = now();
                run(shots);
                const double elapsed = now() - before;
                trial.shots += shots;
                trial.elapsed_seconds += elapsed;
                // Longer chunks amortize worker launch and clock overhead on
                // cheap circuits. Candidate order and effort remain a fixed sweep.
                if (elapsed < 0.001) {
                    shots = static_cast<uint32_t>(std::min<uint64_t>(
                        static_cast<uint64_t>(shots) * 2, candidate.max_probe_shots));
                }
            }
        }
        report.trials.push_back(trial);
        if (trial.shots != 0 && trial.elapsed_seconds > 0) {
            ++measured_candidates;
            const double rate = static_cast<double>(trial.shots) / trial.elapsed_seconds;
            if (rate > best_rate) {
                best_rate = rate;
                report.batch_size = trial.batch_size;
                report.shot_workers = trial.shot_workers;
            }
        }
    }
    // An unmeasured baseline cannot justify selecting an alternative, even if
    // the latter happened to finish a cheap probe before the deadline.
    if (measured_candidates < 2 || report.trials.empty() || report.trials.front().shots == 0) {
        report.batch_size = baseline.lane_capacity;
        report.shot_workers = baseline.worker_count;
        report.stop_reason = "insufficient_measurements";
    }
    report.elapsed_seconds = now() - start;
    return report;
}

}  // namespace clifft::sampling::batch_detail
