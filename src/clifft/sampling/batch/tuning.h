#pragma once

#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/results.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling::batch_detail {

struct BatchTuningCandidate {
    BatchExecutionPolicy policy;
    uint32_t min_probe_shots;
    uint32_t max_probe_shots;
};

void validate_batch_tuning_options(std::optional<uint32_t> batch_size,
                                   std::optional<double> tuning_budget_seconds);

[[nodiscard]] std::vector<BatchTuningCandidate> batch_tuning_candidates(
    const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers, uint32_t intra_shot_workers,
    BatchOutputMode output_mode, BatchSamplingMode sampling_mode, uint64_t additional_worker_bytes,
    BatchExecutionPolicy baseline, uint64_t additional_worker_bit_columns = 0,
    uint64_t additional_output_row_bytes = 0);

// The clock and probe factory are injected so budget and selection behavior can
// be tested without depending on machine load or the speed of a particular CPU.
// production_shots amortizes setup and first use over one call when choosing a
// capacity. Unset preserves throughput-only selection for ordinary sampling.
template <typename Now, typename MakeProbe>
BatchTuningReport sweep_batch_candidates(std::span<const BatchTuningCandidate> candidates,
                                         BatchExecutionPolicy baseline, uint32_t intra_shot_workers,
                                         double budget_seconds, const SeedRoot& calibration_root,
                                         Now&& now, MakeProbe&& make_probe,
                                         std::optional<uint32_t> production_shots = std::nullopt) {
    assert(!production_shots || *production_shots > 0);
    const double start = now();
    BatchTuningReport report;
    report.batch_size = baseline.lane_capacity;
    report.baseline_batch_size = baseline.lane_capacity;
    report.shot_workers = baseline.worker_count;
    report.intra_shot_workers = intra_shot_workers;
    report.stop_reason = "completed";
    report.trials.reserve(candidates.size());
    double best_selection_rate = 0;
    double best_packed_rate_per_worker = 0;
    uint32_t smallest_measured_packed_capacity = 0;
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
        // Scalar timing cannot predict a packed speedup. After a packed probe,
        // use its throughput to avoid starting larger probes near the deadline.
        if (best_packed_rate_per_worker > 0 &&
            candidate.policy.lane_capacity > smallest_measured_packed_capacity &&
            candidate.min_probe_shots /
                    (best_packed_rate_per_worker * candidate.policy.worker_count) >
                remaining) {
            report.stop_reason = "budget_exhausted";
            continue;
        }
        const double slice = remaining / static_cast<double>(candidates.size() - i);
        BatchTuningTrial trial;
        trial.batch_size = candidate.policy.lane_capacity;
        trial.shot_workers = candidate.policy.worker_count;
        {
            auto probe = make_probe(candidate.policy);
            trial.setup_seconds = now() - candidate_start;
            auto run = [&](uint32_t shots) {
                const auto words =
                    derive_state(calibration_root, probe_index++, kBatchCalibrationDomain);
                const double before = now();
                probe(shots, words[0]);
                const double elapsed = now() - before;
                report.trial_shots += shots;
                return elapsed;
            };
            if (now() - candidate_start < slice && now() - start < budget_seconds) {
                trial.warmup_seconds = run(candidate.min_probe_shots);
                trial.warmup_shots = candidate.min_probe_shots;
            }
            uint32_t shots = candidate.min_probe_shots;
            while (now() - candidate_start < slice && now() - start < budget_seconds) {
                const double rate = trial.shots_per_second();
                if (rate > 0 && shots / rate > budget_seconds - (now() - start)) {
                    break;
                }
                const double elapsed = run(shots);
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
        if (trial.warmup_shots == 0 && trial.shots == 0) {
            report.stop_reason = "budget_exhausted";
        }
        const double rate = trial.shots_per_second();
        if (rate > 0) {
            ++measured_candidates;
            if (trial.batch_size > 1) {
                best_packed_rate_per_worker =
                    std::max(best_packed_rate_per_worker, rate / trial.shot_workers);
                smallest_measured_packed_capacity =
                    smallest_measured_packed_capacity == 0
                        ? trial.batch_size
                        : std::min(smallest_measured_packed_capacity, trial.batch_size);
            }
            // Preallocated buffers can incur page faults on their first use.
            // Charge this excess once, not once per chunk or once per shot.
            const double first_use_seconds =
                trial.used_warmup()
                    ? 0.0
                    : std::max(0.0, trial.warmup_seconds - trial.warmup_shots / rate);
            const double selection_rate =
                production_shots ? *production_shots / (trial.setup_seconds + first_use_seconds +
                                                        *production_shots / rate)
                                 : rate;
            if (selection_rate > best_selection_rate) {
                best_selection_rate = selection_rate;
                report.batch_size = trial.batch_size;
                report.shot_workers = trial.shot_workers;
            }
        }
    }
    report.elapsed_seconds = now() - start;
    // Completing a candidate's repetitions is normal. The budget only leaves
    // the sweep incomplete when it prevents measuring a candidate at all.
    if (report.stop_reason == "completed" && measured_candidates != candidates.size()) {
        report.stop_reason = "insufficient_measurements";
    }
    // An unmeasured baseline cannot justify selecting an alternative, even if
    // the latter happened to finish a cheap probe before the deadline.
    report.sufficient_measurements = measured_candidates >= 2 && !report.trials.empty() &&
                                     report.trials.front().shots_per_second() > 0;
    if (!report.sufficient_measurements) {
        report.batch_size = baseline.lane_capacity;
        report.shot_workers = baseline.worker_count;
        if (report.stop_reason != "budget_exhausted") {
            report.stop_reason = "insufficient_measurements";
        }
    }
    return report;
}

}  // namespace clifft::sampling::batch_detail
