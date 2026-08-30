#pragma once

#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/results.h"
#include "clifft/util/config.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling {

// Sampling pipeline:
//   optimized HirModule -> SamplingPlan -> ExecutablePlan -> Executor -> results
// Planning produces semantic actions, lowering prepares fixed CPU descriptors,
// Executor owns mutable state for one shot, BatchExecutor owns a packed lane
// group, and the functions below select a prepared path and collect outputs.

// Explicitly partitions sampling workers across independent shots and the
// coefficient kernels within each shot. When supplied, this layout overrides
// the automatic policy selected by threads, whose value is then ignored. The
// minimum active width is evaluated during executor setup and at coefficient
// actions; it does not introduce work inside coefficient loops.
struct ThreadLayout {
    uint32_t shot_workers = 1;
    uint32_t intra_shot_workers = 1;
    uint32_t intra_shot_min_active_width = kDefaultIntraShotMinActiveWidth;
};

// Samples a fixed number of shots into row-major visible-record storage. The
// plan and executor are prepared once, and all output is allocated before the
// first shot enters hot execution. Plans with presampled symbols are rejected
// until their sampling distribution is part of the executable contract.
// threads is a total worker budget. threads=0 selects the implementation-
// reported hardware concurrency; the public Python API spells this as
// threads="auto". Automatic scheduling uses either cross-shot or intra-shot
// workers, not a hybrid layout. batch_size is empty for adaptive selection,
// one for scalar execution, or an explicit packed lane-capacity limit. A seed
// replays within one execution configuration; scalar and packed modes, or two
// packed capacities, are statistically equivalent but need not return the same
// individual rows.
[[nodiscard]] std::vector<uint8_t> sample_records(
    const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
    uint32_t threads = 1, std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt);

// Replays each row-major visible record and returns its joint log probability.
// Unreachable records map to the lowest finite double because release builds
// assume finite arithmetic. Plans with presampled symbols or hidden records
// are rejected because this API does not yet marginalize over either source
// of hidden stochastic state.
[[nodiscard]] std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                                           std::span<const uint8_t> forced_records,
                                                           size_t num_records);

[[nodiscard]] SamplingResult sample(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed = std::nullopt,
                                    uint32_t threads = 1,
                                    std::optional<ThreadLayout> thread_layout = std::nullopt,
                                    std::optional<uint32_t> batch_size = std::nullopt);

// Samples only the selected row-major output matrices. Unselected vectors are
// empty, and selected storage is allocated before hot execution begins.
[[nodiscard]] SamplingResult sample_selected(
    const ExecutablePlan& plan, uint32_t shots, SamplingOutputSelection outputs,
    std::optional<uint64_t> seed = std::nullopt, uint32_t threads = 1,
    std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt);

// Samples fixed rows directly into caller-owned destinations. Boolean sources
// may be emitted more than once and composed at arbitrary column offsets.
// Storage is validated and initialized before hot execution.
void sample_into(const ExecutablePlan& plan, uint32_t shots, SamplingOutputBuffer output,
                 std::optional<uint64_t> seed = std::nullopt, uint32_t threads = 1,
                 std::optional<ThreadLayout> thread_layout = std::nullopt,
                 std::optional<uint32_t> batch_size = std::nullopt);

// Allocates little-endian packed Boolean rows for the selected sources.
[[nodiscard]] PackedSamplingResult sample_packed_selected(
    const ExecutablePlan& plan, uint32_t shots, SamplingOutputSelection outputs,
    std::optional<uint64_t> seed = std::nullopt, uint32_t threads = 1,
    std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt);

[[nodiscard]] SamplingSurvivorResult sample_survivors(
    const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
    bool keep_records = false, uint32_t threads = 1,
    std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt);

// Postselects shots while retaining only selected row-major matrices. Survivor
// counts and observable aggregates are populated independently of selection.
[[nodiscard]] SamplingSurvivorResult sample_survivors_selected(
    const ExecutablePlan& plan, uint32_t shots, SamplingOutputSelection outputs,
    std::optional<uint64_t> seed = std::nullopt, uint32_t threads = 1,
    std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt);

[[nodiscard]] SamplingResult sample_k(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                                      std::optional<uint64_t> seed = std::nullopt,
                                      uint32_t threads = 1,
                                      std::optional<ThreadLayout> thread_layout = std::nullopt,
                                      std::optional<uint32_t> batch_size = std::nullopt);

[[nodiscard]] SamplingSurvivorResult sample_k_survivors(
    const ExecutablePlan& plan, uint32_t shots, uint32_t k,
    std::optional<uint64_t> seed = std::nullopt, bool keep_records = false, uint32_t threads = 1,
    std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt);

}  // namespace clifft::sampling
