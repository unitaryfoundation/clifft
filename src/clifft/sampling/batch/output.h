#pragma once

// Materializes retained rows after executor dispatch and assembles survivor batches.

#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/results.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace clifft::sampling::batch_detail {

inline size_t checked_output_size(uint32_t shots, size_t stride) {
    if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error("sampling output size exceeds size_t range");
    }
    return static_cast<size_t>(shots) * stride;
}

template <typename Result>
void resize_output_rows(Result& result, uint32_t shots, const ExecutablePlan& plan,
                        SamplingOutputs outputs) {
    result.measurements.resize(checked_output_size(
        shots, outputs.measurements() ? outputs.stride(plan.num_visible_records()) : 0));
    result.detectors.resize(
        checked_output_size(shots, outputs.syndromes() ? outputs.stride(plan.num_detectors()) : 0));
    result.observables.resize(checked_output_size(
        shots, outputs.syndromes() ? outputs.stride(plan.num_observables()) : 0));
    result.exp_vals.resize(
        checked_output_size(shots, outputs.expectations() ? plan.num_exp_vals() : 0));
}

// Retains only completed survivor rows. Workers write disjoint batch slots
// after dispatch; final assembly restores shot order independently of scheduling.
class SurvivorRowCollector {
  public:
    SurvivorRowCollector(uint32_t shots, uint32_t capacity, bool retain)
        : batches_(retain ? (uint64_t{shots} + capacity - 1) / capacity : 0), capacity_(capacity) {}

    SamplingResult& batch(uint32_t first_shot) { return batches_[first_shot / capacity_]; }

    void finish(SamplingSurvivorResult& result, const ExecutablePlan& plan,
                SamplingOutputs outputs) {
        resize_output_rows(result, result.passed_shots, plan, outputs);
        size_t measurements = 0, detectors = 0, observables = 0, expectations = 0;
        for (auto& batch : batches_) {
            std::ranges::copy(batch.measurements, result.measurements.begin() + measurements);
            std::ranges::copy(batch.detectors, result.detectors.begin() + detectors);
            std::ranges::copy(batch.observables, result.observables.begin() + observables);
            std::ranges::copy(batch.exp_vals, result.exp_vals.begin() + expectations);
            measurements += batch.measurements.size();
            detectors += batch.detectors.size();
            observables += batch.observables.size();
            expectations += batch.exp_vals.size();
            batch = {};
        }
    }

  private:
    std::vector<SamplingResult> batches_;
    uint32_t capacity_;
};

// Scatters the finalized lanes of one batch into row-major result vectors, one
// row per shot index. Transposing whole columns reads each packed word once
// instead of once per lane.
template <typename Output>
void copy_batch_rows(Output& output, const BatchClassicalState& executor,
                     const ExecutablePlan& plan, SamplingOutputs outputs,
                     std::span<const uint32_t> rows) noexcept {
    if (outputs.measurements()) {
        transpose_bit_columns_to_rows(
            executor.record_columns(), plan.num_visible_records(), rows, output.measurements,
            outputs.stride(plan.num_visible_records()), outputs.bit_packed);
    }
    if (outputs.syndromes()) {
        transpose_bit_columns_to_rows(executor.detector_columns(), plan.num_detectors(), rows,
                                      output.detectors, outputs.stride(plan.num_detectors()),
                                      outputs.bit_packed);
        transpose_bit_columns_to_rows(executor.observable_columns(), plan.num_observables(), rows,
                                      output.observables, outputs.stride(plan.num_observables()),
                                      outputs.bit_packed);
    }
}

template <typename Output>
void copy_batch_rows(Output& output, const BatchClassicalState& executor,
                     const ExecutablePlan& plan, SamplingOutputs outputs = {}) noexcept {
    copy_batch_rows(output, executor, plan, outputs, executor.shot_indices());
}

}  // namespace clifft::sampling::batch_detail
