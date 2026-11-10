#pragma once

#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/executable_plan.h"

#include <cstddef>
#include <cstdint>

namespace clifft::sampling::batch_detail {

// Scatters the finalized lanes of one batch into row-major result vectors, one
// row per shot index. Transposing whole columns reads each packed word once
// instead of once per lane.
template <typename Output>
void copy_batch_rows(Output& output, const BatchExecutor& executor, const ExecutablePlan& plan,
                     SamplingOutputs outputs = {}) noexcept {
    const std::span<const uint32_t> rows = executor.shot_indices();
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

}  // namespace clifft::sampling::batch_detail
