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
void copy_batch_rows(Output& output, const BatchExecutor& executor,
                     const ExecutablePlan& plan) noexcept {
    const std::span<const uint32_t> rows = executor.shot_indices();
    transpose_bit_columns_to_rows(executor.record_columns(), plan.num_visible_records(), rows,
                                  output.measurements, plan.num_visible_records());
    transpose_bit_columns_to_rows(executor.detector_columns(), plan.num_detectors(), rows,
                                  output.detectors, plan.num_detectors());
    transpose_bit_columns_to_rows(executor.observable_columns(), plan.num_observables(), rows,
                                  output.observables, plan.num_observables());
}

}  // namespace clifft::sampling::batch_detail
