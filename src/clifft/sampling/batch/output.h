#pragma once

#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/executable_plan.h"

#include <cstddef>
#include <cstdint>

namespace clifft::sampling::batch_detail {

template <typename Output>
void copy_batch_lane(Output& output, const BatchExecutor& executor, uint32_t lane, uint32_t shot,
                     const ExecutablePlan& plan) noexcept {
    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
        output.measurements[static_cast<size_t>(shot) * plan.num_visible_records() + record] =
            static_cast<uint8_t>(executor.measurement(lane, record));
    }
    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
        output.detectors[static_cast<size_t>(shot) * plan.num_detectors() + detector] =
            static_cast<uint8_t>(executor.detector(lane, detector));
    }
    for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
        output.observables[static_cast<size_t>(shot) * plan.num_observables() + observable] =
            static_cast<uint8_t>(executor.observable(lane, observable));
    }
}

}  // namespace clifft::sampling::batch_detail
