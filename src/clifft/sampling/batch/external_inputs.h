#pragma once

#include "clifft/sampling/batch/bits.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace clifft::sampling {

class BatchExecutor;
class ExecutablePlan;

// External noise/readout decisions let a driver reuse one quantum plan across
// packed shots, including fixed-plan sampling for supported leakage/loss models.
//
// Caller-owned, reusable input columns for one packed worker. The plan must
// outlive this buffer. Construction fixes the shape; filling and consuming
// columns never allocate. Noise columns follow ascending Presampled SymbolId,
// and readout columns follow execution-site order, as in Executor::run_shot.
class BatchExternalInputs {
  public:
    BatchExternalInputs(const ExecutablePlan& plan, uint32_t lane_capacity);

    void clear() noexcept { bits_.clear(); }
    [[nodiscard]] std::span<uint64_t> noise(uint32_t input) noexcept;
    // Flip bits for a source record of zero or one. Selection happens at the
    // readout action, after any earlier channels have updated that record.
    [[nodiscard]] std::span<uint64_t> readout_zero(uint32_t site) noexcept;
    [[nodiscard]] std::span<uint64_t> readout_one(uint32_t site) noexcept;

    [[nodiscard]] static size_t storage_bytes(const ExecutablePlan& plan, uint32_t lane_capacity);

  private:
    friend class BatchExecutor;

    const ExecutablePlan* plan_;
    uint32_t lane_capacity_;
    uint32_t noise_columns_;
    uint32_t readout_sites_;
    PackedBitColumns bits_;
};

}  // namespace clifft::sampling
