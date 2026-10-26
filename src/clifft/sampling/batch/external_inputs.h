#pragma once

#include "clifft/sampling/batch/bits.h"

#include <cassert>
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
    // Borrow the fixed storage once when filling many shots. Start from clear()
    // and set the true inputs; unset bits remain zero. Same lifetime as the
    // owner's storage, like MutableBitColumnsView.
    class MutableView {
      public:
        [[nodiscard]] std::span<uint64_t> noise(uint32_t input) const noexcept {
            assert(input < noise_columns_);
            return bits_.column(input);
        }
        [[nodiscard]] std::span<uint64_t> readout_zero(uint32_t site) const noexcept {
            assert(site < readout_sites_);
            return bits_.column(noise_columns_ + size_t{2} * site);
        }
        [[nodiscard]] std::span<uint64_t> readout_one(uint32_t site) const noexcept {
            assert(site < readout_sites_);
            return bits_.column(noise_columns_ + size_t{2} * site + 1);
        }
        void set_noise(uint32_t input, uint32_t lane) const noexcept {
            assert(input < noise_columns_ && "external noise input must belong to the plan");
            bits_.set_bit(input, lane);
        }
        void set_readout_zero(uint32_t site, uint32_t lane) const noexcept {
            assert(site < readout_sites_ && "external readout input must belong to the plan");
            bits_.set_bit(noise_columns_ + size_t{2} * site, lane);
        }
        void set_readout_one(uint32_t site, uint32_t lane) const noexcept {
            assert(site < readout_sites_ && "external readout input must belong to the plan");
            bits_.set_bit(noise_columns_ + size_t{2} * site + 1, lane);
        }

      private:
        friend class BatchExternalInputs;
        MutableView(MutableBitColumnsView bits, uint32_t noise_columns, uint32_t readout_sites)
            : bits_(bits), noise_columns_(noise_columns), readout_sites_(readout_sites) {}
        MutableBitColumnsView bits_;
        uint32_t noise_columns_;
        uint32_t readout_sites_;
    };

    BatchExternalInputs(const ExecutablePlan& plan, uint32_t lane_capacity);

    void clear() noexcept { bits_.clear(); }
    [[nodiscard]] MutableView mutable_view() noexcept {
        return {bits_.mutable_view(), noise_columns_, readout_sites_};
    }
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
