#pragma once

#include "clifft/util/page_allocation.h"

#include <cstddef>
#include <cstdint>

namespace clifft::sampling {

// Dense active-coordinate storage whose innermost dimension is the shot lane.
// Every shot in a batch follows the same prepared active-width transitions, so
// one width describes the whole live prefix. Construction allocates the
// maximum coefficient and measurement-scratch footprint used by hot kernels.
class InterleavedBatchState {
  public:
    InterleavedBatchState(uint32_t max_active_width, uint32_t initial_active_width,
                          uint32_t lane_capacity);

    InterleavedBatchState(const InterleavedBatchState&) = delete;
    InterleavedBatchState& operator=(const InterleavedBatchState&) = delete;
    InterleavedBatchState(InterleavedBatchState&&) noexcept = default;
    InterleavedBatchState& operator=(InterleavedBatchState&&) noexcept = default;

    void reset(uint32_t active_lanes) noexcept;

    [[nodiscard]] uint32_t active_width() const noexcept { return active_width_; }
    [[nodiscard]] uint32_t initial_active_width() const noexcept {
        return initial_active_width_;
    }
    [[nodiscard]] uint32_t max_active_width() const noexcept { return max_active_width_; }
    [[nodiscard]] uint32_t active_lanes() const noexcept { return active_lanes_; }
    [[nodiscard]] uint32_t lane_capacity() const noexcept { return lane_capacity_; }
    [[nodiscard]] uint32_t lane_pitch() const noexcept { return lane_pitch_; }
    [[nodiscard]] uint64_t size() const noexcept { return uint64_t{1} << active_width_; }
    [[nodiscard]] uint64_t capacity() const noexcept { return coefficient_capacity_; }

    [[nodiscard]] double* real_basis(uint64_t basis) noexcept;
    [[nodiscard]] const double* real_basis(uint64_t basis) const noexcept;
    [[nodiscard]] double* imag_basis(uint64_t basis) noexcept;
    [[nodiscard]] const double* imag_basis(uint64_t basis) const noexcept;
    [[nodiscard]] double* scratch_real_basis(uint64_t basis) noexcept;
    [[nodiscard]] double* scratch_imag_basis(uint64_t basis) noexcept;

    void set_active_width(uint32_t width) noexcept;

  private:
    PageAlignedAllocation allocation_;
    double* real_ = nullptr;
    double* imag_ = nullptr;
    double* scratch_real_ = nullptr;
    double* scratch_imag_ = nullptr;
    uint64_t coefficient_capacity_ = 0;
    uint64_t scratch_capacity_ = 0;
    uint32_t lane_capacity_ = 0;
    uint32_t lane_pitch_ = 0;
    uint32_t initial_active_width_ = 0;
    uint32_t active_width_ = 0;
    uint32_t max_active_width_ = 0;
    uint32_t active_lanes_ = 0;
};

}  // namespace clifft::sampling
