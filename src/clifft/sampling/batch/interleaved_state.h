#pragma once

#include "clifft/util/page_allocation.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace clifft::sampling {

// Return the SIMD-aligned lane stride used by interleaved allocations.
[[nodiscard]] uint32_t interleaved_batch_lane_pitch(uint32_t lane_capacity);

// Return the complete coefficient and measurement-scratch allocation size.
[[nodiscard]] size_t interleaved_batch_state_bytes(uint32_t max_active_width,
                                                   uint32_t lane_capacity);

// Dense complex state-vector storage for a batch of shots. Basis b is the
// computational-basis bit pattern over the current active coordinates, and s
// is a shot lane:
//
//   real amplitude(b, s) = real_[b * lane_pitch_ + s]
//   imag amplitude(b, s) = imag_[b * lane_pitch_ + s]
//
// Every lane follows the same prepared active-width transitions. Construction
// allocates amplitudes and measurement scratch for the maximum active width so
// hot kernels only reuse retained storage.
class InterleavedBatchState {
  public:
    InterleavedBatchState(uint32_t max_active_width, uint32_t initial_active_width,
                          uint32_t lane_capacity);

    InterleavedBatchState(const InterleavedBatchState&) = delete;
    InterleavedBatchState& operator=(const InterleavedBatchState&) = delete;
    InterleavedBatchState(InterleavedBatchState&&) noexcept = default;
    InterleavedBatchState& operator=(InterleavedBatchState&&) noexcept = default;

    void reset(uint32_t active_lanes) noexcept;
    void compact_lanes(std::span<const uint32_t> source_lanes) noexcept;

    [[nodiscard]] uint32_t active_width() const noexcept { return active_width_; }
    [[nodiscard]] uint32_t max_active_width() const noexcept { return max_active_width_; }
    [[nodiscard]] uint32_t active_lanes() const noexcept { return active_lanes_; }
    [[nodiscard]] uint32_t lane_capacity() const noexcept { return lane_capacity_; }
    [[nodiscard]] uint32_t lane_pitch() const noexcept { return lane_pitch_; }
    [[nodiscard]] uint64_t size() const noexcept { return uint64_t{1} << active_width_; }
    [[nodiscard]] uint64_t capacity() const noexcept { return coefficient_capacity_; }

    [[nodiscard]] double* real_basis(uint64_t basis) noexcept {
        assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
        return real_ + static_cast<size_t>(basis) * lane_pitch_;
    }
    [[nodiscard]] const double* real_basis(uint64_t basis) const noexcept {
        assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
        return real_ + static_cast<size_t>(basis) * lane_pitch_;
    }
    [[nodiscard]] double* imag_basis(uint64_t basis) noexcept {
        assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
        return imag_ + static_cast<size_t>(basis) * lane_pitch_;
    }
    [[nodiscard]] const double* imag_basis(uint64_t basis) const noexcept {
        assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
        return imag_ + static_cast<size_t>(basis) * lane_pitch_;
    }
    [[nodiscard]] double* scratch_real_basis(uint64_t basis) noexcept {
        assert(basis < scratch_capacity_ && "basis index must fit measurement scratch");
        return scratch_real_ + static_cast<size_t>(basis) * lane_pitch_;
    }
    [[nodiscard]] double* scratch_imag_basis(uint64_t basis) noexcept {
        assert(basis < scratch_capacity_ && "basis index must fit measurement scratch");
        return scratch_imag_ + static_cast<size_t>(basis) * lane_pitch_;
    }

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
