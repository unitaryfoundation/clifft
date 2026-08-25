#include "clifft/sampling/interleaved_batch_state.h"

#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>

namespace clifft::sampling {

namespace {

constexpr uint32_t kLaneAlignment = 8;

uint64_t checked_coefficient_capacity(uint32_t max_active_width) {
    if (max_active_width >= kDenseActiveWidthLimit) {
        throw std::invalid_argument("interleaved batch maximum active width must be below " +
                                    std::to_string(kDenseActiveWidthLimit));
    }
    return uint64_t{1} << max_active_width;
}

uint32_t rounded_lane_pitch(uint32_t lanes) {
    if (lanes == 0) {
        throw std::invalid_argument("interleaved batch lane capacity must be positive");
    }
    if (lanes > std::numeric_limits<uint32_t>::max() - (kLaneAlignment - 1)) {
        throw std::length_error("interleaved batch lane pitch exceeds uint32_t");
    }
    return (lanes + kLaneAlignment - 1) & ~(kLaneAlignment - 1);
}

size_t checked_array_bytes(uint64_t entries, uint32_t pitch, uint32_t arrays) {
    constexpr size_t kElementBytes = sizeof(double);
    const uint64_t total_entries = entries * static_cast<uint64_t>(pitch);
    if (pitch != 0 && total_entries / pitch != entries) {
        throw std::length_error("interleaved batch allocation exceeds uint64_t");
    }
    if (total_entries > std::numeric_limits<size_t>::max() / kElementBytes / arrays) {
        throw std::length_error("interleaved batch allocation exceeds size_t");
    }
    return static_cast<size_t>(total_entries) * kElementBytes * arrays;
}

}  // namespace

InterleavedBatchState::InterleavedBatchState(uint32_t max_active_width,
                                             uint32_t initial_active_width,
                                             uint32_t lane_capacity)
    : coefficient_capacity_(checked_coefficient_capacity(max_active_width)),
      scratch_capacity_(std::max(uint64_t{1}, coefficient_capacity_ >> 1)),
      lane_capacity_(lane_capacity),
      lane_pitch_(rounded_lane_pitch(lane_capacity)),
      initial_active_width_(initial_active_width),
      active_width_(initial_active_width),
      max_active_width_(max_active_width) {
    if (initial_active_width > max_active_width) {
        throw std::invalid_argument(
            "interleaved batch initial active width exceeds its maximum");
    }
    const size_t coefficient_bytes =
        checked_array_bytes(coefficient_capacity_, lane_pitch_, 2);
    const size_t scratch_bytes = checked_array_bytes(scratch_capacity_, lane_pitch_, 2);
    if (coefficient_bytes > std::numeric_limits<size_t>::max() - scratch_bytes) {
        throw std::length_error("interleaved batch allocation exceeds size_t");
    }
    allocation_ = PageAlignedAllocation(coefficient_bytes + scratch_bytes);
    auto* storage = static_cast<double*>(allocation_.data());
    const size_t coefficient_entries =
        static_cast<size_t>(coefficient_capacity_) * lane_pitch_;
    const size_t scratch_entries = static_cast<size_t>(scratch_capacity_) * lane_pitch_;
    real_ = storage;
    imag_ = real_ + coefficient_entries;
    scratch_real_ = imag_ + coefficient_entries;
    scratch_imag_ = scratch_real_ + scratch_entries;
    reset(lane_capacity_);
}

void InterleavedBatchState::reset(uint32_t active_lanes) noexcept {
    assert(active_lanes <= lane_capacity_ && "active batch lanes must fit retained storage");
    active_lanes_ = active_lanes;
    active_width_ = initial_active_width_;
    const size_t entries = static_cast<size_t>(size()) * lane_pitch_;
    std::fill_n(real_, entries, 0.0);
    std::fill_n(imag_, entries, 0.0);
    std::fill_n(real_, active_lanes_, 1.0);
}

double* InterleavedBatchState::real_basis(uint64_t basis) noexcept {
    assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
    return real_ + static_cast<size_t>(basis) * lane_pitch_;
}

const double* InterleavedBatchState::real_basis(uint64_t basis) const noexcept {
    assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
    return real_ + static_cast<size_t>(basis) * lane_pitch_;
}

double* InterleavedBatchState::imag_basis(uint64_t basis) noexcept {
    assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
    return imag_ + static_cast<size_t>(basis) * lane_pitch_;
}

const double* InterleavedBatchState::imag_basis(uint64_t basis) const noexcept {
    assert(basis < coefficient_capacity_ && "basis index must fit coefficient storage");
    return imag_ + static_cast<size_t>(basis) * lane_pitch_;
}

double* InterleavedBatchState::scratch_real_basis(uint64_t basis) noexcept {
    assert(basis < scratch_capacity_ && "basis index must fit measurement scratch");
    return scratch_real_ + static_cast<size_t>(basis) * lane_pitch_;
}

double* InterleavedBatchState::scratch_imag_basis(uint64_t basis) noexcept {
    assert(basis < scratch_capacity_ && "basis index must fit measurement scratch");
    return scratch_imag_ + static_cast<size_t>(basis) * lane_pitch_;
}

void InterleavedBatchState::set_active_width(uint32_t width) noexcept {
    assert(width <= max_active_width_ && "active width must fit interleaved batch storage");
    active_width_ = width;
}

}  // namespace clifft::sampling
