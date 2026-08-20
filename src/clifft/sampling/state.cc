#include "clifft/sampling/state.h"

#include "clifft/util/intra_shot_parallel.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft::sampling {

namespace {

constexpr size_t kStateAlignment = 64;
constexpr uint64_t kDoublesPerAlignment = kStateAlignment / sizeof(double);

uint64_t round_array_stride(uint64_t entries) {
    return std::max(kDoublesPerAlignment,
                    (entries + kDoublesPerAlignment - 1) & ~(kDoublesPerAlignment - 1));
}

}  // namespace

State::State(uint32_t max_active_width, uint32_t initial_active_width,
             uint32_t initialization_workers, uint32_t intra_shot_min_active_width)
    : initial_active_width_(initial_active_width),
      active_width_(initial_active_width),
      max_active_width_(max_active_width) {
    if (max_active_width >= kDenseActiveWidthLimit) {
        throw std::invalid_argument("sampling state maximum active width must be below " +
                                    std::to_string(kDenseActiveWidthLimit));
    }
    if (initial_active_width > max_active_width) {
        throw std::invalid_argument("sampling state initial active width exceeds its maximum");
    }
    capacity_ = uint64_t{1} << max_active_width;
    coefficient_stride_ = round_array_stride(capacity_);
    const uint64_t scratch_capacity = std::max(uint64_t{1}, capacity_ >> 1);
    scratch_stride_ = round_array_stride(scratch_capacity);
    const uint64_t allocated_doubles = 2 * coefficient_stride_ + 2 * scratch_stride_;
    if (allocated_doubles > std::numeric_limits<size_t>::max() / sizeof(double)) {
        throw std::length_error("sampling state allocation exceeds addressable memory");
    }
    const size_t allocation_bytes = static_cast<size_t>(allocated_doubles) * sizeof(double);
    allocation_ = PageAlignedAllocation(allocation_bytes);
    real_ = static_cast<double*>(allocation_.data());
    imag_ = real_ + coefficient_stride_;
    scratch_real_ = imag_ + coefficient_stride_;
    scratch_imag_ = scratch_real_ + scratch_stride_;
    if (should_parallelize_intra_shot(max_active_width_, initialization_workers,
                                      intra_shot_min_active_width)) {
        // Matching each worker's future coefficient range here lets the OS
        // place physical pages by first touch without changing allocation.
        intra_shot_parallel_ranges(coefficient_stride_, initialization_workers,
                                   [&](uint64_t begin, uint64_t end) noexcept {
                                       std::fill(real_ + begin, real_ + end, 0.0);
                                       std::fill(imag_ + begin, imag_ + end, 0.0);
                                   });
        real_[0] = 1.0;
    } else {
        reset();
    }
}

State::~State() {
    release();
}

State::State(State&& other) noexcept {
    move_from(std::move(other));
}

State& State::operator=(State&& other) noexcept {
    if (this != &other) {
        release();
        move_from(std::move(other));
    }
    return *this;
}

void State::reset() noexcept {
    assert(!allocation_.empty() && "cannot reset a moved-from sampling state");
    active_width_ = initial_active_width_;
    // Only the live prefix is observable. Promotions overwrite both halves of
    // each newly active range, and measurement scratch is overwritten on use.
    std::fill_n(real_, static_cast<size_t>(size()), 0.0);
    std::fill_n(imag_, static_cast<size_t>(size()), 0.0);
    real_[0] = 1.0;
}

void State::reset_parallel(uint32_t workers, uint32_t min_active_width) noexcept {
    if (!should_parallelize_intra_shot(initial_active_width_, workers, min_active_width)) {
        reset();
        return;
    }
    assert(!allocation_.empty() && "cannot reset a moved-from sampling state");
    active_width_ = initial_active_width_;
    intra_shot_parallel_ranges(size(), workers, [&](uint64_t begin, uint64_t end) noexcept {
        std::fill(real_ + begin, real_ + end, 0.0);
        std::fill(imag_ + begin, imag_ + end, 0.0);
    });
    real_[0] = 1.0;
}

void State::ensure_capacity(uint32_t max_active_width) {
    if (max_active_width <= max_active_width_) {
        return;
    }
    if (max_active_width >= kDenseActiveWidthLimit) {
        throw std::invalid_argument("sampling state maximum active width must be below " +
                                    std::to_string(kDenseActiveWidthLimit));
    }

    const uint64_t new_capacity = uint64_t{1} << max_active_width;
    const uint64_t new_coefficient_stride = round_array_stride(new_capacity);
    const uint64_t new_scratch_capacity = std::max(uint64_t{1}, new_capacity >> 1);
    const uint64_t new_scratch_stride = round_array_stride(new_scratch_capacity);
    const uint64_t allocated_doubles = 2 * new_coefficient_stride + 2 * new_scratch_stride;
    if (allocated_doubles > std::numeric_limits<size_t>::max() / sizeof(double)) {
        throw std::length_error("sampling state allocation exceeds addressable memory");
    }

    // The old block must remain live until both coefficient arrays are copied.
    // This transient old-plus-new peak preserves the shot if allocation fails;
    // portable aligned allocation has no in-place growth operation.
    PageAlignedAllocation allocation(static_cast<size_t>(allocated_doubles) * sizeof(double));
    double* const new_real = static_cast<double*>(allocation.data());
    double* const new_imag = new_real + new_coefficient_stride;
    std::copy_n(real_, static_cast<size_t>(size()), new_real);
    std::copy_n(imag_, static_cast<size_t>(size()), new_imag);

    allocation_ = std::move(allocation);
    real_ = new_real;
    imag_ = new_imag;
    scratch_real_ = imag_ + new_coefficient_stride;
    scratch_imag_ = scratch_real_ + new_scratch_stride;
    capacity_ = new_capacity;
    coefficient_stride_ = new_coefficient_stride;
    scratch_stride_ = new_scratch_stride;
    max_active_width_ = max_active_width;
}

void State::set_active_width(uint32_t width) noexcept {
    assert(width <= max_active_width_ && "active width must fit the sampling state allocation");
    active_width_ = width;
}

void State::release() noexcept {
    allocation_.reset();
    real_ = nullptr;
    imag_ = nullptr;
    scratch_real_ = nullptr;
    scratch_imag_ = nullptr;
}

void State::move_from(State&& other) noexcept {
    allocation_ = std::move(other.allocation_);
    real_ = std::exchange(other.real_, nullptr);
    imag_ = std::exchange(other.imag_, nullptr);
    scratch_real_ = std::exchange(other.scratch_real_, nullptr);
    scratch_imag_ = std::exchange(other.scratch_imag_, nullptr);
    capacity_ = std::exchange(other.capacity_, 0);
    coefficient_stride_ = std::exchange(other.coefficient_stride_, 0);
    scratch_stride_ = std::exchange(other.scratch_stride_, 0);
    initial_active_width_ = std::exchange(other.initial_active_width_, 0);
    active_width_ = std::exchange(other.active_width_, 0);
    max_active_width_ = std::exchange(other.max_active_width_, 0);
}

}  // namespace clifft::sampling
