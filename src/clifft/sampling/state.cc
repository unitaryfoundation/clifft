#include "clifft/sampling/state.h"

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

void validate_scalar(std::complex<double> value) {
    if (!is_finite_robust(value.real()) || !is_finite_robust(value.imag())) {
        throw std::invalid_argument("sampling state global scalar must be finite");
    }
}

}  // namespace

State::State(uint32_t max_active_width, uint32_t initial_active_width,
             std::complex<double> initial_global_scalar)
    : initial_active_width_(initial_active_width),
      active_width_(initial_active_width),
      max_active_width_(max_active_width),
      initial_global_scalar_(initial_global_scalar),
      global_scalar_(initial_global_scalar) {
    if (max_active_width >= kDenseActiveWidthLimit) {
        throw std::invalid_argument("sampling state maximum active width must be below " +
                                    std::to_string(kDenseActiveWidthLimit));
    }
    if (initial_active_width > max_active_width) {
        throw std::invalid_argument("sampling state initial active width exceeds its maximum");
    }
    validate_scalar(initial_global_scalar);

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
    reset();
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

void State::reset() {
    if (allocation_.empty()) {
        throw std::logic_error("cannot reset a moved-from sampling state");
    }
    active_width_ = initial_active_width_;
    global_scalar_ = initial_global_scalar_;
    // Only the live prefix is observable. Promotions overwrite both halves of
    // each newly active range, and measurement scratch is overwritten on use.
    std::fill_n(real_, static_cast<size_t>(size()), 0.0);
    std::fill_n(imag_, static_cast<size_t>(size()), 0.0);
    real_[0] = 1.0;
}

void State::set_global_scalar(std::complex<double> value) {
    validate_scalar(value);
    global_scalar_ = value;
}

void State::multiply_global_scalar(std::complex<double> value) {
    validate_scalar(value);
    const std::complex<double> updated = global_scalar_ * value;
    validate_scalar(updated);
    global_scalar_ = updated;
}

void State::set_active_width(uint32_t width) {
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
    initial_global_scalar_ =
        std::exchange(other.initial_global_scalar_, std::complex<double>{1.0, 0.0});
    global_scalar_ = std::exchange(other.global_scalar_, std::complex<double>{1.0, 0.0});
}

}  // namespace clifft::sampling
