#include "clifft/sampling/soa_state.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft::sampling {

namespace {

constexpr size_t kStateAlignment = 64;
constexpr uint64_t kDoublesPerAlignment = kStateAlignment / sizeof(double);

uint64_t round_plane_stride(uint64_t entries) {
    return std::max(kDoublesPerAlignment,
                    (entries + kDoublesPerAlignment - 1) & ~(kDoublesPerAlignment - 1));
}

void validate_scalar(std::complex<double> value) {
    if (!is_finite_robust(value.real()) || !is_finite_robust(value.imag())) {
        throw std::invalid_argument("SoA state global scalar must be finite");
    }
}

}  // namespace

SoaState::SoaState(uint32_t max_active_width, uint32_t initial_active_width,
                   std::complex<double> initial_global_scalar)
    : initial_active_width_(initial_active_width),
      active_width_(initial_active_width),
      max_active_width_(max_active_width),
      initial_global_scalar_(initial_global_scalar),
      global_scalar_(initial_global_scalar) {
    if (max_active_width >= kDenseActiveWidthLimit) {
        throw std::invalid_argument("SoA state maximum active width must be below " +
                                    std::to_string(kDenseActiveWidthLimit));
    }
    if (initial_active_width > max_active_width) {
        throw std::invalid_argument("SoA state initial active width exceeds its maximum");
    }
    validate_scalar(initial_global_scalar);

    capacity_ = uint64_t{1} << max_active_width;
    coefficient_stride_ = round_plane_stride(capacity_);
    const uint64_t scratch_capacity = std::max(uint64_t{1}, capacity_ >> 1);
    scratch_stride_ = round_plane_stride(scratch_capacity);
    const uint64_t allocated_doubles = 2 * coefficient_stride_ + 2 * scratch_stride_;
    if (allocated_doubles > std::numeric_limits<size_t>::max() / sizeof(double)) {
        throw std::length_error("SoA state allocation exceeds addressable memory");
    }
    allocation_bytes_ = static_cast<size_t>(allocated_doubles) * sizeof(double);
    allocation_ = static_cast<double*>(
        ::operator new[](allocation_bytes_, std::align_val_t{kStateAlignment}));
    real_ = allocation_;
    imag_ = real_ + coefficient_stride_;
    scratch_real_ = imag_ + coefficient_stride_;
    scratch_imag_ = scratch_real_ + scratch_stride_;
    reset();
}

SoaState::~SoaState() {
    release();
}

SoaState::SoaState(SoaState&& other) noexcept {
    move_from(std::move(other));
}

SoaState& SoaState::operator=(SoaState&& other) noexcept {
    if (this != &other) {
        release();
        move_from(std::move(other));
    }
    return *this;
}

void SoaState::reset() {
    if (allocation_ == nullptr) {
        throw std::logic_error("cannot reset a moved-from SoA state");
    }
    active_width_ = initial_active_width_;
    global_scalar_ = initial_global_scalar_;
    // Only the live prefix is observable. Promotions overwrite both halves of
    // each newly active range, and measurement scratch is overwritten on use.
    std::fill_n(real_, static_cast<size_t>(size()), 0.0);
    std::fill_n(imag_, static_cast<size_t>(size()), 0.0);
    real_[0] = 1.0;
}

void SoaState::set_global_scalar(std::complex<double> value) {
    validate_scalar(value);
    global_scalar_ = value;
}

void SoaState::multiply_global_scalar(std::complex<double> value) {
    validate_scalar(value);
    const std::complex<double> updated = global_scalar_ * value;
    validate_scalar(updated);
    global_scalar_ = updated;
}

void SoaState::set_active_width(uint32_t width) {
    if (width > max_active_width_) {
        throw std::out_of_range("SoA state active width exceeds its allocation");
    }
    active_width_ = width;
}

void SoaState::release() noexcept {
    if (allocation_ != nullptr) {
        ::operator delete[](allocation_, std::align_val_t{kStateAlignment});
    }
    allocation_ = nullptr;
    real_ = nullptr;
    imag_ = nullptr;
    scratch_real_ = nullptr;
    scratch_imag_ = nullptr;
}

void SoaState::move_from(SoaState&& other) noexcept {
    allocation_ = std::exchange(other.allocation_, nullptr);
    real_ = std::exchange(other.real_, nullptr);
    imag_ = std::exchange(other.imag_, nullptr);
    scratch_real_ = std::exchange(other.scratch_real_, nullptr);
    scratch_imag_ = std::exchange(other.scratch_imag_, nullptr);
    capacity_ = std::exchange(other.capacity_, 0);
    coefficient_stride_ = std::exchange(other.coefficient_stride_, 0);
    scratch_stride_ = std::exchange(other.scratch_stride_, 0);
    allocation_bytes_ = std::exchange(other.allocation_bytes_, 0);
    initial_active_width_ = std::exchange(other.initial_active_width_, 0);
    active_width_ = std::exchange(other.active_width_, 0);
    max_active_width_ = std::exchange(other.max_active_width_, 0);
    initial_global_scalar_ =
        std::exchange(other.initial_global_scalar_, std::complex<double>{1.0, 0.0});
    global_scalar_ = std::exchange(other.global_scalar_, std::complex<double>{1.0, 0.0});
}

}  // namespace clifft::sampling
