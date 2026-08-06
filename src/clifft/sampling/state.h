#pragma once

#include "clifft/util/numeric.h"
#include "clifft/util/page_allocation.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

namespace clifft::sampling {

// Holds one shot's state-vector coefficients for direct-Pauli execution. An
// executor initializes the scalar from SamplingPlan::global_weight; it may have
// non-unit magnitude and accumulates signed-identity phases while the real and
// imaginary coefficient arrays remain normalized. The same allocation contains
// temporary arrays used while collapsing a non-diagonal measurement.
class State {
  public:
    // Immediately allocates all coefficient and temporary storage required by
    // max_active_width, even when initial_active_width is smaller.
    explicit State(uint32_t max_active_width, uint32_t initial_active_width = 0,
                   std::complex<double> initial_global_scalar = {1.0, 0.0});
    ~State();

    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&& other) noexcept;
    State& operator=(State&& other) noexcept;

    // Restore the configured initial width, scalar, and |0...0> coefficients.
    // The allocation and all array addresses remain unchanged.
    void reset();

    [[nodiscard]] uint32_t active_width() const { return active_width_; }
    [[nodiscard]] uint32_t initial_active_width() const { return initial_active_width_; }
    [[nodiscard]] uint32_t max_active_width() const { return max_active_width_; }
    [[nodiscard]] uint64_t size() const { return uint64_t{1} << active_width_; }
    [[nodiscard]] uint64_t capacity() const { return capacity_; }

    [[nodiscard]] std::span<double> real() { return {real_, static_cast<size_t>(size())}; }
    [[nodiscard]] std::span<const double> real() const {
        return {real_, static_cast<size_t>(size())};
    }
    [[nodiscard]] std::span<double> imag() { return {imag_, static_cast<size_t>(size())}; }
    [[nodiscard]] std::span<const double> imag() const {
        return {imag_, static_cast<size_t>(size())};
    }

    [[nodiscard]] double* real_data() { return real_; }
    [[nodiscard]] const double* real_data() const { return real_; }
    [[nodiscard]] double* imag_data() { return imag_; }
    [[nodiscard]] const double* imag_data() const { return imag_; }
    [[nodiscard]] double* scratch_real_data() { return scratch_real_; }
    [[nodiscard]] const double* scratch_real_data() const { return scratch_real_; }
    [[nodiscard]] double* scratch_imag_data() { return scratch_imag_; }
    [[nodiscard]] const double* scratch_imag_data() const { return scratch_imag_; }

    [[nodiscard]] std::complex<double> global_scalar() const { return global_scalar_; }
    void set_global_scalar(std::complex<double> value);
    void multiply_global_scalar(std::complex<double> value);

    // Kernel-only width transition. The caller must stay within the maximum
    // chosen at construction.
    void set_active_width(uint32_t width);

  private:
    void release() noexcept;
    void move_from(State&& other) noexcept;

    PageAlignedAllocation allocation_;
    double* real_ = nullptr;
    double* imag_ = nullptr;
    double* scratch_real_ = nullptr;
    double* scratch_imag_ = nullptr;
    uint64_t capacity_ = 0;
    uint64_t coefficient_stride_ = 0;
    uint64_t scratch_stride_ = 0;
    uint32_t initial_active_width_ = 0;
    uint32_t active_width_ = 0;
    uint32_t max_active_width_ = 0;
    std::complex<double> initial_global_scalar_ = {1.0, 0.0};
    std::complex<double> global_scalar_ = {1.0, 0.0};
};

}  // namespace clifft::sampling
