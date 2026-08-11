#pragma once

#include "clifft/sampling/kernels.h"

#include <cstdint>

namespace clifft::internal {
enum class RuntimeIsa;
}

namespace clifft::sampling {

// Architecture-neutral selection stored in a prepared instrument action.
enum class NewXInstrumentKernel : uint8_t {
    NotApplicable,
    Scalar,
    Avx2,
};

static_assert(sizeof(NewXInstrumentKernel) == 1);

[[nodiscard]] NewXInstrumentKernel resolve_new_x_instrument_kernel(
    bool activates_new_x, uint32_t active_width, internal::RuntimeIsa runtime_isa) noexcept;

void apply_new_x_instrument_no_fire(State& state, double factor_zero, double factor_one,
                                    double no_fire_probability,
                                    NewXInstrumentKernel kernel) noexcept;

}  // namespace clifft::sampling
