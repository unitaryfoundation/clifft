#pragma once

#include "clifft/sampling/kernels.h"

namespace clifft::sampling {

void apply_new_x_instrument_no_fire_avx2(State& state, double factor_zero, double factor_one,
                                         double no_fire_probability) noexcept;

}  // namespace clifft::sampling
