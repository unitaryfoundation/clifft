#pragma once

#include <complex>
#include <numbers>

namespace clifft {

inline constexpr double kInvSqrt2 = std::numbers::sqrt2 / 2.0;
inline constexpr std::complex<double> kI{0.0, 1.0};
inline constexpr std::complex<double> kMinusI{0.0, -1.0};
inline constexpr std::complex<double> kExpIPiOver4{kInvSqrt2, kInvSqrt2};        // e^{i*pi/4}
inline constexpr std::complex<double> kExpMinusIPiOver4{kInvSqrt2, -kInvSqrt2};  // e^{-i*pi/4}

}  // namespace clifft
