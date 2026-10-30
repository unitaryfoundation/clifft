#pragma once

// Phases of single-copy parametric ZX diagrams.
//
// A phase is (value + parity(params)) * pi mod 2pi. The parameters are Boolean
// variables (measurement outcomes and noise bits), so their parity adds pi.
// This follows zxparam's Phase (https://pypi.org/project/zxparam/) and quizx's
// per-vertex Parity (https://github.com/zxcalc/quizx).
//
// The concrete value is a double in half-turns, reduced to [0, 2) and snapped
// to the pi/4 grid. Clifford+T phases therefore stay exact, so the Pauli and
// Clifford tests that select rewrites agree with zxparam's exact rationals.

#include "clifft/util/numeric.h"

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::zx {

// A set of Boolean parameters, combined by XOR. Kept sorted and duplicate-free.
class Params {
  public:
    Params() = default;
    explicit Params(std::vector<uint32_t> ids);

    [[nodiscard]] bool empty() const { return ids_.empty(); }
    [[nodiscard]] std::span<const uint32_t> ids() const { return ids_; }

    void xor_with(const Params& other);

    [[nodiscard]] bool operator==(const Params&) const = default;
    [[nodiscard]] auto operator<=>(const Params&) const = default;

  private:
    std::vector<uint32_t> ids_;
};

// Reduce a half-turn value to [0, 2) and snap it to the pi/4 grid when close.
[[nodiscard]] inline double normalize_half_turns(double value) {
    value = std::fmod(value, 2.0);
    if (value < 0.0) {
        value += 2.0;
    }
    const double quarters = std::round(value * 4.0);
    if (std::abs(value * 4.0 - quarters) < 4.0 * kRotationCanonicalizationTolerance) {
        value = quarters / 4.0;
    }
    if (value >= 2.0) {
        value = 0.0;
    }
    return value;
}

struct Phase {
    double value = 0.0;
    Params params;

    Phase() = default;
    explicit Phase(double half_turns, Params p = {})
        : value(normalize_half_turns(half_turns)), params(std::move(p)) {}

    [[nodiscard]] bool is_zero() const { return value == 0.0 && params.empty(); }
    [[nodiscard]] bool is_pi() const { return value == 1.0; }
    // Pauli up to parameters: the concrete part is 0 or pi.
    [[nodiscard]] bool is_pauli() const { return value == 0.0 || value == 1.0; }
    // A concrete part of +-pi/2, the phases local complementation removes.
    [[nodiscard]] bool is_proper_clifford() const { return value == 0.5 || value == 1.5; }
    // On the pi/4 grid, the case where printing and comparison are exact.
    [[nodiscard]] bool is_dyadic() const { return value * 4.0 == std::round(value * 4.0); }

    [[nodiscard]] Phase plus(const Phase& other) const {
        Phase sum(value + other.value, params);
        sum.params.xor_with(other.params);
        return sum;
    }
    [[nodiscard]] Phase negated() const { return Phase(-value, params); }
    [[nodiscard]] Phase minus(const Phase& other) const { return plus(other.negated()); }
    [[nodiscard]] Phase plus_pi() const { return Phase(value + 1.0, params); }

    [[nodiscard]] bool operator==(const Phase&) const = default;
};

}  // namespace clifft::zx
