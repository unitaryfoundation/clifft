#include "clifft/sampling/pauli_preparation.h"

#include "clifft/util/numeric.h"

#include <bit>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <string>

namespace clifft::sampling {

namespace {

uint64_t width_mask(uint32_t active_width) {
    if (active_width >= kDenseActiveWidthLimit) {
        throw std::invalid_argument("prepared Pauli active width must be below " +
                                    std::to_string(kDenseActiveWidthLimit));
    }
    return active_width == 0 ? 0 : (uint64_t{1} << active_width) - 1;
}

}  // namespace

PreparedPauli prepare_pauli(ActivePauli pauli, uint32_t active_width) {
    const uint64_t valid_bits = width_mask(active_width);
    if ((pauli.x & ~valid_bits) != 0 || (pauli.z & ~valid_bits) != 0) {
        throw std::invalid_argument("prepared Pauli has bits outside its active width");
    }

    const uint32_t overlap = std::popcount(pauli.x & pauli.z);
    return PreparedPauli{.active_width = active_width,
                         .x = pauli.x,
                         .z = pauli.z,
                         .pairing_bit = std::bit_floor(pauli.x),
                         .even_phase = i_power(overlap)};
}

PreparedRotation prepare_rotation(ActivePauli pauli, uint32_t active_width, double half_turns) {
    if (!is_finite_robust(half_turns)) {
        throw std::invalid_argument("prepared rotation angle must be finite");
    }
    if (pauli.is_identity()) {
        throw std::invalid_argument("cannot prepare an identity rotation");
    }
    const double angle = std::numbers::pi * reduce_phase_half_turns(half_turns) / 2.0;
    return PreparedRotation{prepare_pauli(pauli, active_width), std::cos(angle), std::sin(angle)};
}

PreparedPromotion prepare_promotion(double half_turns) {
    if (!is_finite_robust(half_turns)) {
        throw std::invalid_argument("prepared promotion angle must be finite");
    }
    const double angle = std::numbers::pi * reduce_phase_half_turns(half_turns) / 2.0;
    return PreparedPromotion{std::cos(angle), std::sin(angle)};
}

PreparedMeasurement prepare_measurement(ActivePauli pauli, uint32_t active_width, uint32_t pivot) {
    PreparedPauli prepared = prepare_pauli(pauli, active_width);
    if (active_width == 0 || prepared.is_identity()) {
        throw std::invalid_argument("cannot prepare an active identity measurement");
    }
    if (pivot >= active_width) {
        throw std::invalid_argument("prepared measurement pivot is outside its active width");
    }
    const uint64_t pivot_bit = uint64_t{1} << pivot;
    const bool pivot_is_valid =
        prepared.x != 0 ? (prepared.x & pivot_bit) != 0 : (prepared.z & pivot_bit) != 0;
    if (!pivot_is_valid) {
        throw std::invalid_argument("prepared measurement pivot is outside Pauli support");
    }
    return PreparedMeasurement{prepared, pivot, uint64_t{1} << (active_width - 1),
                               prepared.z & ~pivot_bit};
}

}  // namespace clifft::sampling
