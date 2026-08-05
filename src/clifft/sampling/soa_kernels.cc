#include "clifft/sampling/soa_kernels.h"

#include "clifft/util/numeric.h"

#include <bit>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <string>

namespace clifft::sampling {

namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;

uint64_t width_mask(uint32_t active_width) {
    if (active_width >= kDenseActiveWidthLimit) {
        throw std::invalid_argument("prepared Pauli active width must be below " +
                                    std::to_string(kDenseActiveWidthLimit));
    }
    return active_width == 0 ? 0 : (uint64_t{1} << active_width) - 1;
}

void validate_descriptor_width(const SoaState& state, const PreparedPauli& pauli) {
    if (state.active_width() != pauli.active_width) {
        throw std::invalid_argument("prepared Pauli width does not match the active state");
    }
}

std::complex<double> phase_at(const PreparedPauli& pauli, uint64_t basis) {
    return (std::popcount(basis & pauli.z) & 1U) != 0 ? -pauli.even_phase : pauli.even_phase;
}

uint64_t insert_zero_bit(uint64_t packed, uint32_t pivot) {
    const uint64_t lower_mask = (uint64_t{1} << pivot) - 1;
    return (packed & lower_mask) | ((packed & ~lower_mask) << 1);
}

uint64_t diagonal_source(const PreparedMeasurement& measurement, uint64_t packed, bool branch) {
    const uint64_t without_pivot = insert_zero_bit(packed, measurement.pivot);
    const bool other_parity =
        (std::popcount(without_pivot & measurement.z_without_pivot) & 1U) != 0;
    const bool pivot_value = branch != other_parity;
    return without_pivot | (static_cast<uint64_t>(pivot_value) << measurement.pivot);
}

std::complex<double> load(const SoaState& state, uint64_t index) {
    return {state.real_data()[index], state.imag_data()[index]};
}

void store(SoaState& state, uint64_t index, std::complex<double> value) {
    state.real_data()[index] = value.real();
    state.imag_data()[index] = value.imag();
}

std::complex<double> compact_nondiagonal(const SoaState& state,
                                         const PreparedMeasurement& measurement, uint64_t packed,
                                         bool branch) {
    const uint64_t source0 = insert_zero_bit(packed, measurement.pivot);
    const uint64_t source1 = source0 ^ measurement.pauli.x;
    const double eigenvalue = branch ? -1.0 : 1.0;
    const std::complex<double> coefficient1 =
        eigenvalue * std::conj(phase_at(measurement.pauli, source0));
    return kInvSqrt2 * (load(state, source0) + coefficient1 * load(state, source1));
}

}  // namespace

PreparedPauli prepare_pauli(ActivePauli pauli, uint32_t active_width) {
    const uint64_t valid_bits = width_mask(active_width);
    if ((pauli.x & ~valid_bits) != 0 || (pauli.z & ~valid_bits) != 0) {
        throw std::invalid_argument("prepared Pauli has bits outside its active width");
    }

    static constexpr std::complex<double> kIPowers[4] = {
        {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
    const uint32_t overlap = std::popcount(pauli.x & pauli.z);
    return PreparedPauli{active_width, pauli.x, pauli.z,
                         pauli.x == 0 ? 0 : pauli.x & (~pauli.x + 1), kIPowers[overlap & 3U]};
}

PreparedRotation prepare_rotation(ActivePauli pauli, uint32_t active_width, double half_turns) {
    if (!is_finite_robust(half_turns)) {
        throw std::invalid_argument("prepared rotation angle must be finite");
    }
    const double angle = std::numbers::pi * half_turns / 2.0;
    return PreparedRotation{prepare_pauli(pauli, active_width), std::cos(angle), std::sin(angle)};
}

PreparedPromotion prepare_promotion(double half_turns) {
    if (!is_finite_robust(half_turns)) {
        throw std::invalid_argument("prepared promotion angle must be finite");
    }
    const double angle = std::numbers::pi * half_turns / 2.0;
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

void apply_rotation(SoaState& state, const PreparedRotation& rotation, bool sign) {
    validate_descriptor_width(state, rotation.pauli);
    const double sine = sign ? -rotation.sine : rotation.sine;
    if (rotation.pauli.is_identity()) {
        state.multiply_global_scalar({rotation.cosine, -sine});
        return;
    }

    double* real = state.real_data();
    double* imag = state.imag_data();
    const uint64_t size = state.size();
    if (rotation.pauli.is_diagonal()) {
        for (uint64_t basis = 0; basis < size; ++basis) {
            const double eigenvalue =
                (std::popcount(basis & rotation.pauli.z) & 1U) != 0 ? -1.0 : 1.0;
            const double r = real[basis];
            const double i = imag[basis];
            const double signed_sine = sine * eigenvalue;
            real[basis] = rotation.cosine * r + signed_sine * i;
            imag[basis] = rotation.cosine * i - signed_sine * r;
        }
        return;
    }

    const std::complex<double> minus_i_sine{0.0, -sine};
    for (uint64_t left = 0; left < size; ++left) {
        if ((left & rotation.pauli.pair_selector) != 0) {
            continue;
        }
        const uint64_t right = left ^ rotation.pauli.x;
        const std::complex<double> left_value{real[left], imag[left]};
        const std::complex<double> right_value{real[right], imag[right]};
        const std::complex<double> left_phase = phase_at(rotation.pauli, left);
        const std::complex<double> right_phase = phase_at(rotation.pauli, right);
        store(state, left, rotation.cosine * left_value + minus_i_sine * right_phase * right_value);
        store(state, right, rotation.cosine * right_value + minus_i_sine * left_phase * left_value);
    }
}

void apply_promotion(SoaState& state, const PreparedPromotion& promotion, bool sign) {
    if (state.active_width() >= state.max_active_width()) {
        throw std::out_of_range("promotion exceeds the SoA state maximum active width");
    }
    const double sine = sign ? -promotion.sine : promotion.sine;
    double* real = state.real_data();
    double* imag = state.imag_data();
    const uint64_t old_size = state.size();
    for (uint64_t basis = 0; basis < old_size; ++basis) {
        const double r = real[basis];
        const double i = imag[basis];
        real[basis] = promotion.cosine * r;
        imag[basis] = promotion.cosine * i;
        real[old_size + basis] = sine * i;
        imag[old_size + basis] = -sine * r;
    }
    state.set_active_width(state.active_width() + 1);
}

MeasurementProbabilities measurement_probabilities(const SoaState& state,
                                                   const PreparedMeasurement& measurement) {
    validate_descriptor_width(state, measurement.pauli);
    MeasurementProbabilities result;
    if (measurement.pauli.is_diagonal()) {
        for (uint64_t packed = 0; packed < measurement.output_size; ++packed) {
            const uint64_t source0 = diagonal_source(measurement, packed, false);
            const uint64_t source1 = diagonal_source(measurement, packed, true);
            result.zero += std::norm(load(state, source0));
            result.one += std::norm(load(state, source1));
        }
        return result;
    }

    for (uint64_t packed = 0; packed < measurement.output_size; ++packed) {
        result.zero += std::norm(compact_nondiagonal(state, measurement, packed, false));
        result.one += std::norm(compact_nondiagonal(state, measurement, packed, true));
    }
    return result;
}

void collapse_measurement(SoaState& state, const PreparedMeasurement& measurement, bool branch,
                          double branch_probability) {
    validate_descriptor_width(state, measurement.pauli);
    if (!is_finite_robust(branch_probability) || branch_probability <= 0.0) {
        throw std::invalid_argument("measurement collapse requires a positive finite probability");
    }
    const double inv_norm = 1.0 / std::sqrt(branch_probability);
    if (measurement.pauli.is_diagonal()) {
        // Each selected source is at or above its packed destination, so a
        // forward compaction cannot overwrite a source needed later.
        for (uint64_t packed = 0; packed < measurement.output_size; ++packed) {
            const uint64_t source = diagonal_source(measurement, packed, branch);
            store(state, packed, inv_norm * load(state, source));
        }
    } else {
        // Pauli pairings can cross earlier packed destinations. Write every
        // projected pair to preallocated scratch before replacing the prefix.
        double* scratch_real = state.scratch_real_data();
        double* scratch_imag = state.scratch_imag_data();
        for (uint64_t packed = 0; packed < measurement.output_size; ++packed) {
            const std::complex<double> value =
                inv_norm * compact_nondiagonal(state, measurement, packed, branch);
            scratch_real[packed] = value.real();
            scratch_imag[packed] = value.imag();
        }
        for (uint64_t packed = 0; packed < measurement.output_size; ++packed) {
            state.real_data()[packed] = scratch_real[packed];
            state.imag_data()[packed] = scratch_imag[packed];
        }
    }
    state.set_active_width(state.active_width() - 1);
}

}  // namespace clifft::sampling
