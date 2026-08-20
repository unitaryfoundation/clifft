#include "clifft/sampling/kernel_dispatch.h"

#include "clifft/sampling/simd_width.h"

#include <stdexcept>

namespace clifft::sampling {

namespace {

constexpr uint32_t kMinProfitableAvx2MeasurementWidth = kAvx2LaneIndexBits;
constexpr uint32_t kMinProfitableAvx512MeasurementWidth = 4;

ActiveMeasurementKernel select_active_measurement(const PreparedMeasurement& measurement,
                                                  uint64_t vector_lanes, uint32_t lane_index_bits,
                                                  uint32_t min_active_width) noexcept {
    if (measurement.pauli.is_diagonal() || measurement.pauli.active_width < min_active_width) {
        return ActiveMeasurementKernel::Scalar;
    }
    const uint64_t pivot_bit = uint64_t{1} << measurement.pivot;
    if (measurement.pauli.pairing_bit == pivot_bit && pivot_bit >= vector_lanes) {
        return ActiveMeasurementKernel::HighPivot;
    }
    if (measurement.pauli.x < vector_lanes && measurement.pivot < lane_index_bits) {
        return ActiveMeasurementKernel::LanePaired;
    }
    return ActiveMeasurementKernel::Scalar;
}

constexpr uint64_t kNoExcludedPairingBit = 0;
constexpr uint64_t kPivotFourPairingBit = uint64_t{1} << 4;

DirectRotationKernel select_direct_rotation(const PreparedRotation& rotation, uint64_t vector_lanes,
                                            uint32_t min_active_width,
                                            uint64_t excluded_pairing_bit) noexcept {
    if (rotation.pauli.is_identity()) {
        return DirectRotationKernel::Scalar;
    }
    if (rotation.pauli.is_diagonal()) {
        return rotation.pauli.active_width >= min_active_width ? DirectRotationKernel::Diagonal
                                                               : DirectRotationKernel::Scalar;
    }
    const uint64_t pairing_bit = rotation.pauli.pairing_bit;
    if (pairing_bit < vector_lanes) {
        return rotation.pauli.active_width >= min_active_width ? DirectRotationKernel::LanePaired
                                                               : DirectRotationKernel::Scalar;
    }
    if (pairing_bit == excluded_pairing_bit) {
        return DirectRotationKernel::Scalar;
    }
    return DirectRotationKernel::HighPivot;
}

// One four-coefficient block was neutral in the direct microbenchmark, while
// every wider measured width won; smaller states stay on the scalar path.
constexpr uint32_t kMinProfitableAvx2InstrumentWidth = kAvx2LaneIndexBits;

}  // namespace

ExecutorBackend resolve_executor_backend(internal::RuntimeIsa runtime_isa) {
    internal::validate_runtime_isa(runtime_isa);
    switch (runtime_isa) {
        case internal::RuntimeIsa::Scalar:
            return ExecutorBackend::Scalar;
        case internal::RuntimeIsa::Avx2:
            return ExecutorBackend::Avx2;
        case internal::RuntimeIsa::Avx512:
            return ExecutorBackend::Avx512;
        case internal::RuntimeIsa::TrapAvx2:
        case internal::RuntimeIsa::TrapAvx512:
        case internal::RuntimeIsa::TrapUnknown:
            break;
    }
    throw std::invalid_argument("unrecognized sampling executor backend");
}

DirectRotationKernel resolve_direct_rotation_kernel(const PreparedRotation& rotation,
                                                    ExecutorBackend backend) noexcept {
    switch (backend) {
        case ExecutorBackend::Scalar:
            return DirectRotationKernel::Scalar;
        case ExecutorBackend::Avx2:
            // Stride-16 pairing was neutral or faster than scalar across the
            // measured AVX2 widths, so every high pivot uses the vector kernel.
            return select_direct_rotation(rotation, kAvx2DoubleLanes, kAvx2LaneIndexBits,
                                          kNoExcludedPairingBit);
        case ExecutorBackend::Avx512:
            // Stride-16 pairing regressed against scalar at every measured
            // width on the AVX-512 performance host.
            return select_direct_rotation(rotation, kAvx512DoubleLanes, kAvx512LaneIndexBits,
                                          kPivotFourPairingBit);
    }
    return DirectRotationKernel::Scalar;
}

ActiveMeasurementKernel resolve_active_measurement_kernel(const PreparedMeasurement& measurement,
                                                          ExecutorBackend backend) noexcept {
    switch (backend) {
        case ExecutorBackend::Scalar:
            return ActiveMeasurementKernel::Scalar;
        case ExecutorBackend::Avx2:
            // The probability-plus-collapse pair wins from one AVX2 block onward.
            return select_active_measurement(measurement, kAvx2DoubleLanes, kAvx2LaneIndexBits,
                                             kMinProfitableAvx2MeasurementWidth);
        case ExecutorBackend::Avx512:
            // A single AVX-512 block regressed against scalar on the performance host.
            return select_active_measurement(measurement, kAvx512DoubleLanes, kAvx512LaneIndexBits,
                                             kMinProfitableAvx512MeasurementWidth);
    }
    return ActiveMeasurementKernel::Scalar;
}

NewXInstrumentKernel resolve_new_x_instrument_kernel(uint32_t active_width,
                                                     ExecutorBackend backend) noexcept {
    // The AVX-512 backend can use the AVX2 implementation because its required
    // AVX2, BMI2, and FMA features are a subset of that backend's requirements.
    if (backend != ExecutorBackend::Scalar && active_width >= kMinProfitableAvx2InstrumentWidth) {
        return NewXInstrumentKernel::Vectorized;
    }
    return NewXInstrumentKernel::Scalar;
}

}  // namespace clifft::sampling
