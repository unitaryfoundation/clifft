#pragma once

#include "clifft/sampling/fused_rotation.h"
#include "clifft/sampling/kernels.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>
#include <cstdint>
#include <memory>

namespace clifft::sampling {

// The executable plan records one process-selected backend. Its action tags
// describe only the operation shape, such as a diagonal rotation or amplitude
// pairs within one vector, so that backend can call its matching kernel. The
// tags do not encode or rediscover the process ISA.
enum class ExecutorBackend : uint8_t {
    Scalar,
    Neon,
    Avx2,
    Avx512,
};

[[nodiscard]] ExecutorBackend resolve_executor_backend(internal::RuntimeIsa runtime_isa);

enum class DirectRotationKernel : uint8_t {
    Scalar,
    Diagonal,
    HighPivot,
    LanePaired,
};

[[nodiscard]] DirectRotationKernel resolve_direct_rotation_kernel(const PreparedRotation& rotation,
                                                                  ExecutorBackend backend) noexcept;

enum class ActiveMeasurementKernel : uint8_t {
    Scalar,
    Diagonal,
    HighPivot,
    LanePaired,
};

[[nodiscard]] ActiveMeasurementKernel resolve_active_measurement_kernel(
    const PreparedMeasurement& measurement, ExecutorBackend backend) noexcept;

using FusedRotationKernel = void (*)(State&, const PreparedFusedRotation&, const void*) noexcept;
using FusedRotationParallelKernel = void (*)(State&, const PreparedFusedRotation&, const void*,
                                             uint32_t, uint32_t) noexcept;

// Type-erases optional backend-specific preparation without exposing vector
// types to the portable executable plan.
struct FusedRotationSidecar {
    std::shared_ptr<const void> storage;
    FusedRotationKernel kernel = nullptr;
    FusedRotationParallelKernel parallel_kernel = nullptr;
};

enum class NewXInstrumentKernel : uint8_t {
    Scalar,
    Vectorized,
};

[[nodiscard]] NewXInstrumentKernel resolve_new_x_instrument_kernel(
    uint32_t active_width, ExecutorBackend backend) noexcept;

// Entry points implemented in architecture-specific translation units. Only
// a matching backend-specialized executor calls them.
void apply_direct_rotation_neon(State& state, const PreparedRotation& rotation,
                                DirectRotationKernel kernel, bool sign) noexcept;
void apply_direct_rotation_avx2(State& state, const PreparedRotation& rotation,
                                DirectRotationKernel kernel, bool sign) noexcept;
void apply_direct_rotation_avx512(State& state, const PreparedRotation& rotation,
                                  DirectRotationKernel kernel, bool sign) noexcept;
void apply_direct_rotation_avx2_parallel(State& state, const PreparedRotation& rotation,
                                         DirectRotationKernel kernel, bool sign, uint32_t workers,
                                         uint32_t min_active_width) noexcept;
void apply_direct_rotation_avx512_parallel(State& state, const PreparedRotation& rotation,
                                           DirectRotationKernel kernel, bool sign, uint32_t workers,
                                           uint32_t min_active_width) noexcept;
void apply_direct_rotation_neon_parallel(State& state, const PreparedRotation& rotation,
                                         DirectRotationKernel kernel, bool sign, uint32_t workers,
                                         uint32_t min_active_width) noexcept;

[[nodiscard]] MeasurementProbabilities active_measurement_probabilities_avx2(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept;
void collapse_active_measurement_avx2(State& state, const PreparedMeasurement& measurement,
                                      ActiveMeasurementKernel kernel, bool branch,
                                      double branch_probability) noexcept;
[[nodiscard]] MeasurementProbabilities active_measurement_probabilities_avx512(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept;
void collapse_active_measurement_avx512(State& state, const PreparedMeasurement& measurement,
                                        ActiveMeasurementKernel kernel, bool branch,
                                        double branch_probability) noexcept;
[[nodiscard]] MeasurementProbabilities active_measurement_probabilities_neon(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept;
void collapse_active_measurement_neon(State& state, const PreparedMeasurement& measurement,
                                      ActiveMeasurementKernel kernel, bool branch,
                                      double branch_probability) noexcept;

[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx2_sidecar(
    const PreparedFusedRotation& rotation);
[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx512_sidecar(
    const PreparedFusedRotation& rotation);
[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_neon_sidecar(
    const PreparedFusedRotation& rotation);

void apply_new_x_instrument_no_fire_avx2(State& state, double factor_zero, double factor_one,
                                         double no_fire_probability) noexcept;

// Compile-time backend dispatch shared by ordinary and trajectory-sharing executors.
template <ExecutorBackend Backend, bool Parallel = false>
void apply_direct_rotation(State& state, const PreparedRotation& rotation,
                           DirectRotationKernel kernel, bool sign, uint32_t workers = 1,
                           uint32_t min_active_width = kDefaultIntraShotMinActiveWidth) noexcept {
    if (kernel == DirectRotationKernel::Scalar) {
        if constexpr (Parallel) {
            apply_rotation_parallel(state, rotation, sign, workers, min_active_width);
        } else {
            apply_rotation(state, rotation, sign);
        }
        return;
    }
    if constexpr (Backend == ExecutorBackend::Neon) {
        if constexpr (Parallel) {
            apply_direct_rotation_neon_parallel(state, rotation, kernel, sign, workers,
                                                min_active_width);
        } else {
            apply_direct_rotation_neon(state, rotation, kernel, sign);
        }
    } else if constexpr (Backend == ExecutorBackend::Avx2) {
        if constexpr (Parallel) {
            apply_direct_rotation_avx2_parallel(state, rotation, kernel, sign, workers,
                                                min_active_width);
        } else {
            apply_direct_rotation_avx2(state, rotation, kernel, sign);
        }
    } else if constexpr (Backend == ExecutorBackend::Avx512) {
        if constexpr (Parallel) {
            apply_direct_rotation_avx512_parallel(state, rotation, kernel, sign, workers,
                                                  min_active_width);
        } else {
            apply_direct_rotation_avx512(state, rotation, kernel, sign);
        }
    } else {
        assert(false && "scalar executor requires scalar direct-rotation actions");
        apply_rotation(state, rotation, sign);
    }
}

template <ExecutorBackend Backend>
[[nodiscard]] MeasurementProbabilities active_measurement_probabilities(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept {
    if (kernel == ActiveMeasurementKernel::Scalar) {
        return measurement_probabilities(state, measurement);
    }
    if constexpr (Backend == ExecutorBackend::Neon) {
        return active_measurement_probabilities_neon(state, measurement, kernel);
    } else if constexpr (Backend == ExecutorBackend::Avx2) {
        return active_measurement_probabilities_avx2(state, measurement, kernel);
    } else if constexpr (Backend == ExecutorBackend::Avx512) {
        return active_measurement_probabilities_avx512(state, measurement, kernel);
    } else {
        assert(false && "scalar executor requires scalar active-measurement actions");
        return measurement_probabilities(state, measurement);
    }
}

template <ExecutorBackend Backend>
void collapse_active_measurement(State& state, const PreparedMeasurement& measurement,
                                 ActiveMeasurementKernel kernel, bool branch,
                                 double branch_probability) noexcept {
    if (kernel == ActiveMeasurementKernel::Scalar) {
        collapse_measurement(state, measurement, branch, branch_probability);
    } else if constexpr (Backend == ExecutorBackend::Neon) {
        collapse_active_measurement_neon(state, measurement, kernel, branch, branch_probability);
    } else if constexpr (Backend == ExecutorBackend::Avx2) {
        collapse_active_measurement_avx2(state, measurement, kernel, branch, branch_probability);
    } else if constexpr (Backend == ExecutorBackend::Avx512) {
        collapse_active_measurement_avx512(state, measurement, kernel, branch, branch_probability);
    } else {
        assert(false && "scalar executor requires scalar active-measurement actions");
        collapse_measurement(state, measurement, branch, branch_probability);
    }
}

}  // namespace clifft::sampling
