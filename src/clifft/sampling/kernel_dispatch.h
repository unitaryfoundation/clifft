#pragma once

#include "clifft/sampling/fused_rotation.h"
#include "clifft/sampling/kernels.h"

#include <cstdint>
#include <memory>

namespace clifft::internal {
enum class RuntimeIsa;
}

namespace clifft::sampling {

// Compact, architecture-neutral choices stored in executable actions. Lowering
// resolves them once; execution uses the scalar implementation as the portable
// fallback.

// Direct rotations.
enum class DirectRotationKernel : uint8_t {
    Scalar,
    Diagonal,
    HighPivot,
    LanePaired,
};

static_assert(sizeof(DirectRotationKernel) == 1);

[[nodiscard]] DirectRotationKernel resolve_direct_rotation_kernel(
    const PreparedRotation& rotation, internal::RuntimeIsa runtime_isa) noexcept;
void apply_direct_rotation(State& state, const PreparedRotation& rotation,
                           DirectRotationKernel kernel, bool sign) noexcept;

// Active measurements.
enum class ActiveMeasurementKernel : uint8_t {
    Scalar,
    LanePaired,
};

static_assert(sizeof(ActiveMeasurementKernel) == 1);

[[nodiscard]] ActiveMeasurementKernel resolve_active_measurement_kernel(
    const PreparedMeasurement& measurement, internal::RuntimeIsa runtime_isa) noexcept;
[[nodiscard]] MeasurementProbabilities active_measurement_probabilities(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept;
void collapse_active_measurement(State& state, const PreparedMeasurement& measurement,
                                 ActiveMeasurementKernel kernel, bool branch,
                                 double branch_probability) noexcept;

// Fused rotations.
using FusedRotationKernel = void (*)(State&, const PreparedFusedRotation&, const void*) noexcept;

// Type-erases optional host-specific preparation without exposing vector types
// to the portable executable plan.
struct FusedRotationSidecar {
    std::shared_ptr<const void> storage;
    FusedRotationKernel kernel = nullptr;

    [[nodiscard]] explicit operator bool() const noexcept {
        return storage != nullptr && kernel != nullptr;
    }
};

// Owns the portable fused descriptor and any optional host-specific
// preparation selected for it. Construction happens before hot execution.
class PreparedFusedRotationExecution {
  public:
    PreparedFusedRotationExecution(PreparedFusedRotation rotation,
                                   internal::RuntimeIsa runtime_isa);

    void apply(State& state) const noexcept {
        if (sidecar_) {
            sidecar_.kernel(state, rotation_, sidecar_.storage.get());
        } else {
            apply_fused_rotation(state, rotation_);
        }
    }

  private:
    PreparedFusedRotation rotation_;
    FusedRotationSidecar sidecar_;
};

// New-X instrument activation.
enum class NewXInstrumentKernel : uint8_t {
    Scalar,
    Avx2,
};

static_assert(sizeof(NewXInstrumentKernel) == 1);

[[nodiscard]] NewXInstrumentKernel resolve_new_x_instrument_kernel(
    uint32_t active_width, internal::RuntimeIsa runtime_isa) noexcept;
void apply_new_x_instrument_no_fire_dispatched(State& state, double factor_zero, double factor_one,
                                               double no_fire_probability,
                                               NewXInstrumentKernel kernel) noexcept;

}  // namespace clifft::sampling
