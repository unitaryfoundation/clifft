#pragma once

// Private support analysis for experimental accelerator backends. Backends
// consume SamplingPlan semantics directly and own their prepared forms; this
// file does not define a public backend ABI or a shared device command stream.

#include "clifft/sampling/plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace clifft::accelerator {

// Requirements that an execution backend can accept or reject before it
// prepares storage. Action requirements correspond to SamplingAction
// alternatives; the final two capture plan-level behavior that is not an
// independent action.
enum class PlanFeature : uint8_t {
    RotateActivePauli,
    PromoteDormantRotation,
    MeasureActivePauli,
    MeasureDormantRandom,
    RecordClassical,
    DefineSymbol,
    ApplyReadoutNoise,
    WriteDetector,
    WriteObservable,
    WriteExpectationValue,
    ApplyInstrument,
    InstrumentBoundary,
    PresampledNoise,
    Postselection,
    Count,
};

inline constexpr size_t kNumPlanFeatures = static_cast<size_t>(PlanFeature::Count);

[[nodiscard]] std::string_view plan_feature_name(PlanFeature feature);
[[nodiscard]] PlanFeature plan_feature(const sampling::SamplingAction& action);

struct PlanRequirements {
    uint32_t peak_active_width = 0;
    std::array<size_t, kNumPlanFeatures> occurrences{};

    [[nodiscard]] size_t count(PlanFeature feature) const;
    [[nodiscard]] bool uses(PlanFeature feature) const;
};

// Validates and summarizes the semantic work a backend must prepare. This is
// construction-time analysis, not part of hot execution.
[[nodiscard]] PlanRequirements analyze_plan_requirements(const sampling::SamplingPlan& plan);

}  // namespace clifft::accelerator
