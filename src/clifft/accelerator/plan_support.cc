#include "clifft/accelerator/plan_support.h"

#include <stdexcept>

namespace clifft::accelerator {

namespace {

static_assert(kNumPlanFeatures == std::variant_size_v<sampling::SamplingAction> + 2);

constexpr std::array<std::string_view, kNumPlanFeatures> kPlanFeatureNames = {
    "rotate_active_pauli",  "promote_dormant_rotation",
    "measure_active_pauli", "measure_dormant_random",
    "record_classical",     "define_symbol",
    "apply_readout_noise",  "write_detector",
    "write_observable",     "write_expectation_value",
    "apply_instrument",     "instrument_boundary",
    "presampled_noise",     "postselection",
};

size_t feature_index(PlanFeature feature) {
    const size_t result = static_cast<size_t>(feature);
    if (result >= kNumPlanFeatures) {
        throw std::invalid_argument("invalid accelerator plan feature");
    }
    return result;
}

}  // namespace

std::string_view plan_feature_name(PlanFeature feature) {
    return kPlanFeatureNames[feature_index(feature)];
}

PlanFeature plan_feature(const sampling::SamplingAction& action) {
    // The action features intentionally mirror SamplingAction's alternative
    // order. The mapping test catches accidental reordering, while the static
    // assertion above catches additions that omit a feature.
    return static_cast<PlanFeature>(action.index());
}

size_t PlanRequirements::count(PlanFeature feature) const {
    return occurrences[feature_index(feature)];
}

bool PlanRequirements::uses(PlanFeature feature) const {
    return count(feature) != 0;
}

PlanRequirements analyze_plan_requirements(const sampling::SamplingPlan& plan) {
    plan.validate();

    PlanRequirements requirements;
    requirements.peak_active_width = plan.peak_active_width;
    for (const sampling::PlannedAction& action : plan.actions) {
        ++requirements.occurrences[feature_index(plan_feature(action.action))];
        if (const auto* detector = std::get_if<sampling::WriteDetector>(&action.action);
            detector != nullptr && detector->postselected) {
            ++requirements.occurrences[feature_index(PlanFeature::Postselection)];
        }
    }
    requirements.occurrences[feature_index(PlanFeature::PresampledNoise)] =
        plan.presampled_noise_sites.size();
    return requirements;
}

}  // namespace clifft::accelerator
