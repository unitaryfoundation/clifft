#include "clifft/accelerator/plan_support.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <utility>

using clifft::accelerator::analyze_plan_requirements;
using clifft::accelerator::plan_feature;
using clifft::accelerator::PlanFeature;
using clifft::accelerator::PlanRequirements;
using clifft::sampling::ActivePauli;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::ApplyReadoutNoise;
using clifft::sampling::DefineSymbol;
using clifft::sampling::DetectorSlot;
using clifft::sampling::ExpValSlot;
using clifft::sampling::InstrumentBoundary;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::PlannedAction;
using clifft::sampling::PresampledNoiseOutcome;
using clifft::sampling::PresampledNoiseSite;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingAction;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolInfo;
using clifft::sampling::SymbolKind;
using clifft::sampling::WriteDetector;
using clifft::sampling::WriteExpectationValue;
using clifft::sampling::WriteObservable;

namespace {

SamplingPlan supported_feature_plan() {
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.peak_active_width = 1;
    plan.num_detectors = 1;
    plan.num_exp_vals = 1;
    plan.has_postselection = true;
    plan.actions = {
        PlannedAction{0, 1, PromoteDormantRotation{0.25, {}}},
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{1, 0}, 0.25, {}}},
        PlannedAction{1, 1, WriteExpectationValue{ActivePauli{1, 0}, {}, ExpValSlot{0}}},
        PlannedAction{1, 1, WriteDetector{{}, DetectorSlot{0}, true}},
    };
    return plan;
}

SamplingPlan presampled_noise_plan() {
    const SymbolId noise{0};
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.num_noise_sites = 1;
    plan.symbols = {
        SymbolInfo{SymbolKind::Presampled, std::nullopt, clifft::sampling::NoiseSiteId{0}}};
    plan.presampled_noise_sites = {PresampledNoiseSite{
        clifft::sampling::NoiseSiteId{0}, 0.125, {PresampledNoiseOutcome{noise, 0.125}}}};
    return plan;
}

}  // namespace

TEST_CASE("Accelerator support maps every sampling action") {
    const std::array<std::pair<SamplingAction, PlanFeature>, 12> cases = {{
        {RotateActivePauli{}, PlanFeature::RotateActivePauli},
        {PromoteDormantRotation{}, PlanFeature::PromoteDormantRotation},
        {MeasureActivePauli{}, PlanFeature::MeasureActivePauli},
        {MeasureDormantRandom{}, PlanFeature::MeasureDormantRandom},
        {RecordClassical{}, PlanFeature::RecordClassical},
        {DefineSymbol{}, PlanFeature::DefineSymbol},
        {ApplyReadoutNoise{}, PlanFeature::ApplyReadoutNoise},
        {WriteDetector{}, PlanFeature::WriteDetector},
        {WriteObservable{}, PlanFeature::WriteObservable},
        {WriteExpectationValue{}, PlanFeature::WriteExpectationValue},
        {ApplyInstrument{}, PlanFeature::ApplyInstrument},
        {InstrumentBoundary{}, PlanFeature::InstrumentBoundary},
    }};

    for (const auto& [action, expected] : cases) {
        REQUIRE(plan_feature(action) == expected);
    }
}

TEST_CASE("Accelerator support summarizes validated plan requirements") {
    const PlanRequirements requirements = analyze_plan_requirements(supported_feature_plan());

    REQUIRE(requirements.peak_active_width == 1);
    REQUIRE(requirements.count(PlanFeature::PromoteDormantRotation) == 1);
    REQUIRE(requirements.count(PlanFeature::RotateActivePauli) == 1);
    REQUIRE(requirements.count(PlanFeature::WriteExpectationValue) == 1);
    REQUIRE(requirements.count(PlanFeature::WriteDetector) == 1);
    REQUIRE(requirements.count(PlanFeature::Postselection) == 1);
    REQUIRE_FALSE(requirements.uses(PlanFeature::PresampledNoise));

    const PlanRequirements noise_requirements = analyze_plan_requirements(presampled_noise_plan());
    REQUIRE(noise_requirements.count(PlanFeature::PresampledNoise) == 1);
}
