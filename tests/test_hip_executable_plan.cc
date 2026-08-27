#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/hip/executable_plan.h"
#include "clifft/sampling/kernels.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/shot_seed_domains.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>
#include <string_view>
#include <variant>

using clifft::sampling::AffineBool;
using clifft::sampling::PresampledNoiseOutcome;
using clifft::sampling::PresampledNoiseSite;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolKind;
using clifft::sampling::hip::ExecutablePlan;
using clifft::sampling::hip::detail::ActionTag;

namespace {

SamplingPlan plan_from(std::string_view circuit_text) {
    return clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit_text)));
}

}  // namespace

TEST_CASE("HIP sampling has an independent random stream domain") {
    STATIC_REQUIRE(clifft::kHipSamplingExecutorDomain != clifft::kSamplingExecutorDomain);
}

TEST_CASE("HIP device layout sizes coefficient workspace consistently") {
    STATIC_REQUIRE(clifft::sampling::hip::detail::coefficient_state_capacity(0) == 1);
    STATIC_REQUIRE(clifft::sampling::hip::detail::coefficient_scratch_capacity(0) == 1);
    STATIC_REQUIRE(clifft::sampling::hip::detail::coefficient_elements_per_shot(0) == 4);
    STATIC_REQUIRE(clifft::sampling::hip::detail::coefficient_state_capacity(4) == 16);
    STATIC_REQUIRE(clifft::sampling::hip::detail::coefficient_scratch_capacity(4) == 8);
    STATIC_REQUIRE(clifft::sampling::hip::detail::coefficient_elements_per_shot(4) == 48);
}

TEST_CASE("HIP executable lowers existing sampling action names") {
    const SamplingPlan plan = plan_from(R"(
        H 0
        T 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        EXP_VAL Z0
    )");

    const ExecutablePlan executable(plan);

    REQUIRE(executable.actions().size() == plan.actions.size());
    REQUIRE(executable.actions()[0].tag == ActionTag::PromoteDormantRotation);
    REQUIRE(executable.actions()[1].tag == ActionTag::MeasureActivePauli);
    REQUIRE(executable.actions()[2].tag == ActionTag::WriteDetector);
    REQUIRE((executable.actions()[2].flags & clifft::sampling::hip::detail::kRecordParity) != 0);
    REQUIRE(executable.actions()[3].tag == ActionTag::WriteExpectationValue);
    REQUIRE(executable.actions()[4].tag == ActionTag::WriteObservable);
    REQUIRE((executable.actions()[4].flags & clifft::sampling::hip::detail::kRecordParity) != 0);
    REQUIRE(executable.num_exp_vals() == 1);
}

TEST_CASE("HIP executable inspection identifies packed modification points") {
    using Catch::Matchers::ContainsSubstring;

    const ExecutablePlan executable(plan_from("H 0\nT 0\nM 0\nDETECTOR rec[-1]\n"));
    const std::string diagnostic = executable.inspect();

    REQUIRE(executable.num_actions() == executable.actions().size());
    REQUIRE(executable.packed_bytes() >= executable.actions().size_bytes());
    REQUIRE_THAT(diagnostic, ContainsSubstring("HIP executable: actions="));
    REQUIRE_THAT(diagnostic, ContainsSubstring("PromoteDormantRotation"));
    REQUIRE_THAT(diagnostic, ContainsSubstring("MeasureActivePauli"));
    REQUIRE_THAT(diagnostic, ContainsSubstring("WriteDetector"));
}

TEST_CASE("HIP executable preserves selected output representations") {
    const SamplingPlan plan = plan_from(R"(
        X_ERROR(1) 1
        X_ERROR(1) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        READOUT_NOISE(1) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-1]
    )");
    const ExecutablePlan executable(plan);

    REQUIRE(executable.actions().size() == 4);
    const auto& historical = executable.actions()[2];
    REQUIRE(historical.tag == ActionTag::WriteObservable);
    REQUIRE((historical.flags & clifft::sampling::hip::detail::kRecordParity) == 0);
    const auto& historical_expression = executable.expressions()[historical.expression];
    REQUIRE(historical_expression.term_count == 1);
    const auto& planned_record =
        std::get<clifft::sampling::RecordClassical>(plan.actions[0].action);
    REQUIRE(planned_record.outcome.terms().size() == 1);
    const uint32_t historical_symbol = static_cast<uint32_t>(planned_record.outcome.terms()[0]);
    REQUIRE(historical_symbol != 0);
    REQUIRE(executable.expression_terms()[historical_expression.term_begin] == historical_symbol);

    const auto& current = executable.actions()[3];
    REQUIRE(current.tag == ActionTag::WriteObservable);
    REQUIRE((current.flags & clifft::sampling::hip::detail::kRecordParity) != 0);
    const auto& current_expression = executable.expressions()[current.expression];
    REQUIRE(current_expression.term_count == 1);
    REQUIRE(executable.expression_terms()[current_expression.term_begin] == 0);
}

TEST_CASE("HIP executable packs affine terms and categorical noise") {
    SamplingPlan plan;
    plan.symbols = {SymbolKind::Presampled, SymbolKind::Presampled};
    plan.presampled_noise_sites = {PresampledNoiseSite{
        0.25,
        {PresampledNoiseOutcome{SymbolId{0}, 0.125}, PresampledNoiseOutcome{SymbolId{1}, 0.125}}}};

    const ExecutablePlan executable(plan);

    REQUIRE(executable.noise_sites().size() == 1);
    REQUIRE(executable.noise_outcomes().size() == 2);
    REQUIRE(executable.noise_outcomes()[0].cumulative_probability == 0.125);
    REQUIRE(executable.noise_outcomes()[1].cumulative_probability == 0.25);
}

TEST_CASE("HIP executable flattens shared Pauli preparation") {
    const SamplingPlan plan = plan_from(R"(
        R_X(0.13) 0
        R_Y(-0.27) 1
        R_PAULI(0.31) X0*Y1
        MPP X0*Z1
    )");
    const ExecutablePlan executable(plan);

    bool saw_promotion = false;
    bool saw_rotation = false;
    bool saw_measurement = false;
    for (size_t index = 0; index < plan.actions.size(); ++index) {
        const auto& planned = plan.actions[index];
        const auto& packed = executable.actions()[index];
        if (const auto* rotation =
                std::get_if<clifft::sampling::RotateActivePauli>(&planned.action)) {
            const auto prepared = clifft::sampling::prepare_rotation(
                rotation->pauli, planned.active_before, rotation->half_turns);
            REQUIRE(packed.x == prepared.pauli.x);
            REQUIRE(packed.z == prepared.pauli.z);
            REQUIRE(packed.pair_stride == prepared.pauli.pairing_bit);
            REQUIRE(packed.phase_real == prepared.pauli.even_phase.real());
            REQUIRE(packed.phase_imag == prepared.pauli.even_phase.imag());
            REQUIRE(packed.value0 == prepared.cosine);
            REQUIRE(packed.value1 == prepared.sine);
            saw_rotation = true;
        } else if (const auto* promotion =
                       std::get_if<clifft::sampling::PromoteDormantRotation>(&planned.action)) {
            const auto prepared = clifft::sampling::prepare_promotion(promotion->half_turns);
            REQUIRE(packed.value0 == prepared.cosine);
            REQUIRE(packed.value1 == prepared.sine);
            saw_promotion = true;
        } else if (const auto* measurement =
                       std::get_if<clifft::sampling::MeasureActivePauli>(&planned.action)) {
            const auto prepared = clifft::sampling::prepare_measurement(
                measurement->pauli, planned.active_before, measurement->active_pivot);
            REQUIRE(packed.x == prepared.pauli.x);
            REQUIRE(packed.z == prepared.pauli.z);
            REQUIRE(packed.pair_stride == 0);
            REQUIRE(packed.phase_real == prepared.pauli.even_phase.real());
            REQUIRE(packed.phase_imag == prepared.pauli.even_phase.imag());
            REQUIRE(packed.index2 == prepared.pivot);
            saw_measurement = true;
        }
    }
    REQUIRE(saw_promotion);
    REQUIRE(saw_rotation);
    REQUIRE(saw_measurement);
}

TEST_CASE("HIP executable rejects work outside the first device tier") {
    using Catch::Matchers::ContainsSubstring;

    SamplingPlan wide;
    wide.num_qubits = 5;
    wide.initial_active_width = 5;
    wide.peak_active_width = 5;
    REQUIRE_THROWS_WITH(ExecutablePlan(wide), ContainsSubstring("peak active width"));

    SamplingPlan unbound;
    unbound.symbols = {SymbolKind::Presampled};
    REQUIRE_THROWS_WITH(ExecutablePlan(unbound), ContainsSubstring("presampled symbol"));
}

TEST_CASE("HIP executable identifies cultivation cooperative width") {
    using Catch::Matchers::ContainsSubstring;

    const SamplingPlan plan = clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse_file(CLIFFT_FIXTURES_DIR "/cultivation_d5.stim")));

    REQUIRE(plan.peak_active_width == 10);
    REQUIRE_THROWS_WITH(ExecutablePlan(plan), ContainsSubstring("peak active width"));
}
