#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/hip/executable.h"
#include "clifft/sampling/planner.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string_view>

using clifft::sampling::AffineBool;
using clifft::sampling::NoiseSiteId;
using clifft::sampling::PresampledNoiseOutcome;
using clifft::sampling::PresampledNoiseSite;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolInfo;
using clifft::sampling::SymbolKind;
using clifft::sampling::hip::Executable;
using clifft::sampling::hip::detail::ActionTag;

namespace {

SamplingPlan plan_from(std::string_view circuit_text) {
    return clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit_text)));
}

}  // namespace

TEST_CASE("HIP executable lowers existing sampling action names") {
    const SamplingPlan plan = plan_from(R"(
        H 0
        T 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        EXP_VAL Z0
    )");

    const Executable executable(plan);

    REQUIRE(executable.actions().size() == plan.actions.size());
    REQUIRE(executable.actions()[0].tag == ActionTag::PromoteDormantRotation);
    REQUIRE(executable.actions()[1].tag == ActionTag::MeasureActivePauli);
    REQUIRE(executable.actions()[2].tag == ActionTag::WriteDetector);
    REQUIRE(executable.actions()[3].tag == ActionTag::WriteExpectationValue);
    REQUIRE(executable.actions()[4].tag == ActionTag::WriteObservable);
    REQUIRE(executable.num_exp_vals() == 1);
}

TEST_CASE("HIP executable packs affine terms and categorical noise") {
    SamplingPlan plan;
    plan.num_noise_sites = 1;
    plan.symbols = {
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{0}},
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{0}},
    };
    plan.presampled_noise_sites = {PresampledNoiseSite{
        NoiseSiteId{0},
        0.25,
        {PresampledNoiseOutcome{SymbolId{0}, 0.125}, PresampledNoiseOutcome{SymbolId{1}, 0.125}}}};

    const Executable executable(plan);

    REQUIRE(executable.noise_sites().size() == 1);
    REQUIRE(executable.noise_outcomes().size() == 2);
    REQUIRE(executable.noise_outcomes()[0].cumulative_probability == 0.125);
    REQUIRE(executable.noise_outcomes()[1].cumulative_probability == 0.25);
}

TEST_CASE("HIP executable rejects work outside the first device tier") {
    using Catch::Matchers::ContainsSubstring;

    SamplingPlan wide;
    wide.num_qubits = 5;
    wide.initial_active_width = 5;
    wide.peak_active_width = 5;
    REQUIRE_THROWS_WITH(Executable(wide), ContainsSubstring("peak active width"));

    SamplingPlan unbound;
    unbound.symbols = {
        SymbolInfo{SymbolKind::Presampled, std::nullopt, std::nullopt},
    };
    REQUIRE_THROWS_WITH(Executable(unbound), ContainsSubstring("presampled symbol"));
}

TEST_CASE("HIP executable identifies cultivation cooperative width") {
    using Catch::Matchers::ContainsSubstring;

    const SamplingPlan plan = clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse_file(CLIFFT_FIXTURES_DIR "/cultivation_d5.stim")));

    REQUIRE(plan.peak_active_width == 10);
    REQUIRE_THROWS_WITH(Executable(plan), ContainsSubstring("peak active width"));
}
