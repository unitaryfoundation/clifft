#include "clifft/sampling/plan.h"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::DefineSymbol;
using clifft::sampling::InstrumentBoundary;
using clifft::sampling::InstrumentSiteId;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::NoiseSiteId;
using clifft::sampling::PlannedAction;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RecordSlot;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolInfo;
using clifft::sampling::SymbolKind;

namespace {

SamplingPlan valid_plan() {
    const SymbolId noise{0};
    const SymbolId branch{1};
    const SymbolId derived{2};

    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.max_active_width = 1;
    plan.num_visible_records = 1;
    plan.num_hidden_records = 1;
    plan.num_noise_sites = 1;
    plan.num_instrument_sites = 1;
    plan.symbols = {
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{0}},
        SymbolInfo{SymbolKind::Branch, 1, std::nullopt},
        SymbolInfo{SymbolKind::Derived, 2, std::nullopt},
    };
    plan.actions = {
        PlannedAction{0, 1, PromoteDormantRotation{0, 0.25, AffineBool::symbol(noise)}},
        PlannedAction{1, 0,
                      MeasureActivePauli{ActivePauli{1, 0}, 0, branch,
                                         AffineBool::symbol(branch) ^ AffineBool::symbol(noise),
                                         RecordSlot{0}}},
        PlannedAction{0, 0, DefineSymbol{derived, AffineBool::symbol(branch) ^ true}},
        PlannedAction{0, 0, RecordClassical{AffineBool::symbol(derived), RecordSlot{1}}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}}},
    };
    return plan;
}

}  // namespace

TEST_CASE("Sampling plan affine expressions are canonical") {
    const SymbolId s0{0};
    const SymbolId s1{1};
    const SymbolId s2{2};
    AffineBool expression(false, {s2, s0, s2, s1, s1});

    REQUIRE(expression == AffineBool::symbol(s0));
    REQUIRE((expression ^ expression) == AffineBool(false));
    REQUIRE((true ^ expression).constant());
    REQUIRE((true ^ expression).terms() == std::vector<SymbolId>{s0});
}

TEST_CASE("Sampling plan validates symbolic and active state invariants") {
    SamplingPlan plan = valid_plan();
    REQUIRE_NOTHROW(plan.validate());

    const std::string text = plan.inspect();
    REQUIRE(text.find("sampling_plan qubits=2 initial_width=0 max_width=1") != std::string::npos);
    REQUIRE(text.find("s0 kind=presampled noise_site=0") != std::string::npos);
    REQUIRE(text.find("1 active_width=1->0 dense_passes=1 measure_active") != std::string::npos);
    REQUIRE(text.find("outcome=s0 ^ s1 record=0") != std::string::npos);
    REQUIRE(text.find("4 active_width=0->0 dense_passes=0 instrument_boundary site=0") !=
            std::string::npos);
}

TEST_CASE("Sampling plan rejects symbolic use before assignment") {
    SamplingPlan plan = valid_plan();
    auto& promotion = std::get<PromoteDormantRotation>(plan.actions[0].action);
    promotion.sign = AffineBool::symbol(SymbolId{1});

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects inconsistent symbol definitions") {
    SamplingPlan plan = valid_plan();
    plan.symbols[2].defining_action = 3;

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects Pauli bits above active width") {
    SamplingPlan plan = valid_plan();
    auto& measurement = std::get<MeasureActivePauli>(plan.actions[1].action);
    measurement.pauli.x = 2;

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects record slots outside stable storage") {
    SamplingPlan plan = valid_plan();
    auto& record = std::get<RecordClassical>(plan.actions[3].action);
    record.record = RecordSlot{2};

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects duplicate record writes") {
    SamplingPlan plan = valid_plan();
    auto& record = std::get<RecordClassical>(plan.actions[3].action);
    record.record = RecordSlot{0};

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects measurement pivots outside Pauli support") {
    const SymbolId branch{0};
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.initial_active_width = 2;
    plan.max_active_width = 2;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Branch, 0, std::nullopt}};
    plan.actions = {
        PlannedAction{2, 1,
                      MeasureActivePauli{ActivePauli{1, 0}, 1, branch, AffineBool::symbol(branch),
                                         RecordSlot{0}}},
    };

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);

    auto& measurement = std::get<MeasureActivePauli>(plan.actions[0].action);
    measurement.pauli = ActivePauli{0, 1};
    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);

    measurement.active_pivot = 0;
    REQUIRE_NOTHROW(plan.validate());
}

TEST_CASE("Sampling plan rejects active widths unsupported by dense storage") {
    SamplingPlan plan;
    plan.num_qubits = 60;
    plan.initial_active_width = 60;
    plan.max_active_width = 60;

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan distinguishes dormant branch labels from records") {
    const SymbolId branch{0};
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Branch, 0, std::nullopt}};
    plan.actions = {
        PlannedAction{
            0, 0,
            MeasureDormantRandom{0, branch, AffineBool::symbol(branch) ^ true, RecordSlot{0}}},
    };

    REQUIRE_NOTHROW(plan.validate());
    REQUIRE(plan.inspect().find("branch=s0 outcome=1 ^ s0 record=0") != std::string::npos);
}

TEST_CASE("Sampling plan predicts only state touching dense passes") {
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.initial_active_width = 1;
    plan.max_active_width = 1;
    plan.actions = {
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{1, 0}, 0.25, AffineBool{}}},
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{0, 0}, 0.5, AffineBool{}}},
    };

    REQUIRE_NOTHROW(plan.validate());
    REQUIRE(clifft::sampling::predicted_dense_passes(plan.actions[0].action) == 1);
    REQUIRE(clifft::sampling::predicted_dense_passes(plan.actions[1].action) == 0);
}
