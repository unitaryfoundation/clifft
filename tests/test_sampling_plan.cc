#include "clifft/sampling/plan.h"
#include "clifft/util/numeric.h"

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::ApplyReadoutNoise;
using clifft::sampling::DefineSymbol;
using clifft::sampling::DetectorSlot;
using clifft::sampling::ExpValSlot;
using clifft::sampling::InstrumentBoundary;
using clifft::sampling::InstrumentDistribution;
using clifft::sampling::InstrumentMode;
using clifft::sampling::InstrumentSiteId;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::NoiseSiteId;
using clifft::sampling::ObservableSlot;
using clifft::sampling::PlannedAction;
using clifft::sampling::PresampledNoiseOutcome;
using clifft::sampling::PresampledNoiseSite;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RecordSlot;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolInfo;
using clifft::sampling::SymbolKind;
using clifft::sampling::WriteDetector;
using clifft::sampling::WriteExpectationValue;
using clifft::sampling::WriteObservable;

namespace {

SamplingPlan valid_plan() {
    const SymbolId noise{0};
    const SymbolId branch{1};
    const SymbolId derived{2};
    const SymbolId instrument_flip{3};

    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.peak_active_width = 1;
    plan.num_visible_records = 1;
    plan.num_hidden_records = 1;
    plan.num_noise_sites = 1;
    plan.num_instrument_sites = 1;
    plan.symbols = {
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{0}},
        SymbolInfo{SymbolKind::Branch, 1, std::nullopt},
        SymbolInfo{SymbolKind::Derived, 2, std::nullopt},
        SymbolInfo{SymbolKind::Instrument, 4, std::nullopt},
    };
    plan.presampled_noise_sites = {
        PresampledNoiseSite{NoiseSiteId{0}, 0.1, {PresampledNoiseOutcome{noise, 0.1}}}};
    plan.instrument_distributions = {InstrumentDistribution{InstrumentSiteId{0}, {0.0, 0.0}, {}}};
    plan.actions = {
        PlannedAction{0, 1, PromoteDormantRotation{0.25, AffineBool::symbol(noise)}},
        PlannedAction{1, 0,
                      MeasureActivePauli{ActivePauli{1, 0}, 0, branch,
                                         AffineBool::symbol(branch) ^ AffineBool::symbol(noise),
                                         RecordSlot{0}}},
        PlannedAction{0, 0, DefineSymbol{derived, AffineBool::symbol(branch) ^ true}},
        PlannedAction{0, 0, RecordClassical{AffineBool::symbol(derived), RecordSlot{1}}},
        PlannedAction{
            0, 0,
            ApplyInstrument{
                InstrumentSiteId{0}, InstrumentMode::Classical, {}, AffineBool{}, instrument_flip}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 1, 4}},
    };
    return plan;
}

SamplingPlan valid_rotation_plan() {
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.initial_active_width = 1;
    plan.peak_active_width = 1;
    plan.actions = {
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{1, 0}, 0.25, AffineBool{}}},
    };
    return plan;
}

SamplingPlan valid_dormant_measurement_plan() {
    const SymbolId branch{0};
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Branch, 0, std::nullopt}};
    plan.actions = {
        PlannedAction{0, 0,
                      MeasureDormantRandom{0, branch, AffineBool::symbol(branch), RecordSlot{0}}},
    };
    return plan;
}

SamplingPlan valid_syndrome_plan() {
    const SymbolId readout{0};
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.num_visible_records = 1;
    plan.num_detectors = 1;
    plan.num_observables = 1;
    plan.has_postselection = true;
    plan.symbols = {SymbolInfo{SymbolKind::Readout, 1, std::nullopt}};
    plan.actions = {
        PlannedAction{0, 0, RecordClassical{AffineBool{}, RecordSlot{0}}},
        PlannedAction{0, 0, ApplyReadoutNoise{readout, AffineBool{}, RecordSlot{0}, 0.1, 0.2}},
        PlannedAction{0, 0, WriteDetector{AffineBool::symbol(readout), DetectorSlot{0}, true}},
        PlannedAction{0, 0, WriteObservable{AffineBool::symbol(readout), ObservableSlot{0}}},
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

    AffineBool inserted(false, {s0, s2});
    inserted ^= AffineBool::symbol(s1);
    REQUIRE(inserted.terms() == std::vector<SymbolId>{s0, s1, s2});
    inserted ^= AffineBool::symbol(s2);
    REQUIRE(inserted.terms() == std::vector<SymbolId>{s0, s1});

    AffineBool moved;
    moved ^= AffineBool(true, {s2, s0});
    REQUIRE(moved == AffineBool(true, {s0, s2}));
    moved ^= AffineBool(false, {s0, s1});
    REQUIRE(moved == AffineBool(true, {s1, s2}));
}

TEST_CASE("Sampling plan validates symbolic and active state invariants") {
    SamplingPlan plan = valid_plan();
    REQUIRE_NOTHROW(plan.validate());

    const std::string text = plan.inspect();
    REQUIRE(text.find("sampling_plan qubits=2 initial_width=0 peak_width=1") != std::string::npos);
    REQUIRE(text.find("s0 kind=presampled noise_site=0") != std::string::npos);
    REQUIRE(text.find("1 active_width=1->0 dense_passes=2 measure_active") != std::string::npos);
    REQUIRE(text.find("outcome=s0 ^ s1 record=0") != std::string::npos);
    REQUIRE(text.find("5 active_width=0->0 dense_passes=0 instrument_boundary site=0 ") !=
            std::string::npos);
}

TEST_CASE("Sampling plan compact inspection bounds affine expressions") {
    SamplingPlan plan;
    std::vector<SymbolId> terms;
    for (uint32_t i = 0; i < 12; ++i) {
        terms.push_back(SymbolId{i});
    }
    plan.actions.push_back(PlannedAction{
        0,
        0,
        RecordClassical{AffineBool(false, std::move(terms)), RecordSlot{0}},
    });

    REQUIRE(plan.inspect_action(0).find("s11") != std::string::npos);
    const std::string compact = plan.inspect_action_compact(0);
    REQUIRE(compact.find("s7 ^ ... (12 terms)") != std::string::npos);
    REQUIRE(compact.find("s8") == std::string::npos);
}

TEST_CASE("Sampling plan rejects inconsistent noise site totals") {
    SamplingPlan plan = valid_plan();
    plan.presampled_noise_sites[0].total_probability = 0.2;

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects symbolic use before assignment") {
    SamplingPlan plan = valid_plan();
    auto& promotion = std::get<PromoteDormantRotation>(plan.actions[0].action);
    promotion.sign = AffineBool::symbol(SymbolId{1});

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects invalid symbol metadata and definitions") {
    SECTION("definition metadata disagrees with the action") {
        SamplingPlan plan = valid_plan();
        plan.symbols[2].defining_action = 3;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("presampled symbol names a defining action") {
        SamplingPlan plan = valid_plan();
        plan.symbols[0].defining_action = 0;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("noise site is out of range") {
        SamplingPlan plan = valid_plan();
        plan.symbols[0].noise_site = NoiseSiteId{1};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("non-presampled symbol names a noise site") {
        SamplingPlan plan = valid_plan();
        plan.symbols[1].noise_site = NoiseSiteId{0};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("action defines an out-of-range symbol") {
        SamplingPlan plan = valid_plan();
        std::get<DefineSymbol>(plan.actions[2].action).symbol = SymbolId{4};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("two actions define the same symbol") {
        SamplingPlan plan = valid_plan();
        std::get<DefineSymbol>(plan.actions[2].action).symbol = SymbolId{1};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("branch kind is defined by a derived-symbol action") {
        SamplingPlan plan = valid_plan();
        plan.symbols[2].kind = SymbolKind::Branch;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("derived kind is defined by a measurement") {
        SamplingPlan plan = valid_plan();
        plan.symbols[1].kind = SymbolKind::Derived;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("expression references an out-of-range symbol") {
        SamplingPlan plan = valid_plan();
        std::get<PromoteDormantRotation>(plan.actions[0].action).sign =
            AffineBool::symbol(SymbolId{4});
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("derived symbol references itself") {
        SamplingPlan plan = valid_plan();
        std::get<DefineSymbol>(plan.actions[2].action).value = AffineBool::symbol(SymbolId{2});
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("symbol kind is unrecognized") {
        SamplingPlan plan = valid_plan();
        plan.symbols[2].kind = static_cast<SymbolKind>(std::numeric_limits<uint8_t>::max());
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }
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

TEST_CASE("Sampling plan requires every declared record slot to be written") {
    SamplingPlan plan = valid_plan();
    ++plan.num_hidden_records;

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects measurement pivots outside Pauli support") {
    const SymbolId branch{0};
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.initial_active_width = 2;
    plan.peak_active_width = 2;
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

TEST_CASE("Sampling plan requires sampled branches in measurement outcomes") {
    SamplingPlan active = valid_plan();
    auto& active_measurement = std::get<MeasureActivePauli>(active.actions[1].action);
    active_measurement.outcome = AffineBool::symbol(SymbolId{0});
    REQUIRE_THROWS_AS(active.validate(), std::invalid_argument);

    SamplingPlan dormant = valid_dormant_measurement_plan();
    std::get<MeasureDormantRandom>(dormant.actions[0].action).outcome = AffineBool{};
    REQUIRE_THROWS_AS(dormant.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan rejects invalid dimensions and action contracts") {
    SECTION("initial width exceeds the circuit") {
        SamplingPlan plan;
        plan.initial_active_width = 1;
        plan.peak_active_width = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("peak width is below the initial width") {
        SamplingPlan plan;
        plan.num_qubits = 1;
        plan.initial_active_width = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("declared peak is not reached") {
        SamplingPlan plan = valid_plan();
        plan.peak_active_width = 2;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("action stream breaks the width chain") {
        SamplingPlan plan = valid_plan();
        plan.actions[1].active_before = 0;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("active rotation changes width") {
        SamplingPlan plan = valid_rotation_plan();
        plan.actions[0].active_after = 0;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("dormant promotion does not increase width") {
        SamplingPlan plan = valid_plan();
        plan.actions[0].active_after = 0;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("active measurement does not decrease width") {
        SamplingPlan plan = valid_plan();
        plan.actions[1].active_after = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("dormant measurement changes width") {
        SamplingPlan plan = valid_dormant_measurement_plan();
        plan.actions[0].active_after = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("classical record changes width") {
        SamplingPlan plan = valid_plan();
        plan.actions[3].active_after = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("symbol definition changes width") {
        SamplingPlan plan = valid_plan();
        plan.actions[2].active_after = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("instrument boundary changes width") {
        SamplingPlan plan = valid_plan();
        plan.actions[5].active_after = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("active measurement Pauli is identity") {
        SamplingPlan plan = valid_plan();
        std::get<MeasureActivePauli>(plan.actions[1].action).pauli = ActivePauli{};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("active measurement pivot is out of range") {
        SamplingPlan plan = valid_plan();
        std::get<MeasureActivePauli>(plan.actions[1].action).active_pivot = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("dormant measurement pivot is out of range") {
        SamplingPlan plan = valid_dormant_measurement_plan();
        std::get<MeasureDormantRandom>(plan.actions[0].action).dormant_pivot = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("instrument site is out of range") {
        SamplingPlan plan = valid_plan();
        std::get<InstrumentBoundary>(plan.actions[5].action).site = InstrumentSiteId{1};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("instrument mode is unrecognized") {
        SamplingPlan plan = valid_plan();
        std::get<ApplyInstrument>(plan.actions[4].action).mode =
            static_cast<InstrumentMode>(std::numeric_limits<uint8_t>::max());
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }
}

TEST_CASE("Sampling plan rejects invalid numeric metadata") {
    SECTION("record count overflows stable slots") {
        SamplingPlan plan;
        plan.num_visible_records = std::numeric_limits<uint32_t>::max();
        plan.num_hidden_records = 1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("global weight is not finite") {
        SamplingPlan plan;
        plan.global_weight = {std::numeric_limits<double>::infinity(), 0.0};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("active rotation angle is not finite") {
        SamplingPlan plan = valid_rotation_plan();
        std::get<RotateActivePauli>(plan.actions[0].action).half_turns =
            std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("dormant promotion angle is not finite") {
        SamplingPlan plan = valid_plan();
        std::get<PromoteDormantRotation>(plan.actions[0].action).half_turns =
            std::numeric_limits<double>::infinity();
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }
}

TEST_CASE("Sampling plan validates noise distributions and syndrome actions") {
    REQUIRE_NOTHROW(valid_syndrome_plan().validate());

    SECTION("a noise symbol cannot name two outcomes") {
        SamplingPlan plan = valid_plan();
        plan.presampled_noise_sites[0].outcomes.push_back(PresampledNoiseOutcome{SymbolId{0}, 0.2});
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("noise outcome probabilities cannot exceed one in total") {
        SamplingPlan plan = valid_plan();
        const SymbolId second_noise{3};
        plan.symbols.push_back(SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{0}});
        plan.presampled_noise_sites[0].outcomes[0].probability = 0.6;
        plan.presampled_noise_sites[0].outcomes.push_back(
            PresampledNoiseOutcome{second_noise, 0.6});
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("readout requires an assigned record") {
        SamplingPlan plan = valid_syndrome_plan();
        plan.actions.erase(plan.actions.begin());
        plan.symbols[0].defining_action = 0;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("readout probabilities are bounded") {
        SamplingPlan plan = valid_syndrome_plan();
        std::get<ApplyReadoutNoise>(plan.actions[1].action).prob_zero_to_one = 1.1;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("every declared output is written exactly once") {
        SamplingPlan plan = valid_syndrome_plan();
        plan.actions.pop_back();
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("postselection metadata matches detector actions") {
        SamplingPlan plan = valid_syndrome_plan();
        plan.has_postselection = false;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }
}

TEST_CASE("Sampling plan rejects active widths unsupported by dense storage") {
    SamplingPlan plan;
    plan.num_qubits = clifft::kDenseActiveWidthLimit;
    plan.initial_active_width = clifft::kDenseActiveWidthLimit;
    plan.peak_active_width = clifft::kDenseActiveWidthLimit;

    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("Sampling plan safely rejects a transition stream above dense width") {
    constexpr uint32_t kMalformedWidth = clifft::kDenseActiveWidthLimit + 10;
    SamplingPlan plan;
    plan.num_qubits = kMalformedWidth;
    plan.peak_active_width = clifft::kDenseActiveWidthLimit - 1;
    for (uint32_t width = 0; width < kMalformedWidth; ++width) {
        plan.actions.push_back(
            PlannedAction{width, width + 1, PromoteDormantRotation{0.25, AffineBool{}}});
    }
    plan.actions.push_back(PlannedAction{kMalformedWidth, kMalformedWidth,
                                         RotateActivePauli{ActivePauli{1, 0}, 0.25, AffineBool{}}});

    try {
        plan.validate();
        FAIL("malformed active-width stream should be rejected");
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        const std::string expected =
            "action " + std::to_string(clifft::kDenseActiveWidthLimit - 1) +
            " reaches active width " + std::to_string(clifft::kDenseActiveWidthLimit);
        REQUIRE(message.find(expected) != std::string::npos);
    }
}

TEST_CASE("Sampling plan distinguishes dormant branch labels from records") {
    const SymbolId branch{0};
    SamplingPlan plan = valid_dormant_measurement_plan();
    std::get<MeasureDormantRandom>(plan.actions[0].action).outcome =
        AffineBool::symbol(branch) ^ true;

    REQUIRE_NOTHROW(plan.validate());
    REQUIRE(plan.inspect().find("branch=s0 outcome=1 ^ s0 record=0") != std::string::npos);
}

TEST_CASE("Sampling plan validates expectation output slots and zero probes") {
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.initial_active_width = 1;
    plan.peak_active_width = 1;
    plan.num_exp_vals = 2;
    plan.actions = {
        PlannedAction{1, 1,
                      WriteExpectationValue{ActivePauli{1, 1}, AffineBool(true), ExpValSlot{0}}},
        PlannedAction{1, 1, WriteExpectationValue{std::nullopt, AffineBool{}, ExpValSlot{1}}},
    };

    REQUIRE_NOTHROW(plan.validate());
    const std::string text = plan.inspect();
    REQUIRE(text.find("dense_passes=1 write_expectation") != std::string::npos);
    REQUIRE(text.find("dense_passes=0 write_expectation zero exp_val=1") != std::string::npos);

    SECTION("duplicate slot") {
        std::get<WriteExpectationValue>(plan.actions[1].action).exp_val = ExpValSlot{0};
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }
    SECTION("zero probe carries irrelevant sign") {
        std::get<WriteExpectationValue>(plan.actions[1].action).sign = AffineBool(true);
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }
}

TEST_CASE("Sampling plan validates exact-query metadata") {
    SamplingPlan unitary = valid_rotation_plan();
    unitary.final_tableau = stim::Tableau<clifft::kStimWidth>(1);
    REQUIRE_NOTHROW(unitary.validate());
    REQUIRE(unitary.inspect().find("final_state_queries=1") != std::string::npos);

    SECTION("tableau width") {
        unitary.final_tableau = stim::Tableau<clifft::kStimWidth>(2);
        REQUIRE_THROWS_AS(unitary.validate(), std::invalid_argument);
    }

    SECTION("nonunitary action stream") {
        SamplingPlan measured = valid_dormant_measurement_plan();
        measured.final_tableau = stim::Tableau<clifft::kStimWidth>(1);
        REQUIRE_THROWS_AS(measured.validate(), std::invalid_argument);
    }
}

TEST_CASE("Sampling plan predicts only state touching dense passes") {
    SamplingPlan plan = valid_rotation_plan();
    REQUIRE_NOTHROW(plan.validate());
    REQUIRE(clifft::sampling::predicted_dense_passes(plan.actions[0].action) == 1);
    REQUIRE(clifft::sampling::predicted_dense_passes(
                RotateActivePauli{ActivePauli{}, 0.5, AffineBool{}}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(PromoteDormantRotation{0.25, AffineBool{}}) ==
            1);
    REQUIRE(clifft::sampling::predicted_dense_passes(MeasureActivePauli{}) == 2);
    REQUIRE(clifft::sampling::predicted_dense_passes(MeasureDormantRandom{}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(RecordClassical{}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(DefineSymbol{}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(ApplyReadoutNoise{}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(WriteDetector{}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(WriteObservable{}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(
                WriteExpectationValue{ActivePauli{1, 0}, AffineBool{}, ExpValSlot{0}}) == 1);
    REQUIRE(clifft::sampling::predicted_dense_passes(
                WriteExpectationValue{ActivePauli{}, AffineBool{}, ExpValSlot{0}}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(
                WriteExpectationValue{std::nullopt, AffineBool{}, ExpValSlot{0}}) == 0);
    REQUIRE(clifft::sampling::predicted_dense_passes(InstrumentBoundary{}) == 0);
}
