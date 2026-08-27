#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/inspection_format.h"
#include "clifft/sampling/plan.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numbers>
#include <set>
#include <string>
#include <variant>
#include <vector>

using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::ApplyReadoutNoise;
using clifft::sampling::BatchRecordParity;
using clifft::sampling::DefineSymbol;
using clifft::sampling::DetectorSlot;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::ExpValSlot;
using clifft::sampling::format_double_roundtrip;
using clifft::sampling::format_pauli_product;
using clifft::sampling::format_width_prefix;
using clifft::sampling::InstrumentBoundary;
using clifft::sampling::InstrumentMode;
using clifft::sampling::InstrumentSiteId;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::ObservableSlot;
using clifft::sampling::PlannedAction;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RecordSlot;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingAction;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::WriteDetector;
using clifft::sampling::WriteExpectationValue;
using clifft::sampling::WriteObservable;

namespace {

// Checks equality by bit pattern rather than value so -0.0 and NaN behave as
// the format contract intends instead of relying on IEEE comparison rules.
bool bit_identical(double left, double right) {
    uint64_t left_bits = 0;
    uint64_t right_bits = 0;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

double round_trip_parse(const std::string& text) {
    return std::strtod(text.c_str(), nullptr);
}

}  // namespace

TEST_CASE("format_double_roundtrip renders minimal round-trip decimals") {
    CHECK(format_double_roundtrip(0.02) == "0.02");
    CHECK(format_double_roundtrip(0.04) == "0.04");
    CHECK(format_double_roundtrip(0.25) == "0.25");
    CHECK(format_double_roundtrip(-0.0) == "-0");

    const double cos_pi_8 = std::cos(std::numbers::pi / 8.0);
    CHECK(bit_identical(round_trip_parse(format_double_roundtrip(cos_pi_8)), cos_pi_8));

    const double one_third = 1.0 / 3.0;
    CHECK(bit_identical(round_trip_parse(format_double_roundtrip(one_third)), one_third));

    const double tiny = 1e-308;
    CHECK(bit_identical(round_trip_parse(format_double_roundtrip(tiny)), tiny));

    const double huge = 1e100;
    CHECK(bit_identical(round_trip_parse(format_double_roundtrip(huge)), huge));
}

TEST_CASE("format_pauli_product renders sparse ascending Pauli letters") {
    CHECK(format_pauli_product(0, 0) == "I");
    CHECK(format_pauli_product(1, 0) == "X0");
    CHECK(format_pauli_product(0b101, 0b010) == "X0*Z1*X2");
    CHECK(format_pauli_product(uint64_t{1} << 59, uint64_t{1} << 59) == "Y59");
}

TEST_CASE("format_width_prefix reports the active-width transition") {
    CHECK(format_width_prefix(4, 4) == "w4");
    CHECK(format_width_prefix(1, 0) == "w1->0");
}

TEST_CASE("Sampling plan promotion inspection starts with the width change and mnemonic") {
    SamplingPlan plan;
    plan.actions = {
        PlannedAction{0, 1, PromoteDormantRotation{0.25, AffineBool{}}},
    };

    CHECK(plan.inspect_action_compact(0).starts_with("w0->1 PROMOTE_DORMANT"));
}

TEST_CASE("Full inspection reports dense_passes while compact reports a trailing passes field") {
    SamplingPlan dense_plan;
    dense_plan.actions = {
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{1, 0}, 0.02, AffineBool{}}},
    };
    CHECK(dense_plan.inspect_action(0) ==
          "w1 dense_passes=1 ROTATE_ACTIVE X0 half_turns=0.02 sign=0");
    CHECK(dense_plan.inspect_action_compact(0) ==
          "w1 ROTATE_ACTIVE X0 half_turns=0.02 sign=0 passes=1");

    SamplingPlan classical_plan;
    classical_plan.actions = {
        PlannedAction{0, 0, RecordClassical{AffineBool{}, RecordSlot{0}}},
    };
    CHECK(classical_plan.inspect_action(0) ==
          "w0 dense_passes=0 RECORD_CLASSICAL outcome=0 record=r0");
    const std::string classical_compact = classical_plan.inspect_action_compact(0);
    CHECK(classical_compact == "w0 RECORD_CLASSICAL outcome=0 record=r0");
    CHECK(classical_compact.find("passes") == std::string::npos);
}

TEST_CASE("Compact inspection caps affine expressions at four symbol terms") {
    std::vector<SymbolId> terms;
    for (uint32_t i = 0; i < 14; ++i) {
        terms.push_back(SymbolId{i});
    }
    SamplingPlan plan;
    plan.actions = {
        PlannedAction{0, 0, RecordClassical{AffineBool(false, terms), RecordSlot{0}}},
    };

    const std::string full = plan.inspect_action(0);
    CHECK(full.find("s13") != std::string::npos);
    CHECK(full.find("...(+") == std::string::npos);

    const std::string compact = plan.inspect_action_compact(0);
    CHECK(compact.find("outcome=s0^s1^s2^s3^...(+10)") != std::string::npos);
    CHECK(compact.find("s4") == std::string::npos);
}

TEST_CASE("Detector inspection appends postselect only when the detector is postselected") {
    SamplingPlan plan;
    plan.actions = {
        PlannedAction{0, 0, WriteDetector{AffineBool{}, DetectorSlot{0}, true, std::nullopt}},
        PlannedAction{0, 0, WriteDetector{AffineBool{}, DetectorSlot{1}, false, std::nullopt}},
    };

    CHECK(plan.inspect_action(0) ==
          "w0 dense_passes=0 WRITE_DETECTOR outcome=0 detector=d0 postselect");
    CHECK(plan.inspect_action(1) == "w0 dense_passes=0 WRITE_DETECTOR outcome=0 detector=d1");
}

TEST_CASE("Syndrome inspection renders packed record parity sidecars") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.num_detectors = 1;
    plan.num_observables = 1;
    plan.actions = {
        PlannedAction{0, 0, RecordClassical{AffineBool(true), RecordSlot{0}}},
        PlannedAction{0, 0,
                      WriteDetector{AffineBool(true), DetectorSlot{0}, false,
                                    BatchRecordParity{false, {RecordSlot{0}}}}},
        PlannedAction{0, 0,
                      WriteObservable{AffineBool(true), ObservableSlot{0},
                                      BatchRecordParity{false, {RecordSlot{0}}}}},
    };

    CHECK(plan.inspect_action(1) ==
          "w0 dense_passes=0 WRITE_DETECTOR outcome=1 detector=d0 batch_parity=r0");
    CHECK(plan.inspect_action(2) ==
          "w0 dense_passes=0 WRITE_OBSERVABLE outcome=1 observable=o0 batch_parity=r0");
    const ExecutablePlan executable(plan);
    CHECK(executable.inspect_action(1) == "WRITE_DETECTOR detector=d0 outcome=e1 batch_parity=r0");
    CHECK(executable.inspect_action(2) ==
          "WRITE_OBSERVABLE observable=o0 outcome=e2 batch_parity=r0");
}

TEST_CASE("Executable rotation prints a pairing index only for X-type prepared Paulis") {
    SamplingPlan x_type_plan;
    x_type_plan.num_qubits = 1;
    x_type_plan.initial_active_width = 1;
    x_type_plan.peak_active_width = 1;
    x_type_plan.actions = {
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{1, 0}, 0.25, AffineBool{}}},
    };
    const ExecutablePlan x_type_executable(x_type_plan);
    CHECK(x_type_executable.inspect_action(0).find(" pair=0") != std::string::npos);

    SamplingPlan diagonal_plan;
    diagonal_plan.num_qubits = 1;
    diagonal_plan.initial_active_width = 1;
    diagonal_plan.peak_active_width = 1;
    diagonal_plan.actions = {
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{0, 1}, 0.25, AffineBool{}}},
    };
    const ExecutablePlan diagonal_executable(diagonal_plan);
    CHECK(diagonal_executable.inspect_action(0).find(" pair=") == std::string::npos);
}

TEST_CASE("Compact inspection mnemonics cover every SamplingAction and its docs entry") {
    // Adding, removing, or renaming a SamplingAction alternative changes this
    // count. Extend the action list below and the plan_actions table in
    // docs/compiler_ir.json together, then update this assertion.
    static_assert(std::variant_size_v<SamplingAction> == 12);

    const SymbolId s0{0};
    const AffineBool empty{};
    const AffineBool branch_outcome = AffineBool::symbol(s0);

    SamplingPlan plan;
    plan.actions = {
        PlannedAction{1, 1, RotateActivePauli{ActivePauli{1, 0}, 0.25, empty}},
        PlannedAction{0, 1, PromoteDormantRotation{0.25, empty}},
        PlannedAction{1, 0,
                      MeasureActivePauli{ActivePauli{1, 0}, 0, s0, branch_outcome, RecordSlot{0}}},
        PlannedAction{0, 0, MeasureDormantRandom{0, s0, branch_outcome, RecordSlot{0}}},
        PlannedAction{0, 0, RecordClassical{empty, RecordSlot{0}}},
        PlannedAction{0, 0, DefineSymbol{s0, empty}},
        PlannedAction{0, 0, ApplyReadoutNoise{s0, empty, RecordSlot{0}, 0.0, 0.0}},
        PlannedAction{0, 0, WriteDetector{empty, DetectorSlot{0}, false, std::nullopt}},
        PlannedAction{0, 0, WriteObservable{empty, ObservableSlot{0}, std::nullopt}},
        PlannedAction{0, 0, WriteExpectationValue{ActivePauli{}, empty, ExpValSlot{0}}},
        PlannedAction{0, 0,
                      ApplyInstrument{InstrumentSiteId{0}, InstrumentMode::Classical, ActivePauli{},
                                      empty, s0}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 0}},
    };

    std::set<std::string> mnemonics;
    for (size_t i = 0; i < plan.actions.size(); ++i) {
        const std::string compact = plan.inspect_action_compact(i);
        // The compact form is "<width> <MNEMONIC> ...": the mnemonic is the
        // token following the width prefix's single trailing space.
        const size_t width_end = compact.find(' ');
        REQUIRE(width_end != std::string::npos);
        const size_t mnemonic_end = compact.find(' ', width_end + 1);
        mnemonics.insert(compact.substr(width_end + 1, mnemonic_end - (width_end + 1)));
    }

    const std::set<std::string> expected = {
        "ROTATE_ACTIVE",    "PROMOTE_DORMANT",   "MEASURE_ACTIVE",   "MEASURE_DORMANT",
        "RECORD_CLASSICAL", "DEFINE_SYMBOL",     "READOUT_NOISE",    "WRITE_DETECTOR",
        "WRITE_OBSERVABLE", "WRITE_EXPECTATION", "APPLY_INSTRUMENT", "INSTRUMENT_BOUNDARY",
    };
    REQUIRE(mnemonics == expected);

    // Cross-check that every mnemonic also has an entry in the docs metadata
    // that the playground tokenizer and hover text derive from. This is a
    // plain substring search, not a JSON parser, so it only confirms the key
    // is present, not the shape of its value.
    std::ifstream docs_file(std::string(CLIFFT_DOCS_DIR) + "/compiler_ir.json");
    REQUIRE(docs_file.is_open());
    const std::string docs_text((std::istreambuf_iterator<char>(docs_file)),
                                std::istreambuf_iterator<char>());
    for (const std::string& name : expected) {
        CAPTURE(name);
        REQUIRE(docs_text.find("\"" + name + "\":") != std::string::npos);
    }
}
