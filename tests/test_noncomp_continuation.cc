// Exact-mode continuation rewrite: the circuit-level half of the trap
// protocol. rewrite_continuation() rebuilds the full circuit under a
// shot's resolved events -- prefix verbatim (bit-identical compilation is
// what re-entry relies on), annotations kept wherever their qubit is
// still computational, classical-source consults consumed with pre-drawn
// outcomes, and the trapped jump's carrier edit inserted at the suffix
// start, with its hidden trace-out slot reported when the driver must
// force it.

#include "clifft/circuit/parser.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/rewriter.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

namespace {

// Default 5-level set: two computational, then leak_g, leak_e, lost.
constexpr uint8_t kLeak = 3;

// A model with one named leak transition (e leaks at 0.4, g at 0.1),
// seepage back from the leaked level (0.2 to e), and a faithful
// two-symbol classifier whose leaked column reads 1.
NonComputationalModel demo_model(LostLeakedOpsPolicy lost_leaked = LostLeakedOpsPolicy::Drop) {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][0] = 0.1;
    leak[kLeak][1] = 0.4;
    leak[1][kLeak] = 0.2;  // seepage: leaked -> e

    ClassifierSpec classifier;
    classifier.symbols = {"0", "1"};
    classifier.matrix = {{1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.lost_leaked_ops = lost_leaked;
    return NonComputationalModel::from_spec(levels, {1.0, 0.0, 0.0, 0.0, 0.0}, {{"leak", leak}},
                                            classifier, policy);
}

std::vector<QubitStatus> computational_initials(const NonComputationalModel& model, uint32_t n) {
    return std::vector<QubitStatus>(n, model.levels().status_for(0));
}

std::vector<GateType> gate_sequence(const Circuit& circuit) {
    std::vector<GateType> gates;
    gates.reserve(circuit.nodes.size());
    for (const AstNode& node : circuit.nodes) {
        gates.push_back(node.gate);
    }
    return gates;
}

}  // namespace

TEST_CASE("continuation: empty events reproduce the annotated circuit verbatim") {
    // The main line is this rewrite with no jumps: every annotation stays
    // a runtime instrument and (with identity computational classifier
    // columns) no node is added or removed.
    auto model = demo_model();
    auto annotated = annotate(parse("H 0\nLEVEL_TRANSITION[leak] 0\nCX 0 1\nM 0\nM 1"), model);

    ExactShotEvents events;
    events.initial_status = computational_initials(model, 2);

    auto result = rewrite_continuation(annotated, events, /*force_last_traceout=*/false, model);
    REQUIRE(gate_sequence(result.circuit) == gate_sequence(annotated));
    REQUIRE(result.forced_traceout_slot == SIZE_MAX);
    REQUIRE(result.classified_measurements.empty());
}

TEST_CASE("continuation: a known |1> initial adds no X prep") {
    // Exact mode preloads known initials into the Pauli frame per shot;
    // the compiled node stream must not depend on them, or modules could
    // not be shared across shots.
    auto model = demo_model();
    auto annotated = annotate(parse("LEVEL_TRANSITION[leak] 0\nM 0"), model);

    ExactShotEvents events;
    events.initial_status = computational_initials(model, 1);
    events.initial_status[0] = model.levels().status_for(model.levels().computational_one_id());

    auto result = rewrite_continuation(annotated, events, false, model);
    REQUIRE(gate_sequence(result.circuit) == gate_sequence(annotated));
}

TEST_CASE("continuation: a trapped jump keeps its annotation and inserts the trace-out") {
    // The trapped annotation already executed -- it stays in the node
    // stream so the prefix compiles identically -- and the leaked
    // carrier's trace-out R lands right after it. The downstream CX on
    // the leaked qubit drops; the measurement classifies.
    auto model = demo_model();
    auto circuit = parse("H 0\nLEVEL_TRANSITION[leak] 0\nCX 0 1\nM 0\nM 1");
    auto annotated = annotate(circuit, model);

    ExactShotEvents events;
    events.initial_status = computational_initials(model, 2);
    // The annotation is node 1 in the source; annotate() keeps positions.
    events.jumps.push_back({/*op_index=*/1, /*qubit=*/0, /*destination_level=*/kLeak});

    auto result = rewrite_continuation(annotated, events, false, model);
    const auto gates = gate_sequence(result.circuit);
    const std::vector<GateType> want = {
        GateType::H,
        GateType::LEVEL_TRANSITION,  // kept: the prefix must compile identically
        GateType::R,                 // trace-out of the coherent leaked carrier
        // CX dropped whole (leaked control, policy Drop)
        GateType::MPAD,  // M 0 classified: leaked column reads 1 deterministically
        GateType::M,     // M 1 untouched
    };
    REQUIRE(gates == want);
    REQUIRE(result.classified_measurements.size() == 1);
    REQUIRE(result.classified_measurements[0].slot == 0);
    REQUIRE(result.classified_measurements[0].level == kLeak);
}

TEST_CASE("continuation: classical-source consults consume pre-drawn outcomes") {
    // After the trap, the second annotation on the leaked qubit is a
    // classical consult: a pre-drawn seepage jump back to e rematerializes
    // the carrier (R + X) and the third annotation is quantum again.
    auto model = demo_model();
    auto circuit =
        parse("LEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 0\nM 0");
    auto annotated = annotate(circuit, model);

    ExactShotEvents events;
    events.initial_status = computational_initials(model, 1);
    events.jumps.push_back({0, 0, kLeak});
    events.classical_outcomes.push_back(
        {/*op_index=*/1, /*qubit=*/0, /*jumped=*/true,
         /*destination_level=*/model.levels().computational_one_id(), /*source_level=*/kLeak});

    auto result = rewrite_continuation(annotated, events, false, model);
    const std::vector<GateType> want = {
        GateType::LEVEL_TRANSITION,  // trapped site, kept; the carrier is
                                     // Known |0> here, so no trace-out
        // second annotation consumed (classical source), outcome: recapture
        GateType::R,  // carrier materialization at |1>
        GateType::X,
        GateType::LEVEL_TRANSITION,  // third annotation: quantum again, kept
        GateType::M,                 // computational again: ordinary measurement
    };
    REQUIRE(gate_sequence(result.circuit) == want);
}

TEST_CASE("continuation: the forced trace-out slot mirrors trace's hidden numbering") {
    // Hidden slots are assigned per pure-reset target in circuit order,
    // after the visible slots. With one visible measurement and one reset
    // ahead of the trace-out, the trace-out owns hidden slot 2.
    auto model = demo_model();
    auto circuit = parse("R 1\nH 0\nLEVEL_TRANSITION[leak] 0\nM 0");
    auto annotated = annotate(circuit, model);

    ExactShotEvents events;
    events.initial_status = computational_initials(model, 2);
    events.jumps.push_back({2, 0, kLeak});

    auto result = rewrite_continuation(annotated, events, /*force_last_traceout=*/true, model);
    REQUIRE(result.forced_traceout_slot == 2);  // 1 visible + 1 hidden before it
}

TEST_CASE("continuation: events that do not describe the circuit reject") {
    auto model = demo_model();
    auto annotated = annotate(parse("LEVEL_TRANSITION[leak] 0\nM 0"), model);

    ExactShotEvents base;
    base.initial_status = computational_initials(model, 1);

    SECTION("a jump at an op the circuit never consults") {
        ExactShotEvents events = base;
        events.jumps.push_back({5, 0, kLeak});
        REQUIRE_THROWS_WITH(rewrite_continuation(annotated, events, false, model),
                            ContainsSubstring("never consults"));
    }
    SECTION("a classical outcome with no classical-source consult") {
        ExactShotEvents events = base;
        events.classical_outcomes.push_back({0, 0, false, 0});
        REQUIRE_THROWS_WITH(rewrite_continuation(annotated, events, false, model),
                            ContainsSubstring("more classical outcomes"));
    }
    SECTION("a classical outcome drawn at a different source level") {
        // A leaked initial makes op 0 a classical consult; the recorded
        // outcome claims it was drawn at a computational level.
        ExactShotEvents events = base;
        events.initial_status[0] = model.levels().status_for(kLeak);
        events.classical_outcomes.push_back(
            {/*op_index=*/0, /*qubit=*/0, /*jumped=*/false, /*destination_level=*/0,
             /*source_level=*/model.levels().computational_zero_id()});
        REQUIRE_THROWS_WITH(rewrite_continuation(annotated, events, false, model),
                            ContainsSubstring("was drawn at level"));
    }
    SECTION("forcing a trace-out the jump does not emit") {
        // A jump landing computationally from a *known* level emits R+X,
        // so build the definite-carrier case: initial |0> is Known, and
        // the jump lands on the leaked level -- Known carriers need no
        // trace-out, so forcing must reject.
        ExactShotEvents events = base;
        events.initial_status[0] =
            model.levels().status_for(model.levels().computational_zero_id());
        events.jumps.push_back({0, 0, kLeak});
        REQUIRE_THROWS_WITH(rewrite_continuation(annotated, events, true, model),
                            ContainsSubstring("no carrier reset"));
    }
}
