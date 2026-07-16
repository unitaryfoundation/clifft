// Exact-mode continuation rewrite: the circuit-level half of the trap
// protocol. rewrite_continuation() rebuilds the full circuit under a
// shot's resolved events -- prefix verbatim (bit-identical compilation is
// what re-entry relies on), annotations kept wherever their qubit is
// still computational, classical-source consults consumed with pre-drawn
// outcomes, and the trapped jump's carrier edit inserted at the suffix
// start, with its hidden trace-out slot reported when the driver must
// force it.

#include "clifft/circuit/parser.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/transition_hooks.h"

#include "noncomp_test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <optional>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;
using clifft::test::classifier_matrix_with_column;
using clifft::test::level_index;
using clifft::test::pure_initial_state;
using clifft::test::zero_transition_matrix;

namespace {

// A model with one named leak transition (e leaks at 0.4, g at 0.1),
// seepage back from the leaked level (0.2 to e), and a faithful
// two-symbol classifier whose leaked column reads 1.
NonComputationalModel demo_model() {
    auto leak = zero_transition_matrix();
    leak[level_index(Level::LeakE)][level_index(Level::G)] = 0.1;
    leak[level_index(Level::LeakE)][level_index(Level::E)] = 0.4;
    leak[level_index(Level::E)][level_index(Level::LeakE)] = 0.2;

    return NonComputationalModel::from_spec(
        pure_initial_state(Level::G), {{"leak", leak}},
        std::make_optional(classifier_matrix_with_column(Level::LeakE, {0.0, 1.0})),
        NonComputationalPolicy{});
}

std::vector<QubitStatus> computational_initials(uint32_t n) {
    return std::vector<QubitStatus>(n, QubitStatus::Computational);
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
    auto annotated =
        expand_transition_hooks(parse("H 0\nLEVEL_TRANSITION[leak] 0\nCX 0 1\nM 0\nM 1"), model);

    ExactShotEvents events;
    events.initial_status = computational_initials(2);

    auto result = rewrite_continuation(annotated, events, /*force_last_traceout=*/false, model);
    REQUIRE(gate_sequence(result.circuit) == gate_sequence(annotated));
    REQUIRE(!result.forced_traceout_node.has_value());
    REQUIRE(result.classified_measurements.empty());
}

TEST_CASE("continuation: a trapped jump keeps its annotation and inserts the trace-out") {
    // The trapped annotation already executed -- it stays in the node
    // stream so the prefix compiles identically -- and the leaked
    // carrier's trace-out R lands right after it. The downstream CX on
    // the leaked qubit drops; the measurement classifies.
    auto model = demo_model();
    auto circuit = parse("H 0\nLEVEL_TRANSITION[leak] 0\nCX 0 1\nM 0\nM 1");
    auto annotated = expand_transition_hooks(circuit, model);

    ExactShotEvents events;
    events.initial_status = computational_initials(2);
    // This explicit annotation remains node 1 because the model has no gate hooks.
    events.jumps.push_back({{/*op_index=*/1, /*qubit=*/0}, /*destination_level=*/Level::LeakE});

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
    REQUIRE(result.classified_measurements[0].level == Level::LeakE);
}

TEST_CASE("continuation: classical-source consults consume pre-drawn outcomes") {
    // After the trap, the second annotation on the leaked qubit is a
    // classical consult: a pre-drawn seepage jump back to e rematerializes
    // the carrier (R + X) and the third annotation is quantum again.
    auto model = demo_model();
    auto circuit =
        parse("LEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 0\nM 0");
    auto annotated = expand_transition_hooks(circuit, model);

    ExactShotEvents events;
    events.initial_status = computational_initials(1);
    events.jumps.push_back({{0, 0}, Level::LeakE});
    events.classical_outcomes.push_back({{/*op_index=*/1, /*qubit=*/0},
                                         /*destination=*/Level::E,
                                         /*source_level=*/Level::LeakE});

    auto result = rewrite_continuation(annotated, events, false, model);
    const std::vector<GateType> want = {
        GateType::LEVEL_TRANSITION,  // trapped site, kept
        GateType::R,                 // trace-out of the trapped carrier
        // second annotation consumed (classical source), outcome: recapture
        GateType::R,  // carrier materialization at |1>
        GateType::X,
        GateType::LEVEL_TRANSITION,  // third annotation: quantum again, kept
        GateType::M,                 // computational again: ordinary measurement
    };
    REQUIRE(gate_sequence(result.circuit) == want);
}

TEST_CASE("continuation: the forced trace-out node names the trace-out R in the rewritten stream") {
    // The rewrite emits: [R 1, H 0, LEVEL_TRANSITION 0, R 0 (trace-out), MPAD].
    // The forced_traceout_node must point to a GateType::R on the trapped qubit (q0).
    auto model = demo_model();
    auto circuit = parse("R 1\nH 0\nLEVEL_TRANSITION[leak] 0\nM 0");
    auto annotated = expand_transition_hooks(circuit, model);

    ExactShotEvents events;
    events.initial_status = computational_initials(2);
    events.jumps.push_back({{2, 0}, Level::LeakE});

    auto result = rewrite_continuation(annotated, events, /*force_last_traceout=*/true, model);
    REQUIRE(result.forced_traceout_node.has_value());
    const uint32_t idx = result.forced_traceout_node.value();
    REQUIRE(idx < result.circuit.nodes.size());
    REQUIRE(result.circuit.nodes[idx].gate == GateType::R);
    REQUIRE(result.circuit.nodes[idx].targets[0].value() == 0);  // trapped qubit is q0
}

TEST_CASE("continuation: the forced trace-out node is correct with multiple prior resets") {
    // The rewrite emits: [R 1, R 2, H 0, LEVEL_TRANSITION 0, R 0 (trace-out), MPAD].
    // The forced_traceout_node must point to a GateType::R on the trapped qubit (q0).
    auto model = demo_model();
    auto circuit = parse("R 1\nR 2\nH 0\nLEVEL_TRANSITION[leak] 0\nM 0");
    auto annotated = expand_transition_hooks(circuit, model);

    ExactShotEvents events;
    events.initial_status = computational_initials(3);
    events.jumps.push_back({{3, 0}, Level::LeakE});  // the annotation is node 3

    auto result = rewrite_continuation(annotated, events, /*force_last_traceout=*/true, model);
    REQUIRE(result.forced_traceout_node.has_value());
    const uint32_t idx = result.forced_traceout_node.value();
    REQUIRE(idx < result.circuit.nodes.size());
    REQUIRE(result.circuit.nodes[idx].gate == GateType::R);
    REQUIRE(result.circuit.nodes[idx].targets[0].value() == 0);  // trapped qubit is q0
}

TEST_CASE("continuation: events that do not describe the circuit reject") {
    auto model = demo_model();
    auto annotated = expand_transition_hooks(parse("LEVEL_TRANSITION[leak] 0\nM 0"), model);

    ExactShotEvents base;
    base.initial_status = computational_initials(1);

    SECTION("a jump with no matching computational transition") {
        ExactShotEvents events = base;
        events.jumps.push_back({{5, 0}, Level::LeakE});
        REQUIRE_THROWS_WITH(
            rewrite_continuation(annotated, events, false, model),
            ContainsSubstring("do not match transition annotations on computational qubits"));
    }
    SECTION("an outcome with no matching leaked or lost transition") {
        ExactShotEvents events = base;
        events.classical_outcomes.push_back({{0, 0}, std::nullopt, Level::G});
        REQUIRE_THROWS_WITH(rewrite_continuation(annotated, events, false, model),
                            ContainsSubstring("more leaked or lost transition outcomes"));
    }
    SECTION("an outcome drawn at a different source level") {
        // A leaked initial means op 0 uses a pre-drawn outcome; this outcome
        // incorrectly claims it was drawn at a computational level.
        ExactShotEvents events = base;
        events.initial_status[0] = QubitStatus::LeakE;
        events.classical_outcomes.push_back({{/*op_index=*/0, /*qubit=*/0},
                                             /*destination=*/std::nullopt,
                                             /*source_level=*/Level::G});
        REQUIRE_THROWS_WITH(rewrite_continuation(annotated, events, false, model),
                            ContainsSubstring("was drawn at level"));
    }
    SECTION("forcing the trace-out any jump emits") {
        // Every recorded jump emits a carrier reset, so the forced form
        // always has one to point at. Rewritten stream: [LEVEL_TRANSITION,
        // R (trace-out), MPAD], so the trace-out R is at node index 1.
        ExactShotEvents events = base;
        events.jumps.push_back({{0, 0}, Level::LeakE});
        auto result = rewrite_continuation(annotated, events, true, model);
        REQUIRE(result.forced_traceout_node == 1);  // index in the rewritten stream
    }
}
