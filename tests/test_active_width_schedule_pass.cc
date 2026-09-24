// Checks scheduling improvements, legality, determinism, and sampling equivalence.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/active_width_schedule_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pass_registry.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/schedule_dependence.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"

#include "instrument_test_helpers.h"
#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft;
using namespace clifft::test;
using clifft::detail::ScheduleDependence;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::Executor;
using clifft::sampling::SamplingPlan;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Use explicit passes so changes to the default pipeline do not change coverage.
HirModule run_peephole_squeeze_schedule(const HirModule& source, ActiveWidthSchedulePass& pass) {
    HirModule hir = source;
    HirPassManager passes;
    passes.add_pass(std::make_unique<PeepholeFusionPass>());
    passes.add_pass(std::make_unique<StatevectorSqueezePass>());
    passes.run(hir);
    pass.run(hir);
    return hir;
}

// Independent blocks leave many rotations ready at once, exercising candidate
// scoring with a wide frontier.
std::string block_circuit_source(uint32_t num_blocks) {
    std::string source;
    for (uint32_t q = 0; q < num_blocks; ++q) {
        source += "H " + std::to_string(q) + "\n";
        source += "T " + std::to_string(q) + "\n";
        source += "H " + std::to_string(q) + "\n";
        source += "M " + std::to_string(q) + "\n";
    }
    return source;
}

bool ops_equal(const HirModule& a, const HeisenbergOp& op_a, const HirModule& b,
               const HeisenbergOp& op_b) {
    if (op_a.op_type() != op_b.op_type() || op_a.flags() != op_b.flags() ||
        op_a.has_mask() != op_b.has_mask()) {
        return false;
    }
    if (op_a.has_mask() && !(a.mask_view(op_a) == b.mask_view(op_b))) {
        return false;
    }
    switch (op_a.op_type()) {
        case OpType::MEASURE:
            return op_a.meas_record_idx() == op_b.meas_record_idx();
        case OpType::CONDITIONAL_PAULI:
            return op_a.controlling_meas() == op_b.controlling_meas();
        case OpType::NOISE:
            return op_a.noise_site_idx() == op_b.noise_site_idx();
        case OpType::READOUT_NOISE:
            return op_a.readout_noise_idx() == op_b.readout_noise_idx();
        case OpType::DETECTOR:
            return op_a.detector_idx() == op_b.detector_idx();
        case OpType::OBSERVABLE:
            return op_a.observable_idx() == op_b.observable_idx() &&
                   op_a.observable_target_list_idx() == op_b.observable_target_list_idx();
        case OpType::EXP_VAL:
            return op_a.exp_val_idx() == op_b.exp_val_idx();
        case OpType::PHASE_ROTATION:
            return op_a.alpha() == op_b.alpha();
        case OpType::INSTRUMENT:
            return op_a.instrument_site_idx() == op_b.instrument_site_idx();
        case OpType::T_GATE:
        case OpType::NUM_OP_TYPES:
            return true;
    }
    return true;
}

// Compare op payloads and parallel metadata to detect unintended HIR changes.
bool hir_unchanged(const HirModule& a, const HirModule& b) {
    if (a.num_qubits != b.num_qubits || a.ops.size() != b.ops.size()) {
        return false;
    }
    for (size_t i = 0; i < a.ops.size(); ++i) {
        if (!ops_equal(a, a.ops[i], b, b.ops[i])) {
            return false;
        }
    }
    return a.source_map == b.source_map && a.logical_noise_prefix == b.logical_noise_prefix;
}

bool is_fixed_op(OpType type) {
    return type != OpType::T_GATE && type != OpType::PHASE_ROTATION && type != OpType::MEASURE;
}

std::vector<OpType> fixed_op_sequence(const HirModule& hir) {
    std::vector<OpType> fixed;
    for (const HeisenbergOp& op : hir.ops) {
        if (is_fixed_op(op.op_type())) {
            fixed.push_back(op.op_type());
        }
    }
    return fixed;
}

const ApplyInstrument* find_instrument_action(const SamplingPlan& plan) {
    for (const auto& action : plan.actions) {
        if (const auto* instrument = std::get_if<ApplyInstrument>(&action.action)) {
            return instrument;
        }
    }
    return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// Four-operation regression
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass finds the certified optimum for the four operation circuit",
          "[schedule_pass]") {
    HirModule hir(2, 4);
    hir.num_measurements = 2;
    append_phase_rotation(hir, X(0) | X(1), 0, false, 0.3);                  // R_XX
    append_phase_rotation(hir, X(1), Z(0) | Z(1), false, 0.3);               // R_ZY
    append_measure(hir, X(0) | X(1), Z(0) | Z(1), false, MeasRecordIdx{0});  // M_YY
    append_measure(hir, X(0), Z(0), false, MeasRecordIdx{1});                // M_YI

    ActiveWidthSchedulePass pass;
    pass.run(hir);

    REQUIRE(pass.applied());
    const ActiveWidthTrace trace = analyze_active_width(hir);
    std::vector<uint32_t> widths{trace.initial_width};
    for (const WidthTransition& transition : trace.transitions) {
        widths.push_back(transition.after);
    }
    REQUIRE(widths == std::vector<uint32_t>{0, 1, 1, 1, 0});
}

// ---------------------------------------------------------------------------
// Fixture expectations
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass reaches the expected peak and dense work on fixture circuits",
          "[schedule_pass]") {
    SECTION("coherent_d3_r3 reaches peak 4") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
        const HirModule raw = clifft::trace(circuit);
        ActiveWidthSchedulePass pass;  // default options.
        const HirModule scheduled = run_peephole_squeeze_schedule(raw, pass);

        INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                               << " incumbent_dense_work=" << pass.incumbent_dense_work()
                               << " result_dense_work=" << pass.result_dense_work());
        REQUIRE(pass.result_peak() == 4);
        REQUIRE(analyze_active_width(scheduled).peak_width == 4);
    }

    SECTION("coherent_d5_r5 keeps peak 13 and cuts dense work below 0.45 of the incumbent") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d5_r5.stim");
        const HirModule raw = clifft::trace(circuit);
        // A single-candidate beam keeps this wide fixture affordable in Debug
        // while still demonstrating a dense-work improvement.
        ActiveWidthScheduleOptions options;
        options.beam_width = 1;
        ActiveWidthSchedulePass pass(options);
        run_peephole_squeeze_schedule(raw, pass);

        INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                               << " incumbent_dense_work=" << pass.incumbent_dense_work()
                               << " result_dense_work=" << pass.result_dense_work());
        REQUIRE(pass.result_peak() == 13);
        REQUIRE(pass.result_dense_work() <= pass.incumbent_dense_work() * 0.45);
    }

    SECTION("cultivation_d5 keeps peak 10 without increasing dense work") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/cultivation_d5.stim");
        const HirModule raw = clifft::trace(circuit);
        ActiveWidthSchedulePass pass;  // default options.
        run_peephole_squeeze_schedule(raw, pass);

        INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                               << " incumbent_dense_work=" << pass.incumbent_dense_work()
                               << " result_dense_work=" << pass.result_dense_work());
        REQUIRE(pass.result_peak() == 10);
        REQUIRE(pass.result_dense_work() <= pass.incumbent_dense_work());
    }

    SECTION("surface_d7_r7_p001 keeps peak 0 without increasing dense work") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/surface_d7_r7_p001.stim");
        const HirModule raw = clifft::trace(circuit);
        ActiveWidthSchedulePass pass;  // default options.
        run_peephole_squeeze_schedule(raw, pass);

        INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                               << " incumbent_dense_work=" << pass.incumbent_dense_work()
                               << " result_dense_work=" << pass.result_dense_work());
        REQUIRE(pass.result_peak() == 0);
        REQUIRE(pass.result_dense_work() <= pass.incumbent_dense_work());
    }
}

// ---------------------------------------------------------------------------
// Early exit
// ---------------------------------------------------------------------------

// A stabilizer-only circuit needs no scheduling search.
TEST_CASE("Schedule pass exits before building the dependence relation when nothing can move",
          "[schedule_pass]") {
    const Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/surface_d7_r7_p001.stim");
    const HirModule original = clifft::trace(circuit);
    HirModule hir = original;

    ActiveWidthSchedulePass pass;
    pass.run(hir);

    REQUIRE(pass.classification_probes() == 0);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.result_peak() == pass.incumbent_peak());
    REQUIRE(pass.result_dense_work() == pass.incumbent_dense_work());
    REQUIRE(hir_unchanged(hir, original));
}

// This circuit has rotations and nonzero peak width, so search must run.
TEST_CASE("Schedule pass builds the dependence relation when something can move",
          "[schedule_pass]") {
    const Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
    const HirModule raw = clifft::trace(circuit);
    ActiveWidthSchedulePass pass;
    run_peephole_squeeze_schedule(raw, pass);

    REQUIRE(pass.classification_probes() > 0);
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass rejects a zero beam width", "[schedule_pass]") {
    ActiveWidthScheduleOptions options;
    options.beam_width = 0;
    REQUIRE_THROWS_AS(ActiveWidthSchedulePass{options}, std::invalid_argument);
}

TEST_CASE("Schedule pass rejects a negative search budget", "[schedule_pass]") {
    ActiveWidthScheduleOptions options;
    options.search_budget = -1.0;
    REQUIRE_THROWS_AS(ActiveWidthSchedulePass{options}, std::invalid_argument);
}

// Opaque inputs ensure Release -ffast-math cannot fold away the nonfinite checks.
TEST_CASE("Schedule pass rejects a non-finite search budget", "[schedule_pass]") {
    SECTION("infinity") {
        ActiveWidthScheduleOptions options;
        options.search_budget = clifft::test::opaque_infinity();
        REQUIRE_THROWS_AS(ActiveWidthSchedulePass{options}, std::invalid_argument);
    }
    SECTION("negative infinity") {
        ActiveWidthScheduleOptions options;
        options.search_budget = -clifft::test::opaque_infinity();
        REQUIRE_THROWS_AS(ActiveWidthSchedulePass{options}, std::invalid_argument);
    }
    SECTION("NaN") {
        ActiveWidthScheduleOptions options;
        options.search_budget = clifft::test::opaque_nan();
        REQUIRE_THROWS_AS(ActiveWidthSchedulePass{options}, std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// Search budget
// ---------------------------------------------------------------------------

TEST_CASE("A zero search budget preserves greedy scheduling", "[schedule_pass]") {
    const HirModule raw = clifft::trace(
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim"));
    ActiveWidthScheduleOptions options;
    options.search_budget = 0.0;
    ActiveWidthSchedulePass pass(options);
    const HirModule scheduled = run_peephole_squeeze_schedule(raw, pass);

    REQUIRE(pass.classification_probes() > 0);
    REQUIRE(pass.applied());
    REQUIRE(pass.swept_ops() > 0);
    REQUIRE(pass.swept_ops() <= 3 * scheduled.ops.size());
    REQUIRE(pass.result_peak() == 5);
    REQUIRE(pass.result_dense_work() == 598);
}

// This fixture must trigger budget narrowing while retaining the same peak
// as the unbounded search.
TEST_CASE("The default search budget narrows the search on coherent_d3_r3 without losing its peak",
          "[schedule_pass]") {
    const Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
    const HirModule raw = clifft::trace(circuit);

    ActiveWidthSchedulePass default_pass;
    run_peephole_squeeze_schedule(raw, default_pass);

    ActiveWidthScheduleOptions unbounded_options;
    unbounded_options.search_budget = std::nullopt;
    ActiveWidthSchedulePass unbounded_pass(unbounded_options);
    run_peephole_squeeze_schedule(raw, unbounded_pass);

    INFO("default swept_ops=" << default_pass.swept_ops()
                              << " unbounded swept_ops=" << unbounded_pass.swept_ops());
    REQUIRE(default_pass.swept_ops() < unbounded_pass.swept_ops());
    REQUIRE(default_pass.result_peak() == unbounded_pass.result_peak());
}

TEST_CASE("Schedule pass reports swept ops through the search", "[schedule_pass]") {
    SECTION("zero after the early exit") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/surface_d7_r7_p001.stim");
        const HirModule original = clifft::trace(circuit);
        HirModule hir = original;

        ActiveWidthSchedulePass pass;
        pass.run(hir);

        REQUIRE(pass.classification_probes() == 0);
        REQUIRE(pass.swept_ops() == 0);
    }

    SECTION("positive after an applied run") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
        const HirModule raw = clifft::trace(circuit);
        ActiveWidthSchedulePass pass;
        run_peephole_squeeze_schedule(raw, pass);

        REQUIRE(pass.applied());
        REQUIRE(pass.swept_ops() > 0);
    }

    SECTION("a zero budget sweeps at most as much as an unbounded search") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
        const HirModule raw = clifft::trace(circuit);

        ActiveWidthScheduleOptions budget_options;
        budget_options.search_budget = 0.0;
        ActiveWidthSchedulePass budget_pass(budget_options);
        run_peephole_squeeze_schedule(raw, budget_pass);

        ActiveWidthScheduleOptions unbounded_options;
        unbounded_options.search_budget = std::nullopt;
        ActiveWidthSchedulePass unbounded_pass(unbounded_options);
        run_peephole_squeeze_schedule(raw, unbounded_pass);

        REQUIRE(budget_pass.swept_ops() <= unbounded_pass.swept_ops());
    }
}

// The execution budget does not bound probes of rotations that stay ready.
TEST_CASE("Search statistics distinguish executions from classification probes",
          "[schedule_pass]") {
    const HirModule raw = clifft::trace(clifft::parse(block_circuit_source(128)));
    ActiveWidthSchedulePass pass;
    const HirModule scheduled = run_peephole_squeeze_schedule(raw, pass);
    REQUIRE(pass.swept_ops() <= 20 * scheduled.ops.size());
    REQUIRE(pass.classification_probes() > pass.swept_ops());
    REQUIRE(pass.result_peak() == 1);
}

TEST_CASE("Greedy schedules preserve noisy circuit sampling", "[schedule_pass]") {
    const HirModule original = clifft::trace(
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim"));
    for (double budget : {0.0, 0.001, 0.1}) {
        ActiveWidthScheduleOptions options;
        options.search_budget = budget;
        ActiveWidthSchedulePass pass(options);
        const HirModule scheduled = run_peephole_squeeze_schedule(original, pass);
        CAPTURE(budget);
        REQUIRE(pass.swept_ops() > 0);
        check_sampling_equivalent(original, scheduled, 20000, 0x5C4E1, 0x5C4E2);
    }
}

namespace {

void check_wide_coherent_improvement(uint32_t distance) {
    const HirModule raw = clifft::trace(clifft::parse_file(
        std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d" + std::to_string(distance) + "_r3.stim"));
    ActiveWidthSchedulePass pass;
    run_peephole_squeeze_schedule(raw, pass);
    CAPTURE(distance);
    // These structural checks need no dense array at the circuit's peak width.
    REQUIRE(pass.applied());
    REQUIRE(pass.incumbent_peak() == (distance * distance + 1) / 2);
    REQUIRE(pass.result_peak() == pass.incumbent_peak() - 1);
    REQUIRE(pass.result_dense_work() < pass.incumbent_dense_work() / 50);
}

}  // namespace

TEST_CASE("Default scheduling retains distance seven coherent improvements", "[schedule_pass]") {
    check_wide_coherent_improvement(7);
}

TEST_CASE("Default scheduling retains distance nine coherent improvements",
          "[schedule_pass][large-schedule]") {
    check_wide_coherent_improvement(9);
}

TEST_CASE("A single ready expansion completes without survivor replay", "[schedule_pass]") {
    HirModule hir(1, 100);
    for (uint32_t i = 0; i < 100; ++i) {
        append_phase_rotation(hir, i % 2 == 0 ? X(0) : 0, i % 2 == 0 ? 0 : Z(0), false, 0.3);
    }
    ActiveWidthScheduleOptions options;
    options.search_budget = std::nullopt;
    ActiveWidthSchedulePass pass(options);
    pass.run(hir);
    REQUIRE(pass.swept_ops() == hir.ops.size());
}

TEST_CASE("Search counters reset when a reused pass exits early", "[schedule_pass]") {
    HirModule hir = clifft::trace(clifft::parse(block_circuit_source(64)));
    ActiveWidthSchedulePass pass;
    pass.run(hir);
    REQUIRE(pass.classification_probes() > 0);
    REQUIRE(pass.swept_ops() > 0);

    HirModule clifford = clifft::trace(clifft::parse("H 0\nM 0"));
    pass.run(clifford);
    REQUIRE(pass.swept_ops() == 0);
    REQUIRE(pass.classification_probes() == 0);
}

// ---------------------------------------------------------------------------
// Never worse than the incumbent
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass never regresses peak or dense work and leaves an unimproved HIR untouched",
          "[schedule_pass]") {
    constexpr uint32_t kSeed = 0x5C4EDA1;
    constexpr int kTrials = 100;

    clifft::Xoshiro256PlusPlus rng(kSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 7);
        const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
        const std::string source = generate_noisy_source(rng, num_qubits, num_ops);
        CAPTURE(trial, source);

        HirModule hir = clifft::trace(clifft::parse(source));
        const HirModule before = hir;
        const ActiveWidthTrace incumbent_trace = analyze_active_width(hir);
        const uint32_t incumbent_peak = incumbent_trace.peak_width;
        const double incumbent_dense_work = estimate_dense_work(incumbent_trace);

        ActiveWidthSchedulePass pass;
        pass.run(hir);

        REQUIRE(pass.incumbent_peak() == incumbent_peak);
        REQUIRE(pass.incumbent_dense_work() == incumbent_dense_work);

        const bool no_worse = (pass.result_peak() < incumbent_peak) ||
                              (pass.result_peak() == incumbent_peak &&
                               pass.result_dense_work() <= incumbent_dense_work);
        REQUIRE(no_worse);

        if (!pass.applied()) {
            REQUIRE(hir_unchanged(hir, before));
            REQUIRE(pass.result_peak() == incumbent_peak);
            REQUIRE(pass.result_dense_work() == incumbent_dense_work);
        }
    }
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

TEST_CASE("Beam ranking breaks score ties by parent rank", "[schedule_pass]") {
    // These candidates share their score and first choice but differ later.
    // Truncating the beam without resolving that tie discards the order that
    // reaches dense work 40 and instead finishes at 48.
    const HirModule raw = trace(parse(R"(R_PAULI(0.3) Y1*X3
R_PAULI(0.3) Y0
R_PAULI(0.3) Y0*X2
R_PAULI(0.3) Z3
R_PAULI(0.3) Z2*Y3
R_PAULI(0.3) Z3
)"));
    ActiveWidthScheduleOptions options;
    options.beam_width = 2;
    options.search_budget = std::nullopt;
    ActiveWidthSchedulePass pass(options);
    const HirModule scheduled = run_peephole_squeeze_schedule(raw, pass);
    REQUIRE(pass.result_peak() == 4);
    REQUIRE(pass.result_dense_work() == 40);
    const std::vector<std::vector<uint32_t>> expected_sources = {{1}, {4}, {3}, {5}, {6}, {2}};
    REQUIRE(scheduled.source_map == expected_sources);
}

TEST_CASE("Schedule pass is deterministic across repeated runs", "[schedule_pass]") {
    constexpr uint32_t kSeed = 0x0DE7511;
    constexpr int kTrials = 20;

    clifft::Xoshiro256PlusPlus rng(kSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 7);
        const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
        const std::string source = generate_noisy_source(rng, num_qubits, num_ops);
        CAPTURE(trial, source);

        HirModule hir_a = clifft::trace(clifft::parse(source));
        HirModule hir_b = hir_a;

        ActiveWidthSchedulePass pass_a;
        ActiveWidthSchedulePass pass_b;
        pass_a.run(hir_a);
        pass_b.run(hir_b);

        REQUIRE(pass_a.applied() == pass_b.applied());
        REQUIRE(pass_a.result_peak() == pass_b.result_peak());
        REQUIRE(pass_a.result_dense_work() == pass_b.result_dense_work());
        REQUIRE(hir_unchanged(hir_a, hir_b));
    }
}

// ---------------------------------------------------------------------------
// Sampling equivalence
// ---------------------------------------------------------------------------

TEST_CASE("Scheduled programs remain sampling equivalent to the unoptimized program",
          "[schedule_pass]") {
    constexpr uint32_t kShots = 20000;

    SECTION("coherent_d3_r3 fixture") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
        const HirModule original = clifft::trace(circuit);

        ActiveWidthSchedulePass pass;
        const HirModule scheduled = run_peephole_squeeze_schedule(original, pass);

        check_sampling_equivalent(original, scheduled, kShots, 0x5C4E1, 0x5C4E2);
    }

    SECTION("random noisy circuits") {
        constexpr int kTrials = 20;
        clifft::Xoshiro256PlusPlus circuit_rng(0x5A17C3);
        clifft::Xoshiro256PlusPlus control_rng(0x5EED173);
        for (int trial = 0; trial < kTrials; ++trial) {
            const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 7);
            const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
            const std::string source = generate_noisy_source(circuit_rng, num_qubits, num_ops);
            CAPTURE(trial, source);

            const HirModule original = clifft::trace(clifft::parse(source));
            ActiveWidthSchedulePass pass;
            const HirModule scheduled = run_peephole_squeeze_schedule(original, pass);

            check_sampling_equivalent(original, scheduled, kShots, control_rng(), control_rng());
        }
    }
}

TEST_CASE("Scheduled programs are exactly sampling equivalent for every checked noise realization",
          "[schedule_pass]") {
    constexpr uint32_t kCircuitSeed = 0xFACE1;
    constexpr uint32_t kControlSeed = 0xFACE2;
    constexpr int kTrials = 5000;

    clifft::Xoshiro256PlusPlus circuit_rng(kCircuitSeed);
    clifft::Xoshiro256PlusPlus control_rng(kControlSeed);
    int skipped = 0;
    int applied_count = 0;
    int crossed_count = 0;
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 3 + static_cast<uint32_t>(trial % 4);
        const uint32_t num_ops = 12 + static_cast<uint32_t>(trial % 13);
        const std::string source = generate_noisy_source(circuit_rng, num_qubits, num_ops);
        const HirModule original = clifft::trace(clifft::parse(source));
        CAPTURE(trial, num_qubits, num_ops, source);
        // Bound exponential record enumeration. The exact checker also requires
        // no hidden measurements or readout noise.
        if (original.num_measurements > 8 || original.num_hidden_measurements > 0 ||
            !original.readout_noise.empty()) {
            ++skipped;
            continue;
        }

        ActiveWidthSchedulePass pass;
        const HirModule scheduled = run_peephole_squeeze_schedule(original, pass);
        applied_count += pass.applied() ? 1 : 0;
        crossed_count += crossed_noise(scheduled) ? 1 : 0;

        check_exact_equivalent(original, scheduled, control_rng);
    }

    INFO("skipped=" << skipped << " applied=" << applied_count << " crossed=" << crossed_count);
    REQUIRE(applied_count >= 8);
    REQUIRE(crossed_count >= 5);
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass does not reorder around an EXP VAL barrier", "[schedule_pass]") {
    HirModule hir = clifft::trace(clifft::parse(
        "H 0\nT 0\nT 1\nX_ERROR(0.1) 2\nEXP_VAL Z0\nT 1\nM 0\nM 1\nM 2\nDETECTOR rec[-1]\n"));

    const std::vector<OpType> fixed_before = fixed_op_sequence(hir);
    const SamplingPlan plan_before = clifft::sampling::plan_sampling(hir);

    ActiveWidthSchedulePass pass;
    REQUIRE_NOTHROW(pass.run(hir));

    REQUIRE(fixed_op_sequence(hir) == fixed_before);
    const SamplingPlan plan_after = clifft::sampling::plan_sampling(hir);
    REQUIRE(plan_after.num_exp_vals == plan_before.num_exp_vals);
}

TEST_CASE("Schedule pass does not reorder around an INSTRUMENT barrier", "[schedule_pass]") {
    const InstrumentTraceOptions options = clifft::test::source_dependent_jump_options(false);
    HirModule hir = clifft::trace(
        clifft::parse("H 0\nT 0\nH 1\nT 1\nLEVEL_TRANSITION[jump] 1\nT 0\nM 1\nM 0\n"), &options);

    const std::vector<OpType> fixed_before = fixed_op_sequence(hir);
    const SamplingPlan plan_before = clifft::sampling::plan_sampling(hir);
    const ApplyInstrument* instrument_before = find_instrument_action(plan_before);
    REQUIRE(instrument_before != nullptr);
    const auto mode_before = instrument_before->mode;

    ActiveWidthSchedulePass pass;
    REQUIRE_NOTHROW(pass.run(hir));

    REQUIRE(fixed_op_sequence(hir) == fixed_before);
    const SamplingPlan plan_after = clifft::sampling::plan_sampling(hir);
    const ApplyInstrument* instrument_after = find_instrument_action(plan_after);
    REQUIRE(instrument_after != nullptr);
    REQUIRE(instrument_after->mode == mode_before);
    REQUIRE(plan_after.num_exp_vals == plan_before.num_exp_vals);
}

// A noise crossing after an instrument must account for the instrument's
// symbol when correcting signs. The disjoint instrument precedes the entire
// rotation/measurement group, so the reordering does not cross its barrier.
TEST_CASE(
    "Schedule pass moves a rotation across a noise site positioned after an INSTRUMENT boundary",
    "[schedule_pass]") {
    const InstrumentTraceOptions options = clifft::test::source_dependent_jump_options(false);
    const std::string source = R"(X_ERROR(0.5) 1
DEPOLARIZE1(0.3) 0
LEVEL_TRANSITION[jump] 2
R_PAULI(0.3) X0*X1
Z_ERROR(0.5) 0
R_PAULI(0.3) Z0*Y1
MPP Y0*Y1
MPP Y0
M 2
)";
    const HirModule original = clifft::trace(clifft::parse(source), &options);

    const std::vector<OpType> fixed_before = fixed_op_sequence(original);
    const SamplingPlan plan_before = clifft::sampling::plan_sampling(original);
    const ApplyInstrument* instrument_before = find_instrument_action(plan_before);
    REQUIRE(instrument_before != nullptr);
    const auto mode_before = instrument_before->mode;

    ActiveWidthSchedulePass pass;
    const HirModule scheduled = run_peephole_squeeze_schedule(original, pass);

    REQUIRE(pass.applied());
    REQUIRE(crossed_noise(scheduled));

    size_t instrument_position = scheduled.ops.size();
    for (size_t i = 0; i < scheduled.ops.size(); ++i) {
        if (scheduled.ops[i].op_type() == OpType::INSTRUMENT) {
            instrument_position = i;
            break;
        }
    }
    REQUIRE(instrument_position < scheduled.ops.size());

    // Require the noise crossing to occur after the instrument's symbol is live.
    bool disagreement_after_instrument = false;
    uint32_t schedule_count = 0;
    for (size_t i = 0; i < scheduled.ops.size(); ++i) {
        if (scheduled.logical_noise_prefix[i] != schedule_count) {
            disagreement_after_instrument |= i > instrument_position;
        }
        if (scheduled.ops[i].op_type() == OpType::NOISE) {
            ++schedule_count;
        }
    }
    REQUIRE(disagreement_after_instrument);

    REQUIRE(fixed_op_sequence(scheduled) == fixed_before);
    SamplingPlan plan_after;
    REQUIRE_NOTHROW(plan_after = clifft::sampling::plan_sampling(scheduled));
    const ApplyInstrument* instrument_after = find_instrument_action(plan_after);
    REQUIRE(instrument_after != nullptr);
    REQUIRE(instrument_after->mode == mode_before);

    // Only completed, undiscarded shots have meaningful tail records.
    constexpr uint32_t kShots = 20000;
    const uint32_t num_columns = plan_before.num_visible_records;
    REQUIRE(plan_after.num_visible_records == num_columns);

    auto collect_non_trapped = [&](const SamplingPlan& plan, uint64_t seed) {
        const ExecutablePlan executable(plan);
        Executor executor(executable, seed);
        std::vector<uint8_t> rows;
        rows.reserve(static_cast<size_t>(kShots) * num_columns);
        uint32_t kept = 0;
        for (uint32_t shot = 0; shot < kShots; ++shot) {
            executor.run_shot();
            if (executor.pending_trap().has_value() || executor.discarded()) {
                continue;
            }
            const std::span<const uint8_t> records = executor.visible_records();
            rows.insert(rows.end(), records.begin(), records.end());
            ++kept;
        }
        return std::pair<std::vector<uint8_t>, uint32_t>(std::move(rows), kept);
    };

    const auto [rows_before, kept_before] = collect_non_trapped(plan_before, 0x5C4E1);
    const auto [rows_after, kept_after] = collect_non_trapped(plan_after, 0x5C4E2);

    const double fraction_before = static_cast<double>(kept_before) / kShots;
    const double fraction_after = static_cast<double>(kept_after) / kShots;
    INFO("fraction_before=" << fraction_before << " fraction_after=" << fraction_after);
    REQUIRE(fraction_before >= 0.5);
    REQUIRE(fraction_after >= 0.5);
    const double fraction_tol = tolerance_at_6_sigma(fraction_before, fraction_after, kShots);
    REQUIRE_THAT(fraction_before, Catch::Matchers::WithinAbs(fraction_after, fraction_tol));

    // Equal sample counts simplify comparison without changing either distribution.
    const uint32_t common_rows = std::min(kept_before, kept_after);
    const std::span<const uint8_t> a(rows_before.data(),
                                     static_cast<size_t>(common_rows) * num_columns);
    const std::span<const uint8_t> b(rows_after.data(),
                                     static_cast<size_t>(common_rows) * num_columns);
    check_columns_agree(a, b, num_columns, common_rows, "record");
    check_parities_agree(a, b, num_columns, common_rows, "record");
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass registry entry resolves and the factory produces a working pass",
          "[schedule_pass]") {
    bool found = false;
    for (const auto& info : clifft::kRegisteredPasses) {
        if (info.name == "ActiveWidthSchedulePass") {
            found = true;
            REQUIRE_FALSE(info.default_enabled);
            // The trajectory pipeline excludes passes that can reorder records.
            REQUIRE_FALSE(clifft::is_trajectory_compatible(info));
        }
    }
    REQUIRE(found);

    const std::unique_ptr<HirPass> pass = clifft::make_hir_pass("ActiveWidthSchedulePass");
    REQUIRE(pass != nullptr);

    // Exercise the registry factory in a real pipeline, beyond checking its metadata.
    const Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
    HirModule hir = clifft::trace(circuit);
    HirPassManager passes;
    passes.add_pass(std::make_unique<PeepholeFusionPass>());
    passes.add_pass(std::make_unique<StatevectorSqueezePass>());
    passes.add_pass(clifft::make_hir_pass("ActiveWidthSchedulePass"));
    passes.run(hir);
    REQUIRE(analyze_active_width(hir).peak_width == 4);
}

// ---------------------------------------------------------------------------
// Beam dedup regression: peak before dense work
// ---------------------------------------------------------------------------

namespace {

// Score an order directly to avoid copying the HIR for every exhaustive candidate.
struct OrderCost {
    uint32_t peak = 0;
    double dense_work = 0.0;
};

OrderCost cost_for_order(const HirModule& hir, const std::vector<uint32_t>& order) {
    DormantSubspace subspace(hir.num_qubits);
    OrderCost cost;
    cost.peak = subspace.active_width();
    for (uint32_t op : order) {
        const WidthTransition transition = classify_and_apply(hir, hir.ops[op], subspace);
        cost.peak = std::max(cost.peak, transition.after);
        cost.dense_work += clifft::detail::dense_work_contribution(
            transition.effect, transition.before, transition.after);
    }
    return cost;
}

// Enumerate every legal order, restoring predecessor counts between branches.
void enumerate_linear_extensions(const ScheduleDependence& dep,
                                 std::vector<uint32_t>& remaining_preds,
                                 std::vector<bool>& executed, std::vector<uint32_t>& order,
                                 const HirModule& hir, OrderCost& best) {
    if (order.size() == dep.num_ops()) {
        const OrderCost cost = cost_for_order(hir, order);
        const bool better =
            cost.peak < best.peak || (cost.peak == best.peak && cost.dense_work < best.dense_work);
        if (better) {
            best = cost;
        }
        return;
    }
    for (uint32_t op = 0; op < dep.num_ops(); ++op) {
        if (executed[op] || remaining_preds[op] != 0) {
            continue;
        }
        executed[op] = true;
        order.push_back(op);
        for (uint32_t succ : dep.successors(op)) {
            --remaining_preds[succ];
        }
        enumerate_linear_extensions(dep, remaining_preds, executed, order, hir, best);
        for (uint32_t succ : dep.successors(op)) {
            ++remaining_preds[succ];
        }
        order.pop_back();
        executed[op] = false;
    }
}

// Exhaustive reference for the small regression circuits; impractical for the corpus.
OrderCost brute_force_optimum(const HirModule& hir) {
    const ScheduleDependence dep = ScheduleDependence::build(hir);
    std::vector<uint32_t> remaining_preds(dep.num_ops());
    for (uint32_t op = 0; op < dep.num_ops(); ++op) {
        remaining_preds[op] = static_cast<uint32_t>(dep.predecessors(op).size());
    }
    std::vector<bool> executed(dep.num_ops(), false);
    std::vector<uint32_t> order;
    OrderCost best;
    best.peak = std::numeric_limits<uint32_t>::max();
    best.dense_work = std::numeric_limits<double>::infinity();
    enumerate_linear_extensions(dep, remaining_preds, executed, order, hir, best);
    return best;
}

// Peak component alone, for regressions that only certify the primary
// objective.
uint32_t brute_force_min_peak(const HirModule& hir) {
    return brute_force_optimum(hir).peak;
}

}  // namespace

// Two candidates reach the same executed set with conflicting peak and dense-work
// costs. Keeping only the cheaper dense-work candidate misses the optimum peak of 3.
TEST_CASE("Schedule pass reaches the brute-force optimal peak on a beam dedup regression circuit",
          "[schedule_pass]") {
    HirModule hir(4, 8);
    hir.num_measurements = 1;
    clifft::test::append_tgate(hir, 0x8, 0x8, false);                      // Y3
    clifft::test::append_phase_rotation(hir, 0x8, 0x8, false, 0.3);        // Y3
    clifft::test::append_tgate(hir, 0x8, 0x0, false);                      // X3
    clifft::test::append_phase_rotation(hir, 0x8, 0x0, false, 0.3);        // X3
    clifft::test::append_tgate(hir, 0x6, 0x2, false);                      // Y1 X2
    clifft::test::append_tgate(hir, 0x2, 0x2, false);                      // Y1
    clifft::test::append_phase_rotation(hir, 0x1, 0x3, false, 0.3);        // Y0 Z1
    clifft::test::append_measure(hir, 0x2, 0x2, false, MeasRecordIdx{0});  // Y1

    ActiveWidthScheduleOptions options;
    options.beam_width = 16;
    options.sink_neutral_rotations = false;
    ActiveWidthSchedulePass pass(options);
    HirModule scheduled = hir;
    pass.run(scheduled);

    INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak());
    REQUIRE(pass.result_peak() == brute_force_min_peak(hir));
}

// Candidates with costs (peak 3, work 54) and (peak 4, work 52) reach the same
// executed set. A shared suffix forces peak 4, making the second candidate better.
// Keeping both reaches (4, 68); discarding the higher-peak candidate gives (4, 70).
TEST_CASE("Schedule pass reaches the brute-force optimum on a beam dedup Pareto regression circuit",
          "[schedule_pass]") {
    HirModule hir(4, 12);
    hir.num_measurements = 3;
    clifft::test::append_phase_rotation(hir, 0x2, 0x2, false, 0.875);      // Y1
    clifft::test::append_tgate(hir, 0x2, 0x0, false);                      // X1
    clifft::test::append_measure(hir, 0x5, 0x1, false, MeasRecordIdx{0});  // Y0 X2
    clifft::test::append_tgate(hir, 0x1, 0x0, false);                      // X0
    clifft::test::append_tgate(hir, 0x0, 0x9, false);                      // Z0 Z3
    clifft::test::append_tgate(hir, 0x0, 0x2, false);                      // Z1
    clifft::test::append_tgate(hir, 0x8, 0x9, false);                      // Z0 Y3
    clifft::test::append_phase_rotation(hir, 0x2, 0x2, false, 0.25);       // Y1
    clifft::test::append_measure(hir, 0x8, 0x0, false, MeasRecordIdx{1});  // X3
    clifft::test::append_phase_rotation(hir, 0x8, 0x0, false, 0.875);      // X3
    clifft::test::append_measure(hir, 0x8, 0x0, false, MeasRecordIdx{2});  // X3
    clifft::test::append_tgate(hir, 0x8, 0x8, false);                      // Y3

    ActiveWidthScheduleOptions options;
    options.beam_width = 16;
    options.sink_neutral_rotations = false;
    ActiveWidthSchedulePass pass(options);
    HirModule scheduled = hir;
    pass.run(scheduled);

    const OrderCost optimum = brute_force_optimum(hir);
    INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                           << " result_dense_work=" << pass.result_dense_work());
    REQUIRE(pass.result_peak() == optimum.peak);
    REQUIRE(pass.result_dense_work() == optimum.dense_work);
}
