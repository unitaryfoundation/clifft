// Tests for ActiveWidthSchedulePass: the beam-search scheduling pass that
// reduces peak active width, then dense work, seeded by PeepholeFusionPass
// and StatevectorSqueezePass and never worse than that incumbent.

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
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"

#include "instrument_test_helpers.h"
#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft;
using namespace clifft::test;
using clifft::detail::ScheduleDependence;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::SamplingPlan;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Built from named passes through HirPassManager rather than
// default_hir_pass_manager(), so this suite's runtime and behavior stay
// independent of whatever passes happen to be default-enabled.
HirModule run_peephole_squeeze_schedule(const HirModule& source, ActiveWidthSchedulePass& pass) {
    HirModule hir = source;
    HirPassManager passes;
    passes.add_pass(std::make_unique<PeepholeFusionPass>());
    passes.add_pass(std::make_unique<StatevectorSqueezePass>());
    passes.run(hir);
    pass.run(hir);
    return hir;
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

// Field-by-field comparison standing in for a byte-for-byte diff: ops
// (including type-specific payload and any Pauli mask content), source_map,
// and logical_noise_prefix all have to match for two HirModules to count as
// the same program.
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
        // This fixture alone narrows the beam to keep the Debug build of
        // this test comfortably under about 15 seconds: default beam_width
        // 8 costs on the order of 20 seconds here in Debug, versus a few
        // seconds for beam_width 1, and 1 also reaches the lower (better)
        // of the two dense-work figures on this fixture (about 39% of the
        // incumbent either way).
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

// surface_d7_r7_p001 has no T_GATE or PHASE_ROTATION op at all (a pure
// stabilizer QEC memory circuit), so both early-exit conditions hold on its
// raw, unpassed HIR: incumbent peak 0, and no rotation to branch on either
// way. built_dependence() observes directly whether run() built the
// detail::ScheduleDependence relation -- the O(N^2) can_swap scan the early
// exit is supposed to skip -- rather than inferring it from a timing
// side-channel.
TEST_CASE("Schedule pass exits before building the dependence relation when nothing can move",
          "[schedule_pass]") {
    const Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/surface_d7_r7_p001.stim");
    const HirModule original = clifft::trace(circuit);
    HirModule hir = original;

    ActiveWidthSchedulePass pass;
    pass.run(hir);

    REQUIRE_FALSE(pass.built_dependence());
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.result_peak() == pass.incumbent_peak());
    REQUIRE(pass.result_dense_work() == pass.incumbent_dense_work());
    REQUIRE(hir_unchanged(hir, original));
}

// coherent_d3_r3 has T_GATE/PHASE_ROTATION ops and a nonzero incumbent
// peak, so neither early-exit condition holds: run() must build the
// dependence relation to have anything to search over.
TEST_CASE("Schedule pass builds the dependence relation when something can move",
          "[schedule_pass]") {
    const Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
    const HirModule raw = clifft::trace(circuit);
    ActiveWidthSchedulePass pass;
    run_peephole_squeeze_schedule(raw, pass);

    REQUIRE(pass.built_dependence());
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass rejects a zero beam width", "[schedule_pass]") {
    ActiveWidthScheduleOptions options;
    options.beam_width = 0;
    REQUIRE_THROWS_AS(ActiveWidthSchedulePass{options}, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Never worse than the incumbent
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass never regresses peak or dense work and leaves an unimproved HIR untouched",
          "[schedule_pass]") {
    constexpr uint32_t kSeed = 0x5C4EDA1;
    constexpr int kTrials = 100;

    std::mt19937 rng(kSeed);
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

TEST_CASE("Schedule pass is deterministic across repeated runs", "[schedule_pass]") {
    constexpr uint32_t kSeed = 0x0DE7511;
    constexpr int kTrials = 20;

    std::mt19937 rng(kSeed);
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
        std::mt19937 circuit_rng(0x5A17C3);
        std::mt19937 control_rng(0x5EED173);
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

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

TEST_CASE("Schedule pass registry entry resolves and the factory produces a working pass",
          "[schedule_pass]") {
    bool found = false;
    for (const auto& info : clifft::kRegisteredPasses) {
        if (info.name == "ActiveWidthSchedulePass") {
            found = true;
            // Off by default: see the pass's own header comment for why its
            // compile-time cost keeps it opt-in.
            REQUIRE_FALSE(info.default_enabled);
            // Record-order breaking keeps it out of clifft::noncomp::sample's
            // trajectory pipeline regardless of default_enabled: see
            // trajectory_hir_pass_manager() and the trajectory compatibility
            // test in test_optimizer.cc.
            REQUIRE_FALSE(clifft::is_trajectory_compatible(info));
        }
    }
    REQUIRE(found);

    const std::unique_ptr<HirPass> pass = clifft::make_hir_pass("ActiveWidthSchedulePass");
    REQUIRE(pass != nullptr);

    // Built entirely from named passes -- including the registry's own
    // factory function for ActiveWidthSchedulePass, rather than constructing
    // it directly -- through HirPassManager, so this proves the registry
    // wiring actually schedules, not just that the metadata says a pass by
    // this name exists, without depending on default_hir_pass_manager().
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

// Width the trace would peak at if `hir`'s ops executed in `order` (a
// permutation of op indices), replayed directly against a fresh subspace.
// Equivalent to analyze_active_width on a copy of `hir` reordered to
// `order`, without paying for that copy on every candidate order the
// brute-force search below tries.
uint32_t peak_for_order(const HirModule& hir, const std::vector<uint32_t>& order) {
    DormantSubspace subspace(hir.num_qubits);
    uint32_t peak = subspace.active_width();
    for (uint32_t op : order) {
        const WidthTransition transition = classify_and_apply(hir, hir.ops[op], subspace);
        peak = std::max(peak, transition.after);
    }
    return peak;
}

// Backtracking enumeration of every linear extension of `dep`: at each
// step, branches on every currently ready op (the same readiness
// bookkeeping SearchFrontier uses internally), recurses, then restores
// remaining_preds for every successor of the op just tried -- not only the
// ones that became newly ready -- so a sibling branch sees the same counts
// the current one started with.
void enumerate_linear_extensions(const ScheduleDependence& dep,
                                 std::vector<uint32_t>& remaining_preds,
                                 std::vector<bool>& executed, std::vector<uint32_t>& order,
                                 const HirModule& hir, uint32_t& best_peak) {
    if (order.size() == dep.num_ops()) {
        best_peak = std::min(best_peak, peak_for_order(hir, order));
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
        enumerate_linear_extensions(dep, remaining_preds, executed, order, hir, best_peak);
        for (uint32_t succ : dep.successors(op)) {
            ++remaining_preds[succ];
        }
        order.pop_back();
        executed[op] = false;
    }
}

// Minimum peak active width over every legal reordering of `hir`'s ops
// (every linear extension of its ScheduleDependence), found by exhaustive
// search rather than the pass's own beam heuristic. Only affordable because
// the regression circuit below is small (8 ops): the search space is at
// most 8! = 40320 and smaller in practice once fixed-order and
// non-commuting edges prune it.
uint32_t brute_force_min_peak(const HirModule& hir) {
    const ScheduleDependence dep = ScheduleDependence::build(hir);
    std::vector<uint32_t> remaining_preds(dep.num_ops());
    for (uint32_t op = 0; op < dep.num_ops(); ++op) {
        remaining_preds[op] = static_cast<uint32_t>(dep.predecessors(op).size());
    }
    std::vector<bool> executed(dep.num_ops(), false);
    std::vector<uint32_t> order;
    uint32_t best = std::numeric_limits<uint32_t>::max();
    enumerate_linear_extensions(dep, remaining_preds, executed, order, hir, best);
    return best;
}

}  // namespace

// Found by a seeded random search over small circuits, run against an
// instrumented copy of the dedup step that flagged every case where two
// candidates in the same beam generation converged on the same executed-op
// set at different peaks. At beam_width 16 this eight-op circuit produces
// exactly that: the lower-peak duplicate has the higher dense_work.
// Deduping by dense_work alone (the pre-fix rule) keeps the higher-peak
// duplicate instead, and the beam search never recovers -- it reports the
// incumbent's own peak (4) as final, finding no improvement at all.
// Deduping by (peak, dense_work, first_op) keeps the lower-peak duplicate
// and the pass reaches the brute-force optimum (3).
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
