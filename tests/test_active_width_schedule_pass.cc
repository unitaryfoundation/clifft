// Tests for ActiveWidthSchedulePass: the beam-search scheduling pass that
// reduces peak active width, then dense work, seeded by PeepholeFusionPass
// and StatevectorSqueezePass and never worse than that incumbent.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/active_width_schedule_pass.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pass_registry.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"

#include "instrument_test_helpers.h"
#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace clifft;
using namespace clifft::test;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::SamplingPlan;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

HirModule run_peephole_squeeze_schedule(const HirModule& source, ActiveWidthSchedulePass& pass) {
    HirModule hir = source;
    PeepholeFusionPass().run(hir);
    StatevectorSqueezePass().run(hir);
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
// Fixture certificates
// ---------------------------------------------------------------------------

// exact_node_budget defaults to 0 (off): measured on the clifft-paper
// corpus, exact repair never lowered a peak the beam search alone had not
// already reached, and its own ScheduleDependence::build on the repair
// copy (a documented O(N^2) can_swap scan) cost seconds on the larger
// fixtures for no numeric benefit. The coherent_d3_r3 section below passes
// a nonzero budget explicitly so the repair code path stays covered by a
// fixture cheap enough that the extra build does not matter.
//
// coherent_d5_r5 is a few thousand ops wide. Its beam search branches on
// tens of simultaneously-ready, mutually independent expanding rotations
// at many steps; the two-phase scoring in run_beam_search (score every
// candidate against a shared frontier and a scratch subspace copy, then
// materialize a full BeamState only for the beam_width survivors) keeps a
// full beam_width of 8 well under a second in Release, but Debug's lack of
// inlining leaves per-candidate SearchFrontier::execute/undo (a std::set
// insert/erase pair) as the dominant cost regardless of how little else is
// cloned, so this section still narrows the beam to keep the Debug test
// fast; see the next section for the beam_width 1 vs 8 dense-work
// comparison this narrowing relies on.
TEST_CASE("Schedule pass reaches the expected peak and dense work on fixture circuits",
          "[schedule_pass]") {
    SECTION("coherent_d3_r3 reaches peak 4 with exact repair explicitly enabled") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
        const HirModule raw = clifft::trace(circuit);
        ActiveWidthScheduleOptions options;
        options.exact_node_budget = 2000;  // nonzero: keeps maybe_exact_repair covered.
        ActiveWidthSchedulePass pass(options);
        const HirModule scheduled = run_peephole_squeeze_schedule(raw, pass);

        INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                               << " incumbent_dense_work=" << pass.incumbent_dense_work()
                               << " result_dense_work=" << pass.result_dense_work());
        REQUIRE(pass.result_peak() == 4);
        REQUIRE(analyze_active_width(scheduled).peak_width == 4);
    }

    SECTION("coherent_d5_r5 reaches peak at most 13 and meaningfully reduces dense work") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d5_r5.stim");
        const HirModule raw = clifft::trace(circuit);
        // beam_width 1 (greedy) reaches 38.8% of the incumbent's dense work
        // on this fixture, beam_width 8 39.8% -- beam_width 1 is actually
        // the lower (better) of the two here, by about 2.6%, rather than
        // merely close to beam_width 8; both are comfortably under half.
        // See the header comment above for why this section narrows the
        // beam despite beam_width 8 being affordable in Release.
        ActiveWidthScheduleOptions options;
        options.beam_width = 1;
        ActiveWidthSchedulePass pass(options);
        run_peephole_squeeze_schedule(raw, pass);

        INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                               << " incumbent_dense_work=" << pass.incumbent_dense_work()
                               << " result_dense_work=" << pass.result_dense_work());
        REQUIRE(pass.result_peak() <= 13);
        REQUIRE(pass.result_dense_work() < pass.incumbent_dense_work());
        REQUIRE(pass.result_dense_work() <= pass.incumbent_dense_work() * 0.7);
    }

    SECTION("cultivation_d5 keeps peak 10 without increasing dense work") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/cultivation_d5.stim");
        const HirModule raw = clifft::trace(circuit);
        ActiveWidthSchedulePass pass;  // default options (exact_node_budget = 0).
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
        ActiveWidthSchedulePass pass;  // default options (exact_node_budget = 0).
        run_peephole_squeeze_schedule(raw, pass);

        INFO("incumbent_peak=" << pass.incumbent_peak() << " result_peak=" << pass.result_peak()
                               << " incumbent_dense_work=" << pass.incumbent_dense_work()
                               << " result_dense_work=" << pass.result_dense_work());
        REQUIRE(pass.result_peak() == 0);
        REQUIRE(pass.result_dense_work() <= pass.incumbent_dense_work());
    }
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

TEST_CASE("Schedule pass registry entry resolves and stays out of the default pipeline",
          "[schedule_pass]") {
    bool found = false;
    for (const auto& info : clifft::kRegisteredPasses) {
        if (info.name == "ActiveWidthSchedulePass") {
            found = true;
            REQUIRE_FALSE(info.default_enabled);
            REQUIRE_FALSE(clifft::is_trajectory_compatible(info));
        }
    }
    REQUIRE(found);

    const std::unique_ptr<HirPass> pass = clifft::make_hir_pass("ActiveWidthSchedulePass");
    REQUIRE(pass != nullptr);

    auto circuit = clifft::parse("H 0\nCNOT 0 1\nM 0\nM 1");
    auto default_hir = clifft::trace(circuit);
    clifft::default_hir_pass_manager().run(default_hir);
    // default_hir_pass_manager() only adds default_enabled passes, so this
    // circuit's width (0 for a Clifford-only program) is untouched by
    // ActiveWidthSchedulePass regardless: the real check above is the
    // registry metadata, not this circuit's own numbers.
    REQUIRE(default_hir.num_qubits == 2);
}
