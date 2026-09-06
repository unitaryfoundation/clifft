// Tests for HirModule::logical_noise_prefix: the side vector that lets the
// sampling planner resolve an operation's noise-dependent sign from its
// logical (original-circuit) position instead of its schedule position, so
// a T_GATE, PHASE_ROTATION, or MEASURE can move across a NOISE op.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/drop_non_unitary_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/symplectic.h"

#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace clifft;
using namespace clifft::test;
using clifft::sampling::AffineBool;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

template <typename T>
const T& action_as(const SamplingPlan& plan, size_t index) {
    return std::get<T>(plan.actions.at(index).action);
}

// ---------------------------------------------------------------------------
// Random noisy circuit generation
// ---------------------------------------------------------------------------
//
// GeneratedCircuit, generate_noisy_circuit, join_lines, generate_noisy_source,
// realize_noise, and crossed_noise live in sampling_equivalence_helpers.h,
// shared with test_schedule_dependence.cc and
// test_active_width_schedule_pass.cc.

// ---------------------------------------------------------------------------
// Pipeline helpers
// ---------------------------------------------------------------------------

// Built from named passes rather than default_hir_pass_manager(), so this
// suite's runtime and behavior track only the passes it actually exercises
// -- what this file's tests are about -- and not whatever else the default
// pipeline happens to grow.

HirModule run_peephole_only(const HirModule& source) {
    HirModule hir = source;
    HirPassManager passes;
    passes.add_pass(std::make_unique<PeepholeFusionPass>());
    passes.run(hir);
    return hir;
}

HirModule run_production_pipeline(const HirModule& source) {
    HirModule hir = source;
    HirPassManager passes;
    passes.add_pass(std::make_unique<PeepholeFusionPass>());
    passes.add_pass(std::make_unique<StatevectorSqueezePass>());
    passes.run(hir);
    return hir;
}

// ---------------------------------------------------------------------------
// Inertness
// ---------------------------------------------------------------------------

// Materializing logical_noise_prefix must never change what plan_sampling
// produces: an empty vector and a freshly materialized one both describe
// schedule semantics, just with the counts spelled out explicitly. A HIR
// that the planner cannot plan at all because it exceeds the dense-state
// active-width limit (legitimately, e.g. a raw or peephole-only wide
// circuit that has not been through the squeeze pass yet) is skipped
// rather than failed, matching the active-width analysis tests' precedent
// for the same class of fixture. Any other exception -- including the
// planner's own logical_noise_prefix validation -- is a bug and must
// propagate to fail the test rather than being treated as an expected skip.
void check_inert_if_plannable(const HirModule& hir) {
    SamplingPlan schedule_plan;
    try {
        schedule_plan = clifft::sampling::plan_sampling(hir);
    } catch (const std::overflow_error&) {
        return;
    }
    HirModule materialized = hir;
    materialized.materialize_logical_noise_prefix();
    const SamplingPlan materialized_plan = clifft::sampling::plan_sampling(materialized);
    REQUIRE(schedule_plan.inspect() == materialized_plan.inspect());
}

TEST_CASE("Logical noise prefix is inert on fixture circuits", "[logical_noise_prefix]") {
    static const char* const fixtures[] = {
        "coherent_d3_r3.stim",     "coherent_d5_r5.stim", "cultivation_d5.stim",
        "surface_d7_r7_p001.stim", "qv10.stim",           "surface_d11_r11_p001.stim",
        "surface_d5_r5_p05.stim",  "target_qec.stim",
    };
    for (const char* fixture : fixtures) {
        DYNAMIC_SECTION(fixture) {
            const Circuit circuit =
                clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/" + fixture);
            const HirModule raw = clifft::trace(circuit);
            check_inert_if_plannable(raw);
            check_inert_if_plannable(run_peephole_only(raw));
            check_inert_if_plannable(run_production_pipeline(raw));
        }
    }
}

TEST_CASE("Logical noise prefix is inert on random noisy circuits", "[logical_noise_prefix]") {
    constexpr uint32_t kSeed = 0x1e94a1;
    constexpr int kTrials = 200;
    clifft::Xoshiro256PlusPlus rng(kSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 6);
        const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
        const std::string source = generate_noisy_source(rng, num_qubits, num_ops);
        CAPTURE(trial, num_qubits, num_ops, source);

        const HirModule raw = clifft::trace(clifft::parse(source));
        check_inert_if_plannable(raw);
        check_inert_if_plannable(run_peephole_only(raw));
        check_inert_if_plannable(run_production_pipeline(raw));
    }
}

// ---------------------------------------------------------------------------
// Exact sign correction
// ---------------------------------------------------------------------------

TEST_CASE("Logical noise prefix corrects a measurement moved earlier than its noise site",
          "[logical_noise_prefix]") {
    const HirModule original = clifft::trace(clifft::parse("H 0\nZ_ERROR(0.3) 0\nMX 0\n"));
    REQUIRE(original.ops.size() == 2);
    REQUIRE(original.ops[0].op_type() == OpType::NOISE);
    REQUIRE(original.ops[1].op_type() == OpType::MEASURE);

    const SamplingPlan original_plan = clifft::sampling::plan_sampling(original);
    REQUIRE(original_plan.actions.size() == 1);
    const SymbolId original_noise =
        original_plan.presampled_noise_sites.at(0).outcomes.at(0).symbol;
    REQUIRE(action_as<RecordClassical>(original_plan, 0).outcome ==
            AffineBool::symbol(original_noise));

    // Reorder to H 0; MX 0; Z_ERROR(0.3) 0, keeping the measurement's
    // logical prefix at 1: it still logically follows the noise site.
    HirModule reordered = original;
    std::swap(reordered.ops[0], reordered.ops[1]);
    reordered.logical_noise_prefix = {1, 0};

    const SamplingPlan reordered_plan = clifft::sampling::plan_sampling(reordered);
    REQUIRE(reordered_plan.actions.size() == 1);
    const SymbolId reordered_noise =
        reordered_plan.presampled_noise_sites.at(0).outcomes.at(0).symbol;
    REQUIRE(action_as<RecordClassical>(reordered_plan, 0).outcome ==
            AffineBool::symbol(reordered_noise));

    // With the vector cleared (naive schedule semantics), the identical
    // reordered ops resolve to the wrong, constant outcome.
    HirModule naive = reordered;
    naive.logical_noise_prefix.clear();
    const SamplingPlan naive_plan = clifft::sampling::plan_sampling(naive);
    REQUIRE(action_as<RecordClassical>(naive_plan, 0).outcome == AffineBool(false));
}

TEST_CASE("Logical noise prefix corrects a rotation moved earlier than its noise site",
          "[logical_noise_prefix]") {
    const HirModule original = clifft::trace(clifft::parse("H 0\nX_ERROR(0.3) 0\nT 0\nM 0\n"));
    REQUIRE(original.ops.size() == 3);
    REQUIRE(original.ops[0].op_type() == OpType::NOISE);
    REQUIRE(original.ops[1].op_type() == OpType::T_GATE);
    REQUIRE(original.ops[2].op_type() == OpType::MEASURE);

    const SamplingPlan original_plan = clifft::sampling::plan_sampling(original);
    const SymbolId original_noise =
        original_plan.presampled_noise_sites.at(0).outcomes.at(0).symbol;
    REQUIRE(action_as<PromoteDormantRotation>(original_plan, 0).sign ==
            AffineBool::symbol(original_noise));

    // Move T before X_ERROR, keeping its logical prefix at 1.
    HirModule reordered = original;
    std::swap(reordered.ops[0], reordered.ops[1]);  // [T_GATE, NOISE, MEASURE]
    reordered.logical_noise_prefix = {1, 0, 1};

    const SamplingPlan reordered_plan = clifft::sampling::plan_sampling(reordered);
    const SymbolId reordered_noise =
        reordered_plan.presampled_noise_sites.at(0).outcomes.at(0).symbol;
    REQUIRE(action_as<PromoteDormantRotation>(reordered_plan, 0).sign ==
            AffineBool::symbol(reordered_noise));

    HirModule naive = reordered;
    naive.logical_noise_prefix.clear();
    const SamplingPlan naive_plan = clifft::sampling::plan_sampling(naive);
    REQUIRE(action_as<PromoteDormantRotation>(naive_plan, 0).sign == AffineBool(false));
}

TEST_CASE("Logical noise prefix corrects a rotation moved later than its noise site",
          "[logical_noise_prefix]") {
    const HirModule original = clifft::trace(clifft::parse("H 0\nT 0\nX_ERROR(0.3) 0\nM 0\n"));
    REQUIRE(original.ops.size() == 3);
    REQUIRE(original.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(original.ops[1].op_type() == OpType::NOISE);
    REQUIRE(original.ops[2].op_type() == OpType::MEASURE);

    const SamplingPlan original_plan = clifft::sampling::plan_sampling(original);
    // No noise logically precedes T in the original circuit.
    REQUIRE(action_as<PromoteDormantRotation>(original_plan, 0).sign == AffineBool(false));

    // Move T after X_ERROR, keeping its logical prefix at 0: the site never
    // logically preceded it, even though it now sits after it in schedule.
    HirModule reordered = original;
    std::swap(reordered.ops[0], reordered.ops[1]);  // [NOISE, T_GATE, MEASURE]
    reordered.logical_noise_prefix = {0, 0, 1};

    const SamplingPlan reordered_plan = clifft::sampling::plan_sampling(reordered);
    REQUIRE(action_as<PromoteDormantRotation>(reordered_plan, 0).sign == AffineBool(false));

    // Naive schedule semantics incorrectly picks up the noise dependency.
    HirModule naive = reordered;
    naive.logical_noise_prefix.clear();
    const SamplingPlan naive_plan = clifft::sampling::plan_sampling(naive);
    const SymbolId naive_noise = naive_plan.presampled_noise_sites.at(0).outcomes.at(0).symbol;
    REQUIRE(action_as<PromoteDormantRotation>(naive_plan, 0).sign ==
            AffineBool::symbol(naive_noise));
}

TEST_CASE("Logical noise prefix folds in exactly the anticommuting DEPOLARIZE1 channels",
          "[logical_noise_prefix]") {
    const HirModule original = clifft::trace(clifft::parse("DEPOLARIZE1(0.3) 0\nM 0\n"));
    REQUIRE(original.ops.size() == 2);
    REQUIRE(original.ops[0].op_type() == OpType::NOISE);
    REQUIRE(original.noise_sites.at(0).channels.size() == 3);

    const SamplingPlan original_plan = clifft::sampling::plan_sampling(original);
    // Two of DEPOLARIZE1's three channels (X, Y) anticommute with the pure
    // Z body a plain M measures; the third (Z) commutes.
    REQUIRE(action_as<RecordClassical>(original_plan, 0).outcome.terms().size() == 2);

    HirModule reordered = original;
    std::swap(reordered.ops[0], reordered.ops[1]);  // [MEASURE, NOISE]
    reordered.logical_noise_prefix = {1, 0};

    const SamplingPlan reordered_plan = clifft::sampling::plan_sampling(reordered);
    const auto& reordered_outcomes = reordered_plan.presampled_noise_sites.at(0).outcomes;
    REQUIRE(reordered_outcomes.size() == 3);

    // Build the expected outcome from this plan's own channel symbols
    // rather than assuming a fixed channel order.
    AffineBool expected;
    const std::vector<NoiseChannel>& channels = reordered.noise_sites.at(0).channels;
    for (size_t ch = 0; ch < channels.size(); ++ch) {
        const PauliMaskView mask = reordered.noise_channel_masks.at(channels[ch].mask);
        if (mask.x().bit_get(0)) {  // anticommutes with the pure Z body iff X is set
            expected ^= AffineBool::symbol(reordered_outcomes[ch].symbol);
        }
    }
    REQUIRE(expected.terms().size() == 2);
    REQUIRE(action_as<RecordClassical>(reordered_plan, 0).outcome == expected);

    HirModule naive = reordered;
    naive.logical_noise_prefix.clear();
    const SamplingPlan naive_plan = clifft::sampling::plan_sampling(naive);
    REQUIRE(action_as<RecordClassical>(naive_plan, 0).outcome == AffineBool(false));
}

TEST_CASE(
    "Logical noise prefix folds in exactly the anticommuting DEPOLARIZE2 channels for a "
    "moved rotation",
    "[logical_noise_prefix]") {
    const HirModule original =
        clifft::trace(clifft::parse("H 0\nH 1\nDEPOLARIZE2(0.2) 0 1\nT 0\nM 0\nM 1\n"));
    REQUIRE(original.ops.size() == 4);
    REQUIRE(original.ops[0].op_type() == OpType::NOISE);
    REQUIRE(original.ops[1].op_type() == OpType::T_GATE);
    REQUIRE(original.ops[2].op_type() == OpType::MEASURE);
    REQUIRE(original.ops[3].op_type() == OpType::MEASURE);
    REQUIRE(original.noise_sites.at(0).channels.size() == 15);

    const SamplingPlan original_plan = clifft::sampling::plan_sampling(original);
    const AffineBool original_sign = action_as<PromoteDormantRotation>(original_plan, 0).sign;

    // Move T before DEPOLARIZE2, keeping its logical prefix at 1: it still
    // logically follows the noise site.
    HirModule reordered = original;
    std::swap(reordered.ops[0], reordered.ops[1]);  // [T_GATE, NOISE, MEASURE, MEASURE]
    reordered.logical_noise_prefix = {1, 0, 1, 1};

    const SamplingPlan reordered_plan = clifft::sampling::plan_sampling(reordered);
    const auto& reordered_outcomes = reordered_plan.presampled_noise_sites.at(0).outcomes;
    REQUIRE(reordered_outcomes.size() == 15);

    // Build the expected sign from this plan's own channel symbols. A
    // two-qubit channel folds in exactly when its Pauli anticommutes with
    // the rotation's initial-frame body under the full symplectic inner
    // product, not just the component on the rotation's own qubit.
    AffineBool expected;
    const std::vector<NoiseChannel>& channels = reordered.noise_sites.at(0).channels;
    const HeisenbergOp& t_op = reordered.ops[0];
    for (size_t ch = 0; ch < channels.size(); ++ch) {
        const PauliMaskView mask = reordered.noise_channel_masks.at(channels[ch].mask);
        if (anti_commute(reordered.destab_mask(t_op), reordered.stab_mask(t_op), mask.x(),
                         mask.z())) {
            expected ^= AffineBool::symbol(reordered_outcomes[ch].symbol);
        }
    }
    REQUIRE(action_as<PromoteDormantRotation>(reordered_plan, 0).sign == expected);

    // T_GATE never reserves a prepass symbol, so moving it ahead of the
    // site does not shift any channel's symbol numbering: the correction
    // reproduces the original plan's sign expression term for term.
    REQUIRE(action_as<PromoteDormantRotation>(reordered_plan, 0).sign == original_sign);

    HirModule naive = reordered;
    naive.logical_noise_prefix.clear();
    const SamplingPlan naive_plan = clifft::sampling::plan_sampling(naive);
    REQUIRE(action_as<PromoteDormantRotation>(naive_plan, 0).sign == AffineBool(false));
}

TEST_CASE(
    "Logical noise prefix indexes a moved measurement's correction by nonzero "
    "PAULI_CHANNEL_1 channel",
    "[logical_noise_prefix]") {
    const HirModule original =
        clifft::trace(clifft::parse("H 0\nPAULI_CHANNEL_1(0.1, 0, 0.1) 0\nMX 0\n"));
    REQUIRE(original.ops.size() == 2);
    REQUIRE(original.ops[0].op_type() == OpType::NOISE);
    REQUIRE(original.ops[1].op_type() == OpType::MEASURE);
    // The zero-probability P(Y) argument never becomes a channel: only the
    // X and Z channels survive, so this site's outcomes are indexed by
    // nonzero channel only, with no reserved gap where Y would have been.
    REQUIRE(original.noise_sites.at(0).channels.size() == 2);

    const HeisenbergOp& measure_op = original.ops[1];
    const std::vector<NoiseChannel>& channels = original.noise_sites.at(0).channels;

    // Build the expected record from a plan's own channel symbols, keyed by
    // position in noise_sites(0).channels -- which the reorder below leaves
    // untouched -- rather than by PAULI_CHANNEL_1's original 3-argument
    // (X, Y, Z) position.
    auto expected_outcome = [&](const SamplingPlan& plan) {
        AffineBool expected;
        const auto& outcomes = plan.presampled_noise_sites.at(0).outcomes;
        REQUIRE(outcomes.size() == channels.size());
        for (size_t ch = 0; ch < channels.size(); ++ch) {
            const PauliMaskView mask = original.noise_channel_masks.at(channels[ch].mask);
            if (anti_commute(original.destab_mask(measure_op), original.stab_mask(measure_op),
                             mask.x(), mask.z())) {
                expected ^= AffineBool::symbol(outcomes[ch].symbol);
            }
        }
        return expected;
    };

    const SamplingPlan original_plan = clifft::sampling::plan_sampling(original);
    REQUIRE(action_as<RecordClassical>(original_plan, 0).outcome ==
            expected_outcome(original_plan));

    // Move MX before the channel, keeping its logical prefix at 1: it still
    // logically follows the noise site.
    HirModule reordered = original;
    std::swap(reordered.ops[0], reordered.ops[1]);  // [MEASURE, NOISE]
    reordered.logical_noise_prefix = {1, 0};

    const SamplingPlan reordered_plan = clifft::sampling::plan_sampling(reordered);
    REQUIRE(action_as<RecordClassical>(reordered_plan, 0).outcome ==
            expected_outcome(reordered_plan));

    // The reordered plan's record expression matches the original plan's:
    // the same channel-index selection, computed the same way, just
    // renumbered because MX's own branch symbol is now reserved before the
    // site's channel symbols instead of after.
    REQUIRE(action_as<RecordClassical>(original_plan, 0).outcome.terms().size() ==
            action_as<RecordClassical>(reordered_plan, 0).outcome.terms().size());

    HirModule naive = reordered;
    naive.logical_noise_prefix.clear();
    const SamplingPlan naive_plan = clifft::sampling::plan_sampling(naive);
    REQUIRE(action_as<RecordClassical>(naive_plan, 0).outcome == AffineBool(false));
}

// ---------------------------------------------------------------------------
// Reordering helper shared by sampling-equivalence and legality-oracle tests
// ---------------------------------------------------------------------------

bool is_movable_across_noise(const HeisenbergOp& op) {
    return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION ||
           op.op_type() == OpType::MEASURE;
}

// Swaps hir.ops[i] and hir.ops[i+1], carrying every parallel side vector
// along with its own operation: source_map and logical_noise_prefix when
// present, and the caller's own original-index bookkeeping.
void swap_adjacent_ops(HirModule& hir, std::vector<size_t>& original_index, size_t i) {
    std::swap(hir.ops[i], hir.ops[i + 1]);
    std::swap(original_index[i], original_index[i + 1]);
    if (hir.source_map.size() == hir.ops.size()) {
        std::swap(hir.source_map[i], hir.source_map[i + 1]);
    }
    if (hir.has_logical_noise_prefix()) {
        std::swap(hir.logical_noise_prefix[i], hir.logical_noise_prefix[i + 1]);
    }
}

// Randomly walks movable operations (T_GATE, PHASE_ROTATION, MEASURE) one
// position left or right at a time: unconditionally across an adjacent
// NOISE op (bypassing can_swap's noise clause -- the crossing this feature
// makes sound), or otherwise only when can_swap allows it. Materializes
// logical_noise_prefix first so every moved operation keeps the entry it
// started with. original_index must start as 0..ops.size()-1; after the
// walk, original_index[i] is the original position of the op now at i.
void randomly_reorder_across_noise(HirModule& hir, std::vector<size_t>& original_index,
                                   clifft::Xoshiro256PlusPlus& rng, int iterations) {
    hir.materialize_logical_noise_prefix();
    for (int iter = 0; iter < iterations && hir.ops.size() >= 2; ++iter) {
        std::vector<size_t> movable;
        for (size_t i = 0; i < hir.ops.size(); ++i) {
            if (is_movable_across_noise(hir.ops[i])) {
                movable.push_back(i);
            }
        }
        if (movable.empty()) {
            return;
        }
        const size_t i = movable[rng() % movable.size()];
        const bool go_right = (rng() % 2) == 0;
        if (go_right && i + 1 < hir.ops.size()) {
            const bool neighbor_is_noise = hir.ops[i + 1].op_type() == OpType::NOISE;
            if (neighbor_is_noise || can_swap(hir.ops[i], hir.ops[i + 1], hir)) {
                swap_adjacent_ops(hir, original_index, i);
            }
        } else if (!go_right && i > 0) {
            const bool neighbor_is_noise = hir.ops[i - 1].op_type() == OpType::NOISE;
            if (neighbor_is_noise || can_swap(hir.ops[i - 1], hir.ops[i], hir)) {
                swap_adjacent_ops(hir, original_index, i - 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling equivalence
// ---------------------------------------------------------------------------
//
// tolerance_at_6_sigma, column_mean, parity_mean, check_columns_agree,
// check_parities_agree, and check_sampling_equivalent live in
// sampling_equivalence_helpers.h, shared with test_schedule_dependence.cc.

TEST_CASE("Logical noise prefix preserves the sampling distribution across noise-crossing reorders",
          "[logical_noise_prefix]") {
    constexpr uint32_t kShots = 20000;

    SECTION("random circuits") {
        constexpr uint32_t kTrials = 30;
        clifft::Xoshiro256PlusPlus circuit_rng(0xC0FFEE);
        clifft::Xoshiro256PlusPlus control_rng(0x51DE9A1);
        for (uint32_t trial = 0; trial < kTrials; ++trial) {
            const uint32_t num_qubits = 4 + (trial % 5);
            const uint32_t num_ops = 25 + (trial % 20);
            const std::string source = generate_noisy_source(circuit_rng, num_qubits, num_ops);
            CAPTURE(trial, num_qubits, num_ops, source);

            const HirModule original = clifft::trace(clifft::parse(source));
            HirModule reordered = original;
            std::vector<size_t> original_index(reordered.ops.size());
            std::iota(original_index.begin(), original_index.end(), 0);
            clifft::Xoshiro256PlusPlus reorder_rng(control_rng());
            randomly_reorder_across_noise(reordered, original_index, reorder_rng,
                                          8 * static_cast<int>(reordered.ops.size()) + 8);

            check_sampling_equivalent(original, reordered, kShots, control_rng(), control_rng());
        }
    }

    SECTION("coherent_d3_r3 fixture") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
        const HirModule original = clifft::trace(circuit);
        HirModule reordered = original;
        std::vector<size_t> original_index(reordered.ops.size());
        std::iota(original_index.begin(), original_index.end(), 0);
        clifft::Xoshiro256PlusPlus reorder_rng(0x517EE7);
        randomly_reorder_across_noise(reordered, original_index, reorder_rng,
                                      8 * static_cast<int>(reordered.ops.size()) + 8);

        check_sampling_equivalent(original, reordered, kShots, 0x1234, 0x5678);
    }
}

TEST_CASE(
    "Logical noise prefix preserves the sampling distribution across noise-crossing reorders "
    "for an independent random sample",
    "[logical_noise_prefix]") {
    constexpr uint32_t kShots = 20000;
    constexpr uint32_t kTrials = 10;
    clifft::Xoshiro256PlusPlus circuit_rng(0xD0DE2);
    clifft::Xoshiro256PlusPlus control_rng(0x51DE9A2);
    for (uint32_t trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 4 + (trial % 5);
        const uint32_t num_ops = 25 + (trial % 20);
        const std::string source = generate_noisy_source(circuit_rng, num_qubits, num_ops);
        CAPTURE(trial, num_qubits, num_ops, source);

        const HirModule original = clifft::trace(clifft::parse(source));
        HirModule reordered = original;
        std::vector<size_t> original_index(reordered.ops.size());
        std::iota(original_index.begin(), original_index.end(), 0);
        clifft::Xoshiro256PlusPlus reorder_rng(control_rng());
        randomly_reorder_across_noise(reordered, original_index, reorder_rng,
                                      8 * static_cast<int>(reordered.ops.size()) + 8);

        check_sampling_equivalent(original, reordered, kShots, control_rng(), control_rng());
    }
}

TEST_CASE(
    "A noise-crossing reorder from the test-only walk is exactly sampling equivalent for every "
    "checked noise realization",
    "[logical_noise_prefix]") {
    constexpr uint32_t kCircuitSeed = 0xC0DE71;
    constexpr uint32_t kReorderSeed = 0xC0DE72;
    constexpr uint32_t kControlSeed = 0xC0DE73;
    constexpr int kTrials = 150;

    clifft::Xoshiro256PlusPlus circuit_rng(kCircuitSeed);
    clifft::Xoshiro256PlusPlus reorder_rng(kReorderSeed);
    clifft::Xoshiro256PlusPlus control_rng(kControlSeed);
    int checked = 0;
    int skipped = 0;
    int crossed_count = 0;
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 3 + static_cast<uint32_t>(trial % 4);
        const uint32_t num_ops = 12 + static_cast<uint32_t>(trial % 13);
        const std::string source = generate_noisy_source(circuit_rng, num_qubits, num_ops);
        const HirModule original = clifft::trace(clifft::parse(source));
        CAPTURE(trial, num_qubits, num_ops, source);
        // A high measurement count makes the exact check's per-record
        // enumeration (2^num_visible_records replays per realization)
        // expensive; that cost buys nothing this test needs, so skip it. A
        // plain R or a noisy measurement disqualifies a trial outright: the
        // former lowers to a hidden measurement and the latter to a
        // READOUT_NOISE action, and check_exact_equivalent requires neither.
        if (original.num_measurements > 8 || original.num_hidden_measurements > 0 ||
            !original.readout_noise.empty()) {
            ++skipped;
            continue;
        }

        HirModule reordered = original;
        std::vector<size_t> original_index(reordered.ops.size());
        std::iota(original_index.begin(), original_index.end(), 0);
        randomly_reorder_across_noise(reordered, original_index, reorder_rng,
                                      8 * static_cast<int>(reordered.ops.size()) + 8);
        crossed_count += crossed_noise(reordered) ? 1 : 0;

        check_exact_equivalent(original, reordered, control_rng);
        ++checked;
    }

    INFO("checked=" << checked << " skipped=" << skipped << " crossed=" << crossed_count);
    REQUIRE(checked >= 10);
    // Without this, "checked" alone would pass even if the test-only walk
    // never actually moved an operation across a noise site, leaving the
    // exact check exercising only trivial (non-crossing) reorders.
    REQUIRE(crossed_count >= 10);
}

// ---------------------------------------------------------------------------
// Legality oracle
// ---------------------------------------------------------------------------

TEST_CASE("A logical-noise-prefix reorder is a legal can_swap reordering of the realized circuit",
          "[logical_noise_prefix]") {
    constexpr uint32_t kTrials = 30;
    clifft::Xoshiro256PlusPlus circuit_rng(0xA11CE);
    size_t fired = 0;
    size_t dropped = 0;
    for (uint32_t trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 4 + (trial % 5);
        const uint32_t num_ops = 20 + (trial % 20);
        const GeneratedCircuit generated = generate_noisy_circuit(circuit_rng, num_qubits, num_ops);
        const std::string noisy_source = join_lines(generated.lines);
        CAPTURE(trial, num_qubits, num_ops, noisy_source);

        HirModule noisy = clifft::trace(clifft::parse(noisy_source));
        std::vector<uint8_t> is_noise_original(noisy.ops.size());
        for (size_t i = 0; i < noisy.ops.size(); ++i) {
            is_noise_original[i] = noisy.ops[i].op_type() == OpType::NOISE ? 1 : 0;
        }

        std::vector<size_t> original_index(noisy.ops.size());
        std::iota(original_index.begin(), original_index.end(), 0);
        clifft::Xoshiro256PlusPlus reorder_rng(0xB0B0 + trial);
        randomly_reorder_across_noise(noisy, original_index, reorder_rng,
                                      6 * static_cast<int>(noisy.ops.size()) + 6);

        clifft::Xoshiro256PlusPlus realize_rng(0xFACADE + trial);
        const RealizedCircuit realized_circuit = realize_noise(generated, realize_rng);
        fired += realized_circuit.fired;
        dropped += realized_circuit.dropped;
        const std::string realized_source = join_lines(realized_circuit.lines);
        CAPTURE(realized_source);
        const HirModule realized = clifft::trace(clifft::parse(realized_source));

        // Map each non-noise original index to its position in the
        // noise-free realized HIR by counting how many earlier original
        // positions were noise sites.
        std::vector<size_t> orig_to_realized(is_noise_original.size(), 0);
        size_t realized_cursor = 0;
        for (size_t i = 0; i < is_noise_original.size(); ++i) {
            if (!is_noise_original[i]) {
                orig_to_realized[i] = realized_cursor++;
            }
        }
        REQUIRE(realized_cursor == realized.ops.size());

        // The reordering is legal exactly when every pair that ended up
        // inverted relative to original order is a can_swap-legal
        // transposition: that is a sufficient condition for reachability by
        // adjacent transpositions, since any sequence of legal adjacent
        // swaps that sorts the permutation back to original order must, at
        // some point, swap every such pair directly.
        for (size_t p = 0; p < original_index.size(); ++p) {
            if (is_noise_original[original_index[p]]) {
                continue;
            }
            for (size_t q = p + 1; q < original_index.size(); ++q) {
                if (is_noise_original[original_index[q]]) {
                    continue;
                }
                const size_t orig_a = original_index[p];
                const size_t orig_b = original_index[q];
                if (orig_a <= orig_b) {
                    continue;  // not inverted relative to original order
                }
                const size_t ra = orig_to_realized[orig_a];
                const size_t rb = orig_to_realized[orig_b];
                CAPTURE(p, q, orig_a, orig_b, ra, rb);
                CHECK(can_swap(realized.ops[rb], realized.ops[ra], realized));
            }
        }
    }

    // A realization stream that only ever fires (or only ever drops) would
    // leave this oracle checking a single realization class per circuit --
    // e.g. always all channels active -- rather than the mix of fired and
    // dropped sites a real noise draw produces.
    INFO("fired=" << fired << " dropped=" << dropped);
    REQUIRE(fired > 0);
    REQUIRE(dropped > 0);
}

// ---------------------------------------------------------------------------
// Pass maintenance
// ---------------------------------------------------------------------------

TEST_CASE("StatevectorSqueezePass keeps logical_noise_prefix parallel and attached to its op",
          "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse(R"(
        T 0
        T 1
        H 1
        DEPOLARIZE1(0.1) 0
        M 1
        M 0
    )"));
    REQUIRE(hir.source_map.size() == hir.ops.size());
    hir.materialize_logical_noise_prefix();

    // source_map (the source line) uniquely tags each surviving op here, so
    // pairing it with logical_noise_prefix lets us confirm the entry
    // followed the same op through however the pass permutes them.
    auto snapshot = [](const HirModule& module) {
        std::vector<std::pair<std::vector<uint32_t>, uint32_t>> tagged;
        for (size_t i = 0; i < module.ops.size(); ++i) {
            tagged.emplace_back(module.source_map[i], module.logical_noise_prefix[i]);
        }
        std::ranges::sort(tagged);
        return tagged;
    };
    const auto before = snapshot(hir);

    StatevectorSqueezePass pass;
    pass.run(hir);

    REQUIRE(hir.has_logical_noise_prefix());
    REQUIRE(hir.source_map.size() == hir.ops.size());
    REQUIRE(snapshot(hir) == before);
}

TEST_CASE("PeepholeFusionPass fuses normally when logical_noise_prefix is consistent",
          "[logical_noise_prefix]") {
    HirModule with_prefix = clifft::trace(clifft::parse("H 0\nT 0\nT 0\nM 0\n"));
    with_prefix.materialize_logical_noise_prefix();
    HirModule without_prefix = clifft::trace(clifft::parse("H 0\nT 0\nT 0\nM 0\n"));

    PeepholeFusionPass pass_with;
    pass_with.run(with_prefix);
    PeepholeFusionPass pass_without;
    pass_without.run(without_prefix);

    REQUIRE(clifft::sampling::plan_sampling(with_prefix).inspect() ==
            clifft::sampling::plan_sampling(without_prefix).inspect());
    if (!with_prefix.ops.empty()) {
        REQUIRE(with_prefix.has_logical_noise_prefix());
    }
}

TEST_CASE("PeepholeFusionPass leaves an inconsistent logical_noise_prefix unchanged",
          "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse("H 0\nZ_ERROR(0.3) 0\nT 0\nT 0\nM 0\n"));
    hir.materialize_logical_noise_prefix();
    REQUIRE(hir.ops.size() == 4);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[1].op_type() == OpType::T_GATE);
    // Force the first T to look like it logically precedes the noise site,
    // which the schedule count (1) disagrees with.
    hir.logical_noise_prefix[1] = 0;

    const size_t ops_before = hir.ops.size();
    const std::vector<uint32_t> prefix_before = hir.logical_noise_prefix;

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == ops_before);
    REQUIRE(hir.logical_noise_prefix == prefix_before);
    size_t t_gate_count = 0;
    for (const HeisenbergOp& op : hir.ops) {
        t_gate_count += op.op_type() == OpType::T_GATE ? 1 : 0;
    }
    REQUIRE(t_gate_count == 2);  // the fusable pair was not fused
}

TEST_CASE("RemoveNoisePass clears logical_noise_prefix", "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse("Z_ERROR(0.3) 0\nM 0\n"));
    hir.materialize_logical_noise_prefix();
    REQUIRE(hir.has_logical_noise_prefix());

    RemoveNoisePass pass;
    pass.run(hir);

    REQUIRE(hir.logical_noise_prefix.empty());
    REQUIRE(hir.noise_sites.empty());
}

TEST_CASE("DropNonUnitaryPass clears logical_noise_prefix", "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse("Z_ERROR(0.3) 0\nT 0\nM 0\n"));
    hir.materialize_logical_noise_prefix();
    REQUIRE(hir.has_logical_noise_prefix());

    DropNonUnitaryPass pass;
    pass.run(hir);

    REQUIRE(hir.logical_noise_prefix.empty());
    REQUIRE(hir.noise_sites.empty());
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

TEST_CASE("Logical noise prefix validation rejects a wrong-size vector", "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse("Z_ERROR(0.3) 0\nM 0\n"));
    REQUIRE(hir.ops.size() == 2);
    hir.logical_noise_prefix = {0};

    REQUIRE_THROWS_AS(clifft::sampling::plan_sampling(hir), std::invalid_argument);
}

TEST_CASE("Logical noise prefix validation rejects a DETECTOR entry that disagrees with schedule",
          "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse("Z_ERROR(0.3) 0\nM 0\nDETECTOR rec[-1]\n"));
    hir.materialize_logical_noise_prefix();
    REQUIRE(hir.ops.back().op_type() == OpType::DETECTOR);
    hir.logical_noise_prefix.back() = 0;  // was 1

    REQUIRE_THROWS_AS(clifft::sampling::plan_sampling(hir), std::invalid_argument);
}

TEST_CASE(
    "Logical noise prefix validation rejects a NOISE entry that disagrees with its site index",
    "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse("Z_ERROR(0.3) 0\nM 0\n"));
    hir.materialize_logical_noise_prefix();
    REQUIRE(hir.ops.front().op_type() == OpType::NOISE);
    hir.logical_noise_prefix.front() = 1;  // only one noise site exists; must be 0

    REQUIRE_THROWS_AS(clifft::sampling::plan_sampling(hir), std::invalid_argument);
}

TEST_CASE("Logical noise prefix validation rejects an entry above the noise site count",
          "[logical_noise_prefix]") {
    HirModule hir = clifft::trace(clifft::parse("Z_ERROR(0.3) 0\nT 0\nM 0\n"));
    hir.materialize_logical_noise_prefix();
    const auto t_gate = std::ranges::find_if(
        hir.ops, [](const HeisenbergOp& op) { return op.op_type() == OpType::T_GATE; });
    REQUIRE(t_gate != hir.ops.end());
    const size_t t_index = static_cast<size_t>(t_gate - hir.ops.begin());
    hir.logical_noise_prefix[t_index] = 2;  // only one noise site exists

    REQUIRE_THROWS_AS(clifft::sampling::plan_sampling(hir), std::invalid_argument);
}

}  // namespace
