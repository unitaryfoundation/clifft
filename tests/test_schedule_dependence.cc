// Tests for ScheduleDependence and apply_schedule: the conservative
// dependence relation over HIR operations that bounds which reorderings a
// scheduler may use, and the machinery that commits one such reordering
// back into an HirModule.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/schedule_dependence.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/tableau/pauli_string.h"

#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace clifft;
using namespace clifft::test;
using clifft::sampling::SamplingPlan;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// Independent copy of the movable-op classification from
// schedule_dependence.cc, so the oracle tests below do not lean on the
// class under test to define what "movable" means.
bool is_movable_ref(OpType type) {
    return type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::MEASURE;
}

// Deterministic generator over a small gate set: T, T_DAG, R_Z, M, MX, MR,
// R, X_ERROR, DEPOLARIZE1, and DETECTORs. Wraps the shared generator with
// this file's qubit-count convention (4 to 10, per the relation's movable
// op mix) so every call site does not have to repeat the modulus.
std::string random_noisy_source(std::mt19937& rng, int trial) {
    const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 7);
    const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
    return clifft::test::generate_noisy_source(rng, num_qubits, num_ops);
}

// Draws a uniformly random linear extension by repeatedly picking a
// uniformly random ready op (one with no unscheduled predecessor) -- a
// randomized Kahn's algorithm. Every ScheduleDependence is acyclic by
// construction (every edge i -> j has i < j), so this always terminates
// with a full permutation.
std::vector<uint32_t> random_linear_extension(const ScheduleDependence& dep, std::mt19937& rng) {
    const size_t n = dep.num_ops();
    std::vector<uint32_t> remaining_preds(n);
    std::vector<uint32_t> ready;
    for (size_t op = 0; op < n; ++op) {
        remaining_preds[op] = static_cast<uint32_t>(dep.predecessors(op).size());
        if (remaining_preds[op] == 0) {
            ready.push_back(static_cast<uint32_t>(op));
        }
    }

    std::vector<uint32_t> order;
    order.reserve(n);
    while (!ready.empty()) {
        const size_t pick = rng() % ready.size();
        const uint32_t op = ready[pick];
        ready[pick] = ready.back();
        ready.pop_back();
        order.push_back(op);
        for (uint32_t succ : dep.successors(op)) {
            if (--remaining_preds[succ] == 0) {
                ready.push_back(succ);
            }
        }
    }
    return order;
}

// Replays a T_GATE/PHASE_ROTATION/MEASURE-only walk over `hir` through a
// fresh DormantSubspace, using the same per-op-type dispatch
// analyze_active_width uses (active_width_analysis.cc). Every other op type
// leaves S untouched there, which is all this file's generator ever
// produces (it never emits INSTRUMENT, the one op type with a nontrivial
// "leaves S untouched" story of its own).
DormantSubspace final_subspace(const HirModule& hir) {
    DormantSubspace subspace(hir.num_qubits);
    for (const HeisenbergOp& op : hir.ops) {
        const bool is_rotation =
            op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
        const bool is_measurement = op.op_type() == OpType::MEASURE;
        if (!is_rotation && !is_measurement) {
            continue;
        }
        PauliString body(hir.num_qubits);
        body.mut_x().xor_with(hir.destab_mask(op));
        body.mut_z().xor_with(hir.stab_mask(op));
        if (is_measurement) {
            subspace.apply_measurement(body);
        } else {
            subspace.apply_rotation(body);
        }
    }
    return subspace;
}

// For every pair with at least one movable op, the relation must place an
// edge exactly when allowed(i, j) is false: allowed is can_swap, extended
// by the noise-transparency carve-out for a NOISE-versus-movable pair.
// Fixed-fixed pairs are exempted here (checked separately below) since the
// relation only chains consecutive ones, not every pair.
void check_edges_match_can_swap(const HirModule& hir, bool noise_transparent) {
    ScheduleDependenceOptions options;
    options.noise_transparent = noise_transparent;
    const ScheduleDependence dep = ScheduleDependence::build(hir, options);

    REQUIRE(dep.num_ops() == hir.ops.size());
    REQUIRE(dep.noise_transparent() == noise_transparent);
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        REQUIRE(dep.is_movable(i) == is_movable_ref(hir.ops[i].op_type()));
    }

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const bool i_movable = is_movable_ref(hir.ops[i].op_type());
        for (size_t j = i + 1; j < hir.ops.size(); ++j) {
            const bool j_movable = is_movable_ref(hir.ops[j].op_type());
            if (!i_movable && !j_movable) {
                continue;
            }

            const bool has_edge =
                std::ranges::binary_search(dep.predecessors(j), static_cast<uint32_t>(i));
            const bool noise_pair =
                noise_transparent && ((hir.ops[i].op_type() == OpType::NOISE && j_movable) ||
                                      (hir.ops[j].op_type() == OpType::NOISE && i_movable));
            CAPTURE(i, j, noise_pair);
            if (noise_pair) {
                REQUIRE_FALSE(has_edge);
                continue;
            }
            REQUIRE(has_edge == !can_swap(hir.ops[i], hir.ops[j], hir));
        }
    }

    // Every consecutive pair of fixed ops is chained.
    std::optional<uint32_t> previous_fixed;
    for (uint32_t k = 0; k < hir.ops.size(); ++k) {
        if (is_movable_ref(hir.ops[k].op_type())) {
            continue;
        }
        if (previous_fixed.has_value()) {
            CAPTURE(*previous_fixed, k);
            REQUIRE(std::ranges::binary_search(dep.predecessors(k), *previous_fixed));
        }
        previous_fixed = k;
    }
}

// Builds both relations for `hir`, draws two independent random linear
// extensions of each, applies each to its own copy, and requires the
// resulting DormantSubspace to agree: same active width, and every
// generator of one contained in the other. This is the confluence property
// that makes searching over the relation's linear extensions well defined.
void check_confluence(const HirModule& hir, bool noise_transparent, std::mt19937& order_rng) {
    ScheduleDependenceOptions options;
    options.noise_transparent = noise_transparent;
    const ScheduleDependence dep = ScheduleDependence::build(hir, options);

    const std::vector<uint32_t> order_a = random_linear_extension(dep, order_rng);
    const std::vector<uint32_t> order_b = random_linear_extension(dep, order_rng);
    REQUIRE(dep.is_linear_extension(order_a));
    REQUIRE(dep.is_linear_extension(order_b));

    HirModule hir_a = hir;
    apply_schedule(hir_a, dep, order_a);
    HirModule hir_b = hir;
    apply_schedule(hir_b, dep, order_b);

    const ActiveWidthTrace trace_a = analyze_active_width(hir_a);
    const ActiveWidthTrace trace_b = analyze_active_width(hir_b);
    REQUIRE(trace_a.final_width == trace_b.final_width);

    const DormantSubspace subspace_a = final_subspace(hir_a);
    const DormantSubspace subspace_b = final_subspace(hir_b);
    REQUIRE(subspace_a.active_width() == trace_a.final_width);
    REQUIRE(subspace_b.active_width() == trace_b.final_width);

    const std::vector<PauliString> gens_a = subspace_a.generators();
    const std::vector<PauliString> gens_b = subspace_b.generators();
    REQUIRE(gens_a.size() == gens_b.size());
    for (const PauliString& g : gens_b) {
        REQUIRE(subspace_a.contains(g));
    }
    for (const PauliString& g : gens_a) {
        REQUIRE(subspace_b.contains(g));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Relation matches can_swap
// ---------------------------------------------------------------------------

TEST_CASE("Schedule dependence edges match can_swap on random noisy circuits",
          "[schedule_dependence]") {
    constexpr uint32_t kSeed = 0x5C4ED;
    constexpr int kTrials = 200;

    std::mt19937 rng(kSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const std::string source = random_noisy_source(rng, trial);
        CAPTURE(trial, source);

        const HirModule hir = clifft::trace(clifft::parse(source));
        check_edges_match_can_swap(hir, /*noise_transparent=*/false);
        check_edges_match_can_swap(hir, /*noise_transparent=*/true);
    }
}

// ---------------------------------------------------------------------------
// Random linear extensions
// ---------------------------------------------------------------------------

TEST_CASE("A random linear extension keeps fixed ops in order and rejects an inversion",
          "[schedule_dependence]") {
    constexpr uint32_t kCircuitSeed = 0x11FE1;
    constexpr uint32_t kOrderSeed = 0x11FE2;
    constexpr int kTrials = 60;

    std::mt19937 circuit_rng(kCircuitSeed);
    std::mt19937 order_rng(kOrderSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const std::string source = random_noisy_source(circuit_rng, trial);
        CAPTURE(trial, source);

        const HirModule hir = clifft::trace(clifft::parse(source));
        const ScheduleDependence dep = ScheduleDependence::build(hir);
        const std::vector<uint32_t> order = random_linear_extension(dep, order_rng);

        REQUIRE(order.size() == hir.ops.size());
        REQUIRE(dep.is_linear_extension(order));

        std::vector<uint32_t> position(hir.ops.size());
        for (size_t pos = 0; pos < order.size(); ++pos) {
            position[order[pos]] = static_cast<uint32_t>(pos);
        }

        std::optional<uint32_t> previous_fixed;
        for (uint32_t i = 0; i < hir.ops.size(); ++i) {
            if (is_movable_ref(hir.ops[i].op_type())) {
                continue;
            }
            if (previous_fixed.has_value()) {
                REQUIRE(position[*previous_fixed] < position[i]);
            }
            previous_fixed = i;
        }

        // A deliberately inverted dependent pair is rejected: take a real
        // edge i -> j and swap the two ops' slots in the (still valid)
        // order, which places j before i and must violate that edge.
        for (uint32_t j = 0; j < hir.ops.size(); ++j) {
            const std::span<const uint32_t> preds = dep.predecessors(j);
            if (preds.empty()) {
                continue;
            }
            const uint32_t i = preds[0];
            std::vector<uint32_t> broken = order;
            std::swap(broken[position[i]], broken[position[j]]);
            REQUIRE_FALSE(dep.is_linear_extension(broken));
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Confluence of the structural subspace
// ---------------------------------------------------------------------------

TEST_CASE("Different linear extensions reach the same final structural subspace",
          "[schedule_dependence]") {
    constexpr uint32_t kCircuitSeed = 0x9A5E1;
    constexpr uint32_t kOrderSeed = 0x9A5E2;
    constexpr int kTrials = 100;

    std::mt19937 circuit_rng(kCircuitSeed);
    std::mt19937 order_rng(kOrderSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const std::string source = random_noisy_source(circuit_rng, trial);
        const HirModule hir = clifft::trace(clifft::parse(source));

        for (const bool noise_transparent : {false, true}) {
            CAPTURE(trial, source, noise_transparent);
            check_confluence(hir, noise_transparent, order_rng);
        }
    }
}

// ---------------------------------------------------------------------------
// Four-operation regression
// ---------------------------------------------------------------------------

TEST_CASE("Reordering a four operation circuit reduces its peak active width",
          "[schedule_dependence]") {
    // Non-Clifford R_XX(0.3), R_ZY(0.3), M_YY (record 0), M_YI (record 1).
    // Mask bits follow PauliString::from_text's convention (character index
    // q maps to qubit q), matching the equivalent DormantSubspace-only
    // cases in test_active_width_analysis.cc.
    HirModule hir(2, 4);
    hir.num_measurements = 2;
    clifft::test::append_phase_rotation(hir, X(0) | X(1), 0, false, 0.3);                  // R_XX
    clifft::test::append_phase_rotation(hir, X(1), Z(0) | Z(1), false, 0.3);               // R_ZY
    clifft::test::append_measure(hir, X(0) | X(1), Z(0) | Z(1), false, MeasRecordIdx{0});  // M_YY
    clifft::test::append_measure(hir, X(0), Z(0), false, MeasRecordIdx{1});                // M_YI

    const ActiveWidthTrace original_trace = analyze_active_width(hir);
    std::vector<uint32_t> original_widths{original_trace.initial_width};
    for (const WidthTransition& transition : original_trace.transitions) {
        original_widths.push_back(transition.after);
    }
    REQUIRE(original_widths == std::vector<uint32_t>{0, 1, 2, 1, 0});

    const ScheduleDependence dep = ScheduleDependence::build(hir);
    const std::vector<uint32_t> order = {1, 2, 0, 3};
    REQUIRE(dep.is_linear_extension(order));

    HirModule reordered = hir;
    apply_schedule(reordered, dep, order);

    const ActiveWidthTrace reordered_trace = analyze_active_width(reordered);
    std::vector<uint32_t> reordered_widths{reordered_trace.initial_width};
    for (const WidthTransition& transition : reordered_trace.transitions) {
        reordered_widths.push_back(transition.after);
    }
    REQUIRE(reordered_widths == std::vector<uint32_t>{0, 1, 1, 1, 0});

    const SamplingPlan original_plan = clifft::sampling::plan_sampling(hir);
    const SamplingPlan reordered_plan = clifft::sampling::plan_sampling(reordered);
    REQUIRE(original_plan.peak_active_width == original_trace.peak_width);
    REQUIRE(reordered_plan.peak_active_width == reordered_trace.peak_width);
    REQUIRE(original_plan.peak_active_width == 2);
    REQUIRE(reordered_plan.peak_active_width == 1);
}

// ---------------------------------------------------------------------------
// apply_schedule
// ---------------------------------------------------------------------------

TEST_CASE("apply_schedule rejects an order that is not a linear extension",
          "[schedule_dependence]") {
    SECTION("inverted dependent pair") {
        HirModule hir = clifft::trace(clifft::parse("X_ERROR(0.3) 0\nM 0\nDETECTOR rec[-1]\n"));
        REQUIRE(hir.ops.size() == 3);
        REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
        REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
        REQUIRE(hir.ops[2].op_type() == OpType::DETECTOR);

        // With noise transparency off, the plain X_ERROR/Z-measurement
        // anticommutation makes NOISE -> MEASURE a real edge.
        ScheduleDependenceOptions options;
        options.noise_transparent = false;
        const ScheduleDependence dep = ScheduleDependence::build(hir, options);
        REQUIRE_FALSE(can_swap(hir.ops[0], hir.ops[1], hir));

        const std::vector<uint32_t> bad_order = {1, 0, 2};
        REQUIRE_THROWS_AS(apply_schedule(hir, dep, bad_order), std::invalid_argument);
    }

    SECTION("wrong length order") {
        HirModule hir = clifft::trace(clifft::parse("M 0\nM 0\n"));
        const ScheduleDependence dep = ScheduleDependence::build(hir);
        const std::vector<uint32_t> short_order = {0};
        REQUIRE_THROWS_AS(apply_schedule(hir, dep, short_order), std::invalid_argument);
    }

    SECTION("dependence built from a different operation count") {
        HirModule small = clifft::trace(clifft::parse("M 0\n"));
        const ScheduleDependence dep = ScheduleDependence::build(small);

        HirModule other = clifft::trace(clifft::parse("M 0\nM 0\n"));
        const std::vector<uint32_t> order = {0};
        REQUIRE_THROWS_AS(apply_schedule(other, dep, order), std::invalid_argument);
    }
}

TEST_CASE("apply_schedule carries source_map and logical_noise_prefix with their ops",
          "[schedule_dependence]") {
    std::mt19937 circuit_rng(0xF00D1);
    const std::string source = clifft::test::generate_noisy_source(circuit_rng, 6, 30);
    CAPTURE(source);

    HirModule hir = clifft::trace(clifft::parse(source));
    REQUIRE(hir.source_map.size() == hir.ops.size());
    hir.materialize_logical_noise_prefix();

    // source_map (the source line) uniquely tags each op here, so pairing
    // it with logical_noise_prefix confirms the entry followed the same op
    // through however apply_schedule permutes them.
    auto tagged_snapshot = [](const HirModule& module) {
        std::vector<std::pair<std::vector<uint32_t>, uint32_t>> tagged;
        for (size_t i = 0; i < module.ops.size(); ++i) {
            tagged.emplace_back(module.source_map[i], module.logical_noise_prefix[i]);
        }
        std::ranges::sort(tagged);
        return tagged;
    };
    const auto before = tagged_snapshot(hir);

    ScheduleDependenceOptions options;
    options.noise_transparent = true;
    const ScheduleDependence dep = ScheduleDependence::build(hir, options);
    std::mt19937 order_rng(0xBEEF1);
    const std::vector<uint32_t> order = random_linear_extension(dep, order_rng);

    apply_schedule(hir, dep, order);

    REQUIRE(hir.source_map.size() == hir.ops.size());
    REQUIRE(hir.has_logical_noise_prefix());
    REQUIRE(tagged_snapshot(hir) == before);
}

TEST_CASE("A random linear extension under noise transparency is sampling equivalent",
          "[schedule_dependence]") {
    constexpr uint32_t kShots = 20000;

    SECTION("random circuits") {
        constexpr int kTrials = 20;
        std::mt19937 circuit_rng(0x5A17C);
        std::mt19937 control_rng(0x5EED17);
        for (int trial = 0; trial < kTrials; ++trial) {
            const std::string source = random_noisy_source(circuit_rng, trial);
            CAPTURE(trial, source);

            const HirModule original = clifft::trace(clifft::parse(source));
            ScheduleDependenceOptions options;
            options.noise_transparent = true;
            const ScheduleDependence dep = ScheduleDependence::build(original, options);
            std::mt19937 order_rng(control_rng());
            const std::vector<uint32_t> order = random_linear_extension(dep, order_rng);

            HirModule reordered = original;
            apply_schedule(reordered, dep, order);

            clifft::test::check_sampling_equivalent(original, reordered, kShots, control_rng(),
                                                    control_rng());
        }
    }

    SECTION("coherent_d3_r3 fixture") {
        const Circuit circuit =
            clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
        const HirModule original = clifft::trace(circuit);

        ScheduleDependenceOptions options;
        options.noise_transparent = true;
        const ScheduleDependence dep = ScheduleDependence::build(original, options);
        std::mt19937 order_rng(0x517EE7);
        const std::vector<uint32_t> order = random_linear_extension(dep, order_rng);

        HirModule reordered = original;
        apply_schedule(reordered, dep, order);

        clifft::test::check_sampling_equivalent(original, reordered, kShots, 0x1234, 0x5678);
    }
}
