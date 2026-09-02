// Tests for search_width_schedule: the exact, budgeted search for the
// minimum peak active width over the legal schedules of an HIR.
//
// The core cross-check is brute force agreement: on circuits small enough to
// enumerate every linear extension directly, the closure-based search (which
// only ever branches on expanding ops) must still land on the same optimum
// as trying every legal order. That is the closure theorem
// clifft/optimizer/active_width_closure.h documents, exercised rather than
// just asserted.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/schedule_dependence.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/tableau/pauli_string.h"

#include "active_width_search.h"
#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace clifft;
using namespace clifft::research;
using namespace clifft::test;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

// Non-Clifford R_XX(0.3), R_ZY(0.3), M_YY (record 0), M_YI (record 1) --
// the same circuit test_schedule_dependence.cc's four-operation regression
// uses. Its original order has peak active width 2; reordering to
// [1, 2, 0, 3] drops the peak to 1.
HirModule make_four_op_circuit() {
    HirModule hir(2, 4);
    hir.num_measurements = 2;
    append_phase_rotation(hir, X(0) | X(1), 0, false, 0.3);                  // R_XX
    append_phase_rotation(hir, X(1), Z(0) | Z(1), false, 0.3);               // R_ZY
    append_measure(hir, X(0) | X(1), Z(0) | Z(1), false, MeasRecordIdx{0});  // M_YY
    append_measure(hir, X(0), Z(0), false, MeasRecordIdx{1});                // M_YI
    return hir;
}

// Two copies of make_four_op_circuit()'s pattern on disjoint qubit pairs
// (0, 1) and (2, 3). Each pair alone resolves in a single search node (its
// one accepted branch's own closure sweep finishes it, since both of its
// remaining ops are MEASURE and therefore never expanding); with both pairs
// present, closing out one pair's branch leaves the other pair entirely
// unexecuted (disjoint qubits mean neither pair's ops ever depend on or
// close out the other's), so resolving the whole circuit needs at least one
// further node. That gap is what the budget test below depends on: a
// single-pair circuit would already finish within one node regardless of
// the budget, so it could never demonstrate a budget running out.
HirModule make_two_independent_widgets_circuit() {
    HirModule hir(4, 8);
    hir.num_measurements = 4;
    append_phase_rotation(hir, X(0) | X(1), 0, false, 0.3);                  // R_XX on 0,1
    append_phase_rotation(hir, X(1), Z(0) | Z(1), false, 0.3);               // R_ZY on 0,1
    append_measure(hir, X(0) | X(1), Z(0) | Z(1), false, MeasRecordIdx{0});  // M_YY on 0,1
    append_measure(hir, X(0), Z(0), false, MeasRecordIdx{1});                // M_YI on 0,1
    append_phase_rotation(hir, X(2) | X(3), 0, false, 0.3);                  // R_XX on 2,3
    append_phase_rotation(hir, X(3), Z(2) | Z(3), false, 0.3);               // R_ZY on 2,3
    append_measure(hir, X(2) | X(3), Z(2) | Z(3), false, MeasRecordIdx{2});  // M_YY on 2,3
    append_measure(hir, X(2), Z(2), false, MeasRecordIdx{3});                // M_YI on 2,3
    return hir;
}

// Traced and optimized the same way width_certificate's --pipeline
// production runs it (PeepholeFusionPass then StatevectorSqueezePass),
// spelled out here rather than via default_hir_pass_manager() so this file
// and the driver agree on "production" by construction rather than by
// coincidentally matching the library's current default-pass set.
HirModule coherent_d3_r3_production() {
    const Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
    HirModule hir = clifft::trace(circuit);
    PeepholeFusionPass{}.run(hir);
    StatevectorSqueezePass{}.run(hir);
    return hir;
}

std::vector<uint32_t> identity_order(size_t n) {
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), uint32_t{0});
    return order;
}

// ---------------------------------------------------------------------------
// Brute force oracle: peak active width of an explicit op order, and
// exhaustive enumeration of every linear extension of a dependence relation.
// ---------------------------------------------------------------------------

// Peak active width of hir.ops replayed in `order` (a permutation of
// 0..hir.ops.size()-1), without copying or reordering the HIR. Mirrors
// analyze_active_width's rotation/measurement dispatch directly against
// DormantSubspace's public surface; this file's circuit generator never
// emits INSTRUMENT, so that branch is not needed here (mirroring
// test_schedule_dependence.cc's final_subspace helper, which makes the same
// simplification for the same reason).
uint32_t peak_width_for_order(const HirModule& hir, std::span<const uint32_t> order) {
    DormantSubspace subspace(hir.num_qubits);
    uint32_t peak = subspace.active_width();
    for (uint32_t idx : order) {
        const HeisenbergOp& op = hir.ops[idx];
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
        peak = std::max(peak, subspace.active_width());
    }
    return peak;
}

struct BruteForceState {
    const HirModule& hir;
    const ScheduleDependence& dep;
    std::vector<uint32_t> remaining_preds;
    std::vector<bool> executed;
    std::vector<uint32_t> order;
    uint32_t best_peak = std::numeric_limits<uint32_t>::max();
    size_t leaves = 0;
    size_t leaf_cap = 0;
    bool capped = false;
};

void brute_force_recurse(BruteForceState& state) {
    const size_t n = state.dep.num_ops();
    if (state.order.size() == n) {
        ++state.leaves;
        if (state.leaves > state.leaf_cap) {
            state.capped = true;
            return;
        }
        state.best_peak = std::min(state.best_peak, peak_width_for_order(state.hir, state.order));
        return;
    }
    for (uint32_t op = 0; op < n; ++op) {
        if (state.capped) {
            return;
        }
        if (state.executed[op] || state.remaining_preds[op] != 0) {
            continue;
        }
        state.executed[op] = true;
        state.order.push_back(op);
        for (uint32_t succ : state.dep.successors(op)) {
            --state.remaining_preds[succ];
        }

        brute_force_recurse(state);

        for (uint32_t succ : state.dep.successors(op)) {
            ++state.remaining_preds[succ];
        }
        state.order.pop_back();
        state.executed[op] = false;
    }
}

struct BruteForceResult {
    uint32_t min_peak = 0;
    // False when the leaf cap below was hit before every linear extension
    // was tried -- an escape hatch alongside the movable-op-count skip for
    // the rare small-op-count circuit whose movable ops are nonetheless
    // mostly mutually independent (for example several same-axis rotations
    // on one qubit with nothing interleaved to constrain their order).
    bool completed = false;
};

BruteForceResult brute_force_min_peak(const HirModule& hir, const ScheduleDependence& dep,
                                      size_t leaf_cap) {
    BruteForceState state{hir,
                          dep,
                          std::vector<uint32_t>(dep.num_ops()),
                          std::vector<bool>(dep.num_ops(), false),
                          {},
                          std::numeric_limits<uint32_t>::max(),
                          0,
                          leaf_cap,
                          false};
    for (uint32_t op = 0; op < dep.num_ops(); ++op) {
        state.remaining_preds[op] = static_cast<uint32_t>(dep.predecessors(op).size());
    }
    brute_force_recurse(state);
    return {state.best_peak, !state.capped};
}

}  // namespace

// ---------------------------------------------------------------------------
// Brute force agreement
// ---------------------------------------------------------------------------

TEST_CASE("Search finds the same optimum as brute force enumeration on small circuits",
          "[width_search]") {
    constexpr uint32_t kSeed = 0x77A5F17;
    constexpr int kTrials = 150;
    constexpr uint32_t kMaxMovableOps = 9;
    // Nine fully independent movable ops alone already permute 9! = 362880
    // ways, so this cap (not the movable-op count) is what actually bounds
    // enumeration cost; it exists purely to keep this test fast; see
    // BruteForceResult::completed.
    constexpr size_t kLeafCap = 3000;

    std::mt19937 rng(kSeed);
    int checked = 0;
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 2 + static_cast<uint32_t>(trial % 3);
        const uint32_t num_ops = 6 + static_cast<uint32_t>(trial % 9);
        const std::string source = generate_noisy_source(rng, num_qubits, num_ops);

        const HirModule hir = clifft::trace(clifft::parse(source));
        uint32_t movable_count = 0;
        for (const HeisenbergOp& op : hir.ops) {
            const OpType type = op.op_type();
            if (type == OpType::T_GATE || type == OpType::PHASE_ROTATION ||
                type == OpType::MEASURE) {
                ++movable_count;
            }
        }
        if (movable_count > kMaxMovableOps) {
            continue;
        }

        for (const bool noise_transparent : {false, true}) {
            CAPTURE(trial, source, noise_transparent);

            ScheduleDependenceOptions options;
            options.noise_transparent = noise_transparent;
            const ScheduleDependence dep = ScheduleDependence::build(hir, options);

            const BruteForceResult brute = brute_force_min_peak(hir, dep, kLeafCap);
            if (!brute.completed) {
                continue;
            }

            const WidthSearchResult result = search_width_schedule(hir, dep);
            REQUIRE(result.upper_bound == brute.min_peak);
            REQUIRE(result.lower_bound == brute.min_peak);
            REQUIRE(result.optimal());
            REQUIRE(dep.is_linear_extension(result.best_order));
            REQUIRE(peak_width_for_order(hir, result.best_order) == brute.min_peak);
            ++checked;
        }
    }

    // A sanity floor on how much the movable-op-count and leaf-count skips
    // actually let through, so a parameter change cannot silently degrade
    // this test into checking almost nothing.
    REQUIRE(checked >= 60);
}

// ---------------------------------------------------------------------------
// Four-operation regression
// ---------------------------------------------------------------------------

TEST_CASE("Search finds the optimal schedule for the four operation circuit", "[width_search]") {
    const HirModule hir = make_four_op_circuit();
    REQUIRE(analyze_active_width(hir).peak_width == 2);

    const ScheduleDependence dep = ScheduleDependence::build(hir);
    const WidthSearchResult result = search_width_schedule(hir, dep);

    REQUIRE(result.incumbent_peak == 2);
    REQUIRE(result.upper_bound == 1);
    REQUIRE(result.lower_bound == 1);
    REQUIRE(result.optimal());

    HirModule reordered = hir;
    apply_schedule(reordered, dep, result.best_order);
    const ActiveWidthTrace reordered_trace = analyze_active_width(reordered);
    std::vector<uint32_t> widths{reordered_trace.initial_width};
    for (const WidthTransition& transition : reordered_trace.transitions) {
        widths.push_back(transition.after);
    }
    REQUIRE(widths == std::vector<uint32_t>{0, 1, 1, 1, 0});
}

// ---------------------------------------------------------------------------
// Never worse than the incumbent
// ---------------------------------------------------------------------------

TEST_CASE("Search never reports a worse upper bound than the incumbent", "[width_search]") {
    constexpr uint32_t kSeed = 0x0B4D9C1;
    constexpr int kTrials = 60;

    std::mt19937 rng(kSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 7);
        const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
        const std::string source = generate_noisy_source(rng, num_qubits, num_ops);
        const HirModule hir = clifft::trace(clifft::parse(source));

        for (const bool noise_transparent : {false, true}) {
            CAPTURE(trial, source, noise_transparent);

            ScheduleDependenceOptions options;
            options.noise_transparent = noise_transparent;
            const ScheduleDependence dep = ScheduleDependence::build(hir, options);
            const WidthSearchResult result = search_width_schedule(hir, dep);

            REQUIRE(result.upper_bound <= result.incumbent_peak);
            if (result.upper_bound == result.incumbent_peak) {
                REQUIRE(result.best_order == identity_order(hir.ops.size()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Budget behaviour
// ---------------------------------------------------------------------------

TEST_CASE("An exhausted node budget is reported rather than thrown", "[width_search]") {
    const HirModule hir = make_two_independent_widgets_circuit();
    const ScheduleDependence dep = ScheduleDependence::build(hir);

    WidthSearchOptions options;
    options.node_budget = 1;
    WidthSearchResult result;
    REQUIRE_NOTHROW(result = search_width_schedule(hir, dep, options));

    REQUIRE(result.budget_exhausted);
    REQUIRE(result.upper_bound == result.incumbent_peak);
    REQUIRE(result.lower_bound == analyze_active_width(hir).final_width);
}

// ---------------------------------------------------------------------------
// Fixture certificates
// ---------------------------------------------------------------------------

TEST_CASE("Search certifies coherent_d3_r3's peak under both relation options", "[width_search]") {
    const HirModule hir = coherent_d3_r3_production();
    WidthSearchOptions options;
    options.node_budget = 200000;

    SECTION("without noise transparency the production peak is already optimal") {
        ScheduleDependenceOptions dep_options;
        dep_options.noise_transparent = false;
        const ScheduleDependence dep = ScheduleDependence::build(hir, dep_options);
        const WidthSearchResult result = search_width_schedule(hir, dep, options);

        INFO("explored_nodes=" << result.explored_nodes
                               << " budget_exhausted=" << result.budget_exhausted);
        REQUIRE(result.optimal());
        REQUIRE(result.incumbent_peak == 5);
        REQUIRE(result.upper_bound == 5);
        REQUIRE(result.lower_bound == 5);
    }

    SECTION("noise transparency lets the search certify a lower peak") {
        ScheduleDependenceOptions dep_options;
        dep_options.noise_transparent = true;
        const ScheduleDependence dep = ScheduleDependence::build(hir, dep_options);
        const WidthSearchResult result = search_width_schedule(hir, dep, options);

        INFO("explored_nodes=" << result.explored_nodes
                               << " budget_exhausted=" << result.budget_exhausted);
        REQUIRE(result.optimal());
        REQUIRE(result.incumbent_peak == 5);
        REQUIRE(result.upper_bound == 4);
        REQUIRE(result.lower_bound == 4);
    }
}

// ---------------------------------------------------------------------------
// Sampling equivalence of the found schedule
// ---------------------------------------------------------------------------

TEST_CASE("Applying the search's noise-transparent schedule preserves sampling statistics",
          "[width_search]") {
    constexpr uint32_t kShots = 20000;

    SECTION("coherent_d3_r3 fixture") {
        const HirModule hir = coherent_d3_r3_production();

        ScheduleDependenceOptions options;
        options.noise_transparent = true;
        const ScheduleDependence dep = ScheduleDependence::build(hir, options);
        const WidthSearchResult result = search_width_schedule(hir, dep);

        HirModule reordered = hir;
        apply_schedule(reordered, dep, result.best_order);
        check_sampling_equivalent(hir, reordered, kShots, 0x51D1, 0x51D2);
    }

    SECTION("random noisy circuits") {
        constexpr int kTrials = 10;
        std::mt19937 circuit_rng(0x5A17C2);
        std::mt19937 control_rng(0x5EED172);
        for (int trial = 0; trial < kTrials; ++trial) {
            const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 7);
            const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
            const std::string source = generate_noisy_source(circuit_rng, num_qubits, num_ops);
            CAPTURE(trial, source);

            const HirModule hir = clifft::trace(clifft::parse(source));
            ScheduleDependenceOptions options;
            options.noise_transparent = true;
            const ScheduleDependence dep = ScheduleDependence::build(hir, options);
            const WidthSearchResult result = search_width_schedule(hir, dep);

            HirModule reordered = hir;
            apply_schedule(reordered, dep, result.best_order);
            check_sampling_equivalent(hir, reordered, kShots, control_rng(), control_rng());
        }
    }
}
