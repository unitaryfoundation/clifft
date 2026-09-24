// Checks scheduling constraints against the full pairwise relation and verifies
// that applying legal orders preserves metadata and sampling behavior.

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
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace clifft;
using namespace clifft::test;
using clifft::detail::apply_schedule;
using clifft::detail::ScheduleDependence;
using clifft::detail::ScheduleDependenceOptions;
using clifft::sampling::SamplingPlan;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// Reference rules for checking the reduced graph against all operation pairs.
bool is_movable_ref(OpType type) {
    return type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::MEASURE;
}

bool allowed_ref(const HirModule& hir, const HeisenbergOp& left, const HeisenbergOp& right,
                 bool noise_transparent) {
    if (can_swap(left, right, hir)) {
        return true;
    }
    if (!noise_transparent) {
        return false;
    }
    const bool left_noise = left.op_type() == OpType::NOISE;
    const bool right_noise = right.op_type() == OpType::NOISE;
    const bool left_movable = is_movable_ref(left.op_type());
    const bool right_movable = is_movable_ref(right.op_type());
    return (left_noise && right_movable) || (right_noise && left_movable);
}

// Recompute the fixed-op chain independently of the graph under test.
std::vector<std::optional<uint32_t>> compute_prev_fixed_of(const HirModule& hir) {
    std::vector<std::optional<uint32_t>> prev_fixed_of(hir.ops.size());
    std::optional<uint32_t> previous_fixed;
    for (uint32_t k = 0; k < hir.ops.size(); ++k) {
        if (is_movable_ref(hir.ops[k].op_type())) {
            continue;
        }
        prev_fixed_of[k] = previous_fixed;
        previous_fixed = k;
    }
    return prev_fixed_of;
}

std::string random_noisy_source(clifft::Xoshiro256PlusPlus& rng, int trial) {
    const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 7);
    const uint32_t num_ops = 15 + static_cast<uint32_t>(trial % 25);
    return clifft::test::generate_noisy_source(rng, num_qubits, num_ops);
}

// Choose a random ready op at each step of Kahn's algorithm.
std::vector<uint32_t> random_linear_extension(const ScheduleDependence& dep,
                                              clifft::Xoshiro256PlusPlus& rng) {
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

// Replay the generator's rotations and measurements to compare final
// subspaces. It emits no instruments, so other ops leave the subspace unchanged.
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

// Build the full pairwise relation without edge reduction. ref[j][i] records
// whether i must precede j, including constraints inherited through earlier ops.
std::vector<std::vector<bool>> reference_ancestor_closure(const HirModule& hir,
                                                          bool noise_transparent) {
    const size_t n = hir.ops.size();
    const std::vector<std::optional<uint32_t>> prev_fixed_of = compute_prev_fixed_of(hir);

    std::vector<std::vector<bool>> ref(n, std::vector<bool>(n, false));
    for (size_t j = 0; j < n; ++j) {
        const bool j_movable = is_movable_ref(hir.ops[j].op_type());
        for (size_t i = 0; i < j; ++i) {
            const bool i_movable = is_movable_ref(hir.ops[i].op_type());
            const bool constrained =
                (!i_movable && !j_movable)
                    ? (prev_fixed_of[j].has_value() && *prev_fixed_of[j] == i)
                    : !allowed_ref(hir, hir.ops[i], hir.ops[j], noise_transparent);
            if (!constrained) {
                continue;
            }
            ref[j][i] = true;
            for (size_t k = 0; k < n; ++k) {
                if (ref[i][k]) {
                    ref[j][k] = true;
                }
            }
        }
    }
    return ref;
}

// Compute the reduced graph's closure for comparison with the pairwise reference.
std::vector<std::vector<bool>> built_ancestor_closure(const ScheduleDependence& dep) {
    const size_t n = dep.num_ops();
    std::vector<std::vector<bool>> built(n, std::vector<bool>(n, false));
    for (size_t j = 0; j < n; ++j) {
        for (uint32_t i : dep.predecessors(j)) {
            built[j][i] = true;
            for (size_t k = 0; k < n; ++k) {
                if (built[i][k]) {
                    built[j][k] = true;
                }
            }
        }
    }
    return built;
}

// Require the same closure as the pairwise reference, no spurious edges,
// and direct links between consecutive fixed ops, for any cache budget.
void check_closure_matches_can_swap(const HirModule& hir, bool noise_transparent,
                                    ScheduleDependenceOptions options = {}) {
    options.noise_transparent = noise_transparent;
    const ScheduleDependence dep = ScheduleDependence::build(hir, options);

    REQUIRE(dep.num_ops() == hir.ops.size());
    REQUIRE(dep.noise_transparent() == noise_transparent);
    for (size_t i = 0; i < hir.ops.size(); ++i) {}

    const std::vector<std::vector<bool>> reference =
        reference_ancestor_closure(hir, noise_transparent);
    const std::vector<std::vector<bool>> built = built_ancestor_closure(dep);
    REQUIRE(built == reference);

    const std::vector<std::optional<uint32_t>> prev_fixed_of = compute_prev_fixed_of(hir);

    // Reduction must not introduce constraints absent from the reference.
    for (size_t j = 0; j < hir.ops.size(); ++j) {
        const bool j_movable = is_movable_ref(hir.ops[j].op_type());
        for (uint32_t i : dep.predecessors(j)) {
            const bool i_movable = is_movable_ref(hir.ops[i].op_type());
            CAPTURE(i, j);
            if (!i_movable && !j_movable) {
                REQUIRE(prev_fixed_of[j].has_value());
                REQUIRE(*prev_fixed_of[j] == i);
            } else {
                REQUIRE_FALSE(allowed_ref(hir, hir.ops[i], hir.ops[j], noise_transparent));
            }
        }
    }

    for (uint32_t k = 0; k < hir.ops.size(); ++k) {
        if (prev_fixed_of[k].has_value()) {
            CAPTURE(*prev_fixed_of[k], k);
            REQUIRE(std::ranges::binary_search(dep.predecessors(k), *prev_fixed_of[k]));
        }
    }
}

// Different legal orders must produce the same dormant subspace. Compare
// mutual containment because equivalent bases can have different generators.
void check_confluence(const HirModule& hir, bool noise_transparent,
                      clifft::Xoshiro256PlusPlus& order_rng) {
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

TEST_CASE("Schedule dependence closure matches can_swap on random noisy circuits",
          "[schedule_dependence]") {
    constexpr uint32_t kSeed = 0x5C4ED;
    constexpr int kTrials = 200;

    clifft::Xoshiro256PlusPlus rng(kSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const std::string source = random_noisy_source(rng, trial);
        CAPTURE(trial, source);

        const HirModule hir = clifft::trace(clifft::parse(source));
        check_closure_matches_can_swap(hir, /*noise_transparent=*/false);
        check_closure_matches_can_swap(hir, /*noise_transparent=*/true);
    }
}

TEST_CASE("Schedule dependence closure matches can_swap with an evicting ancestor cache",
          "[schedule_dependence]") {
    constexpr uint32_t kSeed = 0x5C4ED2;
    constexpr int kTrials = 60;

    clifft::Xoshiro256PlusPlus rng(kSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const std::string source = random_noisy_source(rng, trial);
        const HirModule hir = clifft::trace(clifft::parse(source));
        const size_t words = (hir.ops.size() + 63) / 64;

        for (const bool noise_transparent : {false, true}) {
            // Exercise both complete and partial eviction of cached ancestor rows.
            for (const size_t rows : {size_t{1}, size_t{3}}) {
                ScheduleDependenceOptions options;
                options.ancestor_cache_bytes = rows * words * sizeof(uint64_t);
                CAPTURE(trial, source, noise_transparent, rows);
                check_closure_matches_can_swap(hir, noise_transparent, options);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chain reduction has teeth
// ---------------------------------------------------------------------------

TEST_CASE("An anticommuting measurement chain reduces to N minus one edges",
          "[schedule_dependence]") {
    constexpr int kMeasurements = 40;

    std::string source;
    for (int t = 0; t < kMeasurements; ++t) {
        source += (t % 2 == 0) ? "MX 0\n" : "MZ 0\n";
    }

    const HirModule hir = clifft::trace(clifft::parse(source));
    // Exclude extra frontend ops that could change the expected edge count.
    REQUIRE(hir.ops.size() == static_cast<size_t>(kMeasurements));
    for (const HeisenbergOp& op : hir.ops) {
        REQUIRE(op.op_type() == OpType::MEASURE);
    }

    const ScheduleDependence dep = ScheduleDependence::build(hir);

    // Alternating X and Z measurements have quadratically many conflicts,
    // but the consecutive chain implies them all with kMeasurements - 1 edges.
    size_t total_successors = 0;
    for (size_t op = 0; op < dep.num_ops(); ++op) {
        total_successors += dep.successors(op).size();
    }
    REQUIRE(total_successors == static_cast<size_t>(kMeasurements - 1));

    for (uint32_t t = 0; t + 1 < static_cast<uint32_t>(kMeasurements); ++t) {
        CAPTURE(t);
        REQUIRE(std::ranges::binary_search(dep.successors(t), t + 1));
    }

    std::vector<uint32_t> identity_order(dep.num_ops());
    for (size_t i = 0; i < identity_order.size(); ++i) {
        identity_order[i] = static_cast<uint32_t>(i);
    }
    REQUIRE(dep.is_linear_extension(identity_order));

    std::vector<uint32_t> swapped_order = identity_order;
    std::swap(swapped_order[0], swapped_order[1]);
    REQUIRE_FALSE(dep.is_linear_extension(swapped_order));
}

// ---------------------------------------------------------------------------
// Random linear extensions
// ---------------------------------------------------------------------------

TEST_CASE("A random linear extension keeps fixed ops in order and rejects an inversion",
          "[schedule_dependence]") {
    constexpr uint32_t kCircuitSeed = 0x11FE1;
    constexpr uint32_t kOrderSeed = 0x11FE2;
    constexpr int kTrials = 60;

    clifft::Xoshiro256PlusPlus circuit_rng(kCircuitSeed);
    clifft::Xoshiro256PlusPlus order_rng(kOrderSeed);
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

        // Reversing an edge's endpoints must invalidate an otherwise legal order.
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

    clifft::Xoshiro256PlusPlus circuit_rng(kCircuitSeed);
    clifft::Xoshiro256PlusPlus order_rng(kOrderSeed);
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
    // Use the same Pauli bodies as the DormantSubspace confluence regression.
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

        // Without sign correction, X noise cannot cross a Z measurement.
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
}

TEST_CASE("apply_schedule carries source_map and logical_noise_prefix with their ops",
          "[schedule_dependence]") {
    clifft::Xoshiro256PlusPlus circuit_rng(0xF00D1);
    const std::string source = clifft::test::generate_noisy_source(circuit_rng, 6, 30);
    CAPTURE(source);

    HirModule hir = clifft::trace(clifft::parse(source));
    REQUIRE(hir.source_map.size() == hir.ops.size());
    hir.materialize_logical_noise_prefix();

    // Source lines uniquely tag these ops, letting us check that logical
    // noise positions stay paired with the same op after permutation.
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
    clifft::Xoshiro256PlusPlus order_rng(0xBEEF1);
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
        clifft::Xoshiro256PlusPlus circuit_rng(0x5A17C);
        clifft::Xoshiro256PlusPlus control_rng(0x5EED17);
        for (int trial = 0; trial < kTrials; ++trial) {
            const std::string source = random_noisy_source(circuit_rng, trial);
            CAPTURE(trial, source);

            const HirModule original = clifft::trace(clifft::parse(source));
            ScheduleDependenceOptions options;
            options.noise_transparent = true;
            const ScheduleDependence dep = ScheduleDependence::build(original, options);
            clifft::Xoshiro256PlusPlus order_rng(control_rng());
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
        clifft::Xoshiro256PlusPlus order_rng(0x517EE7);
        const std::vector<uint32_t> order = random_linear_extension(dep, order_rng);

        HirModule reordered = original;
        apply_schedule(reordered, dep, order);

        clifft::test::check_sampling_equivalent(original, reordered, kShots, 0x1234, 0x5678);
    }
}

TEST_CASE(
    "A random linear extension under noise transparency is exactly sampling equivalent for "
    "every checked noise realization",
    "[schedule_dependence]") {
    constexpr uint32_t kCircuitSeed = 0xDEC0DE1;
    constexpr uint32_t kOrderSeed = 0xDEC0DE2;
    constexpr uint32_t kControlSeed = 0xDEC0DE3;
    constexpr int kTrials = 200;

    clifft::Xoshiro256PlusPlus circuit_rng(kCircuitSeed);
    clifft::Xoshiro256PlusPlus order_rng(kOrderSeed);
    clifft::Xoshiro256PlusPlus control_rng(kControlSeed);
    int skipped = 0;
    int crossed_count = 0;
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 3 + static_cast<uint32_t>(trial % 4);
        const uint32_t num_ops = 12 + static_cast<uint32_t>(trial % 13);
        const std::string source =
            clifft::test::generate_noisy_source(circuit_rng, num_qubits, num_ops);
        const HirModule original = clifft::trace(clifft::parse(source));
        CAPTURE(trial, num_qubits, num_ops, source);
        // Bound the exponential record enumeration. The exact checker also
        // requires no hidden measurements or readout noise.
        if (original.num_measurements > 8 || original.num_hidden_measurements > 0 ||
            !original.readout_noise.empty()) {
            ++skipped;
            continue;
        }

        ScheduleDependenceOptions options;
        options.noise_transparent = true;
        const ScheduleDependence dep = ScheduleDependence::build(original, options);
        const std::vector<uint32_t> order = random_linear_extension(dep, order_rng);

        HirModule reordered = original;
        apply_schedule(reordered, dep, order);
        crossed_count += clifft::test::crossed_noise(reordered) ? 1 : 0;

        clifft::test::check_exact_equivalent(original, reordered, control_rng);
    }

    INFO("skipped=" << skipped << " crossed=" << crossed_count);
    REQUIRE(crossed_count >= 10);
}
