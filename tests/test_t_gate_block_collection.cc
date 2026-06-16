#include "clifft/backend/backend.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/exact_phase_polynomial_t_count_pass.h"
#include "clifft/optimizer/t_gate_block_collection_pass.h"
#include "clifft/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using namespace clifft;
using clifft::test::append_phase_rotation;
using clifft::test::append_tgate;
using clifft::test::check_complex;

namespace {

void append_z_parity_t(HirModule& hir, uint32_t parity) {
    append_tgate(hir, 0, parity, false);
}

void append_z_separator(HirModule& hir) {
    append_phase_rotation(hir, 0, 1, false, 0.125);
}

std::vector<std::complex<double>> statevector_after_lower_execute(const HirModule& hir) {
    CompiledModule program = lower(hir);
    SchrodingerState state({.peak_rank = program.peak_rank,
                            .num_measurements = program.total_meas_slots,
                            .num_qubits = program.num_qubits,
                            .seed = 42});
    execute(program, state);
    return get_statevector(program, state);
}

void require_statevector_equal(const HirModule& actual, const HirModule& expected) {
    const auto actual_sv = statevector_after_lower_execute(actual);
    const auto expected_sv = statevector_after_lower_execute(expected);
    REQUIRE(actual_sv.size() == expected_sv.size());
    for (size_t i = 0; i < actual_sv.size(); ++i) {
        CAPTURE(i);
        check_complex(actual_sv[i], expected_sv[i], 1e-9);
    }
}

}  // namespace

TEST_CASE("T-gate block collection exposes separated rank-4 zero word", "[optimizer]") {
    HirModule original(4, 30);
    HirModule hir(4, 30);
    for (uint32_t parity = 1; parity < 16; ++parity) {
        append_z_parity_t(original, parity);
        append_z_parity_t(hir, parity);
        if (parity != 15) {
            append_z_separator(original);
            append_z_separator(hir);
        }
    }

    ExactPhasePolynomialTCountPass exact_without_collection;
    exact_without_collection.run(hir);
    REQUIRE(hir.num_t_gates() == 15);
    REQUIRE(exact_without_collection.blocks_optimized() == 0);

    TGateBlockCollectionPass collect;
    collect.run(hir);
    REQUIRE(collect.blocks_collected() == 1);
    REQUIRE(collect.t_gates_moved() == 14);
    REQUIRE(hir.num_t_gates() == 15);

    ExactPhasePolynomialTCountPass exact_after_collection;
    exact_after_collection.run(hir);
    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(exact_after_collection.blocks_optimized() == 1);
    REQUIRE(exact_after_collection.t_removed() == 15);
    require_statevector_equal(hir, original);
}

TEST_CASE("T-gate block collection preserves HIR source-map order under swaps", "[optimizer]") {
    HirModule hir(1, 3);
    append_tgate(hir, 0, 1, false);
    append_phase_rotation(hir, 0, 1, false, 0.125);
    append_tgate(hir, 0, 1, false);
    hir.source_map = {{1}, {2}, {3}};

    TGateBlockCollectionPass collect;
    collect.run(hir);

    REQUIRE(collect.t_gates_moved() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[2].op_type() == OpType::PHASE_ROTATION);
    REQUIRE(hir.source_map == std::vector<std::vector<uint32_t>>{{1}, {3}, {2}});
}

TEST_CASE("T-gate block collection stops at noncommuting barriers", "[optimizer]") {
    HirModule hir(1, 3);
    append_tgate(hir, 0, 1, false);
    append_phase_rotation(hir, 1, 0, false, 0.125);
    append_tgate(hir, 0, 1, false);

    TGateBlockCollectionPass collect;
    collect.run(hir);

    REQUIRE(collect.blocks_collected() == 0);
    REQUIRE(collect.t_gates_moved() == 0);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::PHASE_ROTATION);
    REQUIRE(hir.ops[2].op_type() == OpType::T_GATE);
}

TEST_CASE("T-gate block collection preflights only candidate swaps", "[optimizer]") {
    HirModule hir(1, 2, 1);
    append_tgate(hir, 0, 1, false);

    NoiseSite site;
    auto noise_mask = clifft::test::claim_noise_channel_mask(hir, 0, 1);
    site.channels.push_back({noise_mask, 0.1});
    hir.noise_sites.push_back(std::move(site));
    hir.append_noise(NoiseSiteIdx{0});

    hir.observable_targets.push_back({});
    hir.append_observable(ObservableIdx{0}, 0);
    append_tgate(hir, 0, 1, false);

    TGateBlockCollectionPass collect;
    collect.run(hir);

    REQUIRE(collect.blocks_collected() == 1);
    REQUIRE(collect.t_gates_moved() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[2].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[3].op_type() == OpType::OBSERVABLE);
}
