#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/exact_phase_polynomial_t_count_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

using namespace clifft;
using clifft::test::append_tgate;
using clifft::test::check_complex;
using clifft::test::dense_axis_rotation;
using clifft::test::dense_matmul;
using clifft::test::DenseMatrix;

namespace {

void append_z_parity_t(HirModule& hir, uint32_t parity, bool dagger = false, bool sign = false) {
    append_tgate(hir, 0, parity, sign, dagger);
}

void append_all_nonzero_z_parities(HirModule& hir, uint8_t rank) {
    for (uint32_t parity = 1; parity < (1u << rank); ++parity) {
        append_z_parity_t(hir, parity);
    }
}

std::pair<uint32_t, uint32_t> rank4_product_masks(uint32_t coord) {
    uint32_t x = 0;
    uint32_t z = 0;
    if ((coord & 0b0001) != 0)
        x ^= 0b0011;
    if ((coord & 0b0010) != 0)
        z ^= 0b0011;
    if ((coord & 0b0100) != 0)
        x ^= 0b1100;
    if ((coord & 0b1000) != 0)
        z ^= 0b1100;
    return {x, z};
}

bool rank4_product_sign(uint32_t coord) {
    return ((coord & 0b0011) == 0b0011) != ((coord & 0b1100) == 0b1100);
}

void append_rank4_product_t(HirModule& hir, uint32_t coord, bool use_product_sign,
                            bool dagger = false) {
    const auto [x, z] = rank4_product_masks(coord);
    append_tgate(hir, x, z, use_product_sign && rank4_product_sign(coord), dagger);
}

void append_rank4_product_t_with_sign(HirModule& hir, uint32_t coord, bool sign,
                                      bool dagger = false) {
    const auto [x, z] = rank4_product_masks(coord);
    append_tgate(hir, x, z, sign, dagger);
}

void append_rank4_product_word(HirModule& hir, bool use_product_sign) {
    for (uint32_t coord = 1; coord < 16; ++coord) {
        append_rank4_product_t(hir, coord, use_product_sign);
    }
}

DenseMatrix dense_hir_ops_value(const HirModule& hir) {
    const uint64_t dim = uint64_t{1} << hir.num_qubits;
    DenseMatrix value(dim * dim, {0.0, 0.0});
    for (uint64_t i = 0; i < dim; ++i) {
        value[i * dim + i] = {1.0, 0.0};
    }

    for (size_t i = hir.ops.size(); i-- > 0;) {
        const auto& op = hir.ops[i];
        REQUIRE((op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION));

        const uint64_t x = hir.destab_mask(op).words[0];
        const uint64_t z = hir.stab_mask(op).words[0];
        const bool sign = hir.sign(op);
        DenseMatrix rotation;
        if (op.op_type() == OpType::T_GATE) {
            rotation =
                dense_axis_rotation(x, z, sign, op.is_dagger() ? 1.75 : 0.25, hir.num_qubits);
        } else {
            rotation =
                dense_axis_rotation(x, z, false, sign ? -op.alpha() : op.alpha(), hir.num_qubits);
        }
        value = dense_matmul(value, rotation, dim);
    }

    for (auto& v : value) {
        v *= hir.global_weight;
    }
    return value;
}

void require_dense_equal(const DenseMatrix& actual, const DenseMatrix& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::abs(actual[i] - expected[i]) < 1e-9);
    }
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

TEST_CASE("Exact phase-polynomial pass removes rank-4 Reed-Muller zero word", "[optimizer]") {
    HirModule hir(4, 15);
    append_all_nonzero_z_parities(hir, 4);

    ExactPhasePolynomialTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
    REQUIRE(pass.blocks_considered() == 1);
    REQUIRE(pass.blocks_optimized() == 1);
    REQUIRE(pass.t_removed() == 15);
}

TEST_CASE("Exact phase-polynomial pass reduces rank-4 zero word plus one T", "[optimizer]") {
    HirModule hir(4, 16);
    append_all_nonzero_z_parities(hir, 4);
    append_z_parity_t(hir, 1);

    ExactPhasePolynomialTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.stab_mask(hir.ops[0]) == 1);
    REQUIRE(pass.blocks_optimized() == 1);
    REQUIRE(pass.t_removed() == 15);
}

TEST_CASE("Exact phase-polynomial pass tracks affine signs from Pauli products", "[optimizer]") {
    HirModule original(4, 15);
    append_rank4_product_word(original, false);

    HirModule hir(4, 15);
    append_rank4_product_word(hir, false);

    ExactPhasePolynomialTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(hir.ops.size() == 6);
    for (const auto& op : hir.ops) {
        REQUIRE(op.op_type() == OpType::PHASE_ROTATION);
        REQUIRE(op.alpha() == 1.5);
    }
    REQUIRE(std::abs(hir.global_weight.real()) < 1e-12);
    REQUIRE(std::abs(hir.global_weight.imag() - 1.0) < 1e-12);
    REQUIRE(pass.blocks_optimized() == 1);
    REQUIRE(pass.t_removed() == 15);
    require_dense_equal(dense_hir_ops_value(hir), dense_hir_ops_value(original));
    require_statevector_equal(hir, original);
}

TEST_CASE("Exact phase-polynomial pass preserves global phase for signed odd emissions",
          "[optimizer]") {
    HirModule original(4, 16);
    append_rank4_product_word(original, true);
    append_rank4_product_t(original, 0b0011, true);

    HirModule hir(4, 16);
    append_rank4_product_word(hir, true);
    append_rank4_product_t(hir, 0b0011, true);

    ExactPhasePolynomialTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[0].is_dagger());
    REQUIRE(hir.destab_mask(hir.ops[0]) == 0b0011);
    REQUIRE(hir.stab_mask(hir.ops[0]) == 0b0011);
    REQUIRE(std::abs(hir.global_weight.real() - test::kInvSqrt2) < 1e-12);
    REQUIRE(std::abs(hir.global_weight.imag() - test::kInvSqrt2) < 1e-12);
    REQUIRE(pass.blocks_optimized() == 1);
    REQUIRE(pass.t_removed() == 15);
    require_dense_equal(dense_hir_ops_value(hir), dense_hir_ops_value(original));
}

TEST_CASE("Exact phase-polynomial pass reduces source-level complete CCZ_4 after peephole",
          "[optimizer]") {
    const char* source = "CCZ 0 1 2\nCCZ 0 1 3\nCCZ 0 2 3\nCCZ 1 2 3\n";
    auto original = trace(parse(source));
    auto hir = trace(parse(source));

    REQUIRE(hir.num_t_gates() == 28);

    PeepholeFusionPass peephole;
    peephole.run(hir);
    REQUIRE(hir.num_t_gates() == 8);

    ExactPhasePolynomialTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 7);
    REQUIRE(pass.blocks_optimized() == 1);
    REQUIRE(pass.t_removed() == 1);
    require_statevector_equal(hir, original);
}

TEST_CASE("Exact phase-polynomial pass preserves randomized commuting signed blocks",
          "[optimizer]") {
    bool saw_rewrite = false;

    for (uint64_t trial_seed = 1; trial_seed <= 32; ++trial_seed) {
        uint64_t rng = trial_seed;
        const uint8_t rank = static_cast<uint8_t>(1 + (test::test_lcg(rng) % 4));
        const size_t term_count = 2 + (test::test_lcg(rng) % 22);
        const uint32_t coord_limit = 1u << rank;

        HirModule original(4, term_count);
        HirModule hir(4, term_count);
        for (size_t i = 0; i < term_count; ++i) {
            const uint32_t coord =
                1 + static_cast<uint32_t>(test::test_lcg(rng) % (coord_limit - 1));
            bool sign = rank4_product_sign(coord);
            if ((test::test_lcg(rng) & 1ULL) != 0) {
                sign = !sign;
            }
            const bool dagger = (test::test_lcg(rng) & 1ULL) != 0;
            append_rank4_product_t_with_sign(original, coord, sign, dagger);
            append_rank4_product_t_with_sign(hir, coord, sign, dagger);
        }

        ExactPhasePolynomialTCountPass pass;
        pass.run(hir);
        saw_rewrite = saw_rewrite || pass.blocks_optimized() != 0;

        CAPTURE(trial_seed, rank, term_count, pass.blocks_optimized(), pass.t_removed());
        require_dense_equal(dense_hir_ops_value(hir), dense_hir_ops_value(original));
    }

    REQUIRE(saw_rewrite);
}

TEST_CASE("Exact phase-polynomial pass skips blocks above rank cap", "[optimizer]") {
    HirModule hir(5, 31);
    append_all_nonzero_z_parities(hir, 5);

    ExactPhasePolynomialTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 31);
    REQUIRE(pass.blocks_considered() == 1);
    REQUIRE(pass.blocks_optimized() == 0);
}

TEST_CASE("Exact phase-polynomial pass skips noncommuting T blocks", "[optimizer]") {
    HirModule hir(1, 3);
    append_tgate(hir, 0, 1, false);
    append_tgate(hir, 1, 0, false);
    append_tgate(hir, 0, 1, false);

    ExactPhasePolynomialTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 3);
    REQUIRE(pass.blocks_considered() == 1);
    REQUIRE(pass.blocks_optimized() == 0);
}
