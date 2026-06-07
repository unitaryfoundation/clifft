// Experimental global T-count pass tests (issue #40)

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/phase_poly_tcount_pass.h"
#include "clifft/optimizer/todd_gf2.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>

using namespace clifft;

static HirModule hir_from(const char* text) {
    return clifft::trace(clifft::parse(text));
}

TEST_CASE("TODD GF2: duplicate columns merge to Clifford", "[optimizer][global_tcount]") {
    Gf2Matrix mat;
    mat.n = 2;
    mat.num_words = 1;

    uint64_t c0 = 0b01;
    uint64_t c1 = 0b01;
    mat.append_col(&c0);
    mat.append_col(&c1);

    std::vector<int> coeffs = {1, 1};
    properize(mat, coeffs);

    REQUIRE(mat.num_cols() == 1);
    REQUIRE(coeffs[0] == 2);
}

TEST_CASE("ExperimentalGlobalTcount: MCR reduces toggle sandwich", "[optimizer][global_tcount]") {
    auto hir = hir_from(
        "R_XX(0.25) 0 1\n"
        "R_PAULI(0.25) X0*Y1\n"
        "R_PAULI(0.25) Y0*X1\n"
        "R_XX(0.25) 0 1\n"
        "R_YY(0.25) 0 1\n"
        "R_PAULI(0.25) Y0*X1");

    PeepholeFusionPass peephole;
    peephole.run(hir);
    const size_t before = hir.num_t_gates();

    ExperimentalGlobalTcountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() < before);
    REQUIRE(pass.mcr_stats().swaps_applied >= 1);
}

TEST_CASE("ExperimentalGlobalTcount: no-op on single T gate", "[optimizer][global_tcount]") {
    auto hir = hir_from("H 0\nT 0");

    ExperimentalGlobalTcountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 1);
}

TEST_CASE("ExperimentalGlobalTcount: registry resolves pass", "[optimizer][global_tcount]") {
    auto pass = make_hir_pass("ExperimentalGlobalTcountPass");
    REQUIRE(pass != nullptr);
}

TEST_CASE("ExperimentalGlobalTcount: evaluation table for small circuits",
          "[optimizer][global_tcount]") {
    struct Row {
        const char* name;
        const char* circuit;
    };
    const Row rows[] = {
        {"toggle_sandwich",
         "R_XX(0.25) 0 1\nR_PAULI(0.25) X0*Y1\nR_PAULI(0.25) Y0*X1\nR_XX(0.25) 0 1\nR_YY(0.25) "
         "0 1\nR_PAULI(0.25) Y0*X1"},
        {"two_disjoint_pair_blocks",
         "R_XX(0.25) 0 1\nR_Z(0.25) 0\nR_Z(0.25) 1\nR_XX(0.25) 0 1\nR_YY(0.25) 0 1\n"
         "R_XX(0.25) 2 3\nR_Z(0.25) 2\nR_Z(0.25) 3\nR_XX(0.25) 2 3\nR_YY(0.25) 2 3"},
    };

    for (const auto& row : rows) {
        auto hir = hir_from(row.circuit);
        PeepholeFusionPass peephole;
        peephole.run(hir);
        const size_t after_peephole = hir.num_t_gates();

        ExperimentalGlobalTcountPass pass;
        pass.run(hir);
        const size_t after_global = hir.num_t_gates();

        INFO("circuit=" << row.name << " peephole_T=" << after_peephole
                        << " global_T=" << after_global);
        REQUIRE(after_global <= after_peephole);
    }
}
