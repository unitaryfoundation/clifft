// Experimental MCR T-count pass tests

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/experimental_mcr_t_count_pass.h"
#include "clifft/optimizer/peephole.h"

#include <catch2/catch_test_macros.hpp>

using namespace clifft;

static HirModule hir_from(const char* text) {
    return clifft::trace(clifft::parse(text));
}

TEST_CASE("ExperimentalMcr: reduces kicked XY style T window", "[optimizer][mcr]") {
    auto hir = hir_from(
        "R_XX(0.25) 0 1\n"
        "R_Z(0.25) 0\n"
        "R_Z(0.25) 1\n"
        "R_XX(0.25) 0 1\n"
        "R_YY(0.25) 0 1");

    PeepholeFusionPass peephole;
    peephole.run(hir);
    REQUIRE(hir.num_t_gates() == 5);

    ExperimentalMcrTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 3);
    REQUIRE(pass.window_scans() >= 1);
    REQUIRE(pass.quadruples_found() >= 1);
    REQUIRE(pass.swaps_applied() == 1);
    REQUIRE(pass.merges() == 1);
    REQUIRE(pass.t_removed() == 2);
}

TEST_CASE("ExperimentalMcr: barrier splits window", "[optimizer][mcr]") {
    auto hir = hir_from(
        "R_XX(0.25) 0 1\n"
        "R_Z(0.25) 0\n"
        "M 0\n"
        "R_Z(0.25) 1\n"
        "R_XX(0.25) 0 1\n"
        "R_YY(0.25) 0 1");

    PeepholeFusionPass peephole;
    peephole.run(hir);
    REQUIRE(hir.num_t_gates() == 5);

    ExperimentalMcrTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 5);
    REQUIRE(pass.swaps_applied() == 0);
    REQUIRE(pass.t_removed() == 0);
}

TEST_CASE("ExperimentalMcr: rejects non MCR quadruple", "[optimizer][mcr]") {
    auto hir = hir_from(
        "R_XX(0.25) 0 1\n"
        "R_Z(0.25) 0\n"
        "R_Z(0.25) 1\n"
        "R_XX(0.25) 0 1\n"
        "R_ZZ(0.25) 0 1");

    PeepholeFusionPass peephole;
    peephole.run(hir);
    REQUIRE(hir.num_t_gates() == 5);

    ExperimentalMcrTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 5);
    REQUIRE(pass.swaps_applied() == 0);
    REQUIRE(pass.t_removed() == 0);
}

TEST_CASE("ExperimentalMcr: rejects blocked gather that changes semantics", "[optimizer][mcr]") {
    auto hir = hir_from(
        "R_XX(0.25) 1 0\n"
        "R_Y(0.25) 1\n"
        "R_Y(0.25) 0\n"
        "R_XX(0.25) 0 1\n"
        "R_X(0.25) 0\n"
        "R_ZZ(0.25) 1 0");

    PeepholeFusionPass peephole;
    peephole.run(hir);
    REQUIRE(hir.num_t_gates() == 6);

    ExperimentalMcrTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 6);
    REQUIRE(pass.candidates_considered() >= 1);
    REQUIRE(pass.equivalence_checks() >= 1);
    REQUIRE(pass.swaps_applied() == 0);
    REQUIRE(pass.t_removed() == 0);
}

TEST_CASE("ExperimentalMcr: rejects mixed T directions in MCR quadruple", "[optimizer][mcr]") {
    auto hir = hir_from(
        "R_XX(0.25) 0 1\n"
        "R_ZZ(0.25) 1 0\n"
        "S 1\n"
        "R_Y(0.25) 0\n"
        "R_X(0.25) 1\n"
        "R_ZZ(0.25) 1 0");

    PeepholeFusionPass peephole;
    peephole.run(hir);
    REQUIRE(hir.num_t_gates() == 5);

    ExperimentalMcrTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 5);
    REQUIRE(pass.swaps_applied() == 0);
    REQUIRE(pass.t_removed() == 0);
}

TEST_CASE("ExperimentalMcr: sliding anchor finds late window candidate", "[optimizer][mcr]") {
    auto hir = hir_from(
        "R_Z(0.25) 10\n"
        "R_Z(0.25) 11\n"
        "R_Z(0.25) 12\n"
        "R_Z(0.25) 13\n"
        "R_Z(0.25) 14\n"
        "R_Z(0.25) 15\n"
        "R_Z(0.25) 16\n"
        "R_Z(0.25) 17\n"
        "R_Z(0.25) 18\n"
        "R_Z(0.25) 19\n"
        "R_Z(0.25) 20\n"
        "R_Z(0.25) 21\n"
        "R_Z(0.25) 22\n"
        "R_Z(0.25) 23\n"
        "R_Z(0.25) 24\n"
        "R_Z(0.25) 25\n"
        "R_Z(0.25) 26\n"
        "R_Z(0.25) 27\n"
        "R_XX(0.25) 0 1\n"
        "R_Z(0.25) 0\n"
        "R_Z(0.25) 1\n"
        "R_XX(0.25) 0 1\n"
        "R_YY(0.25) 0 1");

    PeepholeFusionPass peephole;
    peephole.run(hir);
    REQUIRE(hir.num_t_gates() == 23);

    ExperimentalMcrTCountPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 21);
    REQUIRE(pass.window_scans() >= 1);
    REQUIRE(pass.window_scans_over_lookahead_cap() >= 1);
    REQUIRE(pass.swaps_applied() == 1);
    REQUIRE(pass.t_removed() == 2);
}
