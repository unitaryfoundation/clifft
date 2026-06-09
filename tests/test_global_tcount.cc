#include "clifft/optimizer/global_tcount_pass.h"
#include "clifft/optimizer/mcr_reorder.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pass_registry.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/todd_phase_pass.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace clifft;
using namespace clifft::test;

TEST_CASE("GlobalTcountPass is registered and disabled by default", "[optimizer][global_tcount]") {
    bool found = false;
    for (size_t i = 0; i < kNumRegisteredPasses; ++i) {
        if (kRegisteredPasses[i].name == "GlobalTcountPass") {
            found = true;
            REQUIRE(kRegisteredPasses[i].kind == PassKind::HIR);
            REQUIRE_FALSE(kRegisteredPasses[i].default_enabled);
            REQUIRE(kRegisteredPasses[i].make_hir != nullptr);
        }
    }
    REQUIRE(found);
}

TEST_CASE("MCR reorder reduces T count on kicked XY block", "[optimizer][global_tcount]") {
    HirModule hir(2, 16);
    append_tgate(hir, X(0), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);
    append_tgate(hir, X(1), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);
    append_tgate(hir, X(0), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);
    append_tgate(hir, X(1), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);

    PeepholeFusionPass peep;
    peep.run(hir);
    size_t before = hir.num_t_gates();
    REQUIRE(before == 8);

    McrReorderPass mcr;
    mcr.run(hir);
    peep.run(hir);

    REQUIRE(hir.num_t_gates() < before);
    REQUIRE(mcr.stats().swaps_applied > 0);
}

TEST_CASE("GlobalTcountPass reports phase statistics", "[optimizer][global_tcount]") {
    HirModule hir(2, 16);
    append_tgate(hir, X(0), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);
    append_tgate(hir, X(1), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);
    append_tgate(hir, X(0), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);
    append_tgate(hir, X(1), 0, false);
    append_tgate(hir, X(0) | X(1), X(1), false);

    PeepholeFusionPass peep;
    peep.run(hir);

    GlobalTcountPass pass;
    pass.run(hir);
    peep.run(hir);

    REQUIRE(pass.t_gates_before() >= pass.t_gates_after());
    REQUIRE(pass.mcr_stats().t_removed + pass.todd_t_removed() > 0);
}

TEST_CASE("pass_registry_json includes global T-count passes", "[optimizer][global_tcount]") {
    std::string json = clifft::pass_registry_json();
    REQUIRE(json.find("GlobalTcountPass") != std::string::npos);
    REQUIRE(json.find("McrReorderPass") != std::string::npos);
    REQUIRE(json.find("ToddPhasePass") != std::string::npos);
}
