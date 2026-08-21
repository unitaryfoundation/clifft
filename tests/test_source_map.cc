// Source map propagation tests
//
// Verifies that source line provenance threads correctly through the
// front-end and HIR optimization pipeline: parse -> trace -> optimize.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/peephole.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace clifft;

// Helper: full pipeline through trace (no optimizer)
static HirModule hir_from(const char* text) {
    return clifft::trace(clifft::parse(text));
}

// Helper: full pipeline through trace + peephole
static HirModule hir_optimized(const char* text) {
    auto hir = hir_from(text);
    HirPassManager pm;
    pm.add_pass(std::make_unique<PeepholeFusionPass>());
    pm.run(hir);
    return hir;
}

// =============================================================================
// Front-End source map population
// =============================================================================

TEST_CASE("Source map: trace populates source_map parallel to ops", "[source_map]") {
    auto hir = hir_from("H 0\nT 0\nM 0");
    REQUIRE(hir.source_map.size() == hir.ops.size());
    // T gate on line 2 produces at least one HIR op with source_line == 2
    bool found_t = false;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() == OpType::T_GATE) {
            REQUIRE(hir.source_map[i].size() == 1);
            REQUIRE(hir.source_map[i][0] == 2);
            found_t = true;
        }
    }
    REQUIRE(found_t);
}

TEST_CASE("Source map: Clifford-only circuit still has parallel source_map", "[source_map]") {
    auto hir = hir_from("H 0\nS 0\nM 0");
    REQUIRE(hir.source_map.size() == hir.ops.size());
    // Only ops should be MEASURE -- Cliffords are absorbed by the tableau
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        REQUIRE(hir.source_map[i].size() == 1);
        REQUIRE(hir.source_map[i][0] > 0);
    }
}

TEST_CASE("Source map: multi-target gate emits one source_map entry per op", "[source_map]") {
    auto hir = hir_from("T 0 1 2");
    REQUIRE(hir.source_map.size() == hir.ops.size());
    // All T ops should trace back to line 1
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() == OpType::T_GATE) {
            REQUIRE(hir.source_map[i] == std::vector<uint32_t>{1});
        }
    }
}

// =============================================================================
// Optimizer maintains source_map invariant
// =============================================================================

TEST_CASE("Source map: peephole fusion preserves parallel size", "[source_map]") {
    auto hir = hir_optimized("H 0\nT 0\nT 0");
    REQUIRE(hir.source_map.size() == hir.ops.size());
}

TEST_CASE("Source map: T plus T fusion deletes both source entries", "[source_map]") {
    // H 0 (line 1) -> absorbed by tableau
    // T 0 (line 2) + T 0 (line 3) -> fused to S, absorbed offline
    auto hir = hir_optimized("H 0\nT 0\nT 0");
    // S absorption eliminates both T-gate ops. No ops should remain.
    REQUIRE(hir.ops.empty());
    REQUIRE(hir.source_map.empty());
}

TEST_CASE("Source map: T plus T_dag cancellation removes both from map", "[source_map]") {
    // T 0 (line 1) + T_DAG 0 (line 2) -> cancelled entirely
    // M 0 (line 3) -> measurement remains
    auto hir = hir_optimized("T 0\nT_DAG 0\nM 0");
    REQUIRE(hir.source_map.size() == hir.ops.size());
    // No T_GATE should remain (CLIFFORD_PHASE no longer exists)
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        REQUIRE(hir.ops[i].op_type() != OpType::T_GATE);
    }
}

TEST_CASE("Source map: absorbed Pauli residue removes only its source entries", "[source_map]") {
    auto hir = hir_optimized("R_Z(0.3) 0\nR_Z(0.7) 0\nR_X(0.2) 0");

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::PHASE_ROTATION);
    REQUIRE(hir.source_map == std::vector<std::vector<uint32_t>>{{3}});
}

TEST_CASE("Source map: terminal reset phase elimination removes only phase entries",
          "[source_map]") {
    auto hir = hir_optimized(
        "R 0 1\n"
        "H 0 1\n"
        "R_Z(0.3) 0 1\n"
        "X_ERROR(0.01) 0 1\n"
        "MR 0 1");

    REQUIRE(hir.source_map.size() == hir.ops.size());
    REQUIRE(hir.ops.size() == 10);

    size_t reset_entries = 0;
    size_t noise_entries = 0;
    size_t measure_reset_entries = 0;
    for (const auto& entry : hir.source_map) {
        REQUIRE(entry.size() == 1);
        REQUIRE(entry[0] != 3);
        reset_entries += entry[0] == 1;
        noise_entries += entry[0] == 4;
        measure_reset_entries += entry[0] == 5;
    }
    REQUIRE(reset_entries == 4);
    REQUIRE(noise_entries == 2);
    REQUIRE(measure_reset_entries == 4);
}
