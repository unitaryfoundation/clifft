// Source map propagation tests
//
// Verifies that source line provenance threads correctly through the
// full compile pipeline: parse -> trace -> optimize -> lower.

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/pass_manager.h"
#include "ucc/optimizer/peephole.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace ucc;

// Helper: full pipeline through trace (no optimizer)
static HirModule hir_from(const char* text) {
    return ucc::trace(ucc::parse(text));
}

// Helper: full pipeline through trace + peephole
static HirModule hir_optimized(const char* text) {
    auto hir = hir_from(text);
    PassManager pm;
    pm.add_pass(std::make_unique<PeepholeFusionPass>());
    pm.run(hir);
    return hir;
}

// Helper: full pipeline through lower
static CompiledModule compiled_from(const char* text, bool optimize = false) {
    auto hir = hir_from(text);
    if (optimize) {
        PassManager pm;
        pm.add_pass(std::make_unique<PeepholeFusionPass>());
        pm.run(hir);
    }
    return ucc::lower(hir);
}

// =============================================================================
// Task 2.2: Front-End source map population
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
// Task 2.3: Optimizer maintains source_map invariant
// =============================================================================

TEST_CASE("Source map: peephole fusion preserves parallel size", "[source_map]") {
    auto hir = hir_optimized("H 0\nT 0\nT 0");
    REQUIRE(hir.source_map.size() == hir.ops.size());
}

TEST_CASE("Source map: T plus T fusion carries both source lines", "[source_map]") {
    // H 0 (line 1) -> absorbed by tableau
    // T 0 (line 2) + T 0 (line 3) -> fused to S
    auto hir = hir_optimized("H 0\nT 0\nT 0");
    // Find the fused CLIFFORD_PHASE op
    bool found = false;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() == OpType::CLIFFORD_PHASE) {
            REQUIRE(hir.source_map[i].size() == 2);
            // Both original T gate lines should be present
            auto& lines = hir.source_map[i];
            bool has_2 = (lines[0] == 2 || lines[1] == 2);
            bool has_3 = (lines[0] == 3 || lines[1] == 3);
            REQUIRE(has_2);
            REQUIRE(has_3);
            found = true;
        }
    }
    REQUIRE(found);
}

TEST_CASE("Source map: T plus T_dag cancellation removes both from map", "[source_map]") {
    // T 0 (line 1) + T_DAG 0 (line 2) -> cancelled entirely
    // M 0 (line 3) -> measurement remains
    auto hir = hir_optimized("T 0\nT_DAG 0\nM 0");
    REQUIRE(hir.source_map.size() == hir.ops.size());
    // No T_GATE or CLIFFORD_PHASE should remain
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        REQUIRE(hir.ops[i].op_type() != OpType::T_GATE);
        REQUIRE(hir.ops[i].op_type() != OpType::CLIFFORD_PHASE);
    }
}

// =============================================================================
// Task 2.4: Back-End source map and k-history
// =============================================================================

TEST_CASE("Source map: lower produces parallel source_map and k_history", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nM 0");
    REQUIRE(prog.source_map.size() == prog.bytecode.size());
    REQUIRE(prog.active_k_history.size() == prog.bytecode.size());
}

TEST_CASE("Source map: k_history shows expansion from T gate", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nM 0");
    // T gate expands k from 0 to 1 -- at least some entries should be 1
    bool found_k1 = false;
    for (auto k : prog.active_k_history) {
        if (k == 1) {
            found_k1 = true;
            break;
        }
    }
    REQUIRE(found_k1);
}

TEST_CASE("Source map: k_history shows compaction after measurement", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nM 0");
    // After measurement compacts, the final k entries should be 0
    REQUIRE(!prog.active_k_history.empty());
    REQUIRE(prog.active_k_history.back() == 0);
}

TEST_CASE("Source map: bytecode source lines trace back to correct input", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nM 0");
    // Every source_map entry should contain lines from {1, 2, 3}
    for (const auto& lines : prog.source_map) {
        REQUIRE(!lines.empty());
        for (auto line : lines) {
            REQUIRE(line >= 1);
            REQUIRE(line <= 3);
        }
    }
}

TEST_CASE("Source map: optimized pipeline still produces valid maps", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nT 0\nM 0", /*optimize=*/true);
    REQUIRE(prog.source_map.size() == prog.bytecode.size());
    REQUIRE(prog.active_k_history.size() == prog.bytecode.size());
}

TEST_CASE("Source map: empty circuit produces empty maps", "[source_map]") {
    auto prog = compiled_from("");
    REQUIRE(prog.bytecode.empty());
    REQUIRE(prog.source_map.empty());
    REQUIRE(prog.active_k_history.empty());
}
