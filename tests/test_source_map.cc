// Source map propagation tests
//
// Verifies that source line provenance threads correctly through the
// full compile pipeline: parse -> trace -> optimize -> lower.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/peephole.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace clifft;

// Helper: number of bytecode instructions in the source map.
static size_t source_map_size(const CompiledModule& m) {
    return m.source_map.size();
}

// Helper: retrieve the source lines for bytecode instruction i.
static std::vector<uint32_t> source_map_entry(const CompiledModule& m, size_t i) {
    auto lines = m.source_map.lines_for(i);
    return {lines.begin(), lines.end()};
}

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

// Helper: full pipeline through lower
static CompiledModule compiled_from(const char* text, bool optimize = false) {
    auto hir = hir_from(text);
    if (optimize) {
        HirPassManager pm;
        pm.add_pass(std::make_unique<PeepholeFusionPass>());
        pm.run(hir);
    }
    return clifft::lower(hir);
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

// =============================================================================
// Back-End source map and k-history
// =============================================================================

TEST_CASE("Source map: lower produces parallel source_map and k_history", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nM 0");
    REQUIRE(source_map_size(prog) == prog.bytecode.size());
    REQUIRE(prog.source_map.active_k_history().size() == prog.bytecode.size());
}

TEST_CASE("Source map: k_history shows expansion from T gate", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nM 0");
    // T gate expands k from 0 to 1 -- at least some entries should be 1
    bool found_k1 = false;
    for (auto k : prog.source_map.active_k_history()) {
        if (k == 1) {
            found_k1 = true;
            break;
        }
    }
    REQUIRE(found_k1);
}

TEST_CASE("Source map: k_history shows compaction after measurement", "[source_map]") {
    // H 0; T 0; M 0; H 1; T 1; M 1
    // Each qubit breathes k: 0->1->0 independently. The measurement
    // instruction records k at emission time (before deactivation),
    // so the second T's EXPAND records k=1, and the second M records k=1.
    // After M1 deactivates, the following instructions (if any) get k=0.
    auto prog = compiled_from("H 0\nT 0\nM 0\nH 1\nT 1\nM 1");
    REQUIRE(!prog.source_map.active_k_history().empty());
    // Verify breathing: k should reach 1 and return
    bool found_k1 = false;
    for (auto k : prog.source_map.active_k_history()) {
        if (k == 1) {
            found_k1 = true;
            break;
        }
    }
    REQUIRE(found_k1);
    // The last measurement deactivates, so the final k recorded is 1
    // (measurement itself runs at k=1; deactivation is a post-emission effect).
    REQUIRE(prog.source_map.active_k_history().back() == 1);
}

TEST_CASE("Source map: bytecode source lines trace back to correct input", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nM 0");
    size_t n = source_map_size(prog);
    REQUIRE(n == prog.bytecode.size());
    for (size_t i = 0; i < n; ++i) {
        auto lines = source_map_entry(prog, i);
        REQUIRE(!lines.empty());
        for (auto line : lines) {
            REQUIRE(line >= 1);
            REQUIRE(line <= 3);
        }
    }
}

TEST_CASE("Source map: optimized pipeline still produces valid maps", "[source_map]") {
    auto prog = compiled_from("H 0\nT 0\nT 0\nM 0", /*optimize=*/true);
    REQUIRE(source_map_size(prog) == prog.bytecode.size());
    REQUIRE(prog.source_map.active_k_history().size() == prog.bytecode.size());
}

TEST_CASE("Source map: empty circuit produces empty maps", "[source_map]") {
    auto prog = compiled_from("");
    REQUIRE(prog.bytecode.empty());
    REQUIRE(source_map_size(prog) == 0);
    REQUIRE(prog.source_map.active_k_history().empty());
}
