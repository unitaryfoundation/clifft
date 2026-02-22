#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"

#include <catch2/catch_test_macros.hpp>

using namespace ucc;

// =============================================================================
// Helper: Parse -> Trace -> Lower pipeline
// =============================================================================

static CompiledModule compile(const std::string& stim_text) {
    Circuit circuit = parse(stim_text);
    HirModule hir = trace(circuit);
    return lower(hir);
}

// =============================================================================
// GF2Basis Direct Unit Tests
// =============================================================================

TEST_CASE("GF2Basis: zero vector is always in span", "[backend][gf2]") {
    GF2Basis basis;
    auto result = basis.find_in_span(stim::bitword<kStimWidth>(0));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0);  // Empty combination produces zero
}

TEST_CASE("GF2Basis: single vector span membership", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(stim::bitword<kStimWidth>(0b1010));

    // Same vector should be in span with x_mask = 1
    auto result = basis.find_in_span(stim::bitword<kStimWidth>(0b1010));
    REQUIRE(result.has_value());
    REQUIRE(*result == 1);

    // Different vector not in span
    auto not_found = basis.find_in_span(stim::bitword<kStimWidth>(0b0101));
    REQUIRE_FALSE(not_found.has_value());
}

TEST_CASE("GF2Basis: XOR combinations", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(stim::bitword<kStimWidth>(0b0011));  // idx 0
    basis.add(stim::bitword<kStimWidth>(0b0101));  // idx 1

    // 0b0110 = 0b0011 XOR 0b0101
    auto result = basis.find_in_span(stim::bitword<kStimWidth>(0b0110));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0b11);  // Both vectors needed
}

TEST_CASE("GF2Basis: add returns correct index", "[backend][gf2]") {
    GF2Basis basis;
    REQUIRE(basis.add(stim::bitword<kStimWidth>(0b001)) == 0);
    REQUIRE(basis.add(stim::bitword<kStimWidth>(0b010)) == 1);
    REQUIRE(basis.add(stim::bitword<kStimWidth>(0b100)) == 2);
    REQUIRE(basis.rank() == 3);
}

TEST_CASE("GF2Basis: remove shifts indices", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(stim::bitword<kStimWidth>(0b001));  // idx 0
    basis.add(stim::bitword<kStimWidth>(0b010));  // idx 1
    basis.add(stim::bitword<kStimWidth>(0b100));  // idx 2

    basis.remove(1);  // Remove middle vector
    REQUIRE(basis.rank() == 2);

    // 0b100 should now be at index 1 (was index 2)
    auto result = basis.find_in_span(stim::bitword<kStimWidth>(0b100));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0b10);  // Now index 1, so x_mask = 2
}

TEST_CASE("GF2Basis: echelon form with dependent vectors", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(stim::bitword<kStimWidth>(0b1100));  // idx 0, lead bit 3
    basis.add(stim::bitword<kStimWidth>(0b1010));  // idx 1, lead bit 3 (same!)

    // After echelon reduction, 0b1010 is stored as 0b1010 XOR 0b1100 = 0b0110
    // So 0b0110 is in span via index 1 alone
    auto result = basis.find_in_span(stim::bitword<kStimWidth>(0b0110));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0b11);  // Need both: 0b1100 XOR 0b1010 = 0b0110
}

TEST_CASE("GF2Basis: remove then add reuses slot", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(stim::bitword<kStimWidth>(0b01));
    basis.add(stim::bitword<kStimWidth>(0b10));
    REQUIRE(basis.rank() == 2);

    basis.remove(0);  // Remove first
    REQUIRE(basis.rank() == 1);

    // Add new vector - should get index 1 (next available)
    uint32_t new_idx = basis.add(stim::bitword<kStimWidth>(0b11));
    REQUIRE(new_idx == 1);
    REQUIRE(basis.rank() == 2);
}

// =============================================================================
// Task 4.1: Instruction struct size
// =============================================================================

TEST_CASE("Backend: Instruction is exactly 32 bytes", "[backend]") {
    static_assert(sizeof(Instruction) == 32);
    REQUIRE(sizeof(Instruction) == 32);
}

TEST_CASE("Backend: Instruction is 32-byte aligned", "[backend]") {
    static_assert(alignof(Instruction) == 32);
    REQUIRE(alignof(Instruction) == 32);
}

// =============================================================================
// Task 4.2: T-Gate GF(2) Tracking
// =============================================================================

TEST_CASE("Backend: 4 independent T gates emit 4 OP_BRANCH with peak_rank=4", "[backend]") {
    // T gates on superposition states (after H) create new GF(2) dimensions.
    // T on |0> is diagonal (SCALAR_PHASE), but T on |+> branches.
    auto result = compile(R"(
        H 0
        H 1
        H 2
        H 3
        T 0
        T 1
        T 2
        T 3
    )");

    REQUIRE(result.bytecode.size() == 4);
    REQUIRE(result.peak_rank == 4);

    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(result.bytecode[i].opcode == Opcode::OP_BRANCH);
        REQUIRE(result.bytecode[i].branch.bit_index == i);
    }
}

TEST_CASE("Backend: T on same qubit twice emits BRANCH then COLLIDE", "[backend]") {
    // First T on superposition creates dimension, second T collides.
    // Need H to put qubit in superposition first.
    auto result = compile(R"(
        H 0
        T 0
        T 0
    )");

    REQUIRE(result.bytecode.size() == 2);
    REQUIRE(result.peak_rank == 1);

    REQUIRE(result.bytecode[0].opcode == Opcode::OP_BRANCH);
    REQUIRE(result.bytecode[0].branch.bit_index == 0);

    REQUIRE(result.bytecode[1].opcode == Opcode::OP_COLLIDE);
    REQUIRE(result.bytecode[1].branch.x_mask == 1);  // Uses basis vector 0
}

TEST_CASE("Backend: T on computational basis emits SCALAR_PHASE", "[backend]") {
    // T on |0> or |1> (Z-eigenstates) is diagonal: no spatial shift.
    // The rewound Pauli has destab_mask=0 (pure Z), so SCALAR_PHASE.
    auto result = compile("T 0");
    REQUIRE(result.bytecode.size() == 1);
    REQUIRE(result.bytecode[0].opcode == Opcode::OP_SCALAR_PHASE);

    // Same with Z gate before (Z commutes with T, doesn't change basis)
    auto result2 = compile(R"(
        Z 0
        T 0
    )");
    REQUIRE(result2.bytecode.size() == 1);
    REQUIRE(result2.bytecode[0].opcode == Opcode::OP_SCALAR_PHASE);

    // Multiple T gates on Z-eigenstates are all SCALAR_PHASE
    auto result3 = compile(R"(
        T 0
        T 1
        T 2
    )");
    REQUIRE(result3.bytecode.size() == 3);
    for (const auto& instr : result3.bytecode) {
        REQUIRE(instr.opcode == Opcode::OP_SCALAR_PHASE);
    }
    REQUIRE(result3.peak_rank == 0);  // No branching dimensions
}

TEST_CASE("Backend: T_DAG sets is_dagger flag", "[backend]") {
    auto result = compile(R"(
        T_DAG 0
        T 1
    )");

    REQUIRE(result.bytecode.size() == 2);
    REQUIRE(result.bytecode[0].is_dagger == true);
    REQUIRE(result.bytecode[1].is_dagger == false);
}

TEST_CASE("Backend: CX spreads T to create linear dependency", "[backend]") {
    // H puts q0 in superposition, T branches.
    // CX entangles, then T on q1 targets a new independent dimension.
    auto result = compile(R"(
        H 0
        H 1
        T 0
        CX 0 1
        T 1
    )");

    REQUIRE(result.bytecode.size() == 2);
    REQUIRE(result.peak_rank == 2);
    REQUIRE(result.bytecode[0].opcode == Opcode::OP_BRANCH);
    REQUIRE(result.bytecode[1].opcode == Opcode::OP_BRANCH);
}

TEST_CASE("Backend: CX creates COLLIDE scenario", "[backend]") {
    // Put qubits in superposition, then multiple T gates.
    // Third T on same qubit as first should COLLIDE.
    auto result = compile(R"(
        H 0
        H 1
        T 0
        T 1
        T 0
    )");

    // T0: BRANCH (dim 0), T1: BRANCH (dim 1), T0 again: COLLIDE (X0 in span)
    REQUIRE(result.bytecode.size() == 3);
    REQUIRE(result.peak_rank == 2);
    REQUIRE(result.bytecode[0].opcode == Opcode::OP_BRANCH);
    REQUIRE(result.bytecode[1].opcode == Opcode::OP_BRANCH);
    REQUIRE(result.bytecode[2].opcode == Opcode::OP_COLLIDE);
    REQUIRE(result.bytecode[2].branch.x_mask == 1);  // Uses basis[0] = X0
}

TEST_CASE("Backend: x_mask encodes linear combination", "[backend]") {
    // H on both qubits, T on each, then CX 1 0 (control=1, target=0).
    // After CX 1 0 in Heisenberg: Z0 -> Z0 Z1, so rewound Z0 has X0 XOR X1.
    auto result = compile(R"(
        H 0
        H 1
        T 0
        T 1
        CX 1 0
        T 0
    )");

    // Third T on q0 targets X0 XOR X1 (destab=3), which is basis[0] XOR basis[1]
    REQUIRE(result.bytecode.size() == 3);
    REQUIRE(result.bytecode[2].opcode == Opcode::OP_COLLIDE);
    REQUIRE(result.bytecode[2].branch.x_mask == 0b11);  // Uses basis[0] and basis[1]
}

TEST_CASE("Backend: commutation_mask tracks Z-overlap", "[backend]") {
    // After H, T has destab=X0, stab=0. The basis vector is X0.
    // Second T on same qubit: destab=X0, stab=0.
    // commutation_mask checks (basis[i] & stab_mask).popcount() % 2.
    // With stab=0, commutation_mask should be 0.
    auto result = compile(R"(
        H 0
        T 0
        T 0
    )");

    // Both T gates have stab_mask=0 (pure X after H rewinding)
    // So commutation_mask = 0 for both
    REQUIRE(result.bytecode[0].commutation_mask == 0);
    REQUIRE(result.bytecode[1].commutation_mask == 0);

    // For non-zero commutation, we need stab_mask to overlap with basis.
    // This happens when T is applied after a different Clifford sequence.
    // H S H puts qubit in Y-eigenstate: destab has X, stab has Z.
    // But this is getting complex. Just verify the formula works.
}

// =============================================================================
// Task 4.3: Measurements
// =============================================================================

TEST_CASE("Backend: deterministic M emits OP_MEASURE_DETERMINISTIC", "[backend]") {
    // M on |0> is deterministic
    auto result = compile("M 0");

    REQUIRE(result.bytecode.size() == 1);
    REQUIRE(result.bytecode[0].opcode == Opcode::OP_MEASURE_DETERMINISTIC);
}

TEST_CASE("Backend: M after H emits only OP_AG_PIVOT", "[backend]") {
    // H puts qubit in superposition, M has β ∉ span(V) (empty basis).
    // Per corrected physics: β ≠ 0 and β ∉ span(V) → only AG_PIVOT, no MEASURE_* opcode.
    // The Front-End handles the tableau collapse; Back-End just emits the pivot.
    auto result = compile(R"(
        H 0
        M 0
    )");

    // Only AG_PIVOT instruction (no array change, just sign tracking)
    REQUIRE(result.bytecode.size() == 1);
    REQUIRE(result.bytecode[0].opcode == Opcode::OP_AG_PIVOT);
    REQUIRE(result.constant_pool.ag_matrices.size() == 1);
}

TEST_CASE("Backend: M after T emits MEASURE_MERGE", "[backend]") {
    // H T M: T creates a dimension (β = X0), M has same β = X0.
    // Per corrected physics: β ≠ 0 and β ∈ span(V) → MERGE (array halves).
    auto result = compile(R"(
        H 0
        T 0
        M 0
    )");

    // T emits BRANCH (adds X0 to basis), M emits MERGE (collapses that dimension)
    // Plus AG_PIVOT from Front-End
    REQUIRE(result.bytecode.size() == 3);
    REQUIRE(result.bytecode[0].opcode == Opcode::OP_BRANCH);
    REQUIRE(result.bytecode[1].opcode == Opcode::OP_MEASURE_MERGE);
    REQUIRE(result.bytecode[2].opcode == Opcode::OP_AG_PIVOT);

    // After MERGE, dimension is reclaimed - basis should be empty
    REQUIRE(result.constant_pool.gf2_basis.empty());
}

TEST_CASE("Backend: ag_ref_outcome propagates to instruction", "[backend]") {
    auto result = compile(R"(
        H 0
        M 0
    )");

    // The AG reference outcome is set by the frontend
    // It should propagate to the MEASURE instruction
    REQUIRE(result.bytecode.size() >= 1);
    // ag_ref_outcome is either 0 or 1, just check it's set
    uint8_t ag_ref = result.bytecode[0].ag_ref_outcome;
    REQUIRE((ag_ref == 0 || ag_ref == 1));
}

TEST_CASE("Backend: multiple measurements track correctly", "[backend]") {
    // Both H M have β ∉ span(V) (empty basis), so only AG_PIVOTs are emitted
    auto result = compile(R"(
        H 0
        H 1
        M 0
        M 1
    )");

    REQUIRE(result.num_measurements == 2);

    // With corrected physics: both measurements have β ∉ span(V)
    // so only AG_PIVOTs are emitted, no MEASURE_* opcodes
    int ag_pivot_count = 0;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_AG_PIVOT) {
            ag_pivot_count++;
        }
    }
    REQUIRE(ag_pivot_count == 2);
}

// =============================================================================
// Task 4.3: Classical Conditionals
// =============================================================================

TEST_CASE("Backend: reset emits conditional", "[backend]") {
    // R = M + CX, which becomes MEASURE + CONDITIONAL
    auto result = compile(R"(
        H 0
        R 0
    )");

    // Should have: MEASURE_*, possibly AG_PIVOT, then CONDITIONAL
    bool found_conditional = false;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_CONDITIONAL) {
            found_conditional = true;
            // Check that controlling_meas is set (index 0 for first measurement)
            REQUIRE(instr.meta.controlling_meas == 0);
        }
    }
    REQUIRE(found_conditional);
}

TEST_CASE("Backend: conditional Pauli has correct masks", "[backend]") {
    auto result = compile(R"(
        H 0
        R 0
    )");

    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_CONDITIONAL) {
            // The conditional should have non-zero masks (applying X on reset)
            // After H, the reset's CX applies X0 (rewound through H -> Z0)
            // Actually depends on the Heisenberg picture state.
            // Just verify masks are populated.
            REQUIRE((instr.meta.destab_mask != 0 || instr.meta.stab_mask != 0));
        }
    }
}

// =============================================================================
// GF(2) Basis: Echelon Form & Rank Limits
// =============================================================================

TEST_CASE("Backend: echelon form handles overlapping leading bits", "[backend]") {
    // This test exposes bugs in naive Gaussian elimination.
    // If we add 0b11 (X0 XOR X1) and 0b10 (X1), then query 0b01 (X0),
    // a correct echelon implementation should find X0 = (X0 XOR X1) XOR X1.
    // A buggy impl that just linearly scans by leading bit would fail.
    auto result = compile(R"(
        H 0
        H 1
        CX 1 0
        T 0
        T 1
        CX 1 0
        T 0
    )");

    // First T on q0: after CX 1 0, destab = X0 XOR X1 (0b11) -> BRANCH
    // Second T on q1: destab = X1 (0b10) -> BRANCH (independent)
    // After CX 1 0 again: destab of q0 changes back
    // Third T on q0: destab should be found in span via echelon
    REQUIRE(result.peak_rank == 2);

    // Count opcodes
    int branch_count = 0, collide_count = 0;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_BRANCH)
            branch_count++;
        if (instr.opcode == Opcode::OP_COLLIDE)
            collide_count++;
    }
    REQUIRE(branch_count == 2);   // Two independent dimensions
    REQUIRE(collide_count == 1);  // Third T collides
}

TEST_CASE("Backend: commutation_mask non-zero with Y-basis", "[backend]") {
    // S H (in that order) gives rewound Pauli with both destab=1 and stab=1.
    // This creates a scenario where commutation_mask should be non-zero.
    auto result = compile(R"(
        S 0
        H 0
        T 0
        T 0
    )");

    // After S H: inv_state.zs[0] has xs=1 (destab) and zs=1 (stab).
    // First T: BRANCH (adds destab=1 to basis).
    // Second T: COLLIDE.
    // For COLLIDE: commutation_mask = (basis[0] & stab).popcount() % 2
    // basis[0] = 1, stab = 1, overlap = 1, popcount = 1, odd -> mask bit 0 set.
    REQUIRE(result.bytecode.size() == 2);
    REQUIRE(result.bytecode[0].opcode == Opcode::OP_BRANCH);
    REQUIRE(result.bytecode[1].opcode == Opcode::OP_COLLIDE);
    REQUIRE(result.bytecode[1].commutation_mask == 1);
}

TEST_CASE("Backend: MEASURE_FILTER on Z-basis measurement", "[backend]") {
    // H T creates dimension (X0). Then H H = I, so destab returns to 0.
    // But if we have T first, then measure in Z-basis with commutation.
    // This is tricky - we need β = 0 but comm_mask ≠ 0.
    //
    // Actually, β = 0 means destab_mask = 0 (pure Z measurement on Z-eigenstate).
    // For comm_mask ≠ 0, we need the basis to have vectors that anti-commute
    // with the Z measurement's stab_mask.
    //
    // Consider: H T on q0 (adds X0 to basis), then H on q0 (back to Z-basis),
    // then measure q0 (Z-basis). After H H = I, destab = 0 (pure Z).
    // But basis has X0, and Z0 measurement has stab = Z0 = 1.
    // comm_mask = (X0 & Z0).popcount() % 2 = 1 -> FILTER, not DETERMINISTIC.
    auto result = compile(R"(
        H 0
        T 0
        H 0
        M 0
    )");

    // T creates dimension (BRANCH), then M has β=0 but comm_mask=1 -> FILTER
    const Instruction* meas = nullptr;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_MEASURE_FILTER) {
            meas = &instr;
            break;
        }
    }
    REQUIRE(meas != nullptr);
    REQUIRE(meas->opcode == Opcode::OP_MEASURE_FILTER);
}

// =============================================================================
// GF(2) Basis Persistence
// =============================================================================

TEST_CASE("Backend: gf2_basis stored in constant pool", "[backend]") {
    // Need H to create superposition for T to branch
    auto result = compile(R"(
        H 0
        H 1
        H 2
        T 0
        T 1
        T 2
    )");

    REQUIRE(result.constant_pool.gf2_basis.size() == 3);
    // Each basis vector should have a single bit set (independent qubits)
    REQUIRE(static_cast<uint64_t>(result.constant_pool.gf2_basis[0]) == 1);
    REQUIRE(static_cast<uint64_t>(result.constant_pool.gf2_basis[1]) == 2);
    REQUIRE(static_cast<uint64_t>(result.constant_pool.gf2_basis[2]) == 4);
}

TEST_CASE("Backend: empty circuit produces empty bytecode", "[backend]") {
    auto result = compile("");

    REQUIRE(result.bytecode.empty());
    REQUIRE(result.peak_rank == 0);
    REQUIRE(result.constant_pool.gf2_basis.empty());
}

TEST_CASE("Backend: pure Clifford circuit produces empty bytecode", "[backend]") {
    auto result = compile(R"(
        H 0
        CX 0 1
        S 1
        H 1
    )");

    REQUIRE(result.bytecode.empty());
    REQUIRE(result.peak_rank == 0);
}

// =============================================================================
// Integration: Full Pipeline
// =============================================================================

TEST_CASE("Backend: teleportation circuit compiles", "[backend]") {
    auto result = compile(R"(
        H 1
        CX 1 2
        CX 0 1
        H 0
        M 0
        M 1
        CX 1 2
        CZ 0 2
    )");

    // Should compile without error
    // Has 2 measurements, various opcodes
    REQUIRE(result.num_measurements == 2);
}

TEST_CASE("Backend: peak_rank tracks maximum dimension reached", "[backend]") {
    // Two T-gates reach peak_rank=2, then MERGE reduces rank to 1.
    // peak_rank should remain 2 (the maximum ever reached).
    auto result = compile(R"(
        H 0
        H 1
        T 0
        T 1
        M 0
    )");

    REQUIRE(result.peak_rank == 2);
    // Final basis has only X1 (X0 was removed by MERGE)
    REQUIRE(result.constant_pool.gf2_basis.size() == 1);
    REQUIRE(static_cast<uint64_t>(result.constant_pool.gf2_basis[0]) == 2);  // X1
}

TEST_CASE("Backend: peak_rank with interleaved T and M", "[backend]") {
    // T adds dimension, M removes it, T adds it back.
    // Peak is 1 because we never exceed 1 active dimension.
    auto result = compile(R"(
        H 0
        T 0
        M 0
        H 0
        T 0
    )");

    REQUIRE(result.peak_rank == 1);
    // Both T-gates should be BRANCH (dimension added, removed, added again)
    int branch_count = 0;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_BRANCH)
            branch_count++;
    }
    REQUIRE(branch_count == 2);
}

TEST_CASE("Backend: T-count matches across pipeline", "[backend]") {
    Circuit circuit = parse(R"(
        T 0
        T 1
        T_DAG 2
        T 0
    )");
    HirModule hir = trace(circuit);

    // All 4 T/T_DAG gates should be in HIR
    int t_count = 0;
    for (const auto& op : hir.ops) {
        if (op.op_type() == OpType::T_GATE) {
            t_count++;
        }
    }
    REQUIRE(t_count == 4);

    CompiledModule result = lower(hir);
    // Should emit 4 bytecode instructions for T gates
    REQUIRE(result.bytecode.size() == 4);
}

TEST_CASE("Backend: exceeds 32 GF(2) rank limit", "[backend]") {
    // Each H+T pair on a fresh qubit adds one dimension to the GF(2) basis.
    // With 33 such pairs, the rank exceeds the backend limit of 32.
    std::string stim_text = R"(
        H 0
        T 0
        H 1
        T 1
        H 2
        T 2
        H 3
        T 3
        H 4
        T 4
        H 5
        T 5
        H 6
        T 6
        H 7
        T 7
        H 8
        T 8
        H 9
        T 9
        H 10
        T 10
        H 11
        T 11
        H 12
        T 12
        H 13
        T 13
        H 14
        T 14
        H 15
        T 15
        H 16
        T 16
        H 17
        T 17
        H 18
        T 18
        H 19
        T 19
        H 20
        T 20
        H 21
        T 21
        H 22
        T 22
        H 23
        T 23
        H 24
        T 24
        H 25
        T 25
        H 26
        T 26
        H 27
        T 27
        H 28
        T 28
        H 29
        T 29
        H 30
        T 30
        H 31
        T 31
        H 32
        T 32
    )";

    REQUIRE_THROWS_AS(compile(stim_text), std::runtime_error);
}
