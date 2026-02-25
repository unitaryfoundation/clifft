#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <utility>

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
    auto result = basis.find_in_span(static_cast<uint64_t>(0));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0);  // Empty combination produces zero
}

TEST_CASE("GF2Basis: single vector span membership", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(static_cast<uint64_t>(0b1010));

    // Same vector should be in span with x_mask = 1
    auto result = basis.find_in_span(static_cast<uint64_t>(0b1010));
    REQUIRE(result.has_value());
    REQUIRE(*result == 1);

    // Different vector not in span
    auto not_found = basis.find_in_span(static_cast<uint64_t>(0b0101));
    REQUIRE_FALSE(not_found.has_value());
}

TEST_CASE("GF2Basis: XOR combinations", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(static_cast<uint64_t>(0b0011));  // idx 0
    basis.add(static_cast<uint64_t>(0b0101));  // idx 1

    // 0b0110 = 0b0011 XOR 0b0101
    auto result = basis.find_in_span(static_cast<uint64_t>(0b0110));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0b11);  // Both vectors needed
}

TEST_CASE("GF2Basis: add returns correct index", "[backend][gf2]") {
    GF2Basis basis;
    REQUIRE(basis.add(static_cast<uint64_t>(0b001)) == 0);
    REQUIRE(basis.add(static_cast<uint64_t>(0b010)) == 1);
    REQUIRE(basis.add(static_cast<uint64_t>(0b100)) == 2);
    REQUIRE(basis.rank() == 3);
}

TEST_CASE("GF2Basis: remove shifts indices", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(static_cast<uint64_t>(0b001));  // idx 0
    basis.add(static_cast<uint64_t>(0b010));  // idx 1
    basis.add(static_cast<uint64_t>(0b100));  // idx 2

    basis.remove(1);  // Remove middle vector
    REQUIRE(basis.rank() == 2);

    // 0b100 should now be at index 1 (was index 2)
    auto result = basis.find_in_span(static_cast<uint64_t>(0b100));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0b10);  // Now index 1, so x_mask = 2
}

TEST_CASE("GF2Basis: echelon form with dependent vectors", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(static_cast<uint64_t>(0b1100));  // idx 0, lead bit 3
    basis.add(static_cast<uint64_t>(0b1010));  // idx 1, lead bit 3 (same!)

    // After echelon reduction, 0b1010 is stored as 0b1010 XOR 0b1100 = 0b0110
    // So 0b0110 is in span via index 1 alone
    auto result = basis.find_in_span(static_cast<uint64_t>(0b0110));
    REQUIRE(result.has_value());
    REQUIRE(*result == 0b11);  // Need both: 0b1100 XOR 0b1010 = 0b0110
}

TEST_CASE("GF2Basis: remove then add reuses slot", "[backend][gf2]") {
    GF2Basis basis;
    basis.add(static_cast<uint64_t>(0b01));
    basis.add(static_cast<uint64_t>(0b10));
    REQUIRE(basis.rank() == 2);

    basis.remove(0);  // Remove first
    REQUIRE(basis.rank() == 1);

    // Add new vector - should get index 1 (next available)
    uint32_t new_idx = basis.add(static_cast<uint64_t>(0b11));
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
    REQUIRE((result.bytecode[0].flags & Instruction::FLAG_IS_DAGGER) != 0);
    REQUIRE((result.bytecode[1].flags & Instruction::FLAG_IS_DAGGER) == 0);
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
    // H puts qubit in superposition, M has beta not in span(V) (empty basis).
    // Per corrected physics: beta != 0 and beta not in span(V) -> only AG_PIVOT, no MEASURE_*
    // opcode. The Front-End handles the tableau collapse; Back-End just emits the pivot.
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
    // H T M: T creates a dimension (beta = X0), M has same beta = X0.
    // Per corrected physics: beta != 0 and beta in span(V) -> MERGE (array halves).
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
    // Both H M have beta not in span(V) (empty basis), so only AG_PIVOTs are emitted
    auto result = compile(R"(
        H 0
        H 1
        M 0
        M 1
    )");

    REQUIRE(result.num_measurements == 2);

    // With corrected physics: both measurements have beta not in span(V)
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

TEST_CASE("Backend: reset emits hidden measurement and conditional", "[backend]") {
    // R decomposes into hidden MEASURE + CONDITIONAL with FLAG_USE_LAST_OUTCOME
    // Use T to ensure there's a GF(2) basis so we get a MEASURE_MERGE opcode
    auto result = compile(R"(
        H 0
        T 0
        R 0
    )");

    // Should have hidden measurement or hidden AG_PIVOT, and conditional
    bool found_hidden_meas = false;
    bool found_conditional = false;
    for (const auto& instr : result.bytecode) {
        // Check for hidden MEASURE_* or hidden AG_PIVOT (when beta not in span(V))
        if (((instr.opcode == Opcode::OP_MEASURE_MERGE ||
              instr.opcode == Opcode::OP_MEASURE_FILTER ||
              instr.opcode == Opcode::OP_MEASURE_DETERMINISTIC ||
              instr.opcode == Opcode::OP_AG_PIVOT)) &&
            (instr.flags & Instruction::FLAG_HIDDEN)) {
            found_hidden_meas = true;
        }
        if (instr.opcode == Opcode::OP_CONDITIONAL &&
            (instr.flags & Instruction::FLAG_USE_LAST_OUTCOME)) {
            found_conditional = true;
        }
    }
    REQUIRE(found_hidden_meas);
    REQUIRE(found_conditional);

    // R has no visible measurement
    REQUIRE(result.num_measurements == 0);
}

TEST_CASE("Backend: conditional Pauli has correct masks", "[backend]") {
    // Test using explicit classical feedback to get OP_CONDITIONAL
    auto result = compile(R"(
        M 0
        CX rec[-1] 1
    )");

    bool found_conditional = false;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_CONDITIONAL) {
            found_conditional = true;
            // The conditional should have non-zero masks
            REQUIRE((instr.meta.destab_mask != 0 || instr.meta.stab_mask != 0));
        }
    }
    REQUIRE(found_conditional);
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
    // This is tricky - we need beta = 0 but comm_mask != 0.
    //
    // Actually, beta = 0 means destab_mask = 0 (pure Z measurement on Z-eigenstate).
    // For comm_mask != 0, we need the basis to have vectors that anti-commute
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

    // T creates dimension (BRANCH), then M has beta=0 but comm_mask=1 -> FILTER
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

// =============================================================================
// Phase 2.3: Noise Scheduling & Constant Pool
// =============================================================================

TEST_CASE("Backend: DEPOLARIZE1 creates noise_schedule entry with 3 channels", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        DEPOLARIZE1(0.03) 0
        M 0
    )");

    // DEPOLARIZE1(p) emits NOISE op which goes to noise_schedule, not bytecode
    REQUIRE(result.constant_pool.noise_schedule.size() == 1);

    const auto& entry = result.constant_pool.noise_schedule[0];
    REQUIRE(entry.channels.size() == 3);  // X, Y, Z channels

    // Total probability = p (not 3*p/3, just p because one of X/Y/Z fires)
    REQUIRE(entry.total_probability == Catch::Approx(0.03));

    // Each channel has prob = p/3
    for (const auto& ch : entry.channels) {
        REQUIRE(ch.prob == Catch::Approx(0.01));
    }
}

TEST_CASE("Backend: DEPOLARIZE2 creates noise_schedule entry with 15 channels",
          "[backend][noise]") {
    auto result = compile(R"(
        H 0
        H 1
        CX 0 1
        DEPOLARIZE2(0.15) 0 1
        M 0 1
    )");

    REQUIRE(result.constant_pool.noise_schedule.size() == 1);

    const auto& entry = result.constant_pool.noise_schedule[0];
    REQUIRE(entry.channels.size() == 15);  // All non-II two-qubit Paulis

    // Total probability = p
    REQUIRE(entry.total_probability == Catch::Approx(0.15));

    // Each channel has prob = p/15
    for (const auto& ch : entry.channels) {
        REQUIRE(ch.prob == Catch::Approx(0.01));
    }
}

TEST_CASE("Backend: noise_schedule pc points to next instruction", "[backend][noise]") {
    // The noise pc should be the index of the next quantum instruction after the noise.
    auto result = compile(R"(
        H 0
        T 0
        DEPOLARIZE1(0.01) 0
        T 0
        M 0
    )");

    // Bytecode: T (BRANCH idx=0), T (COLLIDE idx=1), ...
    // Noise pc should point to instruction after first T (i.e., the second T)
    REQUIRE(result.constant_pool.noise_schedule.size() == 1);
    REQUIRE(result.constant_pool.noise_schedule[0].pc == 1);  // Points to second T
}

TEST_CASE("Backend: quantum noise does NOT emit bytecode", "[backend][noise]") {
    auto result = compile(R"(
        X_ERROR(0.01) 0
        Y_ERROR(0.02) 1
        Z_ERROR(0.03) 2
        DEPOLARIZE1(0.04) 0
        DEPOLARIZE2(0.05) 0 1
    )");

    // No quantum instructions emitted - bytecode should be empty
    REQUIRE(result.bytecode.empty());

    // But noise_schedule should have 5 entries
    REQUIRE(result.constant_pool.noise_schedule.size() == 5);
}

TEST_CASE("Backend: OP_READOUT_NOISE emitted with inline payload", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        M(0.01) 0
    )");

    // Find the OP_READOUT_NOISE instruction
    const Instruction* readout = nullptr;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_READOUT_NOISE) {
            readout = &instr;
            break;
        }
    }
    REQUIRE(readout != nullptr);

    // Verify inline payload
    REQUIRE(readout->readout.prob == Catch::Approx(0.01));
    REQUIRE(readout->readout.meas_idx == 0);
}

TEST_CASE("Backend: multiple OP_READOUT_NOISE for multi-target noisy measurement",
          "[backend][noise]") {
    auto result = compile(R"(
        H 0
        H 1
        H 2
        M(0.02) 0 1 2
    )");

    // Count OP_READOUT_NOISE instructions
    int readout_count = 0;
    std::vector<uint32_t> meas_indices;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_READOUT_NOISE) {
            readout_count++;
            REQUIRE(instr.readout.prob == Catch::Approx(0.02));
            meas_indices.push_back(instr.readout.meas_idx);
        }
    }
    REQUIRE(readout_count == 3);

    // Measurement indices should be 0, 1, 2
    std::sort(meas_indices.begin(), meas_indices.end());
    REQUIRE(meas_indices == std::vector<uint32_t>{0, 1, 2});
}

TEST_CASE("Backend: OP_DETECTOR emitted with target list", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        M 0
        DETECTOR rec[-1]
    )");

    // Find OP_DETECTOR
    const Instruction* detector = nullptr;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_DETECTOR) {
            detector = &instr;
            break;
        }
    }
    REQUIRE(detector != nullptr);

    // Verify target list in constant pool
    uint32_t target_idx = detector->detector.target_idx;
    REQUIRE(target_idx < result.constant_pool.detector_targets.size());

    const auto& targets = result.constant_pool.detector_targets[target_idx];
    REQUIRE(targets.size() == 1);
    REQUIRE(targets[0] == 0);  // rec[-1] resolved to absolute index 0
}

TEST_CASE("Backend: OP_DETECTOR with multiple targets", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        H 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    )");

    // Find OP_DETECTOR
    const Instruction* detector = nullptr;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_DETECTOR) {
            detector = &instr;
            break;
        }
    }
    REQUIRE(detector != nullptr);

    uint32_t target_idx = detector->detector.target_idx;
    const auto& targets = result.constant_pool.detector_targets[target_idx];
    REQUIRE(targets.size() == 2);

    // rec[-1] = 1, rec[-2] = 0 (absolute indices)
    std::vector<uint32_t> sorted_targets = targets;
    std::sort(sorted_targets.begin(), sorted_targets.end());
    REQUIRE(sorted_targets == std::vector<uint32_t>{0, 1});
}

TEST_CASE("Backend: OP_OBSERVABLE emitted with obs_idx", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )");

    // Find OP_OBSERVABLE
    const Instruction* observable = nullptr;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_OBSERVABLE) {
            observable = &instr;
            break;
        }
    }
    REQUIRE(observable != nullptr);

    // Verify observable index
    REQUIRE(observable->observable.obs_idx == 0);

    // Verify target list in constant pool
    uint32_t target_idx = observable->observable.target_idx;
    REQUIRE(target_idx < result.constant_pool.observable_targets.size());

    const auto& targets = result.constant_pool.observable_targets[target_idx];
    REQUIRE(targets.size() == 1);
    REQUIRE(targets[0] == 0);
}

TEST_CASE("Backend: multiple observables with different indices", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        H 1
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(2) rec[-1]
    )");

    // Collect OP_OBSERVABLE instructions
    std::vector<const Instruction*> observables;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_OBSERVABLE) {
            observables.push_back(&instr);
        }
    }
    REQUIRE(observables.size() == 2);

    // Find obs_idx 0 and 2
    bool found_0 = false, found_2 = false;
    for (const auto* obs : observables) {
        if (obs->observable.obs_idx == 0)
            found_0 = true;
        if (obs->observable.obs_idx == 2)
            found_2 = true;
    }
    REQUIRE(found_0);
    REQUIRE(found_2);
}

TEST_CASE("Backend: num_detectors and num_observables propagate", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        H 1
        M 0 1
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-2]
    )");

    REQUIRE(result.num_detectors == 3);
    REQUIRE(result.num_observables == 2);
}

TEST_CASE("Backend: MPP with noise emits READOUT_NOISE", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        H 1
        MPP(0.005) X0*Z1
    )");

    // MPP(p) should decompose to MPP + READOUT_NOISE
    const Instruction* readout = nullptr;
    for (const auto& instr : result.bytecode) {
        if (instr.opcode == Opcode::OP_READOUT_NOISE) {
            readout = &instr;
            break;
        }
    }
    REQUIRE(readout != nullptr);
    REQUIRE(readout->readout.prob == Catch::Approx(0.005));
}

TEST_CASE("Backend: noise_schedule ordering matches HIR order", "[backend][noise]") {
    auto result = compile(R"(
        X_ERROR(0.01) 0
        H 0
        DEPOLARIZE1(0.02) 0
        T 0
        Y_ERROR(0.03) 0
        M 0
    )");

    REQUIRE(result.constant_pool.noise_schedule.size() == 3);

    // Noise entries should be in order with increasing pc
    REQUIRE(result.constant_pool.noise_schedule[0].pc == 0);  // Before T (BRANCH)
    REQUIRE(result.constant_pool.noise_schedule[1].pc == 0);  // Also before T
    REQUIRE(result.constant_pool.noise_schedule[2].pc == 1);  // Before MEASURE

    // Verify probabilities match order of noise gates
    REQUIRE(result.constant_pool.noise_schedule[0].total_probability == Catch::Approx(0.01));
    REQUIRE(result.constant_pool.noise_schedule[1].total_probability == Catch::Approx(0.02));
    REQUIRE(result.constant_pool.noise_schedule[2].total_probability == Catch::Approx(0.03));
}

TEST_CASE("Backend: noise at circuit end has pc equal to bytecode size", "[backend][noise]") {
    // Noise after the last quantum instruction should have pc = bytecode.size()
    auto result = compile(R"(
        H 0
        T 0
        M 0
        DEPOLARIZE1(0.01) 0
    )");

    // The DEPOLARIZE1 comes after M, so its pc should point past all bytecode
    REQUIRE(result.constant_pool.noise_schedule.size() == 1);

    // Count non-noise bytecode instructions
    size_t expected_bc_size = result.bytecode.size();
    REQUIRE(result.constant_pool.noise_schedule[0].pc == expected_bc_size);
}

// =============================================================================
// AGMatrix: Sparse GF(2) Transform
// =============================================================================

TEST_CASE("AGMatrix: sparse application matches Stim Tableau for entangled circuit",
          "[backend][ag]") {
    // Build a scrambling transformation (highly entangled)
    std::mt19937_64 rng(0);
    stim::TableauSimulator<kStimWidth> sim(std::move(rng), 4);
    stim::Circuit circuit;
    circuit.safe_append_u("H", {0, 1, 2, 3});
    circuit.safe_append_u("CX", {0, 1, 1, 2, 2, 3});
    circuit.safe_append_u("S", {0, 2});
    sim.safe_do_circuit(circuit);

    // Extract our fast AGMatrix from the simulator's inverse tableau
    AGMatrix mat(sim.inv_state);

    // Test various error frames: clean, sparse, dense
    std::vector<std::pair<uint64_t, uint64_t>> test_frames = {
        {0, 0},            // Clean state
        {1ULL << 2, 0},    // Single X error on qubit 2
        {0, 1ULL << 1},    // Single Z error on qubit 1
        {0b0101, 0b1010},  // Mixed X and Z errors
    };

    for (auto [test_x, test_z] : test_frames) {
        // Ground truth: Stim's PauliString-based evaluation
        stim::PauliString<kStimWidth> ps(4);
        ps.xs.u64[0] = test_x;
        ps.zs.u64[0] = test_z;
        stim::PauliString<kStimWidth> expected = sim.inv_state(ps);

        // Our sparse matrix evaluation
        uint64_t ucc_x = test_x;
        uint64_t ucc_z = test_z;
        mat.apply(ucc_x, ucc_z);

        REQUIRE(ucc_x == expected.xs.u64[0]);
        REQUIRE(ucc_z == expected.zs.u64[0]);
    }
}

TEST_CASE("AGMatrix: identity tableau is passthrough", "[backend][ag]") {
    // Identity tableau should not change the error frame
    stim::Tableau<kStimWidth> identity(4);
    AGMatrix mat(identity);

    uint64_t x = 0b1010;
    uint64_t z = 0b0101;
    mat.apply(x, z);

    REQUIRE(x == 0b1010);
    REQUIRE(z == 0b0101);
}

TEST_CASE("AGMatrix: zero error frame stays zero", "[backend][ag]") {
    std::mt19937_64 rng(42);
    stim::TableauSimulator<kStimWidth> sim(std::move(rng), 3);
    stim::Circuit circuit;
    circuit.safe_append_u("H", {0, 1, 2});
    circuit.safe_append_u("CX", {0, 1, 1, 2});
    sim.safe_do_circuit(circuit);

    AGMatrix mat(sim.inv_state);

    uint64_t x = 0;
    uint64_t z = 0;
    mat.apply(x, z);

    REQUIRE(x == 0);
    REQUIRE(z == 0);
}

TEST_CASE("AGMatrix: all single-qubit errors for 6-qubit circuit", "[backend][ag]") {
    // Exhaustively test every single-qubit X and Z error
    std::mt19937_64 rng(123);
    stim::TableauSimulator<kStimWidth> sim(std::move(rng), 6);
    stim::Circuit circuit;
    circuit.safe_append_u("H", {0, 2, 4});
    circuit.safe_append_u("CX", {0, 1, 2, 3, 4, 5});
    circuit.safe_append_u("S", {1, 3, 5});
    circuit.safe_append_u("H", {1, 3, 5});
    sim.safe_do_circuit(circuit);

    AGMatrix mat(sim.inv_state);

    for (int q = 0; q < 6; ++q) {
        // Test X error on qubit q
        {
            stim::PauliString<kStimWidth> ps(6);
            ps.xs.u64[0] = 1ULL << q;
            ps.zs.u64[0] = 0;
            auto expected = sim.inv_state(ps);

            uint64_t ux = 1ULL << q;
            uint64_t uz = 0;
            mat.apply(ux, uz);
            REQUIRE(ux == expected.xs.u64[0]);
            REQUIRE(uz == expected.zs.u64[0]);
        }
        // Test Z error on qubit q
        {
            stim::PauliString<kStimWidth> ps(6);
            ps.xs.u64[0] = 0;
            ps.zs.u64[0] = 1ULL << q;
            auto expected = sim.inv_state(ps);

            uint64_t ux = 0;
            uint64_t uz = 1ULL << q;
            mat.apply(ux, uz);
            REQUIRE(ux == expected.xs.u64[0]);
            REQUIRE(uz == expected.zs.u64[0]);
        }
    }
}

TEST_CASE("Backend: cumulative_hazards computed alongside noise_schedule", "[backend][noise]") {
    auto result = compile(R"(
        X_ERROR(0.01) 0
        H 0
        DEPOLARIZE1(0.02) 0
        T 0
        Y_ERROR(0.03) 0
        M 0
    )");

    const auto& hazards = result.constant_pool.cumulative_hazards;
    const auto& schedule = result.constant_pool.noise_schedule;

    REQUIRE(hazards.size() == schedule.size());
    REQUIRE(hazards.size() == 3);

    // h_i = -ln(1 - p_i), H_k = sum h_0..h_k
    double h0 = -std::log(1.0 - 0.01);
    double h1 = -std::log(1.0 - 0.02);
    double h2 = -std::log(1.0 - 0.03);

    REQUIRE(hazards[0] == Catch::Approx(h0));
    REQUIRE(hazards[1] == Catch::Approx(h0 + h1));
    REQUIRE(hazards[2] == Catch::Approx(h0 + h1 + h2));

    // Strictly monotonically increasing
    REQUIRE(hazards[0] > 0.0);
    REQUIRE(hazards[1] > hazards[0]);
    REQUIRE(hazards[2] > hazards[1]);
}

TEST_CASE("Backend: cumulative_hazards empty when no noise", "[backend][noise]") {
    auto result = compile(R"(
        H 0
        T 0
        M 0
    )");

    REQUIRE(result.constant_pool.cumulative_hazards.empty());
    REQUIRE(result.constant_pool.noise_schedule.empty());
}

TEST_CASE("Backend: cumulative_hazards clamps probability near 1.0", "[backend][noise]") {
    // X_ERROR(1.0) would make p=1, log(0)=-inf. Must clamp.
    auto result = compile(R"(
        X_ERROR(1.0) 0
        M 0
    )");

    REQUIRE(result.constant_pool.cumulative_hazards.size() == 1);
    // Must be finite (clamped, not -inf)
    REQUIRE(std::isfinite(result.constant_pool.cumulative_hazards[0]));
    REQUIRE(result.constant_pool.cumulative_hazards[0] > 0.0);
}
