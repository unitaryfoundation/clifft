#include "clifft/backend/compiler_context.h"
#include "clifft/backend/reference_syndrome.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/remove_noise_pass.h"

#include "test_helpers.h"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <string>

using namespace clifft;
using namespace clifft::internal;
using clifft::test::test_lcg;
using clifft::test::X;
using clifft::test::Z;

// =============================================================================
// Helpers
// =============================================================================

// Build a PauliString from explicit X and Z bitmasks.
static stim::PauliString<kStimWidth> make_pauli(uint32_t n, uint64_t x_bits, uint64_t z_bits,
                                                bool sign = false) {
    stim::PauliString<kStimWidth> p(n);
    p.xs.u64[0] = x_bits;
    p.zs.u64[0] = z_bits;
    p.sign = sign;
    return p;
}

// Build a Stim Tableau from a sequence of VM bytecode instructions.
// Only handles frame/array gate opcodes (CNOT, CZ, H, S, SWAP).
// This reconstructs the virtual-coordinate transformation from the raw
// bytecode, independently of v_cum, to verify the compiler's emission.
static stim::Tableau<kStimWidth> bytecode_to_tableau(const std::vector<Instruction>& bytecode,
                                                     uint32_t num_qubits) {
    stim::Tableau<kStimWidth> tab(num_qubits);
    {
        // Use transposed append to mirror how the compiler builds v_local.
        // This ensures sign tracking matches the compiler exactly.
        stim::TableauTransposedRaii<kStimWidth> trans(tab);
        for (const auto& instr : bytecode) {
            uint16_t a = instr.axis_1;
            uint16_t b = instr.axis_2;
            switch (instr.opcode) {
                case Opcode::OP_FRAME_CNOT:
                case Opcode::OP_ARRAY_CNOT:
                    trans.append_ZCX(a, b);
                    break;
                case Opcode::OP_FRAME_CZ:
                case Opcode::OP_ARRAY_CZ:
                    trans.append_ZCZ(a, b);
                    break;
                case Opcode::OP_FRAME_H:
                case Opcode::OP_ARRAY_H:
                    trans.append_H_XZ(a);
                    break;
                case Opcode::OP_FRAME_S:
                case Opcode::OP_ARRAY_S:
                    trans.append_S(a);
                    break;
                case Opcode::OP_FRAME_S_DAG:
                case Opcode::OP_ARRAY_S_DAG:
                    // S^dag = S^3; the transposed RAII has no append_S_DAG
                    trans.append_S(a);
                    trans.append_S(a);
                    trans.append_S(a);
                    break;
                case Opcode::OP_FRAME_SWAP:
                case Opcode::OP_ARRAY_SWAP:
                    // Match the 3-CNOT decomposition emitted by the compiler
                    trans.append_ZCX(a, b);
                    trans.append_ZCX(b, a);
                    trans.append_ZCX(a, b);
                    break;
                default:
                    break;
            }
        }
    }  // trans goes out of scope and un-transposes
    return tab;
}

// Verify that V_cum P V_cum^dag is a single-qubit Pauli on the expected pivot.
// Returns the localized PauliString for further inspection.
static stim::PauliString<kStimWidth> verify_localization(CompilerContext& ctx,
                                                         const stim::PauliString<kStimWidth>& input,
                                                         const LocalizationResult& result) {
    ctx.virtual_frame.flush();
    stim::PauliString<kStimWidth> localized = ctx.virtual_frame.materialized_tableau()(input);
    uint64_t cx = localized.xs.u64[0];
    uint64_t cz = localized.zs.u64[0];

    // Must act on exactly one qubit
    uint64_t support = cx | cz;
    REQUIRE(support != 0);
    REQUIRE((support & (support - 1)) == 0);  // Power of two = single bit

    // The single bit must be the declared pivot
    uint16_t actual_pivot = static_cast<uint16_t>(std::countr_zero(static_cast<uint64_t>(support)));
    REQUIRE(actual_pivot == result.pivot);

    // Basis must match
    if (result.basis == LocalizedBasis::X_BASIS) {
        REQUIRE((cx & (1ULL << result.pivot)) != 0);
        REQUIRE((cz & (1ULL << result.pivot)) == 0);
    } else {
        REQUIRE((cx & (1ULL << result.pivot)) == 0);
        REQUIRE((cz & (1ULL << result.pivot)) != 0);
    }

    // Sign must match
    REQUIRE(localized.sign == result.sign);

    return localized;
}

// Verify that the emitted bytecode independently transforms the input Pauli
// to a weight-1 Pauli on the declared pivot. This catches bugs where v_cum
// is updated correctly but the bytecode emission is wrong (e.g., swapped
// control/target on a CNOT).
static void verify_bytecode_localization(const CompilerContext& ctx,
                                         const stim::PauliString<kStimWidth>& input,
                                         const LocalizationResult& result) {
    stim::Tableau<kStimWidth> tab =
        bytecode_to_tableau(ctx.bytecode, static_cast<uint32_t>(input.num_qubits));
    stim::PauliString<kStimWidth> localized = tab(input);
    uint64_t cx = localized.xs.u64[0];
    uint64_t cz = localized.zs.u64[0];

    uint64_t support = cx | cz;
    REQUIRE(support != 0);
    REQUIRE((support & (support - 1)) == 0);

    uint16_t actual_pivot = static_cast<uint16_t>(std::countr_zero(static_cast<uint64_t>(support)));
    REQUIRE(actual_pivot == result.pivot);

    if (result.basis == LocalizedBasis::X_BASIS) {
        REQUIRE((cx & (1ULL << result.pivot)) != 0);
        REQUIRE((cz & (1ULL << result.pivot)) == 0);
    } else {
        REQUIRE((cx & (1ULL << result.pivot)) == 0);
        REQUIRE((cz & (1ULL << result.pivot)) != 0);
    }

    REQUIRE(localized.sign == result.sign);

    // Verify ARRAY vs FRAME opcode semantics: array opcodes must target
    // active axes (< active_k), frame opcodes must target dormant axes
    // (>= active_k). localize_pauli never changes active_k, so the final
    // value is valid for all emitted instructions.
    uint32_t k = ctx.reg_manager.active_k();
    for (const auto& instr : ctx.bytecode) {
        uint16_t a = instr.axis_1;
        uint16_t b = instr.axis_2;
        switch (instr.opcode) {
            case Opcode::OP_ARRAY_CNOT:
            case Opcode::OP_ARRAY_CZ:
            case Opcode::OP_ARRAY_SWAP:
            case Opcode::OP_ARRAY_H:
            case Opcode::OP_ARRAY_S:
            case Opcode::OP_ARRAY_S_DAG:
                REQUIRE(a < k);
                if (instr.opcode == Opcode::OP_ARRAY_CNOT || instr.opcode == Opcode::OP_ARRAY_CZ ||
                    instr.opcode == Opcode::OP_ARRAY_SWAP) {
                    REQUIRE(b < k);
                }
                break;
            case Opcode::OP_FRAME_CNOT:
                // Frame CNOT: control is dormant, target can be active or dormant
                REQUIRE(a >= k);
                break;
            case Opcode::OP_FRAME_CZ:
                // Frame CZ: at least one operand is dormant
                REQUIRE((a >= k || b >= k));
                break;
            case Opcode::OP_FRAME_H:
            case Opcode::OP_FRAME_S:
            case Opcode::OP_FRAME_S_DAG:
                REQUIRE(a >= k);
                break;
            case Opcode::OP_FRAME_SWAP:
                REQUIRE(a >= k);
                REQUIRE(b >= k);
                break;
            default:
                break;
        }
    }
}

// Count opcodes of a given type in the bytecode.
static uint32_t count_opcodes(const std::vector<Instruction>& bytecode, Opcode op) {
    uint32_t count = 0;
    for (const auto& instr : bytecode) {
        if (instr.opcode == op)
            ++count;
    }
    return count;
}

// Check that NO array opcodes were emitted (all frame-only).
static bool all_frame_opcodes(const std::vector<Instruction>& bytecode) {
    for (const auto& instr : bytecode) {
        if (instr.opcode == Opcode::OP_ARRAY_CNOT || instr.opcode == Opcode::OP_ARRAY_CZ ||
            instr.opcode == Opcode::OP_ARRAY_SWAP || instr.opcode == Opcode::OP_ARRAY_H ||
            instr.opcode == Opcode::OP_ARRAY_S) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Single-qubit Paulis: no localization needed
// =============================================================================

TEST_CASE("Localize: single-qubit Z needs no gates") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, 0, Z(2));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.pivot == 2);
    REQUIRE(result.basis == LocalizedBasis::Z_BASIS);
    REQUIRE(result.sign == false);
    REQUIRE(ctx.bytecode.empty());
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: single-qubit X needs no gates") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(3), 0);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.pivot == 3);
    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    REQUIRE(result.sign == false);
    REQUIRE(ctx.bytecode.empty());
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: single-qubit Y emits S gate") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(1), Z(1));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.pivot == 1);
    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    REQUIRE(result.sign == true);
    REQUIRE(count_opcodes(ctx.bytecode, Opcode::OP_FRAME_S) == 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// Pure Z-strings: Case 2
// =============================================================================

TEST_CASE("Localize: two-qubit ZZ string") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, 0, Z(0) | Z(2));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::Z_BASIS);
    // One CNOT needed to fold the second Z onto the pivot
    REQUIRE(ctx.bytecode.size() == 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: three-qubit ZZZ string") {
    CompilerContext ctx(5);
    auto pauli = make_pauli(5, 0, Z(0) | Z(2) | Z(4));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::Z_BASIS);
    // Two CNOTs needed
    REQUIRE(ctx.bytecode.size() == 2);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// X-support strings: Case 1
// =============================================================================

TEST_CASE("Localize: two-qubit XX string") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0) | X(1), 0);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    REQUIRE(ctx.bytecode.size() == 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: XX with Z residue needs CNOT plus CZ") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0) | X(1), Z(1));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: XZ mixed two-qubit") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0), Z(1));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.pivot == 0);
    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    REQUIRE(ctx.bytecode.size() == 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// Sign tracking
// =============================================================================

TEST_CASE("Localize: negative Z preserves sign") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, 0, Z(0), /*sign=*/true);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.sign == true);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: negative X preserves sign") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0), 0, /*sign=*/true);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.sign == true);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: negative Y gives positive X after S") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0), Z(0), /*sign=*/true);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    // -Y -> S -> -(-X) = +X, so sign should be false
    REQUIRE(result.sign == false);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// Dormant pivot preference
// =============================================================================

TEST_CASE("Localize: X-localization prefers dormant pivot") {
    CompilerContext ctx(4);
    // Activate axis 0 (k=1), so axis 0 is active, 1..3 are dormant.
    ctx.reg_manager.activate();

    // X0 X1: axis 0 is active, axis 1 is dormant -> prefer dormant pivot.
    auto pauli = make_pauli(4, X(0) | X(1), 0);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.pivot == 1);
    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    REQUIRE(all_frame_opcodes(ctx.bytecode));
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: Z-localization prefers active pivot") {
    CompilerContext ctx(4);
    // k=1: axis 0 active, axes 1..3 dormant.
    ctx.reg_manager.activate();

    // Z0 Z1: axis 0 is active -> prefer it as pivot.
    // CNOT(1->0) has dormant control -> frame opcode.
    auto pauli = make_pauli(4, 0, Z(0) | Z(1));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.pivot == 0);
    REQUIRE(result.basis == LocalizedBasis::Z_BASIS);
    REQUIRE(all_frame_opcodes(ctx.bytecode));
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// Active-active opcodes
// =============================================================================

TEST_CASE("Localize: all-active X-support emits array opcodes") {
    CompilerContext ctx(4);
    // k=2: axes 0,1 active.
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();

    auto pauli = make_pauli(4, X(0) | X(1), 0);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    REQUIRE(count_opcodes(ctx.bytecode, Opcode::OP_ARRAY_CNOT) == 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: all-active CZ emits array CZ") {
    CompilerContext ctx(4);
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();

    auto pauli = make_pauli(4, X(0), Z(1));
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.pivot == 0);
    REQUIRE(count_opcodes(ctx.bytecode, Opcode::OP_ARRAY_CZ) == 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// Heavy random Pauli strings: fuzz test
// =============================================================================

TEST_CASE("Localize: random heavy Paulis localize to weight-1") {
    uint64_t seed = 0xDEADBEEF;

    for (int trial = 0; trial < 100; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        const uint32_t n = 20;
        CompilerContext ctx(n);

        // Activate a random contiguous prefix 0..k-1.
        uint32_t k = static_cast<uint32_t>(test_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = test_lcg(seed) & qubit_mask;
        uint64_t z_bits = test_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = localize_pauli(ctx, pauli);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

TEST_CASE("Localize: random heavy Paulis with sign") {
    uint64_t seed = 0xCAFEBABE;

    for (int trial = 0; trial < 50; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        const uint32_t n = 15;
        CompilerContext ctx(n);

        uint32_t k = static_cast<uint32_t>(test_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = test_lcg(seed) & qubit_mask;
        uint64_t z_bits = test_lcg(seed) & qubit_mask;
        bool sign = (test_lcg(seed) & 1) != 0;
        if ((x_bits | z_bits) == 0) {
            z_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, sign);
        auto result = localize_pauli(ctx, pauli);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

// =============================================================================
// All-dormant: frame-only opcodes
// =============================================================================

TEST_CASE("Localize: all-dormant heavy Pauli emits only frame opcodes") {
    uint64_t seed = 0x12345678;

    for (int trial = 0; trial < 50; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        const uint32_t n = 16;
        CompilerContext ctx(n);  // All dormant

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = test_lcg(seed) & qubit_mask;
        uint64_t z_bits = test_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = localize_pauli(ctx, pauli);

        REQUIRE(all_frame_opcodes(ctx.bytecode));
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

// =============================================================================
// Sequential localizations: V_cum accumulates correctly
// =============================================================================

// Verify localization for a sequential call.
// v_cum_before: snapshot of v_cum BEFORE this localize_pauli call.
// v_cum_after: v_cum AFTER this localize_pauli call.
// The local frame is: v_local = v_cum_before^{-1}.then(v_cum_after)
// and v_local(input) should be the weight-1 localized Pauli.
static void verify_sequential_localization(const CompilerContext& ctx,
                                           const stim::Tableau<kStimWidth>& v_cum_before,
                                           const stim::Tableau<kStimWidth>& v_cum_after,
                                           const stim::PauliString<kStimWidth>& input,
                                           const LocalizationResult& result, size_t bc_before) {
    stim::Tableau<kStimWidth> v_local = v_cum_before.inverse().then(v_cum_after);
    stim::PauliString<kStimWidth> localized = v_local(input);

    uint64_t cx = localized.xs.u64[0];
    uint64_t cz = localized.zs.u64[0];
    uint64_t support = cx | cz;
    REQUIRE(support != 0);
    REQUIRE((support & (support - 1)) == 0);

    uint16_t actual_pivot = static_cast<uint16_t>(std::countr_zero(static_cast<uint64_t>(support)));
    REQUIRE(actual_pivot == result.pivot);

    if (result.basis == LocalizedBasis::X_BASIS) {
        REQUIRE((cx & (1ULL << result.pivot)) != 0);
        REQUIRE((cz & (1ULL << result.pivot)) == 0);
    } else {
        REQUIRE((cx & (1ULL << result.pivot)) == 0);
        REQUIRE((cz & (1ULL << result.pivot)) != 0);
    }

    REQUIRE(localized.sign == result.sign);

    // Verify bytecode emitted during this localization step independently
    // produces the correct single-qubit Pauli on the declared pivot.
    std::vector<Instruction> step_bytecode(ctx.bytecode.begin() + static_cast<ptrdiff_t>(bc_before),
                                           ctx.bytecode.end());
    stim::Tableau<kStimWidth> tab =
        bytecode_to_tableau(step_bytecode, static_cast<uint32_t>(input.num_qubits));
    stim::PauliString<kStimWidth> bc_localized = tab(input);
    uint64_t bcx = bc_localized.xs.u64[0];
    uint64_t bcz = bc_localized.zs.u64[0];
    uint64_t bc_support = bcx | bcz;
    REQUIRE(bc_support != 0);
    REQUIRE((bc_support & (bc_support - 1)) == 0);
    REQUIRE(static_cast<uint16_t>(std::countr_zero(static_cast<uint64_t>(bc_support))) ==
            result.pivot);
    REQUIRE(bc_localized.sign == result.sign);
}

TEST_CASE("Localize: sequential localizations accumulate in V_cum") {
    const uint32_t n = 6;
    CompilerContext ctx(n);

    auto p1 = make_pauli(n, X(0) | X(1) | X(2), 0);
    auto p2 = make_pauli(n, 0, Z(3) | Z(4) | Z(5));
    auto p3 = make_pauli(n, X(2) | X(4), Z(1) | Z(3));

    // First localization: v_cum starts as identity, so verify_localization works.
    auto r1 = localize_pauli(ctx, p1);
    verify_localization(ctx, p1, r1);

    // Snapshot v_cum before second localization.
    ctx.virtual_frame.flush();
    stim::Tableau<kStimWidth> snap1 = ctx.virtual_frame.materialized_tableau();
    size_t bc1 = ctx.bytecode.size();
    auto r2 = localize_pauli(ctx, p2);
    ctx.virtual_frame.flush();
    verify_sequential_localization(ctx, snap1, ctx.virtual_frame.materialized_tableau(), p2, r2,
                                   bc1);

    // Snapshot v_cum before third localization.
    stim::Tableau<kStimWidth> snap2 = ctx.virtual_frame.materialized_tableau();
    size_t bc2 = ctx.bytecode.size();
    auto r3 = localize_pauli(ctx, p3);
    ctx.virtual_frame.flush();
    verify_sequential_localization(ctx, snap2, ctx.virtual_frame.materialized_tableau(), p3, r3,
                                   bc2);

    // V_cum should be non-identity after multiple localizations.
    bool is_identity = true;
    for (uint32_t q = 0; q < n; ++q) {
        const auto& v_cum = ctx.virtual_frame.materialized_tableau();
        if (v_cum.xs[q].xs.u64[0] != (1ULL << q) || v_cum.xs[q].zs.u64[0] != 0 ||
            v_cum.zs[q].xs.u64[0] != 0 || v_cum.zs[q].zs.u64[0] != (1ULL << q)) {
            is_identity = false;
            break;
        }
    }
    REQUIRE_FALSE(is_identity);
}

// =============================================================================
// Opcode axis values use mapped array axes, not virtual qubit indices
// =============================================================================

TEST_CASE("Localize: array opcode axes are literal axis indices") {
    CompilerContext ctx(8);
    // k=3: axes 0,1,2 are active.
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();

    // X0 X2: both active, CNOT will use literal axes 0 and 2.
    auto pauli = make_pauli(8, X(0) | X(2), 0);
    auto result = localize_pauli(ctx, pauli);

    bool found = false;
    for (const auto& instr : ctx.bytecode) {
        if (instr.opcode == Opcode::OP_ARRAY_CNOT) {
            // Axes in the opcode are the literal axis values.
            REQUIRE((instr.axis_1 == 0 || instr.axis_1 == 2));
            REQUIRE((instr.axis_2 == 0 || instr.axis_2 == 2));
            REQUIRE(instr.axis_1 != instr.axis_2);
            found = true;
        }
    }
    REQUIRE(found);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// Edge case: wide Pauli covering all qubits
// =============================================================================

TEST_CASE("Localize: full-width X string on 20 qubits") {
    const uint32_t n = 20;
    CompilerContext ctx(n);
    uint64_t all_x = (1ULL << n) - 1;
    auto pauli = make_pauli(n, all_x, 0);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    // n-1 CNOTs to clear all X except pivot
    REQUIRE(ctx.bytecode.size() == n - 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: full-width Z string on 20 qubits") {
    const uint32_t n = 20;
    CompilerContext ctx(n);
    uint64_t all_z = (1ULL << n) - 1;
    auto pauli = make_pauli(n, 0, all_z);
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::Z_BASIS);
    REQUIRE(ctx.bytecode.size() == n - 1);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize: full-width Y string on 10 qubits") {
    const uint32_t n = 10;
    CompilerContext ctx(n);
    uint64_t all = (1ULL << n) - 1;
    auto pauli = make_pauli(n, all, all);  // Y on every qubit
    auto result = localize_pauli(ctx, pauli);

    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

// =============================================================================
// VirtualRegisterManager unit tests
// =============================================================================

TEST_CASE("VirtualRegisterManager: contiguous active-dormant split") {
    VirtualRegisterManager mgr(8);

    REQUIRE(mgr.active_k() == 0);
    REQUIRE(mgr.is_dormant(0));
    REQUIRE(mgr.is_dormant(7));

    // activate() promotes axis k to active.
    mgr.activate();  // k=1: axis 0 active
    REQUIRE(mgr.is_active(0));
    REQUIRE(mgr.is_dormant(1));
    REQUIRE(mgr.active_k() == 1);

    mgr.activate();  // k=2: axes 0,1 active
    REQUIRE(mgr.is_active(1));
    REQUIRE(mgr.is_dormant(2));
    REQUIRE(mgr.active_k() == 2);

    // deactivate() demotes axis k-1 to dormant.
    mgr.deactivate();  // k=1: axis 1 now dormant
    REQUIRE(mgr.is_dormant(1));
    REQUIRE(mgr.active_k() == 1);
    REQUIRE(mgr.is_active(0));
}

TEST_CASE("VirtualRegisterManager: peak tracking") {
    VirtualRegisterManager mgr(4);

    mgr.activate();
    REQUIRE(mgr.peak_k() == 1);
    mgr.activate();
    REQUIRE(mgr.peak_k() == 2);

    mgr.deactivate();
    REQUIRE(mgr.peak_k() == 2);  // Peak doesn't decrease
    REQUIRE(mgr.active_k() == 1);
}

// =============================================================================
// Gap Sampling Hazard Array Tests
// =============================================================================

TEST_CASE("Backend: Gap sampling hazard array accumulation") {
    CompilerContext ctx(3);

    NoiseSite site1;
    site1.channels.push_back({1, 0, 0.5});
    NoiseSite site2;
    site2.channels.push_back({2, 0, 0.75});
    NoiseSite site3;
    site3.channels.push_back({4, 0, 1.0});  // clamped to 1.0 - 2^-53

    HirModule hir;
    hir.num_qubits = 3;
    hir.noise_sites.push_back(std::move(site1));
    hir.noise_sites.push_back(std::move(site2));
    hir.noise_sites.push_back(std::move(site3));

    hir.ops.push_back(HeisenbergOp::make_noise(NoiseSiteIdx{0}));
    hir.ops.push_back(HeisenbergOp::make_noise(NoiseSiteIdx{1}));
    hir.ops.push_back(HeisenbergOp::make_noise(NoiseSiteIdx{2}));

    CompiledModule prog = lower(hir);

    REQUIRE(prog.constant_pool.noise_hazards.size() == 3);

    double h1 = -std::log1p(-0.5);
    double h2 = h1 - std::log1p(-0.75);
    double h3 = h2 - std::log1p(-(1.0 - 0x1.0p-53));

    CHECK_THAT(prog.constant_pool.noise_hazards[0], Catch::Matchers::WithinAbs(h1, 1e-5));
    CHECK_THAT(prog.constant_pool.noise_hazards[1], Catch::Matchers::WithinAbs(h2, 1e-5));
    CHECK_THAT(prog.constant_pool.noise_hazards[2], Catch::Matchers::WithinAbs(h3, 1e-5));
}

// =============================================================================
// Scaled Localizer Fuzzing
// =============================================================================

TEST_CASE("Localize fuzz: 30-qubit heavy Paulis 500 trials") {
    uint64_t seed = 0xA5A5A5A5;
    const uint32_t n = 30;

    for (int trial = 0; trial < 500; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        CompilerContext ctx(n);

        uint32_t k = static_cast<uint32_t>(test_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = test_lcg(seed) & qubit_mask;
        uint64_t z_bits = test_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = localize_pauli(ctx, pauli);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

TEST_CASE("Localize fuzz: 64-qubit max-width Paulis") {
    uint64_t seed = 0xFEEDFACE;
    const uint32_t n = 64;

    for (int trial = 0; trial < 200; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        CompilerContext ctx(n);

        uint32_t k = static_cast<uint32_t>(test_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        // Full 64-bit masks, no masking needed
        uint64_t x_bits = test_lcg(seed);
        uint64_t z_bits = test_lcg(seed);
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = localize_pauli(ctx, pauli);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

// =============================================================================
// Adversarial Patterns
// =============================================================================

TEST_CASE("Localize adversarial: all-Y strings") {
    // Every qubit is Y = XZ. Maximizes S-gate emissions.
    for (uint32_t n = 2; n <= 30; n += 4) {
        CompilerContext ctx(n);
        uint64_t mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;
        auto pauli = make_pauli(n, mask, mask);  // Y on every qubit
        auto result = localize_pauli(ctx, pauli);

        REQUIRE(result.basis == LocalizedBasis::X_BASIS);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

TEST_CASE("Localize adversarial: all-Y with varied active partitions") {
    const uint32_t n = 20;
    uint64_t mask = (1ULL << n) - 1;

    for (uint32_t k = 0; k <= n; k += 3) {
        CompilerContext ctx(n);
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }
        auto pauli = make_pauli(n, mask, mask);
        auto result = localize_pauli(ctx, pauli);

        REQUIRE(result.basis == LocalizedBasis::X_BASIS);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

TEST_CASE("Localize adversarial: checkerboard X-Z pattern") {
    // Even qubits get X, odd qubits get Z. Stresses CZ residue cleanup.
    for (uint32_t n = 4; n <= 30; n += 4) {
        CompilerContext ctx(n);
        uint64_t x_bits = 0;
        uint64_t z_bits = 0;
        for (uint32_t q = 0; q < n; ++q) {
            if (q % 2 == 0)
                x_bits |= (1ULL << q);
            else
                z_bits |= (1ULL << q);
        }
        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = localize_pauli(ctx, pauli);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

TEST_CASE("Localize adversarial: single X rest Z") {
    // One X bit on q0, Z on all others. Stresses Z-cleanup after X pivot.
    for (uint32_t n = 2; n <= 30; n += 4) {
        CompilerContext ctx(n);
        uint64_t z_mask = ((n < 64) ? ((1ULL << n) - 1) : ~0ULL) & ~1ULL;
        auto pauli = make_pauli(n, X(0), z_mask);
        auto result = localize_pauli(ctx, pauli);

        REQUIRE(result.pivot == 0);
        REQUIRE(result.basis == LocalizedBasis::X_BASIS);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

TEST_CASE("Localize adversarial: single X rest Z with active pivot") {
    // Same pattern but with the X qubit active and some Z qubits dormant.
    const uint32_t n = 20;
    CompilerContext ctx(n);
    ctx.reg_manager.activate();  // axis 0 active

    uint64_t z_mask = ((1ULL << n) - 1) & ~1ULL;  // Z on qubits 1..19
    auto pauli = make_pauli(n, X(0), z_mask);
    auto result = localize_pauli(ctx, pauli);

    // Pivot should still be 0 (the X qubit) since it's the only X bit
    REQUIRE(result.pivot == 0);
    REQUIRE(result.basis == LocalizedBasis::X_BASIS);
    verify_localization(ctx, pauli, result);
    verify_bytecode_localization(ctx, pauli, result);
}

TEST_CASE("Localize adversarial: dense XZ overlap") {
    // High Hamming weight on both X and Z masks (many Y qubits mixed with
    // pure X and pure Z). Maximizes total gate count.
    uint64_t seed = 0xBAADF00D;
    const uint32_t n = 30;

    for (int trial = 0; trial < 100; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        CompilerContext ctx(n);
        uint32_t k = static_cast<uint32_t>(test_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        // Generate masks with high density (~75% bits set)
        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = (test_lcg(seed) | test_lcg(seed)) & qubit_mask;
        uint64_t z_bits = (test_lcg(seed) | test_lcg(seed)) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = localize_pauli(ctx, pauli);
        verify_localization(ctx, pauli, result);
        verify_bytecode_localization(ctx, pauli, result);
    }
}

// =============================================================================
// Sequential Localization Stress
// =============================================================================

TEST_CASE("Localize sequential: 20 localizations on 20 qubits") {
    uint64_t seed = 0x1337C0DE;
    const uint32_t n = 20;
    CompilerContext ctx(n);

    // Activate half the qubits for a realistic mixed partition
    for (uint32_t i = 0; i < 10; ++i) {
        ctx.reg_manager.activate();
    }

    for (int step = 0; step < 20; ++step) {
        ctx.virtual_frame.flush();
        stim::Tableau<kStimWidth> snap = ctx.virtual_frame.materialized_tableau();
        size_t bc_snap = ctx.bytecode.size();

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = test_lcg(seed) & qubit_mask;
        uint64_t z_bits = test_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, (test_lcg(seed) & 1) != 0);
        auto result = localize_pauli(ctx, pauli);
        ctx.virtual_frame.flush();
        verify_sequential_localization(ctx, snap, ctx.virtual_frame.materialized_tableau(), pauli,
                                       result, bc_snap);
    }
}

TEST_CASE("Localize sequential: 30 localizations on 30 qubits all-active") {
    uint64_t seed = 0xABCDEF01;
    const uint32_t n = 30;
    CompilerContext ctx(n);

    for (uint32_t i = 0; i < n; ++i) {
        ctx.reg_manager.activate();
    }

    for (int step = 0; step < 30; ++step) {
        ctx.virtual_frame.flush();
        stim::Tableau<kStimWidth> snap = ctx.virtual_frame.materialized_tableau();
        size_t bc_snap = ctx.bytecode.size();

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = test_lcg(seed) & qubit_mask;
        uint64_t z_bits = test_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            z_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, (test_lcg(seed) & 1) != 0);
        auto result = localize_pauli(ctx, pauli);
        ctx.virtual_frame.flush();
        verify_sequential_localization(ctx, snap, ctx.virtual_frame.materialized_tableau(), pauli,
                                       result, bc_snap);
    }
}

TEST_CASE("Localize sequential: 30 localizations on 30 qubits all-dormant") {
    uint64_t seed = 0x99887766;
    const uint32_t n = 30;
    CompilerContext ctx(n);  // k=0, all dormant

    for (int step = 0; step < 30; ++step) {
        ctx.virtual_frame.flush();
        stim::Tableau<kStimWidth> snap = ctx.virtual_frame.materialized_tableau();
        size_t bc_snap = ctx.bytecode.size();

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = test_lcg(seed) & qubit_mask;
        uint64_t z_bits = test_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, (test_lcg(seed) & 1) != 0);
        auto result = localize_pauli(ctx, pauli);
        ctx.virtual_frame.flush();
        verify_sequential_localization(ctx, snap, ctx.virtual_frame.materialized_tableau(), pauli,
                                       result, bc_snap);

        // All-dormant should only emit frame opcodes
        REQUIRE(all_frame_opcodes(ctx.bytecode));
    }
}

// =============================================================================
// Active/Dormant Boundary - EXPAND Emission
// =============================================================================

// Helper: compile a circuit string through the full pipeline and return the module.
static CompiledModule compile_circuit(const std::string& text) {
    auto circuit = clifft::parse(text);
    auto hir = clifft::trace(circuit);
    return clifft::lower(hir);
}

TEST_CASE("Lower: T on single qubit emits EXPAND") {
    // H 0; T 0: the H is absorbed into the tableau. The T gate's rewound Pauli
    // lands on a dormant X-basis axis, requiring EXPAND to activate it.
    auto mod = compile_circuit("H 0\nT 0");

    uint32_t expand_count = count_opcodes(mod.bytecode, Opcode::OP_EXPAND);
    CHECK(expand_count == 1);
    CHECK(mod.peak_rank >= 1);
}

TEST_CASE("Lower: T on dormant Z-basis needs no EXPAND") {
    // T 0 (no H): the rewound Pauli is Z_0, which is dormant Z-basis.
    // This should NOT emit EXPAND -- zero-cost dormant property.
    auto mod = compile_circuit("T 0");

    uint32_t expand_count = count_opcodes(mod.bytecode, Opcode::OP_EXPAND);
    CHECK(expand_count == 0);
    CHECK(mod.peak_rank == 0);
}

TEST_CASE("Lower: two T gates on same dormant Z-basis emit no EXPAND") {
    // T 0; T 0: both Paulis are Z_0 on dormant axis. No expansion needed.
    auto mod = compile_circuit("T 0\nT 0");

    uint32_t expand_count = count_opcodes(mod.bytecode, Opcode::OP_EXPAND);
    CHECK(expand_count == 0);
    CHECK(mod.peak_rank == 0);
}

TEST_CASE("Lower: T after entanglement may require EXPAND") {
    // H 0; CX 0 1; T 0: the T gate's rewound Pauli is multi-qubit (X0*X1)
    // because the CX entangled the qubits. After localization, the pivot
    // may land on a dormant X-basis qubit, requiring EXPAND.
    auto mod = compile_circuit("H 0\nCX 0 1\nT 0");

    uint32_t expand_count = count_opcodes(mod.bytecode, Opcode::OP_EXPAND);
    CHECK(expand_count >= 1);
    CHECK(mod.peak_rank >= 1);
}

TEST_CASE("Lower: measurement after T deactivates - peak rank is bounded") {
    // H 0; T 0; M 0: EXPAND activates axis for T, measurement deactivates.
    // peak_rank should be exactly 1 (breathes 0 -> 1 -> 0).
    auto mod = compile_circuit("H 0\nT 0\nM 0");

    CHECK(mod.peak_rank == 1);
    uint32_t expand_count = count_opcodes(mod.bytecode, Opcode::OP_EXPAND);
    CHECK(expand_count == 1);

    // Should have exactly one active measurement opcode
    uint32_t diag_count = count_opcodes(mod.bytecode, Opcode::OP_MEAS_ACTIVE_DIAGONAL);
    uint32_t interf_count = count_opcodes(mod.bytecode, Opcode::OP_MEAS_ACTIVE_INTERFERE);
    CHECK(diag_count + interf_count == 1);
}

TEST_CASE("Lower: breathing lifecycle - repeated T then M keeps peak rank low") {
    // H 0; T 0; M 0; H 1; T 1; M 1
    // Each qubit breathes 0->1->0 independently. Peak rank should be 1.
    auto mod = compile_circuit("H 0\nT 0\nM 0\nH 1\nT 1\nM 1");

    CHECK(mod.peak_rank == 1);
}

TEST_CASE("Lower: two simultaneous T gates may raise peak rank to 2") {
    // H 0; H 1; T 0; T 1; M 0; M 1
    // Both T gates need activation before either is measured.
    auto mod = compile_circuit("H 0\nH 1\nT 0\nT 1\nM 0\nM 1");

    CHECK(mod.peak_rank >= 1);
    // Verify correct number of measurements
    CHECK(mod.num_measurements == 2);
}

TEST_CASE("Lower: pure Clifford circuit emits no EXPAND or T opcodes") {
    // A pure Clifford circuit should emit no EXPAND, no ARRAY_T, no ARRAY_T_DAG.
    // All gates are absorbed into the tableau.
    auto mod = compile_circuit("H 0\nCX 0 1\nS 1\nH 1\nM 0\nM 1");

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_EXPAND) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T_DAG) == 0);
}

// =============================================================================
// S Absorption via Peephole + Backend Lowering Tests
// =============================================================================

// compile_optimized runs parse -> trace -> peephole -> lower.
// T+T fusion now absorbs S into the Pauli frame (no CLIFFORD_PHASE).
static CompiledModule compile_optimized(const std::string& text) {
    auto circuit = clifft::parse(text);
    auto hir = clifft::trace(circuit);
    clifft::PeepholeFusionPass pass;
    pass.run(hir);
    return clifft::lower(hir);
}

TEST_CASE("Lower: H-T-T absorbs S -- no S or T opcodes emitted") {
    // H 0; T 0; T 0: optimizer absorbs the fused S into the coordinate frame.
    // No runtime S or T opcodes should appear.
    auto mod = compile_optimized("H 0\nT 0\nT 0");

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T_DAG) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_S) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_FRAME_S) == 0);
    CHECK(mod.peak_rank == 0);
}

TEST_CASE("Lower: T-T on dormant Z absorbs S offline -- no runtime ops") {
    // T 0; T 0 (no H): fused S absorbed into frame. Nothing to lower.
    auto mod = compile_optimized("T 0\nT 0");

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_EXPAND) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_FRAME_S) == 0);
    CHECK(mod.peak_rank == 0);
}

TEST_CASE("Lower: T_dag-T_dag absorbs S_dag offline") {
    auto mod = compile_optimized("T_DAG 0\nT_DAG 0");

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T_DAG) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_S) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_S_DAG) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_FRAME_S) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_FRAME_S_DAG) == 0);
}

TEST_CASE("Lower: T-T_dag cancels completely") {
    auto mod = compile_optimized("T 0\nT_DAG 0");

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_T_DAG) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_S) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_FRAME_S) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_EXPAND) == 0);
    CHECK(mod.peak_rank == 0);
}

// =============================================================================
// Postselection Mask Tests
// =============================================================================

TEST_CASE("Lower: postselection_mask emits OP_POSTSELECT for flagged detectors") {
    // Circuit with 2 detectors: flag detector 0 for postselection.
    std::string text =
        "M 0\n"
        "M 1\n"
        "DETECTOR rec[-1]\n"
        "DETECTOR rec[-2]\n";

    auto circuit = clifft::parse(text);
    auto hir = clifft::trace(circuit);

    // Mask: detector 0 is postselected, detector 1 is normal
    std::vector<uint8_t> mask = {1, 0};
    auto mod = clifft::lower(hir, mask);

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_POSTSELECT) == 1);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_DETECTOR) == 1);
}

TEST_CASE("Lower: empty postselection_mask emits only OP_DETECTOR") {
    std::string text =
        "M 0\n"
        "DETECTOR rec[-1]\n";

    auto circuit = clifft::parse(text);
    auto hir = clifft::trace(circuit);
    auto mod = clifft::lower(hir);

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_POSTSELECT) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_DETECTOR) == 1);
}

TEST_CASE("Lower: short postselection_mask only affects present indices") {
    // Mask shorter than detector count: only first detector is postselected.
    std::string text =
        "M 0\n"
        "M 1\n"
        "DETECTOR rec[-1]\n"
        "DETECTOR rec[-2]\n"
        "DETECTOR rec[-1]\n";

    auto circuit = clifft::parse(text);
    auto hir = clifft::trace(circuit);

    std::vector<uint8_t> mask = {1};  // Only 1 element, 3 detectors
    auto mod = clifft::lower(hir, mask);

    CHECK(count_opcodes(mod.bytecode, Opcode::OP_POSTSELECT) == 1);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_DETECTOR) == 2);
}

TEST_CASE("Lower: ARRAY_ROTATION geometric artifact correction", "[backend][rotation]") {
    // Manually construct a ARRAY_ROTATION with a +Y Pauli on qubit 0.
    // +Y localizes to -Z via the virtual frame, flipping result.sign.
    // The Back-End must correct global_weight for this geometric artifact.
    HirModule hir;
    hir.num_qubits = 1;
    hir.global_weight = {1.0, 0.0};

    // +Y on qubit 0: destab=X(0), stab=Z(0), sign=false
    hir.ops.push_back(HeisenbergOp::make_phase_rotation(X(0), Z(0), false, 0.5));

    auto mod = clifft::lower(hir);

    // +Y localizes to -Z, so result.sign is true while op.sign() is false.
    // The Back-End should apply a phase correction of e^{i * 0.5 * pi} = +i.
    CHECK_THAT(mod.constant_pool.global_weight.real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    CHECK_THAT(mod.constant_pool.global_weight.imag(), Catch::Matchers::WithinAbs(1.0, 1e-12));

    // The emitted VM instruction should have alpha inverted to -0.5.
    // z = e^{i * (-0.5) * pi} = -i
    bool found_rot = false;
    for (const auto& instr : mod.bytecode) {
        if (instr.opcode == Opcode::OP_ARRAY_ROT || instr.opcode == Opcode::OP_EXPAND_ROT) {
            CHECK_THAT(instr.math.weight_re, Catch::Matchers::WithinAbs(0.0, 1e-12));
            CHECK_THAT(instr.math.weight_im, Catch::Matchers::WithinAbs(-1.0, 1e-12));
            found_rot = true;
        }
    }
    REQUIRE(found_rot);
}

// =============================================================================
// Error Syndrome Normalization - Backend
// =============================================================================

TEST_CASE("lower injects FLAG_EXPECTED_ONE into detector instructions") {
    auto circuit = parse(
        "X 0\n"
        "M 0\n"
        "DETECTOR rec[-1]\n"
        "DETECTOR rec[-1]\n");
    auto hir = clifft::trace(circuit);

    // expected_detectors: det0 -> 1, det1 -> 0
    std::vector<uint8_t> expected_det = {1, 0};
    auto mod = clifft::lower(hir, {}, expected_det, {});

    // Find the two OP_DETECTOR instructions
    std::vector<const Instruction*> dets;
    for (const auto& instr : mod.bytecode) {
        if (instr.opcode == Opcode::OP_DETECTOR) {
            dets.push_back(&instr);
        }
    }
    REQUIRE(dets.size() == 2);
    CHECK((dets[0]->flags & Instruction::FLAG_EXPECTED_ONE) != 0);
    CHECK((dets[1]->flags & Instruction::FLAG_EXPECTED_ONE) == 0);
}

TEST_CASE("lower injects FLAG_EXPECTED_ONE into postselect instructions") {
    auto circuit = parse(
        "X 0\n"
        "M 0\n"
        "DETECTOR rec[-1]\n"
        "DETECTOR rec[-1]\n");
    auto hir = clifft::trace(circuit);

    // Postselect det0, expected_detectors: det0 -> 1
    std::vector<uint8_t> ps_mask = {1, 0};
    std::vector<uint8_t> expected_det = {1, 0};
    auto mod = clifft::lower(hir, ps_mask, expected_det, {});

    // det0 should be OP_POSTSELECT with FLAG_EXPECTED_ONE
    const Instruction* ps = nullptr;
    const Instruction* det = nullptr;
    for (const auto& instr : mod.bytecode) {
        if (instr.opcode == Opcode::OP_POSTSELECT)
            ps = &instr;
        if (instr.opcode == Opcode::OP_DETECTOR)
            det = &instr;
    }
    REQUIRE(ps != nullptr);
    REQUIRE(det != nullptr);
    CHECK((ps->flags & Instruction::FLAG_EXPECTED_ONE) != 0);
    CHECK((det->flags & Instruction::FLAG_EXPECTED_ONE) == 0);
}

TEST_CASE("lower stores expected_observables in CompiledModule") {
    auto circuit = parse(
        "M 0\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n");
    auto hir = clifft::trace(circuit);

    std::vector<uint8_t> expected_obs = {1};
    auto mod = clifft::lower(hir, {}, {}, expected_obs);

    REQUIRE(mod.expected_observables.size() == 1);
    CHECK(mod.expected_observables[0] == 1);
}

TEST_CASE("RemoveNoisePass strips noise ops and clears side-tables") {
    auto circuit = parse(
        "X_ERROR(0.1) 0\n"
        "M 0\n"
        "DETECTOR rec[-1]\n");
    auto hir = clifft::trace(circuit);

    REQUIRE(hir.noise_sites.size() > 0);
    size_t original_ops = hir.ops.size();

    RemoveNoisePass strip;
    strip.run(hir);

    CHECK(hir.ops.size() < original_ops);
    CHECK(hir.noise_sites.empty());
    CHECK(hir.readout_noise.empty());

    // No NOISE ops remain
    for (const auto& op : hir.ops) {
        CHECK(op.op_type() != OpType::NOISE);
        CHECK(op.op_type() != OpType::READOUT_NOISE);
    }
}

TEST_CASE("compute_reference_syndrome extracts expected parities") {
    // X 0 -> M 0 produces measurement=1
    // DETECTOR rec[-1] -> raw parity = 1
    // OBSERVABLE_INCLUDE(0) rec[-1] -> raw parity = 1
    auto circuit = parse(
        "X 0\n"
        "M 0\n"
        "DETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n");
    auto hir = clifft::trace(circuit);

    auto ref = compute_reference_syndrome(hir);
    REQUIRE(ref.detectors.size() == 1);
    REQUIRE(ref.observables.size() == 1);
    CHECK(ref.detectors[0] == 1);
    CHECK(ref.observables[0] == 1);
}

TEST_CASE("compute_reference_syndrome ignores noise") {
    // Same circuit but with noise - reference should still be clean
    auto circuit = parse(
        "X 0\n"
        "X_ERROR(1.0) 0\n"  // Would flip back to 0 if not stripped
        "M 0\n"
        "DETECTOR rec[-1]\n");
    auto hir = clifft::trace(circuit);

    auto ref = compute_reference_syndrome(hir);
    REQUIRE(ref.detectors.size() == 1);
    // Reference should see M0=1 (X applied), not M0=0 (X then X_ERROR)
    CHECK(ref.detectors[0] == 1);
}

// =============================================================================
// EXP_VAL Lowering
// =============================================================================

TEST_CASE("Lower: EXP_VAL emits OP_EXP_VAL") {
    auto mod = compile_circuit("EXP_VAL Z0");
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_EXP_VAL) == 1);
    CHECK(mod.num_exp_vals == 1);
}

TEST_CASE("Lower: EXP_VAL multi-product emits multiple OP_EXP_VAL") {
    auto mod = compile_circuit("EXP_VAL X0 Z1");
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_EXP_VAL) == 2);
    CHECK(mod.num_exp_vals == 2);
}

TEST_CASE("Lower: EXP_VAL does not emit routing instructions") {
    auto mod = compile_circuit("H 0\nEXP_VAL X0");
    // No SWAPs, EXPANDs, or other routing should be emitted around EXP_VAL
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_EXPAND) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_ARRAY_SWAP) == 0);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_EXP_VAL) == 1);
}

TEST_CASE("Lower: EXP_VAL constant pool mask is correct") {
    // Z0 after no Cliffords: virtual Pauli should be Z on axis 0
    auto mod = compile_circuit("EXP_VAL Z0");
    REQUIRE(mod.constant_pool.exp_val_masks.size() == 1);
    const auto& pm = mod.constant_pool.exp_val_masks[0];
    CHECK(pm.x.is_zero());  // No X support
    CHECK(pm.z.w[0] == 1);  // Z on qubit 0
    CHECK(!pm.sign);
}

TEST_CASE("Lower: EXP_VAL does not affect measurement count") {
    auto mod = compile_circuit("H 0\nEXP_VAL X0\nM 0");
    CHECK(mod.num_measurements == 1);
    CHECK(mod.num_exp_vals == 1);
}

TEST_CASE("Lower: queued virtual gates affect later EXP_VAL masks") {
    HirModule hir;
    hir.num_qubits = 1;
    hir.num_exp_vals = 1;

    // T on an X-basis dormant axis queues a virtual H and EXPAND without
    // immediately materializing the H into v_cum.
    hir.ops.push_back(HeisenbergOp::make_tgate(X(0), 0, false));
    hir.ops.push_back(HeisenbergOp::make_exp_val(X(0), 0, false, ExpValIdx{0}));

    auto mod = lower(hir);

    REQUIRE(mod.constant_pool.exp_val_masks.size() == 1);
    const auto& pm = mod.constant_pool.exp_val_masks[0];
    CHECK(pm.x.is_zero());
    CHECK(pm.z == Z(0));
    CHECK(!pm.sign);
}

TEST_CASE("Lower: queued virtual gates affect later measurements") {
    HirModule hir;
    hir.num_qubits = 1;
    hir.num_measurements = 1;

    // The queued virtual H from the T gate should map a later X probe to Z,
    // making the active measurement diagonal instead of interfering.
    hir.ops.push_back(HeisenbergOp::make_tgate(X(0), 0, false));
    hir.ops.push_back(HeisenbergOp::make_measure(X(0), 0, false, MeasRecordIdx{0}));

    auto mod = lower(hir);

    CHECK(mod.peak_rank == 1);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_MEAS_ACTIVE_DIAGONAL) == 1);
    CHECK(count_opcodes(mod.bytecode, Opcode::OP_MEAS_ACTIVE_INTERFERE) == 0);
}

TEST_CASE("Lower: queued virtual gates affect later noise masks") {
    HirModule hir;
    hir.num_qubits = 1;

    NoiseSite site;
    site.channels.push_back({X(0), 0, 0.25});
    hir.noise_sites.push_back(site);

    // Leave a virtual H pending, then map an X-error noise site through it.
    hir.ops.push_back(HeisenbergOp::make_tgate(X(0), 0, false));
    hir.ops.push_back(HeisenbergOp::make_noise(NoiseSiteIdx{0}));

    auto mod = lower(hir);

    REQUIRE(mod.constant_pool.noise_sites.size() == 1);
    REQUIRE(mod.constant_pool.noise_sites[0].channels.size() == 1);
    const auto& ch = mod.constant_pool.noise_sites[0].channels[0];
    CHECK(ch.destab_mask.is_zero());
    CHECK(ch.stab_mask == Z(0));
    CHECK_THAT(ch.prob, Catch::Matchers::WithinAbs(0.25, 1e-12));
}
