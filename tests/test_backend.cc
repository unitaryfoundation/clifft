#include "ucc/backend/compiler_context.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>

using namespace ucc;
using namespace ucc::internal;
using ucc::test::X;
using ucc::test::Y;
using ucc::test::Z;

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

// Verify that V_cum P V_cum^dag is a single-qubit Pauli on the expected pivot.
// Returns the compressed PauliString for further inspection.
static stim::PauliString<kStimWidth> verify_compression(const CompilerContext& ctx,
                                                        const stim::PauliString<kStimWidth>& input,
                                                        const CompressionResult& result) {
    stim::PauliString<kStimWidth> compressed = ctx.v_cum(input);
    uint64_t cx = compressed.xs.u64[0];
    uint64_t cz = compressed.zs.u64[0];

    // Must act on exactly one qubit
    uint64_t support = cx | cz;
    REQUIRE(support != 0);
    REQUIRE((support & (support - 1)) == 0);  // Power of two = single bit

    // The single bit must be the declared pivot
    uint16_t actual_pivot = static_cast<uint16_t>(__builtin_ctzll(support));
    REQUIRE(actual_pivot == result.pivot);

    // Basis must match
    if (result.basis == CompressedBasis::X_BASIS) {
        REQUIRE((cx & (1ULL << result.pivot)) != 0);
        REQUIRE((cz & (1ULL << result.pivot)) == 0);
    } else {
        REQUIRE((cx & (1ULL << result.pivot)) == 0);
        REQUIRE((cz & (1ULL << result.pivot)) != 0);
    }

    // Sign must match
    REQUIRE(compressed.sign == result.sign);

    return compressed;
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
            instr.opcode == Opcode::OP_ARRAY_SWAP) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Single-qubit Paulis: no compression needed
// =============================================================================

TEST_CASE("Compress: single-qubit Z needs no gates") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, 0, Z(2));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.pivot == 2);
    REQUIRE(result.basis == CompressedBasis::Z_BASIS);
    REQUIRE(result.sign == false);
    REQUIRE(ctx.bytecode.empty());
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: single-qubit X needs no gates") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(3), 0);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.pivot == 3);
    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    REQUIRE(result.sign == false);
    REQUIRE(ctx.bytecode.empty());
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: single-qubit Y emits S gate") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, Y(1), Y(1));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.pivot == 1);
    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    // Y_1 -> S maps to -X_1, so sign flips from the original
    REQUIRE(result.sign == true);
    REQUIRE(count_opcodes(ctx.bytecode, Opcode::OP_FRAME_S) == 1);
    verify_compression(ctx, pauli, result);
}

// =============================================================================
// Pure Z-strings: Case 2
// =============================================================================

TEST_CASE("Compress: two-qubit ZZ string") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, 0, Z(0) | Z(2));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::Z_BASIS);
    // One CNOT needed to fold the second Z onto the pivot
    REQUIRE(ctx.bytecode.size() == 1);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: three-qubit ZZZ string") {
    CompilerContext ctx(5);
    auto pauli = make_pauli(5, 0, Z(0) | Z(2) | Z(4));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::Z_BASIS);
    // Two CNOTs needed
    REQUIRE(ctx.bytecode.size() == 2);
    verify_compression(ctx, pauli, result);
}

// =============================================================================
// X-support strings: Case 1
// =============================================================================

TEST_CASE("Compress: two-qubit XX string") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0) | X(1), 0);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    REQUIRE(ctx.bytecode.size() == 1);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: XX with Z residue needs CNOT plus CZ") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0) | X(1), Z(1));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: XZ mixed two-qubit") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0), Z(1));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.pivot == 0);
    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    REQUIRE(ctx.bytecode.size() == 1);
    verify_compression(ctx, pauli, result);
}

// =============================================================================
// Sign tracking
// =============================================================================

TEST_CASE("Compress: negative Z preserves sign") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, 0, Z(0), /*sign=*/true);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.sign == true);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: negative X preserves sign") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, X(0), 0, /*sign=*/true);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.sign == true);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: negative Y gives positive X after S") {
    CompilerContext ctx(4);
    auto pauli = make_pauli(4, Y(0), Y(0), /*sign=*/true);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    // -Y -> S -> -(-X) = +X, so sign should be false
    REQUIRE(result.sign == false);
    verify_compression(ctx, pauli, result);
}

// =============================================================================
// Dormant pivot preference
// =============================================================================

TEST_CASE("Compress: X-compression prefers dormant pivot") {
    CompilerContext ctx(4);
    // Activate axis 0 (k=1), so axis 0 is active, 1..3 are dormant.
    ctx.reg_manager.activate();

    // X0 X1: axis 0 is active, axis 1 is dormant -> prefer dormant pivot.
    auto pauli = make_pauli(4, X(0) | X(1), 0);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.pivot == 1);
    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    REQUIRE(all_frame_opcodes(ctx.bytecode));
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: Z-compression prefers active pivot") {
    CompilerContext ctx(4);
    // k=1: axis 0 active, axes 1..3 dormant.
    ctx.reg_manager.activate();

    // Z0 Z1: axis 0 is active -> prefer it as pivot.
    // CNOT(1->0) has dormant control -> frame opcode.
    auto pauli = make_pauli(4, 0, Z(0) | Z(1));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.pivot == 0);
    REQUIRE(result.basis == CompressedBasis::Z_BASIS);
    REQUIRE(all_frame_opcodes(ctx.bytecode));
    verify_compression(ctx, pauli, result);
}

// =============================================================================
// Active-active opcodes
// =============================================================================

TEST_CASE("Compress: all-active X-support emits array opcodes") {
    CompilerContext ctx(4);
    // k=2: axes 0,1 active.
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();

    auto pauli = make_pauli(4, X(0) | X(1), 0);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    REQUIRE(count_opcodes(ctx.bytecode, Opcode::OP_ARRAY_CNOT) == 1);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: all-active CZ emits array CZ") {
    CompilerContext ctx(4);
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();

    auto pauli = make_pauli(4, X(0), Z(1));
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.pivot == 0);
    REQUIRE(count_opcodes(ctx.bytecode, Opcode::OP_ARRAY_CZ) == 1);
    verify_compression(ctx, pauli, result);
}

// =============================================================================
// Heavy random Pauli strings: fuzz test
// =============================================================================

TEST_CASE("Compress: random heavy Paulis compress to weight-1") {
    uint64_t seed = 0xDEADBEEF;
    auto next_rand = [&seed]() -> uint64_t {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return seed;
    };

    for (int trial = 0; trial < 100; ++trial) {
        const uint32_t n = 20;
        CompilerContext ctx(n);

        // Activate a random contiguous prefix 0..k-1.
        uint32_t k = static_cast<uint32_t>(next_rand() % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = next_rand() & qubit_mask;
        uint64_t z_bits = next_rand() & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = compress_pauli(ctx, pauli);
        verify_compression(ctx, pauli, result);
    }
}

TEST_CASE("Compress: random heavy Paulis with sign") {
    uint64_t seed = 0xCAFEBABE;
    auto next_rand = [&seed]() -> uint64_t {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return seed;
    };

    for (int trial = 0; trial < 50; ++trial) {
        const uint32_t n = 15;
        CompilerContext ctx(n);

        uint32_t k = static_cast<uint32_t>(next_rand() % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = next_rand() & qubit_mask;
        uint64_t z_bits = next_rand() & qubit_mask;
        bool sign = (next_rand() & 1) != 0;
        if ((x_bits | z_bits) == 0) {
            z_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, sign);
        auto result = compress_pauli(ctx, pauli);
        verify_compression(ctx, pauli, result);
    }
}

// =============================================================================
// All-dormant: frame-only opcodes
// =============================================================================

TEST_CASE("Compress: all-dormant heavy Pauli emits only frame opcodes") {
    uint64_t seed = 0x12345678;
    auto next_rand = [&seed]() -> uint64_t {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return seed;
    };

    for (int trial = 0; trial < 50; ++trial) {
        const uint32_t n = 16;
        CompilerContext ctx(n);  // All dormant

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = next_rand() & qubit_mask;
        uint64_t z_bits = next_rand() & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = compress_pauli(ctx, pauli);

        REQUIRE(all_frame_opcodes(ctx.bytecode));
        verify_compression(ctx, pauli, result);
    }
}

// =============================================================================
// Sequential compressions: V_cum accumulates correctly
// =============================================================================

// Verify compression for a sequential call.
// v_cum_before: snapshot of v_cum BEFORE this compress_pauli call.
// v_cum_after: v_cum AFTER this compress_pauli call.
// The local frame is: v_local = v_cum_before^{-1}.then(v_cum_after)
// and v_local(input) should be the weight-1 compressed Pauli.
static void verify_sequential_compression(const stim::Tableau<kStimWidth>& v_cum_before,
                                          const stim::Tableau<kStimWidth>& v_cum_after,
                                          const stim::PauliString<kStimWidth>& input,
                                          const CompressionResult& result) {
    // v_cum gates are appended: v_cum_after = v_local composed with v_cum_before.
    // Stim's append model: v_cum_after(P) = v_local(v_cum_before(P)).
    // So v_cum_after = v_cum_before.then(v_local).
    // Therefore v_local = v_cum_before.inverse().then(v_cum_after).
    stim::Tableau<kStimWidth> v_local = v_cum_before.inverse().then(v_cum_after);
    stim::PauliString<kStimWidth> compressed = v_local(input);

    uint64_t cx = compressed.xs.u64[0];
    uint64_t cz = compressed.zs.u64[0];
    uint64_t support = cx | cz;
    REQUIRE(support != 0);
    REQUIRE((support & (support - 1)) == 0);

    uint16_t actual_pivot = static_cast<uint16_t>(__builtin_ctzll(support));
    REQUIRE(actual_pivot == result.pivot);

    if (result.basis == CompressedBasis::X_BASIS) {
        REQUIRE((cx & (1ULL << result.pivot)) != 0);
        REQUIRE((cz & (1ULL << result.pivot)) == 0);
    } else {
        REQUIRE((cx & (1ULL << result.pivot)) == 0);
        REQUIRE((cz & (1ULL << result.pivot)) != 0);
    }

    REQUIRE(compressed.sign == result.sign);
}

TEST_CASE("Compress: sequential compressions accumulate in V_cum") {
    const uint32_t n = 6;
    CompilerContext ctx(n);

    auto p1 = make_pauli(n, X(0) | X(1) | X(2), 0);
    auto p2 = make_pauli(n, 0, Z(3) | Z(4) | Z(5));
    auto p3 = make_pauli(n, X(2) | X(4), Z(1) | Z(3));

    // First compression: v_cum starts as identity, so verify_compression works.
    auto r1 = compress_pauli(ctx, p1);
    verify_compression(ctx, p1, r1);

    // Snapshot v_cum before second compression.
    stim::Tableau<kStimWidth> snap1 = ctx.v_cum;
    auto r2 = compress_pauli(ctx, p2);
    verify_sequential_compression(snap1, ctx.v_cum, p2, r2);

    // Snapshot v_cum before third compression.
    stim::Tableau<kStimWidth> snap2 = ctx.v_cum;
    auto r3 = compress_pauli(ctx, p3);
    verify_sequential_compression(snap2, ctx.v_cum, p3, r3);

    // V_cum should be non-identity after multiple compressions.
    bool is_identity = true;
    for (uint32_t q = 0; q < n; ++q) {
        if (ctx.v_cum.xs[q].xs.u64[0] != (1ULL << q) || ctx.v_cum.xs[q].zs.u64[0] != 0 ||
            ctx.v_cum.zs[q].xs.u64[0] != 0 || ctx.v_cum.zs[q].zs.u64[0] != (1ULL << q)) {
            is_identity = false;
            break;
        }
    }
    REQUIRE_FALSE(is_identity);
}

// =============================================================================
// Opcode axis values use mapped array axes, not virtual qubit indices
// =============================================================================

TEST_CASE("Compress: array opcode axes are literal axis indices") {
    CompilerContext ctx(8);
    // k=3: axes 0,1,2 are active.
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();
    ctx.reg_manager.activate();

    // X0 X2: both active, CNOT will use literal axes 0 and 2.
    auto pauli = make_pauli(8, X(0) | X(2), 0);
    auto result = compress_pauli(ctx, pauli);

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
    verify_compression(ctx, pauli, result);
}

// =============================================================================
// Edge case: wide Pauli covering all qubits
// =============================================================================

TEST_CASE("Compress: full-width X string on 20 qubits") {
    const uint32_t n = 20;
    CompilerContext ctx(n);
    uint64_t all_x = (1ULL << n) - 1;
    auto pauli = make_pauli(n, all_x, 0);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    // n-1 CNOTs to clear all X except pivot
    REQUIRE(ctx.bytecode.size() == n - 1);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: full-width Z string on 20 qubits") {
    const uint32_t n = 20;
    CompilerContext ctx(n);
    uint64_t all_z = (1ULL << n) - 1;
    auto pauli = make_pauli(n, 0, all_z);
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::Z_BASIS);
    REQUIRE(ctx.bytecode.size() == n - 1);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress: full-width Y string on 10 qubits") {
    const uint32_t n = 10;
    CompilerContext ctx(n);
    uint64_t all = (1ULL << n) - 1;
    auto pauli = make_pauli(n, all, all);  // Y on every qubit
    auto result = compress_pauli(ctx, pauli);

    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    verify_compression(ctx, pauli, result);
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
    site3.channels.push_back({4, 0, 1.0});  // clamped to 0.9999

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

    double h1 = -std::log(1.0 - 0.5);
    double h2 = h1 - std::log(1.0 - 0.75);
    double h3 = h2 - std::log(1.0 - 0.9999);

    CHECK_THAT(prog.constant_pool.noise_hazards[0], Catch::Matchers::WithinAbs(h1, 1e-5));
    CHECK_THAT(prog.constant_pool.noise_hazards[1], Catch::Matchers::WithinAbs(h2, 1e-5));
    CHECK_THAT(prog.constant_pool.noise_hazards[2], Catch::Matchers::WithinAbs(h3, 1e-5));
}

// =============================================================================
// Phase 2 Hardening: Scaled Compressor Fuzzing
// =============================================================================

// Simple LCG for deterministic test-local RNG.
static uint64_t fuzz_lcg(uint64_t& seed) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
}

TEST_CASE("Compress fuzz: 30-qubit heavy Paulis 500 trials") {
    uint64_t seed = 0xA5A5A5A5;
    const uint32_t n = 30;

    for (int trial = 0; trial < 500; ++trial) {
        CompilerContext ctx(n);

        uint32_t k = static_cast<uint32_t>(fuzz_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = fuzz_lcg(seed) & qubit_mask;
        uint64_t z_bits = fuzz_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = compress_pauli(ctx, pauli);
        verify_compression(ctx, pauli, result);
    }
}

TEST_CASE("Compress fuzz: 64-qubit max-width Paulis") {
    uint64_t seed = 0xFEEDFACE;
    const uint32_t n = 64;

    for (int trial = 0; trial < 200; ++trial) {
        CompilerContext ctx(n);

        uint32_t k = static_cast<uint32_t>(fuzz_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        // Full 64-bit masks, no masking needed
        uint64_t x_bits = fuzz_lcg(seed);
        uint64_t z_bits = fuzz_lcg(seed);
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = compress_pauli(ctx, pauli);
        verify_compression(ctx, pauli, result);
    }
}

// =============================================================================
// Phase 2 Hardening: Adversarial Patterns
// =============================================================================

TEST_CASE("Compress adversarial: all-Y strings") {
    // Every qubit is Y = XZ. Maximizes S-gate emissions.
    for (uint32_t n = 2; n <= 30; n += 4) {
        CompilerContext ctx(n);
        uint64_t mask = (n < 64) ? ((1ULL << n) - 1) : ~0ULL;
        auto pauli = make_pauli(n, mask, mask);  // Y on every qubit
        auto result = compress_pauli(ctx, pauli);

        REQUIRE(result.basis == CompressedBasis::X_BASIS);
        verify_compression(ctx, pauli, result);
    }
}

TEST_CASE("Compress adversarial: all-Y with varied active partitions") {
    const uint32_t n = 20;
    uint64_t mask = (1ULL << n) - 1;

    for (uint32_t k = 0; k <= n; k += 3) {
        CompilerContext ctx(n);
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }
        auto pauli = make_pauli(n, mask, mask);
        auto result = compress_pauli(ctx, pauli);

        REQUIRE(result.basis == CompressedBasis::X_BASIS);
        verify_compression(ctx, pauli, result);
    }
}

TEST_CASE("Compress adversarial: checkerboard X-Z pattern") {
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
        auto result = compress_pauli(ctx, pauli);
        verify_compression(ctx, pauli, result);
    }
}

TEST_CASE("Compress adversarial: single X rest Z") {
    // One X bit on q0, Z on all others. Stresses Z-cleanup after X pivot.
    for (uint32_t n = 2; n <= 30; n += 4) {
        CompilerContext ctx(n);
        uint64_t z_mask = ((n < 64) ? ((1ULL << n) - 1) : ~0ULL) & ~1ULL;
        auto pauli = make_pauli(n, X(0), z_mask);
        auto result = compress_pauli(ctx, pauli);

        REQUIRE(result.pivot == 0);
        REQUIRE(result.basis == CompressedBasis::X_BASIS);
        verify_compression(ctx, pauli, result);
    }
}

TEST_CASE("Compress adversarial: single X rest Z with active pivot") {
    // Same pattern but with the X qubit active and some Z qubits dormant.
    const uint32_t n = 20;
    CompilerContext ctx(n);
    ctx.reg_manager.activate();  // axis 0 active

    uint64_t z_mask = ((1ULL << n) - 1) & ~1ULL;  // Z on qubits 1..19
    auto pauli = make_pauli(n, X(0), z_mask);
    auto result = compress_pauli(ctx, pauli);

    // Pivot should still be 0 (the X qubit) since it's the only X bit
    REQUIRE(result.pivot == 0);
    REQUIRE(result.basis == CompressedBasis::X_BASIS);
    verify_compression(ctx, pauli, result);
}

TEST_CASE("Compress adversarial: dense XZ overlap") {
    // High Hamming weight on both X and Z masks (many Y qubits mixed with
    // pure X and pure Z). Maximizes total gate count.
    uint64_t seed = 0xBAADF00D;
    const uint32_t n = 30;

    for (int trial = 0; trial < 100; ++trial) {
        CompilerContext ctx(n);
        uint32_t k = static_cast<uint32_t>(fuzz_lcg(seed) % (n + 1));
        for (uint32_t i = 0; i < k; ++i) {
            ctx.reg_manager.activate();
        }

        // Generate masks with high density (~75% bits set)
        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = (fuzz_lcg(seed) | fuzz_lcg(seed)) & qubit_mask;
        uint64_t z_bits = (fuzz_lcg(seed) | fuzz_lcg(seed)) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits);
        auto result = compress_pauli(ctx, pauli);
        verify_compression(ctx, pauli, result);
    }
}

// =============================================================================
// Phase 2 Hardening: Sequential Compression Stress
// =============================================================================

TEST_CASE("Compress sequential: 20 compressions on 20 qubits") {
    uint64_t seed = 0x1337C0DE;
    const uint32_t n = 20;
    CompilerContext ctx(n);

    // Activate half the qubits for a realistic mixed partition
    for (uint32_t i = 0; i < 10; ++i) {
        ctx.reg_manager.activate();
    }

    for (int step = 0; step < 20; ++step) {
        stim::Tableau<kStimWidth> snap = ctx.v_cum;

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = fuzz_lcg(seed) & qubit_mask;
        uint64_t z_bits = fuzz_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, (fuzz_lcg(seed) & 1) != 0);
        auto result = compress_pauli(ctx, pauli);
        verify_sequential_compression(snap, ctx.v_cum, pauli, result);
    }
}

TEST_CASE("Compress sequential: 30 compressions on 30 qubits all-active") {
    uint64_t seed = 0xABCDEF01;
    const uint32_t n = 30;
    CompilerContext ctx(n);

    for (uint32_t i = 0; i < n; ++i) {
        ctx.reg_manager.activate();
    }

    for (int step = 0; step < 30; ++step) {
        stim::Tableau<kStimWidth> snap = ctx.v_cum;

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = fuzz_lcg(seed) & qubit_mask;
        uint64_t z_bits = fuzz_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            z_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, (fuzz_lcg(seed) & 1) != 0);
        auto result = compress_pauli(ctx, pauli);
        verify_sequential_compression(snap, ctx.v_cum, pauli, result);
    }
}

TEST_CASE("Compress sequential: 30 compressions on 30 qubits all-dormant") {
    uint64_t seed = 0x99887766;
    const uint32_t n = 30;
    CompilerContext ctx(n);  // k=0, all dormant

    for (int step = 0; step < 30; ++step) {
        stim::Tableau<kStimWidth> snap = ctx.v_cum;

        uint64_t qubit_mask = (1ULL << n) - 1;
        uint64_t x_bits = fuzz_lcg(seed) & qubit_mask;
        uint64_t z_bits = fuzz_lcg(seed) & qubit_mask;
        if ((x_bits | z_bits) == 0) {
            x_bits = 1;
        }

        auto pauli = make_pauli(n, x_bits, z_bits, (fuzz_lcg(seed) & 1) != 0);
        auto result = compress_pauli(ctx, pauli);
        verify_sequential_compression(snap, ctx.v_cum, pauli, result);

        // All-dormant should only emit frame opcodes
        REQUIRE(all_frame_opcodes(ctx.bytecode));
    }
}
