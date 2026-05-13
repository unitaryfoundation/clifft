// Tests for the C++ probability_of() entry point.
//
// probability_of() computes exact log-probabilities for a batch of
// measurement records against a compiled circuit. It mirrors what
// sample() would emit, modulo dust-clamping, via a bytecode rewrite that
// turns OP_MEAS_* into their OP_MEAS_*_FORCED siblings and replays each
// record through execute().
//
// These tests cover the entry-point contract: pre-flight validation,
// correct handling of the rewrite + per-record loop, unreachable
// records, EXP_VAL pass-through, and that the input CompiledModule is
// not mutated. Deeper algorithmic correctness is covered by the
// per-kernel and dispatcher tests in earlier steps, and Python-level
// equivalence with probabilities() is covered separately.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/svm/svm.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::WithinAbs;

namespace {

CompiledModule compile_circuit(const std::string& text) {
    auto circuit = clifft::parse(text);
    auto hir = clifft::trace(circuit);
    return clifft::lower(hir);
}

}  // namespace

// =============================================================================
// Basic correctness: small circuits with known closed-form probabilities.
// =============================================================================

TEST_CASE("probability_of: Bell state probabilities") {
    auto mod = compile_circuit("H 0\nCX 0 1\nM 0 1");
    REQUIRE(mod.num_measurements == 2);

    // Records "00", "01", "10", "11" in measurement-record order.
    std::vector<uint8_t> records{0, 0, 0, 1, 1, 0, 1, 1};
    auto log_probs = probability_of(mod, records, /*num_records=*/4);
    REQUIRE(log_probs.size() == 4);

    // Bell state: only 00 and 11 are emitted, each with probability 0.5.
    CHECK_THAT(std::exp(log_probs[0]), WithinAbs(0.5, 1e-12));
    REQUIRE(std::isinf(log_probs[1]));
    REQUIRE(log_probs[1] < 0);  // -inf for unreachable
    REQUIRE(std::isinf(log_probs[2]));
    REQUIRE(log_probs[2] < 0);
    CHECK_THAT(std::exp(log_probs[3]), WithinAbs(0.5, 1e-12));
}

TEST_CASE("probability_of: |+> single-qubit measurements are 1/2 each") {
    auto mod = compile_circuit("H 0\nM 0");
    REQUIRE(mod.num_measurements == 1);

    std::vector<uint8_t> records{0, 1};
    auto log_probs = probability_of(mod, records, /*num_records=*/2);
    REQUIRE(log_probs.size() == 2);
    CHECK_THAT(log_probs[0], WithinAbs(std::log(0.5), 1e-12));
    CHECK_THAT(log_probs[1], WithinAbs(std::log(0.5), 1e-12));
}

TEST_CASE("probability_of: |0> deterministic measurement") {
    auto mod = compile_circuit("M 0");
    REQUIRE(mod.num_measurements == 1);

    std::vector<uint8_t> records{0, 1};
    auto log_probs = probability_of(mod, records, /*num_records=*/2);
    REQUIRE(log_probs.size() == 2);
    CHECK_THAT(log_probs[0], WithinAbs(0.0, 1e-12));  // log(1)
    REQUIRE(std::isinf(log_probs[1]));
    REQUIRE(log_probs[1] < 0);
}

// =============================================================================
// Feedback (OP_APPLY_PAULI) is supported.
// =============================================================================

TEST_CASE("probability_of: feedback circuit produces joint trajectory probability") {
    // Measure qubit 0 (random 0/1), apply X to qubit 1 conditional on the
    // outcome, then measure qubit 1. Joint probability of (m0, m1):
    //   (0, ?): qubit 1 stayed in |0>, so m1=0 deterministically.
    //   (1, ?): qubit 1 was flipped to |1>, so m1=1 deterministically.
    // So records (0,0) and (1,1) each have probability 1/2; (0,1) and (1,0)
    // are unreachable.
    auto mod = compile_circuit(
        "H 0\n"
        "M 0\n"
        "CX rec[-1] 1\n"
        "M 1\n");
    REQUIRE(mod.num_measurements == 2);

    std::vector<uint8_t> records{0, 0, 0, 1, 1, 0, 1, 1};
    auto log_probs = probability_of(mod, records, /*num_records=*/4);
    REQUIRE(log_probs.size() == 4);

    CHECK_THAT(log_probs[0], WithinAbs(std::log(0.5), 1e-12));
    REQUIRE(std::isinf(log_probs[1]));
    REQUIRE(log_probs[1] < 0);
    REQUIRE(std::isinf(log_probs[2]));
    REQUIRE(log_probs[2] < 0);
    CHECK_THAT(log_probs[3], WithinAbs(std::log(0.5), 1e-12));
}

// =============================================================================
// Pre-flight validation.
// =============================================================================

TEST_CASE("probability_of: rejects zero-measurement programs") {
    auto mod = compile_circuit("H 0\nT 0");
    REQUIRE(mod.num_measurements == 0);

    std::vector<uint8_t> records{};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/0), std::invalid_argument);
}

TEST_CASE("probability_of: rejects noise opcodes") {
    auto mod = compile_circuit("X_ERROR(0.1) 0\nM 0");
    std::vector<uint8_t> records{0};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/1), std::invalid_argument);
}

TEST_CASE("probability_of: rejects detector opcodes") {
    auto mod = compile_circuit("M 0\nDETECTOR rec[-1]");
    std::vector<uint8_t> records{0};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/1), std::invalid_argument);
}

TEST_CASE("probability_of: rejects observable opcodes") {
    auto mod = compile_circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]");
    std::vector<uint8_t> records{0};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/1), std::invalid_argument);
}

TEST_CASE("probability_of: rejects record buffer with inconsistent length") {
    auto mod = compile_circuit("M 0 1");
    REQUIRE(mod.num_measurements == 2);

    std::vector<uint8_t> records{0, 0, 1};  // 3 bytes is not a multiple of 2.
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/2), std::invalid_argument);
}

TEST_CASE("probability_of: rejects programs with hidden measurement slots") {
    // R q lowers to a hidden measurement + conditional Pauli. The forced
    // execution path doesn't currently marginalize over hidden outcomes,
    // so any program with hidden slots is rejected up front.
    auto mod = compile_circuit("M 0\nR 1\nM 1");
    REQUIRE(mod.num_measurements == 2);
    REQUIRE(mod.total_meas_slots > mod.num_measurements);

    std::vector<uint8_t> records{0, 0, 1, 1};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/2), std::invalid_argument);
}

TEST_CASE("probability_of: rejects non-bit record bytes") {
    auto mod = compile_circuit("H 0\nM 0");
    REQUIRE(mod.num_measurements == 1);

    std::vector<uint8_t> records{0, 1, 2};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/3), std::invalid_argument);
}

TEST_CASE("probability_of: rejects postselection opcodes") {
    auto mod = clifft::lower(clifft::trace(clifft::parse("M 0\nDETECTOR rec[-1]")),
                             /*postselection_mask=*/std::array<uint8_t, 1>{1});

    std::vector<uint8_t> records{0};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/1), std::invalid_argument);
}

TEST_CASE("probability_of: rejects hand-built bytecode containing forced opcodes") {
    // The validator must reject user-supplied forced opcodes so the
    // rewrite step in probability_of() is the only source of them.
    auto mod = compile_circuit("M 0");
    // Hand-patch the user-visible bytecode to its FORCED variant.
    mod.bytecode[0].opcode = Opcode::OP_MEAS_DORMANT_STATIC_FORCED;

    std::vector<uint8_t> records{0};
    REQUIRE_THROWS_AS(probability_of(mod, records, /*num_records=*/1), std::invalid_argument);
}

// =============================================================================
// EXP_VAL is allowed (probes are ignored).
// =============================================================================

TEST_CASE("probability_of: tolerates EXP_VAL probes") {
    auto mod = compile_circuit("H 0\nEXP_VAL X0\nM 0");
    std::vector<uint8_t> records{0, 1};
    auto log_probs = probability_of(mod, records, /*num_records=*/2);
    REQUIRE(log_probs.size() == 2);
    CHECK_THAT(log_probs[0], WithinAbs(std::log(0.5), 1e-12));
    CHECK_THAT(log_probs[1], WithinAbs(std::log(0.5), 1e-12));
}

// =============================================================================
// Edge cases.
// =============================================================================

TEST_CASE("probability_of: empty record batch returns empty vector") {
    auto mod = compile_circuit("H 0\nM 0");
    std::vector<uint8_t> records{};
    auto log_probs = probability_of(mod, records, /*num_records=*/0);
    REQUIRE(log_probs.empty());
}

TEST_CASE("probability_of: does not mutate the input CompiledModule") {
    auto mod = compile_circuit("H 0\nM 0");
    REQUIRE(mod.bytecode.size() > 0);

    // Byte-level snapshot of the full bytecode. The rewrite touches
    // opcode bytes, but any field shifting would be caught here too.
    std::vector<std::array<uint8_t, sizeof(Instruction)>> snapshot;
    snapshot.reserve(mod.bytecode.size());
    for (const auto& instr : mod.bytecode) {
        std::array<uint8_t, sizeof(Instruction)> bytes{};
        std::memcpy(bytes.data(), &instr, sizeof(Instruction));
        snapshot.push_back(bytes);
    }

    std::vector<uint8_t> records{0, 1};
    (void)probability_of(mod, records, /*num_records=*/2);

    REQUIRE(mod.bytecode.size() == snapshot.size());
    for (size_t i = 0; i < mod.bytecode.size(); ++i) {
        std::array<uint8_t, sizeof(Instruction)> after_bytes{};
        std::memcpy(after_bytes.data(), &mod.bytecode[i], sizeof(Instruction));
        REQUIRE(after_bytes == snapshot[i]);
    }
}

// =============================================================================
// Cross-check against probabilities() on terminal-M-all circuits.
// =============================================================================
//
// For a unitary circuit followed by M on every qubit, probability_of()
// computes the same value probabilities() does (the latter via the
// stabilizer-amplitude path on the unitary-stripped program). This is
// the load-bearing equivalence: any drift between the two paths shows
// up here.

TEST_CASE("probability_of: matches probabilities() on a Clifford circuit") {
    auto unitary = compile_circuit("H 0\nCX 0 1");
    auto measured = compile_circuit("H 0\nCX 0 1\nM 0 1");
    REQUIRE(measured.num_measurements == 2);

    // probabilities() takes word-packed bitmasks. For 2 qubits, each mask
    // fits in one 64-bit word; bit i = qubit i.
    std::vector<uint64_t> masks{0b00, 0b01, 0b10, 0b11};
    auto probs = probabilities(unitary, masks, /*num_basis_masks=*/4,
                               /*words_per_basis_mask=*/1);

    // probability_of() takes byte-packed records in measurement order.
    // M 0 1 records qubit 0 first, qubit 1 second, so record byte 0
    // corresponds to mask bit 0.
    std::vector<uint8_t> records{0, 0, 1, 0, 0, 1, 1, 1};
    auto log_probs = probability_of(measured, records, /*num_records=*/4);

    REQUIRE(probs.size() == 4);
    REQUIRE(log_probs.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        if (std::isfinite(log_probs[i])) {
            CHECK_THAT(std::exp(log_probs[i]), WithinAbs(probs[i], 1e-12));
        } else {
            CHECK_THAT(probs[i], WithinAbs(0.0, 1e-12));
        }
    }
}

TEST_CASE("probability_of: matches probabilities() on a Clifford+T circuit (active kernel)") {
    // H T H on |0> takes the qubit through a non-Clifford rotation that
    // forces the SVM to keep the qubit active at measurement time. The
    // resulting measurement opcode hits the active forced kernels via
    // the dispatcher, which the smaller closed-form tests above don't
    // exercise through the full compile pipeline.
    auto unitary = compile_circuit("H 0\nT 0\nH 0");
    auto measured = compile_circuit("H 0\nT 0\nH 0\nM 0");
    REQUIRE(measured.num_measurements == 1);
    REQUIRE(measured.peak_rank > 0);  // T gate forces a non-zero active rank.

    std::vector<uint64_t> masks{0, 1};
    auto probs = probabilities(unitary, masks, /*num_basis_masks=*/2,
                               /*words_per_basis_mask=*/1);

    std::vector<uint8_t> records{0, 1};
    auto log_probs = probability_of(measured, records, /*num_records=*/2);

    REQUIRE(probs.size() == 2);
    REQUIRE(log_probs.size() == 2);
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE(std::isfinite(log_probs[i]));
        CHECK_THAT(std::exp(log_probs[i]), WithinAbs(probs[i], 1e-12));
    }

    // Sanity: analytic outcome probabilities are (2 +/- sqrt(2)) / 4.
    const double sqrt2 = std::sqrt(2.0);
    CHECK_THAT(probs[0], WithinAbs((2.0 + sqrt2) / 4.0, 1e-12));
    CHECK_THAT(probs[1], WithinAbs((2.0 - sqrt2) / 4.0, 1e-12));
}

TEST_CASE("probability_of: multi-record probabilities sum to 1 for a small circuit") {
    // Unitarity check: enumerate all 2^n measurement records for a circuit
    // ending in M-all and confirm probabilities sum to 1.
    auto mod = compile_circuit("H 0\nCX 0 1\nH 1\nM 0 1");
    REQUIRE(mod.num_measurements == 2);

    std::vector<uint8_t> records{0, 0, 0, 1, 1, 0, 1, 1};
    auto log_probs = probability_of(mod, records, /*num_records=*/4);

    double total = 0.0;
    for (double lp : log_probs) {
        if (std::isfinite(lp)) {
            total += std::exp(lp);
        }
    }
    CHECK_THAT(total, WithinAbs(1.0, 1e-12));
}
