// Boundary-driven width-sweep fuzz harness.
//
// Sweeps n ∈ {32, 64, 100, 128, 129, 150, 1000} and touched qubits
// ∈ {0, 63, 64, 127, 128, n-1} over deterministic patterns that exercise
// frontend → optimizer → backend → SVM at word boundaries. Catches the
// class of word-boundary bug where a scratch buffer or scalar mask field
// clips bits at qubit 64+ -- the runtime-width arena and Pauli frame must
// round-trip every set bit.
//
// Each pattern uses an outcome that is deterministic conditional on the
// circuit, so we can assert exact equality across all shots without a
// statistical oracle. For the random-measurement coverage we just check
// that both branches appear over many shots, which still exercises the
// SVM hot paths.
//
// Tests are tagged [fuzz]. They are included in default ctest.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/svm/svm.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using namespace clifft;

namespace {

// Boundary widths the harness sweeps. 64 lies on the 1-word boundary,
// 128 on the 2-word boundary, 129 and 150 cover the immediate generic-
// fallback path, and 1000 stresses the multi-word loops; 32 and 100
// are non-boundary controls.
constexpr uint32_t kSweep[] = {32, 64, 100, 128, 129, 150, 1000};

// Touched-qubit selection. Always include 0; add the last bit of words
// 0 (63), 1 (127), and any specific within-range word boundary that
// fixed-width scratch buffers historically clipped (64, 128). Always
// include n-1 to land on the last live qubit. Suppress duplicates.
std::vector<uint32_t> touched_qubits(uint32_t n) {
    std::vector<uint32_t> out;
    auto push_unique = [&](uint32_t q) {
        if (q < n && std::find(out.begin(), out.end(), q) == out.end())
            out.push_back(q);
    };
    push_unique(0);
    push_unique(63);
    push_unique(64);
    push_unique(127);
    push_unique(128);
    push_unique(n - 1);
    return out;
}

// Cross-word pairs that exercise inter-word interactions in the live
// region. (63, 64) straddles the first word boundary, (127, 128)
// straddles the second, and (0, n - 1) spans the live region.
std::vector<std::pair<uint32_t, uint32_t>> cross_word_pairs(uint32_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    if (n > 64)
        out.emplace_back(63, 64);
    if (n > 128)
        out.emplace_back(127, 128);
    if (n > 1)
        out.emplace_back(0, n - 1);
    return out;
}

// Compile a circuit through the front-end, peephole pass, and backend.
// Mirrors the production pipeline closely enough to catch end-to-end
// width bugs.
CompiledModule compile_text(uint32_t n, const std::string& body) {
    // Force the active width by appending an identity touch on the
    // highest qubit. Identity gates update num_qubits without emitting
    // an HIR op (see is_identity_noop).
    std::ostringstream s;
    s << body;
    if (!body.empty() && body.back() != '\n')
        s << '\n';
    s << "I " << (n - 1) << "\n";

    auto hir = trace(parse(s.str()));
    HirPassManager pm;
    pm.add_pass(std::make_unique<PeepholeFusionPass>());
    pm.run(hir);
    return lower(hir);
}

// Read measurement m_idx from each shot.
std::vector<uint8_t> meas_column(const SampleResult& r, uint32_t num_meas, size_t m_idx) {
    std::vector<uint8_t> out;
    out.reserve(r.measurements.size() / num_meas);
    for (size_t shot = 0; shot < r.measurements.size() / num_meas; ++shot) {
        out.push_back(r.measurements[shot * num_meas + m_idx]);
    }
    return out;
}

void require_all_equal(const std::vector<uint8_t>& bits, uint8_t expected) {
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] != expected) {
            CAPTURE(i, bits[i], expected);
            FAIL("bit mismatch in deterministic-outcome shot");
        }
    }
}

constexpr uint64_t kFuzzSeed = 0xC11FF7;
constexpr uint32_t kShots = 32;

}  // namespace

// ---------------------------------------------------------------------------
// Single-qubit measurement: |0> deterministic.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: deterministic single-qubit Z measurement", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (uint32_t q : touched_qubits(n)) {
            CAPTURE(n, q);

            // M q on |0>: always 0.
            {
                std::ostringstream body;
                body << "M " << q;
                auto mod = compile_text(n, body.str());
                auto result = sample(mod, kShots, kFuzzSeed);
                require_all_equal(meas_column(result, 1, 0), 0);
            }

            // X q; M q: always 1.
            {
                std::ostringstream body;
                body << "X " << q << "\nM " << q;
                auto mod = compile_text(n, body.str());
                auto result = sample(mod, kShots, kFuzzSeed);
                require_all_equal(meas_column(result, 1, 0), 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reset overrides X. Tests R / MR pathway.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: R after X gives 0", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (uint32_t q : touched_qubits(n)) {
            CAPTURE(n, q);
            std::ostringstream body;
            body << "X " << q << "\nR " << q << "\nM " << q;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kShots, kFuzzSeed);
            require_all_equal(meas_column(result, 1, 0), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic noise: X_ERROR(1.0) is a guaranteed flip; X_ERROR(0.0) never.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: deterministic single-qubit noise", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (uint32_t q : touched_qubits(n)) {
            CAPTURE(n, q);

            // p=1.0 always flips.
            {
                std::ostringstream body;
                body << "X_ERROR(1.0) " << q << "\nM " << q;
                auto mod = compile_text(n, body.str());
                auto result = sample(mod, kShots, kFuzzSeed);
                require_all_equal(meas_column(result, 1, 0), 1);
            }

            // p=0.0 never flips.
            {
                std::ostringstream body;
                body << "X_ERROR(0.0) " << q << "\nM " << q;
                auto mod = compile_text(n, body.str());
                auto result = sample(mod, kShots, kFuzzSeed);
                require_all_equal(meas_column(result, 1, 0), 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Classical feedback. After H q; M q; CX rec[-1] q; M q the second
// measurement is deterministically 0: if the first measurement collapsed
// to 0, no CX fires; if it collapsed to 1, the CX flips q back to 0.
// Exercises the CONDITIONAL_PAULI rewinding path at boundary qubits.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: classical feedback uncomputes random measurement", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (uint32_t q : touched_qubits(n)) {
            CAPTURE(n, q);
            std::ostringstream body;
            body << "H " << q << "\nM " << q << "\nCX rec[-1] " << q << "\nM " << q;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kShots, kFuzzSeed);
            // Second measurement is deterministically 0.
            require_all_equal(meas_column(result, 2, 1), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-Pauli measurement (MPP) spanning a word boundary. State |++> has
// X_a*X_b eigenvalue +1, so MPP yields 0.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: MPP X*X on |++> across word boundary", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (auto [a, b] : cross_word_pairs(n)) {
            CAPTURE(n, a, b);
            std::ostringstream body;
            body << "H " << a << "\nH " << b << "\nMPP X" << a << "*X" << b;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kShots, kFuzzSeed);
            require_all_equal(meas_column(result, 1, 0), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Bell state: H a; CX a b; Z_a*Z_b eigenvalue is +1, MPP yields 0. The
// CX may straddle word boundaries; the rewinding through the tableau
// must preserve all set bits.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: MPP Z*Z on Bell state across word boundary", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (auto [a, b] : cross_word_pairs(n)) {
            CAPTURE(n, a, b);
            std::ostringstream body;
            body << "H " << a << "\nCX " << a << " " << b << "\nMPP Z" << a << "*Z" << b;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kShots, kFuzzSeed);
            require_all_equal(meas_column(result, 1, 0), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// EXP_VAL probes. Each is deterministic given the prepared state.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: EXP_VAL Z on |0> = +1", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (uint32_t q : touched_qubits(n)) {
            CAPTURE(n, q);
            std::ostringstream body;
            body << "EXP_VAL Z" << q;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kShots, kFuzzSeed);
            REQUIRE(result.exp_vals.size() == kShots);
            for (size_t i = 0; i < result.exp_vals.size(); ++i) {
                CAPTURE(i);
                REQUIRE(result.exp_vals[i] == 1.0);
            }
        }
    }
}

TEST_CASE("Fuzz: EXP_VAL Z after X = -1", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (uint32_t q : touched_qubits(n)) {
            CAPTURE(n, q);
            std::ostringstream body;
            body << "X " << q << "\nEXP_VAL Z" << q;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kShots, kFuzzSeed);
            REQUIRE(result.exp_vals.size() == kShots);
            for (size_t i = 0; i < result.exp_vals.size(); ++i) {
                CAPTURE(i);
                REQUIRE(result.exp_vals[i] == -1.0);
            }
        }
    }
}

TEST_CASE("Fuzz: EXP_VAL Z*Z on |11> across word boundary", "[fuzz]") {
    for (uint32_t n : kSweep) {
        for (auto [a, b] : cross_word_pairs(n)) {
            CAPTURE(n, a, b);
            std::ostringstream body;
            body << "X " << a << "\nX " << b << "\nEXP_VAL Z" << a << "*Z" << b;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kShots, kFuzzSeed);
            REQUIRE(result.exp_vals.size() == kShots);
            for (size_t i = 0; i < result.exp_vals.size(); ++i) {
                CAPTURE(i);
                REQUIRE(result.exp_vals[i] == 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Random-measurement coverage. H q; M q is 50/50; we just confirm both
// branches appear over many shots, which exercises the active-measure
// SVM path at every (n, q) combination.
// ---------------------------------------------------------------------------
TEST_CASE("Fuzz: H then M produces both 0 and 1 outcomes", "[fuzz]") {
    constexpr uint32_t kManyShots = 200;
    for (uint32_t n : kSweep) {
        for (uint32_t q : touched_qubits(n)) {
            CAPTURE(n, q);
            std::ostringstream body;
            body << "H " << q << "\nM " << q;
            auto mod = compile_text(n, body.str());
            auto result = sample(mod, kManyShots, kFuzzSeed);
            auto bits = meas_column(result, 1, 0);
            uint32_t zeros = 0, ones = 0;
            for (auto b : bits) {
                if (b)
                    ++ones;
                else
                    ++zeros;
            }
            REQUIRE(zeros > 0);
            REQUIRE(ones > 0);
        }
    }
}
