#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/optimizer/wide_phase_region.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <string>
#include <vector>

using namespace clifft;

namespace {

using phase_detail::wide::Mask;
constexpr uint32_t width = 65;

Mask bit(uint32_t q) {
    Mask mask;
    mask.bit_set(q, true);
    return mask;
}

struct PhaseTerm {
    Mask parity;
    uint8_t coefficient;
    bool negative;
};

unsigned evaluate(const std::vector<PhaseTerm>& terms, Mask assignment) {
    unsigned value = 0;
    for (const auto& term : terms) {
        const bool eigenbit = ((term.parity & assignment).popcount() & 1) != 0;
        value += term.coefficient * (eigenbit != term.negative);
    }
    return value & 7;
}

// For graph generators K_q = X_q * product Z_neighbors, K(m) has X mask m,
// Z mask A*m, and XZ phase (-1) to the number of selected internal edges.
// This scalar formula avoids the optimizer's Pauli products and basis algebra.
PauliString graph_product(const std::vector<Mask>& neighbors, Mask selected) {
    PauliString result(width);
    unsigned edges = 0;
    for (uint32_t q = 0; q < width; ++q) {
        if (!selected.bit_get(q)) {
            continue;
        }
        result.mut_x().bit_set(q, true);
        for (uint32_t r = 0; r < width; ++r) {
            if (neighbors[q].bit_get(r)) {
                result.mut_z().bit_set(r, !result.z().bit_get(r));
                edges += r < q && selected.bit_get(r);
            }
        }
    }
    result.set_phase(static_cast<uint8_t>((2 * edges) & 3));
    return result;
}

std::vector<PhaseTerm> decode(const phase_detail::wide::Rewrite& rewrite,
                              const std::vector<Mask>& neighbors) {
    std::vector<PhaseTerm> terms;
    for (const auto& [axis, coefficient] : rewrite.phases) {
        Mask parity;
        for (uint32_t q = 0; q < width; ++q) {
            parity.bit_set(q, axis.x().bit_get(q));
        }
        const auto expected = graph_product(neighbors, parity);
        for (uint32_t q = 0; q < width; ++q) {
            REQUIRE(axis.z().bit_get(q) == expected.z().bit_get(q));
        }
        const unsigned phase = (unsigned(axis.phase()) + 4 - expected.phase()) & 3;
        REQUIRE((phase & 1) == 0);
        terms.push_back({parity, coefficient, phase == 2});
    }
    return terms;
}

}  // namespace

TEST_CASE("Output aligned wide phase rewrites preserve signed eigenphases", "[optimizer]") {
    std::mt19937_64 rng(5522026);
    std::vector<Mask> neighbors(width);
    std::string clifford;
    for (unsigned edge = 0; edge < 100; ++edge) {
        const uint32_t a = rng() % width;
        const uint32_t b = rng() % width;
        if (a != b) {
            neighbors[a] ^= bit(b);
            neighbors[b] ^= bit(a);
            clifford += "CZ " + std::to_string(a) + " " + std::to_string(b) + "\n";
        }
    }
    for (uint32_t q = 0; q < width; ++q) {
        clifford += "H " + std::to_string(q) + "\n";
    }
    HirModule hir(width, 8 * width + 15 * (width - 9) + 42);
    hir.final_tableau = trace(parse(clifford)).final_tableau;
    std::vector<PhaseTerm> original;
    const auto emit = [&](Mask parity, bool dagger) {
        auto axis = graph_product(neighbors, parity);
        const bool negative = original.size() % 3 == 0;
        if (negative) {
            axis.negate();
        }
        // Paired sign and dagger flips preserve the intended nonconstant phase.
        dagger ^= negative;
        original.push_back({parity, uint8_t(dagger ? 7 : 1), negative});
        hir.append_tgate(
            dagger, [&](MutablePauliMaskView mask) { optimizer_detail::write_axis(mask, axis); });
    };

    // Identity rotations establish a full collection basis whose coordinates
    // obscure the small non-Clifford core.
    std::vector<Mask> basis;
    for (uint32_t q = 0; q < width; ++q) {
        basis.push_back(bit(q));
    }
    for (unsigned step = 0; step < 150; ++step) {
        const auto a = rng() % width;
        const auto b = rng() % width;
        if (a != b) {
            basis[a] ^= basis[b];
        }
    }
    for (auto parity : basis) {
        for (unsigned repeat = 0; repeat < 8; ++repeat) {
            emit(parity, false);
        }
    }

    const std::array<uint32_t, 9> core{0, 1, 2, 31, 32, 33, 62, 63, 64};
    for (uint32_t q = 0; q < width; ++q) {
        if (std::ranges::find(core, q) != core.end()) {
            continue;
        }
        std::array<Mask, 4> quartet{};
        for (unsigned j = 0; j < 3; ++j) {
            const auto selected = 1 + rng() % 511;
            for (unsigned k = 0; k < core.size(); ++k) {
                if (selected & (1U << k)) {
                    quartet[j] |= bit(core[k]);
                }
            }
        }
        quartet[3] = bit(q);
        // All fifteen parities on four bits sum to zero modulo eight. Their
        // projected odd columns obstruct synthesis in the collection basis;
        // the output basis makes the spectator direction disappear.
        for (unsigned subset = 1; subset < 16; ++subset) {
            Mask parity;
            for (unsigned j = 0; j < quartet.size(); ++j) {
                if (subset & (1U << j)) {
                    parity ^= quartet[j];
                }
            }
            if (!parity.is_zero()) {
                emit(parity, false);
            }
        }
    }
    for (unsigned a = 0; a < 3; ++a) {
        for (unsigned b = 0; b < 3; ++b) {
            if (a == b) {
                continue;
            }
            const auto x = bit(core[a]);
            const auto y = bit(core[3 + b]);
            const auto z = bit(core[6 + 3 - a - b]);
            // x+y+z-(x XOR y)-(x XOR z)-(y XOR z)+(x XOR y XOR z) = 4*x*y*z.
            for (auto parity : {x, y, z, x ^ y ^ z}) {
                emit(parity, false);
            }
            for (auto parity : {x ^ y, x ^ z, y ^ z}) {
                emit(parity, true);
            }
        }
    }

    const KnownStabilizers fixed(width);
    const optimizer_detail::CliffordAbsorption frame;
    auto without_output = hir;
    without_output.final_tableau.reset();
    phase_detail::wide::Budget ordinary_budget, aligned_budget;
    const auto ordinary =
        phase_detail::wide::try_region(without_output, 0, fixed, frame, 128, ordinary_budget);
    const auto aligned = phase_detail::wide::try_region(hir, 0, fixed, frame, 128, aligned_budget);
    REQUIRE(ordinary);
    REQUIRE(aligned);
    REQUIRE_FALSE(ordinary_budget.exhausted);
    REQUIRE_FALSE(aligned_budget.exhausted);
    REQUIRE(aligned->end == hir.ops.size());
    REQUIRE(aligned->t_count < ordinary->t_count);

    const auto rewritten = decode(*aligned, neighbors);
    const unsigned global_phase = (evaluate(original, {}) + 8 - evaluate(rewritten, {})) & 7;
    for (unsigned assignment = 0; assignment < 512; ++assignment) {
        for (unsigned sample = 0; sample < 2; ++sample) {
            Mask x;
            x.w[0] = rng();
            x.w[1] = rng() & 1;
            for (unsigned q = 0; q < core.size(); ++q) {
                x.bit_set(core[q], (assignment >> q) & 1);
            }
            CAPTURE(assignment, sample);
            REQUIRE((evaluate(original, x) + 8 - evaluate(rewritten, x)) % 8 == global_phase);
        }
    }
}
