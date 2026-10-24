#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/simplify.h"

#include "zx_circuit_reference.h"
#include "zx_fresh_resets.h"
#include "zx_test_helpers.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft::zx;
using zx_test::amplitude_table;
using zx_test::apply_classical_readout;
using zx_test::Distribution;
using zx_test::Op;
using zx_test::random_ops;
using zx_test::Reference;
using zx_test::TestRng;
using zx_test::to_text;
using zx_test::with_fresh_qubit_per_reset;

namespace {

// ---------------------------------------------------------------------------
// The record distribution the reduced diagram describes.

// Fix the noise parameters to `assignment`, leaving a graph and constraints over
// the outcome parameters only.
Graph bind_noise(const Graph& g, const std::vector<DiagramParam>& params, uint64_t assignment,
                 std::vector<Constraint>& constraints) {
    auto split = [&](const Params& ps, bool& parity) {
        std::vector<uint32_t> kept;
        for (uint32_t p : ps.ids()) {
            if (params[p].kind == DiagramParam::Kind::Noise) {
                parity ^= ((assignment >> p) & 1) != 0;
            } else {
                kept.push_back(p);
            }
        }
        return Params(std::move(kept));
    };
    Graph out = g;
    for (Vertex v : out.vertices()) {
        bool parity = false;
        Params kept = split(out.phase(v).params, parity);
        out.set_phase(v, Phase(out.phase(v).value + (parity ? 1.0 : 0.0), std::move(kept)));
    }
    constraints.clear();
    for (const Constraint& c : g.constraints()) {
        bool parity = false;
        Params kept = split(c.params, parity);
        constraints.push_back(Constraint{.params = std::move(kept), .rhs = c.rhs != parity});
    }
    return out;
}

std::optional<Distribution> diagram_distribution(const SingleCopyDiagram& d, const Graph& reduced,
                                                 const clifft::HirModule& hir) {
    std::vector<uint32_t> outcome_params;
    for (uint32_t p = 0; p < d.params.size(); ++p) {
        if (d.params[p].kind != DiagramParam::Kind::Noise) {
            outcome_params.push_back(p);
        }
    }
    Distribution dist;
    std::vector<size_t> choice(d.noise.size(), 0);
    while (true) {
        double weight = 1.0;
        uint64_t noise_bits = 0;
        for (size_t g = 0; g < d.noise.size(); ++g) {
            const NoiseGroup::Outcome& o = d.noise[g].outcomes[choice[g]];
            weight *= o.probability;
            for (size_t k = 0; k < d.noise[g].bits.size(); ++k) {
                if ((o.pattern >> k) & 1) {
                    noise_bits |= uint64_t{1} << d.noise[g].bits[k];
                }
            }
        }
        if (weight > 0.0) {
            std::vector<Constraint> constraints;
            const Graph bound = bind_noise(reduced, d.params, noise_bits, constraints);
            const auto table = amplitude_table(bound, outcome_params, constraints, 20);
            if (!table) {
                return std::nullopt;
            }
            Distribution given;
            double total = 0.0;
            for (uint64_t a = 0; a < table->size(); ++a) {
                uint64_t records = 0;
                for (size_t k = 0; k < outcome_params.size(); ++k) {
                    const DiagramParam& p = d.params[outcome_params[k]];
                    if (p.kind == DiagramParam::Kind::Record) {
                        records |= ((a >> k) & 1) << p.index;
                    }
                }
                const double p2 = (*table)[a] * (*table)[a];
                given[records] += p2;
                total += p2;
            }
            if (total <= 0.0) {
                return Distribution{};  // a noise outcome the diagram calls impossible
            }
            for (const auto& [records, p2] : given) {
                dist[records] += weight * p2 / total;
            }
        }
        size_t g = 0;
        while (g < choice.size() && ++choice[g] == d.noise[g].outcomes.size()) {
            choice[g++] = 0;
        }
        if (g == choice.size()) {
            break;
        }
    }
    return apply_classical_readout(std::move(dist), hir, d.classical_readout);
}

// The evaluator ignores dropped scalar factors other than constraints, which
// is exact only if each has a parameter-independent modulus.
bool scalar_terms_are_constraints_or_constants(const Graph& g) {
    for (const ScalarTerm& t : g.scalar_terms()) {
        const bool pauli_a = t.alpha == 0.0 || t.alpha == 1.0;
        const bool pauli_b = t.beta == 0.0 || t.beta == 1.0;
        const bool ok = t.kind == ScalarTerm::Kind::Node
                            ? pauli_a || t.a.empty()
                            : (pauli_a && pauli_b) || (t.a.empty() && t.b.empty());
        if (!ok) {
            return false;
        }
    }
    return true;
}

enum class Check { Matched, Rejected };

// Trace the circuit (optionally with fresh qubits per reset), build, reduce,
// and compare the record distribution with the reference.
Check check_circuit(const std::vector<Op>& ops, int num_qubits, bool fresh) {
    const std::string text = to_text(ops);
    INFO(text);
    const clifft::Circuit parsed = clifft::parse(text);
    const clifft::HirModule hir =
        clifft::trace(fresh ? with_fresh_qubit_per_reset(parsed) : parsed);
    std::string reason;
    const std::optional<SingleCopyDiagram> d = build_single_copy_diagram(hir, &reason);
    if (!d) {
        // The one rejection random circuits can trigger.
        REQUIRE(reason == "feedback reads a record with asymmetric readout noise");
        return Check::Rejected;
    }
    Graph reduced = d->graph;
    full_reduce(reduced);
    REQUIRE(scalar_terms_are_constraints_or_constants(reduced));
    const auto got = diagram_distribution(*d, reduced, hir);
    REQUIRE(got);
    const Distribution want = Reference(num_qubits).run(ops);
    for (uint64_t r = 0; r < (uint64_t{1} << hir.num_measurements); ++r) {
        const double a = want.contains(r) ? want.at(r) : 0.0;
        const double b = got->contains(r) ? got->at(r) : 0.0;
        INFO("records " << r);
        REQUIRE(std::abs(a - b) < 1e-9);
    }
    return Check::Matched;
}

// With qubit reuse kept, resets trace as hidden measurements plus corrections,
// which the builder rewrites onto fresh qubits; with fresh qubits per reset in
// the circuit they trace as cut wires. Both must describe the circuit's record
// distribution exactly.
void check_random_circuits(int trials) {
    for (const bool fresh : {false, true}) {
        for (const bool noise_heavy : {false, true}) {
            INFO((fresh ? "fresh qubit per reset" : "qubit reuse kept")
                 << (noise_heavy ? ", noise between basis changes" : ", mixed operations"));
            TestRng rng(4);
            int matched = 0, rejected = 0;
            for (int trial = 0; trial < trials; ++trial) {
                const int num_qubits = 2 + trial % 2;
                const auto ops = random_ops(rng, num_qubits, noise_heavy ? 10 : 14, noise_heavy);
                (check_circuit(ops, num_qubits, fresh) == Check::Matched ? matched : rejected)++;
            }
            CHECK(matched + rejected == trials);
            CHECK(rejected * 10 < trials);
        }
    }
}

}  // namespace

TEST_CASE("Zx diagrams of traced circuits reproduce record distributions", "[zx]") {
    check_random_circuits(25);
}

TEST_CASE("Zx diagrams reproduce record distributions of many random circuits", "[zx][expensive]") {
    check_random_circuits(150);
}

TEST_CASE("Zx diagrams match exact distributions of hand written circuits", "[zx]") {
    // Each case targets one failure: Pauli noise a basis change exposes,
    // dependent channels, Y bases, inverted and padded records, rotations with
    // Y factors, readout noise before and after feedback, and repeated resets.
    const std::vector<std::pair<int, std::vector<Op>>> cases = {
        {1,
         {{.name = "H", .targets = {0}},
          {.name = "DEPOLARIZE1", .targets = {0}, .args = {0.3}},
          {.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "DEPOLARIZE2", .targets = {0, 1}, .args = {0.4}},
          {.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "MX", .targets = {1}}}},
        {2,
         {{.name = "E", .targets = {0, 1}, .paulis = "XZ", .args = {0.3}},
          {.name = "ELSE_CORRELATED_ERROR", .targets = {0}, .paulis = "X", .args = {0.5}},
          {.name = "ELSE_CORRELATED_ERROR", .targets = {1}, .paulis = "Y", .args = {0.2}},
          {.name = "M", .targets = {0}},
          {.name = "MX", .targets = {1}}}},
        {1,
         {{.name = "RY", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "MY", .targets = {0}},
          {.name = "MRY", .targets = {0}},
          {.name = "MY", .targets = {0}}}},
        {1,
         {{.name = "H", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "M", .targets = {0}, .inverted = true},
          {.name = "MPAD", .targets = {1}},
          {.name = "MRX", .targets = {0}}}},
        {2,
         {{.name = "R_PAULI", .targets = {0, 1}, .paulis = "XY", .args = {0.3}},
          {.name = "MPP", .targets = {0, 1}, .paulis = "YZ"},
          {.name = "MY", .targets = {0}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "MR", .targets = {0}, .args = {0.25}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "M", .targets = {1}},
          {.name = "M", .targets = {0}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "READOUT_NOISE", .targets = {-1}, .args = {0.1, 0.4}},
          {.name = "M", .targets = {1}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "READOUT_NOISE", .targets = {-1}, .args = {0.3}},
          {.name = "FEEDBACK_CZ", .targets = {-1, 1}},
          {.name = "H", .targets = {1}},
          {.name = "M", .targets = {1}}}},
        // Feedback on both sides of one flip: the first read sees the record
        // before the flip, the second after it.
        {3,
         {{.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "READOUT_NOISE", .targets = {-1}, .args = {0.3}},
          {.name = "FEEDBACK_CX", .targets = {-1, 2}},
          {.name = "M", .targets = {1}},
          {.name = "M", .targets = {2}}}},
        // Feedback right after a measurement that commutes with the measured
        // Pauli is not a reset: on the measured qubit, and on another one.
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "MX", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 0}},
          {.name = "H", .targets = {0}},
          {.name = "CX", .targets = {0, 1}},
          {.name = "MY", .targets = {0}},
          {.name = "M", .targets = {1}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "H", .targets = {0}},
          {.name = "CX", .targets = {0, 1}},
          {.name = "MX", .targets = {0}},
          {.name = "M", .targets = {1}}}},
        // A Z reset after an X or Y reset of an untouched qubit.
        {1,
         {{.name = "RX", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "M", .targets = {0}}}},
        {1,
         {{.name = "RY", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "M", .targets = {0}}}},
        {1,
         {{.name = "X", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "H", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "M", .targets = {0}}}},
    };
    for (const bool fresh : {false, true}) {
        for (const auto& [num_qubits, ops] : cases) {
            INFO((fresh ? "fresh qubit per reset" : "qubit reuse kept"));
            CHECK(check_circuit(ops, num_qubits, fresh) == Check::Matched);
        }
    }
}

TEST_CASE("Zx fresh qubits per reset relabel later targets", "[zx]") {
    const clifft::Circuit c = clifft::parse("H 0\nMR 0\nRX 1\nR 2\nR 2\nCX 0 2\nM 0 1 2\n");
    const clifft::Circuit fresh = with_fresh_qubit_per_reset(c);
    CHECK(fresh.num_measurements == c.num_measurements);
    // MR 0 moves qubit 0 to qubit 3. Qubits 1 and 2 were still |0>, so their
    // resets keep them, and the repeated R 2 allocates nothing.
    CHECK(fresh.num_qubits == 4);
    const auto cx =
        std::find_if(fresh.nodes.begin(), fresh.nodes.end(),
                     [](const clifft::AstNode& n) { return n.gate == clifft::GateType::CX; });
    REQUIRE(cx != fresh.nodes.end());
    REQUIRE(cx->targets.size() == 2);
    CHECK(cx->targets[0].value() == 3);
    CHECK(cx->targets[1].value() == 2);
}

TEST_CASE("Zx fresh qubits per reset reject loss annotations", "[zx]") {
    const clifft::Circuit c = clifft::parse("LOSS(0.1) 0\nR 0\nM 0\n");
    CHECK_THROWS_AS(with_fresh_qubit_per_reset(c), std::invalid_argument);
}

TEST_CASE("Zx diagrams reject noise rescheduled across operations", "[zx]") {
    clifft::HirModule hir = clifft::trace(clifft::parse("H 0\nZ_ERROR(0.3) 0\nT 0\nM 0\n"));
    // Mark the T as having come before the noise, as a scheduler would.
    hir.materialize_logical_noise_prefix();
    bool found = false;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() == clifft::OpType::T_GATE) {
            hir.logical_noise_prefix[i] = 0;
            found = true;
        }
    }
    REQUIRE(found);
    std::string reason;
    CHECK_FALSE(build_single_copy_diagram(hir, &reason));
    CHECK(reason == "noise has been rescheduled past other operations");
}
