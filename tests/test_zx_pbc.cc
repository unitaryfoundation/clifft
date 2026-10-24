#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/pbc.h"
#include "clifft/zx/schedule.h"
#include "clifft/zx/simplify.h"

#include "zx_circuit_reference.h"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace clifft::zx;
using zx_test::apply_classical_readout;
using zx_test::Distribution;
using zx_test::Op;
using zx_test::random_ops;
using zx_test::Reference;
using zx_test::TestRng;
using zx_test::to_text;

namespace {

using C = std::complex<double>;
using zx_test::apply_pauli;
using zx_test::PauliMasks;
using zx_test::State;

// Masks of a group's Pauli: bit l of x and z for line l.
PauliMasks masks_of(const PbcOp& op) {
    PauliMasks m;
    for (size_t l = 0; l < op.x.size(); ++l) {
        m.x |= uint64_t(op.x.test(l)) << l;
        m.z |= uint64_t(op.z.test(l)) << l;
    }
    return m;
}

// The distribution of the group's records, given its noise bits, running the
// ops in `order` on |0...0> and branching on every measurement.
void simulate(const PbcGroup& g, const std::vector<uint32_t>& order, size_t at, State psi,
              double prob, uint64_t records, const std::vector<bool>& noise, Distribution& out) {
    if (prob < 1e-15) {
        return;
    }
    for (; at < order.size(); ++at) {
        const PbcOp& op = g.ops[order[at]];
        bool s = op.sign.test(0);
        for (size_t k = 0; k < g.noise.size(); ++k) {
            s ^= op.sign.test(1 + k) && noise[g.noise[k]];
        }
        const State p_psi = apply_pauli(psi, masks_of(op));
        const double sigma = s ? -1.0 : 1.0;
        if (op.kind == PbcOp::Kind::Rotation) {
            for (size_t i = 0; i < psi.size(); ++i) {
                psi[i] = std::cos(op.theta) * psi[i] - C(0, sigma * std::sin(op.theta)) * p_psi[i];
            }
            continue;
        }
        for (int bit = 0; bit < 2; ++bit) {
            const double e = (bit ? -1.0 : 1.0) * sigma;
            State post(psi.size());
            double p = 0.0;
            for (size_t i = 0; i < psi.size(); ++i) {
                post[i] = (psi[i] + e * p_psi[i]) / 2.0;
                p += std::norm(post[i]);
            }
            if (p < 1e-15) {
                continue;
            }
            for (auto& a : post) {
                a /= std::sqrt(p);
            }
            uint64_t with = records;
            if (op.record >= 0) {
                with |= uint64_t(bit) << op.record;
            }
            simulate(g, order, at + 1, std::move(post), prob * p, with, noise, out);
        }
        return;
    }
    out[records] += prob;
}

// The record distribution of the whole program, every group run in `order_of`.
Distribution pbc_distribution(const SingleCopyDiagram& d, const PbcProgram& program,
                              const clifft::HirModule& hir,
                              const std::vector<std::vector<uint32_t>>& order_of) {
    Distribution dist;
    std::vector<size_t> choice(d.noise.size(), 0);
    while (true) {
        double weight = 1.0;
        std::vector<bool> noise(d.params.size(), false);
        for (size_t g = 0; g < d.noise.size(); ++g) {
            const NoiseGroup::Outcome& o = d.noise[g].outcomes[choice[g]];
            weight *= o.probability;
            for (size_t k = 0; k < d.noise[g].bits.size(); ++k) {
                noise[d.noise[g].bits[k]] = (o.pattern >> k) & 1;
            }
        }
        if (weight > 0.0) {
            Distribution joint = {{0, 1.0}};
            for (size_t gi = 0; gi < program.groups.size(); ++gi) {
                const PbcGroup& g = program.groups[gi];
                State psi(size_t{1} << g.num_lines, 0.0);
                psi[0] = 1.0;
                Distribution part;
                simulate(g, order_of[gi], 0, std::move(psi), 1.0, 0, noise, part);
                Distribution next;
                for (const auto& [a, pa] : joint) {
                    for (const auto& [b, pb] : part) {
                        next[a | b] += pa * pb;  // groups write disjoint records
                    }
                }
                joint = std::move(next);
            }
            for (const auto& [r, p] : joint) {
                dist[r] += weight * p;
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

enum class Check { Matched, Rejected };

Check check_circuit(const std::vector<Op>& ops, int num_qubits) {
    const std::string text = to_text(ops);
    INFO(text);
    const clifft::HirModule hir = clifft::trace(clifft::parse(text));
    std::string reason;
    const auto d = build_single_copy_diagram(hir, &reason);
    if (!d) {
        REQUIRE(reason == "feedback reads a record with asymmetric readout noise");
        return Check::Rejected;
    }
    Graph reduced = d->graph;
    full_reduce(reduced);
    const auto program = read_pbc(*d, reduced, &reason);
    INFO(reason);
    REQUIRE(program);
    std::vector<std::vector<uint32_t>> given, scheduled;
    for (const PbcGroup& g : program->groups) {
        REQUIRE(g.num_lines <= 20);
        std::vector<uint32_t> order(g.ops.size());
        std::iota(order.begin(), order.end(), 0u);
        given.push_back(order);
        const Schedule s = greedy_schedule(g);
        REQUIRE(s.order.size() == g.ops.size());
        CHECK(peak_width(g, s.order) == s.width);
        scheduled.push_back(s.order);
    }
    const Distribution want = Reference(num_qubits).run(ops);
    // The reader's order and the greedy schedule must both reproduce it.
    for (const auto* orders : {&given, &scheduled}) {
        const Distribution got = pbc_distribution(*d, *program, hir, *orders);
        for (uint64_t r = 0; r < (uint64_t{1} << hir.num_measurements); ++r) {
            const double a = want.contains(r) ? want.at(r) : 0.0;
            const double b = got.contains(r) ? got.at(r) : 0.0;
            INFO("records " << r << (orders == &given ? " in the reader's order" : " scheduled"));
            REQUIRE(std::abs(a - b) < 1e-9);
        }
    }
    return Check::Matched;
}

void check_random_circuits(int trials) {
    for (const bool noise_heavy : {false, true}) {
        INFO((noise_heavy ? "noise between basis changes" : "mixed operations"));
        TestRng rng(9);
        int matched = 0, rejected = 0;
        for (int trial = 0; trial < trials; ++trial) {
            const int num_qubits = 2 + trial % 2;
            const auto ops = random_ops(rng, num_qubits, noise_heavy ? 10 : 14, noise_heavy);
            (check_circuit(ops, num_qubits) == Check::Matched ? matched : rejected)++;
        }
        CHECK(matched + rejected == trials);
        CHECK(rejected * 10 < trials);
    }
}

}  // namespace

TEST_CASE("Zx PBCs reproduce record distributions of traced circuits", "[zx]") {
    check_random_circuits(25);
}

TEST_CASE("Zx PBCs reproduce record distributions of many random circuits", "[zx][expensive]") {
    check_random_circuits(150);
}

TEST_CASE("Zx PBC of cultivation d5 schedules to width 6", "[zx]") {
    // The circuit from SOFT that the ZX lowering was designed around: Tsim
    // reaches width 6 on it, and Clifft's ordinary pipeline reaches 10.
    std::ifstream in(std::string(CLIFFT_FIXTURES_DIR) + "/cultivation_d5.stim");
    REQUIRE(in.good());
    std::stringstream text;
    text << in.rdbuf();
    const clifft::HirModule hir = clifft::trace(clifft::parse(text.str()));
    std::string reason;
    const auto d = build_single_copy_diagram(hir, &reason);
    REQUIRE(d);
    Graph reduced = d->graph;
    full_reduce(reduced);
    const auto program = read_pbc(*d, reduced, &reason);
    INFO(reason);
    REQUIRE(program);
    uint32_t width = 0;
    for (const PbcGroup& g : program->groups) {
        width = std::max(width, greedy_schedule(g).width);
    }
    CHECK(width == 6);
}
