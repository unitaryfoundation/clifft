#pragma once

// Brute-force evaluation of small closed single-copy ZX diagrams, shared by the
// ZX tests as an oracle that does not depend on any rewrite rule.

#include "clifft/zx/graph.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace zx_test {

using namespace clifft::zx;

// Draws from mt19937_64, whose output the standard fixes, without the
// library-defined std:: distributions, so generated cases match everywhere.
struct TestRng {
    std::mt19937_64 engine;

    explicit TestRng(uint64_t seed) : engine(seed) {}
    double uniform() { return static_cast<double>(engine() >> 11) * 0x1.0p-53; }
    uint32_t below(uint32_t n) { return static_cast<uint32_t>(engine() % n); }
    bool coin() { return (engine() & 1) != 0; }
};

// ---------------------------------------------------------------------------
// Brute-force amplitudes of a closed reduced graph, for every assignment of
// the parameters (bit k of the index is params[k]).

inline std::optional<std::vector<double>> amplitude_table(
    const Graph& g, const std::vector<uint32_t>& params, const std::vector<Constraint>& constraints,
    size_t max_groups = 16) {
    const std::vector<Vertex> vs = g.vertices();
    std::vector<size_t> parent(g.id_bound());
    std::iota(parent.begin(), parent.end(), size_t{0});
    auto find = [&](size_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    // An X spider carries a Hadamard on every leg, so an edge contributes a
    // Hadamard matrix iff its own type and its X endpoints have odd parity;
    // otherwise it identifies the two spiders' variables.
    std::vector<std::pair<Vertex, Vertex>> hadamards;
    for (Vertex u : vs) {
        for (const Neighbor& n : g.neighbors(u)) {
            if (n.vertex < u) {
                continue;
            }
            const int k = (g.type(u) == VertexType::X) + (g.type(n.vertex) == VertexType::X) +
                          (n.type == EdgeType::Hadamard);
            if (k % 2 == 0) {
                parent[find(u)] = find(n.vertex);
            } else {
                hadamards.emplace_back(u, n.vertex);
            }
        }
    }
    std::vector<size_t> group(g.id_bound(), 0);
    size_t num_groups = 0;
    std::vector<size_t> root_index(g.id_bound(), SIZE_MAX);
    for (Vertex v : vs) {
        const size_t r = find(v);
        if (root_index[r] == SIZE_MAX) {
            root_index[r] = num_groups++;
        }
        group[v] = root_index[r];
    }
    if (num_groups > max_groups) {
        return std::nullopt;
    }
    const size_t num_x = size_t{1} << num_groups;
    std::vector<int> sign(num_x, 1);
    for (size_t x = 0; x < num_x; ++x) {
        int parity = 0;
        for (const auto& [a, b] : hadamards) {
            parity ^= static_cast<int>((x >> group[a]) & (x >> group[b]) & 1);
        }
        sign[x] = parity ? -1 : 1;
    }
    auto index_of = [&](uint32_t p) {
        const auto it = std::ranges::find(params, p);
        assert(it != params.end());
        return static_cast<size_t>(it - params.begin());
    };
    auto parity_of = [&](const Params& ps, uint64_t assignment) {
        int parity = 0;
        for (uint32_t p : ps.ids()) {
            parity ^= static_cast<int>((assignment >> index_of(p)) & 1);
        }
        return parity;
    };
    std::vector<double> out(size_t{1} << params.size());
    for (uint64_t a = 0; a < out.size(); ++a) {
        bool ok = true;
        for (const Constraint& c : constraints) {
            ok &= parity_of(c.params, a) == static_cast<int>(c.rhs);
        }
        if (!ok) {
            out[a] = 0.0;
            continue;
        }
        std::vector<double> coefficient(num_groups, 0.0);
        for (Vertex v : vs) {
            coefficient[group[v]] += g.phase(v).value + parity_of(g.phase(v).params, a);
        }
        std::complex<double> amp = 0.0;
        for (size_t x = 0; x < num_x; ++x) {
            double angle = 0.0;
            for (size_t r = 0; r < num_groups; ++r) {
                if ((x >> r) & 1) {
                    angle += coefficient[r];
                }
            }
            amp += static_cast<double>(sign[x]) * std::polar(1.0, std::numbers::pi * angle);
        }
        out[a] = std::abs(amp);
    }
    return out;
}

// Rounding leaves magnitudes near 1e-16 where an amplitude vanishes exactly, so
// zero is judged relative to the table's largest entry with an absolute floor.
inline double zero_threshold(double max_value) {
    return 1e-9 * std::max(max_value, 1.0);
}

}  // namespace zx_test
