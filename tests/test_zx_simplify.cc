#include "clifft/zx/canonical.h"
#include "clifft/zx/graph.h"
#include "clifft/zx/simplify.h"

#include "zx_test_helpers.h"

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace clifft::zx;
using zx_test::amplitude_table;
using zx_test::zero_threshold;

namespace {

// ---------------------------------------------------------------------------
// Fixtures from tools/zx_fixtures/generate.py.

struct Step {
    std::string rule;
    uint64_t hash;
};

struct FixtureCase {
    std::string name;
    struct VertexLine {
        VertexType type;
        double phase;
        std::vector<uint32_t> params;
    };
    std::vector<VertexLine> vertices;
    std::vector<std::vector<Neighbor>> adjacency;
    std::vector<Step> steps;
    uint64_t final_hash = 0;
    std::vector<ScalarTerm> nodes;
    std::vector<ScalarTerm> pairs;
    std::vector<uint32_t> params;
    std::vector<bool> param_is_noise;
    std::vector<double> reference;
};

double parse_fraction(const std::string& text) {
    const size_t slash = text.find('/');
    return std::stod(text.substr(0, slash)) / std::stod(text.substr(slash + 1));
}

std::vector<uint32_t> parse_params(const std::string& text) {
    std::vector<uint32_t> out;
    if (text == "-") {
        return out;
    }
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return out;
}

std::vector<FixtureCase> load_fixtures(const std::string& file) {
    std::ifstream in(std::string(CLIFFT_FIXTURES_DIR) + "/zx/" + file);
    REQUIRE(in.good());
    std::vector<FixtureCase> cases;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "case") {
            cases.emplace_back();
            ss >> cases.back().name;
            continue;
        }
        FixtureCase& c = cases.back();
        if (tag == "v") {
            uint32_t id = 0;
            std::string type, phase, params;
            ss >> id >> type >> phase >> params;
            REQUIRE(id == c.vertices.size());
            c.vertices.push_back({.type = type == "Z" ? VertexType::Z : VertexType::X,
                                  .phase = parse_fraction(phase),
                                  .params = parse_params(params)});
        } else if (tag == "a") {
            uint32_t id = 0;
            ss >> id;
            c.adjacency.resize(c.vertices.size());
            std::string item;
            while (ss >> item) {
                c.adjacency[id].push_back(Neighbor{
                    .vertex = static_cast<Vertex>(std::stoul(item.substr(0, item.size() - 1))),
                    .type = item.back() == 's' ? EdgeType::Simple : EdgeType::Hadamard});
            }
        } else if (tag == "step") {
            Step step;
            std::string hash;
            ss >> step.rule >> hash;
            step.hash = std::stoull(hash, nullptr, 16);
            c.steps.push_back(step);
        } else if (tag == "final") {
            std::string hash;
            ss >> hash;
            c.final_hash = std::stoull(hash, nullptr, 16);
        } else if (tag == "node") {
            std::string alpha, params;
            ss >> alpha >> params;
            c.nodes.push_back(ScalarTerm{.kind = ScalarTerm::Kind::Node,
                                         .alpha = parse_fraction(alpha),
                                         .beta = 0.0,
                                         .a = Params(parse_params(params)),
                                         .b = {}});
        } else if (tag == "pair") {
            std::string alpha, beta, a, b;
            ss >> alpha >> beta >> a >> b;
            c.pairs.push_back(ScalarTerm{.kind = ScalarTerm::Kind::PhasePair,
                                         .alpha = parse_fraction(alpha),
                                         .beta = parse_fraction(beta),
                                         .a = Params(parse_params(a)),
                                         .b = Params(parse_params(b))});
        } else if (tag == "params") {
            std::string item;
            while (ss >> item) {
                c.params.push_back(
                    static_cast<uint32_t>(std::stoul(item.substr(0, item.size() - 1))));
                c.param_is_noise.push_back(item.back() == 'e');
            }
        } else if (tag == "ref") {
            std::string item;
            while (ss >> item) {
                c.reference.push_back(std::stod(item));
            }
        }
    }
    return cases;
}

Graph load_graph(const FixtureCase& c) {
    Graph g;
    for (const auto& v : c.vertices) {
        g.add_vertex(v.type, Phase(v.phase, Params(v.params)));
    }
    for (Vertex v = 0; v < c.adjacency.size(); ++v) {
        g.set_neighbors_for_loading(v, c.adjacency[v]);
    }
    return g;
}

const char* const kFixtureFiles[] = {
    "single_copy_random.txt",
    "single_copy_random_t.txt",
    "single_copy_random_clifford.txt",
    "single_copy_distillation.txt",
    "single_copy_msc_d3_inject_cultivate_p1e-3.txt",
    "single_copy_coherent_d3_r3.txt",
};

// Whether `reduced` is proportional to `reference` for each noise pattern.
std::string proportionality_error(const FixtureCase& c, const std::vector<double>& reduced) {
    std::vector<uint64_t> noise_key(reduced.size(), 0);
    for (uint64_t a = 0; a < reduced.size(); ++a) {
        for (size_t k = 0, j = 0; k < c.params.size(); ++k) {
            if (c.param_is_noise[k]) {
                noise_key[a] |= ((a >> k) & 1) << j++;
            }
        }
    }
    const uint64_t num_keys = *std::ranges::max_element(noise_key) + 1;
    for (uint64_t key = 0; key < num_keys; ++key) {
        double max_ref = 0.0, max_red = 0.0;
        for (uint64_t a = 0; a < reduced.size(); ++a) {
            if (noise_key[a] == key) {
                max_ref = std::max(max_ref, c.reference[a]);
                max_red = std::max(max_red, reduced[a]);
            }
        }
        std::optional<double> ratio;
        for (uint64_t a = 0; a < reduced.size(); ++a) {
            if (noise_key[a] != key) {
                continue;
            }
            const bool ref_zero = c.reference[a] <= zero_threshold(max_ref);
            const bool red_zero = reduced[a] <= zero_threshold(max_red);
            if (ref_zero != red_zero) {
                return "support differs at assignment " + std::to_string(a);
            }
            if (ref_zero) {
                continue;
            }
            const double r = reduced[a] / c.reference[a];
            if (!ratio) {
                ratio = r;
            } else if (std::abs(r - *ratio) > 1e-4 * *ratio) {
                return "amplitude ratio varies at assignment " + std::to_string(a);
            }
        }
    }
    return "";
}

// Whether |after| = K |before| for one constant K over every assignment.
bool globally_proportional(const std::vector<double>& before, const std::vector<double>& after) {
    const double max_before = *std::ranges::max_element(before);
    const double max_after = *std::ranges::max_element(after);
    std::optional<double> ratio;
    for (size_t a = 0; a < before.size(); ++a) {
        const bool zero_before = before[a] <= zero_threshold(max_before);
        const bool zero_after = after[a] <= zero_threshold(max_after);
        if (zero_before != zero_after) {
            return false;
        }
        if (zero_before) {
            continue;
        }
        const double r = after[a] / before[a];
        if (!ratio) {
            ratio = r;
        } else if (std::abs(r - *ratio) > 1e-7 * *ratio) {
            return false;
        }
    }
    return true;
}

// A random closed diagram: Z and X spiders with phases on the pi/4 grid (and
// some arbitrary ones), Boolean parameters, and random plain or Hadamard edges.
Graph random_diagram(zx_test::TestRng& rng, size_t num_vertices, uint32_t num_params) {
    Graph g;
    auto coin = [&](zx_test::TestRng& r) { return r.coin() ? 1 : 0; };
    auto quarters = [](zx_test::TestRng& r) { return static_cast<int>(r.below(8)); };
    auto param = [&](zx_test::TestRng& r) { return r.below(num_params); };
    auto uniform = [](zx_test::TestRng& r) { return r.uniform(); };
    for (size_t i = 0; i < num_vertices; ++i) {
        double value = quarters(rng) / 4.0;
        if (uniform(rng) < 0.1) {
            value = uniform(rng) * 2.0;
        }
        std::vector<uint32_t> ps;
        if (uniform(rng) < 0.4) {
            ps.push_back(param(rng));
            if (uniform(rng) < 0.3) {
                ps.push_back(param(rng));
            }
        }
        g.add_vertex(coin(rng) ? VertexType::Z : VertexType::X, Phase(value, Params(ps)));
    }
    const double density = 2.5 / static_cast<double>(num_vertices);
    for (Vertex u = 0; u < num_vertices; ++u) {
        for (Vertex v = u + 1; v < num_vertices; ++v) {
            if (uniform(rng) < density) {
                g.add_edge(u, v, coin(rng) ? EdgeType::Hadamard : EdgeType::Simple);
            }
        }
    }
    // Phase gadgets that share legs, with bodies whose parities sometimes
    // differ: the case gadget fusion must keep apart.
    if (num_vertices >= 3 && uniform(rng) < 0.6) {
        auto pick = [&](zx_test::TestRng& r) {
            return r.below(static_cast<uint32_t>(num_vertices));
        };
        std::vector<Vertex> legs = {pick(rng), pick(rng)};
        std::ranges::sort(legs);
        legs.erase(std::unique(legs.begin(), legs.end()), legs.end());
        for (int k = 0; k < 2 + coin(rng); ++k) {
            std::vector<uint32_t> ps;
            if (coin(rng)) {
                ps.push_back(param(rng));
            }
            const Vertex body = g.add_vertex(VertexType::Z, Phase(coin(rng), Params(ps)));
            const Vertex head = g.add_vertex(VertexType::Z, Phase(quarters(rng) / 4.0));
            g.add_edge(body, head, EdgeType::Hadamard);
            for (Vertex leg : legs) {
                g.add_edge(body, leg, EdgeType::Hadamard);
            }
        }
    }
    return g;
}

}  // namespace

TEST_CASE("Zx every rule pass keeps amplitude magnitudes", "[zx]") {
    // Independent of zxparam: each pass on its own, then the full strategy,
    // must leave |amplitude| proportional for every parameter assignment once
    // the dropped factors' constraints are applied.
    zx_test::TestRng rng(20261021);
    const uint32_t num_params = 4;
    const std::vector<uint32_t> params = {0, 1, 2, 3};
    using Pass = bool (*)(Graph&);
    const std::pair<const char*, Pass> passes[] = {
        {"to_gh", rules::to_gh},
        {"spider_fusion", rules::spider_fusion},
        {"copy", rules::copy},
        {"id_remove", rules::id_remove},
        {"pivot", rules::pivot},
        {"lcomp", rules::lcomp},
        {"pivot_gadget", rules::pivot_gadget},
        // Fusion before normalization, so bodies that differ by a concrete pi
        // reach the fusion rule.
        {"gadget_fusion", rules::gadget_fusion},
        {"gadget_normalize", rules::gadget_normalize},
        {"remove_isolated", [](Graph& g) { return rules::remove_isolated(g, ReduceOptions{}); }},
    };
    std::vector<size_t> fired(std::size(passes), 0);
    for (int trial = 0; trial < 3000; ++trial) {
        const size_t n = 4 + trial % 7;
        Graph g = random_diagram(rng, n, num_params);
        const auto before = amplitude_table(g, params, {});
        REQUIRE(before);
        // Graph-like form first, as the strategy has it before every other rule.
        Graph work = g;
        if (rules::to_gh(work)) {
            ++fired[0];
            INFO("trial " << trial << " pass to_gh");
            REQUIRE(globally_proportional(*before, *amplitude_table(work, params, {})));
        }
        rules::spider_fusion(work);
        for (size_t k = 0; k < std::size(passes); ++k) {
            INFO("trial " << trial << " pass " << passes[k].first);
            if (!passes[k].second(work)) {
                continue;
            }
            ++fired[k];
            const auto after = amplitude_table(work, params, work.constraints());
            REQUIRE(after);
            REQUIRE(globally_proportional(*before, *after));
        }
        Graph reduced = g;
        full_reduce(reduced);
        const auto after = amplitude_table(reduced, params, reduced.constraints());
        REQUIRE(after);
        INFO("trial " << trial << " full_reduce");
        REQUIRE(globally_proportional(*before, *after));
    }
    for (size_t k = 0; k < std::size(passes); ++k) {
        INFO(passes[k].first);
        CHECK(fired[k] >= 10);
    }
}

TEST_CASE("Zx phases snap to the pi over 4 grid and reduce mod 2", "[zx]") {
    CHECK(Phase(2.25).value == 0.25);
    CHECK(Phase(-0.25).value == 1.75);
    CHECK(Phase(0.1 + 0.15).value == 0.25);
    CHECK(Phase(1.0 / 3.0).plus(Phase(2.0 / 3.0)).is_pi());
    CHECK(Phase(0.5).is_proper_clifford());
    CHECK(Phase(1.0, Params({7})).is_pauli());
    CHECK_FALSE(Phase(0.0, Params({7})).is_zero());
    CHECK(Phase(0.25, Params({3})).negated() == Phase(1.75, Params({3})));
}

TEST_CASE("Zx params combine by xor", "[zx]") {
    Params a({5, 1, 3});
    a.xor_with(Params({3, 9}));
    CHECK(std::vector<uint32_t>(a.ids().begin(), a.ids().end()) == std::vector<uint32_t>{1, 5, 9});
    CHECK(Params({2, 2}).empty());
}

TEST_CASE("Zx parallel edges fuse cancel or add pi", "[zx]") {
    Graph g;
    const Vertex a = g.add_vertex(VertexType::Z);
    const Vertex b = g.add_vertex(VertexType::Z);
    const Vertex c = g.add_vertex(VertexType::X);
    g.add_edge(a, b, EdgeType::Simple);
    g.add_edge(a, b, EdgeType::Simple);
    CHECK(g.edge_type(a, b) == EdgeType::Simple);
    g.add_edge(a, b, EdgeType::Hadamard);
    CHECK(g.edge_type(a, b) == EdgeType::Simple);
    CHECK(g.phase(a).is_pi());
    g.add_edge(a, c, EdgeType::Simple);
    g.add_edge(a, c, EdgeType::Simple);
    CHECK_FALSE(g.has_edge(a, c));
}

TEST_CASE("Zx neighbor order follows insertion and survives updates", "[zx]") {
    Graph g;
    const Vertex hub = g.add_vertex(VertexType::Z);
    std::vector<Vertex> leaves;
    for (int i = 0; i < 4; ++i) {
        leaves.push_back(g.add_vertex(VertexType::Z));
        g.add_edge(hub, leaves.back(), EdgeType::Simple);
    }
    g.toggle_edge(hub, leaves[1]);
    g.remove_edge(hub, leaves[0]);
    g.add_edge(hub, leaves[0], EdgeType::Hadamard);
    std::vector<Neighbor> want = {{leaves[1], EdgeType::Hadamard},
                                  {leaves[2], EdgeType::Simple},
                                  {leaves[3], EdgeType::Simple},
                                  {leaves[0], EdgeType::Hadamard}};
    CHECK(std::vector<Neighbor>(g.neighbors(hub).begin(), g.neighbors(hub).end()) == want);
}

TEST_CASE("Zx full reduce replays zxparam pass by pass", "[zx]") {
    size_t num_cases = 0;
    for (const char* file : kFixtureFiles) {
        for (const FixtureCase& c : load_fixtures(file)) {
            INFO(c.name);
            Graph g = load_graph(c);
            std::vector<Step> steps;
            full_reduce(g, ReduceOptions{.keep_weighted_isolated = false},
                        [&](std::string_view rule, bool changed, const Graph& h) {
                            if (changed) {
                                steps.push_back({std::string(rule), fnv1a64(canonical_text(h))});
                            }
                        });
            REQUIRE(steps.size() == c.steps.size());
            for (size_t i = 0; i < steps.size(); ++i) {
                INFO("step " << i);
                REQUIRE(steps[i].rule == c.steps[i].rule);
                REQUIRE(steps[i].hash == c.steps[i].hash);
            }
            CHECK(fnv1a64(canonical_text(g)) == c.final_hash);
            std::vector<ScalarTerm> nodes, pairs;
            for (const ScalarTerm& t : g.scalar_terms()) {
                (t.kind == ScalarTerm::Kind::Node ? nodes : pairs).push_back(t);
            }
            CHECK(nodes == c.nodes);
            CHECK(pairs == c.pairs);
            ++num_cases;
        }
    }
    CHECK(num_cases == 78);
}

TEST_CASE("Zx full reduce keeps single copy amplitudes", "[zx]") {
    size_t checked = 0;
    for (const char* file : {"single_copy_random.txt", "single_copy_random_t.txt",
                             "single_copy_random_clifford.txt"}) {
        for (const FixtureCase& c : load_fixtures(file)) {
            INFO(c.name);
            REQUIRE(c.reference.size() == (size_t{1} << c.params.size()));
            Graph g = load_graph(c);
            full_reduce(g);
            // Every dropped factor now has a parameter-independent modulus or
            // is a constraint, so the constraints alone account for the scalar.
            for (const ScalarTerm& t : g.scalar_terms()) {
                const bool pauli = t.alpha == 0.0 || t.alpha == 1.0;
                if (t.kind == ScalarTerm::Kind::Node) {
                    CHECK((pauli || t.a.empty()));
                } else {
                    CHECK(((pauli && (t.beta == 0.0 || t.beta == 1.0)) ||
                           (t.a.empty() && t.b.empty())));
                }
            }
            const auto table = amplitude_table(g, c.params, g.constraints());
            if (!table) {
                continue;
            }
            CHECK(proportionality_error(c, *table).empty());
            ++checked;
        }
    }
    CHECK(checked >= 72);
}
