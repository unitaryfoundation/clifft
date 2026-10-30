#include "clifft/zx/simplify.h"

#include "clifft/util/numeric.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace clifft::zx {
namespace {

using EdgePairs = std::vector<std::pair<Vertex, Vertex>>;

// The rewrite engine: collect every match, then apply each that still checks.
template <typename Match, typename Collect, typename Check, typename Apply>
bool run_pass(Graph& g, Collect collect, Check check, Apply apply) {
    const std::vector<Match> matches = collect(g);
    bool changed = false;
    for (const Match& m : matches) {
        if (g.over_work_limit()) {
            break;
        }
        if (check(g, m)) {
            apply(g, m);
            changed = true;
        }
    }
    return changed;
}

template <typename Step>
bool saturate(Graph& g, Step step) {
    bool changed = false;
    while (step(g)) {
        changed = true;
    }
    return changed;
}

template <typename Match, typename Valid>
std::vector<Match> vertex_matches(const Graph& g, Valid valid) {
    std::vector<Match> found;
    for (Vertex v : g.vertices()) {
        if (std::optional<Match> m = valid(g, v)) {
            found.push_back(*m);
        }
    }
    return found;
}

template <typename Match, typename Valid>
std::vector<Match> edge_matches(const Graph& g, Valid valid) {
    std::vector<Match> found;
    for (Vertex u : g.vertices()) {
        for (const Neighbor& n : g.neighbors(u)) {
            if (n.vertex > u) {
                if (std::optional<Match> m = valid(g, u, n.vertex)) {
                    found.push_back(*m);
                }
            }
        }
    }
    return found;
}

template <typename Match, typename Valid>
bool vertex_check(const Graph& g, Vertex v, const Match& m, Valid valid) {
    if (!g.contains(v)) {
        return false;
    }
    const std::optional<Match> now = valid(g, v);
    return now && *now == m;
}

template <typename Match, typename Valid>
bool edge_check(const Graph& g, Vertex u, Vertex v, const Match& m, Valid valid) {
    if (!g.contains(u) || !g.contains(v) || !g.has_edge(u, v)) {
        return false;
    }
    const std::optional<Match> now = valid(g, u, v);
    return now && *now == m;
}

std::vector<Vertex> sorted_neighbors(const Graph& g, Vertex v) {
    std::vector<Vertex> out;
    out.reserve(g.degree(v));
    for (const Neighbor& n : g.neighbors(v)) {
        out.push_back(n.vertex);
    }
    std::ranges::sort(out);
    return out;
}

// Whether every neighbor of v outside `exclude` is a Z spider behind a Hadamard edge.
bool all_interior_z(const Graph& g, Vertex v, std::optional<Vertex> exclude) {
    return std::ranges::all_of(g.neighbors(v), [&](const Neighbor& n) {
        return n.vertex == exclude ||
               (g.type(n.vertex) == VertexType::Z && n.type == EdgeType::Hadamard);
    });
}

struct VertexPair {
    Vertex u;
    Vertex v;
    bool operator==(const VertexPair&) const = default;
};

struct Single {
    Vertex v;
    bool operator==(const Single&) const = default;
};

// ---------------------------------------------------------------------------
// Spider fusion: two same-color spiders joined by a plain edge.

std::optional<VertexPair> fusion_valid(const Graph& g, Vertex u, Vertex v) {
    if (g.type(u) != g.type(v) || g.edge_type(u, v) != EdgeType::Simple) {
        return std::nullopt;
    }
    return VertexPair{u, v};
}

void fusion_apply(Graph& g, const VertexPair& m) {
    // Single-copy fusion keeps the lower ID, as zxparam does for classical spiders.
    const Vertex u = m.u;
    const Vertex v = m.v;
    g.add_to_phase(u, g.phase(v));
    const std::vector<Neighbor> moved(g.neighbors(v).begin(), g.neighbors(v).end());
    for (const Neighbor& n : moved) {
        if (n.vertex == u) {
            continue;
        }
        g.remove_edge(v, n.vertex);
        g.add_edge(u, n.vertex, n.type);
    }
    g.remove_vertex(v);
}

// ---------------------------------------------------------------------------
// Copy: an arity-1 Pauli Z spider copied through its Hadamard neighbor.

std::optional<VertexPair> copy_valid(const Graph& g, Vertex v) {
    if (g.type(v) != VertexType::Z || g.degree(v) != 1 || !g.phase(v).is_pauli()) {
        return std::nullopt;
    }
    const Neighbor first = g.neighbors(v).front();
    const Vertex w = first.vertex;
    if (g.type(w) != VertexType::Z || first.type != EdgeType::Hadamard) {
        return std::nullopt;
    }
    const bool absorbs = std::ranges::all_of(g.neighbors(w), [&](const Neighbor& n) {
        return n.vertex == v || (g.type(n.vertex) == VertexType::Z && n.type == EdgeType::Hadamard);
    });
    if (!absorbs) {
        return std::nullopt;
    }
    return VertexPair{v, w};
}

void copy_apply(Graph& g, const VertexPair& m) {
    const Vertex v = m.u;
    const Vertex w = m.v;
    const Phase ph_v = g.phase(v);
    for (Vertex n : sorted_neighbors(g, w)) {
        if (n != v) {
            g.add_to_phase(n, ph_v);
        }
    }
    g.remove_vertex(v);
    g.remove_vertex(w);
}

// ---------------------------------------------------------------------------
// Identity removal: a phase-0 spider with two edges.

std::optional<Single> id_valid(const Graph& g, Vertex v) {
    if (!g.phase(v).is_zero() || g.degree(v) != 2) {
        return std::nullopt;
    }
    return Single{v};
}

void id_apply(Graph& g, const Single& m) {
    const auto nbrs = g.neighbors(m.v);
    const Neighbor a = nbrs[0];
    const Neighbor b = nbrs[1];
    const EdgeType combined = a.type != b.type ? EdgeType::Hadamard : EdgeType::Simple;
    g.remove_vertex(m.v);
    g.add_edge(a.vertex, b.vertex, combined);
}

// ---------------------------------------------------------------------------
// Local complementation at a Z spider with phase +-pi/2.

std::optional<Single> lcomp_valid(const Graph& g, Vertex v) {
    if (g.type(v) != VertexType::Z || !g.phase(v).is_proper_clifford() ||
        !all_interior_z(g, v, std::nullopt)) {
        return std::nullopt;
    }
    return Single{v};
}

void lcomp_apply(Graph& g, const Single& m) {
    const Vertex v = m.v;
    const Phase minus = g.phase(v).negated();
    const std::vector<Vertex> nbrs = sorted_neighbors(g, v);
    EdgePairs pairs;
    for (size_t i = 0; i < nbrs.size(); ++i) {
        for (size_t j = i + 1; j < nbrs.size(); ++j) {
            pairs.emplace_back(nbrs[i], nbrs[j]);
        }
    }
    g.add_hadamard_edges(pairs);
    for (Vertex w : nbrs) {
        g.add_to_phase(w, minus);
    }
    g.remove_vertex(v);
}

// ---------------------------------------------------------------------------
// The pivot family.

struct Parts {
    std::vector<Vertex> only_u;
    std::vector<Vertex> only_v;
    std::vector<Vertex> shared;
};

// Neighbor order decides where toggled edges land and so the order of later
// matches: only_u and shared follow u's order, only_v follows v's.
Parts partitions(const Graph& g, Vertex u, Vertex v) {
    Parts parts;
    for (const Neighbor& n : g.neighbors(u)) {
        if (n.vertex == v) {
            continue;
        }
        (g.has_edge(v, n.vertex) ? parts.shared : parts.only_u).push_back(n.vertex);
    }
    for (const Neighbor& n : g.neighbors(v)) {
        if (n.vertex != u && !g.has_edge(u, n.vertex)) {
            parts.only_v.push_back(n.vertex);
        }
    }
    return parts;
}

void pivot_algebra(Graph& g, const Phase& ph_u, const Phase& ph_v, const Parts& parts) {
    EdgePairs pairs;
    const std::pair<const std::vector<Vertex>*, const std::vector<Vertex>*> products[] = {
        {&parts.only_u, &parts.only_v},
        {&parts.only_u, &parts.shared},
        {&parts.only_v, &parts.shared},
    };
    for (const auto& [xs, ys] : products) {
        for (Vertex a : *xs) {
            for (Vertex b : *ys) {
                if (a != b) {
                    pairs.emplace_back(a, b);
                }
            }
        }
    }
    g.add_hadamard_edges(pairs);
    const Phase pi(1.0);
    for (Vertex w : parts.shared) {
        g.add_to_phase(w, pi);
    }
    if (!ph_v.is_zero()) {
        for (Vertex w : parts.only_u) {
            g.add_to_phase(w, ph_v);
        }
        for (Vertex w : parts.shared) {
            g.add_to_phase(w, ph_v);
        }
    }
    if (!ph_u.is_zero()) {
        for (Vertex w : parts.only_v) {
            g.add_to_phase(w, ph_u);
        }
        for (Vertex w : parts.shared) {
            g.add_to_phase(w, ph_u);
        }
    }
}

std::optional<VertexPair> pivot_valid(const Graph& g, Vertex u, Vertex v) {
    if (g.edge_type(u, v) != EdgeType::Hadamard || g.type(u) != VertexType::Z ||
        g.type(v) != VertexType::Z || !g.phase(u).is_pauli() || !g.phase(v).is_pauli()) {
        return std::nullopt;
    }
    if (!all_interior_z(g, u, v) || !all_interior_z(g, v, u)) {
        return std::nullopt;
    }
    return VertexPair{u, v};
}

void pivot_apply(Graph& g, const VertexPair& m) {
    const Phase ph_u = g.phase(m.u);
    const Phase ph_v = g.phase(m.v);
    pivot_algebra(g, ph_u, ph_v, partitions(g, m.u, m.v));
    g.remove_vertex(m.u);
    g.remove_vertex(m.v);
}

// Gadgetize a non-Pauli spider v by pivoting on its Pauli neighbor u.
std::optional<VertexPair> pivot_gadget_valid(const Graph& g, Vertex a, Vertex b) {
    if (g.edge_type(a, b) != EdgeType::Hadamard || g.type(a) != VertexType::Z ||
        g.type(b) != VertexType::Z) {
        return std::nullopt;
    }
    const bool pauli_a = g.phase(a).is_pauli();
    const bool pauli_b = g.phase(b).is_pauli();
    if (pauli_a == pauli_b) {
        return std::nullopt;
    }
    const Vertex u = pauli_a ? a : b;
    const Vertex v = pauli_a ? b : a;
    // A leaf v is already a gadget head; gadgetizing it again would loop.
    if (g.degree(v) <= 1) {
        return std::nullopt;
    }
    if (!all_interior_z(g, u, v) || !all_interior_z(g, v, u)) {
        return std::nullopt;
    }
    // A gadget body u would lose its leaf to a toggle partition, so saturation
    // would not end.
    for (const Neighbor& n : g.neighbors(u)) {
        if (n.vertex != v && g.degree(n.vertex) == 1) {
            return std::nullopt;
        }
    }
    return VertexPair{u, v};
}

void pivot_gadget_apply(Graph& g, const VertexPair& m) {
    const Vertex u = m.u;
    const Vertex v = m.v;
    const Phase ph_u = g.phase(u);
    const Phase ph_v = g.phase(v);
    pivot_algebra(g, ph_u, Phase{}, partitions(g, u, v));
    g.remove_vertex(v);
    const Vertex head = g.add_vertex(VertexType::Z, ph_v);
    g.add_edge(u, head, EdgeType::Hadamard);
}

// ---------------------------------------------------------------------------
// Phase gadgets: a Pauli body with exactly one Hadamard leaf, the head.

std::optional<Vertex> gadget_head(const Graph& g, Vertex body) {
    if (g.type(body) != VertexType::Z || !g.phase(body).is_pauli() || g.degree(body) < 2) {
        return std::nullopt;
    }
    // zxparam leaves the legs unchecked because the strategy only calls the
    // gadget rules on graph-like diagrams. A plain leg would merge the body
    // into its leg, so requiring Hadamard legs makes the rules sound on their
    // own without changing any match the strategy sees.
    if (!all_interior_z(g, body, std::nullopt)) {
        return std::nullopt;
    }
    std::optional<Vertex> head;
    for (const Neighbor& n : g.neighbors(body)) {
        if (g.degree(n.vertex) == 1) {
            if (head) {
                return std::nullopt;
            }
            head = n.vertex;
        }
    }
    return head;
}

std::optional<VertexPair> normalize_valid(const Graph& g, Vertex v) {
    if (!g.phase(v).is_pi()) {
        return std::nullopt;
    }
    const std::optional<Vertex> head = gadget_head(g, v);
    if (!head) {
        return std::nullopt;
    }
    return VertexPair{v, *head};
}

void normalize_apply(Graph& g, const VertexPair& m) {
    // e^{i a (1 + q)} = e^{i a} e^{-i a q}: the body loses its pi and the head is negated.
    const Phase alpha = g.phase(m.v);
    g.set_phase(m.u, g.phase(m.u).plus_pi());
    g.set_phase(m.v, alpha.negated());
}

// The legs and the body's parameters: a gadget body's parity sets the sign of
// the gadget's axis, so only gadgets with equal parities may fuse.
using GadgetKey = std::pair<std::vector<Vertex>, Params>;

std::optional<std::pair<Vertex, GadgetKey>> gadget(const Graph& g, Vertex body) {
    const std::optional<Vertex> head = gadget_head(g, body);
    if (!head) {
        return std::nullopt;
    }
    std::vector<Vertex> legs;
    for (const Neighbor& n : g.neighbors(body)) {
        if (n.vertex != *head) {
            legs.push_back(n.vertex);
        }
    }
    if (legs.empty()) {
        return std::nullopt;
    }
    std::ranges::sort(legs);
    return std::pair{*head, GadgetKey{std::move(legs), g.phase(body).params}};
}

struct GadgetFusion {
    Vertex gb1, gh1, gb2, gh2;
    bool operator==(const GadgetFusion&) const = default;
};

std::vector<GadgetFusion> gadget_fusion_matches(const Graph& g) {
    std::vector<std::vector<std::pair<Vertex, Vertex>>> buckets;
    std::map<GadgetKey, size_t> bucket_of;
    for (Vertex v : g.vertices()) {
        auto found = gadget(g, v);
        if (!found) {
            continue;
        }
        auto [it, inserted] = bucket_of.try_emplace(std::move(found->second), buckets.size());
        if (inserted) {
            buckets.emplace_back();
        }
        buckets[it->second].emplace_back(v, found->first);
    }
    std::vector<GadgetFusion> out;
    for (const auto& bucket : buckets) {
        for (size_t k = 1; k < bucket.size(); ++k) {
            out.push_back(
                GadgetFusion{bucket[0].first, bucket[0].second, bucket[k].first, bucket[k].second});
        }
    }
    return out;
}

bool gadget_fusion_check(const Graph& g, const GadgetFusion& m) {
    if (m.gb1 == m.gb2) {
        return false;
    }
    for (Vertex x : {m.gb1, m.gh1, m.gb2, m.gh2}) {
        if (!g.contains(x)) {
            return false;
        }
    }
    const auto first = gadget(g, m.gb1);
    const auto second = gadget(g, m.gb2);
    return first && second && first->first == m.gh1 && second->first == m.gh2 &&
           first->second == second->second;
}

void gadget_fusion_apply(Graph& g, const GadgetFusion& m) {
    const Phase alpha2 = g.phase(m.gh2);
    if (g.phase(m.gb1) == g.phase(m.gb2)) {
        g.add_to_phase(m.gh1, alpha2);
    } else {
        // The bodies differ by a concrete pi: merge alpha1 - alpha2.
        g.set_phase(m.gh1, g.phase(m.gh1).minus(alpha2));
    }
    g.remove_vertex(m.gh2);
    g.remove_vertex(m.gb2);
}

bool weighted(double alpha, const Params& params) {
    return !params.empty() && alpha != 0.0 && alpha != 1.0;
}

}  // namespace

namespace rules {

bool to_gh(Graph& g) {
    bool changed = false;
    for (Vertex v : g.vertices()) {
        if (g.type(v) != VertexType::X) {
            continue;
        }
        g.set_type(v, VertexType::Z);
        const std::vector<Neighbor> nbrs(g.neighbors(v).begin(), g.neighbors(v).end());
        for (const Neighbor& n : nbrs) {
            g.toggle_edge(v, n.vertex);
        }
        changed = true;
    }
    return changed;
}

bool spider_fusion(Graph& g) {
    return run_pass<VertexPair>(
        g, [](const Graph& h) { return edge_matches<VertexPair>(h, fusion_valid); },
        [](const Graph& h, const VertexPair& m) {
            return edge_check(h, m.u, m.v, m, fusion_valid);
        },
        fusion_apply);
}

bool copy(Graph& g) {
    return run_pass<VertexPair>(
        g, [](const Graph& h) { return vertex_matches<VertexPair>(h, copy_valid); },
        [](const Graph& h, const VertexPair& m) { return vertex_check(h, m.u, m, copy_valid); },
        copy_apply);
}

bool id_remove(Graph& g) {
    return run_pass<Single>(
        g, [](const Graph& h) { return vertex_matches<Single>(h, id_valid); },
        [](const Graph& h, const Single& m) { return vertex_check(h, m.v, m, id_valid); },
        id_apply);
}

bool pivot(Graph& g) {
    return run_pass<VertexPair>(
        g, [](const Graph& h) { return edge_matches<VertexPair>(h, pivot_valid); },
        [](const Graph& h, const VertexPair& m) { return edge_check(h, m.u, m.v, m, pivot_valid); },
        pivot_apply);
}

bool lcomp(Graph& g) {
    return run_pass<Single>(
        g, [](const Graph& h) { return vertex_matches<Single>(h, lcomp_valid); },
        [](const Graph& h, const Single& m) { return vertex_check(h, m.v, m, lcomp_valid); },
        lcomp_apply);
}

bool pivot_gadget(Graph& g) {
    return run_pass<VertexPair>(
        g, [](const Graph& h) { return edge_matches<VertexPair>(h, pivot_gadget_valid); },
        [](const Graph& h, const VertexPair& m) {
            return edge_check(h, m.u, m.v, m, pivot_gadget_valid);
        },
        pivot_gadget_apply);
}

bool gadget_normalize(Graph& g) {
    return run_pass<VertexPair>(
        g, [](const Graph& h) { return vertex_matches<VertexPair>(h, normalize_valid); },
        [](const Graph& h, const VertexPair& m) {
            return vertex_check(h, m.u, m, normalize_valid);
        },
        normalize_apply);
}

bool gadget_fusion(Graph& g) {
    return run_pass<GadgetFusion>(g, gadget_fusion_matches, gadget_fusion_check,
                                  gadget_fusion_apply);
}

bool remove_isolated(Graph& g, const ReduceOptions& options) {
    const size_t before = g.num_vertices();
    std::vector<bool> remove(g.id_bound(), false);
    std::vector<Vertex> isolated;
    std::vector<std::pair<Vertex, Vertex>> pairs;
    for (Vertex v : g.vertices()) {
        if (remove[v]) {
            continue;
        }
        if (g.degree(v) == 0) {
            isolated.push_back(v);
            remove[v] = true;
            continue;
        }
        if (g.degree(v) != 1) {
            continue;
        }
        const Vertex w = g.neighbors(v).front().vertex;
        if (remove[w] || g.degree(w) != 1) {
            continue;
        }
        pairs.emplace_back(v, w);
        remove[v] = true;
        remove[w] = true;
    }
    for (Vertex v : isolated) {
        const Phase& ph = g.phase(v);
        if (options.keep_weighted_isolated && weighted(ph.value, ph.params)) {
            remove[v] = false;
            continue;
        }
        g.append_scalar_term(ScalarTerm{.kind = ScalarTerm::Kind::Node,
                                        .alpha = ph.value,
                                        .beta = 0.0,
                                        .a = ph.params,
                                        .b = {}});
    }
    for (const auto& [u, v] : pairs) {
        const Phase& pu = g.phase(u);
        const Phase& pv = g.phase(v);
        const bool same_color = g.type(u) == g.type(v);
        const bool simple_edge = g.edge_type(u, v) == EdgeType::Simple;
        ScalarTerm term;
        if (same_color == simple_edge) {
            const Phase sum = pu.plus(pv);
            term = ScalarTerm{.kind = ScalarTerm::Kind::Node,
                              .alpha = sum.value,
                              .beta = 0.0,
                              .a = sum.params,
                              .b = {}};
            if (options.keep_weighted_isolated && weighted(sum.value, sum.params)) {
                remove[u] = remove[v] = false;
                continue;
            }
        } else {
            term = ScalarTerm{.kind = ScalarTerm::Kind::PhasePair,
                              .alpha = pu.value,
                              .beta = pv.value,
                              .a = pu.params,
                              .b = pv.params};
            // With Pauli concrete parts the factor has modulus 2 for every assignment.
            const bool pauli = pu.is_pauli() && pv.is_pauli();
            if (options.keep_weighted_isolated && !pauli &&
                !(pu.params.empty() && pv.params.empty())) {
                remove[u] = remove[v] = false;
                continue;
            }
        }
        g.append_scalar_term(std::move(term));
    }
    for (Vertex v = 0; v < remove.size(); ++v) {
        if (remove[v]) {
            g.remove_vertex(v);
        }
    }
    return g.num_vertices() < before;
}

}  // namespace rules

namespace {

struct Runner {
    Graph& g;
    const ReduceOptions& options;
    const ReduceTrace& trace;

    template <typename Step>
    bool leaf(std::string_view name, Step step) {
        // Reporting no change unwinds every saturating loop.
        if (g.over_work_limit()) {
            return false;
        }
        const bool changed = step(g);
        if (trace) {
            trace(name, changed, g);
        }
        return changed;
    }

    template <typename Step>
    bool sat_rule(std::string_view name, Step step) {
        return saturate(g, [&](Graph&) { return leaf(name, step); });
    }

    bool interior_clifford() {
        return saturate(g, [&](Graph&) {
            bool changed = sat_rule("id_remove", rules::id_remove);
            changed |= sat_rule("spider_fusion", rules::spider_fusion);
            changed |= sat_rule("pivot", rules::pivot);
            changed |= sat_rule("lcomp", rules::lcomp);
            return changed;
        });
    }

    bool run() {
        bool changed = leaf("to_gh", rules::to_gh);
        changed |= sat_rule("spider_fusion", rules::spider_fusion);
        changed |= sat_rule("copy", rules::copy);
        changed |= interior_clifford();
        changed |= sat_rule("pivot_gadget", rules::pivot_gadget);
        changed |= saturate(g, [&](Graph&) {
            bool round = interior_clifford();
            round |= sat_rule("gadget_normalize", rules::gadget_normalize);
            round |= leaf("gadget_fusion", rules::gadget_fusion);
            round |= leaf("copy", rules::copy);
            return round;
        });
        const bool removed =
            leaf("remove_isolated", [&](Graph& h) { return rules::remove_isolated(h, options); });
        return changed || removed;
    }
};

}  // namespace

bool full_reduce(Graph& g, const ReduceOptions& options, const ReduceTrace& trace) {
    g.set_work_limit(saturating_add_u64(g.work(), options.max_work));
    Runner runner{g, options, trace};
    while (runner.run()) {}
    const bool finished = !g.over_work_limit();
    g.set_work_limit(std::numeric_limits<uint64_t>::max());
    return finished;
}

}  // namespace clifft::zx
