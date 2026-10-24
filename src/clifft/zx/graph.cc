#include "clifft/zx/graph.h"

#include <algorithm>
#include <cassert>

namespace clifft::zx {

Vertex Graph::add_vertex(VertexType type, Phase phase) {
    const auto v = static_cast<Vertex>(slots_.size());
    slots_.push_back(Slot{.alive = true, .type = type, .phase = std::move(phase), .adj = {}});
    ++live_;
    return v;
}

void Graph::remove_vertex(Vertex v) {
    assert(contains(v));
    for (const Neighbor& n : slot(v).adj) {
        erase(slot(n.vertex).adj, v);
    }
    Slot& s = slot(v);
    s.alive = false;
    s.adj.clear();
    s.adj.shrink_to_fit();
    --live_;
}

std::vector<Vertex> Graph::vertices() const {
    std::vector<Vertex> out;
    out.reserve(live_);
    for (Vertex v = 0; v < slots_.size(); ++v) {
        if (slots_[v].alive) {
            out.push_back(v);
        }
    }
    return out;
}

std::optional<EdgeType> Graph::edge_type(Vertex u, Vertex v) const {
    for (const Neighbor& n : slot(u).adj) {
        if (n.vertex == v) {
            return n.type;
        }
    }
    return std::nullopt;
}

void Graph::upsert(std::vector<Neighbor>& adj, Vertex v, EdgeType type) {
    for (Neighbor& n : adj) {
        if (n.vertex == v) {
            n.type = type;
            return;
        }
    }
    adj.push_back(Neighbor{.vertex = v, .type = type});
}

bool Graph::erase(std::vector<Neighbor>& adj, Vertex v) {
    const auto it = std::ranges::find(adj, v, &Neighbor::vertex);
    if (it == adj.end()) {
        return false;
    }
    adj.erase(it);
    return true;
}

void Graph::add_edge(Vertex u, Vertex v, EdgeType type) {
    assert(u != v && contains(u) && contains(v));
    const std::optional<EdgeType> existing = edge_type(u, v);
    if (!existing) {
        slot(u).adj.push_back(Neighbor{.vertex = v, .type = type});
        slot(v).adj.push_back(Neighbor{.vertex = u, .type = type});
        return;
    }
    const bool same_color = slot(u).type == slot(v).type;
    const EdgeType fuse = same_color ? EdgeType::Simple : EdgeType::Hadamard;
    const EdgeType hopf = toggled(fuse);
    if (*existing == fuse && type == fuse) {
        return;
    }
    if (*existing == hopf && type == hopf) {
        erase(slot(u).adj, v);
        erase(slot(v).adj, u);
        return;
    }
    upsert(slot(u).adj, v, fuse);
    upsert(slot(v).adj, u, fuse);
    slot(u).phase = slot(u).phase.plus_pi();
}

void Graph::remove_edge(Vertex u, Vertex v) {
    [[maybe_unused]] const bool found = erase(slot(u).adj, v);
    assert(found);
    erase(slot(v).adj, u);
}

void Graph::toggle_edge(Vertex u, Vertex v) {
    const std::optional<EdgeType> existing = edge_type(u, v);
    assert(existing);
    upsert(slot(u).adj, v, toggled(*existing));
    upsert(slot(v).adj, u, toggled(*existing));
}

void Graph::add_hadamard_edges(std::span<const std::pair<Vertex, Vertex>> pairs) {
    work_ += pairs.size();
    for (const auto& [a, b] : pairs) {
        assert(a != b && "Hadamard self-edges are not toggled");
        if (slot(a).type != VertexType::Z || slot(b).type != VertexType::Z) {
            add_edge(a, b, EdgeType::Hadamard);
            continue;
        }
        const std::optional<EdgeType> existing = edge_type(a, b);
        if (!existing) {
            slot(a).adj.push_back(Neighbor{.vertex = b, .type = EdgeType::Hadamard});
            slot(b).adj.push_back(Neighbor{.vertex = a, .type = EdgeType::Hadamard});
        } else if (*existing == EdgeType::Hadamard) {
            erase(slot(a).adj, b);
            erase(slot(b).adj, a);
        } else {
            add_edge(a, b, EdgeType::Hadamard);
        }
    }
}

void Graph::set_neighbors_for_loading(Vertex v, std::vector<Neighbor> neighbors) {
    slot(v).adj = std::move(neighbors);
}

std::vector<Constraint> Graph::constraints() const {
    std::vector<Constraint> out;
    for (const ScalarTerm& term : scalar_terms_) {
        if (term.kind == ScalarTerm::Kind::Node && (term.alpha == 0.0 || term.alpha == 1.0)) {
            out.push_back(Constraint{.params = term.a, .rhs = term.alpha == 1.0});
        }
    }
    return out;
}

}  // namespace clifft::zx
