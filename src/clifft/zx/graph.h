#pragma once

// A closed, single-copy parametric ZX diagram.
//
// The semantics and the rewrite behavior follow zxparam 0.1.0's Rust core
// (https://pypi.org/project/zxparam/), restricted to classical-domain
// vertices, which are single-copy. Vertices are indexed by a dense ID that is never reused, in the
// style of quizx's vector graph. Neighbor order is insertion order: a new
// neighbor is appended, an edge-type update keeps its position, and a removal
// keeps the others in order. Rules take matches in vertex then neighbor order,
// so this order decides which rewrites happen and must match zxparam's.

#include "clifft/zx/phase.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace clifft::zx {

using Vertex = uint32_t;

enum class VertexType : uint8_t { Z, X };
enum class EdgeType : uint8_t { Simple, Hadamard };

[[nodiscard]] inline EdgeType toggled(EdgeType type) {
    return type == EdgeType::Simple ? EdgeType::Hadamard : EdgeType::Simple;
}

struct Neighbor {
    Vertex vertex;
    EdgeType type;

    [[nodiscard]] bool operator==(const Neighbor&) const = default;
};

// A scalar factor that a rewrite removed from the diagram.
//
// Only isolated spiders and isolated spider pairs contribute factors whose
// modulus can depend on the parameters; every other rule's factor is a power of
// sqrt(2) or a phase. A node is 1 + e^{i pi (alpha + parity(a))}. A phase pair is
// 1 + e^{i pi (alpha + parity(a))} + e^{i pi (beta + parity(b))}
//   - e^{i pi (alpha + beta + parity(a) + parity(b))}.
struct ScalarTerm {
    enum class Kind : uint8_t { Node, PhasePair };

    Kind kind = Kind::Node;
    double alpha = 0.0;
    double beta = 0.0;
    Params a;
    Params b;

    [[nodiscard]] bool operator==(const ScalarTerm&) const = default;
};

// An affine GF(2) constraint parity(params) = rhs on the sampled parameters.
struct Constraint {
    Params params;
    bool rhs = false;

    [[nodiscard]] bool operator==(const Constraint&) const = default;
};

class Graph {
  public:
    Graph() = default;

    // Vertices get IDs 0, 1, 2, ... in creation order.
    Vertex add_vertex(VertexType type, Phase phase = {});
    void remove_vertex(Vertex v);

    [[nodiscard]] bool contains(Vertex v) const { return v < slots_.size() && slots_[v].alive; }
    [[nodiscard]] size_t num_vertices() const { return live_; }
    // One past the largest ID ever handed out.
    [[nodiscard]] Vertex id_bound() const { return static_cast<Vertex>(slots_.size()); }
    // The live vertices in ascending ID order.
    [[nodiscard]] std::vector<Vertex> vertices() const;

    [[nodiscard]] VertexType type(Vertex v) const { return slot(v).type; }
    void set_type(Vertex v, VertexType type) { slot(v).type = type; }
    [[nodiscard]] const Phase& phase(Vertex v) const { return slot(v).phase; }
    void set_phase(Vertex v, Phase phase) { slot(v).phase = std::move(phase); }
    void add_to_phase(Vertex v, const Phase& phase) { slot(v).phase = slot(v).phase.plus(phase); }

    [[nodiscard]] std::span<const Neighbor> neighbors(Vertex v) const { return slot(v).adj; }
    [[nodiscard]] size_t degree(Vertex v) const { return slot(v).adj.size(); }
    [[nodiscard]] std::optional<EdgeType> edge_type(Vertex u, Vertex v) const;
    [[nodiscard]] bool has_edge(Vertex u, Vertex v) const { return edge_type(u, v).has_value(); }

    // Add a u-v edge, collapsing a parallel edge. Same-color spiders fuse under
    // plain edges and Hopf-cancel under Hadamard edges; opposite colors swap the
    // roles. A fusing and a cancelling edge leave the fusing one and add pi to u.
    // Precondition: u != v.
    void add_edge(Vertex u, Vertex v, EdgeType type);
    // Precondition: the edge exists.
    void remove_edge(Vertex u, Vertex v);
    // Precondition: the edge exists.
    void toggle_edge(Vertex u, Vertex v);
    // Add a Hadamard edge for each pair in order. Between two Z spiders this
    // toggles the edge directly, the hot loop of local complementation and
    // pivoting.
    void add_hadamard_edges(std::span<const std::pair<Vertex, Vertex>> pairs);

    // Hadamard-edge toggles from local complementation and pivoting, the
    // dominant cost of reduction: complementing a degree-d neighborhood
    // toggles d(d-1)/2 edges. full_reduce sets the limit for its own run.
    [[nodiscard]] uint64_t work() const { return work_; }
    [[nodiscard]] bool over_work_limit() const { return work_ > work_limit_; }
    void set_work_limit(uint64_t limit) { work_limit_ = limit; }

    // Replace v's neighbor list, keeping the given order on both endpoints'
    // sides as listed. For loading fixtures whose neighbor order is fixed;
    // callers list every edge on both endpoints with the same type.
    void set_neighbors_for_loading(Vertex v, std::vector<Neighbor> neighbors);

    [[nodiscard]] const std::vector<ScalarTerm>& scalar_terms() const { return scalar_terms_; }
    void append_scalar_term(ScalarTerm term) { scalar_terms_.push_back(std::move(term)); }
    // The constraints the removed factors impose: each node whose concrete part
    // is Pauli vanishes unless parity(a) equals that part.
    [[nodiscard]] std::vector<Constraint> constraints() const;

  private:
    struct Slot {
        bool alive = false;
        VertexType type = VertexType::Z;
        Phase phase;
        std::vector<Neighbor> adj;
    };

    [[nodiscard]] const Slot& slot(Vertex v) const { return slots_[v]; }
    [[nodiscard]] Slot& slot(Vertex v) { return slots_[v]; }

    // Set the u-side entry for v: in place if present, else appended.
    static void upsert(std::vector<Neighbor>& adj, Vertex v, EdgeType type);
    static bool erase(std::vector<Neighbor>& adj, Vertex v);

    std::vector<Slot> slots_;
    size_t live_ = 0;
    uint64_t work_ = 0;
    uint64_t work_limit_ = std::numeric_limits<uint64_t>::max();
    std::vector<ScalarTerm> scalar_terms_;
};

}  // namespace clifft::zx
