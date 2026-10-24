#include "clifft/zx/pbc.h"

#include "clifft/tableau/pauli_string.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <numbers>
#include <numeric>
#include <utility>

namespace clifft::zx {
namespace {

using Kind = DiagramParam::Kind;

// ---------------------------------------------------------------------------
// Grouping: components of the reduced graph, constraints and records that
// share an outcome parameter belong together. Noise bits do not join groups:
// they are sampled before anything else.

class UnionFind {
  public:
    explicit UnionFind(size_t n) : parent_(n) { std::iota(parent_.begin(), parent_.end(), 0); }
    size_t find(size_t i) {
        while (parent_[i] != i) {
            parent_[i] = parent_[parent_[i]];
            i = parent_[i];
        }
        return i;
    }
    void unite(size_t a, size_t b) { parent_[find(a)] = find(b); }

  private:
    std::vector<size_t> parent_;
};

std::vector<std::vector<Vertex>> components(const Graph& g) {
    std::vector<bool> seen(g.id_bound(), false);
    std::vector<std::vector<Vertex>> out;
    for (Vertex start : g.vertices()) {
        if (seen[start]) {
            continue;
        }
        std::vector<Vertex> members = {start};
        seen[start] = true;
        for (size_t k = 0; k < members.size(); ++k) {
            for (const Neighbor& n : g.neighbors(members[k])) {
                if (!seen[n.vertex]) {
                    seen[n.vertex] = true;
                    members.push_back(n.vertex);
                }
            }
        }
        std::ranges::sort(members);
        out.push_back(std::move(members));
    }
    return out;
}

struct Group {
    std::vector<size_t> components;
    std::vector<size_t> constraints;
    std::vector<size_t> records;  // indices into the record list
};

// ---------------------------------------------------------------------------
// The reading of one group: spider outcomes, postselections and outputs.

struct Reading {
    size_t n = 0;                 // spiders
    std::vector<Bits> neighbors;  // over spiders
    std::vector<char> letter;     // 'X' or 'Y'
    Bits negative;                // spiders whose measured Pauli is negated
    std::vector<double> theta;    // rotation of each spider, where rotated
    std::vector<bool> has_theta;
    std::vector<std::pair<Bits, Bits>> posts;  // (spider mask, rhs)
    struct Output {
        Bits spiders;
        Bits coins;
        Bits rhs;
    };
    std::vector<Output> outputs;
    size_t num_coins = 0;
    size_t num_f = 0;

    [[nodiscard]] bool rotated(size_t v) const { return has_theta[v]; }

    [[nodiscard]] PauliString measured(const Bits& mask) const {
        PauliString p(static_cast<uint32_t>(n));
        for (size_t v : mask.ones()) {
            p.set_pauli(static_cast<uint32_t>(v), true, letter[v] == 'Y');
        }
        p.set_sign((mask & negative).popcount() % 2 == 1);
        return p;
    }

    // K_T, the product of K_u = X_u Z_N(u) over the spiders in t.
    [[nodiscard]] PauliString stabilizer(const Bits& t) const {
        PauliString product(static_cast<uint32_t>(n));
        for (size_t u : t.ones()) {
            PauliString k(static_cast<uint32_t>(n));
            k.set_pauli(static_cast<uint32_t>(u), true, false);
            for (size_t w : neighbors[u].ones()) {
                k.set_pauli(static_cast<uint32_t>(w), false, true);
            }
            k.set_sign(false);
            product.right_multiply(k.view());
        }
        return product;
    }
};

// Reduced row echelon form of affine equations over GF(2), with variables in
// a priority order: the highest bit is the highest priority.
struct Equation {
    Bits mask;
    Bits rhs;
};

std::optional<std::vector<Equation>> reduced_echelon(std::vector<Equation> equations,
                                                     size_t num_vars) {
    std::vector<std::optional<Equation>> rows(num_vars);
    for (Equation& e : equations) {
        while (const auto lead = e.mask.highest()) {
            if (!rows[*lead]) {
                break;
            }
            e.mask ^= rows[*lead]->mask;
            e.rhs ^= rows[*lead]->rhs;
        }
        if (const auto lead = e.mask.highest()) {
            rows[*lead] = std::move(e);
        } else if (e.rhs.any()) {
            return std::nullopt;  // 0 = 1, or 0 = a parity of noise bits
        }
    }
    // Back-substitute from the lowest lead up, so no row keeps another's lead.
    Bits below(num_vars);
    for (size_t lead = 0; lead < num_vars; ++lead) {
        if (!rows[lead]) {
            continue;
        }
        for (size_t b : (rows[lead]->mask & below).ones()) {
            rows[lead]->mask ^= rows[b]->mask;
            rows[lead]->rhs ^= rows[b]->rhs;
        }
        below.set(lead);
    }
    std::vector<Equation> out;
    for (size_t lead = num_vars; lead-- > 0;) {
        if (rows[lead]) {
            out.push_back(std::move(*rows[lead]));
        }
    }
    return out;
}

// Postselections each K_u and Z_v anticommutes with, as masks over posts.
// K_u = X_u Z_N(u) anticommutes with m_w when w is a neighbour of u, or when
// w = u and m_w is a Y; Z_v anticommutes with m_v.
std::pair<std::vector<Bits>, std::vector<Bits>> patterns(const Reading& r) {
    std::vector<Bits> k_pattern(r.n, Bits(r.posts.size()));
    std::vector<Bits> z_pattern(r.n, Bits(r.posts.size()));
    for (size_t k = 0; k < r.posts.size(); ++k) {
        for (size_t w : r.posts[k].first.ones()) {
            z_pattern[w].set(k);
            for (size_t u : r.neighbors[w].ones()) {
                k_pattern[u].flip(k);
            }
            if (r.letter[w] == 'Y') {
                k_pattern[w].flip(k);
            }
        }
    }
    return {std::move(k_pattern), std::move(z_pattern)};
}

// (v, T) per rotated spider in rotation order, with Z_v K_T commuting with
// every postselection and T holding no spider rotated at or before v. Built
// from the back: a spider placed just before the placed ones may use them and
// the unrotated spiders, a set that only grows, so the greedy never
// backtracks and fails only if no order exists.
std::optional<std::vector<std::pair<size_t, Bits>>> rotation_axes(
    const Reading& r, const std::vector<Bits>& k_pattern, const std::vector<Bits>& z_pattern) {
    std::vector<size_t> unplaced;
    for (size_t v = r.n; v-- > 0;) {
        if (r.rotated(v)) {
            unplaced.push_back(v);
        }
    }
    Bits allowed(r.n);
    EchelonBasis basis(r.posts.size(), r.n);
    for (size_t u = 0; u < r.n; ++u) {
        if (!r.rotated(u)) {
            allowed.set(u);
            basis.insert(k_pattern[u], Bits::single(r.n, u));
        }
    }
    auto witness = [&](size_t v) -> std::optional<Bits> {
        if (!z_pattern[v].any()) {
            return Bits(r.n);
        }
        // Single neighbors first: typically a phase gadget's root.
        for (size_t u : (r.neighbors[v] & allowed).ones()) {
            if (k_pattern[u] == z_pattern[v]) {
                return Bits::single(r.n, u);
            }
        }
        return basis.solve(z_pattern[v]);
    };
    std::vector<std::pair<size_t, Bits>> placed;
    while (!unplaced.empty()) {
        bool found = false;
        for (size_t k = 0; k < unplaced.size(); ++k) {
            if (auto t = witness(unplaced[k])) {
                const size_t v = unplaced[k];
                unplaced.erase(unplaced.begin() + static_cast<std::ptrdiff_t>(k));
                placed.emplace_back(v, std::move(*t));
                basis.insert(k_pattern[v], Bits::single(r.n, v));
                allowed.set(v);
                found = true;
                break;
            }
        }
        if (!found) {
            return std::nullopt;
        }
    }
    std::ranges::reverse(placed);
    return placed;
}

// A Clifford frame mapping each generator g_i to Z_i, through destabilizers
// d_i (d_i to X_i) with <d_i, g_j> = delta_ij and commuting with each other.
class Frame {
  public:
    explicit Frame(std::vector<PauliString> generators) : g_(std::move(generators)) {
        const size_t n = g_.size();
        // Solve <d, g_j> = delta_ij: <d, g> is d . swap(g) over (x | z).
        struct Row {
            Bits a;  // swap(g_j) as a 2n vector
            Bits m;  // the row operations applied, over the n equations
        };
        std::vector<Row> rows;
        for (size_t j = 0; j < n; ++j) {
            Bits a(2 * n);
            for (size_t q = 0; q < n; ++q) {
                a.set(q, g_[j].z().bit_get(static_cast<uint32_t>(q)));
                a.set(n + q, g_[j].x().bit_get(static_cast<uint32_t>(q)));
            }
            rows.push_back(Row{std::move(a), Bits::single(n, j)});
        }
        // Reduced row echelon form with pivots in ascending columns.
        std::vector<size_t> pivot(n, 0);
        size_t rank = 0;
        for (size_t col = 0; col < 2 * n && rank < n; ++col) {
            size_t sel = rank;
            while (sel < n && !rows[sel].a.test(col)) {
                ++sel;
            }
            if (sel == n) {
                continue;
            }
            std::swap(rows[sel], rows[rank]);
            for (size_t i = 0; i < n; ++i) {
                if (i != rank && rows[i].a.test(col)) {
                    rows[i].a ^= rows[rank].a;
                    rows[i].m ^= rows[rank].m;
                }
            }
            pivot[rank++] = col;
        }
        valid_ = rank == n;
        if (!valid_) {
            return;
        }
        // M A has identity pivot columns, so d_i = sum_j M[j][i] e_{pivot j}.
        std::vector<Bits> d(n, Bits(2 * n));
        for (size_t j = 0; j < n; ++j) {
            for (size_t i : rows[j].m.ones()) {
                d[i].flip(pivot[j]);
            }
        }
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < j; ++i) {
                if (symplectic(d[j], d[i], n)) {
                    // g_i flips <d_j, d_i> and nothing else d_j pairs with.
                    for (size_t q = 0; q < n; ++q) {
                        if (g_[i].x().bit_get(static_cast<uint32_t>(q))) {
                            d[j].flip(q);
                        }
                        if (g_[i].z().bit_get(static_cast<uint32_t>(q))) {
                            d[j].flip(n + q);
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            PauliString p(static_cast<uint32_t>(n));
            for (size_t q = 0; q < n; ++q) {
                p.set_pauli(static_cast<uint32_t>(q), d[i].test(q), d[i].test(n + q));
            }
            p.set_sign(false);
            d_.push_back(std::move(p));
        }
    }

    [[nodiscard]] bool valid() const { return valid_; }

    struct Image {
        Bits x;
        Bits z;
        bool sign;
    };

    // Q in the frame: X_i where Q anticommutes with g_i, Z_i where with d_i,
    // and the sign from writing Q as a product of the d_i and g_i.
    [[nodiscard]] Image apply(const PauliString& q) const {
        const size_t n = g_.size();
        Image out{Bits(n), Bits(n), false};
        PauliString product(static_cast<uint32_t>(n));
        PauliString image(static_cast<uint32_t>(n));
        for (size_t i = 0; i < n; ++i) {
            const bool a = !q.view().commutes(g_[i].view());
            const bool b = !q.view().commutes(d_[i].view());
            if (a) {
                product.right_multiply(d_[i].view());
                PauliString xi(static_cast<uint32_t>(n));
                xi.set_pauli(static_cast<uint32_t>(i), true, false);
                image.right_multiply(xi.view());
                out.x.set(i);
            }
            if (b) {
                product.right_multiply(g_[i].view());
                PauliString zi(static_cast<uint32_t>(n));
                zi.set_pauli(static_cast<uint32_t>(i), false, true);
                image.right_multiply(zi.view());
                out.z.set(i);
            }
        }
        assert(product.x() == q.x() && product.z() == q.z());
        image.add_phase(static_cast<uint8_t>((q.phase() - product.phase()) & 3U));
        assert(image.is_hermitian());
        out.sign = image.sign();
        return out;
    }

  private:
    static bool symplectic(const Bits& a, const Bits& b, size_t n) {
        bool odd = false;
        for (size_t q = 0; q < n; ++q) {
            odd ^= (a.test(q) && b.test(n + q)) != (a.test(n + q) && b.test(q));
        }
        return odd;
    }

    std::vector<PauliString> g_;
    std::vector<PauliString> d_;
    bool valid_ = false;
};

// Prepend dropped measurements of Paulis that commute with every op. They
// change no outcome, and their Paulis stay in every trajectory's stabilizer
// group, which can only lower the active width. Chosen greedily, pairwise
// commuting, with X parts independent of those already chosen (a Pauli whose
// X part is in their span measures deterministically).
void prepend_commutant(PbcGroup& group) {
    const size_t lines = group.num_lines;
    const size_t num_ops = group.ops.size();
    // X_l anticommutes with ops that have a Z on l, and Z_l with those that
    // have an X, so the Paulis commuting with every op are zero combinations
    // of these columns.
    EchelonBasis columns(num_ops, 2 * lines);
    std::vector<Bits> commutant;
    for (size_t bit = 0; bit < 2 * lines; ++bit) {
        Bits column(num_ops);
        for (size_t j = 0; j < num_ops; ++j) {
            const PbcOp& op = group.ops[j];
            column.set(j, bit < lines ? op.z.test(bit) : op.x.test(bit - lines));
        }
        if (auto zero = columns.insert(std::move(column), Bits::single(2 * lines, bit))) {
            commutant.push_back(std::move(*zero));
        }
    }
    std::ranges::stable_sort(commutant, {}, [](const Bits& q) { return q.popcount(); });
    auto x_part = [&](const Bits& q) {
        Bits x(lines);
        for (size_t l = 0; l < lines; ++l) {
            x.set(l, q.test(l));
        }
        return x;
    };
    auto z_part = [&](const Bits& q) {
        Bits z(lines);
        for (size_t l = 0; l < lines; ++l) {
            z.set(l, q.test(lines + l));
        }
        return z;
    };
    std::vector<Bits> chosen;
    EchelonBasis x_parts(lines, 1);
    for (const Bits& q : commutant) {
        const Bits qx = x_part(q), qz = z_part(q);
        bool clash = false;
        for (const Bits& c : chosen) {
            clash |= qx.parity_with(z_part(c)) != qz.parity_with(x_part(c));
        }
        if (clash) {
            continue;
        }
        if (!x_parts.insert(qx, Bits(1))) {
            chosen.push_back(q);
        }
    }
    std::vector<PbcOp> ops;
    const size_t sign_size = group.noise.size() + 1;
    for (const Bits& q : chosen) {
        ops.push_back(PbcOp{.kind = PbcOp::Kind::Measurement,
                            .x = x_part(q),
                            .z = z_part(q),
                            .theta = 0.0,
                            .sign = Bits(sign_size),
                            .record = -1});
    }
    ops.insert(ops.end(), std::make_move_iterator(group.ops.begin()),
               std::make_move_iterator(group.ops.end()));
    group.ops = std::move(ops);
}

}  // namespace

std::optional<PbcProgram> read_pbc(const SingleCopyDiagram& diagram, const Graph& reduced,
                                   std::string* reason) {
    auto reject = [&](std::string why) -> std::optional<PbcProgram> {
        if (reason) {
            *reason = std::move(why);
        }
        return std::nullopt;
    };
    const std::vector<DiagramParam>& params = diagram.params;
    auto is_outcome = [&](uint32_t p) { return params[p].kind != Kind::Noise; };

    for (const ScalarTerm& t : reduced.scalar_terms()) {
        const bool pauli_a = t.alpha == 0.0 || t.alpha == 1.0;
        const bool pauli_b = t.beta == 0.0 || t.beta == 1.0;
        const bool constant = t.kind == ScalarTerm::Kind::Node
                                  ? pauli_a || t.a.empty()
                                  : (pauli_a && pauli_b) || (t.a.empty() && t.b.empty());
        if (!constant) {
            return reject("a dropped scalar factor weights trajectories");
        }
    }
    const std::vector<Constraint> constraints = reduced.constraints();
    const std::vector<std::vector<Vertex>> comps = components(reduced);
    std::vector<uint32_t> records;  // record parameters, in record order
    for (uint32_t p = 0; p < params.size(); ++p) {
        if (params[p].kind == Kind::Record) {
            records.push_back(p);
        }
    }

    // Union components, constraints and records through shared outcomes.
    const size_t c0 = comps.size(), r0 = c0 + constraints.size();
    UnionFind uf(r0 + records.size());
    std::map<uint32_t, size_t> owner;
    auto touch = [&](size_t node, uint32_t p) {
        if (!is_outcome(p)) {
            return;
        }
        const auto [it, inserted] = owner.emplace(p, node);
        if (!inserted) {
            uf.unite(node, it->second);
        }
    };
    for (size_t c = 0; c < comps.size(); ++c) {
        for (Vertex v : comps[c]) {
            for (uint32_t p : reduced.phase(v).params.ids()) {
                touch(c, p);
            }
        }
    }
    for (size_t k = 0; k < constraints.size(); ++k) {
        for (uint32_t p : constraints[k].params.ids()) {
            touch(c0 + k, p);
        }
    }
    for (size_t j = 0; j < records.size(); ++j) {
        touch(r0 + j, records[j]);
    }
    std::vector<Group> groups;
    std::map<size_t, size_t> group_of_root;
    for (size_t i = 0; i < r0 + records.size(); ++i) {
        const size_t root = uf.find(i);
        const auto [it, inserted] = group_of_root.emplace(root, groups.size());
        if (inserted) {
            groups.emplace_back();
        }
        Group& g = groups[it->second];
        if (i < c0) {
            g.components.push_back(i);
        } else if (i < r0) {
            g.constraints.push_back(i - c0);
        } else {
            g.records.push_back(i - r0);
        }
    }

    PbcProgram program;
    for (const Group& grp : groups) {
        if (grp.records.empty()) {
            continue;
        }
        // Spiders, outcome variables and noise bits of the group.
        std::vector<Vertex> spiders;
        for (size_t c : grp.components) {
            spiders.insert(spiders.end(), comps[c].begin(), comps[c].end());
        }
        std::map<uint32_t, size_t> outcome_var, noise_pos;
        auto note = [&](uint32_t p) { (is_outcome(p) ? outcome_var : noise_pos).emplace(p, 0); };
        for (Vertex v : spiders) {
            if (reduced.type(v) != VertexType::Z) {
                return reject("the reduced diagram has an X spider");
            }
            for (uint32_t p : reduced.phase(v).params.ids()) {
                note(p);
            }
        }
        for (size_t k : grp.constraints) {
            for (uint32_t p : constraints[k].params.ids()) {
                note(p);
            }
        }
        for (size_t j : grp.records) {
            note(records[j]);
        }
        size_t index = 0;
        for (auto& [p, i] : outcome_var) {
            i = index++;
        }
        index = 0;
        PbcGroup out;
        for (auto& [p, i] : noise_pos) {
            i = index++;
            out.noise.push_back(p);
        }

        Reading r;
        r.n = spiders.size();
        r.num_f = noise_pos.size();
        const size_t m = outcome_var.size(), o = grp.records.size();
        const size_t rhs_size = 1 + r.num_f;
        std::map<Vertex, size_t> spider_index;
        for (size_t i = 0; i < r.n; ++i) {
            spider_index.emplace(spiders[i], i);
        }
        r.neighbors.assign(r.n, Bits(r.n));
        r.letter.assign(r.n, 'X');
        r.negative = Bits(r.n);
        r.theta.assign(r.n, 0.0);
        r.has_theta.assign(r.n, false);

        // Variables in priority order, highest bit first: outcome variables
        // (eliminated), then outputs in record order, then spiders.
        const size_t num_vars = m + o + r.n;
        auto outcome_bit = [&](uint32_t p) { return num_vars - 1 - outcome_var.at(p); };
        auto output_bit = [&](size_t j) { return num_vars - 1 - m - j; };
        auto spider_bit = [&](size_t i) { return num_vars - 1 - m - o - i; };
        auto noise_rhs = [&](const Params& ps, bool constant) {
            Bits rhs(rhs_size);
            rhs.set(0, constant);
            for (uint32_t p : ps.ids()) {
                if (!is_outcome(p)) {
                    rhs.flip(1 + noise_pos.at(p));
                }
            }
            return rhs;
        };
        std::vector<Equation> equations;
        for (size_t i = 0; i < r.n; ++i) {
            const Vertex v = spiders[i];
            const Phase& ph = reduced.phase(v);
            // pi/2 * q + 2 theta: rotate by exp(-i theta Z), then measure S^-q X S^q.
            const double quarters = std::nearbyint(2.0 * ph.value);
            const double rest = ph.value - quarters / 2.0;
            if (rest != 0.0) {
                r.theta[i] = std::numbers::pi * rest / 2.0;
                r.has_theta[i] = true;
            }
            const auto q = static_cast<int64_t>(quarters);
            r.letter[i] = q % 2 ? 'Y' : 'X';
            r.negative.set(i, (q % 4 == 1) || (q % 4 == 2));
            Equation e{Bits(num_vars), noise_rhs(ph.params, false)};
            e.mask.set(spider_bit(i));
            for (uint32_t p : ph.params.ids()) {
                if (is_outcome(p)) {
                    e.mask.flip(outcome_bit(p));
                }
            }
            equations.push_back(std::move(e));
            for (const Neighbor& nb : reduced.neighbors(v)) {
                if (nb.type != EdgeType::Hadamard) {
                    return reject("the reduced diagram has a plain edge between spiders");
                }
                r.neighbors[i].set(spider_index.at(nb.vertex));
            }
        }
        for (size_t k : grp.constraints) {
            Equation e{Bits(num_vars), noise_rhs(constraints[k].params, constraints[k].rhs)};
            for (uint32_t p : constraints[k].params.ids()) {
                if (is_outcome(p)) {
                    e.mask.flip(outcome_bit(p));
                }
            }
            equations.push_back(std::move(e));
        }
        for (size_t j = 0; j < o; ++j) {
            Equation e{Bits(num_vars), Bits(rhs_size)};
            e.mask.set(output_bit(j));
            e.mask.flip(outcome_bit(records[grp.records[j]]));
            equations.push_back(std::move(e));
        }
        auto rows = reduced_echelon(std::move(equations), num_vars);
        if (!rows) {
            return reject("the classical part of the diagram is inconsistent");
        }

        // Rows led by an outcome variable only fix that variable; the rest
        // are postselections (led by a spider) and outputs.
        std::vector<Equation> output_rows(o);
        Bits is_pivot_output(o);
        for (Equation& e : *rows) {
            // Bits [0, n) are spiders, [n, n + o) outputs, the rest outcome variables.
            const size_t lead = *e.mask.highest();
            if (lead >= r.n + o) {
                continue;
            }
            if (lead >= r.n) {
                const size_t j = r.n + o - 1 - lead;
                is_pivot_output.set(j);
                output_rows[j] = std::move(e);
                continue;
            }
            Bits mask(r.n);
            for (size_t i = 0; i < r.n; ++i) {
                mask.set(i, e.mask.test(spider_bit(i)));
            }
            r.posts.emplace_back(std::move(mask), std::move(e.rhs));
        }
        std::vector<size_t> coin_of(o, 0);
        for (size_t j = 0; j < o; ++j) {
            if (!is_pivot_output.test(j)) {
                coin_of[j] = r.num_coins++;
            }
        }
        for (size_t j = 0; j < o; ++j) {
            Reading::Output out_j{Bits(r.n), Bits(r.num_coins), Bits(rhs_size)};
            if (!is_pivot_output.test(j)) {
                out_j.coins.set(coin_of[j]);
            } else {
                const Equation& e = output_rows[j];
                for (size_t i = 0; i < r.n; ++i) {
                    out_j.spiders.set(i, e.mask.test(spider_bit(i)));
                }
                for (size_t k = 0; k < o; ++k) {
                    if (k != j && e.mask.test(output_bit(k))) {
                        out_j.coins.set(coin_of[k]);
                    }
                }
                out_j.rhs = e.rhs;
            }
            r.outputs.push_back(std::move(out_j));
        }

        // Fold the postselections into the rotation axes.
        const auto [k_pattern, z_pattern] = patterns(r);
        const auto axes = rotation_axes(r, k_pattern, z_pattern);
        if (!axes) {
            return reject("no rotation order makes every postselection commute");
        }

        // The frame: a maximal independent set of the postselections on lines
        // 0, 1, ..., then graph stabilizers commuting with every one of them.
        EchelonBasis independent(r.posts.size(), r.n);
        std::vector<Bits> kernel;
        for (size_t u = 0; u < r.n; ++u) {
            if (auto zero = independent.insert(k_pattern[u], Bits::single(r.n, u))) {
                kernel.push_back(std::move(*zero));
            }
        }
        const std::vector<size_t> post_lines = independent.leads();
        std::vector<PauliString> posts;
        for (const auto& [mask, rhs] : r.posts) {
            PauliString p = r.measured(mask);
            if (rhs.test(0)) {
                p.negate();
            }
            posts.push_back(std::move(p));
        }
        std::vector<PauliString> generators;
        for (size_t k : post_lines) {
            generators.push_back(posts[k]);
        }
        for (const Bits& t : kernel) {
            generators.push_back(r.stabilizer(t));
        }
        if (generators.size() != r.n) {
            return reject("the postselected graph state is not fully stabilized");
        }
        const Frame frame(std::move(generators));
        if (r.n > 0 && !frame.valid()) {
            return reject("the postselected graph state is not fully stabilized");
        }
        // Line i holds post post_lines[i], starting in |target_i>.
        auto noise_part = [&](const Bits& rhs) {
            Bits f(r.num_f);
            for (size_t k = 0; k < r.num_f; ++k) {
                f.set(k, rhs.test(1 + k));
            }
            return f;
        };
        std::vector<Bits> targets;
        for (size_t k : post_lines) {
            targets.push_back(noise_part(r.posts[k].second));
        }
        // Every other postselection is a product of the independent ones and
        // must hold for every noise pattern.
        for (size_t k = 0; k < r.posts.size(); ++k) {
            if (independent.has_lead(k)) {
                continue;
            }
            const Frame::Image img = frame.apply(posts[k]);
            Bits target(r.num_f);
            for (size_t i = 0; i < post_lines.size(); ++i) {
                if (img.z.test(i)) {
                    target ^= targets[i];
                }
            }
            if (img.sign || img.x.any() || !(target == noise_part(r.posts[k].second))) {
                return reject("the diagram postselects an event of probability zero");
            }
        }

        // The operations' Paulis in the frame.
        struct Source {
            PauliString pauli;
            Bits coins;
            Bits rhs;
            double theta;
            int64_t record;
        };
        std::vector<Source> sources;
        for (const auto& [v, t] : *axes) {
            PauliString axis(static_cast<uint32_t>(r.n));
            axis.set_pauli(static_cast<uint32_t>(v), false, true);
            axis.right_multiply(r.stabilizer(t).view());
            sources.push_back(
                Source{std::move(axis), Bits(r.num_coins), Bits(rhs_size), r.theta[v], -1});
        }
        for (size_t j = 0; j < o; ++j) {
            const Reading::Output& oj = r.outputs[j];
            sources.push_back(Source{r.measured(oj.spiders), oj.coins, oj.rhs, 0.0,
                                     static_cast<int64_t>(params[records[grp.records[j]]].index)});
        }
        std::vector<Frame::Image> images;
        Bits flipped(r.n);
        for (const Source& s : sources) {
            images.push_back(r.n > 0 ? frame.apply(s.pauli)
                                     : Frame::Image{Bits(0), Bits(0), s.pauli.sign()});
            flipped |= images.back().x;
        }
        const std::vector<size_t> lines = flipped.ones();
        out.num_lines = static_cast<uint32_t>(lines.size() + r.num_coins);
        for (size_t j = 0; j < sources.size(); ++j) {
            const Source& s = sources[j];
            const Frame::Image& img = images[j];
            PbcOp op;
            op.kind = j < axes->size() ? PbcOp::Kind::Rotation : PbcOp::Kind::Measurement;
            op.x = Bits(out.num_lines);
            op.z = Bits(out.num_lines);
            for (size_t l = 0; l < lines.size(); ++l) {
                op.x.set(l, img.x.test(lines[l]));
                op.z.set(l, img.z.test(lines[l]));
            }
            for (size_t c = 0; c < r.num_coins; ++c) {
                op.x.set(lines.size() + c, s.coins.test(c));
            }
            op.theta = s.theta;
            op.sign = s.rhs;
            if (img.sign) {
                op.sign.flip(0);
            }
            // A line holding a postselection starts in |target>, so a Z on
            // it flips the sign with that target.
            for (size_t i = 0; i < post_lines.size(); ++i) {
                if (img.z.test(i)) {
                    for (size_t k = 0; k < r.num_f; ++k) {
                        if (targets[i].test(k)) {
                            op.sign.flip(1 + k);
                        }
                    }
                }
            }
            op.record = s.record;
            out.ops.push_back(std::move(op));
        }
        prepend_commutant(out);
        program.groups.push_back(std::move(out));
    }
    return program;
}

}  // namespace clifft::zx
