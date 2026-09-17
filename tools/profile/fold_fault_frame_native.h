// Standalone research descriptor evaluator, outside production dispatch.
#pragma once
#include "fold_blocks_native.h"

namespace fault_frame {
using fold_blocks::History;
using fold_blocks::Mask;
struct Frame {
    Mask flips = 0;
    unsigned phase = 0;
    std::array<unsigned, 128> linear{};
    std::array<uint64_t, 4> edges{};
    bool operator==(const Frame&) const = default;
};
struct Update {
    unsigned source, target;
};
struct Action {
    unsigned kind, q, r, s, e0, e1, e2, offset, count;
    int x, z;
};
struct Plan {
    unsigned width;
    std::span<const Action> actions;
    std::span<const Update> updates;
    std::array<uint64_t, 4> nondata_edges;
    Mask nondata_qubits;
};
inline unsigned edge(const Frame& f, unsigned e) noexcept {
    assert(e < 256);
    return (f.edges[e / 64] >> (e % 64)) & 1;
}
inline void toggle(Frame& f, unsigned e, unsigned bit) noexcept {
    assert(e < 256 && bit <= 1);
    f.edges[e / 64] ^= uint64_t(bit) << (e % 64);
}
inline bool data_only(const Frame& f, const Plan& p) noexcept {
    Mask support = f.flips;
    for (unsigned q = 0; q < p.width; ++q)
        support |= Mask(f.linear[q] != 0) << q;
    uint64_t edges = 0;
    for (unsigned j = 0; j < f.edges.size(); ++j)
        edges |= f.edges[j] & p.nondata_edges[j];
    return !(support & p.nondata_qubits) && !edges;
}
inline void evaluate(const Plan& p, const History& h, Frame& f) noexcept {
    for (const auto& a : p.actions) {
        unsigned x = (f.flips >> a.q) & 1;
        if (a.kind == 0) {
            unsigned u = fold_blocks::bit(h, a.x), v = fold_blocks::bit(h, a.z);
            f.phase += 4 * v * x;
            f.linear[a.q] += 4 * v;
            f.flips ^= Mask(u) << a.q;
        } else if (a.kind == 2 || a.kind == 3) {
            unsigned sign = a.kind == 2 ? 1 : -1;
            f.phase += sign * x;
            f.linear[a.q] -= 2 * sign * x;
        } else if (a.kind == 4) {
            f.flips ^= Mask(x) << a.r;
            f.linear[a.q] += f.linear[a.r] + 4 * edge(f, a.e0);
            toggle(f, a.e0, (f.linear[a.r] / 2) & 1);
            for (unsigned k = 0; k < a.count; ++k) {
                const auto& u = p.updates[a.offset + k];
                toggle(f, u.target, edge(f, u.source));
            }
        } else {
            assert(a.kind == 5);
            unsigned y = (f.flips >> a.r) & 1, z = (f.flips >> a.s) & 1;
            f.phase += 4 * x * y * z;
            f.linear[a.q] += 4 * y * z;
            f.linear[a.r] += 4 * x * z;
            f.linear[a.s] += 4 * x * y;
            toggle(f, a.e0, x);
            toggle(f, a.e1, y);
            toggle(f, a.e2, z);
        }
    }
    f.phase &= 7;
    for (unsigned q = 0; q < p.width; ++q)
        f.linear[q] &= 7;
}
}  // namespace fault_frame
