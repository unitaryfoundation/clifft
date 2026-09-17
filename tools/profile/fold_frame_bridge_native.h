// Standalone measurement bridge; all edge routing is compiled before execution.
#pragma once
#include "fold_fault_frame_native.h"

namespace fault_frame {
struct Route {
    unsigned edge, kind, q, r, output_edge;
};
struct Bridge {
    const Plan* frame;
    std::span<const unsigned> data, cats;
    std::span<const Route> routes;
    std::array<fold_blocks::Monomial, 2> ideal;
};

inline unsigned contract(const Bridge& bridge, const fold_blocks::Fold& fold, const Frame& f,
                         Mask records, std::array<fold_blocks::Term, 2>& out) noexcept {
    using namespace fold_blocks;
    Mask cat_flip = 0;
    unsigned cat_phase = 0;
    for (unsigned j = 0; j < bridge.cats.size(); ++j) {
        auto q = bridge.cats[j];
        cat_flip |= ((f.flips >> q) & 1) << j;
        cat_phase += f.linear[q];
    }
    unsigned count = 0;
    for (unsigned b = 0; b < 2; ++b) {
        Mask cat = (b ? (Mask(1) << fold.cats) - 1 : 0) ^ cat_flip;
        double weight = .5 * fold.decode[(records << fold.cats) | cat];
        if (!weight)
            continue;
        Monomial op;
        op.phase = f.phase + b * cat_phase;
        for (unsigned q = 0; q < bridge.data.size(); ++q) {
            auto physical = bridge.data[q];
            op.flips |= ((f.flips >> physical) & 1) << q;
            op.linear[q] = f.linear[physical];
        }
        for (const auto& route : bridge.routes) {
            unsigned value = edge(f, route.edge);
            if (route.kind == 0)
                op.edges ^= Mask(value) << route.output_edge;
            else if (route.kind == 1)
                op.linear[route.q] += 4 * b * value;
            else {
                assert(route.kind == 2);
                op.phase += 4 * b * value;
            }
        }
        out[count++] = {weight, compose(op, bridge.ideal[b], *fold.geometry)};
    }
    return count;
}

inline unsigned branches(const Bridge& bridge, const fold_blocks::Fold& fold, const History& h,
                         std::array<fold_blocks::Term, 2>& out) noexcept {
    using namespace fold_blocks;
    Mask flags = apply(fold.preparation.records, h);
    if (flags != 0 && flags != fold.equal_flag_mask)
        return 0;
    Mask x = apply(fold.preparation.x, h), z = apply(fold.preparation.z, h);
    Frame f;
    for (unsigned j = 0; j < bridge.cats.size(); ++j) {
        auto q = bridge.cats[j];
        f.flips |= ((x >> j) & 1) << q;
        f.linear[q] = 4 * ((z >> j) & 1);
    }
    evaluate(*bridge.frame, h, f);
    return contract(bridge, fold, f, apply(fold.records, h), out);
}

class Executor {
    fold_blocks::Executor base;
    std::span<const Bridge* const> bridges;

  public:
    explicit Executor(std::span<const Bridge* const> bridges) : bridges(bridges) {}
    std::array<double, 4> evaluate(const fold_blocks::Protocol& plan, const History& h) noexcept {
        assert(bridges.size() == plan.stages.size());
        return base.evaluate_with(
            plan, h,
            [this](size_t index, const fold_blocks::Fold& fold, const History& history,
                   std::array<fold_blocks::Term, 2>& out) {
                return bridges[index] ? branches(*bridges[index], fold, history, out)
                                      : base.branches(fold, history, out);
            });
    }
};
}  // namespace fault_frame
