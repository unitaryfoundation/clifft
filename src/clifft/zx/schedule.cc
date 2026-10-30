#include "clifft/zx/schedule.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace clifft::zx {
namespace {

// A Pauli on L lines as one 2L vector: x bits, then z bits.
Bits joined(const PbcOp& op) {
    const size_t lines = op.x.size();
    Bits out(2 * lines);
    for (size_t l = 0; l < lines; ++l) {
        out.set(l, op.x.test(l));
        out.set(lines + l, op.z.test(l));
    }
    return out;
}

// The same Pauli with x and z swapped, so that a dot product with another
// joined Pauli is their symplectic product.
Bits swapped(const Bits& p) {
    const size_t lines = p.size() / 2;
    Bits out(p.size());
    for (size_t l = 0; l < lines; ++l) {
        out.set(l, p.test(lines + l));
        out.set(lines + l, p.test(l));
    }
    return out;
}

// Generators, up to sign, of the Pauli group stabilizing every trajectory.
class StabilizerGroup {
  public:
    explicit StabilizerGroup(size_t lines) : lines_(lines) {
        for (size_t l = 0; l < lines; ++l) {
            rows_.push_back(Bits::single(2 * lines, lines + l));  // Z_l on |0...0>
        }
    }

    [[nodiscard]] uint32_t width() const { return static_cast<uint32_t>(lines_ - rows_.size()); }

    [[nodiscard]] bool anticommutes_with_any(const Bits& swapped_op) const {
        return std::ranges::any_of(rows_, [&](const Bits& r) { return r.parity_with(swapped_op); });
    }

    // Leave one generator anticommuting with the op and return it, if any.
    std::optional<size_t> eliminate(const Bits& swapped_op) {
        std::optional<size_t> first;
        for (size_t i = 0; i < rows_.size(); ++i) {
            if (!rows_[i].parity_with(swapped_op)) {
                continue;
            }
            if (!first) {
                first = i;
            } else {
                rows_[i] ^= rows_[*first];
            }
        }
        return first;
    }

    void rotate(const Bits& swapped_op) {
        if (const auto first = eliminate(swapped_op)) {
            rows_.erase(rows_.begin() + static_cast<std::ptrdiff_t>(*first));
        }
    }

    void measure(const Bits& op, const Bits& swapped_op) {
        if (const auto first = eliminate(swapped_op)) {
            rows_[*first] = op;
        } else if (!contains(op)) {
            rows_.push_back(op);
        }
    }

  private:
    [[nodiscard]] bool contains(const Bits& op) const {
        EchelonBasis basis(2 * lines_, 1);
        for (const Bits& r : rows_) {
            basis.insert(r, Bits(1));
        }
        return basis.solve(op).has_value();
    }

    size_t lines_;
    std::vector<Bits> rows_;
};

struct Prepared {
    std::vector<Bits> op;
    std::vector<Bits> swapped_op;
    std::vector<bool> is_measurement;
};

Prepared prepare(const PbcGroup& group) {
    Prepared p;
    for (const PbcOp& op : group.ops) {
        p.op.push_back(joined(op));
        p.swapped_op.push_back(swapped(p.op.back()));
        p.is_measurement.push_back(op.kind == PbcOp::Kind::Measurement);
    }
    return p;
}

void advance(StabilizerGroup& s, const Prepared& p, size_t j) {
    if (p.is_measurement[j]) {
        s.measure(p.op[j], p.swapped_op[j]);
    } else {
        s.rotate(p.swapped_op[j]);
    }
}

}  // namespace

uint32_t peak_width(const PbcGroup& group, const std::vector<uint32_t>& order) {
    const Prepared p = prepare(group);
    StabilizerGroup s(group.num_lines);
    uint32_t peak = 0;
    for (uint32_t j : order) {
        advance(s, p, j);
        peak = std::max(peak, s.width());
    }
    return peak;
}

Schedule greedy_schedule(const PbcGroup& group) {
    const Prepared p = prepare(group);
    const size_t n = group.ops.size();
    // successors[i] holds the later ops that anticommute with op i.
    std::vector<Bits> successors(n, Bits(n));
    std::vector<size_t> waiting(n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (p.op[i].parity_with(p.swapped_op[j])) {
                successors[i].set(j);
                ++waiting[j];
            }
        }
    }
    // ancestors[j]: every op that must run before op j, transitively.
    std::vector<Bits> ancestors(n, Bits(n));
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < j; ++i) {
            if (successors[i].test(j)) {
                ancestors[j] |= ancestors[i];
                ancestors[j].set(i);
            }
        }
    }

    Bits done(n);
    StabilizerGroup s(group.num_lines);
    Schedule out;
    auto run = [&](size_t j) {
        advance(s, p, j);
        done.set(j);
        out.order.push_back(static_cast<uint32_t>(j));
        out.width = std::max(out.width, s.width());
        for (size_t k : successors[j].ones()) {
            --waiting[k];
        }
    };
    // The peak width of running `chain` now, or kNoBound once it reaches
    // `bound`. Integers, not infinities: the build may assume finite math.
    constexpr uint32_t kNoBound = std::numeric_limits<uint32_t>::max();
    auto peak = [&](const std::vector<size_t>& chain, uint32_t bound) {
        StabilizerGroup trial = s;
        uint32_t highest = trial.width();
        for (size_t j : chain) {
            advance(trial, p, j);
            highest = std::max(highest, trial.width());
            if (highest >= bound) {
                return kNoBound;
            }
        }
        return highest;
    };
    auto undone = [&] {
        std::vector<size_t> out_ops;
        for (size_t j = 0; j < n; ++j) {
            if (!done.test(j)) {
                out_ops.push_back(j);
            }
        }
        return out_ops;
    };

    while (done.popcount() < n) {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (size_t j : undone()) {
                if (waiting[j]) {
                    continue;
                }
                if (p.is_measurement[j] || !s.anticommutes_with_any(p.swapped_op[j])) {
                    run(j);
                    progressed = true;
                }
            }
        }
        std::vector<size_t> pending;
        for (size_t j : undone()) {
            if (p.is_measurement[j]) {
                pending.push_back(j);
            }
        }
        if (pending.empty()) {
            for (size_t j : undone()) {
                run(j);
            }
            break;
        }
        // A chain that reaches the best peak so far cannot win, as ties go to
        // the earliest, so its simulation stops there.
        size_t best = 0;
        uint32_t lowest = kNoBound;
        std::vector<std::vector<size_t>> chains;
        for (size_t m : pending) {
            std::vector<size_t> chain;
            for (size_t j : ancestors[m].ones()) {
                if (!done.test(j)) {
                    chain.push_back(j);
                }
            }
            chains.push_back(std::move(chain));
        }
        for (size_t k = 0; k < chains.size(); ++k) {
            const uint32_t h = peak(chains[k], lowest);
            if (h < lowest) {
                best = k;
                lowest = h;
            }
        }
        for (size_t j : chains[best]) {
            run(j);
        }
        run(pending[best]);
    }
    return out;
}

}  // namespace clifft::zx
