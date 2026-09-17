// Standalone generated-plan experiment, deliberately outside Clifft dispatch.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>

namespace fold_blocks {
using Mask = __uint128_t;
using Complex = std::complex<double>;
using History = std::array<uint64_t, 128>;
using Pair = std::array<Mask, 2>;
struct Word {
    unsigned index;
    uint64_t mask;
};
struct Row {
    unsigned offset, size;
};
struct Map {
    std::span<const Row> rows;
    std::span<const Word> words;
};
struct Linear {
    Map x, z, records;
};
struct Edge {
    unsigned q, r;
};
struct Leaf {
    unsigned offset, size;
};
struct Step {
    unsigned offset, size, inputs, gather;
};
struct Geometry {
    unsigned width, rank, leaf_storage;
    Mask logical_z;
    std::span<const Mask> z_masks;
    std::span<const Edge> edges;
    std::span<const Leaf> leaves;
    std::span<const Step> steps;
    std::span<const unsigned> gathers, multipliers, outputs;
};
struct Monomial {
    Mask flips = 0, edges = 0;
    unsigned phase = 0;
    std::array<unsigned, 85> linear{};
};
struct Term {
    double weight = 0;
    Monomial op;
};
struct Action {
    unsigned kind, q, r, s, edge;
    int x, z;
};
struct Fold {
    const Geometry* geometry;
    unsigned cats;
    Mask equal_flag_mask;
    Linear preparation;
    Map records;
    std::span<const Action> actions;
    std::span<const int> decode;
};
struct Boundary {
    const Geometry* before;
    Linear noise;
    std::span<const Mask> constraints, syndrome;
    std::span<const Pair> duals, transported;
};
struct Stage {
    const Fold* fold;
    const Boundary* boundary;
};
struct NoiseSite {
    unsigned arity, flip;
    std::array<int, 3> x, z;
};
struct Protocol {
    Linear prefix;
    std::span<const Stage> stages;
    std::span<const NoiseSite> noise;
};
struct Fixture {
    History history;
    std::array<double, 4> expected;
};

inline unsigned parity(Mask value) noexcept {
    return (std::popcount(static_cast<uint64_t>(value)) ^
            std::popcount(static_cast<uint64_t>(value >> 64))) &
           1;
}
inline bool bit(const History& h, int index) noexcept {
    return index >= 0 && ((h[static_cast<unsigned>(index) / 64] >> (index % 64)) & 1);
}
inline Mask apply(Map map, const History& h) noexcept {
    Mask output = 0;
    for (unsigned j = 0; j < map.rows.size(); ++j) {
        Mask value = 0;
        const auto& row = map.rows[j];
        for (unsigned k = 0; k < row.size; ++k) {
            const auto& word = map.words[row.offset + k];
            value ^= h[word.index] & word.mask;
        }
        output |= Mask(parity(value)) << j;
    }
    return output;
}
inline Mask apply(std::span<const Mask> rows, Mask bits) noexcept {
    Mask output = 0;
    for (unsigned j = 0; j < rows.size(); ++j)
        output |= Mask(parity(rows[j] & bits)) << j;
    return output;
}
inline Monomial pauli(Mask x, Mask z, bool inverse = false) noexcept {
    Monomial result;
    result.flips = x;
    result.phase = inverse ? 4 * parity(x & z) : 0;
    for (unsigned q = 0; q < result.linear.size(); ++q)
        result.linear[q] = 4 * ((z >> q) & 1);
    return result;
}
inline Monomial compose(const Monomial& after, const Monomial& before, const Geometry& g) noexcept {
    Monomial out = before;
    out.flips ^= after.flips;
    out.edges ^= after.edges;
    out.phase += after.phase;
    for (unsigned q = 0; q < g.width; ++q) {
        unsigned x = (before.flips >> q) & 1;
        out.phase += after.linear[q] * x;
        out.linear[q] += after.linear[q] * (1 - 2 * x);
    }
    for (unsigned j = 0; j < g.edges.size(); ++j)
        if ((after.edges >> j) & 1) {
            auto [q, r] = g.edges[j];
            unsigned x = (before.flips >> q) & 1, y = (before.flips >> r) & 1;
            out.phase += 4 * x * y;
            out.linear[q] += 4 * y;
            out.linear[r] += 4 * x;
        }
    out.phase &= 7;
    for (auto& c : out.linear)
        c &= 7;
    return out;
}

class Executor {
    std::array<Complex, 2851> values;
    const std::array<Complex, 8> roots = {Complex(1),     Complex(std::sqrt(.5), std::sqrt(.5)),
                                          Complex(0, 1),  Complex(-std::sqrt(.5), std::sqrt(.5)),
                                          Complex(-1),    Complex(-std::sqrt(.5), -std::sqrt(.5)),
                                          Complex(0, -1), Complex(std::sqrt(.5), -std::sqrt(.5))};

    std::array<Complex, 2> sums(const Geometry& g, const Monomial& op) noexcept {
        std::array<Complex, 2> result;
        for (unsigned b = 0; b < 2; ++b) {
            for (unsigned k = 0; k < g.leaves.size(); ++k) {
                const auto& leaf = g.leaves[k];
                unsigned parameter =
                    k < g.width ? op.linear[k] : 4 * ((op.edges >> (k - g.width)) & 1);
                for (unsigned j = 0; j < leaf.size; ++j)
                    values[leaf.offset + j] =
                        roots[parameter * g.multipliers[b * g.leaf_storage + leaf.offset + j]];
            }
            for (const auto& step : g.steps)
                for (unsigned j = 0; j < step.size; ++j) {
                    Complex a = 1, c = 1;
                    for (unsigned k = 0; k < step.inputs; ++k) {
                        unsigned index = step.gather + k * 2 * step.size + 2 * j;
                        a *= values[g.gathers[index]];
                        c *= values[g.gathers[index + 1]];
                    }
                    values[step.offset + j] = a + c;
                }
            Complex total = roots[op.phase] * std::ldexp(1., -static_cast<int>(g.rank));
            for (auto offset : g.outputs)
                total *= values[offset];
            result[b] = total;
        }
        return result;
    }

  public:
    unsigned branches(const Fold& fold, const History& h, std::array<Term, 2>& out) noexcept {
        Mask flags = apply(fold.preparation.records, h);
        if (flags != 0 && flags != fold.equal_flag_mask)
            return 0;
        Mask x = apply(fold.preparation.x, h), z = apply(fold.preparation.z, h);
        Mask records = apply(fold.records, h);
        unsigned count = 0;
        for (unsigned branch = 0; branch < 2; ++branch) {
            Mask initial = branch ? (Mask(1) << fold.cats) - 1 : 0;
            Mask cat = initial ^ x;
            Monomial op;
            for (const auto& a : fold.actions) {
                unsigned xbit = bit(h, a.x), zbit = bit(h, a.z);
                if (a.kind == 0) {
                    op.phase += 4 * zbit * ((op.flips >> a.q) & 1);
                    op.linear[a.q] += 4 * zbit;
                    op.flips ^= Mask(xbit) << a.q;
                } else if (a.kind == 1) {
                    op.phase += 4 * zbit * ((cat >> a.q) & 1);
                    cat ^= Mask(xbit) << a.q;
                } else if (a.kind == 2 || a.kind == 3) {
                    unsigned sign = a.kind == 2 ? 1 : -1;
                    unsigned value = (op.flips >> a.q) & 1;
                    op.phase += sign * value;
                    op.linear[a.q] += sign * (1 - 2 * value);
                } else if ((cat >> a.q) & 1) {
                    if (a.kind == 4)
                        op.flips ^= Mask(1) << a.r;
                    else {
                        unsigned u = (op.flips >> a.r) & 1, v = (op.flips >> a.s) & 1;
                        op.phase += 4 * u * v;
                        op.linear[a.r] += 4 * v;
                        op.linear[a.s] += 4 * u;
                        op.edges ^= Mask(1) << a.edge;
                    }
                }
            }
            double weight =
                .5 * (parity(initial & z) ? -1 : 1) * fold.decode[(records << fold.cats) | cat];
            if (weight) {
                op.phase &= 7;
                for (auto& c : op.linear) {
                    c &= 7;
                    assert(!(c & 1));
                }
                out[count++] = {weight, op};
            }
        }
        return count;
    }

    template <class Branches>
    std::array<double, 4> evaluate_with(const Protocol& plan, const History& h,
                                        Branches&& branch_evaluator) noexcept {
        if (apply(plan.prefix.records, h))
            return {};
        std::array<Complex, 2> state{std::sqrt(.5), Complex(.5, .5)};
        std::array<Term, 4> terms{}, next{};
        terms[0] = {1, pauli(apply(plan.prefix.x, h), apply(plan.prefix.z, h))};
        unsigned count = 1;
        for (size_t stage_index = 0; stage_index < plan.stages.size(); ++stage_index) {
            const auto& stage = plan.stages[stage_index];
            if (stage.fold) {
                std::array<Term, 2> additions{};
                unsigned added = branch_evaluator(stage_index, *stage.fold, h, additions), n = 0;
                assert(count * added <= next.size());
                for (unsigned j = 0; j < count; ++j)
                    for (unsigned k = 0; k < added; ++k)
                        next[n++] = {terms[j].weight * additions[k].weight,
                                     compose(additions[k].op, terms[j].op, *stage.fold->geometry)};
                if (!n)
                    return {};
                terms = next;
                count = n;
            } else {
                const auto& b = *stage.boundary;
                Mask records = apply(b.noise.records, h);
                if (apply(b.constraints, records))
                    return {};
                Mask syndrome = apply(b.syndrome, records);
                Mask x = apply(b.noise.x, h), z = apply(b.noise.z, h), dx = 0, dz = 0;
                for (unsigned j = 0; j < b.duals.size(); ++j)
                    if ((syndrome >> j) & 1) {
                        dx ^= b.duals[j][0];
                        dz ^= b.duals[j][1];
                        x ^= b.transported[j][0];
                        z ^= b.transported[j][1];
                    }
                auto inverse = pauli(dx, dz, true);
                std::array<Complex, 2> output{};
                for (unsigned j = 0; j < count; ++j) {
                    auto op = compose(inverse, terms[j].op, *b.before);
                    if (apply(b.before->z_masks, op.flips))
                        continue;
                    unsigned flip = parity(op.flips & b.before->logical_z);
                    auto amplitudes = sums(*b.before, op);
                    for (unsigned k = 0; k < 2; ++k)
                        output[k ^ flip] += terms[j].weight * amplitudes[k] * state[k];
                }
                state = output;
                if (state[0] == Complex(0) && state[1] == Complex(0))
                    return {};
                terms[0] = {1, pauli(x, z)};
                count = 1;
            }
        }
        Complex cross = std::conj(state[0]) * state[1];
        return {std::norm(state[0]) + std::norm(state[1]), 2 * cross.real(), 2 * cross.imag(),
                std::norm(state[0]) - std::norm(state[1])};
    }

    std::array<double, 4> evaluate(const Protocol& plan, const History& h) noexcept {
        return evaluate_with(
            plan, h,
            [this](size_t, const Fold& fold, const History& history, std::array<Term, 2>& out) {
                return branches(fold, history, out);
            });
    }
};

template <class Probability>
inline void sample_with(const Protocol& plan, History& h, std::mt19937_64& rng,
                        Probability probability) noexcept {
    h.fill(0);
    size_t index = 0;
    for (const auto& site : plan.noise)
        if ((rng() >> 11) * 0x1.0p-53 < probability(index++)) {
            if (site.flip)
                h[site.x[0] / 64] ^= uint64_t(1) << (site.x[0] % 64);
            else {
                uint64_t n = (uint64_t(1) << (2 * site.arity)) - 1, threshold = -n % n, draw;
                do {
                    draw = rng();
                } while (draw < threshold);
                draw = draw % n + 1;
                for (unsigned q = 0; q < site.arity; ++q) {
                    unsigned p = draw & 3;
                    draw >>= 2;
                    if (p == 1 || p == 2)
                        h[site.x[q] / 64] ^= uint64_t(1) << (site.x[q] % 64);
                    if (p == 2 || p == 3)
                        h[site.z[q] / 64] ^= uint64_t(1) << (site.z[q] % 64);
                }
            }
        }
}

inline void sample(const Protocol& plan, History& h, std::mt19937_64& rng,
                   double probability) noexcept {
    sample_with(plan, h, rng, [probability](size_t) { return probability; });
}

inline void sample_sites(const Protocol& plan, History& h, std::mt19937_64& rng,
                         std::span<const double> probabilities) noexcept {
    assert(probabilities.size() == plan.noise.size());
    sample_with(plan, h, rng, [probabilities](size_t k) { return probabilities[k]; });
}

inline int benchmark(const Protocol& plan, std::span<const Fixture> fixtures, int argc,
                     char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: generated_program REPETITIONS SHOTS_PER_TRIAL");
        size_t used_a, used_b;
        auto repeats = std::stoul(argv[1], &used_a), shots = std::stoul(argv[2], &used_b);
        if (used_a != std::string(argv[1]).size() || used_b != std::string(argv[2]).size() ||
            !repeats || repeats > 1000000 || !shots || shots > 100000000)
            throw std::invalid_argument("invalid repetition count");
        Executor executor;
        double error = 0, checksum = 0;
        for (const auto& f : fixtures) {
            auto result = executor.evaluate(plan, f.history);
            for (unsigned k = 0; k < 4; ++k)
                error = std::max(error, std::abs(result[k] - f.expected[k]));
        }
        if (!std::isfinite(error) || error > 2e-12)
            throw std::runtime_error("native/reference mismatch");
        std::array<double, 5> fixed{}, sampled{};
        std::array<double, 3> probes{};
        size_t accepted = 0;
        std::mt19937_64 rng(19331);
        History history{};
        for (unsigned trial = 0; trial < 5; ++trial) {
            auto start = std::chrono::steady_clock::now();
            for (size_t j = 0; j < repeats; ++j)
                for (const auto& f : fixtures) {
                    auto result = executor.evaluate(plan, f.history);
                    for (double value : result)
                        checksum += value;
                }
            fixed[trial] =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                    .count() /
                (repeats * fixtures.size());
            start = std::chrono::steady_clock::now();
            for (size_t j = 0; j < shots; ++j) {
                sample(plan, history, rng, .001);
                auto result = executor.evaluate(plan, history);
                if ((rng() >> 11) * 0x1.0p-53 < result[0]) {
                    ++accepted;
                    for (unsigned k = 0; k < 3; ++k)
                        probes[k] += result[k + 1] / result[0];
                }
            }
            sampled[trial] =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                    .count() /
                shots;
        }
        auto a = fixed, b = sampled;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::cout << std::setprecision(17) << "{\"fixtures\":" << fixtures.size()
                  << ",\"max_error\":" << error << ",\"repetitions\":" << repeats
                  << ",\"shots_per_trial\":" << shots << ",\"fixed_history_median_us\":" << a[2]
                  << ",\"sampled_attempt_median_us\":" << b[2] << ",\"accepted\":" << accepted
                  << ",\"total_attempts\":" << 5 * shots << ",\"fixed_trials_us\":[";
        for (unsigned j = 0; j < 5; ++j)
            std::cout << (j ? "," : "") << fixed[j];
        std::cout << "],\"sampled_trials_us\":[";
        for (unsigned j = 0; j < 5; ++j)
            std::cout << (j ? "," : "") << sampled[j];
        std::cout << "],\"accepted_mean_xyz\":[";
        for (unsigned j = 0; j < 3; ++j)
            std::cout << (j ? "," : "") << (accepted ? probes[j] / accepted : 0);
        std::cout << "],\"checksum\":" << checksum << "}\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
}  // namespace fold_blocks
