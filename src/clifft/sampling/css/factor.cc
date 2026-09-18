#include "clifft/sampling/css/factor.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <tuple>

namespace clifft::sampling::css {
namespace {
constexpr double h = std::numbers::sqrt2 / 2;
const std::array<Complex, 8> roots{Complex{1}, {h, h},   {0, 1},  {-h, h},
                                   {-1, 0},    {-h, -h}, {0, -1}, {h, -h}};
unsigned parity(uint64_t bits) noexcept {
    return std::popcount(bits) & 1;
}
void require(bool value) {
    if (!value)
        throw std::invalid_argument("unsupported or inconsistent CSS block certificate");
}
std::vector<unsigned> positions(uint64_t bits) {
    std::vector<unsigned> result;
    while (bits) {
        unsigned k = std::countr_zero(bits);
        result.push_back(k);
        bits &= bits - 1;
    }
    return result;
}
uint64_t solve(std::vector<uint64_t> rows, uint64_t rhs, unsigned width) {
    std::vector<unsigned> pivots;
    unsigned rank = 0;
    for (unsigned q = 0; q < width; ++q) {
        unsigned pivot = rank;
        while (pivot < rows.size() && !(rows[pivot] >> q & 1))
            ++pivot;
        if (pivot == rows.size())
            continue;
        std::swap(rows[pivot], rows[rank]);
        if (((rhs >> pivot) ^ (rhs >> rank)) & 1)
            rhs ^= (uint64_t{1} << pivot) | (uint64_t{1} << rank);
        for (unsigned k = 0; k < rows.size(); ++k)
            if (k != rank && (rows[k] >> q & 1)) {
                rows[k] ^= rows[rank];
                rhs ^= ((rhs >> rank) & 1) << k;
            }
        pivots.push_back(q);
        ++rank;
    }
    require(rank == rows.size());
    uint64_t result = 0;
    for (unsigned k = 0; k < rank; ++k)
        result |= ((rhs >> k) & 1) << pivots[k];
    return result;
}
std::vector<unsigned> elimination_order(std::span<const uint64_t> scopes, unsigned rank) {
    std::vector<uint64_t> graph(rank);
    for (auto scope : scopes)
        for (auto k : positions(scope))
            graph[k] |= scope & ~(uint64_t{1} << k);
    std::vector<unsigned> order;
    uint64_t remaining = (uint64_t{1} << rank) - 1;
    while (remaining) {
        unsigned best = rank;
        auto best_score = std::tuple{std::numeric_limits<unsigned>::max(), rank, rank};
        for (auto k : positions(remaining)) {
            auto neighbors = positions(graph[k]);
            unsigned fill = 0;
            for (auto a : neighbors)
                for (auto b : neighbors)
                    fill += a < b && !(graph[a] >> b & 1);
            auto score = std::tuple{fill, unsigned(neighbors.size()), k};
            if (score < best_score) {
                best = k;
                best_score = score;
            }
        }
        for (auto a : positions(graph[best]))
            graph[a] = (graph[a] | graph[best]) & ~(uint64_t{1} << a) & ~(uint64_t{1} << best);
        remaining &= ~(uint64_t{1} << best);
        order.push_back(best);
    }
    return order;
}
struct Monomial {
    uint64_t flip = 0;
    unsigned phase = 0;
    std::array<unsigned, kMaxData> linear{};
    double weight = .5;
};
struct Case {
    std::array<unsigned, 2> labels{};
    std::array<std::array<Complex, 2>, 2> amplitude{};
    std::array<std::array<std::array<unsigned, kMaxData>, 2>, 2> phase{};
};
// Local Pauli/T arithmetic is tabulated, including phases of Y, so binding
// physical fault bits cannot perform gate propagation during execution.
struct Local {
    unsigned flip, phase, linear;
};
constexpr auto make_table() {
    std::array<std::array<Local, 2>, 256> result{};
    for (unsigned f = 0; f < 256; ++f)
        for (unsigned branch = 0; branch < 2; ++branch) {
            Local v{};
            auto pauli = [&](unsigned x, unsigned z) {
                v.phase += 4 * z * v.flip + 2 * x * z;
                v.linear += 4 * z;
                v.flip ^= x;
            };
            for (unsigned layer = 0; layer < 4; ++layer) {
                pauli((f >> (2 * layer)) & 1, (f >> (2 * layer + 1)) & 1);
                if (layer == 0 || layer == 2) {
                    unsigned angle = layer == 0 ? 1 : 7;
                    v.phase += angle * v.flip;
                    v.linear += angle * (v.flip ? 7 : 1);
                } else if (layer == 1 && branch)
                    pauli(1, 1);
            }
            v.phase &= 7;
            v.linear &= 7;
            result[f][branch] = v;
        }
    return result;
}
constexpr auto local_table = make_table();
}  // namespace

Factor::Factor(std::span<const std::pair<uint64_t, unsigned>> factors,
               std::span<const unsigned> order, size_t limit) {
    require(limit > 0 && limit <= 2000000);
    std::vector<std::pair<uint64_t, unsigned>> active;
    for (auto [scope, parameter] : factors) {
        unsigned bits = std::popcount(scope);
        require(bits < 31 && (uint64_t{1} << bits) <= limit - storage_);
        unsigned size = 1u << bits;
        Leaf leaf{parameter, storage_, std::vector<uint8_t>(size)};
        for (unsigned k = 0; k < size; ++k)
            leaf.parity[k] = parity(k);
        leaves_.push_back(std::move(leaf));
        active.emplace_back(scope, storage_);
        storage_ += size;
    }
    for (auto variable : order) {
        std::vector<std::pair<uint64_t, unsigned>> inputs, next;
        uint64_t joint = 0;
        for (auto pair : active) {
            if (pair.first >> variable & 1) {
                inputs.push_back(pair);
                joint |= pair.first;
            } else
                next.push_back(pair);
        }
        require(!inputs.empty());
        auto neighbors = positions(joint & ~(uint64_t{1} << variable));
        require(neighbors.size() < 30);
        unsigned size = 1u << neighbors.size();
        require(size <= limit - storage_ && uint64_t{2} * size * inputs.size() <= limit - gathers_);
        neighbors.insert(neighbors.begin(), variable);
        Step step{storage_, size, unsigned(inputs.size()), {}};
        step.gathers.reserve(2 * size * inputs.size());
        for (auto [scope, offset] : inputs) {
            auto local = positions(scope);
            for (unsigned b = 0; b < 2 * size; ++b) {
                unsigned index = 0;
                for (unsigned k = 0; k < local.size(); ++k) {
                    auto pos =
                        std::find(neighbors.begin(), neighbors.end(), local[k]) - neighbors.begin();
                    index |= ((b >> pos) & 1) << k;
                }
                step.gathers.push_back(offset + index);
            }
        }
        gathers_ += step.gathers.size();
        steps_.push_back(std::move(step));
        next.emplace_back(joint & ~(uint64_t{1} << variable), storage_);
        storage_ += size;
        active = std::move(next);
    }
    for (auto [scope, offset] : active) {
        require(scope == 0);
        outputs_.push_back(offset);
    }
}
Complex Factor::evaluate(std::span<const unsigned> parameters,
                         std::span<Complex> scratch) const noexcept {
    assert(scratch.size() >= storage_);
    for (const auto& leaf : leaves_) {
        Complex phase = roots[parameters[leaf.parameter] & 7];
        for (unsigned k = 0; k < leaf.parity.size(); ++k)
            scratch[leaf.offset + k] = leaf.parity[k] ? phase : Complex{1};
    }
    for (const auto& step : steps_)
        for (unsigned j = 0; j < step.size; ++j) {
            Complex a{1}, b{1};
            for (unsigned k = 0; k < step.inputs; ++k) {
                unsigned offset = 2 * step.size * k + 2 * j;
                a *= scratch[step.gathers[offset]];
                b *= scratch[step.gathers[offset + 1]];
            }
            scratch[step.offset + j] = .5 * (a + b);
        }
    Complex result{1};
    for (auto offset : outputs_)
        result *= scratch[offset];
    return result;
}

Code::Code(unsigned width, std::vector<Check> checks, size_t limit)
    : width_(width), checks_(std::move(checks)) {
    require(width > 1 && width <= kMaxData && width % 2 == 1 && checks_.size() == width - 1);
    all_ = (uint64_t{1} << width) - 1;
    for (const auto& check : checks_) {
        require(check.support && check.support <= all_ && !parity(check.support));
        auto& family = check.x ? xchecks_ : zchecks_;
        check_order_.emplace_back(check.x, unsigned(family.size()));
        family.push_back(check.support);
    }
    require(xchecks_.size() == zchecks_.size() && rank() <= kMaxRank);
    for (auto x : xchecks_)
        for (auto z : zchecks_)
            require(!parity(x & z));
    for (auto [x, k] : check_order_) {
        // Including the all-data logical row makes each syndrome correction
        // commute with both logical axes, so terminal readout needs no frame bit.
        auto rows = x ? xchecks_ : zchecks_;
        rows.push_back(all_);
        auto dual = solve(rows, uint64_t{1} << k, width);
        duals_.emplace_back(x ? 0 : dual, x ? dual : 0);
    }
    std::vector<std::pair<uint64_t, unsigned>> basis(width);
    for (unsigned k = 0; k < rank(); ++k) {
        uint64_t key = xchecks_[k];
        unsigned coords = 1u << k;
        while (key) {
            unsigned pivot = 63 - std::countl_zero(key);
            if (!basis[pivot].first) {
                basis[pivot] = {key, coords};
                break;
            }
            key ^= basis[pivot].first;
            coords ^= basis[pivot].second;
        }
        require(key != 0);
    }
    coordinates_.resize(rank());
    for (unsigned q = 0; q < width; ++q) {
        uint64_t key = uint64_t{1} << q;
        unsigned coords = 0;
        for (unsigned p = width; p-- > 0;)
            if (key >> p & 1) {
                key ^= basis[p].first;
                coords ^= basis[p].second;
            }
        for (unsigned k = 0; k < rank(); ++k)
            coordinates_[k] |= uint64_t{(coords >> k) & 1} << q;
    }
    std::vector<uint64_t> scopes(width);
    for (unsigned q = 0; q < width; ++q)
        for (unsigned k = 0; k < rank(); ++k)
            scopes[q] |= ((xchecks_[k] >> q) & 1) << k;
    order_ = elimination_order(scopes, rank());
    std::vector<std::pair<uint64_t, unsigned>> factors;
    for (unsigned q = 0; q < width; ++q)
        factors.emplace_back(scopes[q], q);
    for (unsigned k = 0; k < rank(); ++k)
        factors.emplace_back(uint64_t{1} << k, width + k);
    single_.emplace_back(factors, order_, limit);
    scratch_size_ = single_[0].storage();
    gather_entries_ = single_[0].gather_entries();
    for (unsigned count = 0; count <= rank(); ++count) {
        // Measured X generators have separate bra/ket variables. Unmeasured
        // generators share a variable, which sums their outcomes without
        // materializing an exponentially large syndrome distribution.
        factors.clear();
        std::vector<unsigned> mapping(rank()), elimination;
        for (unsigned k = 0; k < rank(); ++k)
            mapping[k] = k;
        for (unsigned j = 0; j < count; ++j)
            mapping[order_[j]] = rank() + j;
        for (unsigned q = 0; q < width; ++q)
            factors.emplace_back(scopes[q], q);
        for (unsigned q = 0; q < width; ++q) {
            uint64_t scope = 0;
            for (auto k : positions(scopes[q]))
                scope |= uint64_t{1} << mapping[k];
            factors.emplace_back(scope, width + q);
        }
        for (unsigned j = 0; j < count; ++j) {
            unsigned k = order_[j];
            factors.emplace_back(uint64_t{1} << k, 2 * width + k);
            factors.emplace_back(uint64_t{1} << mapping[k], 2 * width + k);
            elimination.push_back(k);
            elimination.push_back(mapping[k]);
        }
        elimination.insert(elimination.end(), order_.begin() + count, order_.end());
        prefixes_.emplace_back(factors, elimination, limit);
        scratch_size_ = std::max(scratch_size_, prefixes_.back().storage());
        gather_entries_ += prefixes_.back().gather_entries();
        require(gather_entries_ <= 8000000);
    }
}
unsigned Code::coordinate(uint64_t bits) const noexcept {
    unsigned result = 0;
    for (unsigned k = 0; k < rank(); ++k)
        result |= parity(bits & coordinates_[k]) << k;
    return result;
}
uint64_t Code::expand(unsigned bits) const noexcept {
    uint64_t result = 0;
    for (unsigned k = 0; k < rank(); ++k)
        if (bits >> k & 1)
            result ^= xchecks_[k];
    return result;
}

struct Engine {
    const Code& code;
    Workspace& work;
    std::array<Complex, 2>& logical;
    Xoshiro256PlusPlus& rng;
    uint64_t frame_x = 0, frame_z = 0;
    std::array<Monomial, 2> terms{};
    Case data;
    Engine(const Code& c, Workspace& w, std::array<Complex, 2>& l, Xoshiro256PlusPlus& random)
        : code(c), work(w), logical(l), rng(random) {
        for (unsigned q = 0; q < code.width_; ++q) {
            frame_x |= uint64_t{work.inputs[2 * q]} << q;
            frame_z |= uint64_t{work.inputs[2 * q + 1]} << q;
            unsigned faults = 0;
            for (unsigned layer = 0; layer < 4; ++layer) {
                unsigned base = 2 * code.width_ * (1 + layer) + 2 * q;
                faults |= unsigned(work.inputs[base]) << (2 * layer);
                faults |= unsigned(work.inputs[base + 1]) << (2 * layer + 1);
            }
            for (unsigned t = 0; t < 2; ++t) {
                const auto& local = local_table[faults][t];
                terms[t].flip |= uint64_t{local.flip} << q;
                terms[t].phase = (terms[t].phase + local.phase) & 7;
                terms[t].linear[q] = local.linear;
            }
        }
        for (unsigned t = 0; t < 2; ++t) {
            const auto& op = terms[t];
            uint64_t offset = frame_x ^ op.flip;
            for (unsigned k = 0; k < code.rank(); ++k)
                data.labels[t] |= parity(offset & code.zchecks_[k]) << k;
            for (unsigned l = 0; l < 2; ++l) {
                unsigned source = l ^ parity(offset);
                uint64_t delta = frame_x ^ (source ? code.all_ : 0) ^
                                 code.expand(code.coordinate(offset ^ (source ? code.all_ : 0)));
                unsigned constant = op.phase + 4 * parity(frame_z & (delta ^ frame_x));
                for (unsigned q = 0; q < code.width_; ++q) {
                    unsigned bit = (delta >> q) & 1;
                    constant += op.linear[q] * bit;
                    data.phase[t][l][q] =
                        (op.linear[q] * (bit ? 7 : 1) + 4 * ((frame_z >> q) & 1)) & 7;
                }
                data.amplitude[t][l] = logical[source] * roots[constant & 7];
            }
        }
        assert(data.labels[0] == data.labels[1]);
    }
    double mass(unsigned count, unsigned syndrome) noexcept {
        std::array<unsigned, 3 * kMaxData> parameters{};
        for (unsigned k = 0; k < code.rank(); ++k)
            parameters[2 * code.width_ + k] = 4 * ((syndrome >> k) & 1);
        Complex result{};
        for (unsigned a = 0; a < 2; ++a)
            for (unsigned b = a; b < 2; ++b)
                for (unsigned l = 0; l < 2; ++l) {
                    Complex coefficient = terms[a].weight * terms[b].weight *
                                          std::conj(data.amplitude[a][l]) * data.amplitude[b][l];
                    if (coefficient == Complex{})
                        continue;
                    for (unsigned q = 0; q < code.width_; ++q) {
                        parameters[q] = (8 - data.phase[a][l][q]) & 7;
                        parameters[code.width_ + q] = data.phase[b][l][q];
                    }
                    Complex value =
                        coefficient * code.prefixes_[count].evaluate(parameters, work.coefficients);
                    result += a == b ? value : Complex{2 * value.real()};
                }
        assert(std::abs(result.imag()) < 1e-10 && result.real() >= -1e-10);
        return std::max(0., result.real());
    }
    double contract(unsigned xs, unsigned zs) noexcept {
        if (zs != data.labels[0])
            return 0;
        uint64_t cx = 0, cz = 0;
        for (unsigned k = 0; k < code.checks_.size(); ++k) {
            auto [x, j] = code.check_order_[k];
            if ((x ? xs : zs) >> j & 1) {
                cx ^= code.duals_[k].first;
                cz ^= code.duals_[k].second;
            }
        }
        uint64_t logical_rep = code.all_ ^ code.expand(code.coordinate(code.all_));
        // Convert the coset representative used by the contraction into the
        // certified syndrome-dual frame carried into the next block. Its phase
        // matters for subsequent checks and for logical X/Y measurements.
        uint64_t representative =
            cx ^ code.expand(code.coordinate(cx)) ^ (parity(cx) ? logical_rep : 0);
        std::array<Complex, 2> output{};
        std::array<unsigned, 3 * kMaxData> parameters{};
        for (unsigned k = 0; k < code.rank(); ++k)
            parameters[code.width_ + k] = 4 * ((xs >> k) & 1);
        for (unsigned t = 0; t < 2; ++t)
            for (unsigned l = 0; l < 2; ++l) {
                for (unsigned q = 0; q < code.width_; ++q)
                    parameters[q] = data.phase[t][l][q];
                uint64_t delta = representative ^ (l ? logical_rep : 0) ^ cx;
                output[l ^ parity(cx)] += terms[t].weight * data.amplitude[t][l] *
                                          code.single_[0].evaluate(parameters, work.coefficients) *
                                          (parity(delta & cz) ? -1. : 1.);
            }
        double weight = std::norm(output[0]) + std::norm(output[1]);
        assert(std::isfinite(weight) && weight >= 0);
        if (weight > 0)
            for (unsigned l = 0; l < 2; ++l)
                logical[l] = output[l] / std::sqrt(weight);
        return weight;
    }
    double run(bool forced) noexcept {
        unsigned xs = 0, zs = data.labels[0];
        if (forced) {
            zs = 0;
            for (unsigned k = 0; k < code.checks_.size(); ++k) {
                auto [x, j] = code.check_order_[k];
                (x ? xs : zs) |= unsigned(work.outcomes[k + 1]) << j;
            }
        } else {
            double zero = mass(0, 0);
            assert(zero <= 1 + 1e-10);
            work.outcomes[0] = rng.next_double() >= std::min(1., zero);
        }
        if (work.outcomes[0])
            terms[1].weight = -.5;
        if (!forced) {
            double current = mass(0, 0);
            assert(current > 0);
            for (unsigned m = 0; m < code.rank(); ++m) {
                double zero = mass(m + 1, xs);
                assert(zero <= current + 1e-10);
                zero = std::min(current, zero);
                bool bit = rng.next_double() * current >= zero;
                xs |= unsigned(bit) << code.order_[m];
                current = bit ? current - zero : zero;
            }
            for (unsigned k = 0; k < code.checks_.size(); ++k) {
                auto [x, j] = code.check_order_[k];
                work.outcomes[k + 1] = ((x ? xs : zs) >> j) & 1;
            }
        }
        return contract(xs, zs);
    }
};
double apply(const Code& code, std::array<Complex, 2>& logical, Workspace& work,
             Xoshiro256PlusPlus& rng, bool forced) noexcept {
    assert(work.coefficients.size() >= code.scratch_size());
    return Engine(code, work, logical, rng).run(forced);
}
}  // namespace clifft::sampling::css
