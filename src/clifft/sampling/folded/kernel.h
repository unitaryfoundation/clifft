#pragma once

#include "clifft/sampling/folded/contraction.h"
#include "clifft/util/xoshiro.h"

#include <array>
#include <cassert>
#include <sstream>

namespace clifft::sampling::folded {
struct Term {
    Complex coefficient;
    std::vector<Complex> local;
};
using Choice = std::array<std::array<Term, 4>, 2>;
using Case = std::array<Choice, 4>;

struct KernelPlan {
    size_t rank, leaves, characters;
    std::vector<Contraction> marginals;
    Contraction amplitude;
    size_t scratch_size, product_size;

    static Contraction read_plan(const std::string& text, size_t rank, size_t leaves) {
        std::istringstream input(text);
        Reader reader{input};
        return Contraction(reader, rank, leaves, 4);
    }

    KernelPlan(const std::vector<std::string>& tables, size_t rank, size_t leaves,
               size_t characters)
        : rank(rank),
          leaves(leaves),
          characters(characters),
          amplitude(read_plan(tables.at(0), rank, leaves)),
          scratch_size(amplitude.scratch_size()),
          product_size(amplitude.product_size()) {
        if (!rank || rank > 63 || leaves > 512 || characters + rank != leaves ||
            tables.size() != rank + 2)
            throw std::invalid_argument("invalid folded contraction dimensions");
        for (size_t m = 0; m <= rank; ++m) {
            marginals.push_back(read_plan(tables[m + 1], rank + m, 2 * leaves));
            scratch_size = std::max(scratch_size, marginals.back().scratch_size());
            product_size = std::max(product_size, marginals.back().product_size());
        }
    }
};

class Kernel {
    size_t rank_, leaves_, characters_;
    const std::vector<Contraction>& marginals_;
    const Contraction& amplitude_;
    ContractionWorkspace workspace_;
    std::vector<Complex> paired_;
    clifft::Xoshiro256PlusPlus* rng_ = nullptr;
    double root_probability_ = 0;

    double marginal(const Choice& choice, size_t measured, uint64_t syndrome) noexcept {
        double total = 0;
        const size_t stride = 4 * leaves_;
        for (const auto& terms : choice) {
            for (size_t i = 0; i < terms.size(); ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    for (size_t k = 0; k < stride; ++k) {
                        paired_[k] = terms[i].local[k];
                        paired_[stride + k] = std::conj(terms[j].local[k]);
                    }
                    for (size_t k = 0; k < measured; ++k) {
                        if ((syndrome >> k) & 1) {
                            paired_[4 * (characters_ + k) + 1] *= -1;
                            paired_[stride + 4 * (characters_ + k) + 1] *= -1;
                        }
                    }
                    total += ((i == j ? 1.0 : 2.0) * terms[i].coefficient *
                              std::conj(terms[j].coefficient) *
                              marginals_[measured].evaluate(paired_.data(), workspace_))
                                 .real();
                }
            }
        }
        assert(total >= -1e-9);
        return std::max(0.0, total);
    }

    bool draw(double zero, double total) noexcept {
        assert(total > 0 && zero >= -1e-9 && zero <= total + 1e-9);
        return ((*rng_)() >> 11) * 0x1.0p-53 * total >= std::clamp(zero, 0.0, total);
    }

  public:
    size_t root_outcomes = 0;
    uint64_t syndrome = 0;
    std::array<Complex, 2> logical{};

    Kernel(const KernelPlan& plan)
        : rank_(plan.rank),
          leaves_(plan.leaves),
          characters_(plan.characters),
          marginals_(plan.marginals),
          amplitude_(plan.amplitude) {
        workspace_.prepare(plan.scratch_size, plan.product_size);
        paired_.resize(8 * leaves_);
    }

    void use_rng(clifft::Xoshiro256PlusPlus& rng) noexcept { rng_ = &rng; }
    double probability(const Choice& choice) noexcept { return marginal(choice, 0, 0); }
    bool sample_binary(double zero, double total) noexcept { return draw(zero, total); }

    void sample_roots(const Case& instance) noexcept {
        std::array<double, 4> probabilities;
        for (size_t i = 0; i < 4; ++i)
            probabilities[i] = marginal(instance[i], 0, 0);
        const double total =
            probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3];
        assert(std::abs(total - 1) < 1e-8);
        const auto first = draw(probabilities[0] + probabilities[1], total);
        const size_t start = 2 * first;
        root_outcomes =
            start + draw(probabilities[start], probabilities[start] + probabilities[start + 1]);
        root_probability_ = probabilities[root_outcomes];
    }

    void finish(const Choice& choice) noexcept {
        double parent = root_probability_;
        syndrome = 0;
        for (size_t m = 1; m <= rank_; ++m) {
            const double zero = marginal(choice, m, syndrome);
            const bool bit = draw(zero, parent);
            parent = bit ? std::max(0.0, parent - zero) : zero;
            syndrome |= uint64_t(bit) << (m - 1);
        }
        double norm = 0;
        for (size_t l = 0; l < 2; ++l) {
            logical[l] = 0;
            for (const auto& term : choice[l]) {
                std::copy(term.local.begin(), term.local.end(), paired_.begin());
                for (size_t i = 0; i < rank_; ++i)
                    if ((syndrome >> i) & 1)
                        paired_[4 * (characters_ + i) + 1] *= -1;
                logical[l] += term.coefficient * amplitude_.evaluate(paired_.data(), workspace_);
            }
            norm += std::norm(logical[l]);
        }
        assert(norm > 0 && parent > 0);
        assert(std::abs(norm - parent) < 1e-8);
        for (auto& value : logical)
            value /= std::sqrt(norm);
    }
};
}  // namespace clifft::sampling::folded
