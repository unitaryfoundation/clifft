#pragma once

#include "clifft/util/xoshiro.h"

#include "gadget_contraction_kernel.h"

#include <array>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace gadget_study;

namespace folded_study {
struct Term {
    Complex coefficient;
    std::vector<Complex> local;
};
using Choice = std::array<std::array<Term, 4>, 2>;
using Case = std::array<Choice, 4>;

class FoldedSampler {
    size_t rank_, leaves_, characters_;
    std::vector<Case> cases_;
    std::vector<Contraction> marginals_;
    Contraction amplitude_;
    ContractionWorkspace workspace_;
    std::vector<Complex> paired_;
    clifft::Xoshiro256PlusPlus rng_;
    double root_probability_ = 0;

    static Contraction read_plan(const std::filesystem::path& path, size_t rank, size_t leaves) {
        std::ifstream input(path);
        Reader reader{input};
        return Contraction(reader, rank, leaves, 4);
    }

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
        return (rng_() >> 11) * 0x1.0p-53 * total >= std::clamp(zero, 0.0, total);
    }

  public:
    size_t root_outcomes = 0;
    uint64_t syndrome = 0;
    std::array<Complex, 2> logical{};
    double log_probability = 0;
    double max_normalization_error = 0;
    double max_continuation_error = 0;

    FoldedSampler(const std::filesystem::path& directory, uint64_t seed, size_t rank, size_t leaves,
                  size_t characters, bool load_inputs = true)
        : rank_(rank),
          leaves_(leaves),
          characters_(characters),
          amplitude_(read_plan(directory / "amplitude.txt", rank, leaves)),
          rng_(seed) {
        if (!rank_ || rank_ > 63 || leaves_ > 512 || characters_ + rank_ != leaves_)
            throw std::invalid_argument("invalid folded contraction dimensions");
        std::ifstream input(directory / "inputs.txt");
        Reader reader{input};
        const auto count = load_inputs ? reader.size() : 1;
        if (!count || count > 128)
            throw std::invalid_argument("invalid folded input case count");
        cases_.resize(count);
        for (auto& instance : cases_)
            for (auto& choice : instance)
                for (auto& terms : choice)
                    for (auto& term : terms) {
                        const double re = load_inputs ? reader.number() : 0;
                        const double im = load_inputs ? reader.number() : 0;
                        term.coefficient = {re, im};
                        if (std::abs(term.coefficient) > 1)
                            throw std::invalid_argument("invalid folded path coefficient");
                        term.local.resize(4 * leaves_);
                        for (auto& value : term.local) {
                            const double x = load_inputs ? reader.number() : 1;
                            const double y = load_inputs ? reader.number() : 0;
                            value = {x, y};
                            if (std::abs(std::norm(value) - 1) > 1e-9)
                                throw std::invalid_argument("folded local factor is not a phase");
                        }
                    }
        for (size_t m = 0; m <= rank_; ++m)
            marginals_.push_back(read_plan(directory / ("marginal_" + std::to_string(m) + ".txt"),
                                           rank_ + m, 2 * leaves_));
        size_t scratch_size = amplitude_.scratch_size();
        size_t product_size = amplitude_.product_size();
        for (const auto& plan : marginals_) {
            scratch_size = std::max(scratch_size, plan.scratch_size());
            product_size = std::max(product_size, plan.product_size());
        }
        workspace_.prepare(scratch_size, product_size);
        paired_.resize(8 * leaves_);
        if (load_inputs)
            for (const auto& instance : cases_) {
                double norm = 0;
                for (const auto& choice : instance)
                    norm += marginal(choice, 0, 0);
                if (std::abs(norm - 1) > 1e-9)
                    throw std::invalid_argument("folded input is not normalized");
            }
    }

    Case& input() noexcept { return cases_[0]; }
    size_t lookup_bytes() const noexcept {
        size_t result = amplitude_.lookup_bytes();
        for (const auto& plan : marginals_)
            result += plan.lookup_bytes();
        return result;
    }
    size_t workspace_bytes() const noexcept {
        return sizeof(Complex) * (workspace_.scratch.capacity() + workspace_.product.capacity());
    }
    double probability(const Choice& choice) noexcept { return marginal(choice, 0, 0); }
    bool sample_binary(double zero, double total) noexcept { return draw(zero, total); }

    void sample_roots(const Case& instance) noexcept {
        std::array<double, 4> probabilities;
        for (size_t i = 0; i < 4; ++i)
            probabilities[i] = marginal(instance[i], 0, 0);
        const double total =
            probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3];
        max_normalization_error = std::max(max_normalization_error, std::abs(total - 1));
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
        max_continuation_error = std::max(max_continuation_error, std::abs(norm - parent));
        for (auto& value : logical)
            value /= std::sqrt(norm);
        log_probability = std::log(norm);
    }

    void run(size_t shot) noexcept {
        const auto& instance = cases_[shot % cases_.size()];
        sample_roots(instance);
        finish(instance[root_outcomes]);
    }
};
}  // namespace folded_study
