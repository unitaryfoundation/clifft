#include "clifft/util/fault_sampling.h"

#include "clifft/util/numeric.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace clifft {

KFaultSampler::KFaultSampler(std::span<const double> probabilities, uint32_t k) : k_(k) {
    if (probabilities.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("fault-site count exceeds uint32 range");
    }
    num_sites_ = static_cast<uint32_t>(probabilities.size());
    if (k > num_sites_) {
        throw std::invalid_argument("k (" + std::to_string(k) + ") exceeds total fault sites (" +
                                    std::to_string(num_sites_) + ")");
    }

    uint32_t impossible_count = 0;
    for (uint32_t site = 0; site < num_sites_; ++site) {
        const double probability = probabilities[site];
        if (!is_probability(probability)) {
            throw std::invalid_argument("fault-site probability must be finite and in [0, 1]");
        }
        if (probability == 0.0) {
            ++impossible_count;
        } else if (probability == 1.0) {
            certain_sites_.push_back(site);
        } else {
            uncertain_sites_.push_back(site);
        }
    }

    const uint32_t certain_count = static_cast<uint32_t>(certain_sites_.size());
    if (k < certain_count || k > num_sites_ - impossible_count) {
        throw std::invalid_argument("k-fault stratum k=" + std::to_string(k) +
                                    " has zero probability mass (" + std::to_string(certain_count) +
                                    " sites have p=1, " + std::to_string(impossible_count) +
                                    " sites have p=0)");
    }
    remaining_k_ = k - certain_count;
    selected_sites_.reserve(k);

    if (remaining_k_ == 0) {
        return;
    }

    const double first_probability = probabilities[uncertain_sites_.front()];
    uniform_mode_ = std::ranges::all_of(
        uncertain_sites_, [&](uint32_t site) { return probabilities[site] == first_probability; });
    if (uniform_mode_) {
        uniform_pool_ = uncertain_sites_;
        swap_targets_.resize(remaining_k_);
        return;
    }

    odds_ratios_.reserve(uncertain_sites_.size());
    for (uint32_t site : uncertain_sites_) {
        const double probability = probabilities[site];
        odds_ratios_.push_back(probability / (1.0 - probability));
    }
    const double sum = std::accumulate(odds_ratios_.begin(), odds_ratios_.end(), 0.0);
    const double scale = static_cast<double>(odds_ratios_.size()) / sum;
    for (double& odds : odds_ratios_) {
        odds *= scale;
    }

    const size_t rows = uncertain_sites_.size() + 1;
    const size_t stride = static_cast<size_t>(remaining_k_) + 1;
    if (rows > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error("fault-conditioning table exceeds size_t range");
    }
    dp_.assign(rows * stride, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        dp_[row * stride] = 1.0;
    }
    for (size_t row = uncertain_sites_.size(); row-- > 0;) {
        const uint32_t remaining = static_cast<uint32_t>(uncertain_sites_.size() - row);
        const uint32_t max_selected = std::min(remaining, remaining_k_);
        for (uint32_t selected = 1; selected <= max_selected; ++selected) {
            dp_[row * stride + selected] =
                dp_[(row + 1) * stride + selected] +
                odds_ratios_[row] * dp_[(row + 1) * stride + selected - 1];
        }
    }
}

}  // namespace clifft
