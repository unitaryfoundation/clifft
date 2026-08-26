#include "clifft/util/fault_sampling.h"

#include "clifft/util/numeric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

namespace {

inline constexpr double kMaxConditioningRow = 0x1p500;

}  // namespace

KFaultDistribution::KFaultDistribution(std::span<const double> probabilities, uint32_t k) : k_(k) {
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
    if (remaining_k_ == 0) {
        return;
    }

    const double first_probability = probabilities[uncertain_sites_.front()];
    uniform_mode_ = std::ranges::all_of(
        uncertain_sites_, [&](uint32_t site) { return probabilities[site] == first_probability; });
    if (uniform_mode_) {
        return;
    }

    odds_ratios_.reserve(uncertain_sites_.size());
    for (uint32_t site : uncertain_sites_) {
        const double probability = probabilities[site];
        odds_ratios_.push_back(probability / (1.0 - probability));
    }
    // A common odds factor cancels when conditioning on a fixed count. Scale
    // by the largest odds so every recurrence coefficient stays in [0, 1].
    const double scale = *std::ranges::max_element(odds_ratios_);
    for (double& odds : odds_ratios_) {
        odds /= scale;
    }

    const size_t rows = uncertain_sites_.size() + 1;
    const size_t stride = static_cast<size_t>(remaining_k_) + 1;
    if (rows > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error("fault-conditioning table exceeds size_t range");
    }
    dp_.assign(rows * stride, 0.0);
    row_scale_ratios_.resize(uncertain_sites_.size());
    dp_[uncertain_sites_.size() * stride] = 1.0;
    for (size_t row = uncertain_sites_.size(); row-- > 0;) {
        const uint32_t remaining = static_cast<uint32_t>(uncertain_sites_.size() - row);
        const uint32_t max_selected = std::min(remaining, remaining_k_);
        double row_max = 0.0;
        for (uint32_t selected = 0; selected <= max_selected; ++selected) {
            double value = dp_[(row + 1) * stride + selected];
            if (selected != 0) {
                value += odds_ratios_[row] * dp_[(row + 1) * stride + selected - 1];
            }
            dp_[row * stride + selected] = value;
            row_max = std::max(row_max, value);
        }
        if (!(row_max > 0.0) || !std::isfinite(row_max)) {
            throw std::overflow_error("fault-conditioning table is numerically unstable");
        }
        row_scale_ratios_[row] = 1.0;
        if (row_max > kMaxConditioningRow) {
            // Only ratios between adjacent rows are needed while sampling.
            // Rescale before coefficients approach overflow and retain the
            // scale change explicitly for the conditional probability.
            const double inverse_row_max = 1.0 / row_max;
            row_scale_ratios_[row] = inverse_row_max;
            for (uint32_t selected = 0; selected <= max_selected; ++selected) {
                dp_[row * stride + selected] *= inverse_row_max;
            }
        }
    }
}

uint64_t KFaultDistribution::worker_scratch_bytes() const noexcept {
    uint64_t entries = k_;
    if (uniform_mode_ && remaining_k_ != 0) {
        entries += uncertain_sites_.size();
        entries += remaining_k_;
    }
    return entries * sizeof(uint32_t);
}

KFaultSampler::KFaultSampler(std::span<const double> probabilities, uint32_t k)
    : KFaultSampler(std::make_shared<const KFaultDistribution>(probabilities, k)) {}

KFaultSampler::KFaultSampler(std::shared_ptr<const KFaultDistribution> distribution)
    : distribution_(std::move(distribution)) {
    if (!distribution_) {
        throw std::invalid_argument("fault distribution must not be null");
    }
    selected_sites_.reserve(distribution_->k_);
    if (distribution_->uniform_mode_ && distribution_->remaining_k_ != 0) {
        uniform_pool_ = distribution_->uncertain_sites_;
        swap_targets_.resize(distribution_->remaining_k_);
    }
}

}  // namespace clifft
