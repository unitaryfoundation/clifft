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

double log_add_exp(double first, double second) noexcept {
    if (first == -std::numeric_limits<double>::infinity()) {
        return second;
    }
    if (second == -std::numeric_limits<double>::infinity()) {
        return first;
    }
    const double greater = std::max(first, second);
    const double lesser = std::min(first, second);
    return greater + std::log1p(std::exp(lesser - greater));
}

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

    std::vector<double> odds_ratios;
    odds_ratios.reserve(uncertain_sites_.size());
    long double odds_sum = 0.0L;
    for (uint32_t site : uncertain_sites_) {
        const double probability = probabilities[site];
        const double odds = probability / (1.0 - probability);
        odds_ratios.push_back(odds);
        odds_sum += static_cast<long double>(odds);
    }

    const size_t rows = uncertain_sites_.size() + 1;
    const size_t stride = static_cast<size_t>(remaining_k_) + 1;
    if (rows > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error("fault-conditioning table exceeds size_t range");
    }

    // Most fixed-k tables remain well conditioned after removing their common
    // mean odds. Keep this multiply-add path fast, but reject it if any
    // structurally possible coefficient or branch is lost to floating point.
    bool stable = true;
    const long double normalization = static_cast<long double>(uncertain_sites_.size()) / odds_sum;
    for (double& odds : odds_ratios) {
        odds = static_cast<double>(static_cast<long double>(odds) * normalization);
        stable &= odds > 0.0 && std::isfinite(odds);
    }
    selection_probabilities_.assign(rows * stride, 0.0);
    std::vector<double> row_scale_ratios(uncertain_sites_.size(), 1.0);
    selection_probabilities_[uncertain_sites_.size() * stride] = 1.0;
    for (size_t row = uncertain_sites_.size(); stable && row-- > 0;) {
        const uint32_t remaining = static_cast<uint32_t>(uncertain_sites_.size() - row);
        const uint32_t min_selected = remaining_k_ > row ? remaining_k_ - row : 0;
        const uint32_t max_selected = std::min(remaining, remaining_k_);
        double row_max = 0.0;
        for (uint32_t selected = min_selected; selected <= max_selected; ++selected) {
            double value = selection_probabilities_[(row + 1) * stride + selected];
            if (selected != 0) {
                value +=
                    odds_ratios[row] * selection_probabilities_[(row + 1) * stride + selected - 1];
            }
            selection_probabilities_[row * stride + selected] = value;
            stable &= value > 0.0 && std::isfinite(value);
            row_max = std::max(row_max, value);
        }
        if (stable && row_max > kMaxConditioningRow) {
            const double inverse_row_max = 1.0 / row_max;
            row_scale_ratios[row] = inverse_row_max;
            for (uint32_t selected = min_selected; selected <= max_selected; ++selected) {
                double& value = selection_probabilities_[row * stride + selected];
                value *= inverse_row_max;
                stable &= value > 0.0;
            }
        }
    }
    for (size_t row = 0; stable && row < uncertain_sites_.size(); ++row) {
        const uint32_t remaining = static_cast<uint32_t>(uncertain_sites_.size() - row);
        const uint32_t min_selected = remaining_k_ > row ? remaining_k_ - row : 0;
        const uint32_t max_selected = std::min(remaining, remaining_k_);
        for (uint32_t selected = std::max(uint32_t{1}, min_selected); selected <= max_selected;
             ++selected) {
            if (selected == remaining) {
                selection_probabilities_[row * stride + selected] = 1.0;
                continue;
            }
            const double probability =
                odds_ratios[row] * selection_probabilities_[(row + 1) * stride + selected - 1] *
                row_scale_ratios[row] / selection_probabilities_[row * stride + selected];
            if (!(probability > 0.0 && probability < 1.0 && std::isfinite(probability))) {
                stable = false;
                break;
            }
            selection_probabilities_[row * stride + selected] = probability;
        }
    }
    if (stable) {
        return;
    }

    // Extreme odds or strata can span more than the double exponent range.
    // Log partitions preserve those cases; only setup pays the fallback cost,
    // while workers still consume the same direct-probability table.
    std::vector<double> log_odds_ratios;
    log_odds_ratios.reserve(uncertain_sites_.size());
    for (uint32_t site : uncertain_sites_) {
        const double probability = probabilities[site];
        log_odds_ratios.push_back(std::log(probability) - std::log1p(-probability));
    }
    const double log_zero = -std::numeric_limits<double>::infinity();
    selection_probabilities_.assign(rows * stride, log_zero);
    selection_probabilities_[uncertain_sites_.size() * stride] = 0.0;
    for (size_t row = uncertain_sites_.size(); row-- > 0;) {
        const uint32_t remaining = static_cast<uint32_t>(uncertain_sites_.size() - row);
        const uint32_t min_selected = remaining_k_ > row ? remaining_k_ - row : 0;
        const uint32_t max_selected = std::min(remaining, remaining_k_);
        for (uint32_t selected = min_selected; selected <= max_selected; ++selected) {
            const double without_site = selection_probabilities_[(row + 1) * stride + selected];
            double with_site = log_zero;
            if (selected != 0) {
                with_site = log_odds_ratios[row] +
                            selection_probabilities_[(row + 1) * stride + selected - 1];
            }
            selection_probabilities_[row * stride + selected] =
                log_add_exp(without_site, with_site);
        }
    }

    // Convert the partition table in place. Rows are visited from the front
    // so the suffix row needed by each ratio is still in the log domain.
    for (size_t row = 0; row < uncertain_sites_.size(); ++row) {
        const uint32_t remaining = static_cast<uint32_t>(uncertain_sites_.size() - row);
        const uint32_t min_selected = remaining_k_ > row ? remaining_k_ - row : 0;
        const uint32_t max_selected = std::min(remaining, remaining_k_);
        for (uint32_t selected = std::max(uint32_t{1}, min_selected); selected <= max_selected;
             ++selected) {
            const double log_probability =
                log_odds_ratios[row] + selection_probabilities_[(row + 1) * stride + selected - 1] -
                selection_probabilities_[row * stride + selected];
            // Roundoff can put a probability a few ulps above one. Values
            // below the double range are indistinguishable to the sampler.
            selection_probabilities_[row * stride + selected] =
                std::clamp(std::exp(log_probability), 0.0, 1.0);
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
