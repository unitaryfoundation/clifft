#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

// Precomputes and repeatedly samples the exact distribution of independent
// Bernoulli sites conditioned on exactly k sites firing. Construction owns all
// allocation and validation; sample() only mutates preallocated buffers.
class KFaultSampler {
  public:
    KFaultSampler(std::span<const double> probabilities, uint32_t k);
    KFaultSampler(const KFaultSampler&) = delete;
    KFaultSampler& operator=(const KFaultSampler&) = delete;
    KFaultSampler(KFaultSampler&&) noexcept = default;
    KFaultSampler& operator=(KFaultSampler&&) noexcept = default;

    [[nodiscard]] uint32_t num_sites() const noexcept { return num_sites_; }
    [[nodiscard]] uint32_t k() const noexcept { return k_; }

    template <typename RandomDouble>
    [[nodiscard]] std::span<const uint32_t> sample(RandomDouble&& random_double) noexcept {
        selected_sites_.clear();
        selected_sites_.insert(selected_sites_.end(), certain_sites_.begin(), certain_sites_.end());

        if (uniform_mode_) {
            const uint32_t count = static_cast<uint32_t>(uniform_pool_.size());
            for (uint32_t j = 0; j < remaining_k_; ++j) {
                const uint32_t remaining = count - j;
                const double draw = random_double();
                assert(draw >= 0.0 && draw < 1.0 && "fault selection draw must be in [0, 1)");
                const uint32_t pick = j + static_cast<uint32_t>(draw * remaining);
                std::swap(uniform_pool_[j], uniform_pool_[pick]);
            }
            // Keep the pool permutation across calls so a seeded sampler has
            // one stable RNG evolution while still returning circuit order.
            std::sort(uniform_pool_.begin(), uniform_pool_.begin() + remaining_k_);
            for (uint32_t j = 0; j < remaining_k_; ++j) {
                selected_sites_.push_back(uniform_pool_[j]);
            }
        } else {
            const uint32_t count = static_cast<uint32_t>(uncertain_sites_.size());
            const uint32_t stride = remaining_k_ + 1;
            uint32_t needed = remaining_k_;
            for (uint32_t i = 0; i < count && needed > 0; ++i) {
                double probability = 1.0;
                const uint32_t remaining = count - i;
                if (needed != remaining) {
                    const double denominator = dp_[static_cast<size_t>(i) * stride + needed];
                    probability = denominator > 0.0
                                      ? odds_ratios_[i] *
                                            dp_[static_cast<size_t>(i + 1) * stride + needed - 1] /
                                            denominator
                                      : 0.0;
                }
                const double draw = random_double();
                assert(draw >= 0.0 && draw < 1.0 && "fault selection draw must be in [0, 1)");
                if (draw < probability) {
                    selected_sites_.push_back(uncertain_sites_[i]);
                    --needed;
                }
            }
            assert(needed == 0 && "conditioned fault sampler must select the requested stratum");
        }

        std::sort(selected_sites_.begin(), selected_sites_.end());
        assert(selected_sites_.size() == k_ && "conditioned fault sampler returned wrong size");
        return selected_sites_;
    }

  private:
    uint32_t num_sites_ = 0;
    uint32_t k_ = 0;
    uint32_t remaining_k_ = 0;
    bool uniform_mode_ = true;
    std::vector<uint32_t> certain_sites_;
    std::vector<uint32_t> uncertain_sites_;
    std::vector<double> odds_ratios_;
    std::vector<double> dp_;
    std::vector<uint32_t> uniform_pool_;
    std::vector<uint32_t> selected_sites_;
};

}  // namespace clifft
