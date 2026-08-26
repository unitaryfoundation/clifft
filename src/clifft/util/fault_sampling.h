#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace clifft {

// Immutable preparation for independent Bernoulli sites conditioned on exactly
// k sites firing. Sharing the potentially large DP table keeps threaded
// sampling memory proportional to one preparation plus small worker scratch.
class KFaultDistribution {
  public:
    KFaultDistribution(std::span<const double> probabilities, uint32_t k);

    KFaultDistribution(const KFaultDistribution&) = delete;
    KFaultDistribution& operator=(const KFaultDistribution&) = delete;
    KFaultDistribution(KFaultDistribution&&) noexcept = default;
    KFaultDistribution& operator=(KFaultDistribution&&) noexcept = default;

    [[nodiscard]] uint32_t num_sites() const noexcept { return num_sites_; }
    [[nodiscard]] uint32_t k() const noexcept { return k_; }
    [[nodiscard]] uint64_t worker_scratch_bytes() const noexcept;

  private:
    friend class KFaultSampler;

    uint32_t num_sites_ = 0;
    uint32_t k_ = 0;
    uint32_t remaining_k_ = 0;
    bool uniform_mode_ = true;
    std::vector<uint32_t> certain_sites_;
    std::vector<uint32_t> uncertain_sites_;
    std::vector<double> selection_probabilities_;
};

// Repeatedly samples a prepared fixed-k distribution. Construction allocates
// all worker-local scratch; sample() only mutates those preallocated buffers.
class KFaultSampler {
  public:
    KFaultSampler(std::span<const double> probabilities, uint32_t k);
    explicit KFaultSampler(std::shared_ptr<const KFaultDistribution> distribution);
    KFaultSampler(const KFaultSampler&) = delete;
    KFaultSampler& operator=(const KFaultSampler&) = delete;
    KFaultSampler(KFaultSampler&&) noexcept = default;
    KFaultSampler& operator=(KFaultSampler&&) noexcept = default;

    [[nodiscard]] uint32_t num_sites() const noexcept { return distribution_->num_sites_; }
    [[nodiscard]] uint32_t k() const noexcept { return distribution_->k_; }

    template <typename RandomDouble>
    [[nodiscard]] std::span<const uint32_t> sample(RandomDouble&& random_double) noexcept {
        const KFaultDistribution& distribution = *distribution_;
        selected_sites_.clear();
        selected_sites_.insert(selected_sites_.end(), distribution.certain_sites_.begin(),
                               distribution.certain_sites_.end());

        if (distribution.uniform_mode_) {
            const uint32_t count = static_cast<uint32_t>(uniform_pool_.size());
            for (uint32_t j = 0; j < distribution.remaining_k_; ++j) {
                const uint32_t remaining = count - j;
                const double draw = random_double();
                assert(draw >= 0.0 && draw < 1.0 && "fault selection draw must be in [0, 1)");
                const uint32_t pick = j + static_cast<uint32_t>(draw * remaining);
                std::swap(uniform_pool_[j], uniform_pool_[pick]);
                swap_targets_[j] = pick;
            }
            for (uint32_t j = 0; j < distribution.remaining_k_; ++j) {
                selected_sites_.push_back(uniform_pool_[j]);
            }
            // Restore the canonical pool so a shot's subset depends only on
            // its own RNG stream, not on which earlier shots used this worker.
            for (uint32_t j = distribution.remaining_k_; j > 0; --j) {
                const uint32_t index = j - 1;
                std::swap(uniform_pool_[index], uniform_pool_[swap_targets_[index]]);
            }
        } else {
            const uint32_t count = static_cast<uint32_t>(distribution.uncertain_sites_.size());
            const uint32_t stride = distribution.remaining_k_ + 1;
            uint32_t needed = distribution.remaining_k_;
            for (uint32_t i = 0; i < count && needed > 0; ++i) {
                double probability = 1.0;
                const uint32_t remaining = count - i;
                if (needed != remaining) {
                    probability =
                        distribution
                            .selection_probabilities_[static_cast<size_t>(i) * stride + needed];
                }
                const double draw = random_double();
                assert(draw >= 0.0 && draw < 1.0 && "fault selection draw must be in [0, 1)");
                if (draw < probability) {
                    selected_sites_.push_back(distribution.uncertain_sites_[i]);
                    --needed;
                }
            }
            assert(needed == 0 && "conditioned fault sampler must select the requested stratum");
        }

        std::sort(selected_sites_.begin(), selected_sites_.end());
        assert(selected_sites_.size() == distribution.k_ &&
               "conditioned fault sampler returned wrong size");
        return selected_sites_;
    }

  private:
    std::shared_ptr<const KFaultDistribution> distribution_;
    std::vector<uint32_t> uniform_pool_;
    std::vector<uint32_t> swap_targets_;
    std::vector<uint32_t> selected_sites_;
};

}  // namespace clifft
