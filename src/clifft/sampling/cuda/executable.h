#pragma once

#include "clifft/sampling/cuda/device_program.h"
#include "clifft/sampling/plan.h"

#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling::cuda {

// Widest state one thread should own outright; beyond it a cooperative block
// owns the shot, with the coefficients in shared memory when they fit and in
// a per-block global slab otherwise.
inline constexpr uint32_t kThreadPerShotMaxActiveWidth = 4;

class Executable {
  public:
    explicit Executable(const SamplingPlan& plan);

    [[nodiscard]] uint32_t initial_active_width() const { return initial_active_width_; }
    [[nodiscard]] uint32_t peak_active_width() const { return peak_active_width_; }
    [[nodiscard]] uint32_t num_symbols() const { return num_symbols_; }
    [[nodiscard]] uint32_t num_records() const {
        return num_visible_records_ + num_hidden_records_;
    }
    [[nodiscard]] uint32_t num_visible_records() const { return num_visible_records_; }
    [[nodiscard]] uint32_t num_detectors() const { return num_detectors_; }
    [[nodiscard]] uint32_t num_observables() const { return num_observables_; }
    [[nodiscard]] uint32_t num_exp_vals() const { return num_exp_vals_; }
    [[nodiscard]] bool has_postselection() const { return has_postselection_; }

    [[nodiscard]] std::span<const detail::Action> actions() const { return actions_; }
    [[nodiscard]] std::span<const detail::Expression> expressions() const { return expressions_; }
    [[nodiscard]] std::span<const uint32_t> expression_terms() const { return expression_terms_; }
    [[nodiscard]] std::span<const detail::NoiseSite> noise_sites() const { return noise_sites_; }
    [[nodiscard]] std::span<const detail::NoiseOutcome> noise_outcomes() const {
        return noise_outcomes_;
    }

  private:
    uint32_t append_expression(const AffineBool& expression);
    detail::Action lower_action(const PlannedAction& planned);

    uint32_t initial_active_width_ = 0;
    uint32_t peak_active_width_ = 0;
    uint32_t num_symbols_ = 0;
    uint32_t num_visible_records_ = 0;
    uint32_t num_hidden_records_ = 0;
    uint32_t num_detectors_ = 0;
    uint32_t num_observables_ = 0;
    uint32_t num_exp_vals_ = 0;
    bool has_postselection_ = false;
    std::vector<detail::Action> actions_;
    std::vector<detail::Expression> expressions_;
    std::vector<uint32_t> expression_terms_;
    std::vector<detail::NoiseSite> noise_sites_;
    std::vector<detail::NoiseOutcome> noise_outcomes_;
};

}  // namespace clifft::sampling::cuda
