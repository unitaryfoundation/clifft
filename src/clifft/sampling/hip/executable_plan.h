#pragma once

#include "clifft/sampling/hip/device_program.h"
#include "clifft/sampling/plan.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace clifft::sampling::hip {

// Auto selects from the active width, precision, and device memory limits.
// Explicit tiers are rejected if the plan does not fit.
enum class ExecutionTier : uint8_t {
    Auto,
    ThreadPerShot,  // one thread per shot, global coefficients
    BlockShared,    // one block per shot, shared coefficients
    BlockGlobal,    // one block per shot, global coefficients
};

[[nodiscard]] constexpr const char* tier_name(ExecutionTier tier) {
    switch (tier) {
        case ExecutionTier::Auto:
            return "auto";
        case ExecutionTier::ThreadPerShot:
            return "thread_per_shot";
        case ExecutionTier::BlockShared:
            return "block_shared";
        case ExecutionTier::BlockGlobal:
            return "block_global";
    }
    return "unknown";
}

inline constexpr uint32_t kThreadPerShotMaxActiveWidth = 4;

// Device memory can impose a lower limit.
inline constexpr uint32_t kMaxSupportedActiveWidth = 30;

[[nodiscard]] constexpr uint64_t coefficient_bytes_per_shot(uint32_t peak_active_width,
                                                            uint64_t element_bytes) {
    return detail::coefficient_elements_per_shot(peak_active_width) * element_bytes;
}

[[nodiscard]] constexpr ExecutionTier select_execution_tier(uint32_t peak_active_width,
                                                            uint64_t element_bytes,
                                                            uint64_t lds_bytes_per_workgroup) {
    if (peak_active_width <= kThreadPerShotMaxActiveWidth) {
        return ExecutionTier::ThreadPerShot;
    }
    // Coefficients share LDS with the kernel's static reduction scratch.
    const uint64_t usable = lds_bytes_per_workgroup > detail::kCooperativeReductionBytes
                                ? lds_bytes_per_workgroup - detail::kCooperativeReductionBytes
                                : 0;
    if (coefficient_bytes_per_shot(peak_active_width, element_bytes) <= usable) {
        return ExecutionTier::BlockShared;
    }
    return ExecutionTier::BlockGlobal;
}

// The automatic thread-per-shot cutoff does not restrict forced tiers.
// Only BlockShared requires the state to fit per-block shared memory.
[[nodiscard]] constexpr bool tier_supports(ExecutionTier tier, uint32_t peak_active_width,
                                           uint64_t element_bytes,
                                           uint64_t lds_bytes_per_workgroup) {
    if (tier != ExecutionTier::BlockShared) {
        return true;
    }
    const uint64_t usable = lds_bytes_per_workgroup > detail::kCooperativeReductionBytes
                                ? lds_bytes_per_workgroup - detail::kCooperativeReductionBytes
                                : 0;
    return coefficient_bytes_per_shot(peak_active_width, element_bytes) <= usable;
}

class ExecutablePlan {
  public:
    explicit ExecutablePlan(const SamplingPlan& plan);

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
    [[nodiscard]] uint32_t num_actions() const { return static_cast<uint32_t>(actions_.size()); }
    [[nodiscard]] size_t packed_bytes() const;
    [[nodiscard]] std::string inspect() const;

    [[nodiscard]] std::span<const detail::Action> actions() const { return actions_; }
    [[nodiscard]] std::span<const detail::Expression> expressions() const { return expressions_; }
    [[nodiscard]] std::span<const uint32_t> expression_terms() const { return expression_terms_; }
    [[nodiscard]] std::span<const detail::NoiseSite> noise_sites() const { return noise_sites_; }
    [[nodiscard]] std::span<const detail::NoiseOutcome> noise_outcomes() const {
        return noise_outcomes_;
    }

  private:
    uint32_t append_expression(const AffineBool& expression);
    uint32_t append_record_parity(const RecordParity& parity);
    void lower_observable_value(detail::Action& action, const ObservableValue& value);
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

}  // namespace clifft::sampling::hip
