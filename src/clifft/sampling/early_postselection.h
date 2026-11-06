#pragma once

#include "clifft/sampling/executable_plan.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling {

// Prepares noise-only entrance constraints and schedules selected detectors
// immediately after their contributing record values become available.
class EarlyPostselection {
  public:
    explicit EarlyPostselection(const ExecutablePlan& plan, bool enabled = true);
    [[nodiscard]] size_t num_checks() const noexcept { return checks_.size(); }
    [[nodiscard]] std::span<const size_t> action_order() const noexcept { return action_order_; }

  private:
    friend class BatchClassicalState;
    friend class SharingExecutor;
    struct Check {
        bool constant = false;
        std::vector<uint32_t> registers;
    };
    const ExecutablePlan* plan_;
    std::vector<Check> checks_;
    std::vector<size_t> action_order_;
};

}  // namespace clifft::sampling
