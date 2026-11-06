#pragma once

#include "clifft/sampling/executable_plan.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft::sampling {

// Necessary selected-detector constraints that depend only on presampled noise.
// Each check reads the initial affine registers, before branch assignments.
class EarlyPostselection {
  public:
    explicit EarlyPostselection(const ExecutablePlan& plan, bool enabled = true);
    [[nodiscard]] size_t num_checks() const noexcept { return checks_.size(); }

  private:
    friend class BatchClassicalState;
    friend class SharingExecutor;
    struct Check {
        bool constant = false;
        std::vector<uint32_t> registers;
    };
    const ExecutablePlan* plan_;
    std::vector<Check> checks_;
};

}  // namespace clifft::sampling
