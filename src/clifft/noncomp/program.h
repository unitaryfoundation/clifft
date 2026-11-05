#pragma once

#include "clifft/noncomp/fixed.h"
#include "clifft/noncomp/preparation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace clifft {

enum class NonComputationalExecution { Auto, Fixed, Trajectory, Online };

std::string_view noncomputational_execution_name(NonComputationalExecution execution) noexcept;

std::string_view fixed_rejection_name(FixedNonComputationalReason reason);

namespace detail {
struct NonComputationalProgramData;
}

// Immutable owned preparation. Mutable workers and RNG belong to each sample call.
class NonComputationalProgram {
  public:
    NonComputationalProgram(const Circuit& circuit, const NonComputationalModel& model,
                            NonComputationalExecution execution = NonComputationalExecution::Auto,
                            NonComputationalRequirements requirements = {.final_status = false});

    [[nodiscard]] NonComputationalExecution strategy() const;
    [[nodiscard]] bool returns_final_status() const;
    [[nodiscard]] const std::optional<FixedNonComputationalRejection>& rejection() const;
    [[nodiscard]] std::string explain() const;

    NonComputationalSample sample(uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
                                  uint32_t threads = 1,
                                  std::optional<uint32_t> batch_size = std::nullopt,
                                  bool collect_stats = false,
                                  std::optional<double> tuning_budget_seconds = std::nullopt) const;

  private:
    std::shared_ptr<const detail::NonComputationalProgramData> data_;
};

}  // namespace clifft
