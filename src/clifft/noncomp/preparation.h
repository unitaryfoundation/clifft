#pragma once

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/sample.h"

#include <cstdint>
#include <optional>

namespace clifft {

struct NonComputationalRequirements {
    bool final_status = true;
    std::optional<uint32_t> max_active_width = std::nullopt;
};

enum class NonComputationalStrategy { Trajectory };

struct NonComputationalPreparationReport {
    NonComputationalStrategy strategy = NonComputationalStrategy::Trajectory;
};

// Internal preparation boundary. Owning the expanded circuit and model lets
// callers reuse a request after its inputs have changed or gone out of scope.
// Only execution calls own RNG, workers, and compiled continuations.
class PreparedNonComputationalRequest {
  public:
    PreparedNonComputationalRequest(const Circuit& circuit, const NonComputationalModel& model,
                                    NonComputationalRequirements requirements = {});

    [[nodiscard]] const Circuit& circuit() const { return circuit_; }
    [[nodiscard]] const NonComputationalModel& model() const { return model_; }
    [[nodiscard]] const NonComputationalRequirements& requirements() const { return requirements_; }
    [[nodiscard]] NonComputationalPreparationReport report() const { return {}; }

    NonComputationalSample sample(uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
                                  uint32_t threads = 1) const;

  private:
    NonComputationalModel model_;
    Circuit circuit_;
    NonComputationalRequirements requirements_;
};

}  // namespace clifft
