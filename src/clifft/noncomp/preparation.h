#pragma once

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/resolution.h"
#include "clifft/noncomp/sample.h"

#include <cstdint>
#include <optional>

namespace clifft {

struct NonComputationalRequirements {
    bool final_status = true;
    std::optional<uint32_t> max_active_width = std::nullopt;
};

// Internal preparation boundary. Owning the expanded circuit and model lets
// callers reuse a request after its inputs have changed or gone out of scope.
// Only execution calls own RNG, workers, and compiled continuations.
class PreparedNonComputationalRequest {
  public:
    PreparedNonComputationalRequest(const Circuit& circuit, const NonComputationalModel& model,
                                    NonComputationalRequirements requirements = {});

    [[nodiscard]] const ResolvedNonComputationalCircuit& resolved_circuit() const {
        return resolved_circuit_;
    }
    [[nodiscard]] const NonComputationalModel& model() const { return model_; }
    [[nodiscard]] const NonComputationalRequirements& requirements() const { return requirements_; }

    NonComputationalSample sample(uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
                                  uint32_t threads = 1) const;

  private:
    NonComputationalModel model_;
    ResolvedNonComputationalCircuit resolved_circuit_;
    NonComputationalRequirements requirements_;
};

}  // namespace clifft
