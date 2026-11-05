#pragma once

// Owns a circuit's compiled ExecutablePlan together with optional reconstructed
// sampling plans for keep_records=true (records, detectors, and observables)
// and keep_records=false (detectors and observables only). State queries, exact
// probabilities, replay, and fixed-fault sampling always use original().

#include "clifft/optimizer/global_reconstruction.h"
#include "clifft/sampling/executable_plan.h"

#include <array>
#include <optional>
#include <string>

namespace clifft {

struct HirModule;

class Program {
  public:
    explicit Program(sampling::ExecutablePlan original);

    [[nodiscard]] static Program compile(const Circuit& source, const HirModule& original_hir,
                                         bool reconstruction = false,
                                         bool passes_preserve_sampling = true,
                                         const ReconstructionLimits& limits = {});

    [[nodiscard]] const sampling::ExecutablePlan& original() const noexcept { return original_; }
    [[nodiscard]] const sampling::ExecutablePlan& sampling_plan(bool keep_records) const noexcept;
    [[nodiscard]] bool uses_reconstruction(bool keep_records) const noexcept;
    [[nodiscard]] const std::string& reconstruction_fallback_reason(bool keep_records) const;

  private:
    struct SamplingVariant {
        std::optional<sampling::ExecutablePlan> plan;
        std::string fallback_reason = "reconstruction is disabled";
    };

    sampling::ExecutablePlan original_;
    // Prepared once during compilation, then immutable for concurrent sampling.
    std::array<SamplingVariant, 2> variants_;
};

}  // namespace clifft
