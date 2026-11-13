#pragma once

// Owns a circuit's compiled ExecutablePlan together with optional reconstructed
// sampling plans with or without measurement records, plus the sampling
// preparation derived from them on first use. Copies share both. Mutable
// workers belong to each sampler or sampling call. State queries, exact
// probabilities, replay, fixed-fault sampling, and plan inspection use the
// compiled plans.

#include "clifft/optimizer/global_reconstruction.h"
#include "clifft/sampling/executable_plan.h"

#include <memory>
#include <string>

namespace clifft {

struct HirModule;
class SamplingPreparation;

// Selects execution requirements independently of the retained output rows.
struct SamplingPreparationOptions {
    bool early_postselection = true;
    bool fixed_faults = false;
    bool survivor_counts = false;
};

class Program {
  public:
    explicit Program(sampling::ExecutablePlan original);

    [[nodiscard]] static Program compile(const Circuit& source, const HirModule& original_hir,
                                         bool reconstruction = false,
                                         bool passes_preserve_sampling = true,
                                         const ReconstructionLimits& limits = {});

    [[nodiscard]] const sampling::ExecutablePlan& original() const noexcept;
    [[nodiscard]] const sampling::ExecutablePlan& sampling_plan(bool keep_records) const noexcept;
    [[nodiscard]] bool uses_reconstruction(bool keep_records) const noexcept;
    [[nodiscard]] const std::string& reconstruction_fallback_reason(bool keep_records) const;

    // Output-specific preparation, shared by later calls and Program copies.
    // Fixed-fault sampling uses the original plan to preserve fault identities.
    // Survivor summaries retain observables even without their per-shot rows.
    // Bit packing affects collection only and shares the same preparation.
    [[nodiscard]] const SamplingPreparation& sampling_preparation(
        sampling::SamplingOutputs outputs, SamplingPreparationOptions options = {}) const;

  private:
    struct Data;
    explicit Program(std::shared_ptr<const Data> data);

    std::shared_ptr<const Data> data_;
};

}  // namespace clifft
