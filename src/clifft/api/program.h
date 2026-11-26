#pragma once

// Owns a circuit's compiled ExecutablePlan (when below the dense active-width
// limit) together with optional reconstructed sampling plans with or without
// measurement records, plus the sampling preparation derived from them on
// first use. Copies share both, including bounded idle shared-worker storage
// checked out exclusively by each call. State queries, exact probabilities,
// replay, and fixed-fault sampling require the original plan. On request the
// original plan carries a runtime Clifford frame, which keeps signed T regions
// at the active width of their fixed-sign core.

#include "clifft/optimizer/global_reconstruction.h"
#include "clifft/sampling/executable_plan.h"

#include <cstdint>
#include <memory>
#include <optional>
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

    // An unset reconstruction selects automatically. With runtime_frame, the
    // circuit is also planned on a runtime Clifford frame. HIR passes split
    // the regions a frame reduces, so that plan starts from a fresh trace of
    // source when the passes preserve the sampling distribution, and from
    // original_hir otherwise. When the plan carries a frame it becomes the
    // original plan.
    [[nodiscard]] static Program compile(const Circuit& source, const HirModule& original_hir,
                                         std::optional<bool> reconstruction = std::nullopt,
                                         bool passes_preserve_sampling = true,
                                         const ReconstructionLimits& limits = {},
                                         bool runtime_frame = false);

    [[nodiscard]] bool has_original() const noexcept;
    [[nodiscard]] bool has_sampling_plan(bool keep_records) const noexcept;
    [[nodiscard]] const sampling::ExecutablePlan& original() const;
    [[nodiscard]] const sampling::ExecutablePlan& sampling_plan(bool keep_records) const;
    [[nodiscard]] bool uses_reconstruction(bool keep_records) const noexcept;
    [[nodiscard]] const std::string& reconstruction_fallback_reason(bool keep_records) const;
    // Why the original plan carries no runtime Clifford frame; empty when it does.
    [[nodiscard]] const std::string& runtime_frame_fallback_reason() const noexcept;

    [[nodiscard]] uint32_t num_qubits() const noexcept;
    [[nodiscard]] uint32_t num_visible_records() const noexcept;
    [[nodiscard]] uint32_t num_hidden_records() const noexcept;
    [[nodiscard]] uint32_t num_detectors() const noexcept;
    [[nodiscard]] uint32_t num_observables() const noexcept;
    [[nodiscard]] uint32_t num_exp_vals() const noexcept;
    [[nodiscard]] bool has_postselection() const noexcept;

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
