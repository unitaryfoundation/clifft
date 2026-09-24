#pragma once

// Searches legal HIR orders to reduce peak active width, then estimated dense
// work. Applies only a strict improvement; ties leave the input unchanged.
// Run after fusion and squeezing, since noise crossings can inhibit later fusion.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/hir_pass.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace clifft {

struct ActiveWidthScheduleOptions {
    // Allow noise crossings with symbolic sign correction.
    bool noise_transparent = true;

    // Partial schedules retained at each search step. Must be positive;
    // one keeps only the best candidate, while larger beams explore alternatives.
    uint32_t beam_width = 8;

    // Search executions per HIR op. At half the budget, retain one parent;
    // after the full budget, choose the lowest-index ready expansion. Ongoing
    // sweeps and replays finish, so these thresholds can be exceeded.
    //
    // This does not bound classification probes or wall time. Zero requests
    // greedy continuation; nullopt disables narrowing. Must be finite and nonnegative.
    std::optional<double> search_budget = 16.0;

    // Delay neutral rotations past independent non-expanding ops to reduce
    // dense work and expose rotation fusion opportunities.
    bool sink_neutral_rotations = true;
};

class ActiveWidthSchedulePass : public HirPass {
  public:
    explicit ActiveWidthSchedulePass(ActiveWidthScheduleOptions options = {});

    void run(HirModule& hir) override;

    // Statistics from the last run() call. All read as zero/false before
    // the first call.
    [[nodiscard]] uint32_t incumbent_peak() const { return incumbent_peak_; }
    [[nodiscard]] uint32_t result_peak() const { return result_peak_; }
    [[nodiscard]] double incumbent_dense_work() const { return incumbent_dense_work_; }
    [[nodiscard]] double result_dense_work() const { return result_dense_work_; }
    [[nodiscard]] bool applied() const { return applied_; }

    // Executions during search, including discarded candidates and survivor
    // replays. Excludes dependence construction, sinking, and final traces.
    [[nodiscard]] size_t swept_ops() const { return swept_ops_; }

    // Closure classification probes, including queries that found an
    // expansion and executed nothing. Both counters reset on run()
    // and remain zero on early exit.
    [[nodiscard]] size_t classification_probes() const { return classification_probes_; }

  private:
    ActiveWidthScheduleOptions options_;
    uint32_t incumbent_peak_ = 0;
    uint32_t result_peak_ = 0;
    double incumbent_dense_work_ = 0.0;
    double result_dense_work_ = 0.0;
    bool applied_ = false;
    size_t swept_ops_ = 0;
    size_t classification_probes_ = 0;
};

}  // namespace clifft
