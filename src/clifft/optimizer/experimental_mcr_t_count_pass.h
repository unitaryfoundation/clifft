#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

/// Experimental bounded MCR pass: scans HIR windows for four-gate
/// multiplicative commutator relation patterns that unlock local
/// same-axis T fusion.
///
/// Finds four-gate MCR patterns inside contiguous T-gate runs, swaps the
/// commuting pairs as a unit, then applies only the local same-axis T fusion
/// made reachable by that swap. Disabled by default.
class ExperimentalMcrTCountPass : public HirPass {
  public:
    void run(HirModule& hir) override;

    /// Statistics from the last run.
    [[nodiscard]] size_t lookahead_cap() const { return kLookaheadCap; }
    /// Number of barrier-bounded T windows scanned across all rewrite rounds.
    [[nodiscard]] size_t window_scans() const { return window_scans_; }
    /// Number of scanned windows whose T-count exceeded lookahead_cap().
    [[nodiscard]] size_t window_scans_over_lookahead_cap() const {
        return window_scans_over_lookahead_cap_;
    }
    [[nodiscard]] size_t quadruples_found() const { return quadruples_found_; }
    [[nodiscard]] size_t swaps_applied() const { return swaps_applied_; }
    [[nodiscard]] size_t merges() const { return merges_; }
    [[nodiscard]] size_t t_removed() const { return t_removed_; }

  private:
    static constexpr size_t kLookaheadCap = 16;

    size_t window_scans_ = 0;
    size_t window_scans_over_lookahead_cap_ = 0;
    size_t quadruples_found_ = 0;
    size_t swaps_applied_ = 0;
    size_t merges_ = 0;
    size_t t_removed_ = 0;
};

}  // namespace clifft
