#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

/// Experimental bounded MCR pass: scans HIR T windows for exact four-gate
/// block swaps that unlock lower T-count rewrites after peephole cleanup.
///
/// Enumerates bounded four-gate block swaps inside contiguous T-gate runs,
/// validates each candidate against the full local span up to global phase,
/// then accepts only rewrites that reduce the post-peephole T count.
/// Disabled by default.
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
    [[nodiscard]] size_t candidates_considered() const { return candidates_considered_; }
    [[nodiscard]] size_t merge_potential_rejects() const { return merge_potential_rejects_; }
    [[nodiscard]] size_t equivalence_checks() const { return equivalence_checks_; }
    [[nodiscard]] size_t quadruples_found() const { return quadruples_found_; }
    [[nodiscard]] size_t swaps_applied() const { return swaps_applied_; }
    [[nodiscard]] size_t merges() const { return merges_; }
    [[nodiscard]] size_t t_removed() const { return t_removed_; }

  private:
    static constexpr size_t kLookaheadCap = 16;

    size_t window_scans_ = 0;
    size_t window_scans_over_lookahead_cap_ = 0;
    size_t candidates_considered_ = 0;
    size_t merge_potential_rejects_ = 0;
    size_t equivalence_checks_ = 0;
    size_t quadruples_found_ = 0;
    size_t swaps_applied_ = 0;
    size_t merges_ = 0;
    size_t t_removed_ = 0;
};

}  // namespace clifft
