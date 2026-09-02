#pragma once

// Closure and readiness bookkeeping shared by the active-width schedulers.
//
// active_width_search.cc's threshold-feasibility search and
// active_width_schedule_pass.cc's beam search both replay HIR ops against a
// DormantSubspace over a ScheduleDependence DAG, and both need the same
// "which ops are ready" bookkeeping and the same "sweep every ready
// non-expanding op" step. This header hosts that shared mechanism once so
// the two callers cannot drift apart on what counts as ready or expanding;
// see active_width_search.h for the closure theorem and confluence
// argument the mechanism itself depends on, which this header does not
// restate.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/schedule_dependence.h"

#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace clifft::detail {

// True when executing `op` against `subspace` would raise the active width:
// a T_GATE/PHASE_ROTATION whose axis does not commute with every generator
// of S, or an INSTRUMENT that takes the Activate branch (see
// active_width_analysis.h's WidthEffect). Pure query, no mutation, so a
// caller can test every ready op before committing to one.
[[nodiscard]] bool is_expanding(const HirModule& hir, const HeisenbergOp& op,
                                const DormantSubspace& subspace);

// Applies `op`'s effect to `subspace` and reports whether it was expanding.
// A thin wrapper over classify_and_apply for callers that only need the
// bool; safe to call on any ready op regardless of its classification.
bool apply_op(const HirModule& hir, const HeisenbergOp& op, DormantSubspace& subspace);

// Tracks which ops are executed and which are ready (every predecessor
// executed) as one mutable structure a caller updates via execute()/undo()
// pairs bracketing each recursive or speculative step. Readiness and the
// executed set are both pure functions of the executed-op set alone (see
// active_width_search.h's confluence note), so undo() only has to reverse
// exactly what its matching execute() call did, never recompute anything
// from scratch. Copyable: a caller exploring several independent
// speculative continuations (rather than backtracking one shared instance)
// can clone a frontier instead of using execute()/undo().
class SearchFrontier {
  public:
    explicit SearchFrontier(const ScheduleDependence& dependence);

    [[nodiscard]] const std::set<uint32_t>& ready() const { return ready_; }
    [[nodiscard]] const std::vector<uint64_t>& executed_bits() const { return executed_; }
    [[nodiscard]] size_t executed_count() const { return executed_count_; }

    // Marks `op` executed and returns the successors that newly became
    // ready as a result, which undo() needs to reverse this call exactly.
    std::vector<uint32_t> execute(uint32_t op);

    // Reverses exactly the execute(op) call that returned `newly_ready`.
    void undo(uint32_t op, const std::vector<uint32_t>& newly_ready);

  private:
    const ScheduleDependence* dependence_;
    std::vector<uint64_t> executed_;
    std::vector<uint32_t> remaining_preds_;
    std::set<uint32_t> ready_;
    size_t executed_count_ = 0;
};

// An executed op paired with the successors execute() found newly ready:
// everything undo() needs to reverse that one call.
using UndoStep = std::pair<uint32_t, std::vector<uint32_t>>;

void undo_all(SearchFrontier& frontier, const std::vector<UndoStep>& log);

// Lowest-index ready op that is not expanding, or nullopt when every
// currently ready op (if any) is expanding.
[[nodiscard]] std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir,
                                                               const SearchFrontier& frontier,
                                                               const DormantSubspace& subspace);

// Executes every ready non-expanding op, lowest index first, until none is
// ready: the closure step both callers' correctness depends on. Appends
// each executed op, in execution order, to `order` and logs it in `log` for
// the caller to undo later if needed. When `transitions` is non-null, each
// executed op's classification (before/after width and effect) is appended
// to it in the same order, so a caller that needs per-op dense-work
// contributions (the scheduling pass) can accumulate them without a second
// pass over the same ops; the exact search leaves this null since threshold
// feasibility only needs width.
void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log,
                 std::vector<WidthTransition>* transitions = nullptr);

}  // namespace clifft::detail
