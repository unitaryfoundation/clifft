#pragma once

// Closure and readiness bookkeeping shared by the active-width schedulers.
//
// active_width_schedule_pass.cc's beam search replays HIR ops against a
// DormantSubspace over a ScheduleDependence DAG, choosing at each step which
// ready expanding rotation to execute next and closing out everything else
// deterministically. This header hosts that "which ops are ready" and
// "sweep every ready non-expanding op" mechanism once, plus the theory it
// depends on, so any other caller walking the same DAG -- including a
// research tool outside this library -- cannot drift from what counts as
// ready or expanding.
//
// Definitions, over the dependence DAG (predecessors/successors, ops 0..N-1
// in original index order):
//
//   ready       every predecessor of the op has already executed.
//   expanding   executing the op would raise the active width k: a T_GATE or
//               PHASE_ROTATION whose axis does not commute with every
//               generator of the current DormantSubspace S, or an
//               INSTRUMENT that takes the Activate branch (see
//               active_width_analysis.h's WidthEffect). Every other ready op
//               is non-expanding: any MEASURE, any rotation whose axis
//               commutes with S, and every other op type.
//   closure     starting from a set of executed ops, repeatedly execute the
//               lowest-index ready non-expanding op until none remains.
//
// Closure theorem: some schedule that minimizes peak active width executes
// every ready non-expanding op as soon as it is ready. Sketch: readiness
// only grows as ops execute, a rotation that currently commutes with S stays
// commuting under any op independent of it (an independent op's body cannot
// be one of the directions S changes along, by can_swap's own soundness),
// and no non-expanding op's execution can make another already-ready
// non-expanding op expanding, so delaying a ready non-expanding op can only
// ever leave the peak the same or larger, never smaller. Consequently a
// scheduler only has to branch on which ready expanding op to execute next;
// closure fills in the rest deterministically. An INSTRUMENT is a
// positional barrier under ScheduleDependence (commutation.cc's can_swap
// refuses to reorder anything across one), so every op before it in
// original order is transitively its predecessor and every op after it is
// transitively its successor -- when it is ready, it is therefore the only
// ready op, and it executes regardless of whether it is expanding.
//
// Confluence: the subspace S reached by executing a given set of ops is the
// same regardless of the order they executed in (independent GF(2) updates
// commute), so a scheduling state is fully determined by its set of
// executed ops, not by the sequence that reached it. That is what makes the
// executed-op bitset below a sound identity for deduplicating candidates:
// two different branches that happen to execute the same set of ops are
// provably in the same state.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/schedule_dependence.h"

#include <cstdint>
#include <optional>
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

// One execute()/undo() bracket: the op that executed, and how many
// successors newly became ready as a result. The count indexes into the
// caller's own shared newly-ready log (see SearchFrontier::execute), so
// backtracking never pays for a fresh heap allocation per step the way a
// per-step std::vector<uint32_t> would.
struct UndoStep {
    uint32_t op = 0;
    uint32_t newly_ready_count = 0;
};

// Tracks which ops are executed and which are ready (every predecessor
// executed) as one mutable structure a caller updates via execute()/undo()
// pairs bracketing each recursive or speculative step. Readiness and the
// executed set are both pure functions of the executed-op set alone (see
// the file comment's confluence note), so undo() only has to reverse
// exactly what its matching execute() call did, never recompute anything
// from scratch. Copyable: a caller exploring several independent
// speculative continuations (rather than backtracking one shared instance)
// can clone a frontier instead of using execute()/undo().
//
// Readiness lives in a flat ready_flag_ array rather than a std::set: it
// churns on every execute()/undo() call in a beam or exact search's inner
// loop, and a flag flip is both allocation-free and branch-cheap where a
// tree insert or erase is neither. lowest_ready_hint_ is a safe lower bound
// on the smallest ready op index (never above the true minimum, though it
// can undershoot after a removal until the next scan walks past the gap),
// letting find_ready_non_expanding's ascending scan skip the already-known-
// empty prefix below it while still visiting ready ops in the same
// lowest-index-first order a std::set would and stopping at the first
// non-expanding one -- which is what makes each avoided is_expanding() call
// (a PauliString allocation plus a GF(2) commutation scan) worth avoiding.
class SearchFrontier {
  public:
    explicit SearchFrontier(const ScheduleDependence& dependence);

    [[nodiscard]] bool is_ready(uint32_t op) const { return ready_flag_[op] != 0; }
    [[nodiscard]] uint32_t num_ops() const { return static_cast<uint32_t>(ready_flag_.size()); }
    [[nodiscard]] uint32_t lowest_ready_hint() const { return lowest_ready_hint_; }
    [[nodiscard]] const std::vector<uint64_t>& executed_bits() const { return executed_; }
    [[nodiscard]] size_t executed_count() const { return executed_count_; }

    // Marks `op` executed. Appends each successor that newly became ready to
    // the end of `newly_ready_log` (grown, never cleared here, so a caller
    // threading one log across a whole closure sweep or search amortizes
    // its allocations instead of paying for a fresh vector per call) and
    // returns how many entries it appended -- store this (typically in an
    // UndoStep) to undo exactly this call later.
    uint32_t execute(uint32_t op, std::vector<uint32_t>& newly_ready_log);

    // Reverses exactly an execute() call that appended `newly_ready_count`
    // entries to the end of `newly_ready_log`, popping them back off.
    void undo(uint32_t op, uint32_t newly_ready_count, std::vector<uint32_t>& newly_ready_log);

  private:
    void mark_ready(uint32_t op);
    void mark_not_ready(uint32_t op);

    const ScheduleDependence* dependence_;
    std::vector<uint64_t> executed_;
    std::vector<uint32_t> remaining_preds_;
    std::vector<uint8_t> ready_flag_;
    uint32_t lowest_ready_hint_ = 0;
    size_t executed_count_ = 0;
};

// Undoes every step in `log`, most recent first, against the shared
// `newly_ready_log` every step's SearchFrontier::execute() call appended to.
void undo_all(SearchFrontier& frontier, const std::vector<UndoStep>& log,
              std::vector<uint32_t>& newly_ready_log);

// Lowest-index ready op that is not expanding, or nullopt when every
// currently ready op (if any) is expanding.
[[nodiscard]] std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir,
                                                               const SearchFrontier& frontier,
                                                               const DormantSubspace& subspace);

// Executes every ready non-expanding op, lowest index first, until none is
// ready: the closure step the closure theorem above justifies. Appends
// each executed op, in execution order, to `order` and logs it in `log`
// (backed by `newly_ready_log`, which every logged step's execute() call
// appends to) for the caller to undo later if needed. When `transitions` is
// non-null, each executed op's classification (before/after width and
// effect) is appended to it in the same order, so a caller that needs
// per-op dense-work contributions (the scheduling pass) can accumulate them
// without a second pass over the same ops; a caller that only needs width
// leaves this null.
void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log,
                 std::vector<uint32_t>& newly_ready_log,
                 std::vector<WidthTransition>* transitions = nullptr);

}  // namespace clifft::detail
