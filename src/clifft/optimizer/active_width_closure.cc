#include "clifft/optimizer/active_width_closure.h"

#include <algorithm>
#include <cassert>

namespace clifft::detail {

namespace {

void bitset_set(std::vector<uint64_t>& bits, uint32_t index) {
    bits[index / 64] |= (uint64_t{1} << (index % 64));
}

void bitset_clear(std::vector<uint64_t>& bits, uint32_t index) {
    bits[index / 64] &= ~(uint64_t{1} << (index % 64));
}

}  // namespace

bool is_expanding(const HirModule& hir, const HeisenbergOp& op, const DormantSubspace& subspace) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
            return !subspace.commutes_with_all(hir.destab_mask(op), hir.stab_mask(op));
        case OpType::INSTRUMENT: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            if (subspace.commutes_with_all(x, z)) {
                return false;  // Classical or Active: non-expanding.
            }
            const InstrumentSite& site =
                hir.instrument_sites.at(static_cast<uint32_t>(op.instrument_site_idx()));
            const bool traps = hir.neglect_instrument_damping ||
                               site.probabilities.p_fire[0] == site.probabilities.p_fire[1];
            return !traps;  // Activate iff it does not trap.
        }
        default:
            return false;
    }
}

bool apply_op(const HirModule& hir, const HeisenbergOp& op, DormantSubspace& subspace) {
    return is_expanding_effect(classify_and_apply(hir, op, subspace).effect);
}

SearchFrontier::SearchFrontier(const ScheduleDependence& dependence)
    : dependence_(&dependence),
      executed_((dependence.num_ops() + 63) / 64, 0),
      remaining_preds_(dependence.num_ops()),
      ready_flag_(dependence.num_ops(), 0) {
    for (uint32_t op = 0; op < dependence.num_ops(); ++op) {
        remaining_preds_[op] = static_cast<uint32_t>(dependence.predecessors(op).size());
        if (remaining_preds_[op] == 0) {
            mark_ready(op);
        }
    }
}

void SearchFrontier::mark_ready(uint32_t op) {
    ready_flag_[op] = 1;
    lowest_ready_hint_ = std::min(lowest_ready_hint_, op);
}

void SearchFrontier::mark_not_ready(uint32_t op) {
    ready_flag_[op] = 0;
    // Only a cheap, exact-match bump: the op just removed was the hint's own
    // witness, so the hint must move at least past it, but the true new
    // minimum among whatever remains ready is not known without a scan --
    // find_ready_non_expanding's own scan discovers it lazily instead.
    if (op == lowest_ready_hint_) {
        ++lowest_ready_hint_;
    }
}

uint32_t SearchFrontier::execute(uint32_t op, std::vector<uint32_t>& newly_ready_log) {
    assert(is_ready(op) && "execute() called on a non-ready op");
    mark_not_ready(op);
    bitset_set(executed_, op);
    ++executed_count_;

    uint32_t count = 0;
    for (uint32_t succ : dependence_->successors(op)) {
        if (--remaining_preds_[succ] == 0) {
            mark_ready(succ);
            newly_ready_log.push_back(succ);
            ++count;
        }
    }
    return count;
}

void SearchFrontier::undo(uint32_t op, uint32_t newly_ready_count,
                          std::vector<uint32_t>& newly_ready_log) {
    for (uint32_t succ : dependence_->successors(op)) {
        ++remaining_preds_[succ];
    }

    assert(newly_ready_log.size() >= newly_ready_count &&
           "undo() asked to reverse more newly-ready entries than the log holds");
    for (uint32_t i = 0; i < newly_ready_count; ++i) {
        mark_not_ready(newly_ready_log.back());
        newly_ready_log.pop_back();
    }

    bitset_clear(executed_, op);
    --executed_count_;
    // op is ready again, and may be lower than the current hint (it was
    // ready before this undo's matching execute() call raised the hint past
    // it), so this goes through mark_ready rather than a raw flag set.
    mark_ready(op);
}

void undo_all(SearchFrontier& frontier, const std::vector<UndoStep>& log,
              std::vector<uint32_t>& newly_ready_log) {
    for (auto it = log.rbegin(); it != log.rend(); ++it) {
        frontier.undo(it->op, it->newly_ready_count, newly_ready_log);
    }
}

std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir,
                                                 const SearchFrontier& frontier,
                                                 const DormantSubspace& subspace) {
    // Ascending scan starting from the hint (a safe lower bound, never above
    // the true minimum ready index): a cheap is_ready() check skips every
    // not-ready index, so is_expanding() only ever runs on an actual ready
    // op, in the same lowest-index-first order a sorted container would
    // visit them, stopping at the first non-expanding one.
    for (uint32_t op = frontier.lowest_ready_hint(); op < frontier.num_ops(); ++op) {
        if (!frontier.is_ready(op)) {
            continue;
        }
        if (!is_expanding(hir, hir.ops[op], subspace)) {
            return op;
        }
    }
    return std::nullopt;
}

void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log,
                 std::vector<uint32_t>& newly_ready_log,
                 std::vector<WidthTransition>* transitions) {
    while (const std::optional<uint32_t> op = find_ready_non_expanding(hir, frontier, subspace)) {
        const uint32_t newly_ready_count = frontier.execute(*op, newly_ready_log);
        log.push_back(UndoStep{*op, newly_ready_count});
        order.push_back(*op);
        const WidthTransition transition = classify_and_apply(hir, hir.ops[*op], subspace);
        assert(!is_expanding_effect(transition.effect) &&
               "find_ready_non_expanding chose an op classify_and_apply treats as expanding");
        if (transitions != nullptr) {
            transitions->push_back(transition);
        }
    }
}

}  // namespace clifft::detail
