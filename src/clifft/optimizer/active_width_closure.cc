#include "clifft/optimizer/active_width_closure.h"

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
            return !subspace.commutes_with_all(pauli_body(hir, op));
        case OpType::INSTRUMENT: {
            const PauliString body = pauli_body(hir, op);
            if (subspace.commutes_with_all(body)) {
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
      remaining_preds_(dependence.num_ops()) {
    for (uint32_t op = 0; op < dependence.num_ops(); ++op) {
        remaining_preds_[op] = static_cast<uint32_t>(dependence.predecessors(op).size());
        if (remaining_preds_[op] == 0) {
            ready_.insert(op);
        }
    }
}

std::vector<uint32_t> SearchFrontier::execute(uint32_t op) {
    assert(ready_.contains(op) && "execute() called on a non-ready op");
    ready_.erase(op);
    bitset_set(executed_, op);
    ++executed_count_;
    std::vector<uint32_t> newly_ready;
    for (uint32_t succ : dependence_->successors(op)) {
        if (--remaining_preds_[succ] == 0) {
            ready_.insert(succ);
            newly_ready.push_back(succ);
        }
    }
    return newly_ready;
}

void SearchFrontier::undo(uint32_t op, const std::vector<uint32_t>& newly_ready) {
    for (uint32_t succ : dependence_->successors(op)) {
        ++remaining_preds_[succ];
    }
    for (uint32_t r : newly_ready) {
        ready_.erase(r);
    }
    bitset_clear(executed_, op);
    --executed_count_;
    ready_.insert(op);
}

void undo_all(SearchFrontier& frontier, const std::vector<UndoStep>& log) {
    for (auto it = log.rbegin(); it != log.rend(); ++it) {
        frontier.undo(it->first, it->second);
    }
}

std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir,
                                                 const SearchFrontier& frontier,
                                                 const DormantSubspace& subspace) {
    for (uint32_t op : frontier.ready()) {
        if (!is_expanding(hir, hir.ops[op], subspace)) {
            return op;
        }
    }
    return std::nullopt;
}

void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log,
                 std::vector<WidthTransition>* transitions) {
    while (const std::optional<uint32_t> op = find_ready_non_expanding(hir, frontier, subspace)) {
        log.emplace_back(*op, frontier.execute(*op));
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
