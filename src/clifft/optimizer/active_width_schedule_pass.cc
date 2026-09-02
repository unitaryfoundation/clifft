#include "clifft/optimizer/active_width_schedule_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/active_width_closure.h"
#include "clifft/optimizer/active_width_search.h"
#include "clifft/optimizer/schedule_dependence.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace clifft {

namespace {

// ---------------------------------------------------------------------------
// Beam state: one partial schedule, closed (every ready non-expanding op
// already executed), tracking everything needed to score and extend it
// further without replaying the ops executed so far.
// ---------------------------------------------------------------------------

struct BeamState {
    BeamState(detail::SearchFrontier frontier_in, DormantSubspace subspace_in)
        : frontier(std::move(frontier_in)), subspace(std::move(subspace_in)) {}

    detail::SearchFrontier frontier;
    DormantSubspace subspace;
    uint32_t peak = 0;
    double dense_work = 0.0;
    std::vector<uint32_t> order;
};

void absorb_closure_transitions(BeamState& state, const std::vector<WidthTransition>& transitions) {
    for (const WidthTransition& transition : transitions) {
        state.peak = std::max(state.peak, transition.after);
        state.dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
}

BeamState make_initial_beam_state(const HirModule& hir, const ScheduleDependence& dependence) {
    BeamState state(detail::SearchFrontier(dependence), DormantSubspace(hir.num_qubits));
    std::vector<detail::UndoStep> discarded_log;
    std::vector<WidthTransition> transitions;
    detail::run_closure(hir, state.frontier, state.subspace, state.order, discarded_log,
                        &transitions);
    absorb_closure_transitions(state, transitions);
    return state;
}

// One child of a beam state: commit to `first_op` (a ready expanding op)
// then close. Scored by (peak, width_after_closure, dense_work, first_op)
// ascending, matching search_width_schedule's own candidate ranking so the
// beam and the exact repair step agree on which move looks best whenever
// both consider the same choice.
struct BeamCandidate {
    BeamState state;
    uint32_t width_after_closure = 0;
    uint32_t first_op = 0;
};

std::vector<uint32_t> ready_ops_snapshot(const detail::SearchFrontier& frontier) {
    return std::vector<uint32_t>(frontier.ready().begin(), frontier.ready().end());
}

// Expands `parent` into one child per ready expanding op: each child is an
// independent clone (never a mutation of `parent`), since the beam keeps
// several partial schedules alive at once rather than backtracking one
// shared frontier the way the exact search's DFS does.
std::vector<BeamCandidate> expand_state(const HirModule& hir, const BeamState& parent) {
    std::vector<BeamCandidate> children;
    for (uint32_t op : ready_ops_snapshot(parent.frontier)) {
        if (!detail::is_expanding(hir, hir.ops[op], parent.subspace)) {
            continue;
        }

        BeamState child_state(parent.frontier, parent.subspace);
        child_state.peak = parent.peak;
        child_state.dense_work = parent.dense_work;
        child_state.order = parent.order;

        child_state.frontier.execute(op);
        child_state.order.push_back(op);
        const WidthTransition first_transition =
            classify_and_apply(hir, hir.ops[op], child_state.subspace);
        assert(is_expanding_effect(first_transition.effect) &&
               "expand_state chose a ready op is_expanding did not classify as expanding");
        child_state.peak = std::max(child_state.peak, first_transition.after);
        child_state.dense_work += detail::dense_work_contribution(
            first_transition.effect, first_transition.before, first_transition.after);

        std::vector<detail::UndoStep> discarded_log;
        std::vector<WidthTransition> swept;
        detail::run_closure(hir, child_state.frontier, child_state.subspace, child_state.order,
                            discarded_log, &swept);
        absorb_closure_transitions(child_state, swept);

        const uint32_t width_after_closure = child_state.subspace.active_width();
        children.push_back(BeamCandidate{std::move(child_state), width_after_closure, op});
    }
    return children;
}

// Among completed (fully executed) beam states, the one with the smallest
// (peak, dense_work), tied by comparing the executed order itself so the
// choice is fully deterministic even when two structurally different
// schedules happen to cost exactly the same.
const BeamState* pick_best_completed(const std::vector<BeamState>& completed) {
    const BeamState* best = nullptr;
    for (const BeamState& state : completed) {
        if (best == nullptr) {
            best = &state;
            continue;
        }
        if (state.peak != best->peak) {
            if (state.peak < best->peak) {
                best = &state;
            }
            continue;
        }
        if (state.dense_work != best->dense_work) {
            if (state.dense_work < best->dense_work) {
                best = &state;
            }
            continue;
        }
        if (state.order < best->order) {
            best = &state;
        }
    }
    return best;
}

// Beam search over the closure/readiness machinery active_width_closure.h
// shares with the exact search. beam_width == 1 degenerates to the greedy
// closure scheduler: at every step, take whichever single ready expanding
// op's own closure sweep scores best. Falls back to the identity order if
// no beam_width (e.g. a caller-supplied 0) ever lets a schedule complete;
// the pass's own "never worse" check downstream makes that fallback safe.
std::vector<uint32_t> run_beam_search(const HirModule& hir, const ScheduleDependence& dependence,
                                      uint32_t beam_width) {
    std::vector<BeamState> beam;
    beam.push_back(make_initial_beam_state(hir, dependence));

    std::vector<BeamState> completed;
    while (!beam.empty()) {
        std::vector<BeamCandidate> generation;
        for (BeamState& parent : beam) {
            std::vector<BeamCandidate> children = expand_state(hir, parent);
            if (children.empty()) {
                // Closure already consumed every ready non-expanding op, so
                // no ready expanding candidates left means no ready ops
                // left at all, which is only possible once every op has
                // executed (see active_width_search.h's confluence note).
                assert(
                    parent.frontier.executed_count() == dependence.num_ops() &&
                    "a closed beam state with no ready expanding op must have executed every op");
                completed.push_back(std::move(parent));
                continue;
            }
            for (BeamCandidate& child : children) {
                generation.push_back(std::move(child));
            }
        }

        if (generation.empty()) {
            break;
        }

        // Deduplicate by executed-op bitset: by confluence, the same set of
        // executed ops always reaches the same subspace and width, so only
        // the smaller-dense-work copy is worth keeping.
        std::ranges::sort(generation, [](const BeamCandidate& a, const BeamCandidate& b) {
            return a.state.frontier.executed_bits() < b.state.frontier.executed_bits();
        });
        std::vector<BeamCandidate> deduped;
        for (BeamCandidate& candidate : generation) {
            if (!deduped.empty() && deduped.back().state.frontier.executed_bits() ==
                                        candidate.state.frontier.executed_bits()) {
                if (candidate.state.dense_work < deduped.back().state.dense_work) {
                    deduped.back() = std::move(candidate);
                }
                continue;
            }
            deduped.push_back(std::move(candidate));
        }

        std::ranges::sort(deduped, [](const BeamCandidate& a, const BeamCandidate& b) {
            if (a.state.peak != b.state.peak) {
                return a.state.peak < b.state.peak;
            }
            if (a.width_after_closure != b.width_after_closure) {
                return a.width_after_closure < b.width_after_closure;
            }
            if (a.state.dense_work != b.state.dense_work) {
                return a.state.dense_work < b.state.dense_work;
            }
            return a.first_op < b.first_op;
        });

        beam.clear();
        for (size_t i = 0; i < deduped.size() && i < beam_width; ++i) {
            beam.push_back(std::move(deduped[i].state));
        }
    }

    const BeamState* best = pick_best_completed(completed);
    if (best == nullptr) {
        std::vector<uint32_t> identity(hir.ops.size());
        std::iota(identity.begin(), identity.end(), uint32_t{0});
        return identity;
    }
    return best->order;
}

// ---------------------------------------------------------------------------
// Exact repair: an optional, bounded polish of the beam's own answer.
// ---------------------------------------------------------------------------

// Applies `beam_order` to a copy of `hir` and asks the exact search to
// improve on it within `exact_node_budget` nodes. Adopts the exact search's
// order, composed with `beam_order`, only when it certifies or witnesses a
// strictly lower peak than the beam order's own; otherwise returns
// `beam_order` unchanged. This is a pure improver: the pass never reports
// search_width_schedule's certificate, only whatever order it settles on.
std::vector<uint32_t> maybe_exact_repair(const HirModule& hir, const ScheduleDependence& dependence,
                                         std::vector<uint32_t> beam_order, bool noise_transparent,
                                         uint64_t exact_node_budget) {
    HirModule repaired = hir;
    apply_schedule(repaired, dependence, beam_order);

    ScheduleDependenceOptions repaired_options;
    repaired_options.noise_transparent = noise_transparent;
    const ScheduleDependence repaired_dependence =
        ScheduleDependence::build(repaired, repaired_options);

    WidthSearchOptions search_options;
    search_options.node_budget = exact_node_budget;
    const WidthSearchResult result =
        search_width_schedule(repaired, repaired_dependence, search_options);

    if (result.upper_bound >= result.incumbent_peak) {
        return beam_order;
    }

    // result.best_order indexes into `repaired`'s ops, which are hir's ops
    // permuted by beam_order, so composing the two maps each final position
    // back to hir's own op indices.
    std::vector<uint32_t> composed(beam_order.size());
    for (size_t i = 0; i < result.best_order.size(); ++i) {
        composed[i] = beam_order[result.best_order[i]];
    }
    return composed;
}

// ---------------------------------------------------------------------------
// Neutral-rotation sinking: a rightward bubble per RotationNeutral op.
// ---------------------------------------------------------------------------

bool independent(const ScheduleDependence& dependence, uint32_t a, uint32_t b) {
    return !std::ranges::binary_search(dependence.successors(a), b) &&
           !std::ranges::binary_search(dependence.predecessors(a), b);
}

// Each op's WidthEffect when `order` is replayed from a fresh subspace,
// indexed by op index (not position) so it stays valid as `order` is
// mutated afterward: a RotationNeutral op is a complete no-op on the
// subspace (see active_width_schedule_pass.h), so removing or relocating
// one never changes any other op's classification. Every RotationNeutral
// crossing performed below therefore reads a classification that remains
// accurate no matter how many other neutral rotations have already moved.
std::vector<WidthEffect> effect_by_op_index(const HirModule& hir,
                                            const std::vector<uint32_t>& order) {
    std::vector<WidthEffect> effect(hir.ops.size(), WidthEffect::None);
    DormantSubspace subspace(hir.num_qubits);
    for (uint32_t op : order) {
        effect[op] = classify_and_apply(hir, hir.ops[op], subspace).effect;
    }
    return effect;
}

// Bubbles every RotationNeutral op in `order` as far right as it can legally
// go, one op at a time in original schedule order, stopping each bubble
// before the first op that is dependent (per `dependence`) or expanding.
// Crossing only independent, non-expanding ops never raises the width the
// rotation runs at, and never reorders any pair of non-sunk ops relative to
// each other, so every intermediate and final order stays a legal linear
// extension of `dependence`. Does not move RotationStabilizer ops: they
// emit no planner action, so sinking one would not change dense work.
void sink_neutral_rotations(const HirModule& hir, const ScheduleDependence& dependence,
                            std::vector<uint32_t>& order) {
    const std::vector<WidthEffect> effect = effect_by_op_index(hir, order);

    std::vector<uint32_t> to_sink;
    for (uint32_t op : order) {
        if (effect[op] == WidthEffect::RotationNeutral) {
            to_sink.push_back(op);
        }
    }
    if (to_sink.empty()) {
        return;
    }

    // position[op] tracks op's live index in `order`, kept in sync with
    // every swap below so each rotation's bubble starts from wherever it
    // actually sits, including any shift an earlier rotation's own bubble
    // caused.
    std::vector<uint32_t> position(hir.ops.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        position[order[i]] = i;
    }

    for (uint32_t op : to_sink) {
        size_t curr = position[op];
        while (curr + 1 < order.size()) {
            const uint32_t next_op = order[curr + 1];
            if (!independent(dependence, op, next_op) || is_expanding_effect(effect[next_op])) {
                break;
            }
            std::swap(order[curr], order[curr + 1]);
            position[order[curr]] = static_cast<uint32_t>(curr);
            position[order[curr + 1]] = static_cast<uint32_t>(curr + 1);
            ++curr;
        }
    }
}

}  // namespace

ActiveWidthSchedulePass::ActiveWidthSchedulePass(ActiveWidthScheduleOptions options)
    : options_(options) {}

void ActiveWidthSchedulePass::run(HirModule& hir) {
    const ActiveWidthTrace incumbent_trace = analyze_active_width(hir);
    incumbent_peak_ = incumbent_trace.peak_width;
    incumbent_dense_work_ = estimate_dense_work(incumbent_trace);

    ScheduleDependenceOptions dependence_options;
    dependence_options.noise_transparent = options_.noise_transparent;
    const ScheduleDependence dependence = ScheduleDependence::build(hir, dependence_options);

    std::vector<uint32_t> order = run_beam_search(hir, dependence, options_.beam_width);

    if (options_.exact_node_budget > 0) {
        order = maybe_exact_repair(hir, dependence, std::move(order), options_.noise_transparent,
                                   options_.exact_node_budget);
    }

    if (options_.sink_neutral_rotations) {
        sink_neutral_rotations(hir, dependence, order);
    }

    HirModule candidate = hir;
    apply_schedule(candidate, dependence, order);
    const ActiveWidthTrace candidate_trace = analyze_active_width(candidate);
    const double candidate_dense_work = estimate_dense_work(candidate_trace);

    const bool better = (candidate_trace.peak_width < incumbent_peak_) ||
                        (candidate_trace.peak_width == incumbent_peak_ &&
                         candidate_dense_work < incumbent_dense_work_);

    if (better) {
        hir = std::move(candidate);
        result_peak_ = candidate_trace.peak_width;
        result_dense_work_ = candidate_dense_work;
        applied_ = true;
    } else {
        result_peak_ = incumbent_peak_;
        result_dense_work_ = incumbent_dense_work_;
        applied_ = false;
    }
}

}  // namespace clifft
