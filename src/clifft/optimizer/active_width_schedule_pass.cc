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

void absorb_transitions(uint32_t& peak, double& dense_work,
                        const std::vector<WidthTransition>& transitions) {
    for (const WidthTransition& transition : transitions) {
        peak = std::max(peak, transition.after);
        dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
}

void absorb_closure_transitions(BeamState& state, const std::vector<WidthTransition>& transitions) {
    absorb_transitions(state.peak, state.dense_work, transitions);
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

std::vector<uint32_t> ready_ops_snapshot(const detail::SearchFrontier& frontier) {
    return std::vector<uint32_t>(frontier.ready().begin(), frontier.ready().end());
}

// A scored but not yet materialized child of beam[parent_index]: committing
// to `first_op` (a ready expanding op) and closing would append `ops`
// (first_op followed by its closure sweep, in execution order) to the
// parent's own order and reach `width_after_closure`/peak/dense_work.
// executed_bits is the parent's post-candidate executed-op bitset, captured
// for deduplication without needing a per-candidate SearchFrontier clone.
// Scored by (peak, width_after_closure, dense_work, first_op) ascending,
// matching search_width_schedule's own candidate ranking so the beam and
// the exact repair step agree on which move looks best whenever both
// consider the same choice.
struct ScoredCandidate {
    uint32_t parent_index = 0;
    std::vector<uint32_t> ops;
    std::vector<uint64_t> executed_bits;
    uint32_t width_after_closure = 0;
    uint32_t peak = 0;
    double dense_work = 0.0;
    uint32_t first_op = 0;
};

// Scores every ready expanding op of `parent` without materializing a full
// child BeamState for each: `parent.frontier` is mutated and restored via
// execute()/undo() exactly as active_width_search.cc's simulate_candidate
// mutates and restores its shared DFS frontier, and only `subspace` (which
// has no cheap undo) is cloned, once per candidate. This is the expensive
// step's cost reduction: on a fixture with many simultaneously-ready
// independent expanding ops, most candidates are discarded after scoring,
// so paying only for a DormantSubspace clone and a small ops list here --
// not a SearchFrontier clone and a full copy of the (potentially
// near-complete) order vector -- is what keeps this affordable. `parent` is
// left exactly as found on return.
std::vector<ScoredCandidate> score_candidates(const HirModule& hir, BeamState& parent,
                                              uint32_t parent_index) {
    std::vector<ScoredCandidate> scored;
    for (uint32_t op : ready_ops_snapshot(parent.frontier)) {
        if (!detail::is_expanding(hir, hir.ops[op], parent.subspace)) {
            continue;
        }

        DormantSubspace scratch(parent.subspace);
        std::vector<detail::UndoStep> log;

        log.emplace_back(op, parent.frontier.execute(op));
        ScoredCandidate candidate;
        candidate.parent_index = parent_index;
        candidate.first_op = op;
        candidate.ops.push_back(op);

        const WidthTransition first_transition = classify_and_apply(hir, hir.ops[op], scratch);
        assert(is_expanding_effect(first_transition.effect) &&
               "score_candidates chose a ready op is_expanding did not classify as expanding");
        candidate.peak = std::max(parent.peak, first_transition.after);
        candidate.dense_work =
            parent.dense_work + detail::dense_work_contribution(first_transition.effect,
                                                                first_transition.before,
                                                                first_transition.after);

        std::vector<WidthTransition> swept;
        detail::run_closure(hir, parent.frontier, scratch, candidate.ops, log, &swept);
        absorb_transitions(candidate.peak, candidate.dense_work, swept);

        candidate.width_after_closure = scratch.active_width();
        candidate.executed_bits = parent.frontier.executed_bits();

        detail::undo_all(parent.frontier, log);
        scored.push_back(std::move(candidate));
    }
    return scored;
}

// Materializes a beam_width survivor: clones its parent once (the only
// clone this candidate ever needed) and replays `candidate.ops` -- already
// known from scoring, so this is a deterministic replay rather than a new
// search -- through the clone's own frontier and subspace. The assertions
// cross-check that this replay reaches exactly what score_candidates
// predicted, since the two are computed by independent code paths that
// must agree bit-for-bit (dense_work accumulates the same transitions in
// the same order in both places, so the floating-point sums match exactly,
// not just numerically).
BeamState materialize_candidate(const HirModule& hir, const std::vector<BeamState>& beam,
                                const ScoredCandidate& candidate) {
    const BeamState& parent = beam[candidate.parent_index];
    BeamState state(parent.frontier, parent.subspace);
    state.peak = parent.peak;
    state.dense_work = parent.dense_work;
    state.order = parent.order;
    state.order.reserve(state.order.size() + candidate.ops.size());

    for (uint32_t op : candidate.ops) {
        state.frontier.execute(op);
        const WidthTransition transition = classify_and_apply(hir, hir.ops[op], state.subspace);
        state.peak = std::max(state.peak, transition.after);
        state.dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
        state.order.push_back(op);
    }

    assert(state.subspace.active_width() == candidate.width_after_closure &&
           "materialize_candidate's replay disagrees with score_candidates' speculative width");
    assert(state.peak == candidate.peak &&
           "materialize_candidate's replay disagrees with score_candidates' speculative peak");
    assert(
        state.dense_work == candidate.dense_work &&
        "materialize_candidate's replay disagrees with score_candidates' speculative dense work");

    return state;
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
//
// Two-phase per step: score_candidates ranks every ready expanding op of
// every current beam state cheaply (see its own comment), then only the
// surviving beam_width candidates -- picked by the same dedup-then-rank
// rule the single-phase version used -- pay to materialize a full BeamState
// via materialize_candidate. A wide, mostly-discarded generation is common
// on fixtures with many independent expanding rotations, so this ordering
// (score everything, materialize only the winners) is what makes the beam
// width affordable to scale.
std::vector<uint32_t> run_beam_search(const HirModule& hir, const ScheduleDependence& dependence,
                                      uint32_t beam_width) {
    std::vector<BeamState> beam;
    beam.push_back(make_initial_beam_state(hir, dependence));

    std::vector<BeamState> completed;
    while (!beam.empty()) {
        std::vector<ScoredCandidate> generation;
        std::vector<bool> parent_has_candidates(beam.size(), false);
        for (uint32_t i = 0; i < beam.size(); ++i) {
            std::vector<ScoredCandidate> scored = score_candidates(hir, beam[i], i);
            if (!scored.empty()) {
                parent_has_candidates[i] = true;
                for (ScoredCandidate& candidate : scored) {
                    generation.push_back(std::move(candidate));
                }
            }
        }

        for (uint32_t i = 0; i < beam.size(); ++i) {
            if (parent_has_candidates[i]) {
                continue;
            }
            // Closure already consumed every ready non-expanding op, so no
            // ready expanding candidates left means no ready ops left at
            // all, which is only possible once every op has executed (see
            // active_width_search.h's confluence note).
            assert(beam[i].frontier.executed_count() == dependence.num_ops() &&
                   "a closed beam state with no ready expanding op must have executed every op");
            completed.push_back(std::move(beam[i]));
        }

        if (generation.empty()) {
            break;
        }

        // Deduplicate by executed-op bitset: by confluence, the same set of
        // executed ops always reaches the same subspace and width, so only
        // the smaller-dense-work copy is worth keeping.
        std::ranges::sort(generation, [](const ScoredCandidate& a, const ScoredCandidate& b) {
            return a.executed_bits < b.executed_bits;
        });
        std::vector<ScoredCandidate> deduped;
        for (ScoredCandidate& candidate : generation) {
            if (!deduped.empty() && deduped.back().executed_bits == candidate.executed_bits) {
                if (candidate.dense_work < deduped.back().dense_work) {
                    deduped.back() = std::move(candidate);
                }
                continue;
            }
            deduped.push_back(std::move(candidate));
        }

        std::ranges::sort(deduped, [](const ScoredCandidate& a, const ScoredCandidate& b) {
            if (a.peak != b.peak) {
                return a.peak < b.peak;
            }
            if (a.width_after_closure != b.width_after_closure) {
                return a.width_after_closure < b.width_after_closure;
            }
            if (a.dense_work != b.dense_work) {
                return a.dense_work < b.dense_work;
            }
            return a.first_op < b.first_op;
        });

        std::vector<BeamState> next_beam;
        next_beam.reserve(std::min<size_t>(deduped.size(), beam_width));
        for (size_t i = 0; i < deduped.size() && i < beam_width; ++i) {
            next_beam.push_back(materialize_candidate(hir, beam, deduped[i]));
        }
        beam = std::move(next_beam);
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
