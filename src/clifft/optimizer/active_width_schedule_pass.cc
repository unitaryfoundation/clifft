#include "clifft/optimizer/active_width_schedule_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/active_width_closure.h"
#include "clifft/optimizer/schedule_dependence.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <tuple>
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

BeamState make_initial_beam_state(const HirModule& hir,
                                  const detail::ScheduleDependence& dependence) {
    BeamState state(detail::SearchFrontier(dependence), DormantSubspace(hir.num_qubits));
    std::vector<detail::UndoStep> discarded_log;
    std::vector<uint32_t> discarded_newly_ready;
    std::vector<WidthTransition> transitions;
    detail::run_closure(hir, state.frontier, state.subspace, state.order, discarded_log,
                        discarded_newly_ready, &transitions);
    absorb_closure_transitions(state, transitions);
    return state;
}

// Copies out the parent's current ready set before any candidate scoring
// below mutates it: score_candidates' loop executes and undoes each
// candidate against `parent.frontier` in turn, so iterating a live view of
// its ready ops while that same loop body edits them is unsafe. The scan is
// ascending (lowest_ready_hint() up to num_ops()), so the candidate list
// score_candidates below produces is ordered the same way regardless of the
// history of execute()/undo() calls that got `frontier` to its current
// state -- which matters because the beam ranking sort's key does not fully
// order every pair of candidates, and an input built in a different order
// could lead an otherwise-tied sort to a different, equally-ranked winner.
// This runs once per beam member per beam step, not once per closure step,
// so a scan over a few thousand ops here is negligible next to the cost of
// scoring and closing each candidate.
std::vector<uint32_t> ready_ops_snapshot(const detail::SearchFrontier& frontier) {
    std::vector<uint32_t> ready;
    for (uint32_t op = frontier.lowest_ready_hint(); op < frontier.num_ops(); ++op) {
        if (frontier.is_ready(op)) {
            ready.push_back(op);
        }
    }
    return ready;
}

// A scored but not yet materialized child of beam[parent_index]: committing
// to `first_op` (a ready expanding op) and closing would append `ops`
// (first_op followed by its closure sweep, in execution order) to the
// parent's own order and reach `width_after_closure`/peak/dense_work.
// executed_bits is the parent's post-candidate executed-op bitset, captured
// for deduplication without needing a per-candidate SearchFrontier clone.
// Scored by (peak, width_after_closure, -ops.size(), first_op) ascending:
// among candidates tied on peak and width_after_closure, the one whose
// closure swept the most operations into place (typically measurements)
// ranks first, ahead of dense_work_so_far. The intuition: a wide sweep
// usually means several commuting measurements collapsed the subspace
// immediately, which is a stronger signal that this branch is heading
// toward a state with few remaining live directions than the dense work
// spent to get there so far -- measured on the clifft-paper corpus, this
// tie-break reaches lower summed dense work than tying on dense_work_so_far
// (with or without also considering sweep count).
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
// paired execute()/undo() calls, and only `subspace` (which has no cheap
// undo) is cloned, once per candidate. This is the expensive
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
        std::vector<uint32_t> newly_ready_log;

        log.push_back(detail::UndoStep{op, parent.frontier.execute(op, newly_ready_log)});
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
        detail::run_closure(hir, parent.frontier, scratch, candidate.ops, log, newly_ready_log,
                            &swept);
        absorb_transitions(candidate.peak, candidate.dense_work, swept);

        candidate.width_after_closure = scratch.active_width();
        candidate.executed_bits = parent.frontier.executed_bits();

        detail::undo_all(parent.frontier, log, newly_ready_log);
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

    // This replay never backtracks, so the newly-ready ops execute() reports
    // are never read; one scratch vector reused for the whole loop avoids
    // both a fresh allocation per op and the earlier per-call
    // std::vector<uint32_t> return this replaced.
    std::vector<uint32_t> discarded_newly_ready;
    for (uint32_t op : candidate.ops) {
        state.frontier.execute(op, discarded_newly_ready);
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
// op's own closure sweep scores best. The constructor rejects beam_width ==
// 0, so at least one beam member always survives to complete a schedule.
//
// Two-phase per step: score_candidates ranks every ready expanding op of
// every current beam state cheaply (see its own comment), then only the
// surviving beam_width candidates -- picked by the same dedup-then-rank
// rule the single-phase version used -- pay to materialize a full BeamState
// via materialize_candidate. A wide, mostly-discarded generation is common
// on fixtures with many independent expanding rotations, so this ordering
// (score everything, materialize only the winners) is what makes the beam
// width affordable to scale.
std::vector<uint32_t> run_beam_search(const HirModule& hir,
                                      const detail::ScheduleDependence& dependence,
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
            // active_width_closure.h's confluence note).
            assert(beam[i].frontier.executed_count() == dependence.num_ops() &&
                   "a closed beam state with no ready expanding op must have executed every op");
            completed.push_back(std::move(beam[i]));
        }

        if (generation.empty()) {
            break;
        }

        // Deduplicate by executed-op bitset: by confluence, the same set of
        // executed ops always reaches the same subspace and width, so only
        // one survivor is worth keeping. The peak reached along the way is
        // not confluent, though: it is a running max over a path-dependent
        // sequence of transitions, so two candidates that converge on the
        // same executed set can still disagree on peak. Keeping whichever
        // has the smaller dense_work, regardless of peak, can therefore
        // discard the strictly better duplicate -- lower peak but higher
        // dense_work -- since peak can only stay the same or grow over the
        // rest of the schedule while dense_work is only a secondary
        // objective. Rank by the pass's own lexicographic objective,
        // (peak, dense_work), instead, with first_op breaking a remaining
        // tie deterministically.
        std::ranges::sort(generation, [](const ScoredCandidate& a, const ScoredCandidate& b) {
            return a.executed_bits < b.executed_bits;
        });
        std::vector<ScoredCandidate> deduped;
        for (ScoredCandidate& candidate : generation) {
            if (!deduped.empty() && deduped.back().executed_bits == candidate.executed_bits) {
                const ScoredCandidate& kept = deduped.back();
                const bool candidate_is_better =
                    std::tie(candidate.peak, candidate.dense_work, candidate.first_op) <
                    std::tie(kept.peak, kept.dense_work, kept.first_op);
                if (candidate_is_better) {
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
            if (a.ops.size() != b.ops.size()) {
                return a.ops.size() > b.ops.size();  // -swept_count ascending.
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
    assert(best != nullptr &&
           "beam_width >= 1 (the constructor rejects 0) guarantees at least one completed state");
    return best->order;
}

// ---------------------------------------------------------------------------
// Neutral-rotation sinking: a rightward bubble per RotationNeutral op.
// ---------------------------------------------------------------------------

bool independent(const detail::ScheduleDependence& dependence, uint32_t a, uint32_t b) {
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
void sink_neutral_rotations(const HirModule& hir, const detail::ScheduleDependence& dependence,
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

// True when `hir` has any op a beam search could ever branch on. T_GATE and
// PHASE_ROTATION are the only op types is_expanding ever calls genuinely
// discretionary: an expanding INSTRUMENT is always the sole ready op when it
// fires (detail::ScheduleDependence treats it as a positional barrier), so
// its presence or absence changes nothing a scheduler could choose
// differently.
bool has_rotation_op(const HirModule& hir) {
    return std::ranges::any_of(hir.ops, [](const HeisenbergOp& op) {
        return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
    });
}

}  // namespace

ActiveWidthSchedulePass::ActiveWidthSchedulePass(ActiveWidthScheduleOptions options)
    : options_(options) {
    if (options_.beam_width == 0) {
        throw std::invalid_argument("ActiveWidthSchedulePass: beam_width must be positive");
    }
}

void ActiveWidthSchedulePass::run(HirModule& hir) {
    built_dependence_ = false;

    const ActiveWidthTrace incumbent_trace = analyze_active_width(hir);
    incumbent_peak_ = incumbent_trace.peak_width;
    incumbent_dense_work_ = estimate_dense_work(incumbent_trace);

    // See the header comment's "Early exit": with nothing for a scheduler to
    // choose among, report the incumbent unchanged rather than pay for
    // ScheduleDependence::build's O(N^2) scan.
    if (incumbent_peak_ == 0 || !has_rotation_op(hir)) {
        result_peak_ = incumbent_peak_;
        result_dense_work_ = incumbent_dense_work_;
        applied_ = false;
        return;
    }

    detail::ScheduleDependenceOptions dependence_options;
    dependence_options.noise_transparent = options_.noise_transparent;
    const detail::ScheduleDependence dependence =
        detail::ScheduleDependence::build(hir, dependence_options);
    built_dependence_ = true;

    std::vector<uint32_t> order = run_beam_search(hir, dependence, options_.beam_width);

    if (options_.sink_neutral_rotations) {
        sink_neutral_rotations(hir, dependence, order);
    }

    HirModule candidate = hir;
    detail::apply_schedule(candidate, dependence, order);
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
