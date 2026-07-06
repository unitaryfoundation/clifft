// Exact-mode driver implementation. exact_driver.h is the entry point;
// design/state-dependent-jumps.md holds the full design. The vocabulary
// used throughout this file:
//
//   annotation   A LEVEL_TRANSITION[name] or LOSS(p) node in the user's
//                circuit: the level-transition channel a physical
//                operation applies to each of its qubits.
//   target       One (op_index, qubit) pair of an annotation; a node
//                with n qubit operands is n targets.
//   consult      One target's per-shot draw against its channel: did it
//                fire, and to which destination level. A *classical*
//                consult is one whose qubit's level is definite at that
//                point (known computational, leaked, or lost), so the
//                driver draws it without touching quantum state.
//   fire / jump  A consult that moved the qubit to another level.
//   site         A target kept as a runtime INSTRUMENT barrier in the
//                compiled module, consulted in-line by the VM;
//                trace()'s materialization order assigns site ids.
//   trap         A fire the VM cannot resolve in-line -- the qubit's
//                level is not definite there -- so execution stops at
//                the site and control returns to the driver.
//   carrier      The simulated qubit cell that keeps holding a
//                noncomputational population's correlations while the
//                status ledger tracks its level classically.
//   continuation The circuit rewritten under a shot's events (initial
//                statuses, jumps so far, pre-drawn classical outcomes)
//                and recompiled whole; execution resumes past the
//                trapped site, the prefix guaranteed identical to what
//                the shot already ran.
//   main line    The continuation of "nothing happened": every shot
//                starts in it unless an initial level is
//                noncomputational.
//   forced twin  The forced-outcome variant of a sampling measurement
//                opcode: same record slot, outcome read from
//                state.forced_record instead of drawn. Collapses a
//                neglect-form trap's carrier to the source the trap
//                reported.
//   sidecar      The per-shot herald record: one flag per visible
//                measurement slot, set when a ternary classifier drew
//                its "ambiguous" symbol for that slot.
//   driver draw  Randomness this file consumes between VM runs
//                (initial levels, trap destinations, classical
//                consults, herald flags): one stream per shot, every
//                draw ordered before any cache lookup. The VM's own
//                Born randomness is the separate SVM stream (seed.h).

#include "clifft/noncomp/exact_driver.h"

#include "clifft/backend/backend.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sampler.h"
#include "clifft/noncomp/seed.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_registry.h"
#include "clifft/svm/svm.h"
#include "clifft/svm/svm_math.h"
#include "clifft/util/xoshiro.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

// The transition consulted by one annotation node: a named instrument for
// LEVEL_TRANSITION, a uniform all-lost rate for LOSS.
struct AnnotationChannel {
    const TransitionInstrument* instrument = nullptr;  // null for LOSS
    double loss_p = 0.0;
    uint8_t lost_level = 0;
};

AnnotationChannel resolve_annotation(const AstNode& node, const NonComputationalModel& model,
                                     uint32_t op_index) {
    AnnotationChannel channel;
    if (node.gate == GateType::LOSS) {
        const std::optional<uint8_t> lost = sole_lost_level(model.levels());
        if (!lost.has_value()) {
            throw std::invalid_argument(
                "sample_noncomputational: LOSS at op " + std::to_string(op_index) +
                " requires a level table with exactly one Lost-category level");
        }
        channel.loss_p = loss_probability(node.args, op_index, "sample_noncomputational");
        channel.lost_level = *lost;
        return channel;
    }
    channel.instrument = model.transition_named(node.tag);
    if (channel.instrument == nullptr) {
        throw std::invalid_argument("sample_noncomputational: LEVEL_TRANSITION[" + node.tag +
                                    "] at op " + std::to_string(op_index) +
                                    " does not name a transition in the model");
    }
    return channel;
}

// Draw a destination level from `source`'s column of the instrument,
// restricted to the levels `admit` accepts. `admit` is a pure
// level_id -> bool predicate naming which destinations participate; the
// draw normalizes over the admitted entries' own mass, summed here so a
// caller cannot supply a mismatched normalizer (which would silently
// bias the draw). Uses the measurement kernels' last-positive fallback
// for the floating-point tail, and consumes exactly one RNG draw.
template <typename Admit>
uint8_t draw_from_column(const TransitionInstrument& instrument, uint8_t source,
                         Xoshiro256PlusPlus& rng, Admit&& admit, size_t num_levels) {
    double mass = 0.0;
    for (uint8_t to = 0; to < num_levels; ++to) {
        if (admit(to)) {
            mass += instrument.prob(to, source);
        }
    }
    if (!(mass > 0.0)) {
        // Unreachable through the current call sites, all of which fire
        // only when the column's admitted mass is positive; a real error
        // beats a Release-mode draw of level 255 surfacing downstream.
        throw std::logic_error(
            "sample_noncomputational: destination draw over an empty column (source level " +
            std::to_string(source) + ")");
    }
    const double u = rng.next_double() * mass;
    double acc = 0.0;
    int last_positive = -1;
    for (uint8_t to = 0; to < num_levels; ++to) {
        if (!admit(to)) {
            continue;
        }
        const double p = instrument.prob(to, source);
        if (p > 0.0) {
            last_positive = to;
        }
        acc += p;
        if (u < acc) {
            return to;
        }
    }
    assert(last_positive >= 0 && "destination draw over an empty column");
    return static_cast<uint8_t>(last_positive);
}

// Walk the annotated circuit's statuses under `events` and rebuild the
// classical-outcome stream in the walk's own (circuit) order: outcomes
// already drawn for a consult are reused by annotation target, and
// consults seen for the first time are drawn. Rebuilding rather than
// appending matters after a trap, which turns the trapped qubit's later
// consults classical *between* previously recorded ones -- an
// append-only stream would replay old outcomes at the wrong targets.
// Reused outcomes keep the executed prefix's compilation stable;
// first-seen consults live only in the not-yet-executed suffix, so a
// fresh draw is unbiased. Returns the walk's final statuses (computed
// from the real, uncanonicalized initials). The walk is the single
// source of truth for which consults are classical, so the events
// stream always matches what rewrite_continuation will validate.
std::vector<QubitStatus> extend_classical_outcomes(const Circuit& annotated,
                                                   ExactShotEvents& events,
                                                   const NonComputationalModel& model,
                                                   Xoshiro256PlusPlus& rng) {
    const LevelSet& levels = model.levels();
    std::map<std::pair<uint32_t, uint32_t>, uint8_t> jump_dest;
    for (const ResolvedJump& jump : events.jumps) {
        jump_dest.emplace(std::make_pair(jump.op_index, jump.qubit), jump.destination_level);
    }

    std::map<std::pair<uint32_t, uint32_t>, ClassicalOutcome> drawn;
    for (const ClassicalOutcome& outcome : events.classical_outcomes) {
        drawn.emplace(std::make_pair(outcome.op_index, outcome.qubit), outcome);
    }
    std::vector<ClassicalOutcome> ordered;
    ordered.reserve(events.classical_outcomes.size());

    std::vector<QubitStatus> status = events.initial_status;

    for (uint32_t op_index = 0; op_index < annotated.nodes.size(); ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        const GateType gate = node.gate;
        if (gate == GateType::LEVEL_TRANSITION || gate == GateType::LOSS) {
            for (const Target& target : node.targets) {
                const uint32_t qubit = target.value();
                const QubitStatus pre = status[qubit];
                const bool classical =
                    pre.kind() == QubitStatusKind::Leaked || pre.kind() == QubitStatusKind::Lost;
                if (!classical) {
                    const auto jump = jump_dest.find({op_index, qubit});
                    if (jump != jump_dest.end()) {
                        status[qubit] = levels.status_for(jump->second);
                    }
                    continue;
                }
                const auto seen = drawn.find({op_index, qubit});
                if (seen != drawn.end()) {
                    if (seen->second.source_level != pre.level_id()) {
                        throw std::logic_error(
                            "sample_noncomputational: classical outcome reuse at op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) +
                            " crossed a source-level change (drawn at level " +
                            std::to_string(seen->second.source_level) + ", walk holds level " +
                            std::to_string(pre.level_id()) + ")");
                    }
                    ordered.push_back(seen->second);
                    if (seen->second.jumped) {
                        status[qubit] = levels.status_for(seen->second.destination_level);
                    }
                    continue;
                }
                const AnnotationChannel channel = resolve_annotation(node, model, op_index);
                ClassicalOutcome outcome{op_index, qubit, false, 0, pre.level_id()};
                if (channel.instrument != nullptr) {
                    const uint8_t source = pre.level_id();
                    const double total = channel.instrument->column_sum(source);
                    if (rng.next_double() < total) {
                        outcome.jumped = true;
                        outcome.destination_level = draw_from_column(
                            *channel.instrument, source, rng, [](uint8_t) { return true; },
                            levels.size());
                    }
                } else if (pre.kind() != QubitStatusKind::Lost &&
                           rng.next_double() < channel.loss_p) {
                    // LOSS on a leaked (not lost) qubit can still vacate
                    // it; an already-lost qubit records a no-op outcome
                    // without spending a draw.
                    outcome.jumped = true;
                    outcome.destination_level = channel.lost_level;
                }
                ordered.push_back(outcome);
                if (outcome.jumped) {
                    status[qubit] = levels.status_for(outcome.destination_level);
                }
            }
            continue;
        }
        // Ordinary operations advance statuses exactly as the rewrite
        // does; drops keep entry statuses, which the stepper handles via
        // the shared policy scan semantics.
        bool drop_op = false;
        for (const QubitOperand& operand : qubit_operands(node)) {
            if (operand_action(gate, status[operand.qubit].kind(), model.policy()) ==
                OperandAction::Drop) {
                drop_op = true;
            }
        }
        for (const QubitOperand& operand : qubit_operands(node)) {
            const QubitStatus pre = status[operand.qubit];
            status[operand.qubit] =
                drop_op ? pre
                        : normal_post_op_status(pre, gate, operand.role, model.policy(), levels);
        }
    }
    events.classical_outcomes = std::move(ordered);
    return status;
}

// Cache key: the canonicalized status-outcome delta. Computational
// initial levels canonicalize to the zero level -- the rewrite provably
// depends on initial statuses only through their kinds (plus levels for
// noncomputational initials) -- so shots differing only in |1> preloads
// share one module.
std::string cache_key(const ExactShotEvents& events, const LevelSet& levels) {
    std::string key;
    key.reserve(events.initial_status.size() * 2 + 10 + events.jumps.size() * 9 +
                events.classical_outcomes.size() * 10);
    for (const QubitStatus& s : events.initial_status) {
        key.push_back(static_cast<char>(s.kind()));
        const bool classical =
            s.kind() == QubitStatusKind::Leaked || s.kind() == QubitStatusKind::Lost;
        key.push_back(classical ? static_cast<char>(s.level_id())
                                : static_cast<char>(levels.computational_zero_id()));
    }
    auto push32 = [&key](uint32_t v) {
        for (int b = 0; b < 4; ++b) {
            key.push_back(static_cast<char>((v >> (8 * b)) & 0xFF));
        }
    };
    // Each variable-length section is count-prefixed, so equal keys mean
    // equal events by construction: the fixed-size records cannot be
    // reparsed across a section boundary. The markers are readability.
    key.push_back('J');
    push32(static_cast<uint32_t>(events.jumps.size()));
    for (const ResolvedJump& jump : events.jumps) {
        push32(jump.op_index);
        push32(jump.qubit);
        key.push_back(static_cast<char>(jump.destination_level));
    }
    key.push_back('C');
    push32(static_cast<uint32_t>(events.classical_outcomes.size()));
    for (const ClassicalOutcome& outcome : events.classical_outcomes) {
        push32(outcome.op_index);
        push32(outcome.qubit);
        key.push_back(outcome.jumped ? 1 : 0);
        key.push_back(static_cast<char>(outcome.destination_level));
    }
    return key;
}

ExactShotEvents canonicalize(const ExactShotEvents& events, const LevelSet& levels) {
    ExactShotEvents canonical = events;
    for (QubitStatus& s : canonical.initial_status) {
        if (s.kind() == QubitStatusKind::ComputationalKnown) {
            s = levels.status_for(levels.computational_zero_id());
        }
    }
    return canonical;
}

// One rewritten continuation plus its per-herald-flag compiled modules.
struct ContinuationEntry {
    ContinuationRewrite rw;
    std::map<std::vector<uint8_t>, CompiledModule> modules;
};

void check_max_rank(const CompiledModule& module, std::optional<uint32_t> max_rank) {
    if (!max_rank.has_value() || module.peak_rank <= *max_rank) {
        return;
    }
    // Name the first instruction whose active dimension exceeded the cap,
    // through the compile's per-instruction k history.
    std::string site;
    for (size_t i = 0; i < module.source_map.size(); ++i) {
        if (module.source_map.active_k_at(i) > *max_rank) {
            const auto lines = module.source_map.lines_for(i);
            if (!lines.empty()) {
                site = " (first exceeded at circuit line " + std::to_string(lines[0]) + ")";
            }
            break;
        }
    }
    throw std::invalid_argument(
        "sample_noncomputational: compiled peak rank " + std::to_string(module.peak_rank) +
        " exceeds max_rank " + std::to_string(*max_rank) + site +
        "; consider damping=\"neglect\" for high-rate sites or a larger max_rank");
}

// The exact-mode pipelines: the default-enabled passes that declare
// record-order preservation (pass_registry.h). Forced-outcome execution
// is why the property matters: reordering commuting measurements is
// exchangeable under sampling semantics but wrong once a continuation's
// trace-out is forced -- an entangled partner must not measure before
// the forced collapse it is correlated with. Today the filter excludes
// exactly the statevector squeeze, whose rank compaction is a
// performance loss to quantify; a future pass joins these pipelines
// only by declaring itself order-preserving. One pipeline serves every
// module in the mode, so prefix identity is preserved.
HirPassManager exact_hir_pass_manager() {
    HirPassManager pm;
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::HIR && info.default_enabled && info.record_order.preserved) {
            pm.add_pass(info.make_hir());
        }
    }
    return pm;
}

// Bytecode counterpart. Every current default bytecode pass preserves
// the record sequence (they fuse contiguous instructions, and record
// slots ride inside instruction payloads), so today this matches the
// default pipeline; the filter is the standing contract.
BytecodePassManager exact_bytecode_pass_manager() {
    BytecodePassManager pm;
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::Bytecode && info.default_enabled &&
            info.record_order.preserved) {
            pm.add_pass(info.make_bc());
        }
    }
    return pm;
}

#ifndef NDEBUG
// The sequence the exact pipelines must preserve: every measurement
// record index, visible and hidden alike, in program order. Audited
// around each module compile so a pass misdeclared as order-preserving
// fails loudly here instead of corrupting correlations.
std::vector<uint32_t> record_sequence(const HirModule& hir) {
    std::vector<uint32_t> seq;
    for (const HeisenbergOp& op : hir.ops) {
        if (op.op_type() == OpType::MEASURE) {
            seq.push_back(static_cast<uint32_t>(op.meas_record_idx()));
        }
    }
    return seq;
}
#endif

// The forced-outcome twin of a sampling measurement opcode, or
// NUM_OPCODES when the opcode is not a sampling measurement.
Opcode forced_twin(Opcode op) {
    switch (op) {
        case Opcode::OP_MEAS_DORMANT_STATIC:
            return Opcode::OP_MEAS_DORMANT_STATIC_FORCED;
        case Opcode::OP_MEAS_DORMANT_RANDOM:
            return Opcode::OP_MEAS_DORMANT_RANDOM_FORCED;
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
            return Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED;
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
            return Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED;
        case Opcode::OP_SWAP_MEAS_INTERFERE:
            return Opcode::OP_SWAP_MEAS_INTERFERE_FORCED;
        default:
            return Opcode::NUM_OPCODES;
    }
}

// Swap the trace-out's hidden measurement at `slot` to its forced twin,
// so resume() collapses the trapped carrier to the source the trap
// reported (read from state.forced_record[slot]) instead of redrawing.
// Slot indices ride inside instruction payloads and are never renumbered
// by bytecode passes, so the slot is the durable identity here.
void swap_traceout_to_forced(CompiledModule& module, size_t slot) {
    size_t found = 0;
    for (Instruction& instr : module.bytecode) {
        const Opcode twin = forced_twin(instr.opcode);
        if (twin == Opcode::NUM_OPCODES) {
            continue;
        }
        if (instr.classical.classical_idx == slot) {
            instr.opcode = twin;
            ++found;
        }
    }
    if (found != 1) {
        throw std::logic_error("sample_noncomputational: the forced trace-out slot matched " +
                               std::to_string(found) +
                               " measurement instructions; expected exactly one");
    }
}

// Instructions equal up to the sampling <-> forced opcode swap: the one
// sanctioned prefix difference between a continuation's fresh compile and
// the executed module it must otherwise match byte for byte.
bool equal_modulo_forced_swap(const Instruction& fresh, const Instruction& executed) {
    if (std::memcmp(&fresh, &executed, sizeof(Instruction)) == 0) {
        return true;
    }
    if (forced_twin(fresh.opcode) != executed.opcode) {
        return false;
    }
    Instruction swapped = fresh;
    swapped.opcode = executed.opcode;
    return std::memcmp(&swapped, &executed, sizeof(Instruction)) == 0;
}

}  // namespace

NonComputationalSample sample_noncomputational_exact(const Circuit& circuit,
                                                     const NonComputationalModel& model,
                                                     uint32_t shots, uint64_t global_seed,
                                                     std::optional<uint32_t> max_rank) {
    NonComputationalSample result;
    result.shots = shots;
    result.num_qubits = circuit.num_qubits;
    result.num_measurements = circuit.num_measurements;
    result.num_detectors = circuit.num_detectors;
    result.num_observables = circuit.num_observables;
    if (shots == 0) {
        return result;
    }

    const LevelSet& levels = model.levels();
    const MeasurementClassifier* classifier = model.classifier();
    const bool ternary = classifier != nullptr && classifier->has_herald();

    const Circuit annotated = annotate(circuit, model);
    const InstrumentTraceOptions instrument_options = instrument_trace_options(model);

    std::map<std::string, ContinuationEntry> cache;

    // Level one: the rewrite for `events`, computed once per delta. It is
    // deterministic in the events and consumes no randomness, so a fetch
    // never perturbs sampling.
    auto get_entry = [&](const ExactShotEvents& events, bool force_last) -> ContinuationEntry& {
        const std::string key =
            cache_key(events, levels) + static_cast<char>(force_last ? 'F' : 'f');
        auto [it, inserted] = cache.try_emplace(key);
        ContinuationEntry& entry = it->second;
        if (inserted) {
            entry.rw =
                rewrite_continuation(annotated, canonicalize(events, levels), force_last, model);
        }
        return entry;
    };

    // Level two: the compiled module for one herald-flag assignment (one
    // flag per classified slot, in slot order). Every driver draw feeding
    // this call happens before it, so cache hits and misses sample
    // identically.
    auto get_module = [&](ContinuationEntry& entry, const std::vector<uint8_t>& flags,
                          const CompiledModule* executed_prefix_module,
                          uint32_t prefix_end) -> CompiledModule* {
        auto [mit, module_inserted] = entry.modules.try_emplace(flags);
        if (module_inserted) {
            Circuit patched = entry.rw.circuit;
            assert(flags.size() == entry.rw.classified_measurements.size());
            for (size_t i = 0; i < flags.size(); ++i) {
                if (flags[i] != 0) {
                    const ClassifiedMeasurement& m = entry.rw.classified_measurements[i];
                    assert(m.noise_node != SIZE_MAX);
                    patched.nodes[m.noise_node].args[0] = 0.5;
                }
            }
            HirModule hir = trace(patched, &instrument_options);
#ifndef NDEBUG
            const std::vector<uint32_t> pre_pass_records = record_sequence(hir);
#endif
            exact_hir_pass_manager().run(hir);
#ifndef NDEBUG
            assert(record_sequence(hir) == pre_pass_records &&
                   "an exact-pipeline HIR pass reordered or removed a record op");
#endif
            CompiledModule module = lower(hir);
            exact_bytecode_pass_manager().run(module);
            check_max_rank(module, max_rank);
            if (entry.rw.forced_traceout_slot != SIZE_MAX) {
                swap_traceout_to_forced(module, entry.rw.forced_traceout_slot);
            }
#ifndef NDEBUG
            // Re-entry contract: the continuation's prefix must be
            // bit-identical to the code the shot already executed. A
            // determinism regression shows up here, loudly, rather than
            // as state corruption.
            if (executed_prefix_module != nullptr) {
                assert(prefix_end <= module.bytecode.size() &&
                       prefix_end <= executed_prefix_module->bytecode.size());
                for (uint32_t i = 0; i < prefix_end; ++i) {
                    assert(equal_modulo_forced_swap(module.bytecode[i],
                                                    executed_prefix_module->bytecode[i]) &&
                           "continuation prefix diverged from the executed module");
                }
            }
#else
            (void)executed_prefix_module;
            (void)prefix_end;
#endif
            mit->second = std::move(module);
        }
        return &mit->second;
    };

    // The shared main line: the continuation of "nothing happened".
    ExactShotEvents no_events;
    no_events.initial_status.assign(circuit.num_qubits,
                                    levels.status_for(levels.computational_zero_id()));
    ContinuationEntry& main_entry = get_entry(no_events, false);
    CompiledModule* main_module = get_module(
        main_entry, std::vector<uint8_t>(main_entry.rw.classified_measurements.size(), 0), nullptr,
        0);

    // The state is reused across shots (growth from trap continuations
    // amortizes to the chain maximum); a starting module that outgrows it
    // -- a rare noncomputational-initial shot -- rebuilds it instead,
    // since grow_for_continuation is trap-gated by design.
    auto make_state = [&](const CompiledModule& module) {
        return SchrodingerState(StateConfig{.peak_rank = module.peak_rank,
                                            .num_measurements = module.total_meas_slots,
                                            .num_qubits = module.num_qubits,
                                            .num_detectors = module.num_detectors,
                                            .num_observables = module.num_observables,
                                            .seed = 0});
    };
    SchrodingerState state = make_state(*main_module);
    uint32_t state_rank = main_module->peak_rank;
    uint32_t state_slots = main_module->total_meas_slots;

    for (uint32_t shot = 0; shot < shots; ++shot) {
        Xoshiro256PlusPlus driver_rng(derive_seed(global_seed, shot, kExactDriverDomain));

        ExactShotEvents events;
        events.initial_status.reserve(circuit.num_qubits);
        bool any_noncomp_initial = false;
        for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
            const uint8_t level = draw_initial_level(model, driver_rng);
            events.initial_status.push_back(levels.status_for(level));
            const QubitStatusKind kind = events.initial_status.back().kind();
            any_noncomp_initial |= kind == QubitStatusKind::Leaked || kind == QubitStatusKind::Lost;
        }

        // Forced-outcome buffer for neglect-form trace-outs, one entry
        // per hidden slot the chain forces; reset() clears the state's
        // span each shot.
        std::vector<uint8_t> forced_buffer;

        // Herald flags accumulate per visible slot across the shot's
        // trap chain: a slot's flag is drawn once, when its classified
        // measurement first appears in a continuation.
        std::map<uint32_t, uint8_t> herald_flags;
        auto flags_for = [&](const ContinuationRewrite& rw) {
            std::vector<uint8_t> flags;
            flags.reserve(rw.classified_measurements.size());
            for (const ClassifiedMeasurement& m : rw.classified_measurements) {
                auto [it, inserted] = herald_flags.try_emplace(m.slot, 0);
                if (inserted && ternary) {
                    const double p_herald =
                        classifier->prob(MeasurementClassifier::kHeraldSymbol, m.level);
                    it->second = driver_rng.next_double() < p_herald ? 1 : 0;
                }
                flags.push_back(it->second);
            }
            return flags;
        };

        // Resolve the shot's starting module. A noncomputational initial
        // (rare) compiles its own continuation-from-the-top; classical
        // consults over the whole circuit are pre-drawn for it. The
        // rewrite fetch consumes no randomness, so drawing the herald
        // flags after it keeps every driver draw ahead of module lookup.
        ContinuationEntry* entry = &main_entry;
        CompiledModule* module = main_module;
        std::vector<QubitStatus> final_status =
            extend_classical_outcomes(annotated, events, model, driver_rng);
        if (any_noncomp_initial) {
            entry = &get_entry(events, false);
            module = get_module(*entry, flags_for(entry->rw), nullptr, 0);
        }

        if (shot > 0) {
            state.reset();
        }
        if (module->peak_rank > state_rank || module->total_meas_slots > state_slots) {
            state_rank = std::max(state_rank, module->peak_rank);
            state_slots = std::max(state_slots, module->total_meas_slots);
            state = make_state(*module);
        }
        state.reseed(derive_seed(global_seed, shot, kExactSvmDomain));
        if (state.meas_record.size() < module->total_meas_slots) {
            state.meas_record.resize(module->total_meas_slots, 0);
        }
        // Known |1> initial levels are an X at time zero: a Pauli, so a
        // per-shot frame preload rather than a distinct module.
        for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
            const QubitStatus& s = events.initial_status[q];
            if (s.kind() == QubitStatusKind::ComputationalKnown &&
                s.level_id() == levels.computational_one_id()) {
                bit_set(state.p_x, q, true);
            }
        }
        state.next_noise_idx = 0;
        state.draw_next_noise(module->constant_pool.noise_hazards);
        execute(*module, state);

        while (state.pending_trap.has_value()) {
            const SchrodingerState::InstrumentTrap trap = *state.pending_trap;
            if (trap.site_id >= entry->rw.site_targets.size()) {
                throw std::logic_error(
                    "sample_noncomputational: trap site id is out of range of the executing "
                    "module's site table");
            }
            const auto [op_index, qubit] = entry->rw.site_targets[trap.site_id];
            const AstNode& node = annotated.nodes[op_index];
            const AnnotationChannel channel = resolve_annotation(node, model, op_index);

            // Destination: at a neglect-form site nothing was drawn, so
            // the driver draws over the full column (computational
            // destinations included); elsewhere the class is already
            // leaked/lost and only the level within the trap remainder
            // remains.
            uint8_t dest;
            if (channel.instrument == nullptr) {
                dest = channel.lost_level;
            } else if (trap.destination_pending) {
                dest = draw_from_column(
                    *channel.instrument, trap.source, driver_rng, [](uint8_t) { return true; },
                    levels.size());
            } else {
                dest = draw_from_column(
                    *channel.instrument, trap.source, driver_rng,
                    [&](uint8_t to) {
                        return levels.at(to).category != LevelCategory::Computational;
                    },
                    levels.size());
            }

            events.jumps.push_back({op_index, qubit, dest});
            final_status = extend_classical_outcomes(annotated, events, model, driver_rng);

            // A neglect-form trap hands its carrier over uncollapsed; the
            // continuation's trace-out is forced to the reported source,
            // read from forced_record at the slot the rewrite names. The
            // forced value is per-shot runtime state, so the module stays
            // source-independent.
            const bool force = trap.destination_pending;
            const uint32_t prefix_end = module->instrument_offsets[trap.site_id] + 1;
            ContinuationEntry& next_entry = get_entry(events, force);
            CompiledModule* next_module =
                get_module(next_entry, flags_for(next_entry.rw), module, prefix_end);

            if (force) {
                const size_t slot = next_entry.rw.forced_traceout_slot;
                if (forced_buffer.size() <= slot) {
                    forced_buffer.resize(slot + 1, 0);
                }
                forced_buffer[slot] = trap.source;
                // The span must be re-pointed after any resize.
                state.forced_record = forced_buffer;
            }

            entry = &next_entry;
            module = next_module;
            resume(*module, state, module->instrument_offsets[trap.site_id] + 1);
        }

        // Emit the visible records. The noncomputational path never sets
        // expected_observables, so no reference normalization applies.
        result.measurements.insert(result.measurements.end(), state.meas_record.begin(),
                                   state.meas_record.begin() + circuit.num_measurements);
        result.detectors.insert(result.detectors.end(), state.det_record.begin(),
                                state.det_record.end());
        result.observables.insert(result.observables.end(), state.obs_record.begin(),
                                  state.obs_record.end());
        result.final_status.insert(result.final_status.end(), final_status.begin(),
                                   final_status.end());
        std::vector<uint8_t> shot_heralds(circuit.num_measurements, 0);
        for (const auto& [slot, flag] : herald_flags) {
            shot_heralds[slot] = flag;
        }
        result.heralds.insert(result.heralds.end(), shot_heralds.begin(), shot_heralds.end());
    }

    return result;
}

}  // namespace clifft
