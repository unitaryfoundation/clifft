// Exact-mode driver implementation; exact_driver.h is the entry point.
// The vocabulary used throughout this file:
//
//   annotation         A LEVEL_TRANSITION[name] or LOSS(p) node in the
//                      user's circuit: the level-transition channel a
//                      physical operation applies to each of its qubits.
//   annotation target  One (op_index, qubit) pair of an annotation; a
//                      node with n qubit operands is n annotation
//                      targets. (Bare "Target" is the AST operand type.)
//   consult            One annotation target's per-shot draw against its
//                      channel: did it fire, and to which destination
//                      level. A *classical* consult is one whose qubit's
//                      level is definite at that point (known
//                      computational, leaked, or lost), so the driver
//                      draws it without touching quantum state.
//   fire               A consult that moves the qubit to another level:
//                      the stochastic event itself.
//   jump               A recorded fire -- the ResolvedJump entry in a
//                      shot's events.
//   site               An annotation target kept as a runtime INSTRUMENT
//                      barrier in the compiled module, consulted in-line
//                      by the VM; trace()'s materialization order
//                      assigns site ids.
//   trap               A fire the VM cannot resolve in-line -- the
//                      qubit's level is not definite there -- so
//                      execution stops at the site and control returns
//                      to the driver.
//   carrier            The simulated qubit cell that keeps holding a
//                      noncomputational population's correlations while
//                      the status ledger tracks its level classically.
//   continuation       The circuit rewritten under a shot's events
//                      (initial statuses, jumps so far, pre-drawn
//                      classical outcomes) and recompiled whole;
//                      execution resumes past the trapped site, the
//                      prefix guaranteed identical to what the shot
//                      already ran.
//
// The driver's own randomness (initial levels, trap destinations,
// classical consults, herald flags) is one per-shot stream, every draw
// ordered before any cache lookup; the VM's Born randomness is the
// separate SVM stream (seed.h).

#include "clifft/noncomp/exact_driver.h"

#include "clifft/backend/backend.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/rewriter.h"
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
// LEVEL_TRANSITION, a uniform loss rate for LOSS.
struct AnnotationChannel {
    const TransitionInstrument* instrument = nullptr;  // null for LOSS
    double loss_p = 0.0;
};

AnnotationChannel resolve_annotation(const AstNode& node, const NonComputationalModel& model,
                                     uint32_t op_index) {
    AnnotationChannel channel;
    if (node.gate == GateType::LOSS) {
        channel.loss_p = loss_probability(node.args, op_index, "sample_noncomputational");
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

// Draw one qubit's initial level from the model's shared initial-state
// distribution, with the last positive level catching the floating-point
// tail so a draw always resolves to a level the distribution can produce.
Level draw_initial_level(const NonComputationalModel& model, Xoshiro256PlusPlus& rng) {
    const double u = rng.next_double();
    double acc = 0.0;
    Level last_positive = Level::G;
    for (const Level l : kAllLevels) {
        const double p = model.initial_probability(l);
        if (p > 0.0) {
            last_positive = l;
        }
        acc += p;
        if (u < acc) {
            return l;
        }
    }
    return last_positive;
}

// Draw a destination level from `source`'s column of the instrument,
// restricted to the levels the pure `admit` predicate accepts; the draw
// normalizes over the admitted entries' own mass. Uses the measurement
// kernels' last-positive fallback for the floating-point tail, and
// consumes exactly one RNG draw.
template <typename Admit>
Level draw_from_column(const TransitionInstrument& instrument, Level source,
                       Xoshiro256PlusPlus& rng, Admit&& admit) {
    double mass = 0.0;
    for (const Level to : kAllLevels) {
        if (admit(to)) {
            mass += instrument.prob(to, source);
        }
    }
    if (!(mass > 0.0)) {
        // Unreachable through the current call sites, all of which fire
        // only when the column's admitted mass is positive; a real error
        // beats a Release-mode draw of a bogus level surfacing downstream.
        throw std::logic_error(
            "sample_noncomputational: destination draw over an empty column (source level '" +
            std::string(level_name(source)) + "')");
    }
    const double u = rng.next_double() * mass;
    double acc = 0.0;
    std::optional<Level> last_positive;
    for (const Level to : kAllLevels) {
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
    assert(last_positive.has_value() && "destination draw over an empty column");
    return *last_positive;
}

// Walk the annotated circuit's statuses under `events` and rebuild the
// classical-outcome stream in the walk's own (circuit) order: outcomes
// already drawn for a consult are reused by annotation target, and
// consults seen for the first time are drawn. The stream must be rebuilt,
// not appended: a trap turns the trapped qubit's later consults classical
// *between* previously recorded ones, and an append-only stream would
// replay old outcomes at the wrong targets. Mutates
// events.classical_outcomes; the walk decides which consults are
// classical, matching what rewrite_continuation validates.
void extend_classical_outcomes(const Circuit& annotated, ExactShotEvents& events,
                               const NonComputationalModel& model, Xoshiro256PlusPlus& rng) {
    std::map<std::pair<uint32_t, uint32_t>, Level> jump_dest;
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
                if (is_computational(pre)) {
                    const auto jump = jump_dest.find({op_index, qubit});
                    if (jump != jump_dest.end()) {
                        status[qubit] = status_for(jump->second);
                    }
                    continue;
                }
                const Level source = noncomp_level(pre);
                const auto seen = drawn.find({op_index, qubit});
                if (seen != drawn.end()) {
                    if (seen->second.source_level != source) {
                        throw std::logic_error(
                            "sample_noncomputational: classical outcome reuse at op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) +
                            " crossed a source-level change (drawn at level '" +
                            level_name(seen->second.source_level) + "', walk holds level '" +
                            level_name(source) + "')");
                    }
                    ordered.push_back(seen->second);
                    if (seen->second.destination.has_value()) {
                        status[qubit] = status_for(*seen->second.destination);
                    }
                    continue;
                }
                const AnnotationChannel channel = resolve_annotation(node, model, op_index);
                ClassicalOutcome outcome{op_index, qubit, std::nullopt, source};
                if (channel.instrument != nullptr) {
                    const double total = channel.instrument->column_sum(source);
                    if (rng.next_double() < total) {
                        outcome.destination = draw_from_column(*channel.instrument, source, rng,
                                                               [](Level) { return true; });
                    }
                } else if (!is_lost(pre) && rng.next_double() < channel.loss_p) {
                    // LOSS on a leaked (not lost) qubit can still vacate
                    // it; an already-lost qubit records a no-op outcome
                    // without spending a draw.
                    outcome.destination = Level::Lost;
                }
                ordered.push_back(outcome);
                if (outcome.destination.has_value()) {
                    status[qubit] = status_for(*outcome.destination);
                }
            }
            continue;
        }
        // Ordinary operations advance statuses through the shared walk --
        // the same walk the rewriter uses, now via a single helper.
        advance_ordinary_node(annotated.nodes[op_index], op_index, status, model.policy(),
                              "sample_noncomputational");
    }
    events.classical_outcomes = std::move(ordered);
}

// Cache key: the status-outcome delta. A computational initial status
// carries no level (the |1> preload is a per-shot frame bit, not a
// module property), so shots differing only in computational initials
// share one module by construction; leaked/lost initials contribute
// their level.
std::string cache_key(const ExactShotEvents& events) {
    std::string key;
    key.reserve(events.initial_status.size() * 2 + 10 + events.jumps.size() * 9 +
                events.classical_outcomes.size() * 10);
    for (const QubitStatus s : events.initial_status) {
        key.push_back(static_cast<char>(s));
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
        key.push_back(outcome.destination.has_value() ? 1 : 0);
        key.push_back(static_cast<char>(outcome.destination.value_or(Level::G)));
    }
    return key;
}

// One rewritten continuation plus its per-herald-flag compiled modules.
struct ContinuationEntry {
    ContinuationRewrite rw;
    std::map<std::vector<uint8_t>, CompiledModule> modules;
    // Hidden measurement slot trace() assigned to the forced trace-out reset
    // named by rw.forced_traceout_node. Populated on first compile;
    // subsequent flag-variant compiles assert it is consistent.
    std::optional<size_t> forced_traceout_slot;
    // The force_last flag this entry was created with: derivable from the
    // events (entries with no jumps are never forced; a jumped entry's force
    // equals the last jump's destination_pending flag), fixed per circuit +
    // model + policy.
    bool force_last = false;
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
// the forced collapse it is correlated with. One pipeline serves every
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

// Bytecode counterpart: the same record-order filter applied to the
// bytecode passes.
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

// The forced-outcome variant of a sampling measurement opcode -- same
// record slot, outcome read from state.forced_record instead of drawn
// -- or NUM_OPCODES when the opcode is not a sampling measurement.
Opcode forced_opcode(Opcode op) {
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

// Swap the trace-out's hidden measurement at `slot` to its forced
// opcode,
// so resume() collapses the trapped carrier to the source the trap
// reported (read from state.forced_record[slot]) instead of redrawing.
// Slot indices ride inside instruction payloads and are never renumbered
// by bytecode passes, so the slot is the durable identity here.
void swap_traceout_to_forced(CompiledModule& module, size_t slot) {
    size_t found = 0;
    for (Instruction& instr : module.bytecode) {
        const Opcode forced = forced_opcode(instr.opcode);
        if (forced == Opcode::NUM_OPCODES) {
            continue;
        }
        if (instr.classical.classical_idx == slot) {
            instr.opcode = forced;
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
    if (forced_opcode(fresh.opcode) != executed.opcode) {
        return false;
    }
    Instruction swapped = fresh;
    swapped.opcode = executed.opcode;
    return std::memcmp(&swapped, &executed, sizeof(Instruction)) == 0;
}

// Validate that the circuit is compatible with the model's leak/loss
// semantics. When the model can produce noncomputational qubits at all
// (no per-qubit tracking), the current restrictions are:
//
// - parity measurements (MPP) are not supported; the circuit is rejected.
// - a classifier is required if the circuit has any measurements.
void validate_model_contract(const Circuit& annotated, const NonComputationalModel& model) {
    // Determine whether the model can ever produce a noncomputational qubit.
    bool noncomp_capable = false;
    for (const Level l : kAllLevels) {
        if (category(l) != LevelCategory::Computational && model.initial_probability(l) > 0.0) {
            noncomp_capable = true;
            break;
        }
    }
    if (!noncomp_capable) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(annotated.nodes.size()); ++i) {
            const AstNode& node = annotated.nodes[i];
            if (node.gate == GateType::LOSS && !node.args.empty() && node.args[0] > 0.0) {
                noncomp_capable = true;
                break;
            }
            if (node.gate == GateType::LEVEL_TRANSITION) {
                const TransitionInstrument* instr = model.transition_named(node.tag);
                if (instr == nullptr) {
                    // Unresolvable tags were rejected by the annotation
                    // validation loop that runs before this helper; the
                    // null check is defensive for direct callers.
                    continue;
                }
                for (const Level src : kAllLevels) {
                    if (category(src) != LevelCategory::Computational) {
                        continue;
                    }
                    for (const Level dst : kAllLevels) {
                        if (category(dst) != LevelCategory::Computational &&
                            instr->prob(dst, src) > 0.0) {
                            noncomp_capable = true;
                            break;
                        }
                    }
                    if (noncomp_capable) {
                        break;
                    }
                }
            }
            if (noncomp_capable) {
                break;
            }
        }
    }

    if (!noncomp_capable) {
        return;
    }

    // Gate A: parity measurements are not supported under a capable model.
    for (uint32_t i = 0; i < static_cast<uint32_t>(annotated.nodes.size()); ++i) {
        const AstNode& node = annotated.nodes[i];
        // MPP has MULTI arity; MXX/MYY/MZZ desugar to MPP at parse time.
        if (node.gate == GateType::MPP) {
            throw std::invalid_argument(
                "sample_noncomputational: parity measurement 'MPP' at op " + std::to_string(i) +
                " is not supported under a model that can leak or lose qubits;"
                " expand the parity readout into an explicit ancilla circuit");
        }
    }

    // Gate B: a classifier is required when the circuit has any measurements.
    if (model.classifier() == nullptr) {
        for (const AstNode& node : annotated.nodes) {
            if (is_measurement(node.gate)) {
                throw std::invalid_argument(
                    "sample_noncomputational: this model can leak or lose qubits and the circuit"
                    " measures; a classifier is required to define what a measurement of a leaked"
                    " or lost qubit reads");
            }
        }
    }
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
    const MeasurementClassifier* classifier = model.classifier();
    const bool ternary = classifier != nullptr && classifier->has_herald();

    const Circuit annotated = annotate(circuit, model);

    // Validate every annotation up front -- tag resolution, LOSS shape --
    // so a malformed node fails here, deterministically, rather than on
    // whichever shot first traps or consults it classically: a live site
    // rides through the rewrite verbatim, and trace() must never be the
    // first to look at its arguments.
    for (uint32_t op_index = 0; op_index < static_cast<uint32_t>(annotated.nodes.size());
         ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        if (node.gate == GateType::LEVEL_TRANSITION || node.gate == GateType::LOSS) {
            resolve_annotation(node, model, op_index);
        }
    }

    // Capability contract: check gate A (parity measurements unsupported
    // with a capable model) and gate B (classifier required when the model
    // is capable and the circuit measures). Runs after annotation resolution
    // so LEVEL_TRANSITION tags are already validated.
    validate_model_contract(annotated, model);

    // Validation is shot-count independent: a zero-shot call checks the
    // circuit/model contract and returns empty results.
    if (shots == 0) {
        return result;
    }

    const InstrumentTraceOptions instrument_options = instrument_trace_options(model);

    std::map<std::string, ContinuationEntry> cache;

    // Level one: the rewrite for `events`, computed once per delta. It is
    // deterministic in the events and consumes no randomness, so a fetch
    // never perturbs sampling.
    auto get_entry = [&](const ExactShotEvents& events, bool force_last) -> ContinuationEntry& {
        const std::string key = cache_key(events);
        auto [it, inserted] = cache.try_emplace(key);
        ContinuationEntry& entry = it->second;
        if (inserted) {
            entry.force_last = force_last;
            entry.rw = rewrite_continuation(annotated, events, force_last, model);
        } else {
            assert(entry.force_last == force_last &&
                   "cache hit with mismatched force_last: entries with no jumps are never forced; "
                   "a jumped entry's force equals the last jump's destination_pending flag, "
                   "fixed per circuit+model+policy");
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
                    assert(m.noise_node.has_value() &&
                           "classified measurement must have a READOUT_NOISE node to patch for "
                           "herald");
                    patched.nodes[*m.noise_node].args[0] = 0.5;
                }
            }
            // When the rewrite names a forced trace-out node, ask trace()
            // to report the hidden slot it assigns to that reset.
            HirModule hir = [&]() -> HirModule {
                if (entry.rw.forced_traceout_node.has_value()) {
                    InstrumentTraceOptions opts = instrument_options;
                    opts.forced_traceout_node = entry.rw.forced_traceout_node;
                    return trace(patched, &opts);
                }
                return trace(patched, &instrument_options);
            }();
            if (entry.rw.forced_traceout_node.has_value()) {
                if (!hir.forced_traceout_slot.has_value()) {
                    throw std::logic_error(
                        "sample_noncomputational: trace() did not encounter the forced "
                        "trace-out reset at node " +
                        std::to_string(*entry.rw.forced_traceout_node) +
                        "; the rewrite and trace() must walk the same stream");
                }
                // First compile: record the slot. Later flag-variant
                // compiles: assert it is consistent (same circuit, same R
                // node, same hidden numbering -- the slot cannot differ).
                if (!entry.forced_traceout_slot.has_value()) {
                    entry.forced_traceout_slot = hir.forced_traceout_slot;
                } else {
                    assert(entry.forced_traceout_slot == hir.forced_traceout_slot &&
                           "flag-variant compile yielded a different forced_traceout_slot");
                }
            }
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
            if (module.instrument_offsets.size() != entry.rw.site_targets.size()) {
                throw std::logic_error(
                    "sample_noncomputational: compiled module has " +
                    std::to_string(module.instrument_offsets.size()) +
                    " instrument site(s) but the rewrite's site table has " +
                    std::to_string(entry.rw.site_targets.size()) +
                    "; the rewriter and trace() must elide zero-fire sites identically");
            }
            if (entry.rw.forced_traceout_node.has_value()) {
                swap_traceout_to_forced(module, *entry.forced_traceout_slot);
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
    no_events.initial_status.assign(circuit.num_qubits, QubitStatus::Computational);

    // The state is reused across shots (growth from trap continuations
    // amortizes to the chain maximum); a starting module that outgrows it
    // -- a rare noncomputational-initial shot -- rebuilds it instead,
    // since grow_for_continuation is trap-gated by design. Rebuilds size
    // to the running maxima, never to the triggering module: a starting
    // module can exceed on one axis (say, hidden record slots) while
    // being smaller on the other, and later shots reuse the state.
    auto make_state = [&](const CompiledModule& m, uint32_t peak_rank, uint32_t total_meas_slots) {
        return SchrodingerState(StateConfig{
            .peak_rank = peak_rank,
            .num_measurements = total_meas_slots,
            .num_qubits = m.num_qubits,
            .num_detectors = m.num_detectors,
            .num_observables = m.num_observables,
            .seed = 0});  // placeholder; reseed() supplies the per-shot seed before execution
    };
    uint32_t state_rank = 0;
    uint32_t state_slots = 0;
    std::optional<SchrodingerState> state_storage;

    for (uint32_t shot = 0; shot < shots; ++shot) {
        Xoshiro256PlusPlus driver_rng(derive_seed(global_seed, shot, kExactDriverDomain));

        ExactShotEvents events;
        events.initial_status.reserve(circuit.num_qubits);
        // The drawn levels stay driver-local: the status ledger carries no
        // level for computational qubits, and the only consumer of a
        // computational initial's identity is the |1> frame preload below.
        std::vector<Level> initial_levels;
        initial_levels.reserve(circuit.num_qubits);
        bool any_noncomp_initial = false;
        for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
            const Level level = draw_initial_level(model, driver_rng);
            initial_levels.push_back(level);
            events.initial_status.push_back(status_for(level));
            any_noncomp_initial |= !is_computational(events.initial_status.back());
        }

        // Forced-outcome buffer for trap-form trace-outs, one entry
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
        ContinuationEntry* entry = nullptr;
        CompiledModule* module = nullptr;
        // An all-computational, no-jump walk consumes no randomness and
        // moves no status: statuses leave Computational only via jumps or
        // noncomputational initials, and classical consults happen only for
        // noncomputational statuses.
        std::vector<QubitStatus> final_status;
        if (any_noncomp_initial) {
            extend_classical_outcomes(annotated, events, model, driver_rng);
            entry = &get_entry(events, false);
            module = get_module(*entry, flags_for(entry->rw), nullptr, 0);
            final_status = entry->rw.final_status;
        } else {
            entry = &get_entry(no_events, false);
            module = get_module(*entry, flags_for(entry->rw), nullptr, 0);
            final_status.assign(circuit.num_qubits, QubitStatus::Computational);
        }

        if (!state_storage.has_value()) {
            state_rank = module->peak_rank;
            state_slots = module->total_meas_slots;
            state_storage.emplace(make_state(*module, state_rank, state_slots));
        } else {
            state_storage->reset();
            if (module->peak_rank > state_rank || module->total_meas_slots > state_slots) {
                state_rank = std::max(state_rank, module->peak_rank);
                state_slots = std::max(state_slots, module->total_meas_slots);
                state_storage.emplace(make_state(*module, state_rank, state_slots));
            }
        }
        SchrodingerState& state = *state_storage;
        state.reseed(derive_seed(global_seed, shot, kExactSvmDomain));
        assert(state.meas_record.size() >= module->total_meas_slots &&
               "the rebuild block above guarantees meas_record capacity; "
               "meas_record never shrinks");
        // A |1> initial level is an X at time zero: a Pauli, so a
        // per-shot frame preload rather than a distinct module.
        for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
            if (initial_levels[q] == Level::E) {
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

            // Destination: at a trap-form site nothing was drawn, so
            // the driver draws over the full column (computational
            // destinations included); elsewhere the class is already
            // leaked/lost and only the level within the trap remainder
            // remains. The VM reports the collapsed source as its
            // computational axis index, which is the level index by
            // construction (G = 0, E = 1).
            assert(trap.source <= 1 &&
                   "the VM reports the collapsed source as a computational axis index");
            const Level source = static_cast<Level>(trap.source);
            Level dest = Level::Lost;
            if (channel.instrument != nullptr && trap.destination_pending) {
                dest = draw_from_column(*channel.instrument, source, driver_rng,
                                        [](Level) { return true; });
            } else if (channel.instrument != nullptr) {
                dest = draw_from_column(*channel.instrument, source, driver_rng, [](Level to) {
                    return category(to) != LevelCategory::Computational;
                });
            }

            events.jumps.push_back({op_index, qubit, dest});
            extend_classical_outcomes(annotated, events, model, driver_rng);

            // A trap-form fire hands its carrier over uncollapsed; the
            // continuation's trace-out is forced to the reported source,
            // read from forced_record at the slot trace() assigned to the
            // rewrite's named reset node. The forced value is per-shot
            // runtime state, so the module stays source-independent.
            const bool force = trap.destination_pending;
            const uint32_t prefix_end = module->instrument_offsets[trap.site_id] + 1;
            ContinuationEntry& next_entry = get_entry(events, force);
            CompiledModule* next_module =
                get_module(next_entry, flags_for(next_entry.rw), module, prefix_end);
            final_status = next_entry.rw.final_status;

            if (force) {
                // get_module always populates forced_traceout_slot before
                // returning when forced_traceout_node is set.
                assert(next_entry.forced_traceout_slot.has_value() &&
                       "forced trace-out slot not populated by get_module");
                const size_t slot = *next_entry.forced_traceout_slot;
                if (forced_buffer.size() <= slot) {
                    forced_buffer.resize(slot + 1, 0);
                }
                forced_buffer[slot] = trap.source;
                // The span must be re-pointed after any resize.
                state.forced_record = forced_buffer;
            }

            entry = &next_entry;
            module = next_module;
            // grow_for_continuation sizes the state to this module inside resume();
            // track it so a later starting-module rebuild never shrinks capacity
            // the chain already reached.
            state_rank = std::max(state_rank, module->peak_rank);
            state_slots = std::max(state_slots, module->total_meas_slots);
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
