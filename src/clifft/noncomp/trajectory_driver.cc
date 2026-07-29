// Drives simulation of circuits with noncomputational transitions.
//
// Transition annotations compile to VM instructions. The VM applies an outcome
// directly when the current compiled circuit can continue from it. Otherwise it
// stops at the transition and reports a trap. The driver records the event,
// rewrites and compiles a compatible continuation, and resumes the existing VM
// state after the trapped instruction. Transitions reached while a qubit is
// already leaked or lost are sampled here before compiling the next
// continuation.
//
// Driver draws (initial levels, transition destinations, classical outcomes,
// and herald flags) use one per-shot stream. VM measurements, noise, and
// in-VM transition decisions use a separate per-shot stream; see seed.h.

#include "clifft/noncomp/trajectory_driver.h"

#include "clifft/backend/backend.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/seed.h"
#include "clifft/noncomp/status_walk.h"
#include "clifft/noncomp/transition_hooks.h"
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

// The channel described by one transition annotation: a named instrument for
// LEVEL_TRANSITION or a uniform loss probability for LOSS. A null instrument
// identifies the LOSS representation.
struct AnnotationChannel {
    const TransitionInstrument* instrument;
    double loss_probability;

    bool is_loss() const { return instrument == nullptr; }
};

[[nodiscard]] AnnotationChannel resolve_annotation(const AstNode& node,
                                                   const NonComputationalModel& model,
                                                   uint32_t op_index) {
    if (node.gate == GateType::LOSS) {
        return {nullptr, loss_probability(node.args, op_index, "sample_noncomputational")};
    }
    const TransitionInstrument* instrument = model.transition_named(node.tag);
    if (instrument == nullptr) {
        throw std::invalid_argument("sample_noncomputational: LEVEL_TRANSITION[" + node.tag +
                                    "] at op " + std::to_string(op_index) +
                                    " does not name a transition in the model");
    }
    return {instrument, 0.0};
}

// Resolving validates the LOSS arguments or transition name; the remaining
// checks enforce the target shape expected by the driver and rewriter.
void validate_annotation(const AstNode& node, const NonComputationalModel& model, uint32_t op_index,
                         uint32_t num_qubits) {
    (void)resolve_annotation(node, model, op_index);
    if (node.targets.empty()) {
        throw std::invalid_argument(
            "sample_noncomputational: annotation '" + std::string(gate_name(node.gate)) +
            "' at op " + std::to_string(op_index) + " requires at least one qubit target");
    }
    for (const Target& target : node.targets) {
        if (target.is_rec() || target.has_pauli() || target.is_inverted()) {
            throw std::invalid_argument("sample_noncomputational: annotation '" +
                                        std::string(gate_name(node.gate)) + "' at op " +
                                        std::to_string(op_index) + " requires plain qubit targets");
        }
        if (target.value() >= num_qubits) {
            throw std::invalid_argument("sample_noncomputational: annotation '" +
                                        std::string(gate_name(node.gate)) + "' target qubit " +
                                        std::to_string(target.value()) + " is out of range at op " +
                                        std::to_string(op_index));
        }
    }
}

// Draw one qubit's initial level. If floating-point rounding leaves the draw
// just past the accumulated total, return the last level with nonzero
// probability.
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

// Draw among the destination levels accepted by `admit`, renormalizing their
// probabilities. If rounding leaves the draw just past the accumulated total,
// return the last accepted level with nonzero probability. Consumes one RNG
// draw. `admit(Level)` returns true for destinations included in the draw.
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
        // Current callers only draw when the accepted probability is positive.
        // Keep a runtime error here so release builds cannot continue with an
        // invalid destination.
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

// Rewalk the annotated circuit under `events`, reusing previously sampled
// outcomes by annotation target and sampling outcomes for annotations whose
// source level is now known. Rebuild the outcome vector in circuit order
// instead of appending: a trap at op 2 can make op 3 driver-resolvable even
// if op 5 was already sampled, changing the required order from [op 5] to
// [op 3, op 5]. The rewriter consumes this vector sequentially.
void extend_classical_outcomes(const Circuit& annotated, TrajectoryEvents& events,
                               const NonComputationalModel& model, Xoshiro256PlusPlus& rng) {
    std::map<AnnotationTarget, Level> jump_dest;
    for (const ResolvedJump& jump : events.jumps) {
        jump_dest.emplace(jump.target, jump.destination_level);
    }

    std::map<AnnotationTarget, ClassicalOutcome> drawn;
    for (const ClassicalOutcome& outcome : events.classical_outcomes) {
        drawn.emplace(outcome.target, outcome);
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
                } else {
                    const Level source = noncomp_level(pre);
                    const AnnotationTarget annotation_target{op_index, qubit};
                    const auto seen = drawn.find(annotation_target);
                    ClassicalOutcome outcome{annotation_target, std::nullopt, source};
                    if (seen != drawn.end()) {
                        if (seen->second.source_level != source) {
                            throw std::logic_error(
                                "sample_noncomputational: classical outcome reuse at op " +
                                std::to_string(op_index) + ", qubit " + std::to_string(qubit) +
                                " no longer matches the qubit's source level (drawn at level '" +
                                level_name(seen->second.source_level) + "', qubit now has level '" +
                                level_name(source) + "')");
                        }
                        outcome = seen->second;
                    } else {
                        const AnnotationChannel channel = resolve_annotation(node, model, op_index);
                        if (!channel.is_loss()) {
                            const double total = channel.instrument->column_sum(source);
                            if (rng.next_double() < total) {
                                outcome.destination = draw_from_column(
                                    *channel.instrument, source, rng, [](Level) { return true; });
                            }
                        } else if (!is_lost(pre) && rng.next_double() < channel.loss_probability) {
                            // LOSS can move a leaked qubit to Lost. An already-lost
                            // qubit records a no-op without spending a draw.
                            outcome.destination = Level::Lost;
                        }
                    }
                    ordered.push_back(outcome);
                    if (outcome.destination.has_value()) {
                        status[qubit] = status_for(*outcome.destination);
                    }
                }
            }
        } else {
            // Apply the same status update the rewriter uses for ordinary
            // operations.
            advance_ordinary_node(node, op_index, status, model.policy(),
                                  "sample_noncomputational");
        }
    }
    events.classical_outcomes = std::move(ordered);
}

// Build the continuation cache key from initial statuses, recorded jumps, and
// driver-drawn transition outcomes. QubitStatus::Computational does not
// distinguish g from e, so those initial levels share a compiled module; an
// initial e is supplied separately through the shot's Pauli frame.
std::string cache_key(const TrajectoryEvents& events) {
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
    // Prefix each variable-length section with its count so different event
    // sequences cannot produce the same key. J and C only make the encoding
    // easier to inspect.
    key.push_back('J');
    push32(static_cast<uint32_t>(events.jumps.size()));
    for (const ResolvedJump& jump : events.jumps) {
        push32(jump.target.op_index);
        push32(jump.target.qubit);
        key.push_back(static_cast<char>(jump.destination_level));
    }
    key.push_back('C');
    push32(static_cast<uint32_t>(events.classical_outcomes.size()));
    for (const ClassicalOutcome& outcome : events.classical_outcomes) {
        push32(outcome.target.op_index);
        push32(outcome.target.qubit);
        key.push_back(outcome.destination.has_value() ? 1 : 0);
        key.push_back(static_cast<char>(outcome.destination.value_or(Level::G)));
    }
    return key;
}

// One rewritten continuation and its compiled variants for different herald
// flags.
struct ContinuationEntry {
    ContinuationRewrite rw;
    std::map<std::vector<uint8_t>, CompiledModule> modules;
    // Hidden measurement slot assigned to rw.forced_traceout_node. The first
    // compile records it; later herald variants must assign the same slot.
    std::optional<size_t> forced_traceout_slot;
    // Whether the last jump's trace-out must use a source chosen by the VM.
    // This is false when there are no jumps and otherwise equals the last
    // trap's destination_pending value.
    bool force_last = false;
};

void check_max_rank(const CompiledModule& module, std::optional<uint32_t> max_rank) {
    if (!max_rank.has_value() || module.peak_rank <= *max_rank) {
        return;
    }
    // Use the source map to name the first circuit line that exceeded the
    // requested rank.
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

// Build the HIR pass pipeline from default passes that preserve measurement
// record order. A trajectory may force a hidden trace-out measurement, so moving
// an entangled measurement before that collapse would change the result. The
// same pipeline is used for every continuation to preserve matching prefixes.
HirPassManager trajectory_hir_pass_manager() {
    HirPassManager pm;
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::HIR && info.default_enabled && info.record_order.preserved) {
            pm.add_pass(info.make_hir());
        }
    }
    return pm;
}

// Apply the same record-order requirement to bytecode passes.
BytecodePassManager trajectory_bytecode_pass_manager() {
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
// Measurement record indices in program order, including hidden slots. Debug
// builds compare this sequence before and after optimization to catch a pass
// that incorrectly claims to preserve record order.
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

// A trap-form instrument reports a source but leaves the qubit uncollapsed.
// The rewritten continuation inserts a reset to apply that collapse and
// prepare the selected destination. Its hidden measurement must reuse the
// source already selected by the VM rather than draw a second outcome, so
// replace that measurement with its forced-outcome opcode. Bytecode passes do
// not renumber measurement slots, so `slot` remains valid after optimization.
void swap_traceout_to_forced(CompiledModule& module, size_t slot) {
    size_t found = 0;
    for (Instruction& instr : module.bytecode) {
        const std::optional<Opcode> forced = forced_measurement_opcode(instr.opcode);
        if (!forced.has_value()) {
            continue;
        }
        if (instr.classical.classical_idx == slot) {
            instr.opcode = *forced;
            ++found;
        }
    }
    if (found != 1) {
        throw std::logic_error("sample_noncomputational: the forced trace-out slot matched " +
                               std::to_string(found) +
                               " measurement instructions; expected exactly one");
    }
}

#ifndef NDEBUG
// Compare instructions while allowing the expected sampling-to-forced opcode
// change. Bytewise equality is deliberate: resume requires the continuation
// prefix to match the module that already executed, not merely to be
// semantically equivalent.
bool equal_modulo_forced_swap(const Instruction& fresh, const Instruction& executed) {
    if (std::memcmp(&fresh, &executed, sizeof(Instruction)) == 0) {
        return true;
    }
    const std::optional<Opcode> forced = forced_measurement_opcode(fresh.opcode);
    if (!forced.has_value() || *forced != executed.opcode) {
        return false;
    }
    Instruction swapped = fresh;
    swapped.opcode = executed.opcode;
    return std::memcmp(&swapped, &executed, sizeof(Instruction)) == 0;
}
#endif

// Validate the circuit features that require model-wide knowledge. If the
// model can ever leak or lose a qubit, parity measurements are unsupported and
// ordinary measurements require a classifier.
void validate_model_contract(const Circuit& annotated, const NonComputationalModel& model) {
    // Determine whether the model can ever produce a noncomputational qubit.
    bool noncomp_capable = false;
    for (const Level l : kAllLevels) {
        if (!is_computational(l) && model.initial_probability(l) > 0.0) {
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
                    if (!is_computational(src)) {
                        continue;
                    }
                    for (const Level dst : kAllLevels) {
                        if (!is_computational(dst) && instr->prob(dst, src) > 0.0) {
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

    // A parity measurement has no defined result once an operand leaves the
    // computational subspace.
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

    // A physical-qubit measurement needs a classifier for any leaked or lost
    // operand. MPAD only appends a classical literal to the record.
    if (model.classifier() == nullptr) {
        for (const AstNode& node : annotated.nodes) {
            if (is_measurement(node.gate) && node.gate != GateType::MPAD) {
                throw std::invalid_argument(
                    "sample_noncomputational: this model can leak or lose qubits and the circuit"
                    " measures; a classifier is required to define what a measurement of a leaked"
                    " or lost qubit reads");
            }
        }
    }
}

}  // namespace

NonComputationalSample run_trajectory_driver(const Circuit& circuit,
                                             const NonComputationalModel& model, uint32_t shots,
                                             const SeedRoot& root,
                                             std::optional<uint32_t> max_rank) {
    NonComputationalSample result;
    result.shots = shots;
    result.num_qubits = circuit.num_qubits;
    result.num_measurements = circuit.num_measurements;
    result.num_detectors = circuit.num_detectors;
    result.num_observables = circuit.num_observables;
    const MeasurementClassifier* classifier = model.classifier();
    const bool ternary = classifier != nullptr && classifier->has_herald();

    const Circuit annotated = expand_transition_hooks(circuit, model);

    // Validate transition annotations and measurement shapes before sampling so
    // malformed input fails independently of the path taken by any shot.
    // Computational transitions can reach trace() unchanged, so they must be
    // checked here first.
    for (uint32_t op_index = 0; op_index < static_cast<uint32_t>(annotated.nodes.size());
         ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        if (node.gate == GateType::LEVEL_TRANSITION || node.gate == GateType::LOSS) {
            validate_annotation(node, model, op_index, annotated.num_qubits);
        }
        // The parser emits one node per target for ordinary measurements, and
        // the rewrite relies on that shape when assigning record slots. MPP is
        // intentionally multi-target and still produces one bit. MPAD also
        // produces one bit per node.
        if (is_measurement(node.gate) && node.gate != GateType::MPP && node.targets.size() != 1) {
            throw std::invalid_argument(
                "sample_noncomputational: measurement '" + std::string(gate_name(node.gate)) +
                "' at op " + std::to_string(op_index) + " carries " +
                std::to_string(node.targets.size()) + " targets; the parser" +
                " emits one measurement node per target and the rewrite's" +
                " record accounting relies on that shape; split the node" +
                " into single-target measurements");
        }
    }

    // Check model-wide measurement restrictions after annotations resolve.
    validate_model_contract(annotated, model);

    // Validation is shot-count independent: a zero-shot call checks the
    // circuit/model contract and returns empty results.
    if (shots == 0) {
        return result;
    }

    const size_t shot_count = shots;
    result.measurements.reserve(shot_count * circuit.num_measurements);
    result.detectors.reserve(shot_count * circuit.num_detectors);
    result.observables.reserve(shot_count * circuit.num_observables);
    result.final_status.reserve(shot_count * circuit.num_qubits);
    result.heralds.reserve(shot_count * circuit.num_measurements);

    const InstrumentTraceOptions instrument_options = instrument_trace_options(model);

    std::map<std::string, ContinuationEntry> cache;

    // Cache one rewrite for each event record. Rewriting is deterministic and
    // consumes no randomness, so cache hits do not change sampling.
    auto get_entry = [&](const TrajectoryEvents& events, bool force_last) -> ContinuationEntry& {
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

    // Within each rewrite, cache one compiled module for each set of herald
    // flags. All random draws used by this function happen before the lookup,
    // so hits and misses consume the same randomness.
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
                // Record the hidden slot on the first compile. Later herald
                // variants have the same reset and must assign the same slot.
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
            trajectory_hir_pass_manager().run(hir);
#ifndef NDEBUG
            assert(record_sequence(hir) == pre_pass_records &&
                   "a trajectory-pipeline HIR pass reordered or removed a record op");
#endif
            CompiledModule module = lower(hir);
            trajectory_bytecode_pass_manager().run(module);
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
            // The continuation must reproduce every instruction the shot
            // already executed, except for the forced trace-out opcode handled
            // above. Check this in debug builds before resuming.
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

    // Common starting continuation for a shot with computational initial
    // statuses and no recorded jumps.
    TrajectoryEvents no_events;
    no_events.initial_status.assign(circuit.num_qubits, QubitStatus::Computational);

    // Reuse one state across shots. resume() can grow it for larger
    // continuations. A starting module cannot use that trap-only growth path,
    // so rebuild the state when it needs more rank or measurement slots. Keep
    // the largest capacity seen in either dimension so later shots do not
    // shrink it.
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
        const auto dw = derive_state(root, shot, kTrajectoryDriverDomain);
        Xoshiro256PlusPlus driver_rng(0);
        driver_rng.seed_full(dw[0], dw[1], dw[2], dw[3]);

        TrajectoryEvents events;
        events.initial_status.reserve(circuit.num_qubits);
        // QubitStatus::Computational does not distinguish g from e. Keep the
        // sampled levels separately so an initial e can set the Pauli frame
        // before execution.
        std::vector<Level> initial_levels;
        initial_levels.reserve(circuit.num_qubits);
        bool any_noncomp_initial = false;
        for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
            const Level level = draw_initial_level(model, driver_rng);
            initial_levels.push_back(level);
            events.initial_status.push_back(status_for(level));
            any_noncomp_initial |= !is_computational(events.initial_status.back());
        }

        // Source levels for trace-out measurements that the continuation must
        // force. The vector index is the hidden measurement slot.
        std::vector<uint8_t> forced_buffer;

        // Draw each visible slot's herald flag only the first time that
        // classified measurement appears in a continuation.
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

        // Select the module used at the start of the shot. If any qubit starts
        // leaked or lost, draw all transitions reached in that state before
        // fetching its rewrite. Herald draws also happen before the compiled
        // module lookup so caching cannot change the random stream.
        ContinuationEntry* entry = nullptr;
        CompiledModule* module = nullptr;
        // With computational initial statuses and no jumps, every transition
        // remains a VM instrument and all final statuses stay Computational.
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
        const auto sw = derive_state(root, shot, kTrajectorySvmDomain);
        state.reseed_full(sw[0], sw[1], sw[2], sw[3]);
        assert(state.meas_record.size() >= module->total_meas_slots &&
               "the rebuild block above guarantees meas_record capacity; "
               "meas_record never shrinks");
        // Represent an initial e as an X in the per-shot Pauli frame instead of
        // compiling a separate module.
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

            // If destination_pending is true, the VM selected only the source,
            // so draw from the full transition column. Otherwise the VM already
            // constrained the destination to leaked or lost levels, and the
            // driver selects the specific level when needed. Source indices 0
            // and 1 correspond to g and e.
            assert(trap.source <= 1 &&
                   "the VM reports the collapsed source as a computational axis index");
            const Level source = static_cast<Level>(trap.source);
            Level dest = Level::Lost;
            if (!channel.is_loss() && trap.destination_pending) {
                dest = draw_from_column(*channel.instrument, source, driver_rng,
                                        [](Level) { return true; });
            } else if (!channel.is_loss()) {
                dest = draw_from_column(*channel.instrument, source, driver_rng,
                                        [](Level to) { return !is_computational(to); });
            }

            events.jumps.push_back({{op_index, qubit}, dest});
            extend_classical_outcomes(annotated, events, model, driver_rng);

            // When destination_pending is true, the VM has not collapsed the
            // qubit. Force the continuation's hidden trace-out measurement to
            // the reported source. The source is supplied through per-shot
            // state, so the compiled module can be shared by g and e outcomes.
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
            // resume() grows the state for this continuation when needed. Track
            // the new capacities so rebuilding for a later shot never shrinks
            // them.
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
