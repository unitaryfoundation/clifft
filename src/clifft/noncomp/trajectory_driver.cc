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
#include "clifft/noncomp/continuation_prefix.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/seed.h"
#include "clifft/noncomp/status_walk.h"
#include "clifft/noncomp/transition_hooks.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_registry.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/svm/svm.h"
#include "clifft/svm/svm_math.h"
#include "clifft/util/xoshiro.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace clifft {

namespace {

enum class AnnotationChannelKind { Instrument, Leakage, Loss };

// The channel described by one transition annotation: a named instrument for
// LEVEL_TRANSITION or an inline probability for LEAKAGE/LOSS.
struct AnnotationChannel {
    [[nodiscard]] static AnnotationChannel for_instrument(const TransitionInstrument& instrument) {
        return AnnotationChannel(AnnotationChannelKind::Instrument, &instrument, 0.0);
    }

    [[nodiscard]] static AnnotationChannel for_leakage(double probability) {
        return AnnotationChannel(AnnotationChannelKind::Leakage, nullptr, probability);
    }

    [[nodiscard]] static AnnotationChannel for_loss(double probability) {
        return AnnotationChannel(AnnotationChannelKind::Loss, nullptr, probability);
    }

    AnnotationChannelKind kind;
    const TransitionInstrument* instrument;
    double probability;

  private:
    AnnotationChannel(AnnotationChannelKind kind_in, const TransitionInstrument* instrument_in,
                      double probability_in)
        : kind(kind_in), instrument(instrument_in), probability(probability_in) {}
};

[[nodiscard]] AnnotationChannel resolve_annotation(const AstNode& node,
                                                   const NonComputationalModel& model,
                                                   uint32_t op_index) {
    if (is_inline_noncomputational_annotation(node.gate)) {
        const double p = inline_transition_probability(node.gate, node.args, op_index,
                                                       "sample_noncomputational");
        return node.gate == GateType::LEAKAGE ? AnnotationChannel::for_leakage(p)
                                              : AnnotationChannel::for_loss(p);
    }
    if (node.gate != GateType::LEVEL_TRANSITION) {
        throw std::invalid_argument("sample_noncomputational: unsupported annotation '" +
                                    std::string(gate_name(node.gate)) + "' at op " +
                                    std::to_string(op_index));
    }
    const TransitionInstrument* instrument = model.transition_named(node.tag);
    if (instrument == nullptr) {
        throw std::invalid_argument("sample_noncomputational: LEVEL_TRANSITION[" + node.tag +
                                    "] at op " + std::to_string(op_index) +
                                    " does not name a transition in the model");
    }
    return AnnotationChannel::for_instrument(*instrument);
}

// Resolving validates the inline arguments or transition name; the remaining
// checks enforce the target shape expected by the driver and rewriter.
void validate_annotation(const AstNode& node, const NonComputationalModel& model, uint32_t op_index,
                         uint32_t num_qubits) {
    (void)resolve_annotation(node, model, op_index);
    if (node.targets.empty()) {
        throw std::invalid_argument(
            "sample_noncomputational: annotation '" + std::string(gate_name(node.gate)) +
            "' at op " + std::to_string(op_index) + " requires at least one qubit target");
    }
    std::unordered_set<uint32_t> seen_qubits;
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
        if (!seen_qubits.insert(target.value()).second) {
            throw std::invalid_argument("sample_noncomputational: annotation '" +
                                        std::string(gate_name(node.gate)) +
                                        "' repeats target qubit " + std::to_string(target.value()) +
                                        " at op " + std::to_string(op_index));
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
        if (is_noncomputational_annotation(gate)) {
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
                        switch (channel.kind) {
                            case AnnotationChannelKind::Instrument: {
                                const double total = channel.instrument->column_sum(source);
                                if (rng.next_double() < total) {
                                    outcome.destination =
                                        draw_from_column(*channel.instrument, source, rng,
                                                         [](Level) { return true; });
                                }
                                break;
                            }
                            case AnnotationChannelKind::Leakage:
                                // LEAKAGE is a no-op on every noncomputational source
                                // and spends no draw.
                                break;
                            case AnnotationChannelKind::Loss:
                                // LOSS can move a leaked qubit to Lost. An already-lost
                                // qubit records a no-op without spending a draw.
                                if (!is_lost(pre) && rng.next_double() < channel.probability) {
                                    outcome.destination = Level::Lost;
                                }
                                break;
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

// The compiled program and trajectory metadata needed while executing it.
// The rewritten Circuit and classifier patch table are discarded after
// compilation. Replacing an existing shot-specific continuation invalidates
// pointers and references into its module; the shared all-computational
// continuation persists across shots.
struct CompiledContinuation {
    CompiledModule module;
    std::vector<AnnotationTarget> site_targets;
    std::vector<QubitStatus> final_status;
    std::optional<size_t> forced_traceout_slot;
};

struct SamplingCompiledContinuation {
    sampling::ExecutablePlan program;
    std::vector<AnnotationTarget> site_targets;
    std::vector<QubitStatus> final_status;
    std::optional<size_t> forced_traceout_slot;
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
// record order and explicitly guarantee instrument-prefix stability. A
// trajectory may force a hidden trace-out measurement, so moving an entangled
// measurement before that collapse would change the result.
HirPassManager trajectory_hir_pass_manager() {
    HirPassManager pm;
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::HIR && info.default_enabled && is_trajectory_compatible(info)) {
            pm.add_pass(info.make_hir());
        }
    }
    return pm;
}

// Apply both trajectory requirements to bytecode passes.
BytecodePassManager trajectory_bytecode_pass_manager() {
    BytecodePassManager pm;
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::Bytecode && info.default_enabled &&
            is_trajectory_compatible(info)) {
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

void patch_heralded_measurements(Circuit& circuit,
                                 std::span<const ClassifiedMeasurement> measurements,
                                 std::span<const uint8_t> herald_flags) {
    if (herald_flags.size() != measurements.size()) {
        throw std::logic_error(
            "sample_noncomputational: continuation compile received " +
            std::to_string(herald_flags.size()) + " herald flag(s) for " +
            std::to_string(measurements.size()) +
            " classified measurement(s); every classified measurement needs exactly one flag");
    }
    for (size_t i = 0; i < herald_flags.size(); ++i) {
        if (herald_flags[i] == 0) {
            continue;
        }
        assert(measurements[i].noise_node.has_value() &&
               "classified measurement must have a READOUT_NOISE node to patch for herald");
        circuit.nodes[*measurements[i].noise_node].args[0] = 0.5;
    }
}

struct TracedContinuation {
    HirModule hir;
    std::optional<size_t> forced_traceout_slot;
};

TracedContinuation trace_continuation(Circuit circuit, std::optional<size_t> forced_traceout_node,
                                      const InstrumentTraceOptions& instrument_options) {
    std::optional<InstrumentTraceOptions> forced_trace_options;
    const InstrumentTraceOptions* trace_options = &instrument_options;
    if (forced_traceout_node.has_value()) {
        forced_trace_options = instrument_options;
        forced_trace_options->forced_traceout_node = forced_traceout_node;
        trace_options = &*forced_trace_options;
    }
    HirModule hir = trace(circuit, trace_options);
    std::optional<size_t> forced_traceout_slot;
    if (forced_traceout_node.has_value()) {
        if (!hir.forced_traceout_slot.has_value()) {
            throw std::logic_error(
                "sample_noncomputational: trace() did not encounter the forced trace-out reset "
                "at node " +
                std::to_string(*forced_traceout_node));
        }
        forced_traceout_slot = hir.forced_traceout_slot;
    }

#ifndef NDEBUG
    const std::vector<uint32_t> pre_pass_records = record_sequence(hir);
#endif
    trajectory_hir_pass_manager().run(hir);
#ifndef NDEBUG
    assert(record_sequence(hir) == pre_pass_records &&
           "a trajectory-pipeline HIR pass reordered or removed a record op");
#endif
    return TracedContinuation{
        .hir = std::move(hir),
        .forced_traceout_slot = forced_traceout_slot,
    };
}

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

CompiledContinuation compile_continuation(ContinuationRewrite rewrite,
                                          const std::vector<uint8_t>& herald_flags,
                                          const InstrumentTraceOptions& instrument_options,
                                          std::optional<uint32_t> max_rank,
                                          const CompiledModule* executed_prefix_module,
                                          uint32_t prefix_end) {
    Circuit patched = std::move(rewrite.circuit);
    patch_heralded_measurements(patched, rewrite.classified_measurements, herald_flags);
    TracedContinuation traced =
        trace_continuation(std::move(patched), rewrite.forced_traceout_node, instrument_options);
    CompiledModule module = lower(std::move(traced.hir));
    trajectory_bytecode_pass_manager().run(module);
    check_max_rank(module, max_rank);
    if (module.instrument_offsets.size() != rewrite.site_targets.size()) {
        throw std::logic_error("sample_noncomputational: compiled module has " +
                               std::to_string(module.instrument_offsets.size()) +
                               " instrument site(s) but the rewrite's site table has " +
                               std::to_string(rewrite.site_targets.size()) +
                               "; the rewriter and trace() must elide zero-fire sites identically");
    }
    if (traced.forced_traceout_slot.has_value()) {
        swap_traceout_to_forced(module, *traced.forced_traceout_slot);
    }

    if (executed_prefix_module != nullptr) {
        validate_continuation_prefix(module, *executed_prefix_module, prefix_end);
    }

    return CompiledContinuation{
        .module = std::move(module),
        .site_targets = std::move(rewrite.site_targets),
        .final_status = std::move(rewrite.final_status),
        .forced_traceout_slot = traced.forced_traceout_slot,
    };
}

void prepend_initial_excited_preparations(Circuit& circuit, std::span<const Level> initial_levels,
                                          std::optional<size_t>& forced_traceout_node) {
    std::vector<AstNode> preparations;
    for (uint32_t qubit = 0; qubit < initial_levels.size(); ++qubit) {
        if (initial_levels[qubit] == Level::E) {
            preparations.push_back(AstNode{GateType::X, {Target::qubit(qubit)}, {}, 0, {}});
        }
    }
    if (preparations.empty()) {
        return;
    }
    if (forced_traceout_node.has_value()) {
        *forced_traceout_node += preparations.size();
    }
    circuit.nodes.insert(circuit.nodes.begin(), std::make_move_iterator(preparations.begin()),
                         std::make_move_iterator(preparations.end()));
}

void check_sampling_max_rank(const sampling::SamplingPlan& plan, std::optional<uint32_t> max_rank) {
    if (!max_rank.has_value() || plan.max_active_width <= *max_rank) {
        return;
    }
    throw std::invalid_argument("sample_noncomputational: planned active width " +
                                std::to_string(plan.max_active_width) + " exceeds max_rank " +
                                std::to_string(*max_rank));
}

SamplingCompiledContinuation compile_sampling_continuation(
    ContinuationRewrite rewrite, const std::vector<uint8_t>& herald_flags,
    const InstrumentTraceOptions& instrument_options, std::optional<uint32_t> max_rank,
    std::span<const Level> initial_levels) {
    Circuit patched = std::move(rewrite.circuit);
    patch_heralded_measurements(patched, rewrite.classified_measurements, herald_flags);
    prepend_initial_excited_preparations(patched, initial_levels, rewrite.forced_traceout_node);
    TracedContinuation traced =
        trace_continuation(std::move(patched), rewrite.forced_traceout_node, instrument_options);
    sampling::SamplingPlan plan = sampling::plan_sampling(traced.hir);
    check_sampling_max_rank(plan, max_rank);
    if (plan.num_instrument_sites != rewrite.site_targets.size()) {
        throw std::logic_error("sample_noncomputational: sampling plan has " +
                               std::to_string(plan.num_instrument_sites) +
                               " instrument site(s) but the rewrite has " +
                               std::to_string(rewrite.site_targets.size()));
    }
    return SamplingCompiledContinuation{
        .program = sampling::ExecutablePlan(plan),
        .site_targets = std::move(rewrite.site_targets),
        .final_status = std::move(rewrite.final_status),
        .forced_traceout_slot = traced.forced_traceout_slot,
    };
}

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
            if (is_inline_noncomputational_annotation(node.gate) && !node.args.empty() &&
                node.args[0] > 0.0) {
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

Circuit prepare_trajectory_circuit(const Circuit& circuit, const NonComputationalModel& model) {
    Circuit annotated = expand_transition_hooks(circuit, model);
    for (uint32_t op_index = 0; op_index < static_cast<uint32_t>(annotated.nodes.size());
         ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        if (is_noncomputational_annotation(node.gate)) {
            validate_annotation(node, model, op_index, annotated.num_qubits);
        }
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
    validate_model_contract(annotated, model);
    return annotated;
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

    const Circuit annotated = prepare_trajectory_circuit(circuit, model);

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

    // The all-computational starting continuation is the common low-transition
    // path, so compile it lazily once and reuse it across shots. Other
    // continuations live only for the shot that reaches them.
    TrajectoryEvents no_events;
    no_events.initial_status.assign(circuit.num_qubits, QubitStatus::Computational);
    std::optional<CompiledContinuation> computational_start;

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

        // A noncomputational initial state needs a shot-specific rewrite. The
        // all-computational rewrite contains only VM instruments and is shared
        // across shots. It is compiled lazily so an unreachable starting state
        // cannot fail max_rank validation.
        std::optional<CompiledContinuation> shot_continuation;
        const CompiledContinuation* continuation = nullptr;
        if (any_noncomp_initial) {
            extend_classical_outcomes(annotated, events, model, driver_rng);
            ContinuationRewrite rewrite = rewrite_continuation(annotated, events, false, model);
            const std::vector<uint8_t> flags = flags_for(rewrite);
            shot_continuation.emplace(compile_continuation(
                std::move(rewrite), flags, instrument_options, max_rank, nullptr, 0));
            continuation = &*shot_continuation;
        } else {
            if (!computational_start.has_value()) {
                ContinuationRewrite rewrite =
                    rewrite_continuation(annotated, no_events, false, model);
                assert(rewrite.classified_measurements.empty() &&
                       "an all-computational continuation has no classified measurements");
                computational_start.emplace(compile_continuation(
                    std::move(rewrite), {}, instrument_options, max_rank, nullptr, 0));
            }
            continuation = &*computational_start;
        }

        if (!state_storage.has_value()) {
            state_rank = continuation->module.peak_rank;
            state_slots = continuation->module.total_meas_slots;
            state_storage.emplace(make_state(continuation->module, state_rank, state_slots));
        } else {
            state_storage->reset();
            if (continuation->module.peak_rank > state_rank ||
                continuation->module.total_meas_slots > state_slots) {
                state_rank = std::max(state_rank, continuation->module.peak_rank);
                state_slots = std::max(state_slots, continuation->module.total_meas_slots);
                state_storage.emplace(make_state(continuation->module, state_rank, state_slots));
            }
        }
        SchrodingerState& state = *state_storage;
        const auto sw = derive_state(root, shot, kTrajectoryExecutorDomain);
        state.reseed_full(sw[0], sw[1], sw[2], sw[3]);
        assert(state.meas_record.size() >= continuation->module.total_meas_slots &&
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
        state.draw_next_noise(continuation->module.constant_pool.noise_hazards);
        execute(continuation->module, state);

        while (state.pending_trap.has_value()) {
            const SchrodingerState::InstrumentTrap trap = *state.pending_trap;
            if (trap.site_id >= continuation->site_targets.size()) {
                throw std::logic_error(
                    "sample_noncomputational: trap site id is out of range of the executing "
                    "module's site table");
            }
            const auto [op_index, qubit] = continuation->site_targets[trap.site_id];
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
            Level dest;
            switch (channel.kind) {
                case AnnotationChannelKind::Instrument:
                    if (trap.destination_pending) {
                        dest = draw_from_column(*channel.instrument, source, driver_rng,
                                                [](Level) { return true; });
                    } else {
                        dest = draw_from_column(*channel.instrument, source, driver_rng,
                                                [](Level to) { return !is_computational(to); });
                    }
                    break;
                case AnnotationChannelKind::Leakage:
                    dest = source == Level::G ? Level::LeakG : Level::LeakE;
                    break;
                case AnnotationChannelKind::Loss:
                    dest = Level::Lost;
                    break;
            }

            events.jumps.push_back({{op_index, qubit}, dest});
            extend_classical_outcomes(annotated, events, model, driver_rng);

            // When destination_pending is true, the VM has not collapsed the
            // qubit. Force the continuation's hidden trace-out measurement to
            // the reported source. Supplying it through per-shot state keeps
            // the continuation bytecode independent of the g/e source.
            const bool force = trap.destination_pending;
            const uint32_t prefix_end = continuation->module.instrument_offsets[trap.site_id] + 1;

            // Rebuild from the original circuit using all events seen so far.
            // The new module reproduces the executed prefix and specializes
            // the remaining suffix; resume() skips the prefix and continues
            // immediately after this instrument.
            ContinuationRewrite next_rewrite =
                rewrite_continuation(annotated, events, force, model);
            const std::vector<uint8_t> next_flags = flags_for(next_rewrite);
            CompiledContinuation next =
                compile_continuation(std::move(next_rewrite), next_flags, instrument_options,
                                     max_rank, &continuation->module, prefix_end);

            if (force) {
                assert(next.forced_traceout_slot.has_value() &&
                       "forced trace-out slot not populated by compile_continuation");
                const size_t slot = *next.forced_traceout_slot;
                if (forced_buffer.size() <= slot) {
                    forced_buffer.resize(slot + 1, 0);
                }
                forced_buffer[slot] = trap.source;
                // The span must be re-pointed after any resize.
                state.forced_record = forced_buffer;
            }

            // Reusing one shot-specific slot bounds retained compiled programs
            // independently of the number of traps and shots. The assignment
            // destroys an existing shot-specific module, never the shared
            // all-computational start; compile_continuation has finished
            // reading the executed prefix before this point.
            shot_continuation = std::move(next);
            continuation = &*shot_continuation;
            // resume() grows the state for this continuation when needed. Track
            // the new capacities so rebuilding for a later shot never shrinks
            // them.
            state_rank = std::max(state_rank, continuation->module.peak_rank);
            state_slots = std::max(state_slots, continuation->module.total_meas_slots);
            resume(continuation->module, state,
                   continuation->module.instrument_offsets[trap.site_id] + 1);
        }

        // Emit the visible records. The noncomputational path never sets
        // expected_observables, so no reference normalization applies.
        result.measurements.insert(result.measurements.end(), state.meas_record.begin(),
                                   state.meas_record.begin() + circuit.num_measurements);
        result.detectors.insert(result.detectors.end(), state.det_record.begin(),
                                state.det_record.end());
        result.observables.insert(result.observables.end(), state.obs_record.begin(),
                                  state.obs_record.end());
        result.final_status.insert(result.final_status.end(), continuation->final_status.begin(),
                                   continuation->final_status.end());
        std::vector<uint8_t> shot_heralds(circuit.num_measurements, 0);
        for (const auto& [slot, flag] : herald_flags) {
            shot_heralds[slot] = flag;
        }
        result.heralds.insert(result.heralds.end(), shot_heralds.begin(), shot_heralds.end());
    }

    return result;
}

NonComputationalSample run_sampling_trajectory_driver(const Circuit& circuit,
                                                      const NonComputationalModel& model,
                                                      uint32_t shots, const SeedRoot& root,
                                                      std::optional<uint32_t> max_rank) {
    NonComputationalSample result;
    result.shots = shots;
    result.num_qubits = circuit.num_qubits;
    result.num_measurements = circuit.num_measurements;
    result.num_detectors = circuit.num_detectors;
    result.num_observables = circuit.num_observables;
    const MeasurementClassifier* classifier = model.classifier();
    const bool ternary = classifier != nullptr && classifier->has_herald();
    const Circuit annotated = prepare_trajectory_circuit(circuit, model);
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
    TrajectoryEvents no_events;
    no_events.initial_status.assign(circuit.num_qubits, QubitStatus::Computational);
    std::optional<SamplingCompiledContinuation> all_ground_start;
    std::optional<sampling::Executor> shared_executor;

    for (uint32_t shot = 0; shot < shots; ++shot) {
        const auto driver_words = derive_state(root, shot, kTrajectoryDriverDomain);
        Xoshiro256PlusPlus driver_rng(0);
        driver_rng.seed_full(driver_words[0], driver_words[1], driver_words[2], driver_words[3]);

        TrajectoryEvents events;
        events.initial_status.reserve(circuit.num_qubits);
        std::vector<Level> initial_levels;
        initial_levels.reserve(circuit.num_qubits);
        bool needs_shot_specific_start = false;
        for (uint32_t qubit = 0; qubit < circuit.num_qubits; ++qubit) {
            const Level level = draw_initial_level(model, driver_rng);
            initial_levels.push_back(level);
            events.initial_status.push_back(status_for(level));
            // Noncomputational levels change the rewrite. E remains
            // computational but needs an explicit preparation in this backend.
            needs_shot_specific_start |= level != Level::G;
        }

        std::map<uint32_t, uint8_t> herald_flags;
        auto flags_for = [&](const ContinuationRewrite& rewrite) {
            std::vector<uint8_t> flags;
            flags.reserve(rewrite.classified_measurements.size());
            for (const ClassifiedMeasurement& measurement : rewrite.classified_measurements) {
                auto [it, inserted] = herald_flags.try_emplace(measurement.slot, 0);
                if (inserted && ternary) {
                    const double probability =
                        classifier->prob(MeasurementClassifier::kHeraldSymbol, measurement.level);
                    it->second = driver_rng.next_double() < probability ? 1 : 0;
                }
                flags.push_back(it->second);
            }
            return flags;
        };

        const bool uses_shared_start = !needs_shot_specific_start;
        std::optional<SamplingCompiledContinuation> shot_start;
        const SamplingCompiledContinuation* continuation = nullptr;
        if (needs_shot_specific_start) {
            extend_classical_outcomes(annotated, events, model, driver_rng);
            ContinuationRewrite rewrite = rewrite_continuation(annotated, events, false, model);
            const std::vector<uint8_t> flags = flags_for(rewrite);
            shot_start.emplace(compile_sampling_continuation(
                std::move(rewrite), flags, instrument_options, max_rank, initial_levels));
            continuation = &*shot_start;
        } else {
            if (!all_ground_start.has_value()) {
                ContinuationRewrite rewrite =
                    rewrite_continuation(annotated, no_events, false, model);
                assert(rewrite.classified_measurements.empty() &&
                       "an all-ground continuation has no classified measurements");
                all_ground_start.emplace(compile_sampling_continuation(
                    std::move(rewrite), {}, instrument_options, max_rank, initial_levels));
            }
            continuation = &*all_ground_start;
        }

        std::unique_ptr<SamplingCompiledContinuation> shot_continuation;
        std::optional<sampling::Executor> shot_executor;
        sampling::Executor* executor = nullptr;
        if (uses_shared_start) {
            if (!shared_executor.has_value()) {
                shared_executor.emplace(continuation->program);
            }
            executor = &*shared_executor;
        } else {
            shot_executor.emplace(continuation->program);
            executor = &*shot_executor;
        }
        const auto executor_words = derive_state(root, shot, kTrajectoryExecutorDomain);
        executor->reseed_full(executor_words[0], executor_words[1], executor_words[2],
                              executor_words[3]);
        executor->run_shot();

        while (executor->pending_trap().has_value()) {
            const sampling::InstrumentTrap trap = *executor->pending_trap();
            const uint32_t site = sampling::index(trap.site);
            if (site >= continuation->site_targets.size()) {
                throw std::logic_error(
                    "sample_noncomputational: sampling trap site id is out of range");
            }
            const auto [op_index, qubit] = continuation->site_targets[site];
            const AstNode& node = annotated.nodes[op_index];
            const AnnotationChannel channel = resolve_annotation(node, model, op_index);

            assert(trap.source <= 1 && "the executor reports a computational source as G or E");
            const Level source = static_cast<Level>(trap.source);
            Level destination;
            switch (channel.kind) {
                case AnnotationChannelKind::Instrument:
                    if (trap.destination_pending) {
                        destination = draw_from_column(*channel.instrument, source, driver_rng,
                                                       [](Level) { return true; });
                    } else {
                        destination =
                            draw_from_column(*channel.instrument, source, driver_rng,
                                             [](Level level) { return !is_computational(level); });
                    }
                    break;
                case AnnotationChannelKind::Leakage:
                    destination = source == Level::G ? Level::LeakG : Level::LeakE;
                    break;
                case AnnotationChannelKind::Loss:
                    destination = Level::Lost;
                    break;
            }

            events.jumps.push_back({{op_index, qubit}, destination});
            extend_classical_outcomes(annotated, events, model, driver_rng);
            const bool force = trap.destination_pending;
            ContinuationRewrite next_rewrite =
                rewrite_continuation(annotated, events, force, model);
            const std::vector<uint8_t> next_flags = flags_for(next_rewrite);
            auto next = std::make_unique<SamplingCompiledContinuation>(
                compile_sampling_continuation(std::move(next_rewrite), next_flags,
                                              instrument_options, max_rank, initial_levels));

            std::optional<sampling::ForcedTraceOut> forced_trace_out;
            if (force) {
                if (!next->forced_traceout_slot.has_value() ||
                    *next->forced_traceout_slot > std::numeric_limits<uint32_t>::max()) {
                    throw std::logic_error(
                        "sample_noncomputational: sampling continuation omitted its forced "
                        "trace-out slot");
                }
                forced_trace_out = sampling::ForcedTraceOut{
                    sampling::RecordSlot{static_cast<uint32_t>(*next->forced_traceout_slot)},
                    trap.source};
            }
            executor->resume(next->program, forced_trace_out);
            shot_continuation = std::move(next);
            continuation = shot_continuation.get();
        }

        const std::span<const uint8_t> measurements = executor->visible_records();
        result.measurements.insert(result.measurements.end(), measurements.begin(),
                                   measurements.end());
        const std::span<const uint8_t> detectors = executor->detectors();
        result.detectors.insert(result.detectors.end(), detectors.begin(), detectors.end());
        const std::span<const uint8_t> observables = executor->observables();
        result.observables.insert(result.observables.end(), observables.begin(), observables.end());
        executor->return_to_root_plan();
        result.final_status.insert(result.final_status.end(), continuation->final_status.begin(),
                                   continuation->final_status.end());
        std::vector<uint8_t> shot_heralds(circuit.num_measurements, 0);
        for (const auto& [slot, flag] : herald_flags) {
            shot_heralds[slot] = flag;
        }
        result.heralds.insert(result.heralds.end(), shot_heralds.begin(), shot_heralds.end());
    }

    return result;
}

}  // namespace clifft
