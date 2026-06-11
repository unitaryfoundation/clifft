#include "clifft/noncomp/orchestrator.h"

#include "clifft/backend/backend.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/noncomp/numeric.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sampler.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"
#include "clifft/util/xoshiro.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

// Distinct domain tags so per-shot sub-seeds for the three RNG consumers
// never coincide (seeding all from the same value would correlate them).
constexpr uint64_t kHistoryDomain = 0x1;
constexpr uint64_t kClassifierDomain = 0x2;
constexpr uint64_t kSvmDomain = 0x3;

// SplitMix64 finalizer over a mix of the global seed, shot, and domain.
uint64_t derive_seed(uint64_t global, uint64_t shot, uint64_t domain) {
    uint64_t z = global ^ (shot * 0x9E3779B97F4A7C15ULL) ^ (domain * 0xBF58476D1CE4E5B9ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One sampled classifier outcome for a measurement on a noncomputational
// qubit: the injected record bit, and whether the herald symbol fired.
struct ClassifierDraw {
    uint8_t bit;
    bool herald;
};

// Sample a classifier outcome for a qubit at `level`. The classifier must
// have two or three symbols and a stochastic column for this level. Symbols
// 0 and 1 map directly to the record bit. A third symbol heralds the
// measurement: the herald flag is reported in the sidecar and the record bit
// is drawn uniformly -- the slot still feeds detectors, and a heralded
// outcome carries no preferred computational value, so a uniform bit keeps
// downstream detector statistics unbiased rather than silently pinning them.
// A substochastic (reject) column is unsupported here.
ClassifierDraw classifier_draw(const MeasurementClassifier& classifier, uint8_t level,
                               Xoshiro256PlusPlus& rng, GateType gate, uint32_t op_index,
                               uint32_t qubit) {
    // Record injection writes one visible bit. Symbols 0/1 are that bit; a
    // third symbol is the herald. A still-richer alphabet has no defined
    // mapping onto (bit, herald) and is not representable.
    if (classifier.num_symbols() != 2 && classifier.num_symbols() != 3) {
        throw std::invalid_argument(
            "sample_noncomputational: injecting a measurement on a noncomputational qubit "
            "requires a two- or three-symbol classifier, but the model's has " +
            std::to_string(classifier.num_symbols()) + " symbols (measurement '" +
            std::string(gate_name(gate)) + "' on qubit " + std::to_string(qubit) + " at op " +
            std::to_string(op_index) + ")");
    }
    // A substochastic column reserves probability for a reject (heralded abort)
    // outcome that has no binary record bit. That path is not wired through this
    // entry point yet, so the column must sum to one within tolerance; a reject
    // column is an explicit error rather than a silently dropped or aborted shot.
    const double reject = classifier.reject_probability(level);
    if (reject > kProbTolerance) {
        throw std::invalid_argument(
            "sample_noncomputational: classifier reject columns are not supported yet; the column "
            "for level " +
            std::to_string(level) + " sums to less than one (reject probability " +
            std::to_string(reject) + ", measurement '" + std::string(gate_name(gate)) +
            "' on qubit " + std::to_string(qubit) + " at op " + std::to_string(op_index) + ")");
    }
    const double u = rng.next_double();
    // The column sums to one (checked above): symbol 0 owns [0, prob0),
    // symbol 1 the next interval, and the herald symbol (if any) the
    // remainder. Any tolerance-sized rounding residual falls to the last
    // symbol. A two-symbol draw consumes exactly one random number, so
    // existing two-symbol seeds reproduce.
    if (u < classifier.prob(0, level)) {
        return {0, false};
    }
    if (classifier.num_symbols() == 2 ||
        u < classifier.prob(0, level) + classifier.prob(1, level)) {
        return {1, false};
    }
    return {static_cast<uint8_t>(rng.next_double() < 0.5 ? 0 : 1), true};
}

GateType reset_for(GateType measure_reset) {
    switch (measure_reset) {
        case GateType::MRX:
            return GateType::RX;
        case GateType::MRY:
            return GateType::RY;
        default:
            return GateType::R;  // MR
    }
}

AstNode single_target(GateType gate, uint32_t value) {
    return AstNode{gate, {Target::qubit(value)}, {}, 0};
}

// Replay `original` + `history` to find each leaked/lost measurement's visible
// record slot and its sampled classifier outcome, then swap those slots in
// `rewritten`: M -> MPAD(bit); a measure-and-reset -> MPAD(bit) plus the
// matching reset. Herald flags are written into `heralds` (pre-sized to the
// visible measurement count, zero-filled) at the same slot indices. The
// rewriter has already guaranteed only M / measure-reset measurements reach a
// noncomputational operand.
Circuit inject_classifier(const Circuit& original, const Circuit& rewritten,
                          const NonComputationalHistory& history,
                          const NonComputationalModel& model, Xoshiro256PlusPlus& rng,
                          std::vector<uint8_t>& heralds) {
    const LevelSet& levels = model.levels();
    const NonComputationalPolicy& policy = model.policy();
    const MeasurementClassifier* classifier = model.classifier();

    std::map<uint32_t, ClassifierDraw> slot_to_bit;
    std::vector<QubitStatus> status = history.initial_status;
    size_t trans_cursor = 0;
    uint32_t slot = 0;
    for (uint32_t op_index = 0; op_index < original.nodes.size(); ++op_index) {
        const AstNode& node = original.nodes[op_index];
        const GateType gate = node.gate;
        const TransitionInstrument* instrument = model.transition_for(gate);
        const bool measurement = is_measurement(gate);
        // Mirror the sampler/rewriter policy pre-scan so all three trajectory
        // replays advance statuses identically when an operation is dropped.
        bool drop_op = false;
        for (const QubitOperand& operand : qubit_operands(node)) {
            if (operand_action(gate, status[operand.qubit].kind(), policy) == OperandAction::Drop) {
                drop_op = true;
            }
        }
        for (const QubitOperand& operand : qubit_operands(node)) {
            const uint32_t qubit = operand.qubit;
            const QubitStatus pre = status[qubit];
            TransitionOutcome outcome;
            if (operand.role == OperandRole::Physical && instrument != nullptr) {
                const TransitionRecord& record = history.transitions[trans_cursor++];
                outcome.jumped = record.jumped;
                outcome.destination_level = record.destination_level;
            }
            if (measurement &&
                (pre.kind() == QubitStatusKind::Leaked || pre.kind() == QubitStatusKind::Lost)) {
                if (classifier == nullptr) {
                    throw std::invalid_argument(
                        "sample_noncomputational: measurement '" + std::string(gate_name(gate)) +
                        "' on a noncomputational qubit " + std::to_string(qubit) + " at op " +
                        std::to_string(op_index) +
                        " requires a classifier, but the model has none");
                }
                const ClassifierDraw draw =
                    classifier_draw(*classifier, pre.level_id(), rng, gate, op_index, qubit);
                slot_to_bit[slot] = draw;
                heralds[slot] = draw.herald ? 1 : 0;
            }
            status[qubit] = drop_op ? step_status_dropped(pre, outcome, levels)
                                    : step_status(pre, gate, operand.role, outcome, policy, levels);
        }
        if (measurement) {
            ++slot;
        }
    }

    if (slot_to_bit.empty()) {
        return rewritten;
    }

    Circuit out = rewritten;
    out.nodes.clear();
    out.nodes.reserve(rewritten.nodes.size() + slot_to_bit.size());
    uint32_t out_slot = 0;
    for (const AstNode& node : rewritten.nodes) {
        if (!is_measurement(node.gate)) {
            out.nodes.push_back(node);
            continue;
        }
        auto it = slot_to_bit.find(out_slot);
        ++out_slot;
        if (it == slot_to_bit.end()) {
            out.nodes.push_back(node);  // computational measurement, kept as-is
            continue;
        }
        const uint8_t bit = it->second.bit;
        const uint32_t qubit = qubit_operands(node).front().qubit;
        out.nodes.push_back(single_target(GateType::MPAD, bit));
        if (is_measure_reset(node.gate)) {
            out.nodes.push_back(single_target(reset_for(node.gate), qubit));
        }
    }
    return out;
}

CompiledModule compile_circuit(const Circuit& circuit) {
    HirModule hir = trace(circuit);
    default_hir_pass_manager().run(hir);
    CompiledModule program = lower(hir);
    default_bytecode_pass_manager().run(program);
    return program;
}

}  // namespace

NonComputationalSample sample_noncomputational(const Circuit& circuit,
                                               const NonComputationalModel& model, uint32_t shots,
                                               std::optional<uint64_t> seed) {
    NonComputationalSample result;
    result.shots = shots;
    result.num_qubits = circuit.num_qubits;
    // The visible record layout is invariant under rewriting and injection, so
    // the record widths come straight from the circuit and hold even for zero
    // shots.
    result.num_measurements = circuit.num_measurements;
    result.num_detectors = circuit.num_detectors;
    result.num_observables = circuit.num_observables;

    if (circuit.num_exp_vals != 0) {
        throw std::invalid_argument(
            "sample_noncomputational: EXP_VAL probes are not supported in noncomputational "
            "sampling");
    }
    if (shots == 0) {
        return result;
    }

    uint64_t global_seed;
    if (seed.has_value()) {
        global_seed = *seed;
    } else {
        Xoshiro256PlusPlus entropy;
        entropy.seed_from_entropy();
        global_seed = entropy();
    }

    for (uint32_t shot = 0; shot < shots; ++shot) {
        HistorySample hs =
            sample_history(circuit, model, derive_seed(global_seed, shot, kHistoryDomain));
        Circuit rw = rewrite(circuit, hs.history, model);
        Xoshiro256PlusPlus classifier_rng(derive_seed(global_seed, shot, kClassifierDomain));
        std::vector<uint8_t> shot_heralds(circuit.num_measurements, 0);
        Circuit injected =
            inject_classifier(circuit, rw, hs.history, model, classifier_rng, shot_heralds);

        CompiledModule program = compile_circuit(injected);
        SampleResult sr = sample(program, 1, derive_seed(global_seed, shot, kSvmDomain));

        result.measurements.insert(result.measurements.end(), sr.measurements.begin(),
                                   sr.measurements.end());
        result.detectors.insert(result.detectors.end(), sr.detectors.begin(), sr.detectors.end());
        result.observables.insert(result.observables.end(), sr.observables.begin(),
                                  sr.observables.end());
        result.final_status.insert(result.final_status.end(), hs.final_status.begin(),
                                   hs.final_status.end());
        result.heralds.insert(result.heralds.end(), shot_heralds.begin(), shot_heralds.end());
    }

    return result;
}

}  // namespace clifft
