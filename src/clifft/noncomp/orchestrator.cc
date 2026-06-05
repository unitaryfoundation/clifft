#include "clifft/noncomp/orchestrator.h"

#include "clifft/backend/backend.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
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

// Sample a measurement-record bit from the classifier for a qubit at `level`.
// Symbols partition [0, sum-of-column); the deficit is the reject region, and
// a draw there raises. Symbol index maps to the record bit as zero/non-zero.
uint8_t classifier_bit(const MeasurementClassifier& classifier, uint8_t level,
                       Xoshiro256PlusPlus& rng, GateType gate, uint32_t op_index, uint32_t qubit) {
    // Binary record injection maps the sampled symbol index directly to the
    // record bit, which is faithful only for a two-symbol classifier. A richer
    // alphabet has no defined symbol-to-bit mapping and is not representable.
    if (classifier.num_symbols() != 2) {
        throw std::invalid_argument(
            "sample_noncomputational: injecting a measurement on a noncomputational qubit "
            "requires a two-symbol classifier, but the model's has " +
            std::to_string(classifier.num_symbols()) + " symbols (measurement '" +
            std::string(gate_name(gate)) + "' on qubit " + std::to_string(qubit) + " at op " +
            std::to_string(op_index) + ")");
    }
    const double u = rng.next_double();
    double acc = 0.0;
    for (uint8_t s = 0; s < classifier.num_symbols(); ++s) {
        acc += classifier.prob(s, level);
        if (u < acc) {
            return s;  // symbol index 0 or 1 is the record bit
        }
    }
    throw std::invalid_argument("sample_noncomputational: classifier rejected the measurement '" +
                                std::string(gate_name(gate)) + "' on qubit " +
                                std::to_string(qubit) + " at op " + std::to_string(op_index) +
                                " (level " + std::to_string(level) + ")");
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
// record slot and its sampled classifier bit, then swap those slots in
// `rewritten`: M -> MPAD(bit); a measure-and-reset -> MPAD(bit) plus the
// matching reset. The rewriter has already guaranteed only M / measure-reset
// measurements reach a noncomputational operand.
Circuit inject_classifier(const Circuit& original, const Circuit& rewritten,
                          const NonComputationalHistory& history,
                          const NonComputationalModel& model, Xoshiro256PlusPlus& rng) {
    const LevelSet& levels = model.levels();
    const NonComputationalPolicy& policy = model.policy();
    const MeasurementClassifier* classifier = model.classifier();

    std::map<uint32_t, uint8_t> slot_to_bit;
    std::vector<QubitStatus> status = history.initial_status;
    size_t trans_cursor = 0;
    uint32_t slot = 0;
    for (uint32_t op_index = 0; op_index < original.nodes.size(); ++op_index) {
        const AstNode& node = original.nodes[op_index];
        const GateType gate = node.gate;
        const TransitionInstrument* instrument = model.transition_for(gate);
        const bool measurement = is_measurement(gate);
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
                slot_to_bit[slot] =
                    classifier_bit(*classifier, pre.level_id(), rng, gate, op_index, qubit);
            }
            status[qubit] = step_status(pre, gate, operand.role, outcome, policy, levels);
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
        const uint8_t bit = it->second;
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
        Circuit injected = inject_classifier(circuit, rw, hs.history, model, classifier_rng);

        CompiledModule program = compile_circuit(injected);
        SampleResult sr = sample(program, 1, derive_seed(global_seed, shot, kSvmDomain));

        result.measurements.insert(result.measurements.end(), sr.measurements.begin(),
                                   sr.measurements.end());
        result.detectors.insert(result.detectors.end(), sr.detectors.begin(), sr.detectors.end());
        result.observables.insert(result.observables.end(), sr.observables.begin(),
                                  sr.observables.end());
        result.final_status.insert(result.final_status.end(), hs.final_status.begin(),
                                   hs.final_status.end());
    }

    return result;
}

}  // namespace clifft
