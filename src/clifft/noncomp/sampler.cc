#include "clifft/noncomp/sampler.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/util/xoshiro.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace clifft {

namespace {

// Portable [0, 1) draw: the top 53 bits of a 64-bit word scaled to the
// unit interval. Xoshiro256PlusPlus (the same generator the SVM uses)
// produces an identical sequence across compilers, and this extraction
// avoids std::uniform_real_distribution, whose algorithm varies across
// standard libraries -- together they keep a fixed seed reproducible.
double next_unit(Xoshiro256PlusPlus& rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

}  // namespace

HistorySample sample_history(const Circuit& circuit, const NonComputationalModel& model,
                             uint64_t seed) {
    const LevelSet& levels = model.levels();
    const NonComputationalPolicy& policy = model.policy();
    const size_t num_levels = levels.size();

    Xoshiro256PlusPlus rng(seed);

    HistorySample result;
    result.history.initial_status.reserve(circuit.num_qubits);

    // Sample each qubit's initial level independently from the shared
    // initial-state distribution. The last level catches any
    // floating-point tail so a draw always resolves to a level.
    for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
        const double u = next_unit(rng);
        double acc = 0.0;
        uint8_t chosen = static_cast<uint8_t>(num_levels - 1);
        for (uint8_t l = 0; l < num_levels; ++l) {
            acc += model.initial_probability(l);
            if (u < acc) {
                chosen = l;
                break;
            }
        }
        result.history.initial_status.push_back(levels.status_for(chosen));
    }

    std::vector<QubitStatus> status = result.history.initial_status;

    // Walk the operations, sampling attached transitions per qubit
    // operand and advancing each qubit's status.
    for (uint32_t op_index = 0; op_index < circuit.nodes.size(); ++op_index) {
        const AstNode& node = circuit.nodes[op_index];
        const GateType gate = node.gate;
        const TransitionInstrument* instrument = model.transition_for(gate);

        for (const QubitOperand& operand : qubit_operands(node)) {
            const uint32_t qubit = operand.qubit;
            if (qubit >= status.size()) {
                throw std::invalid_argument(
                    "sample_history: operand qubit " + std::to_string(qubit) +
                    " is out of range (circuit declares " + std::to_string(status.size()) +
                    " qubits) at op " + std::to_string(op_index));
            }

            const QubitStatus s_in = status[qubit];

            // Feedback corrections are virtual: they never fire a
            // transition. Physical operands consult the instrument (if
            // any) for this gate.
            if (operand.role != OperandRole::Physical || instrument == nullptr) {
                status[qubit] = normal_post_op_status(s_in, gate, operand.role, policy, levels);
                continue;
            }

            // Source-context check and column selection: an unknown
            // computational source has no definite level, so a
            // source-dependent instrument cannot pick a column for it.
            uint8_t source_col;
            if (s_in.kind() == QubitStatusKind::ComputationalUnknown) {
                if (!instrument->is_source_independent_on_computational()) {
                    throw std::invalid_argument(
                        "sample_history: source-dependent transition on gate '" +
                        std::string(gate_name(gate)) + "' fired on ComputationalUnknown qubit " +
                        std::to_string(qubit) + " at op " + std::to_string(op_index) +
                        "; a source-dependent instrument cannot be applied to a qubit whose "
                        "computational state is unknown");
                }
                // Computational columns are identical here, so any one
                // serves; use g.
                source_col = levels.computational_zero_id();
            } else {
                source_col = s_in.level_id();
            }

            // Sample the outcome: the no-jump weight occupies [0, w), the
            // jump targets partition [w, 1). last_positive catches a
            // floating-point tail so u >= w always resolves to a jump.
            const double u = next_unit(rng);
            const double no_jump = instrument->no_jump_weight(source_col);
            TransitionOutcome outcome;
            if (u >= no_jump) {
                double acc = no_jump;
                int last_positive = -1;
                for (uint8_t to = 0; to < num_levels; ++to) {
                    const double p = instrument->prob(to, source_col);
                    if (p > 0.0) {
                        last_positive = to;
                    }
                    acc += p;
                    if (u < acc) {
                        outcome.jumped = true;
                        outcome.destination_level = to;
                        break;
                    }
                }
                if (!outcome.jumped && last_positive >= 0) {
                    outcome.jumped = true;
                    outcome.destination_level = static_cast<uint8_t>(last_positive);
                }
            }

            result.history.transitions.push_back(
                TransitionRecord{op_index, qubit, outcome.jumped,
                                 outcome.jumped ? outcome.destination_level : kInvalidLevel});

            status[qubit] = step_status(s_in, gate, OperandRole::Physical, outcome, policy, levels);
        }
    }

    result.final_status = status;
    return result;
}

}  // namespace clifft
