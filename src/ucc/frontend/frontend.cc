#include "ucc/frontend/frontend.h"

#include "ucc/util/config.h"

#include "stim.h"

#include <random>
#include <stdexcept>
#include <string>

namespace ucc {

namespace {

// Measurement outcomes: explicit enum avoids bool ambiguity (is true=0 or true=1?)
enum class MeasOutcome : uint8_t { Zero = 0, One = 1 };

// Measurement result type for determinism detection
struct MeasurementInfo {
    bool is_deterministic;  // true if outcome is fixed by stabilizer state
    MeasOutcome outcome;    // the outcome value, only meaningful if is_deterministic
};

// Check if a Z-basis measurement on a qubit is deterministic.
// Returns the deterministic value (0 or 1) if deterministic, or performs collapse if not.
//
// For the inverse tableau stored in sim.inv_state:
// - inv_state.zs[q] gives the Pauli that conjugates TO Z_q
// - If this Pauli has no X component (only Z and I terms), the measurement is deterministic
// - The outcome is determined by inv_state.zs[q].sign
//
// Stim's is_deterministic_z() uses: !inv_state.zs[target].xs.not_zero()
MeasurementInfo check_deterministic_z(const stim::TableauSimulator<kStimWidth>& sim,
                                      uint32_t qubit) {
    // A Z-basis measurement is deterministic if Z_q commutes with all stabilizers.
    // In the inverse tableau representation, this means the X-component is zero.
    bool is_det = !sim.inv_state.zs[qubit].xs.not_zero();
    MeasOutcome outcome = sim.inv_state.zs[qubit].sign ? MeasOutcome::One : MeasOutcome::Zero;
    return {is_det, outcome};
}

// Check if an X-basis measurement on a qubit is deterministic.
MeasurementInfo check_deterministic_x(const stim::TableauSimulator<kStimWidth>& sim,
                                      uint32_t qubit) {
    // Use Stim's is_deterministic_x: !inv_state.xs[target].xs.not_zero()
    // X-basis measurement is deterministic if the X row of inv_state has no X components.
    bool is_det = sim.is_deterministic_x(qubit);
    MeasOutcome outcome = sim.inv_state.xs[qubit].sign ? MeasOutcome::One : MeasOutcome::Zero;
    return {is_det, outcome};
}

// Check if a Y-basis measurement on a qubit is deterministic.
MeasurementInfo check_deterministic_y(const stim::TableauSimulator<kStimWidth>& sim,
                                      uint32_t qubit) {
    // Y = iXZ requires special handling since it anti-commutes with both X-only and Z-only
    // Paulis. Stim's is_deterministic_y() correctly checks if the y_output has no components
    // that would anti-commute with existing stabilizers.
    auto y_obs = sim.inv_state.y_output(qubit);
    bool is_det = sim.is_deterministic_y(qubit);
    MeasOutcome outcome = y_obs.sign ? MeasOutcome::One : MeasOutcome::Zero;
    return {is_det, outcome};
}

// Check if an MPP observable is deterministic.
// This requires evaluating whether the Pauli product commutes with all stabilizers.
MeasurementInfo check_deterministic_mpp(const stim::TableauSimulator<kStimWidth>& sim,
                                        const stim::PauliString<kStimWidth>& observable) {
    // peek_observable_expectation is O(n) in the number of qubits: it performs a
    // single pass through the stabilizer generators checking if the observable
    // commutes with all of them. Returns 0 if anti-commuting (random outcome),
    // +1 if deterministic outcome 0, or -1 if deterministic outcome 1.
    int8_t expectation = sim.peek_observable_expectation(observable);
    if (expectation == 0) {
        return {false, MeasOutcome::Zero};  // outcome field unused when not deterministic
    }
    return {true, expectation < 0 ? MeasOutcome::One : MeasOutcome::Zero};
}

// Perform measurement collapse for Z-basis.
// Updates the simulator state and returns the AG pivot matrix if anti-commuting.
// If deterministic, returns std::nullopt.
std::optional<stim::Tableau<kStimWidth>> collapse_z_measurement(
    stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit, bool& outcome) {
    auto info = check_deterministic_z(sim, qubit);
    if (info.is_deterministic) {
        outcome = (info.outcome == MeasOutcome::One);
        return std::nullopt;
    }

    // Anti-commuting: save inverse tableau before collapse for AG pivot computation
    stim::Tableau<kStimWidth> inv_before = sim.inv_state;

    // Perform the measurement through Stim (this collapses the state)
    stim::GateTarget targets[] = {stim::GateTarget{qubit}};
    sim.do_MZ({stim::GateType::M, {}, targets, ""});
    outcome = sim.measurement_record.storage.back();

    // The AG pivot captures the change-of-basis induced by measurement collapse.
    // If the SVM later samples a different outcome than our reference, it must
    // apply this unitary to transform between the two post-measurement states.
    // Mathematically: pivot = U_after * U_before^{-1} where U maps computational
    // basis to stabilizer eigenbasis.
    stim::Tableau<kStimWidth> forward_after = sim.inv_state.inverse();
    stim::Tableau<kStimWidth> ag_pivot = forward_after.then(inv_before);

    return ag_pivot;
}

// Perform measurement collapse for X-basis.
std::optional<stim::Tableau<kStimWidth>> collapse_x_measurement(
    stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit, bool& outcome) {
    auto info = check_deterministic_x(sim, qubit);
    if (info.is_deterministic) {
        outcome = (info.outcome == MeasOutcome::One);
        return std::nullopt;
    }

    stim::Tableau<kStimWidth> inv_before = sim.inv_state;

    stim::GateTarget targets[] = {stim::GateTarget{qubit}};
    sim.do_MX({stim::GateType::MX, {}, targets, ""});
    outcome = sim.measurement_record.storage.back();

    // See collapse_z_measurement for explanation of AG pivot computation
    stim::Tableau<kStimWidth> forward_after = sim.inv_state.inverse();
    stim::Tableau<kStimWidth> ag_pivot = forward_after.then(inv_before);

    return ag_pivot;
}

// Perform measurement collapse for Y-basis.
std::optional<stim::Tableau<kStimWidth>> collapse_y_measurement(
    stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit, bool& outcome) {
    auto info = check_deterministic_y(sim, qubit);
    if (info.is_deterministic) {
        outcome = (info.outcome == MeasOutcome::One);
        return std::nullopt;
    }

    stim::Tableau<kStimWidth> inv_before = sim.inv_state;

    stim::GateTarget targets[] = {stim::GateTarget{qubit}};
    sim.do_MY({stim::GateType::MY, {}, targets, ""});
    outcome = sim.measurement_record.storage.back();

    // See collapse_z_measurement for explanation of AG pivot computation
    stim::Tableau<kStimWidth> forward_after = sim.inv_state.inverse();
    stim::Tableau<kStimWidth> ag_pivot = forward_after.then(inv_before);

    return ag_pivot;
}

// Helper to apply a single-qubit Clifford gate to the simulator.
// We directly prepend to inv_state for O(n) performance instead of going through
// safe_do_circuit() which has significant overhead (Circuit allocation, string lookup).
// Since we only need the inverse tableau for Heisenberg rewinding, this is safe.
void apply_single_qubit_clifford(stim::TableauSimulator<kStimWidth>& sim, GateType gate,
                                 uint32_t qubit) {
    switch (gate) {
        case GateType::H:
            // H is self-inverse
            sim.inv_state.prepend_H_XZ(qubit);
            break;
        case GateType::S:
            // S^{-1} = S_DAG, so prepend S_DAG to track inverse
            sim.inv_state.prepend_SQRT_Z_DAG(qubit);
            break;
        case GateType::S_DAG:
            // S_DAG^{-1} = S, so prepend S to track inverse
            sim.inv_state.prepend_SQRT_Z(qubit);
            break;
        case GateType::X:
            // X is self-inverse
            sim.inv_state.prepend_X(qubit);
            break;
        case GateType::Y:
            // Y is self-inverse
            sim.inv_state.prepend_Y(qubit);
            break;
        case GateType::Z:
            // Z is self-inverse
            sim.inv_state.prepend_Z(qubit);
            break;
        default:
            throw std::runtime_error("Not a single-qubit Clifford gate");
    }
}

// Helper to apply a two-qubit Clifford gate to the simulator.
// Same optimization as single-qubit: direct prepend to inv_state.
void apply_two_qubit_clifford(stim::TableauSimulator<kStimWidth>& sim, GateType gate, uint32_t q1,
                              uint32_t q2) {
    switch (gate) {
        case GateType::CX:
            // CNOT is self-inverse
            sim.inv_state.prepend_ZCX(q1, q2);
            break;
        case GateType::CY:
            // CY is self-inverse
            sim.inv_state.prepend_ZCY(q1, q2);
            break;
        case GateType::CZ:
            // CZ is self-inverse
            sim.inv_state.prepend_ZCZ(q1, q2);
            break;
        default:
            throw std::runtime_error("Not a two-qubit Clifford gate");
    }
}

// Extract the rewound Z observable for a qubit as uint64_t masks
void extract_rewound_z(const stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit,
                       uint64_t& destab_mask, uint64_t& stab_mask, bool& sign) {
    const auto& pauli = sim.inv_state.zs[qubit];
    destab_mask = pauli.xs.u64[0];
    stab_mask = pauli.zs.u64[0];
    sign = pauli.sign;
}

// Extract the rewound X observable for a qubit as uint64_t masks
void extract_rewound_x(const stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit,
                       uint64_t& destab_mask, uint64_t& stab_mask, bool& sign) {
    const auto& pauli = sim.inv_state.xs[qubit];
    destab_mask = pauli.xs.u64[0];
    stab_mask = pauli.zs.u64[0];
    sign = pauli.sign;
}

}  // namespace

HirModule trace(const Circuit& circuit) {
    // Check MVP constraint: 64 qubits max
    if (circuit.num_qubits > kMaxInlineQubits) {
        throw std::runtime_error("Circuit exceeds 64-qubit MVP limit: " +
                                 std::to_string(circuit.num_qubits) + " qubits");
    }

    HirModule hir;
    hir.num_qubits = circuit.num_qubits;
    hir.num_measurements = circuit.num_measurements;

    // Initialize tableau simulator with a fixed seed (deterministic for testing)
    // The RNG is only used for measurement collapse, which we handle separately
    std::mt19937_64 rng(0);
    stim::TableauSimulator<kStimWidth> sim(std::move(rng), circuit.num_qubits);

    // Track measurement index for rec[-k] resolution
    MeasRecordIdx meas_idx{0};

    for (const auto& node : circuit.nodes) {
        switch (node.gate) {
            // Single-qubit Clifford gates - absorb into tableau
            case GateType::H:
            case GateType::S:
            case GateType::S_DAG:
            case GateType::X:
            case GateType::Y:
            case GateType::Z: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    apply_single_qubit_clifford(sim, node.gate, qubit);
                }
                break;
            }

            // Two-qubit Clifford gates - absorb into tableau
            case GateType::CX:
            case GateType::CY:
            case GateType::CZ: {
                // Check if this is classical feedback (first target is rec)
                if (!node.targets.empty() && node.targets[0].is_rec()) {
                    // Classical feedback: CX rec[-k] q or CZ rec[-k] q
                    // This was generated by reset decomposition
                    // Supports broadcasting: CX rec[-1] 0 rec[-2] 1 etc.
                    for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                        uint32_t rec_abs_idx = node.targets[i].value();
                        uint32_t target_qubit = node.targets[i + 1].value();

                        // The controlling measurement is the absolute index
                        ControllingMeasIdx controlling_meas{rec_abs_idx};

                        // Extract the rewound Pauli that will be conditionally applied
                        // For CX rec[-k] q: apply X_q if measurement was 1
                        // For CZ rec[-k] q: apply Z_q if measurement was 1
                        uint64_t destab_mask, stab_mask;
                        bool sign;

                        if (node.gate == GateType::CX) {
                            // X on target qubit, rewound through tableau
                            extract_rewound_x(sim, target_qubit, destab_mask, stab_mask, sign);
                        } else if (node.gate == GateType::CZ) {
                            // Z on target qubit, rewound through tableau
                            extract_rewound_z(sim, target_qubit, destab_mask, stab_mask, sign);
                        } else {
                            // CY classical feedback not supported in MVP
                            throw std::runtime_error("CY classical feedback not supported in MVP");
                        }

                        hir.ops.push_back(HeisenbergOp::make_conditional(destab_mask, stab_mask,
                                                                         sign, controlling_meas));
                    }
                } else {
                    // Regular two-qubit Clifford gate
                    for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                        uint32_t q1 = node.targets[i].value();
                        uint32_t q2 = node.targets[i + 1].value();
                        apply_two_qubit_clifford(sim, node.gate, q1, q2);
                    }
                }
                break;
            }

            // T gate - emit HeisenbergOp with rewound Z
            case GateType::T: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;
                    extract_rewound_z(sim, qubit, destab_mask, stab_mask, sign);
                    hir.ops.push_back(
                        HeisenbergOp::make_tgate(destab_mask, stab_mask, sign, /*dagger=*/false));
                }
                break;
            }

            // T_DAG gate - emit HeisenbergOp with rewound Z and is_dagger=true
            case GateType::T_DAG: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;
                    extract_rewound_z(sim, qubit, destab_mask, stab_mask, sign);
                    hir.ops.push_back(
                        HeisenbergOp::make_tgate(destab_mask, stab_mask, sign, /*dagger=*/true));
                }
                break;
            }

            // Z-basis measurement
            case GateType::M: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;

                    // Extract the rewound observable BEFORE potential collapse
                    extract_rewound_z(sim, qubit, destab_mask, stab_mask, sign);

                    // Perform collapse and get AG pivot if anti-commuting
                    bool outcome;
                    auto ag_pivot = collapse_z_measurement(sim, qubit, outcome);

                    // Always record the reference outcome (critical for deterministic
                    // measurements!)
                    uint8_t ag_ref = outcome ? 1 : 0;
                    AgMatrixIdx ag_idx = AgMatrixIdx::None;
                    if (ag_pivot.has_value()) {
                        // Store AG pivot matrix
                        ag_idx = AgMatrixIdx{static_cast<uint32_t>(hir.ag_matrices.size())};
                        hir.ag_matrices.push_back(std::move(*ag_pivot));
                    }

                    hir.ops.push_back(HeisenbergOp::make_measure(destab_mask, stab_mask, sign,
                                                                 meas_idx, ag_idx, ag_ref));
                    ++meas_idx;
                }
                break;
            }

            // X-basis measurement
            case GateType::MX: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;

                    // Extract rewound observable BEFORE collapse
                    extract_rewound_x(sim, qubit, destab_mask, stab_mask, sign);

                    // Perform collapse and get AG pivot if anti-commuting
                    bool outcome;
                    auto ag_pivot = collapse_x_measurement(sim, qubit, outcome);

                    // Always record the reference outcome (critical for deterministic
                    // measurements!)
                    uint8_t ag_ref = outcome ? 1 : 0;
                    AgMatrixIdx ag_idx = AgMatrixIdx::None;
                    if (ag_pivot.has_value()) {
                        ag_idx = AgMatrixIdx{static_cast<uint32_t>(hir.ag_matrices.size())};
                        hir.ag_matrices.push_back(std::move(*ag_pivot));
                    }

                    hir.ops.push_back(HeisenbergOp::make_measure(destab_mask, stab_mask, sign,
                                                                 meas_idx, ag_idx, ag_ref));
                    ++meas_idx;
                }
                break;
            }

            // Y-basis measurement
            case GateType::MY: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();

                    // Use Stim's y_output() which correctly handles Y = iXZ phase
                    auto pauli = sim.inv_state.y_output(qubit);
                    uint64_t destab_mask = pauli.xs.u64[0];
                    uint64_t stab_mask = pauli.zs.u64[0];
                    bool sign = pauli.sign;

                    // Perform collapse and get AG pivot if anti-commuting
                    bool outcome;
                    auto ag_pivot = collapse_y_measurement(sim, qubit, outcome);

                    // Always record the reference outcome (critical for deterministic
                    // measurements!)
                    uint8_t ag_ref = outcome ? 1 : 0;
                    AgMatrixIdx ag_idx = AgMatrixIdx::None;
                    if (ag_pivot.has_value()) {
                        ag_idx = AgMatrixIdx{static_cast<uint32_t>(hir.ag_matrices.size())};
                        hir.ag_matrices.push_back(std::move(*ag_pivot));
                    }

                    hir.ops.push_back(HeisenbergOp::make_measure(destab_mask, stab_mask, sign,
                                                                 meas_idx, ag_idx, ag_ref));
                    ++meas_idx;
                }
                break;
            }

            // Multi-Pauli measurement (MPP)
            case GateType::MPP: {
                // Build the observable as a PauliString, then rewind through
                // the inverse tableau in one shot. This correctly handles all
                // phase accumulation via Stim's scatter_eval().
                stim::PauliString<kStimWidth> obs(circuit.num_qubits);

                for (const auto& target : node.targets) {
                    uint32_t q = target.value();
                    if (target.pauli() == Target::kPauliX) {
                        obs.xs[q] = true;
                    } else if (target.pauli() == Target::kPauliY) {
                        // Y = iXZ, set both bits
                        obs.xs[q] = true;
                        obs.zs[q] = true;
                    } else {
                        // Z target (default)
                        obs.zs[q] = true;
                    }
                }

                // Rewind the entire Pauli product through the inverse tableau
                stim::PauliString<kStimWidth> rewound = sim.inv_state(obs);
                uint64_t destab_mask = rewound.xs.u64[0];
                uint64_t stab_mask = rewound.zs.u64[0];
                bool sign = rewound.sign;

                // Check determinism and handle collapse
                auto info = check_deterministic_mpp(sim, obs);

                AgMatrixIdx ag_idx = AgMatrixIdx::None;
                bool outcome =
                    (info.outcome == MeasOutcome::One);  // Default for deterministic case

                if (!info.is_deterministic) {
                    // Anti-commuting: need to collapse via Stim's MPP
                    stim::Tableau<kStimWidth> inv_before = sim.inv_state;

                    // Build GateTarget array for MPP using Stim's factory methods
                    std::vector<stim::GateTarget> targets;
                    for (const auto& target : node.targets) {
                        uint32_t q = target.value();
                        stim::GateTarget gt;
                        if (target.pauli() == Target::kPauliX) {
                            gt = stim::GateTarget::x(q);
                        } else if (target.pauli() == Target::kPauliY) {
                            gt = stim::GateTarget::y(q);
                        } else {
                            gt = stim::GateTarget::z(q);
                        }
                        if (!targets.empty()) {
                            // Add combiner between terms
                            targets.push_back(stim::GateTarget::combiner());
                        }
                        targets.push_back(gt);
                    }

                    sim.do_MPP({stim::GateType::MPP, {}, targets, ""});
                    outcome = sim.measurement_record.storage.back();

                    stim::Tableau<kStimWidth> forward_after = sim.inv_state.inverse();
                    stim::Tableau<kStimWidth> ag_pivot = forward_after.then(inv_before);

                    ag_idx = AgMatrixIdx{static_cast<uint32_t>(hir.ag_matrices.size())};
                    hir.ag_matrices.push_back(std::move(ag_pivot));
                }

                // Always record the reference outcome (critical for deterministic measurements!)
                uint8_t ag_ref = outcome ? 1 : 0;
                hir.ops.push_back(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx,
                                                             ag_idx, ag_ref));
                ++meas_idx;
                break;
            }

            // TICK is a no-op annotation
            case GateType::TICK:
                break;

            default:
                throw std::runtime_error("Unsupported gate type in Front-End: " +
                                         std::string(gate_name(node.gate)));
        }
    }

    return hir;
}

}  // namespace ucc
