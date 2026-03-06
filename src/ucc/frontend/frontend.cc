#include "ucc/frontend/frontend.h"

#include "ucc/util/config.h"

#include "stim.h"

#include <random>
#include <stdexcept>
#include <string>

namespace ucc {

namespace {

// Helper to apply a single-qubit Clifford gate to the simulator.
// We directly prepend to inv_state for O(n) performance instead of going through
// safe_do_circuit() which has significant overhead (Circuit allocation, string lookup).
// Since we only need the inverse tableau for Heisenberg rewinding, this is safe.
void apply_single_qubit_clifford(stim::TableauSimulator<kStimWidth>& sim, GateType gate,
                                 uint32_t qubit) {
    // Fast path for high-frequency gates using Stim's native inline methods.
    // We prepend to inv_state, so we need the INVERSE of each gate.
    // Self-inverse gates (H, X, Y, Z) are unchanged; S and S_DAG swap.
    size_t q = static_cast<size_t>(qubit);
    switch (gate) {
        case GateType::H:
            sim.inv_state.prepend_H_XZ(q);
            return;
        case GateType::S:
            sim.inv_state.prepend_SQRT_Z_DAG(q);
            return;
        case GateType::S_DAG:
            sim.inv_state.prepend_SQRT_Z(q);
            return;
        case GateType::X:
            sim.inv_state.prepend_X(q);
            return;
        case GateType::Y:
            sim.inv_state.prepend_Y(q);
            return;
        case GateType::Z:
            sim.inv_state.prepend_Z(q);
            return;
        default:
            break;
    }
    // Generic path for the long tail of Cliffords.
    const auto& inv_gate = stim::GATE_DATA.at(gate_name(gate)).inverse();
    auto inv_tab = inv_gate.tableau<kStimWidth>();
    sim.inv_state.inplace_scatter_prepend(inv_tab, {q});
}

// Helper to apply a two-qubit Clifford gate to the simulator.
// Same optimization as single-qubit: direct prepend to inv_state.
void apply_two_qubit_clifford(stim::TableauSimulator<kStimWidth>& sim, GateType gate, uint32_t q1,
                              uint32_t q2) {
    // Fast path for high-frequency gates using Stim's native inline methods.
    size_t a = static_cast<size_t>(q1);
    size_t b = static_cast<size_t>(q2);
    switch (gate) {
        case GateType::CX:
            sim.inv_state.prepend_ZCX(a, b);
            return;
        case GateType::CY:
            sim.inv_state.prepend_ZCY(a, b);
            return;
        case GateType::CZ:
            sim.inv_state.prepend_ZCZ(a, b);
            return;
        case GateType::SWAP:
            sim.inv_state.prepend_SWAP(a, b);
            return;
        default:
            break;
    }
    // Generic path for the long tail of two-qubit Cliffords.
    const auto& inv_gate = stim::GATE_DATA.at(gate_name(gate)).inverse();
    auto inv_tab = inv_gate.tableau<kStimWidth>();
    sim.inv_state.inplace_scatter_prepend(inv_tab, {a, b});
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

// Rewind a single-qubit Pauli (X, Y, or Z) through the tableau.
// pauli_type: 1=X, 2=Y, 3=Z
NoiseChannel rewind_single_pauli(const stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit,
                                 int pauli_type, double prob) {
    stim::PauliString<kStimWidth> pauli(sim.inv_state.num_qubits);
    switch (pauli_type) {
        case 1:  // X
            pauli.xs[qubit] = true;
            break;
        case 2:  // Y = iXZ
            pauli.xs[qubit] = true;
            pauli.zs[qubit] = true;
            break;
        case 3:  // Z
            pauli.zs[qubit] = true;
            break;
    }
    stim::PauliString<kStimWidth> rewound = sim.inv_state(pauli);
    return NoiseChannel{rewound.xs.u64[0], rewound.zs.u64[0], prob};
}

// Rewind a two-qubit Pauli through the tableau.
// pauli1, pauli2: 0=I, 1=X, 2=Y, 3=Z
NoiseChannel rewind_two_qubit_pauli(const stim::TableauSimulator<kStimWidth>& sim, uint32_t q1,
                                    uint32_t q2, int pauli1, int pauli2, double prob) {
    stim::PauliString<kStimWidth> pauli(sim.inv_state.num_qubits);

    // Set Pauli on q1
    if (pauli1 == 1 || pauli1 == 2)
        pauli.xs[q1] = true;  // X or Y
    if (pauli1 == 2 || pauli1 == 3)
        pauli.zs[q1] = true;  // Y or Z

    // Set Pauli on q2
    if (pauli2 == 1 || pauli2 == 2)
        pauli.xs[q2] = true;  // X or Y
    if (pauli2 == 2 || pauli2 == 3)
        pauli.zs[q2] = true;  // Y or Z

    stim::PauliString<kStimWidth> rewound = sim.inv_state(pauli);
    return NoiseChannel{rewound.xs.u64[0], rewound.zs.u64[0], prob};
}

// Create a NoiseSite for a single-qubit noise channel.
NoiseSite make_single_qubit_noise_site(const stim::TableauSimulator<kStimWidth>& sim, GateType gate,
                                       uint32_t qubit, double prob) {
    NoiseSite site;
    switch (gate) {
        case GateType::X_ERROR:
            site.channels.push_back(rewind_single_pauli(sim, qubit, 1, prob));
            break;
        case GateType::Y_ERROR:
            site.channels.push_back(rewind_single_pauli(sim, qubit, 2, prob));
            break;
        case GateType::Z_ERROR:
            site.channels.push_back(rewind_single_pauli(sim, qubit, 3, prob));
            break;
        case GateType::DEPOLARIZE1:
            // DEP1(p) = X, Y, Z each with probability p/3
            site.channels.push_back(rewind_single_pauli(sim, qubit, 1, prob / 3.0));
            site.channels.push_back(rewind_single_pauli(sim, qubit, 2, prob / 3.0));
            site.channels.push_back(rewind_single_pauli(sim, qubit, 3, prob / 3.0));
            break;
        default:
            throw std::runtime_error("Not a single-qubit noise gate");
    }
    return site;
}

// Create a NoiseSite for DEPOLARIZE2 on a qubit pair.
// 15 channels: all non-II two-qubit Paulis, each with prob p/15.
NoiseSite make_depolarize2_noise_site(const stim::TableauSimulator<kStimWidth>& sim, uint32_t q1,
                                      uint32_t q2, double prob) {
    NoiseSite site;
    double channel_prob = prob / 15.0;

    // Enumerate all (p1, p2) where p1,p2 in {0,1,2,3} (I,X,Y,Z) excluding (0,0)
    for (int p1 = 0; p1 <= 3; ++p1) {
        for (int p2 = 0; p2 <= 3; ++p2) {
            if (p1 == 0 && p2 == 0)
                continue;  // Skip II
            site.channels.push_back(rewind_two_qubit_pauli(sim, q1, q2, p1, p2, channel_prob));
        }
    }
    return site;
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
    hir.num_detectors = circuit.num_detectors;
    hir.num_observables = circuit.num_observables;

    // Initialize tableau simulator with a fixed seed (deterministic for testing)
    // The RNG is only used for measurement collapse, which we handle separately
    std::mt19937_64 rng(0);
    stim::TableauSimulator<kStimWidth> sim(std::move(rng), circuit.num_qubits);

    // Track measurement index for rec[-k] resolution
    MeasRecordIdx meas_idx{0};

    for (const auto& node : circuit.nodes) {
        // Emit helper: appends an HIR op and its source provenance in lockstep.
        auto emit = [&](HeisenbergOp op) {
            hir.ops.push_back(op);
            hir.source_map.push_back({node.source_line});
        };

        switch (node.gate) {
            // Single-qubit Clifford gates - absorb into tableau
            case GateType::H:
            case GateType::S:
            case GateType::S_DAG:
            case GateType::X:
            case GateType::Y:
            case GateType::Z:
            case GateType::SQRT_X:
            case GateType::SQRT_X_DAG:
            case GateType::SQRT_Y:
            case GateType::SQRT_Y_DAG:
            case GateType::H_XY:
            case GateType::H_YZ:
            case GateType::H_NXY:
            case GateType::H_NXZ:
            case GateType::H_NYZ:
            case GateType::C_XYZ:
            case GateType::C_ZYX:
            case GateType::C_NXYZ:
            case GateType::C_NZYX:
            case GateType::C_XNYZ:
            case GateType::C_XYNZ:
            case GateType::C_ZNYX:
            case GateType::C_ZYNX: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    apply_single_qubit_clifford(sim, node.gate, qubit);
                }
                break;
            }

            // Two-qubit Clifford gates - absorb into tableau
            case GateType::CX:
            case GateType::CY:
            case GateType::CZ:
            case GateType::SWAP:
            case GateType::ISWAP:
            case GateType::ISWAP_DAG:
            case GateType::SQRT_XX:
            case GateType::SQRT_XX_DAG:
            case GateType::SQRT_YY:
            case GateType::SQRT_YY_DAG:
            case GateType::SQRT_ZZ:
            case GateType::SQRT_ZZ_DAG:
            case GateType::CXSWAP:
            case GateType::CZSWAP:
            case GateType::SWAPCX:
            case GateType::XCX:
            case GateType::XCY:
            case GateType::XCZ:
            case GateType::YCX:
            case GateType::YCY:
            case GateType::YCZ: {
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

                        emit(HeisenbergOp::make_conditional(destab_mask, stab_mask, sign,
                                                            controlling_meas));
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
                    emit(HeisenbergOp::make_tgate(destab_mask, stab_mask, sign, /*dagger=*/false));
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
                    emit(HeisenbergOp::make_tgate(destab_mask, stab_mask, sign, /*dagger=*/true));
                }
                break;
            }

            // Z-basis measurement
            case GateType::M: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;
                    extract_rewound_z(sim, qubit, destab_mask, stab_mask, sign);
                    sign ^= target.is_inverted();
                    emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx));
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
                    extract_rewound_x(sim, qubit, destab_mask, stab_mask, sign);
                    sign ^= target.is_inverted();
                    emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx));
                    ++meas_idx;
                }
                break;
            }

            // Y-basis measurement
            case GateType::MY: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    auto pauli = sim.inv_state.y_output(qubit);
                    uint64_t destab_mask = pauli.xs.u64[0];
                    uint64_t stab_mask = pauli.zs.u64[0];
                    bool sign = pauli.sign ^ target.is_inverted();
                    emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx));
                    ++meas_idx;
                }
                break;
            }

            // Multi-Pauli measurement (MPP)
            case GateType::MPP: {
                stim::PauliString<kStimWidth> obs(circuit.num_qubits);
                bool inversion_parity = false;
                for (const auto& target : node.targets) {
                    uint32_t q = target.value();
                    inversion_parity ^= target.is_inverted();
                    if (target.pauli() == Target::kPauliX) {
                        obs.xs[q] = true;
                    } else if (target.pauli() == Target::kPauliY) {
                        obs.xs[q] = true;
                        obs.zs[q] = true;
                    } else {
                        obs.zs[q] = true;
                    }
                }
                stim::PauliString<kStimWidth> rewound = sim.inv_state(obs);
                uint64_t destab_mask = rewound.xs.u64[0];
                uint64_t stab_mask = rewound.zs.u64[0];
                bool sign = rewound.sign ^ inversion_parity;
                emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx));
                ++meas_idx;
                break;
            }

            // Z-basis reset (hidden measurement + conditional correction)
            case GateType::R: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;

                    // Extract rewound Z observable (measurement) from un-collapsed tableau
                    extract_rewound_z(sim, qubit, destab_mask, stab_mask, sign);
                    auto meas_op =
                        HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx);
                    meas_op.set_hidden(true);
                    emit(meas_op);

                    // Extract rewound X (conditional correction) from same un-collapsed tableau
                    uint64_t corr_destab, corr_stab;
                    bool corr_sign;
                    extract_rewound_x(sim, qubit, corr_destab, corr_stab, corr_sign);
                    auto cond_op = HeisenbergOp::make_conditional(corr_destab, corr_stab, corr_sign,
                                                                  ControllingMeasIdx{0});
                    cond_op.set_use_last_outcome(true);
                    emit(cond_op);
                }
                break;
            }

            // X-basis reset (hidden measurement + conditional correction)
            case GateType::RX: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;

                    // Extract rewound X observable (measurement) from un-collapsed tableau
                    extract_rewound_x(sim, qubit, destab_mask, stab_mask, sign);
                    auto meas_op =
                        HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx);
                    meas_op.set_hidden(true);
                    emit(meas_op);

                    // Extract rewound Z (conditional correction) from same un-collapsed tableau
                    uint64_t corr_destab, corr_stab;
                    bool corr_sign;
                    extract_rewound_z(sim, qubit, corr_destab, corr_stab, corr_sign);
                    auto cond_op = HeisenbergOp::make_conditional(corr_destab, corr_stab, corr_sign,
                                                                  ControllingMeasIdx{0});
                    cond_op.set_use_last_outcome(true);
                    emit(cond_op);
                }
                break;
            }

            // Measure-reset Z-basis (visible measurement + conditional correction)
            case GateType::MR: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;

                    // Emit measurement with TRUE physical sign (no inversion).
                    // The feedback correction reads this value via use_last_outcome.
                    extract_rewound_z(sim, qubit, destab_mask, stab_mask, sign);
                    emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx));
                    uint32_t this_meas = static_cast<uint32_t>(meas_idx);
                    ++meas_idx;

                    // Extract rewound X (conditional correction) from same un-collapsed tableau
                    uint64_t corr_destab, corr_stab;
                    bool corr_sign;
                    extract_rewound_x(sim, qubit, corr_destab, corr_stab, corr_sign);
                    auto cond_op = HeisenbergOp::make_conditional(corr_destab, corr_stab, corr_sign,
                                                                  ControllingMeasIdx{0});
                    cond_op.set_use_last_outcome(true);
                    emit(cond_op);

                    // Apply inversion AFTER feedback so the classical record is flipped
                    // without corrupting the physical reset.
                    if (target.is_inverted()) {
                        ReadoutNoiseIdx idx{static_cast<uint32_t>(hir.readout_noise.size())};
                        hir.readout_noise.push_back({this_meas, 1.0});
                        emit(HeisenbergOp::make_readout_noise(idx));
                    }
                }
                break;
            }

            // Y-basis reset (hidden measurement + conditional correction)
            case GateType::RY: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();

                    // Extract rewound Y observable (measurement)
                    auto pauli = sim.inv_state.y_output(qubit);
                    uint64_t destab_mask = pauli.xs.u64[0];
                    uint64_t stab_mask = pauli.zs.u64[0];
                    bool sign = pauli.sign;
                    auto meas_op =
                        HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx);
                    meas_op.set_hidden(true);
                    emit(meas_op);

                    // X anti-commutes with Y, so X correction flips Y eigenvalue
                    uint64_t corr_destab, corr_stab;
                    bool corr_sign;
                    extract_rewound_x(sim, qubit, corr_destab, corr_stab, corr_sign);
                    auto cond_op = HeisenbergOp::make_conditional(corr_destab, corr_stab, corr_sign,
                                                                  ControllingMeasIdx{0});
                    cond_op.set_use_last_outcome(true);
                    emit(cond_op);
                }
                break;
            }

            // Measure-reset Y-basis (visible measurement + conditional correction)
            case GateType::MRY: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();

                    // Emit measurement with TRUE physical sign (no inversion).
                    auto pauli = sim.inv_state.y_output(qubit);
                    uint64_t destab_mask = pauli.xs.u64[0];
                    uint64_t stab_mask = pauli.zs.u64[0];
                    bool sign = pauli.sign;
                    emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx));
                    uint32_t this_meas = static_cast<uint32_t>(meas_idx);
                    ++meas_idx;

                    // X anti-commutes with Y, so X correction flips Y eigenvalue
                    uint64_t corr_destab, corr_stab;
                    bool corr_sign;
                    extract_rewound_x(sim, qubit, corr_destab, corr_stab, corr_sign);
                    auto cond_op = HeisenbergOp::make_conditional(corr_destab, corr_stab, corr_sign,
                                                                  ControllingMeasIdx{0});
                    cond_op.set_use_last_outcome(true);
                    emit(cond_op);

                    if (target.is_inverted()) {
                        ReadoutNoiseIdx idx{static_cast<uint32_t>(hir.readout_noise.size())};
                        hir.readout_noise.push_back({this_meas, 1.0});
                        emit(HeisenbergOp::make_readout_noise(idx));
                    }
                }
                break;
            }

            // Measure-reset X-basis (visible measurement + conditional correction)
            case GateType::MRX: {
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    uint64_t destab_mask, stab_mask;
                    bool sign;

                    // Emit measurement with TRUE physical sign (no inversion).
                    extract_rewound_x(sim, qubit, destab_mask, stab_mask, sign);
                    emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, meas_idx));
                    uint32_t this_meas = static_cast<uint32_t>(meas_idx);
                    ++meas_idx;

                    // Extract rewound Z (conditional correction) from same un-collapsed tableau
                    uint64_t corr_destab, corr_stab;
                    bool corr_sign;
                    extract_rewound_z(sim, qubit, corr_destab, corr_stab, corr_sign);
                    auto cond_op = HeisenbergOp::make_conditional(corr_destab, corr_stab, corr_sign,
                                                                  ControllingMeasIdx{0});
                    cond_op.set_use_last_outcome(true);
                    emit(cond_op);

                    if (target.is_inverted()) {
                        ReadoutNoiseIdx idx{static_cast<uint32_t>(hir.readout_noise.size())};
                        hir.readout_noise.push_back({this_meas, 1.0});
                        emit(HeisenbergOp::make_readout_noise(idx));
                    }
                }
                break;
            }

            // MPAD: deterministic classical padding (zero-weight measurement)
            case GateType::MPAD: {
                for (const auto& target : node.targets) {
                    bool sign = (target.value() != 0) ^ target.is_inverted();
                    emit(HeisenbergOp::make_measure(0, 0, sign, meas_idx));
                    ++meas_idx;
                }
                break;
            }

            // TICK is a no-op annotation
            case GateType::TICK:
                break;

            // Single-qubit noise gates
            case GateType::X_ERROR:
            case GateType::Y_ERROR:
            case GateType::Z_ERROR:
            case GateType::DEPOLARIZE1: {
                double prob = node.args.empty() ? 0.0 : node.args[0];
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    NoiseSite site = make_single_qubit_noise_site(sim, node.gate, qubit, prob);
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    emit(HeisenbergOp::make_noise(idx));
                }
                break;
            }

            // Single-qubit Pauli channel with 3 explicit probabilities
            case GateType::PAULI_CHANNEL_1: {
                if (node.args.size() < 3) {
                    throw std::runtime_error(
                        "PAULI_CHANNEL_1 requires 3 arguments: P(X), P(Y), P(Z)");
                }
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    NoiseSite site;
                    // pauli_type: 1=X, 2=Y, 3=Z
                    for (int p = 0; p < 3; ++p) {
                        double prob = node.args[static_cast<size_t>(p)];
                        if (prob > 0.0) {
                            site.channels.push_back(rewind_single_pauli(sim, qubit, p + 1, prob));
                        }
                    }
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    emit(HeisenbergOp::make_noise(idx));
                }
                break;
            }

            // Two-qubit Pauli channel with 15 explicit probabilities
            case GateType::PAULI_CHANNEL_2: {
                if (node.args.size() < 15) {
                    throw std::runtime_error("PAULI_CHANNEL_2 requires 15 arguments");
                }
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    NoiseSite site;
                    // Enumerate all non-II two-qubit Paulis in Stim order:
                    // IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
                    size_t arg_idx = 0;
                    for (int p1 = 0; p1 <= 3; ++p1) {
                        for (int p2 = 0; p2 <= 3; ++p2) {
                            if (p1 == 0 && p2 == 0)
                                continue;
                            double prob = node.args[arg_idx];
                            if (prob > 0.0) {
                                site.channels.push_back(
                                    rewind_two_qubit_pauli(sim, q1, q2, p1, p2, prob));
                            }
                            ++arg_idx;
                        }
                    }
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    emit(HeisenbergOp::make_noise(idx));
                }
                break;
            }

            // Two-qubit depolarizing noise
            case GateType::DEPOLARIZE2: {
                double prob = node.args.empty() ? 0.0 : node.args[0];
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    NoiseSite site = make_depolarize2_noise_site(sim, q1, q2, prob);
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    emit(HeisenbergOp::make_noise(idx));
                }
                break;
            }

            // Readout noise (classical bit-flip on measurement result)
            case GateType::READOUT_NOISE: {
                // Parser stores absolute measurement index in target, probability in arg
                for (const auto& target : node.targets) {
                    uint32_t abs_meas_idx = target.value();
                    double prob = node.args.empty() ? 0.0 : node.args[0];
                    ReadoutNoiseIdx idx{static_cast<uint32_t>(hir.readout_noise.size())};
                    hir.readout_noise.push_back({abs_meas_idx, prob});
                    emit(HeisenbergOp::make_readout_noise(idx));
                }
                break;
            }

            // Detector: parity check over measurement records
            case GateType::DETECTOR: {
                std::vector<uint32_t> targets;
                for (const auto& target : node.targets) {
                    targets.push_back(target.value());  // Already absolute indices
                }
                DetectorIdx idx{static_cast<uint32_t>(hir.detector_targets.size())};
                hir.detector_targets.push_back(std::move(targets));
                emit(HeisenbergOp::make_detector(idx));
                break;
            }

            // Observable: logical observable accumulator
            case GateType::OBSERVABLE_INCLUDE: {
                std::vector<uint32_t> targets;
                for (const auto& target : node.targets) {
                    targets.push_back(target.value());  // Already absolute indices
                }
                uint32_t obs_idx = static_cast<uint32_t>(node.args.empty() ? 0.0 : node.args[0]);
                uint32_t target_list_idx = static_cast<uint32_t>(hir.observable_targets.size());
                hir.observable_targets.push_back(std::move(targets));
                emit(HeisenbergOp::make_observable(ObservableIdx{obs_idx}, target_list_idx));
                break;
            }

            default:
                throw std::runtime_error("Unsupported gate type in Front-End: " +
                                         std::string(gate_name(node.gate)));
        }
    }

    // Store forward tableau for statevector expansion
    hir.final_tableau = sim.inv_state.inverse();

    return hir;
}

}  // namespace ucc
