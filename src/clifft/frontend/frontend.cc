#include "clifft/frontend/frontend.h"

#include "stim.h"

#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

namespace clifft {

namespace {

// Helper to apply a single-qubit Clifford gate to the simulator.
// We directly prepend to inv_state for O(n) performance instead of going through
// safe_do_circuit() which has significant overhead (Circuit allocation, string lookup).
// Since we only need the inverse tableau for Heisenberg rewinding, this is safe.
void apply_single_qubit_clifford(stim::TableauSimulator<kStimWidth>& sim, GateType gate,
                                 uint32_t qubit) {
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
    const auto& inv_gate = stim::GATE_DATA.at(gate_name(gate)).inverse();
    auto inv_tab = inv_gate.tableau<kStimWidth>();
    sim.inv_state.inplace_scatter_prepend(inv_tab, {q});
}

void apply_two_qubit_clifford(stim::TableauSimulator<kStimWidth>& sim, GateType gate, uint32_t q1,
                              uint32_t q2) {
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
    const auto& inv_gate = stim::GATE_DATA.at(gate_name(gate)).inverse();
    auto inv_tab = inv_gate.tableau<kStimWidth>();
    sim.inv_state.inplace_scatter_prepend(inv_tab, {a, b});
}

/// Write the rewound Z observable for `qubit` into pre-zeroed MutableMaskViews.
void extract_rewound_z_into(const stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit,
                            MutableMaskView destab, MutableMaskView stab, bool& sign) {
    const auto& pauli = sim.inv_state.zs[qubit];
    uint32_t n = sim.inv_state.num_qubits;
    stim_to_mask_view(pauli.xs, n, destab);
    stim_to_mask_view(pauli.zs, n, stab);
    sign = pauli.sign;
}

/// Write the rewound X observable for `qubit` into pre-zeroed MutableMaskViews.
void extract_rewound_x_into(const stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit,
                            MutableMaskView destab, MutableMaskView stab, bool& sign) {
    const auto& pauli = sim.inv_state.xs[qubit];
    uint32_t n = sim.inv_state.num_qubits;
    stim_to_mask_view(pauli.xs, n, destab);
    stim_to_mask_view(pauli.zs, n, stab);
    sign = pauli.sign;
}

/// Write the rewound Y observable for `qubit` into pre-zeroed MutableMaskViews.
void extract_rewound_y_into(const stim::TableauSimulator<kStimWidth>& sim, uint32_t qubit,
                            MutableMaskView destab, MutableMaskView stab, bool& sign) {
    auto pauli = sim.inv_state.y_output(qubit);
    uint32_t n = sim.inv_state.num_qubits;
    stim_to_mask_view(pauli.xs, n, destab);
    stim_to_mask_view(pauli.zs, n, stab);
    sign = pauli.sign;
}

/// Copy a rewound stim::PauliString into pre-zeroed MutableMaskViews.
void copy_rewound_into(const stim::PauliString<kStimWidth>& rewound, uint32_t n,
                       MutableMaskView destab, MutableMaskView stab) {
    stim_to_mask_view(rewound.xs, n, destab);
    stim_to_mask_view(rewound.zs, n, stab);
}

/// XOR a single-qubit Pauli generator's tableau row into the destination
/// views. pauli_type: 1=X, 2=Y, 3=Z. Sign is irrelevant for noise channels.
void accumulate_pauli_row(const stim::Tableau<kStimWidth>& tab, uint32_t qubit, int pauli_type,
                          uint32_t n, MutableMaskView destab, MutableMaskView stab) {
    const uint32_t words = (n + 63) / 64;
    auto xor_row = [&](const stim::PauliString<kStimWidth>& row) {
        for (uint32_t w = 0; w < words; ++w) {
            destab.words[w] ^= row.xs.u64[w];
            stab.words[w] ^= row.zs.u64[w];
        }
    };
    if (pauli_type == 1 || pauli_type == 2) {
        xor_row(tab.xs[qubit]);
    }
    if (pauli_type == 2 || pauli_type == 3) {
        xor_row(tab.zs[qubit]);
    }
}

/// Rewind a single-qubit Pauli (X, Y, or Z) through the tableau into a
/// freshly claimed noise_channel_masks slot. Returns the channel.
NoiseChannel rewind_single_pauli(HirModule& hir, const stim::TableauSimulator<kStimWidth>& sim,
                                 uint32_t qubit, int pauli_type, double prob) {
    auto h = hir.claim_empty_noise_channel_mask();
    auto slot = hir.noise_channel_masks.mut_at(h);
    slot.x().zero_out();
    slot.z().zero_out();
    accumulate_pauli_row(sim.inv_state, qubit, pauli_type, sim.inv_state.num_qubits, slot.x(),
                         slot.z());
    return NoiseChannel{h, prob};
}

/// Rewind a two-qubit Pauli through the tableau into a freshly claimed slot.
NoiseChannel rewind_two_qubit_pauli(HirModule& hir, const stim::TableauSimulator<kStimWidth>& sim,
                                    uint32_t q1, uint32_t q2, int pauli1, int pauli2, double prob) {
    auto h = hir.claim_empty_noise_channel_mask();
    auto slot = hir.noise_channel_masks.mut_at(h);
    slot.x().zero_out();
    slot.z().zero_out();
    uint32_t n = sim.inv_state.num_qubits;
    if (pauli1 != 0)
        accumulate_pauli_row(sim.inv_state, q1, pauli1, n, slot.x(), slot.z());
    if (pauli2 != 0)
        accumulate_pauli_row(sim.inv_state, q2, pauli2, n, slot.x(), slot.z());
    return NoiseChannel{h, prob};
}

NoiseSite make_single_qubit_noise_site(HirModule& hir,
                                       const stim::TableauSimulator<kStimWidth>& sim, GateType gate,
                                       uint32_t qubit, double prob) {
    NoiseSite site;
    switch (gate) {
        case GateType::X_ERROR:
            site.channels.push_back(rewind_single_pauli(hir, sim, qubit, 1, prob));
            break;
        case GateType::Y_ERROR:
            site.channels.push_back(rewind_single_pauli(hir, sim, qubit, 2, prob));
            break;
        case GateType::Z_ERROR:
            site.channels.push_back(rewind_single_pauli(hir, sim, qubit, 3, prob));
            break;
        case GateType::DEPOLARIZE1:
            site.channels.push_back(rewind_single_pauli(hir, sim, qubit, 1, prob / 3.0));
            site.channels.push_back(rewind_single_pauli(hir, sim, qubit, 2, prob / 3.0));
            site.channels.push_back(rewind_single_pauli(hir, sim, qubit, 3, prob / 3.0));
            break;
        default:
            throw std::runtime_error("Not a single-qubit noise gate");
    }
    return site;
}

NoiseSite make_depolarize2_noise_site(HirModule& hir, const stim::TableauSimulator<kStimWidth>& sim,
                                      uint32_t q1, uint32_t q2, double prob) {
    NoiseSite site;
    double channel_prob = prob / 15.0;
    for (int p1 = 0; p1 <= 3; ++p1) {
        for (int p2 = 0; p2 <= 3; ++p2) {
            if (p1 == 0 && p2 == 0)
                continue;
            site.channels.push_back(rewind_two_qubit_pauli(hir, sim, q1, q2, p1, p2, channel_prob));
        }
    }
    return site;
}

void accumulate_rz_global_phase(HirModule& hir, double alpha) {
    double angle = -alpha * std::numbers::pi / 2.0;
    hir.global_weight *= std::complex<double>(std::cos(angle), std::sin(angle));
}

/// Trace an R_Z(alpha) on a single qubit: extract rewound Z, append
/// PHASE_ROTATION, accumulate global phase. Writes mask data directly
/// into the arena slot.
void trace_rz(stim::TableauSimulator<kStimWidth>& sim, HirModule& hir, uint32_t qubit,
              double alpha) {
    auto& op = hir.append_phase_rotation_empty(alpha);
    auto slot = hir.mask_at(op);
    bool sign;
    extract_rewound_z_into(sim, qubit, slot.x(), slot.z(), sign);
    slot.set_sign(sign);
    double effective_alpha = sign ? -alpha : alpha;
    accumulate_rz_global_phase(hir, effective_alpha);
}

/// Trace an arbitrary Pauli rotation exp(-i*alpha*pi/2 * P).
void trace_pauli_rotation(stim::TableauSimulator<kStimWidth>& sim, HirModule& hir,
                          const stim::PauliString<kStimWidth>& obs, double alpha) {
    stim::PauliString<kStimWidth> rewound = sim.inv_state(obs);
    uint32_t n = sim.inv_state.num_qubits;
    auto& op = hir.append_phase_rotation_empty(alpha);
    auto slot = hir.mask_at(op);
    copy_rewound_into(rewound, n, slot.x(), slot.z());
    slot.set_sign(rewound.sign);
    double effective_alpha = rewound.sign ? -alpha : alpha;
    accumulate_rz_global_phase(hir, effective_alpha);
}

/// Build a stim::PauliString from an MPP/EXP_VAL/R_PAULI target list.
/// Sets `inversion_parity_out` if any target carries an inversion bang.
stim::PauliString<kStimWidth> build_pauli_string(const std::vector<Target>& targets,
                                                 uint32_t num_qubits, bool& inversion_parity_out) {
    stim::PauliString<kStimWidth> obs(num_qubits);
    inversion_parity_out = false;
    for (const auto& target : targets) {
        uint32_t q = target.value();
        inversion_parity_out ^= target.is_inverted();
        if (target.pauli() == Target::kPauliX) {
            obs.xs[q] = true;
        } else if (target.pauli() == Target::kPauliY) {
            obs.xs[q] = true;
            obs.zs[q] = true;
        } else {
            obs.zs[q] = true;
        }
    }
    return obs;
}

/// Conservative upper bound on the number of noise channel masks the
/// trace will emit. Some channels with prob = 0 are skipped, so the
/// actual count may be lower; the unused arena slots stay zero-init.
size_t count_noise_channels(const Circuit& circuit) {
    size_t count = 0;
    for (const auto& node : circuit.nodes) {
        const size_t n_targets = node.targets.size();
        switch (node.gate) {
            case GateType::X_ERROR:
            case GateType::Y_ERROR:
            case GateType::Z_ERROR:
                count += n_targets;
                break;
            case GateType::DEPOLARIZE1:
            case GateType::PAULI_CHANNEL_1:
                count += 3 * n_targets;
                break;
            case GateType::DEPOLARIZE2:
            case GateType::PAULI_CHANNEL_2:
                count += 15 * (n_targets / 2);
                break;
            default:
                break;
        }
    }
    return count;
}

/// Pre-count the number of mask-carrying HIR ops the trace will emit.
/// Must mirror the dispatch in trace().
size_t count_pauli_masks(const Circuit& circuit) {
    size_t count = 0;
    for (const auto& node : circuit.nodes) {
        const size_t n_targets = node.targets.size();
        switch (node.gate) {
            case GateType::T:
            case GateType::T_DAG:
            case GateType::M:
            case GateType::MX:
            case GateType::MY:
            case GateType::MPAD:
            case GateType::R_Z:
            case GateType::R_X:
            case GateType::R_Y:
                count += n_targets;
                break;
            case GateType::U3:
                count += 3 * n_targets;
                break;
            case GateType::R_XX:
            case GateType::R_YY:
            case GateType::R_ZZ:
                count += n_targets / 2;
                break;
            case GateType::R_PAULI:
            case GateType::EXP_VAL:
            case GateType::MPP:
                count += 1;
                break;
            case GateType::R:
            case GateType::RX:
            case GateType::RY:
            case GateType::MR:
            case GateType::MRX:
            case GateType::MRY:
                count += 2 * n_targets;
                break;
            case GateType::CX:
            case GateType::CY:
            case GateType::CZ:
                if (!node.targets.empty() && node.targets[0].is_rec()) {
                    count += n_targets / 2;
                }
                break;
            default:
                break;
        }
    }
    return count;
}

}  // namespace

HirModule trace(const Circuit& circuit) {
    HirModule hir(circuit.num_qubits, count_pauli_masks(circuit), count_noise_channels(circuit));
    hir.num_measurements = circuit.num_measurements;
    hir.num_detectors = circuit.num_detectors;
    hir.num_observables = circuit.num_observables;
    hir.num_exp_vals = circuit.num_exp_vals;

    std::mt19937_64 rng(0);
    stim::TableauSimulator<kStimWidth> sim(std::move(rng), circuit.num_qubits);

    MeasRecordIdx meas_idx{0};
    uint32_t hidden_meas_idx = circuit.num_measurements;
    ExpValIdx exp_val_idx{0};

    for (const auto& node : circuit.nodes) {
        const size_t ops_before = hir.ops.size();

        switch (node.gate) {
            // Single-qubit Cliffords
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
                    apply_single_qubit_clifford(sim, node.gate, target.value());
                }
                break;
            }

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
                if (!node.targets.empty() && node.targets[0].is_rec()) {
                    // Classical feedback: CX rec[-k] q or CZ rec[-k] q.
                    for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                        uint32_t rec_abs_idx = node.targets[i].value();
                        uint32_t target_qubit = node.targets[i + 1].value();
                        ControllingMeasIdx controlling_meas{rec_abs_idx};

                        auto& op = hir.append_conditional_empty(controlling_meas);
                        auto slot = hir.mask_at(op);
                        bool sign;
                        if (node.gate == GateType::CX) {
                            extract_rewound_x_into(sim, target_qubit, slot.x(), slot.z(), sign);
                        } else if (node.gate == GateType::CZ) {
                            extract_rewound_z_into(sim, target_qubit, slot.x(), slot.z(), sign);
                        } else {
                            throw std::runtime_error("CY classical feedback not supported");
                        }
                        slot.set_sign(sign);
                    }
                } else {
                    for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                        apply_two_qubit_clifford(sim, node.gate, node.targets[i].value(),
                                                 node.targets[i + 1].value());
                    }
                }
                break;
            }

            case GateType::T:
            case GateType::T_DAG: {
                bool dagger = (node.gate == GateType::T_DAG);
                for (const auto& target : node.targets) {
                    auto& op = hir.append_tgate_empty(dagger);
                    auto slot = hir.mask_at(op);
                    bool sign;
                    extract_rewound_z_into(sim, target.value(), slot.x(), slot.z(), sign);
                    slot.set_sign(sign);
                }
                break;
            }

            case GateType::R_Z: {
                double alpha = node.args[0];
                for (const auto& target : node.targets) {
                    trace_rz(sim, hir, target.value(), alpha);
                }
                break;
            }

            case GateType::R_X: {
                double alpha = node.args[0];
                for (const auto& target : node.targets) {
                    size_t q = static_cast<size_t>(target.value());
                    sim.inv_state.prepend_H_XZ(q);
                    trace_rz(sim, hir, target.value(), alpha);
                    sim.inv_state.prepend_H_XZ(q);
                }
                break;
            }

            case GateType::R_Y: {
                double alpha = node.args[0];
                for (const auto& target : node.targets) {
                    size_t q = static_cast<size_t>(target.value());
                    sim.inv_state.prepend_H_YZ(q);
                    trace_rz(sim, hir, target.value(), alpha);
                    sim.inv_state.prepend_H_YZ(q);
                }
                break;
            }

            // U3(theta, phi, lambda) = R_Z(phi) * R_Y(theta) * R_Z(lambda)
            case GateType::U3: {
                double theta = node.args[0];
                double phi = node.args[1];
                double lambda = node.args[2];
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    size_t q = static_cast<size_t>(qubit);

                    trace_rz(sim, hir, qubit, lambda);

                    sim.inv_state.prepend_H_YZ(q);
                    trace_rz(sim, hir, qubit, theta);
                    sim.inv_state.prepend_H_YZ(q);

                    trace_rz(sim, hir, qubit, phi);

                    double u3_phase = (phi + lambda) * std::numbers::pi / 2.0;
                    hir.global_weight *=
                        std::complex<double>(std::cos(u3_phase), std::sin(u3_phase));
                }
                break;
            }

            case GateType::R_XX:
            case GateType::R_YY:
            case GateType::R_ZZ: {
                double alpha = node.args[0];
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    if (q1 == q2) {
                        throw std::runtime_error("Duplicate qubit in pair rotation: q" +
                                                 std::to_string(q1));
                    }
                    stim::PauliString<kStimWidth> obs(circuit.num_qubits);
                    if (node.gate == GateType::R_XX) {
                        obs.xs[q1] = true;
                        obs.xs[q2] = true;
                    } else if (node.gate == GateType::R_YY) {
                        obs.xs[q1] = true;
                        obs.zs[q1] = true;
                        obs.xs[q2] = true;
                        obs.zs[q2] = true;
                    } else {
                        obs.zs[q1] = true;
                        obs.zs[q2] = true;
                    }
                    trace_pauli_rotation(sim, hir, obs, alpha);
                }
                break;
            }

            case GateType::R_PAULI: {
                double alpha = node.args[0];
                bool _;
                auto obs = build_pauli_string(node.targets, circuit.num_qubits, _);
                trace_pauli_rotation(sim, hir, obs, alpha);
                break;
            }

            case GateType::M: {
                for (const auto& target : node.targets) {
                    auto& op = hir.append_measure_empty(meas_idx);
                    auto slot = hir.mask_at(op);
                    bool sign;
                    extract_rewound_z_into(sim, target.value(), slot.x(), slot.z(), sign);
                    slot.set_sign(sign ^ target.is_inverted());
                    ++meas_idx;
                }
                break;
            }

            case GateType::MX: {
                for (const auto& target : node.targets) {
                    auto& op = hir.append_measure_empty(meas_idx);
                    auto slot = hir.mask_at(op);
                    bool sign;
                    extract_rewound_x_into(sim, target.value(), slot.x(), slot.z(), sign);
                    slot.set_sign(sign ^ target.is_inverted());
                    ++meas_idx;
                }
                break;
            }

            case GateType::MY: {
                for (const auto& target : node.targets) {
                    auto& op = hir.append_measure_empty(meas_idx);
                    auto slot = hir.mask_at(op);
                    bool sign;
                    extract_rewound_y_into(sim, target.value(), slot.x(), slot.z(), sign);
                    slot.set_sign(sign ^ target.is_inverted());
                    ++meas_idx;
                }
                break;
            }

            case GateType::MPP: {
                bool inversion_parity;
                auto obs = build_pauli_string(node.targets, circuit.num_qubits, inversion_parity);
                stim::PauliString<kStimWidth> rewound = sim.inv_state(obs);
                uint32_t n = sim.inv_state.num_qubits;
                auto& op = hir.append_measure_empty(meas_idx);
                auto slot = hir.mask_at(op);
                copy_rewound_into(rewound, n, slot.x(), slot.z());
                slot.set_sign(rewound.sign ^ inversion_parity);
                ++meas_idx;
                break;
            }

            // Reset / measure-reset decomposition.
            // Pattern: extract measurement observable -> emit MEASURE -> extract
            // correction -> emit CONDITIONAL_PAULI -> (MR only) optional readout noise.
            case GateType::R:
            case GateType::RX:
            case GateType::RY:
            case GateType::MR:
            case GateType::MRX:
            case GateType::MRY: {
                bool hidden = is_reset(node.gate);

                enum class Basis { Z, X, Y };
                Basis basis;
                switch (node.gate) {
                    case GateType::R:
                    case GateType::MR:
                        basis = Basis::Z;
                        break;
                    case GateType::RX:
                    case GateType::MRX:
                        basis = Basis::X;
                        break;
                    default:
                        basis = Basis::Y;
                        break;
                }

                auto extract_meas = [&](uint32_t q, MutableMaskView dm, MutableMaskView sm,
                                        bool& s) {
                    switch (basis) {
                        case Basis::Z:
                            extract_rewound_z_into(sim, q, dm, sm, s);
                            break;
                        case Basis::X:
                            extract_rewound_x_into(sim, q, dm, sm, s);
                            break;
                        case Basis::Y:
                            extract_rewound_y_into(sim, q, dm, sm, s);
                            break;
                    }
                };

                auto extract_corr = [&](uint32_t q, MutableMaskView dm, MutableMaskView sm,
                                        bool& s) {
                    if (basis == Basis::Z)
                        extract_rewound_x_into(sim, q, dm, sm, s);
                    else
                        extract_rewound_z_into(sim, q, dm, sm, s);
                };

                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();

                    uint32_t this_meas;
                    if (hidden) {
                        this_meas = hidden_meas_idx++;
                        auto& meas_op = hir.append_measure_empty(MeasRecordIdx{this_meas});
                        meas_op.set_hidden(true);
                        auto slot = hir.mask_at(meas_op);
                        bool sign;
                        extract_meas(qubit, slot.x(), slot.z(), sign);
                        slot.set_sign(sign);
                    } else {
                        this_meas = static_cast<uint32_t>(meas_idx);
                        auto& meas_op = hir.append_measure_empty(meas_idx);
                        auto slot = hir.mask_at(meas_op);
                        bool sign;
                        extract_meas(qubit, slot.x(), slot.z(), sign);
                        slot.set_sign(sign);
                        ++meas_idx;
                    }

                    auto& cond_op = hir.append_conditional_empty(ControllingMeasIdx{this_meas});
                    auto cond_slot = hir.mask_at(cond_op);
                    bool corr_sign;
                    extract_corr(qubit, cond_slot.x(), cond_slot.z(), corr_sign);
                    cond_slot.set_sign(corr_sign);

                    if (!hidden && target.is_inverted()) {
                        ReadoutNoiseIdx idx{static_cast<uint32_t>(hir.readout_noise.size())};
                        hir.readout_noise.push_back({this_meas, 1.0});
                        hir.append_readout_noise(idx);
                    }
                }
                break;
            }

            case GateType::MPAD: {
                for (const auto& target : node.targets) {
                    bool sign = (target.value() != 0) ^ target.is_inverted();
                    auto& op = hir.append_measure_empty(meas_idx);
                    auto slot = hir.mask_at(op);
                    slot.x().zero_out();
                    slot.z().zero_out();
                    slot.set_sign(sign);
                    ++meas_idx;
                }
                break;
            }

            case GateType::TICK:
                break;

            case GateType::X_ERROR:
            case GateType::Y_ERROR:
            case GateType::Z_ERROR:
            case GateType::DEPOLARIZE1: {
                double prob = node.args.empty() ? 0.0 : node.args[0];
                for (const auto& target : node.targets) {
                    NoiseSite site =
                        make_single_qubit_noise_site(hir, sim, node.gate, target.value(), prob);
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    hir.append_noise(idx);
                }
                break;
            }

            case GateType::PAULI_CHANNEL_1: {
                if (node.args.size() < 3) {
                    throw std::runtime_error(
                        "PAULI_CHANNEL_1 requires 3 arguments: P(X), P(Y), P(Z)");
                }
                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();
                    NoiseSite site;
                    for (int p = 0; p < 3; ++p) {
                        double prob = node.args[static_cast<size_t>(p)];
                        if (prob > 0.0) {
                            site.channels.push_back(
                                rewind_single_pauli(hir, sim, qubit, p + 1, prob));
                        }
                    }
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    hir.append_noise(idx);
                }
                break;
            }

            case GateType::PAULI_CHANNEL_2: {
                if (node.args.size() < 15) {
                    throw std::runtime_error("PAULI_CHANNEL_2 requires 15 arguments");
                }
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    NoiseSite site;
                    size_t arg_idx = 0;
                    for (int p1 = 0; p1 <= 3; ++p1) {
                        for (int p2 = 0; p2 <= 3; ++p2) {
                            if (p1 == 0 && p2 == 0)
                                continue;
                            double prob = node.args[arg_idx];
                            if (prob > 0.0) {
                                site.channels.push_back(
                                    rewind_two_qubit_pauli(hir, sim, q1, q2, p1, p2, prob));
                            }
                            ++arg_idx;
                        }
                    }
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    hir.append_noise(idx);
                }
                break;
            }

            case GateType::DEPOLARIZE2: {
                double prob = node.args.empty() ? 0.0 : node.args[0];
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    NoiseSite site = make_depolarize2_noise_site(hir, sim, q1, q2, prob);
                    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
                    hir.noise_sites.push_back(std::move(site));
                    hir.append_noise(idx);
                }
                break;
            }

            case GateType::READOUT_NOISE: {
                for (const auto& target : node.targets) {
                    uint32_t abs_meas_idx = target.value();
                    double prob = node.args.empty() ? 0.0 : node.args[0];
                    ReadoutNoiseIdx idx{static_cast<uint32_t>(hir.readout_noise.size())};
                    hir.readout_noise.push_back({abs_meas_idx, prob});
                    hir.append_readout_noise(idx);
                }
                break;
            }

            case GateType::DETECTOR: {
                std::vector<uint32_t> targets;
                for (const auto& target : node.targets) {
                    targets.push_back(target.value());
                }
                DetectorIdx idx{static_cast<uint32_t>(hir.detector_targets.size())};
                hir.detector_targets.push_back(std::move(targets));
                hir.append_detector(idx);
                break;
            }

            case GateType::OBSERVABLE_INCLUDE: {
                std::vector<uint32_t> targets;
                for (const auto& target : node.targets) {
                    targets.push_back(target.value());
                }
                uint32_t obs_idx = static_cast<uint32_t>(node.args.empty() ? 0.0 : node.args[0]);
                uint32_t target_list_idx = static_cast<uint32_t>(hir.observable_targets.size());
                hir.observable_targets.push_back(std::move(targets));
                hir.append_observable(ObservableIdx{obs_idx}, target_list_idx);
                break;
            }

            case GateType::EXP_VAL: {
                bool inversion_parity;
                auto obs = build_pauli_string(node.targets, circuit.num_qubits, inversion_parity);
                stim::PauliString<kStimWidth> rewound = sim.inv_state(obs);
                uint32_t n = sim.inv_state.num_qubits;
                auto& op = hir.append_exp_val_empty(exp_val_idx);
                auto slot = hir.mask_at(op);
                copy_rewound_into(rewound, n, slot.x(), slot.z());
                slot.set_sign(rewound.sign ^ inversion_parity);
                exp_val_idx = ExpValIdx{static_cast<uint32_t>(exp_val_idx) + 1};
                break;
            }

            default:
                throw std::runtime_error("Unsupported gate type in Front-End: " +
                                         std::string(gate_name(node.gate)));
        }

        // Source map: append one entry per op produced by this node.
        const size_t ops_after = hir.ops.size();
        for (size_t i = ops_before; i < ops_after; ++i) {
            hir.source_map.push_back({node.source_line});
        }
    }

    hir.num_hidden_measurements = hidden_meas_idx - circuit.num_measurements;
    hir.final_tableau = sim.inv_state.inverse();

    return hir;
}

}  // namespace clifft
