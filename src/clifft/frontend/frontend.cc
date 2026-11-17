#include "clifft/frontend/frontend.h"

#include "clifft/tableau/tableau.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

void apply_single_qubit_clifford(Tableau& inverse, GateType gate, uint32_t qubit) {
    const std::array targets{qubit};
    inverse.prepend_named_gate(Tableau::inverse_named_gate(gate), targets);
}

void apply_two_qubit_clifford(Tableau& inverse, GateType gate, uint32_t q1, uint32_t q2) {
    const std::array targets{q1, q2};
    inverse.prepend_named_gate(Tableau::inverse_named_gate(gate), targets);
}

// Sparse copy of a rewound Pauli, optionally with its sign flipped.
void assign_rewound(PauliMask& mask, PauliStringView pauli, bool flip_sign = false) {
    mask.assign(pauli.x(), pauli.z(), pauli.sign() != flip_sign);
}

enum class PauliAxis : uint8_t { X, Y, Z };

// Sparse copy of a single-qubit Pauli observable rewound through the tableau.
void assign_rewound_axis(PauliMask& mask, const Tableau& inverse, uint32_t qubit, PauliAxis axis,
                         bool flip_sign = false) {
    switch (axis) {
        case PauliAxis::X:
            assign_rewound(mask, inverse.x_output(qubit), flip_sign);
            break;
        case PauliAxis::Y:
            assign_rewound(mask, inverse.y_output(qubit).view(), flip_sign);
            break;
        case PauliAxis::Z:
            assign_rewound(mask, inverse.z_output(qubit), flip_sign);
            break;
    }
}

// Rewound X and Z generators of the qubits one noise site acts on, read
// from the tableau once in sparse form. Each channel of the site is then a
// sparse product of these rows; channel signs are irrelevant. One instance
// serves every site of a trace so its storage is reused.
class RewoundGenerators {
  public:
    void load(const Tableau& inverse, std::initializer_list<uint32_t> qubits) {
        assert(qubits.size() <= kMaxQubits);
        size_t i = 0;
        for (uint32_t qubit : qubits) {
            const PauliStringView x = inverse.x_output(qubit);
            const PauliStringView z = inverse.z_output(qubit);
            x_[i].assign(x.x(), x.z(), false);
            z_[i].assign(z.x(), z.z(), false);
            ++i;
        }
    }

    // Store the product of one Pauli per loaded qubit (0=I, 1=X, 2=Y, 3=Z)
    // as a noise channel of the given probability.
    NoiseChannel channel(HirModule& hir, std::initializer_list<int> pauli_types, double prob) {
        product_.clear();
        size_t i = 0;
        for (int pauli_type : pauli_types) {
            if (pauli_type == 1 || pauli_type == 2) {
                product_.multiply_body(x_[i]);
            }
            if (pauli_type == 2 || pauli_type == 3) {
                product_.multiply_body(z_[i]);
            }
            ++i;
        }
        return NoiseChannel{hir.noise_channel_masks.append(product_), prob};
    }

  private:
    static constexpr size_t kMaxQubits = 3;
    std::array<PauliMask, kMaxQubits> x_;
    std::array<PauliMask, kMaxQubits> z_;
    PauliMask product_;
};

/// Rewind a Pauli product stored as Pauli-tagged targets into a new noise channel.
NoiseChannel rewind_pauli_targets(HirModule& hir, const Tableau& inverse,
                                  const std::vector<Target>& targets, double prob) {
    PauliMask product;
    PauliMask row;
    for (const auto& target : targets) {
        if (!target.has_pauli()) {
            throw std::runtime_error("Expected Pauli target");
        }
        const bool has_x = target.pauli() == Target::kPauliX || target.pauli() == Target::kPauliY;
        const bool has_z = target.pauli() == Target::kPauliZ || target.pauli() == Target::kPauliY;
        if (has_x) {
            assign_rewound(row, inverse.x_output(target.value()));
            product.multiply_body(row);
        }
        if (has_z) {
            assign_rewound(row, inverse.z_output(target.value()));
            product.multiply_body(row);
        }
    }
    product.set_sign(false);
    return NoiseChannel{hir.noise_channel_masks.append(product), prob};
}

void append_noise_site(HirModule& hir, NoiseSite site, double total_probability,
                       size_t source_node) {
    site.total_probability = total_probability;
    site.source_node = static_cast<uint32_t>(source_node);
    NoiseSiteIdx idx{static_cast<uint32_t>(hir.noise_sites.size())};
    hir.noise_sites.push_back(std::move(site));
    hir.append_noise(idx);
}

double sum_noise_probabilities(const NoiseSite& site) {
    double total_probability = 0.0;
    for (const NoiseChannel& channel : site.channels) {
        total_probability += channel.prob;
    }
    // Each channel probability is validated individually and the channels sum to at most
    // one in exact arithmetic, but the floating-point sum can exceed one by an ulp (a
    // correlated chain whose last ELSE link has probability one is the known case) and the
    // plan validator checks the declared total strictly.
    return std::min(total_probability, 1.0);
}

double represented_depolarizing_probability(double probability, double outcome_count) {
    // A channel probability rounded to zero cannot be selected by either
    // executor, so fixed-k conditioning must also treat the site as impossible.
    return probability / outcome_count == 0.0 ? 0.0 : probability;
}

NoiseSite make_single_qubit_noise_site(HirModule& hir, const Tableau& inverse,
                                       RewoundGenerators& generators, GateType gate, uint32_t qubit,
                                       double prob) {
    generators.load(inverse, {qubit});
    NoiseSite site;
    switch (gate) {
        case GateType::X_ERROR:
            site.channels.push_back(generators.channel(hir, {1}, prob));
            break;
        case GateType::Y_ERROR:
            site.channels.push_back(generators.channel(hir, {2}, prob));
            break;
        case GateType::Z_ERROR:
            site.channels.push_back(generators.channel(hir, {3}, prob));
            break;
        case GateType::DEPOLARIZE1:
            site.channels.push_back(generators.channel(hir, {1}, prob / 3.0));
            site.channels.push_back(generators.channel(hir, {2}, prob / 3.0));
            site.channels.push_back(generators.channel(hir, {3}, prob / 3.0));
            break;
        default:
            throw std::runtime_error("Not a single-qubit noise gate");
    }
    return site;
}

NoiseSite make_depolarize2_noise_site(HirModule& hir, const Tableau& inverse,
                                      RewoundGenerators& generators, uint32_t q1, uint32_t q2,
                                      double prob) {
    generators.load(inverse, {q1, q2});
    NoiseSite site;
    site.channels.reserve(15);
    double channel_prob = prob / 15.0;
    for (int p1 = 0; p1 <= 3; ++p1) {
        for (int p2 = 0; p2 <= 3; ++p2) {
            if (p1 == 0 && p2 == 0)
                continue;
            site.channels.push_back(generators.channel(hir, {p1, p2}, channel_prob));
        }
    }
    return site;
}

NoiseSite make_depolarize3_noise_site(HirModule& hir, const Tableau& inverse,
                                      RewoundGenerators& generators, uint32_t q1, uint32_t q2,
                                      uint32_t q3, double prob) {
    generators.load(inverse, {q1, q2, q3});
    NoiseSite site;
    site.channels.reserve(63);
    double channel_prob = prob / 63.0;
    for (int p1 = 0; p1 <= 3; ++p1) {
        for (int p2 = 0; p2 <= 3; ++p2) {
            for (int p3 = 0; p3 <= 3; ++p3) {
                if (p1 == 0 && p2 == 0 && p3 == 0)
                    continue;
                site.channels.push_back(generators.channel(hir, {p1, p2, p3}, channel_prob));
            }
        }
    }
    return site;
}

// Absorb a native Clifford representative. Its omitted scalar phase is
// permitted by the projective statevector contract.
bool try_absorb_clifford_axis_rotation(Tableau& inverse, uint32_t qubit, double alpha,
                                       GateType sqrt_gate, GateType pauli_gate,
                                       GateType sqrt_dag_gate) {
    const auto rotation = classify_clifford_rotation(alpha);
    if (!rotation.has_value()) {
        return false;
    }

    switch (*rotation) {
        case CliffordRotation::IDENTITY:
            break;
        case CliffordRotation::SQRT:
            apply_single_qubit_clifford(inverse, sqrt_gate, qubit);
            break;
        case CliffordRotation::PAULI:
            apply_single_qubit_clifford(inverse, pauli_gate, qubit);
            break;
        case CliffordRotation::SQRT_DAG:
            apply_single_qubit_clifford(inverse, sqrt_dag_gate, qubit);
            break;
    }
    return true;
}

void apply_clifford_pair_rotation(Tableau& inverse, uint32_t q1, uint32_t q2,
                                  CliffordRotation rotation, GateType sqrt_gate,
                                  GateType pauli_gate, GateType sqrt_dag_gate) {
    switch (rotation) {
        case CliffordRotation::IDENTITY:
            break;
        case CliffordRotation::SQRT:
            apply_two_qubit_clifford(inverse, sqrt_gate, q1, q2);
            break;
        case CliffordRotation::PAULI:
            apply_single_qubit_clifford(inverse, pauli_gate, q1);
            apply_single_qubit_clifford(inverse, pauli_gate, q2);
            break;
        case CliffordRotation::SQRT_DAG:
            apply_two_qubit_clifford(inverse, sqrt_dag_gate, q1, q2);
            break;
    }
}

struct PairRotationInfo {
    GateType sqrt_gate;
    GateType pauli_gate;
    GateType sqrt_dag_gate;
    bool x;
    bool z;
};

PairRotationInfo pair_rotation_info(GateType gate) {
    switch (gate) {
        case GateType::R_XX:
            return {GateType::SQRT_XX, GateType::X, GateType::SQRT_XX_DAG, true, false};
        case GateType::R_YY:
            return {GateType::SQRT_YY, GateType::Y, GateType::SQRT_YY_DAG, true, true};
        case GateType::R_ZZ:
            return {GateType::SQRT_ZZ, GateType::Z, GateType::SQRT_ZZ_DAG, false, true};
        default:
            throw std::invalid_argument("Not a pair rotation gate: " +
                                        std::string(gate_name(gate)));
    }
}

void apply_sqrt_pauli_product_clifford(Tableau& inverse, PauliStringView pauli, bool dagger) {
    if (pauli.x().is_zero() && pauli.z().is_zero()) {
        return;
    }
    inverse.prepend_pauli_rotation(pauli, !dagger);
}

// A signed axis swaps the two square-root directions. A full Pauli ignores
// that sign because it changes only the omitted global phase.
bool try_absorb_clifford_pauli_rotation(Tableau& inverse, PauliStringView pauli, double alpha) {
    const auto rotation = classify_clifford_rotation(alpha);
    if (!rotation.has_value()) {
        return false;
    }

    switch (*rotation) {
        case CliffordRotation::IDENTITY:
            break;
        case CliffordRotation::SQRT:
            apply_sqrt_pauli_product_clifford(inverse, pauli, false);
            break;
        case CliffordRotation::PAULI:
            inverse.prepend_pauli(pauli);
            break;
        case CliffordRotation::SQRT_DAG:
            apply_sqrt_pauli_product_clifford(inverse, pauli, true);
            break;
    }
    return true;
}

// Trace R_Z(alpha) on a single qubit by extracting its rewound Z axis. The
// source gate and emitted exponential may differ by a global phase, which is
// intentionally omitted by the projective state contract.
void trace_rz(Tableau& inverse, HirModule& hir, PauliMask& mask, uint32_t qubit, double alpha) {
    if (try_absorb_clifford_axis_rotation(inverse, qubit, alpha, GateType::S, GateType::Z,
                                          GateType::S_DAG)) {
        return;
    }

    assign_rewound_axis(mask, inverse, qubit, PauliAxis::Z);
    hir.append_phase_rotation(alpha, mask);
}

// Trace an arbitrary Pauli rotation exp(-i*alpha*pi/2 * P).
void trace_pauli_rotation(Tableau& inverse, HirModule& hir, PauliStringView observable,
                          double alpha) {
    if (try_absorb_clifford_pauli_rotation(inverse, observable, alpha)) {
        return;
    }

    const PauliString rewound = inverse.apply(observable);
    hir.append_phase_rotation(alpha, PauliMask(rewound.x(), rewound.z(), rewound.sign()));
}

/// Build a PauliString from an MPP/EXP_VAL/R_PAULI target list.
/// Sets `inversion_parity_out` if any target carries an inversion bang.
PauliString build_pauli_string(const std::vector<Target>& targets, uint32_t num_qubits,
                               bool& inversion_parity_out) {
    PauliString observable(num_qubits);
    inversion_parity_out = false;
    for (const auto& target : targets) {
        uint32_t q = target.value();
        inversion_parity_out ^= target.is_inverted();
        if (target.pauli() == Target::kPauliX) {
            observable.set_pauli(q, true, false);
        } else if (target.pauli() == Target::kPauliY) {
            observable.set_pauli(q, true, true);
        } else {
            observable.set_pauli(q, false, true);
        }
    }
    observable.set_sign(false);
    return observable;
}

// InstrumentTraceOptions is a C++ trace boundary in its own right, even
// though the noncomputational model normally constructs it from already
// validated matrices. Validate raw specs here so a malformed compressed
// payload cannot produce a negative square root or an invalid destination
// draw downstream.
void validate_instrument_probabilities(const InstrumentProbabilities& probabilities,
                                       const std::string& site) {
    // Keep this at least as loose as the model layer's kProbTolerance.
    // TransitionInstrument clamps column sums within that tolerance to 1.
    constexpr double kTolerance = 1e-12;

    for (uint8_t source = 0; source < 2; ++source) {
        const double p_fire = probabilities.p_fire[source];
        if (!is_probability(p_fire)) {
            throw std::invalid_argument("trace: " + site + " has invalid p_fire[" +
                                        std::to_string(source) + "] = " + std::to_string(p_fire));
        }

        double p_computational = 0.0;
        for (uint8_t destination = 0; destination < 2; ++destination) {
            const double p = probabilities.p_computational_dest[source][destination];
            if (!is_probability(p)) {
                throw std::invalid_argument(
                    "trace: " + site + " has invalid p_computational_dest[" +
                    std::to_string(source) + "][" + std::to_string(destination) +
                    "] = " + std::to_string(p));
            }
            p_computational += p;
        }
        if (p_computational > p_fire + kTolerance) {
            throw std::invalid_argument("trace: " + site +
                                        " has computational destination probability " +
                                        std::to_string(p_computational) + " above p_fire[" +
                                        std::to_string(source) + "] = " + std::to_string(p_fire));
        }
    }
}

}  // namespace

HirModule trace(const Circuit& circuit, const InstrumentTraceOptions* instruments) {
    if (circuit.nodes.size() > kUnknownSourceNode) {
        throw std::overflow_error("Circuit exceeds the HIR source-node index limit");
    }
    // Avoid an accidental multi-gigabyte tableau allocation. This
    // conservative safety ceiling is far above practical circuit sizes.
    if (circuit.num_qubits > 65536) {
        throw std::runtime_error("Circuit exceeds the 65536-qubit frontend safety limit: " +
                                 std::to_string(circuit.num_qubits) + " qubits");
    }
    if (instruments != nullptr && instruments->forced_traceout_node.has_value()) {
        const size_t node_index = *instruments->forced_traceout_node;
        if (node_index >= circuit.nodes.size() || !is_reset(circuit.nodes[node_index].gate) ||
            circuit.nodes[node_index].targets.size() != 1) {
            throw std::invalid_argument(
                "trace: forced_traceout_node must name a single-target reset");
        }
    }

    HirModule hir(circuit.num_qubits);
    hir.num_measurements = circuit.num_measurements;
    hir.num_detectors = circuit.num_detectors;
    hir.num_observables = circuit.num_observables;
    hir.num_exp_vals = circuit.num_exp_vals;
    hir.neglect_instrument_damping =
        instruments != nullptr && instruments->neglect_instrument_damping;

    Tableau inverse(circuit.num_qubits);
    // Reused for every rewound op mask and noise site so tracing does not
    // allocate per op.
    PauliMask mask;
    RewoundGenerators generators;

    MeasRecordIdx meas_idx{0};
    uint32_t hidden_meas_idx = circuit.num_measurements;
    ExpValIdx exp_val_idx{0};

    for (size_t node_index = 0; node_index < circuit.nodes.size(); ++node_index) {
        const auto& node = circuit.nodes[node_index];
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
                    apply_single_qubit_clifford(inverse, node.gate, target.value());
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

                        if (node.gate == GateType::CX) {
                            assign_rewound_axis(mask, inverse, target_qubit, PauliAxis::X);
                        } else if (node.gate == GateType::CZ) {
                            assign_rewound_axis(mask, inverse, target_qubit, PauliAxis::Z);
                        } else {
                            throw std::runtime_error("CY classical feedback not supported");
                        }
                        hir.append_conditional(controlling_meas, mask);
                    }
                } else {
                    for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                        apply_two_qubit_clifford(inverse, node.gate, node.targets[i].value(),
                                                 node.targets[i + 1].value());
                    }
                }
                break;
            }

            case GateType::T:
            case GateType::T_DAG: {
                bool dagger = (node.gate == GateType::T_DAG);
                for (const auto& target : node.targets) {
                    assign_rewound_axis(mask, inverse, target.value(), PauliAxis::Z);
                    hir.append_tgate(dagger, mask);
                }
                break;
            }

            case GateType::SPP:
            case GateType::SPP_DAG: {
                bool inversion_parity;
                auto obs = build_pauli_string(node.targets, circuit.num_qubits, inversion_parity);
                obs.set_sign(inversion_parity);
                const bool dagger = node.gate == GateType::SPP_DAG;
                trace_pauli_rotation(inverse, hir, obs.view(), dagger ? -0.5 : 0.5);
                break;
            }

            case GateType::TPP:
            case GateType::TPP_DAG: {
                bool inversion_parity;
                auto obs = build_pauli_string(node.targets, circuit.num_qubits, inversion_parity);
                const PauliString rewound = inverse.apply(obs.view());
                bool dagger = node.gate == GateType::TPP_DAG;
                assign_rewound(mask, rewound.view(), inversion_parity);
                hir.append_tgate(dagger, mask);
                break;
            }

            case GateType::R_Z: {
                double alpha = node.args[0];
                for (const auto& target : node.targets) {
                    trace_rz(inverse, hir, mask, target.value(), alpha);
                }
                break;
            }

            case GateType::R_X: {
                double alpha = node.args[0];
                for (const auto& target : node.targets) {
                    if (try_absorb_clifford_axis_rotation(inverse, target.value(), alpha,
                                                          GateType::SQRT_X, GateType::X,
                                                          GateType::SQRT_X_DAG)) {
                        continue;
                    }
                    apply_single_qubit_clifford(inverse, GateType::H, target.value());
                    trace_rz(inverse, hir, mask, target.value(), alpha);
                    apply_single_qubit_clifford(inverse, GateType::H, target.value());
                }
                break;
            }

            case GateType::R_Y: {
                double alpha = node.args[0];
                for (const auto& target : node.targets) {
                    if (try_absorb_clifford_axis_rotation(inverse, target.value(), alpha,
                                                          GateType::SQRT_Y, GateType::Y,
                                                          GateType::SQRT_Y_DAG)) {
                        continue;
                    }
                    apply_single_qubit_clifford(inverse, GateType::H_YZ, target.value());
                    trace_rz(inverse, hir, mask, target.value(), alpha);
                    apply_single_qubit_clifford(inverse, GateType::H_YZ, target.value());
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
                    trace_rz(inverse, hir, mask, qubit, lambda);

                    if (!try_absorb_clifford_axis_rotation(inverse, qubit, theta, GateType::SQRT_Y,
                                                           GateType::Y, GateType::SQRT_Y_DAG)) {
                        apply_single_qubit_clifford(inverse, GateType::H_YZ, qubit);
                        trace_rz(inverse, hir, mask, qubit, theta);
                        apply_single_qubit_clifford(inverse, GateType::H_YZ, qubit);
                    }

                    trace_rz(inverse, hir, mask, qubit, phi);
                }
                break;
            }

            case GateType::R_XX:
            case GateType::R_YY:
            case GateType::R_ZZ: {
                double alpha = node.args[0];
                const auto clifford_rotation = classify_clifford_rotation(alpha);
                const PairRotationInfo info = pair_rotation_info(node.gate);
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    if (q1 == q2) {
                        throw std::runtime_error("Duplicate qubit in pair rotation: q" +
                                                 std::to_string(q1));
                    }

                    if (clifford_rotation.has_value()) {
                        apply_clifford_pair_rotation(inverse, q1, q2, *clifford_rotation,
                                                     info.sqrt_gate, info.pauli_gate,
                                                     info.sqrt_dag_gate);
                        continue;
                    }

                    PauliString observable(circuit.num_qubits);
                    observable.set_pauli(q1, info.x, info.z);
                    observable.set_pauli(q2, info.x, info.z);
                    observable.set_sign(false);
                    trace_pauli_rotation(inverse, hir, observable.view(), alpha);
                }
                break;
            }

            case GateType::R_PAULI: {
                double alpha = node.args[0];
                bool inverted;
                auto obs = build_pauli_string(node.targets, circuit.num_qubits, inverted);
                obs.set_sign(inverted);
                trace_pauli_rotation(inverse, hir, obs.view(), alpha);
                break;
            }

            case GateType::M: {
                for (const auto& target : node.targets) {
                    assign_rewound_axis(mask, inverse, target.value(), PauliAxis::Z,
                                        target.is_inverted());
                    hir.append_measure(meas_idx, mask);
                    ++meas_idx;
                }
                break;
            }

            case GateType::MX: {
                for (const auto& target : node.targets) {
                    assign_rewound_axis(mask, inverse, target.value(), PauliAxis::X,
                                        target.is_inverted());
                    hir.append_measure(meas_idx, mask);
                    ++meas_idx;
                }
                break;
            }

            case GateType::MY: {
                for (const auto& target : node.targets) {
                    assign_rewound_axis(mask, inverse, target.value(), PauliAxis::Y,
                                        target.is_inverted());
                    hir.append_measure(meas_idx, mask);
                    ++meas_idx;
                }
                break;
            }

            case GateType::MPP: {
                bool inversion_parity;
                auto obs = build_pauli_string(node.targets, circuit.num_qubits, inversion_parity);
                const PauliString rewound = inverse.apply(obs.view());
                assign_rewound(mask, rewound.view(), inversion_parity);
                hir.append_measure(meas_idx, mask);
                ++meas_idx;
                break;
            }

            // Reset / measure-reset decomposition.
            // Pattern: extract measurement observable -> emit MEASURE -> extract
            // correction -> emit CONDITIONAL_PAULI.
            case GateType::R:
            case GateType::RX:
            case GateType::RY:
            case GateType::MR:
            case GateType::MRX:
            case GateType::MRY: {
                bool hidden = is_reset(node.gate);

                PauliAxis basis;
                switch (node.gate) {
                    case GateType::R:
                    case GateType::MR:
                        basis = PauliAxis::Z;
                        break;
                    case GateType::RX:
                    case GateType::MRX:
                        basis = PauliAxis::X;
                        break;
                    default:
                        basis = PauliAxis::Y;
                        break;
                }
                // The correction anticommutes with the measured axis.
                const PauliAxis correction = basis == PauliAxis::Z ? PauliAxis::X : PauliAxis::Z;

                for (const auto& target : node.targets) {
                    uint32_t qubit = target.value();

                    // MR !q is X q then MR q (Z q for MRX and MRY): the Pauli
                    // flips the outcome and the reset still prepares the +1
                    // eigenstate, so the frame carries the inversion.
                    if (!hidden && target.is_inverted()) {
                        apply_single_qubit_clifford(
                            inverse, basis == PauliAxis::Z ? GateType::X : GateType::Z, qubit);
                    }

                    uint32_t this_meas;
                    assign_rewound_axis(mask, inverse, qubit, basis);
                    if (hidden) {
                        this_meas = hidden_meas_idx++;
                        auto& meas_op = hir.append_measure(MeasRecordIdx{this_meas}, mask);
                        meas_op.set_hidden(true);
                        // Report the hidden slot to the caller when this node
                        // is the single-target reset validated above.
                        if (instruments != nullptr &&
                            instruments->forced_traceout_node.has_value() &&
                            node_index == *instruments->forced_traceout_node) {
                            hir.forced_traceout_slot = this_meas;
                        }
                    } else {
                        this_meas = static_cast<uint32_t>(meas_idx);
                        hir.append_measure(meas_idx, mask);
                        ++meas_idx;
                    }

                    assign_rewound_axis(mask, inverse, qubit, correction);
                    hir.append_conditional(ControllingMeasIdx{this_meas}, mask);
                }
                break;
            }

            case GateType::LEVEL_TRANSITION:
            case GateType::LEAKAGE:
            case GateType::LOSS:
            case GateType::HERALD_LEAKAGE_EVENT:
            case GateType::HERALD_LOSS_EVENT:
            case GateType::LEAKAGE_INTERACTION:
            case GateType::LOSS_INTERACTION: {
                if (instruments == nullptr) {
                    throw std::invalid_argument(
                        std::string(gate_name(node.gate)) +
                        " is a noncomputational annotation; run the circuit through "
                        "clifft.noncomp.sample instead of compiling it directly");
                }
                if (is_noncomputational_herald(node.gate) ||
                    is_noncomputational_interaction(node.gate)) {
                    throw std::invalid_argument("trace: unlowered " +
                                                std::string(gate_name(node.gate)) + " at node " +
                                                std::to_string(node_index) +
                                                "; status-dependent annotations must be lowered by "
                                                "the noncomputational rewriter "
                                                "before tracing with instruments");
                }
                for (const auto& target : node.targets) {
                    const uint32_t qubit = target.value();
                    InstrumentSite site;
                    site.qubit = qubit;
                    std::string site_description;
                    if (is_inline_noncomputational_annotation(node.gate)) {
                        // Both inline channels have an equal rate from G and E
                        // and destinations entirely in the trap remainder. A
                        // missing argument is a malformed node, not a zero-rate
                        // annotation.
                        if (node.args.size() != 1) {
                            throw std::runtime_error(
                                "trace: " + std::string(gate_name(node.gate)) + " at line " +
                                std::to_string(node.source_line) +
                                " requires exactly one argument (the probability)");
                        }
                        const double p = node.args[0];
                        site.probabilities.p_fire[0] = p;
                        site.probabilities.p_fire[1] = p;
                        site_description = std::string(gate_name(node.gate)) + " at line " +
                                           std::to_string(node.source_line);
                    } else {
                        const auto it = instruments->transitions.find(node.tag);
                        if (it == instruments->transitions.end()) {
                            throw std::runtime_error(
                                "trace: LEVEL_TRANSITION[" + node.tag + "] at line " +
                                std::to_string(node.source_line) +
                                " does not name a transition in the instrument options");
                        }
                        site.probabilities = it->second;
                        site_description = "LEVEL_TRANSITION[" + node.tag + "] at line " +
                                           std::to_string(node.source_line);
                    }
                    validate_instrument_probabilities(site.probabilities, site_description);
                    // A site that can never fire is the identity channel.
                    if (site.probabilities.p_fire[0] == 0.0 &&
                        site.probabilities.p_fire[1] == 0.0) {
                        continue;
                    }

                    // Destination flip: the rewound X observable, stored
                    // in the arena so downstream mask conjugation can
                    // reach it through the side-table handle.
                    assign_rewound_axis(mask, inverse, qubit, PauliAxis::X);
                    site.destination_flip_mask = hir.pauli_masks.append(mask);

                    const InstrumentSiteIdx site_idx{
                        static_cast<uint32_t>(hir.instrument_sites.size())};
                    hir.instrument_sites.push_back(site);
                    assign_rewound_axis(mask, inverse, qubit, PauliAxis::Z);
                    hir.append_instrument(site_idx, mask);
                }
                break;
            }

            case GateType::MPAD: {
                for (const auto& target : node.targets) {
                    bool sign = (target.value() != 0) ^ target.is_inverted();
                    mask.clear();
                    mask.set_sign(sign);
                    hir.append_measure(meas_idx, mask);
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
                    NoiseSite site = make_single_qubit_noise_site(hir, inverse, generators,
                                                                  node.gate, target.value(), prob);
                    const double total_probability =
                        node.gate == GateType::DEPOLARIZE1
                            ? represented_depolarizing_probability(prob, 3.0)
                            : prob;
                    append_noise_site(hir, std::move(site), total_probability, node_index);
                }
                break;
            }

            case GateType::PAULI_CHANNEL_1: {
                if (node.args.size() < 3) {
                    throw std::runtime_error(
                        "PAULI_CHANNEL_1 requires 3 arguments: P(X), P(Y), P(Z)");
                }
                for (const auto& target : node.targets) {
                    generators.load(inverse, {target.value()});
                    NoiseSite site;
                    for (int p = 0; p < 3; ++p) {
                        double prob = node.args[static_cast<size_t>(p)];
                        if (prob > 0.0) {
                            site.channels.push_back(generators.channel(hir, {p + 1}, prob));
                        }
                    }
                    const double total_probability = sum_noise_probabilities(site);
                    append_noise_site(hir, std::move(site), total_probability, node_index);
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
                    generators.load(inverse, {q1, q2});
                    NoiseSite site;
                    size_t arg_idx = 0;
                    for (int p1 = 0; p1 <= 3; ++p1) {
                        for (int p2 = 0; p2 <= 3; ++p2) {
                            if (p1 == 0 && p2 == 0)
                                continue;
                            double prob = node.args[arg_idx];
                            if (prob > 0.0) {
                                site.channels.push_back(generators.channel(hir, {p1, p2}, prob));
                            }
                            ++arg_idx;
                        }
                    }
                    const double total_probability = sum_noise_probabilities(site);
                    append_noise_site(hir, std::move(site), total_probability, node_index);
                }
                break;
            }

            case GateType::PAULI_CHANNEL_3: {
                if (node.args.size() < 63) {
                    throw std::runtime_error("PAULI_CHANNEL_3 requires 63 arguments");
                }
                for (size_t i = 0; i + 2 < node.targets.size(); i += 3) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    uint32_t q3 = node.targets[i + 2].value();
                    generators.load(inverse, {q1, q2, q3});
                    NoiseSite site;
                    size_t arg_idx = 0;
                    for (int p1 = 0; p1 <= 3; ++p1) {
                        for (int p2 = 0; p2 <= 3; ++p2) {
                            for (int p3 = 0; p3 <= 3; ++p3) {
                                if (p1 == 0 && p2 == 0 && p3 == 0)
                                    continue;
                                double prob = node.args[arg_idx];
                                if (prob > 0.0) {
                                    site.channels.push_back(
                                        generators.channel(hir, {p1, p2, p3}, prob));
                                }
                                ++arg_idx;
                            }
                        }
                    }
                    const double total_probability = sum_noise_probabilities(site);
                    append_noise_site(hir, std::move(site), total_probability, node_index);
                }
                break;
            }

            case GateType::CORRELATED_ERROR: {
                NoiseSite site;
                double remaining = 1.0;
                size_t chain_end = node_index;
                while (true) {
                    const auto& link = circuit.nodes[chain_end];
                    if (link.args.empty()) {
                        throw std::runtime_error(
                            "CORRELATED_ERROR requires a probability argument");
                    }

                    double abs_prob = link.args[0] * remaining;
                    if (abs_prob > 0.0 && !link.targets.empty()) {
                        site.channels.push_back(
                            rewind_pauli_targets(hir, inverse, link.targets, abs_prob));
                    }
                    remaining *= 1.0 - link.args[0];

                    size_t next = chain_end + 1;
                    if (next >= circuit.nodes.size() ||
                        circuit.nodes[next].gate != GateType::ELSE_CORRELATED_ERROR) {
                        break;
                    }
                    chain_end = next;
                }
                const double total_probability = sum_noise_probabilities(site);
                append_noise_site(hir, std::move(site), total_probability, node_index);
                node_index = chain_end;
                break;
            }

            case GateType::ELSE_CORRELATED_ERROR:
                throw std::runtime_error(
                    "ELSE_CORRELATED_ERROR must follow CORRELATED_ERROR in the same chain");

            case GateType::DEPOLARIZE2: {
                double prob = node.args.empty() ? 0.0 : node.args[0];
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    NoiseSite site =
                        make_depolarize2_noise_site(hir, inverse, generators, q1, q2, prob);
                    append_noise_site(hir, std::move(site),
                                      represented_depolarizing_probability(prob, 15.0), node_index);
                }
                break;
            }

            case GateType::DEPOLARIZE3: {
                double prob = node.args.empty() ? 0.0 : node.args[0];
                for (size_t i = 0; i + 2 < node.targets.size(); i += 3) {
                    uint32_t q1 = node.targets[i].value();
                    uint32_t q2 = node.targets[i + 1].value();
                    uint32_t q3 = node.targets[i + 2].value();
                    NoiseSite site =
                        make_depolarize3_noise_site(hir, inverse, generators, q1, q2, q3, prob);
                    append_noise_site(hir, std::move(site),
                                      represented_depolarizing_probability(prob, 63.0), node_index);
                }
                break;
            }

            case GateType::READOUT_NOISE: {
                if (node.args.size() != 1 && node.args.size() != 2) {
                    throw std::invalid_argument(
                        "READOUT_NOISE requires one symmetric flip probability or two "
                        "conditional flip probabilities");
                }
                for (const double probability : node.args) {
                    if (!is_probability(probability)) {
                        throw std::invalid_argument(
                            "READOUT_NOISE probabilities must be finite and lie in [0, 1]");
                    }
                }
                for (const auto& target : node.targets) {
                    if (target.is_inverted()) {
                        throw std::runtime_error(
                            "READOUT_NOISE does not support inverted record targets; swap the "
                            "two flip probabilities instead");
                    }
                    uint32_t abs_meas_idx = target.value();
                    // One argument is a symmetric flip; a second argument
                    // splits it into (0->1, 1->0) conditioned on the bit.
                    double p01 = node.args.empty() ? 0.0 : node.args[0];
                    double p10 = node.args.size() > 1 ? node.args[1] : p01;
                    ReadoutNoiseIdx idx{static_cast<uint32_t>(hir.readout_noise.size())};
                    hir.readout_noise.push_back({.meas_idx = abs_meas_idx,
                                                 .prob_zero_to_one = p01,
                                                 .prob_one_to_zero = p10,
                                                 .source_node = static_cast<uint32_t>(node_index)});
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
                const PauliString rewound = inverse.apply(obs.view());
                assign_rewound(mask, rewound.view(), inversion_parity);
                hir.append_exp_val(exp_val_idx, mask);
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
    hir.final_tableau = inverse.inverse();

    return hir;
}

}  // namespace clifft
