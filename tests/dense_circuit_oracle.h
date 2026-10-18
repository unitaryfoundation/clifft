#pragma once

// Independent dense Schrodinger trajectory oracle for small circuits.
//
// Evaluates the exact joint distribution of a parsed clifft::Circuit directly
// from its AST using 2^n statevector linear algebra and explicit trajectory
// branching over quantum noise, projective measurements, resets, readout flips,
// classical feedback, EXP_VAL probes, and postselected detectors.
//
// Shares no code with clifft::trace, HirModule, plan_sampling, ExecutablePlan,
// or Executor, providing an independent reference for calibrating the HIR
// equivalence harness and verifying optimization passes.

#include "clifft/circuit/circuit.h"
#include "clifft/sampling/planner.h"

#include "sampling_equivalence_helpers.h"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft::test {

namespace detail_dense_oracle {

using StateVec = std::vector<std::complex<double>>;

struct PauliSpec {
    uint64_t x_mask = 0;
    uint64_t z_mask = 0;
    bool neg_sign = false;
};

// Computes (-1)^neg_sign * (tensor_q P_q) |psi> in the computational basis.
inline StateVec apply_pauli_spec(const StateVec& psi, uint32_t num_qubits, const PauliSpec& spec) {
    const size_t dim = size_t{1} << num_qubits;
    StateVec out(dim, {0.0, 0.0});
    const int y_count = std::popcount(spec.x_mask & spec.z_mask);
    std::complex<double> y_phase{1.0, 0.0};
    switch (y_count & 3) {
        case 1:
            y_phase = {0.0, 1.0};
            break;
        case 2:
            y_phase = {-1.0, 0.0};
            break;
        case 3:
            y_phase = {0.0, -1.0};
            break;
        default:
            break;
    }
    if (spec.neg_sign) {
        y_phase = -y_phase;
    }

    for (size_t idx = 0; idx < dim; ++idx) {
        const size_t dst = idx ^ static_cast<size_t>(spec.x_mask);
        const bool z_odd = (std::popcount(static_cast<uint64_t>(idx) & spec.z_mask) & 1) != 0;
        const std::complex<double> phase = z_odd ? -y_phase : y_phase;
        out[dst] = phase * psi[idx];
    }
    return out;
}

// Applies exp(-i * (alpha * pi / 2) * P) |psi> = cos(alpha*pi/2)|psi> - i*sin(alpha*pi/2) P|psi>.
inline void apply_pauli_rotation(StateVec& psi, uint32_t num_qubits, const PauliSpec& spec,
                                 double alpha) {
    const double half_theta = alpha * std::numbers::pi * 0.5;
    const double c = std::cos(half_theta);
    const double s = std::sin(half_theta);
    const StateVec p_psi = apply_pauli_spec(psi, num_qubits, spec);
    const std::complex<double> minus_i_s{0.0, -s};
    for (size_t i = 0; i < psi.size(); ++i) {
        psi[i] = c * psi[i] + minus_i_s * p_psi[i];
    }
}

inline PauliSpec build_target_pauli_spec(const std::vector<Target>& targets) {
    PauliSpec spec;
    for (const Target& t : targets) {
        const uint32_t q = t.value();
        spec.neg_sign ^= t.is_inverted();
        if (t.pauli() == Target::kPauliX) {
            spec.x_mask |= (uint64_t{1} << q);
        } else if (t.pauli() == Target::kPauliY) {
            spec.x_mask |= (uint64_t{1} << q);
            spec.z_mask |= (uint64_t{1} << q);
        } else {
            spec.z_mask |= (uint64_t{1} << q);
        }
    }
    return spec;
}

inline double expectation_value(const StateVec& psi, uint32_t num_qubits, const PauliSpec& spec) {
    const StateVec p_psi = apply_pauli_spec(psi, num_qubits, spec);
    double sum = 0.0;
    for (size_t i = 0; i < psi.size(); ++i) {
        sum += std::real(std::conj(psi[i]) * p_psi[i]);
    }
    return sum;
}

inline void apply_single_qubit_hadamard(StateVec& psi, uint32_t num_qubits, uint32_t q) {
    const uint64_t bit = uint64_t{1} << q;
    const StateVec x_psi = apply_pauli_spec(psi, num_qubits, PauliSpec{bit, 0, false});
    const StateVec z_psi = apply_pauli_spec(psi, num_qubits, PauliSpec{0, bit, false});
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (size_t i = 0; i < psi.size(); ++i) {
        psi[i] = (x_psi[i] + z_psi[i]) * inv_sqrt2;
    }
}

inline void apply_two_qubit_clifford_dense(StateVec& psi, GateType gate, uint32_t q1, uint32_t q2) {
    const size_t bit1 = size_t{1} << q1;
    const size_t bit2 = size_t{1} << q2;
    switch (gate) {
        case GateType::CX: {
            for (size_t idx = 0; idx < psi.size(); ++idx) {
                if ((idx & bit1) != 0 && (idx & bit2) == 0) {
                    std::swap(psi[idx], psi[idx | bit2]);
                }
            }
            break;
        }
        case GateType::CZ: {
            for (size_t idx = 0; idx < psi.size(); ++idx) {
                if ((idx & bit1) != 0 && (idx & bit2) != 0) {
                    psi[idx] = -psi[idx];
                }
            }
            break;
        }
        case GateType::CY: {
            for (size_t idx = 0; idx < psi.size(); ++idx) {
                if ((idx & bit1) != 0 && (idx & bit2) == 0) {
                    const std::complex<double> a0 = psi[idx];
                    const std::complex<double> a1 = psi[idx | bit2];
                    psi[idx] = std::complex<double>{0.0, -1.0} * a1;
                    psi[idx | bit2] = std::complex<double>{0.0, 1.0} * a0;
                }
            }
            break;
        }
        case GateType::SWAP: {
            for (size_t idx = 0; idx < psi.size(); ++idx) {
                if ((idx & bit1) != 0 && (idx & bit2) == 0) {
                    std::swap(psi[idx], (psi[idx ^ bit1 ^ bit2]));
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported two-qubit gate in dense oracle: " +
                                     std::string(gate_name(gate)));
    }
}

struct TrajectoryBranch {
    StateVec psi;
    double probability = 1.0;
    std::vector<uint8_t> records;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
    uint32_t next_meas = 0;
    uint32_t next_det = 0;
    uint32_t next_exp_val = 0;
};

// Projects `branch` onto the +1 (b=0) and -1 (b=1) eigenspaces of `spec`.
// If `record_result` is true, appends `b ^ invert_record` to `records`.
// If `reset_spec` is non-empty and physical outcome `b == 1`, applies `reset_spec`.
inline void branch_projective_measurement(const TrajectoryBranch& branch, uint32_t num_qubits,
                                          const PauliSpec& spec, bool record_result,
                                          bool invert_record, const PauliSpec* reset_spec,
                                          std::vector<TrajectoryBranch>& out) {
    const StateVec p_psi = apply_pauli_spec(branch.psi, num_qubits, spec);
    for (uint8_t b = 0; b < 2; ++b) {
        StateVec proj(branch.psi.size());
        double norm_sq = 0.0;
        const double sign = (b == 0) ? 1.0 : -1.0;
        for (size_t i = 0; i < proj.size(); ++i) {
            proj[i] = 0.5 * (branch.psi[i] + sign * p_psi[i]);
            norm_sq += std::norm(proj[i]);
        }
        if (norm_sq <= 1e-15) {
            continue;
        }
        const double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (auto& amp : proj) {
            amp *= inv_norm;
        }
        if (reset_spec != nullptr && b == 1) {
            proj = apply_pauli_spec(proj, num_qubits, *reset_spec);
        }

        TrajectoryBranch child = branch;
        child.psi = std::move(proj);
        child.probability *= norm_sq;
        if (record_result) {
            child.records[child.next_meas++] = static_cast<uint8_t>(b ^ (invert_record ? 1U : 0U));
        }
        out.push_back(std::move(child));
    }
}

}  // namespace detail_dense_oracle

// Evaluates the exact unconditional distribution of `circuit` by stepping a
// dense 2^n statevector through `circuit.nodes` without using trace() or HIR.
inline ExactPlanDistribution compute_dense_circuit_distribution(
    const Circuit& circuit, const clifft::sampling::SamplingPlanOptions& options = {}) {
    using namespace detail_dense_oracle;
    REQUIRE(circuit.num_qubits <= 6);
    REQUIRE(circuit.num_measurements <= 64);
    REQUIRE(circuit.num_detectors <= 64);
    REQUIRE(circuit.num_observables <= 64);

    TrajectoryBranch init;
    init.psi.assign(size_t{1} << circuit.num_qubits, {0.0, 0.0});
    init.psi[0] = {1.0, 0.0};
    init.records.assign(circuit.num_measurements, 0);
    init.detectors.assign(circuit.num_detectors, 0);
    init.observables.assign(circuit.num_observables, 0);
    init.exp_vals.assign(circuit.num_exp_vals, 0.0);

    std::vector<TrajectoryBranch> active{std::move(init)};
    std::vector<TrajectoryBranch> next_active;

    for (size_t node_idx = 0; node_idx < circuit.nodes.size(); ++node_idx) {
        const AstNode& node = circuit.nodes[node_idx];
        switch (node.gate) {
            case GateType::TICK:
            case GateType::I:
            case GateType::II:
            case GateType::I_ERROR:
            case GateType::II_ERROR:
                break;

            case GateType::H:
                for (const Target& t : node.targets) {
                    for (TrajectoryBranch& br : active) {
                        apply_single_qubit_hadamard(br.psi, circuit.num_qubits, t.value());
                    }
                }
                break;

            case GateType::X:
            case GateType::Y:
            case GateType::Z: {
                for (const Target& t : node.targets) {
                    const uint64_t bit = uint64_t{1} << t.value();
                    const PauliSpec spec{
                        .x_mask = (node.gate == GateType::X || node.gate == GateType::Y) ? bit : 0,
                        .z_mask = (node.gate == GateType::Z || node.gate == GateType::Y) ? bit : 0,
                        .neg_sign = false,
                    };
                    for (TrajectoryBranch& br : active) {
                        br.psi = apply_pauli_spec(br.psi, circuit.num_qubits, spec);
                    }
                }
                break;
            }

            case GateType::S:
            case GateType::S_DAG:
            case GateType::T:
            case GateType::T_DAG:
            case GateType::SQRT_X:
            case GateType::SQRT_X_DAG:
            case GateType::SQRT_Y:
            case GateType::SQRT_Y_DAG: {
                const double alpha = (node.gate == GateType::S)            ? 0.5
                                     : (node.gate == GateType::S_DAG)      ? -0.5
                                     : (node.gate == GateType::T)          ? 0.25
                                     : (node.gate == GateType::T_DAG)      ? -0.25
                                     : (node.gate == GateType::SQRT_X)     ? 0.5
                                     : (node.gate == GateType::SQRT_X_DAG) ? -0.5
                                     : (node.gate == GateType::SQRT_Y)     ? 0.5
                                                                           : -0.5;
                for (const Target& t : node.targets) {
                    const uint64_t bit = uint64_t{1} << t.value();
                    const bool is_x =
                        node.gate == GateType::SQRT_X || node.gate == GateType::SQRT_X_DAG;
                    const bool is_y =
                        node.gate == GateType::SQRT_Y || node.gate == GateType::SQRT_Y_DAG;
                    const PauliSpec spec{
                        .x_mask = (is_x || is_y) ? bit : 0,
                        .z_mask = (!is_x || is_y) ? bit : 0,
                        .neg_sign = false,
                    };
                    for (TrajectoryBranch& br : active) {
                        apply_pauli_rotation(br.psi, circuit.num_qubits, spec, alpha);
                    }
                }
                break;
            }

            case GateType::R_X:
            case GateType::R_Y:
            case GateType::R_Z: {
                const double alpha = node.args[0];
                for (const Target& t : node.targets) {
                    const uint64_t bit = uint64_t{1} << t.value();
                    const PauliSpec spec{
                        .x_mask =
                            (node.gate == GateType::R_X || node.gate == GateType::R_Y) ? bit : 0,
                        .z_mask =
                            (node.gate == GateType::R_Z || node.gate == GateType::R_Y) ? bit : 0,
                        .neg_sign = false,
                    };
                    for (TrajectoryBranch& br : active) {
                        apply_pauli_rotation(br.psi, circuit.num_qubits, spec, alpha);
                    }
                }
                break;
            }

            case GateType::R_XX:
            case GateType::R_YY:
            case GateType::R_ZZ: {
                const double alpha = node.args[0];
                for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                    const uint64_t bits = (uint64_t{1} << node.targets[i].value()) |
                                          (uint64_t{1} << node.targets[i + 1].value());
                    const PauliSpec spec{
                        .x_mask =
                            (node.gate == GateType::R_XX || node.gate == GateType::R_YY) ? bits : 0,
                        .z_mask =
                            (node.gate == GateType::R_ZZ || node.gate == GateType::R_YY) ? bits : 0,
                        .neg_sign = false,
                    };
                    for (TrajectoryBranch& br : active) {
                        apply_pauli_rotation(br.psi, circuit.num_qubits, spec, alpha);
                    }
                }
                break;
            }

            case GateType::SPP:
            case GateType::SPP_DAG:
            case GateType::TPP:
            case GateType::TPP_DAG:
            case GateType::R_PAULI: {
                const double alpha = (node.gate == GateType::SPP)       ? 0.5
                                     : (node.gate == GateType::SPP_DAG) ? -0.5
                                     : (node.gate == GateType::TPP)     ? 0.25
                                     : (node.gate == GateType::TPP_DAG) ? -0.25
                                                                        : node.args[0];
                PauliSpec spec = build_target_pauli_spec(node.targets);
                if (node.gate == GateType::R_PAULI) {
                    // R_PAULI ignores target inversion bang per frontend.cc.
                    spec.neg_sign = false;
                }
                for (TrajectoryBranch& br : active) {
                    apply_pauli_rotation(br.psi, circuit.num_qubits, spec, alpha);
                }
                break;
            }

            case GateType::CX:
            case GateType::CY:
            case GateType::CZ:
            case GateType::SWAP: {
                if (!node.targets.empty() && node.targets[0].is_rec()) {
                    for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                        const uint32_t rec_idx = node.targets[i].value();
                        const uint64_t q_bit = uint64_t{1} << node.targets[i + 1].value();
                        const PauliSpec spec{
                            .x_mask = (node.gate == GateType::CX) ? q_bit : 0,
                            .z_mask = (node.gate == GateType::CZ) ? q_bit : 0,
                            .neg_sign = false,
                        };
                        for (TrajectoryBranch& br : active) {
                            if (br.records[rec_idx] != 0) {
                                br.psi = apply_pauli_spec(br.psi, circuit.num_qubits, spec);
                            }
                        }
                    }
                } else {
                    for (size_t i = 0; i + 1 < node.targets.size(); i += 2) {
                        const uint32_t q1 = node.targets[i].value();
                        const uint32_t q2 = node.targets[i + 1].value();
                        for (TrajectoryBranch& br : active) {
                            apply_two_qubit_clifford_dense(br.psi, node.gate, q1, q2);
                        }
                    }
                }
                break;
            }

            case GateType::M:
            case GateType::MX:
            case GateType::MY:
            case GateType::MR:
            case GateType::MRX:
            case GateType::MRY:
            case GateType::R:
            case GateType::RX:
            case GateType::RY: {
                const bool record_result = !is_reset(node.gate);
                const bool do_reset = is_reset(node.gate) || is_measure_reset(node.gate);
                const bool is_x = node.gate == GateType::MX || node.gate == GateType::MRX ||
                                  node.gate == GateType::RX;
                const bool is_y = node.gate == GateType::MY || node.gate == GateType::MRY ||
                                  node.gate == GateType::RY;

                for (const Target& t : node.targets) {
                    const uint64_t bit = uint64_t{1} << t.value();
                    const PauliSpec meas_spec{
                        .x_mask = (is_x || is_y) ? bit : 0,
                        .z_mask = (!is_x || is_y) ? bit : 0,
                        .neg_sign = false,
                    };
                    const PauliSpec reset_spec{
                        .x_mask = (!is_x && !is_y) ? bit : 0,
                        .z_mask = (is_x || is_y) ? bit : 0,
                        .neg_sign = false,
                    };
                    next_active.clear();
                    for (const TrajectoryBranch& br : active) {
                        branch_projective_measurement(
                            br, circuit.num_qubits, meas_spec, record_result, t.is_inverted(),
                            do_reset ? &reset_spec : nullptr, next_active);
                    }
                    active.swap(next_active);
                }
                break;
            }

            case GateType::MPP: {
                const PauliSpec spec = build_target_pauli_spec(node.targets);
                const PauliSpec unsigned_spec{spec.x_mask, spec.z_mask, false};
                next_active.clear();
                for (const TrajectoryBranch& br : active) {
                    branch_projective_measurement(br, circuit.num_qubits, unsigned_spec, true,
                                                  spec.neg_sign, nullptr, next_active);
                }
                active.swap(next_active);
                break;
            }

            case GateType::X_ERROR:
            case GateType::Y_ERROR:
            case GateType::Z_ERROR:
            case GateType::DEPOLARIZE1:
            case GateType::PAULI_CHANNEL_1: {
                for (const Target& t : node.targets) {
                    const uint64_t bit = uint64_t{1} << t.value();
                    std::vector<std::pair<PauliSpec, double>> channels;
                    if (node.gate == GateType::X_ERROR) {
                        channels.push_back({{bit, 0, false}, node.args[0]});
                    } else if (node.gate == GateType::Y_ERROR) {
                        channels.push_back({{bit, bit, false}, node.args[0]});
                    } else if (node.gate == GateType::Z_ERROR) {
                        channels.push_back({{0, bit, false}, node.args[0]});
                    } else if (node.gate == GateType::DEPOLARIZE1) {
                        const double p = node.args[0] / 3.0;
                        channels.push_back({{bit, 0, false}, p});
                        channels.push_back({{bit, bit, false}, p});
                        channels.push_back({{0, bit, false}, p});
                    } else {
                        if (node.args[0] > 0.0) {
                            channels.push_back({{bit, 0, false}, node.args[0]});
                        }
                        if (node.args[1] > 0.0) {
                            channels.push_back({{bit, bit, false}, node.args[1]});
                        }
                        if (node.args[2] > 0.0) {
                            channels.push_back({{0, bit, false}, node.args[2]});
                        }
                    }

                    double total_p = 0.0;
                    for (const auto& ch : channels) {
                        total_p += ch.second;
                    }
                    next_active.clear();
                    for (const TrajectoryBranch& br : active) {
                        if (1.0 - total_p > 1e-15) {
                            TrajectoryBranch no_err = br;
                            no_err.probability *= (1.0 - total_p);
                            next_active.push_back(std::move(no_err));
                        }
                        for (const auto& [spec, prob] : channels) {
                            if (prob <= 1e-15) {
                                continue;
                            }
                            TrajectoryBranch err = br;
                            err.psi = apply_pauli_spec(br.psi, circuit.num_qubits, spec);
                            err.probability *= prob;
                            next_active.push_back(std::move(err));
                        }
                    }
                    active.swap(next_active);
                }
                break;
            }

            case GateType::CORRELATED_ERROR: {
                std::vector<std::pair<PauliSpec, double>> channels;
                double remaining = 1.0;
                size_t chain_end = node_idx;
                while (true) {
                    const AstNode& link = circuit.nodes[chain_end];
                    const double abs_prob = link.args[0] * remaining;
                    if (abs_prob > 0.0 && !link.targets.empty()) {
                        PauliSpec spec = build_target_pauli_spec(link.targets);
                        spec.neg_sign = false;
                        channels.push_back({spec, abs_prob});
                    }
                    remaining *= 1.0 - link.args[0];
                    if (chain_end + 1 >= circuit.nodes.size() ||
                        circuit.nodes[chain_end + 1].gate != GateType::ELSE_CORRELATED_ERROR) {
                        break;
                    }
                    ++chain_end;
                }
                node_idx = chain_end;

                double total_p = 0.0;
                for (const auto& ch : channels) {
                    total_p += ch.second;
                }
                next_active.clear();
                for (const TrajectoryBranch& br : active) {
                    if (1.0 - total_p > 1e-15) {
                        TrajectoryBranch no_err = br;
                        no_err.probability *= (1.0 - total_p);
                        next_active.push_back(std::move(no_err));
                    }
                    for (const auto& [spec, prob] : channels) {
                        if (prob <= 1e-15) {
                            continue;
                        }
                        TrajectoryBranch err = br;
                        err.psi = apply_pauli_spec(br.psi, circuit.num_qubits, spec);
                        err.probability *= prob;
                        next_active.push_back(std::move(err));
                    }
                }
                active.swap(next_active);
                break;
            }

            case GateType::READOUT_NOISE: {
                const double p01 = node.args.empty() ? 0.0 : node.args[0];
                const double p10 = node.args.size() > 1 ? node.args[1] : p01;
                for (const Target& t : node.targets) {
                    const uint32_t rec_idx = t.value();
                    next_active.clear();
                    for (const TrajectoryBranch& br : active) {
                        const uint8_t cur_bit = br.records[rec_idx];
                        const double p_flip = (cur_bit == 0) ? p01 : p10;
                        if (1.0 - p_flip > 1e-15) {
                            TrajectoryBranch keep = br;
                            keep.probability *= (1.0 - p_flip);
                            next_active.push_back(std::move(keep));
                        }
                        if (p_flip > 1e-15) {
                            TrajectoryBranch flip = br;
                            flip.records[rec_idx] ^= 1U;
                            flip.probability *= p_flip;
                            next_active.push_back(std::move(flip));
                        }
                    }
                    active.swap(next_active);
                }
                break;
            }

            case GateType::DETECTOR: {
                for (TrajectoryBranch& br : active) {
                    const uint32_t det_idx = br.next_det++;
                    uint8_t parity =
                        (det_idx < options.expected_detectors.size())
                            ? static_cast<uint8_t>(options.expected_detectors[det_idx] & 1U)
                            : 0U;
                    for (const Target& t : node.targets) {
                        parity ^= br.records[t.value()];
                    }
                    br.detectors[det_idx] = parity;
                }
                break;
            }

            case GateType::OBSERVABLE_INCLUDE: {
                const auto obs_idx = static_cast<uint32_t>(node.args.empty() ? 0.0 : node.args[0]);
                for (TrajectoryBranch& br : active) {
                    uint8_t parity = br.observables[obs_idx];
                    for (const Target& t : node.targets) {
                        parity ^= br.records[t.value()];
                    }
                    br.observables[obs_idx] = parity;
                }
                break;
            }

            case GateType::EXP_VAL: {
                const PauliSpec spec = build_target_pauli_spec(node.targets);
                for (TrajectoryBranch& br : active) {
                    br.exp_vals[br.next_exp_val++] =
                        expectation_value(br.psi, circuit.num_qubits, spec);
                }
                break;
            }

            default:
                throw std::runtime_error("Unsupported gate in dense circuit oracle: " +
                                         std::string(gate_name(node.gate)));
        }
    }

    ExactPlanDistribution dist;
    dist.num_visible_records = circuit.num_measurements;
    dist.num_detectors = circuit.num_detectors;
    dist.num_observables = circuit.num_observables;
    dist.num_exp_vals = circuit.num_exp_vals;
    dist.kept_exp_vals.assign(circuit.num_exp_vals, 0.0);

    uint64_t postselected_mask = 0;
    for (size_t d = 0; d < options.postselection_mask.size() && d < circuit.num_detectors; ++d) {
        if (options.postselection_mask[d] != 0) {
            postselected_mask |= (uint64_t{1} << d);
        }
    }

    double total_prob = 0.0;
    for (const TrajectoryBranch& br : active) {
        total_prob += br.probability;
        uint64_t vis_bits = 0;
        for (uint32_t m = 0; m < circuit.num_measurements; ++m) {
            vis_bits |= static_cast<uint64_t>(br.records[m] & 1U) << m;
        }
        uint64_t det_bits = 0;
        for (uint32_t d = 0; d < circuit.num_detectors; ++d) {
            det_bits |= static_cast<uint64_t>(br.detectors[d] & 1U) << d;
        }
        uint64_t obs_bits = 0;
        for (uint32_t o = 0; o < circuit.num_observables; ++o) {
            const uint8_t ref = (o < options.expected_observables.size())
                                    ? static_cast<uint8_t>(options.expected_observables[o] & 1U)
                                    : 0U;
            obs_bits |= static_cast<uint64_t>((br.observables[o] ^ ref) & 1U) << o;
        }

        if ((det_bits & postselected_mask) != 0) {
            dist.discarded_probability += br.probability;
        } else {
            dist.kept_outcomes.push_back(ExactOutcomeEntry{
                ExactShotOutcome{vis_bits, det_bits, obs_bits, br.exp_vals}, br.probability});
            for (uint32_t e = 0; e < circuit.num_exp_vals; ++e) {
                dist.kept_exp_vals[e] += br.probability * br.exp_vals[e];
            }
        }
    }

    REQUIRE_THAT(total_prob, Catch::Matchers::WithinAbs(1.0, 1e-9));
    canonicalize_outcome_entries(dist.kept_outcomes);
    return dist;
}

}  // namespace clifft::test
