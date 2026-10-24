#pragma once

// Random circuits in Clifft's syntax and their exact record distributions,
// shared by the ZX tests as an oracle independent of HIR and of ZX.

#include "clifft/frontend/hir.h"

#include "zx_test_helpers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace zx_test {

// ---------------------------------------------------------------------------
// Circuits in Clifft's syntax, kept as op lists for the reference.

struct Op {
    std::string name;
    std::vector<int> targets;  // qubits; feedback {lookback, qubit}; readout {lookback}
    std::string paulis = {};   // MPP, R_PAULI, E and ELSE letters, parallel to targets
    std::vector<double> args = {};
    bool inverted = false;  // M !q
};

inline std::string to_text(const std::vector<Op>& ops) {
    std::ostringstream out;
    out.precision(17);
    for (const Op& op : ops) {
        if (op.name == "FEEDBACK_CX" || op.name == "FEEDBACK_CZ") {
            out << (op.name == "FEEDBACK_CX" ? "CX" : "CZ") << " rec[" << op.targets[0] << "] "
                << op.targets[1] << "\n";
            continue;
        }
        out << op.name;
        if (!op.args.empty()) {
            out << "(";
            for (size_t i = 0; i < op.args.size(); ++i) {
                out << (i ? "," : "") << op.args[i];
            }
            out << ")";
        }
        if (op.name == "READOUT_NOISE") {
            out << " rec[" << op.targets[0] << "]\n";
            continue;
        }
        if (op.name == "MPP" || op.name == "R_PAULI") {
            out << " ";
            for (size_t i = 0; i < op.targets.size(); ++i) {
                out << (i ? "*" : "") << op.paulis[i] << op.targets[i];
            }
        } else if (op.name == "E" || op.name == "ELSE_CORRELATED_ERROR") {
            for (size_t i = 0; i < op.targets.size(); ++i) {
                out << " " << op.paulis[i] << op.targets[i];
            }
        } else {
            for (int t : op.targets) {
                out << " " << (op.inverted ? "!" : "") << t;
            }
        }
        out << "\n";
    }
    return out.str();
}

// Three decimals, so the circuit text carries the reference's exact value.
inline double three_decimals(double x) {
    return std::round(x * 1000.0) / 1000.0;
}

inline std::vector<Op> random_ops(TestRng& rng, int num_qubits, int depth, bool noise_heavy) {
    auto qubit = [&] { return static_cast<int>(rng.below(static_cast<uint32_t>(num_qubits))); };
    auto distinct = [&](size_t k) {
        std::vector<int> qs;
        while (qs.size() < k) {
            const int q = qubit();
            if (std::find(qs.begin(), qs.end(), q) == qs.end()) {
                qs.push_back(q);
            }
        }
        return qs;
    };
    auto letters = [&](size_t k) {
        std::string s;
        for (size_t i = 0; i < k; ++i) {
            s += "XYZ"[rng.below(3)];
        }
        return s;
    };
    const char* singles[] = {"H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z", "SQRT_X"};
    const char* paulis_noise[] = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"};
    std::vector<Op> ops;
    int records = 0;
    // The evaluator enumerates every combination of noise outcomes, so keep
    // their product bounded; a noise op over budget is skipped.
    double combinations = 1.0;
    auto afford = [&](double outcomes) {
        if (combinations * outcomes > 4096.0) {
            return false;
        }
        combinations *= outcomes;
        return true;
    };
    auto measure = [&](const std::string& name) {
        Op op{.name = name, .targets = {qubit()}, .inverted = rng.uniform() < 0.2};
        if (rng.uniform() < 0.25 && afford(2)) {
            op.args = {0.2};
        }
        ops.push_back(op);
        ++records;
        // Feedback right after a measurement is a reset when it anticommutes
        // with the measured Pauli and a look-alike when it commutes.
        if (rng.uniform() < 0.3) {
            ops.push_back(
                {.name = rng.coin() ? "FEEDBACK_CX" : "FEEDBACK_CZ", .targets = {-1, qubit()}});
        }
    };
    for (int i = 0; i < depth; ++i) {
        const double r = rng.uniform();
        if (noise_heavy) {
            // Noise between basis changes, measured at the end: noise bugs show.
            if (r < 0.45) {
                ops.push_back({.name = singles[rng.below(9)], .targets = {qubit()}});
            } else if (r < 0.6) {
                ops.push_back({.name = rng.coin() ? "CX" : "CZ", .targets = distinct(2)});
            } else if (r < 0.75) {
                const uint32_t kind = rng.below(4);
                if (afford(kind == 3 ? 4 : 2)) {
                    ops.push_back(
                        {.name = paulis_noise[kind], .targets = {qubit()}, .args = {0.3}});
                }
            } else if (r < 0.85) {
                if (afford(16)) {
                    ops.push_back({.name = "DEPOLARIZE2", .targets = distinct(2), .args = {0.3}});
                }
            } else if (afford(3)) {
                const auto qs = distinct(1 + rng.below(2));
                ops.push_back(
                    {.name = "E", .targets = qs, .paulis = letters(qs.size()), .args = {0.3}});
                if (rng.coin()) {
                    ops.push_back({.name = "ELSE_CORRELATED_ERROR",
                                   .targets = distinct(1),
                                   .paulis = letters(1),
                                   .args = {0.4}});
                }
            }
            continue;
        }
        if (r < 0.26) {
            ops.push_back({.name = singles[rng.below(9)], .targets = {qubit()}});
        } else if (r < 0.3) {
            ops.push_back({.name = "R_Z",
                           .targets = {qubit()},
                           .args = {three_decimals(0.05 + 0.3 * rng.uniform())}});
        } else if (r < 0.34) {
            const auto qs = distinct(1 + rng.below(2));
            ops.push_back({.name = "R_PAULI",
                           .targets = qs,
                           .paulis = letters(qs.size()),
                           .args = {three_decimals(0.4 * rng.uniform())}});
        } else if (r < 0.46) {
            ops.push_back({.name = rng.coin() ? "CX" : "CZ", .targets = distinct(2)});
        } else if (r < 0.54) {
            const uint32_t kind = rng.below(4);
            if (afford(kind == 3 ? 4 : 2)) {
                ops.push_back({.name = paulis_noise[kind], .targets = {qubit()}, .args = {0.15}});
            }
        } else if (r < 0.57) {
            if (afford(16)) {
                ops.push_back({.name = "DEPOLARIZE2", .targets = distinct(2), .args = {0.2}});
            }
        } else if (r < 0.67) {
            const char* kinds[] = {"M", "MX", "MY"};
            measure(kinds[rng.below(3)]);
        } else if (r < 0.75) {
            const char* kinds[] = {"R", "RX", "RY", "MR", "MRX", "MRY"};
            const std::string name = kinds[rng.below(6)];
            if (name[0] == 'M') {
                measure(name);
            } else {
                ops.push_back({.name = name, .targets = {qubit()}});
            }
        } else if (r < 0.81) {
            ops.push_back({.name = "MPP", .targets = distinct(2), .paulis = letters(2)});
            ++records;
        } else if (r < 0.83) {
            ops.push_back({.name = "MPAD", .targets = {static_cast<int>(rng.below(2))}});
            ++records;
        } else if (records > 0 && r < 0.92) {
            const int back = -1 - static_cast<int>(rng.below(static_cast<uint32_t>(records)));
            ops.push_back(
                {.name = rng.coin() ? "FEEDBACK_CX" : "FEEDBACK_CZ", .targets = {back, qubit()}});
        } else if (records > 0 && afford(2)) {
            const int back = -1 - static_cast<int>(rng.below(static_cast<uint32_t>(records)));
            Op op{.name = "READOUT_NOISE", .targets = {back}, .args = {0.25}};
            if (rng.uniform() < 0.3) {
                op.args = {0.1, 0.35};
            }
            ops.push_back(op);
        }
    }
    if (noise_heavy) {
        const char* kinds[] = {"M", "MX", "MY"};
        for (int q = 0; q < num_qubits; ++q) {
            ops.push_back({.name = kinds[rng.below(3)], .targets = {q}});
        }
    }
    return ops;
}

// ---------------------------------------------------------------------------
// Exact record distribution of a circuit, by enumerating every measurement
// branch and noise outcome on a dense state vector.

using State = std::vector<std::complex<double>>;
using Matrix = std::array<std::complex<double>, 4>;
using Distribution = std::map<uint64_t, double>;

class Reference {
  public:
    explicit Reference(int num_qubits) : n_(num_qubits) {}

    Distribution run(const std::vector<Op>& ops) {
        State psi(size_t{1} << n_, 0.0);
        psi[0] = 1.0;
        dist_.clear();
        branch(ops, 0, std::move(psi), 1.0, 0, 0, false);
        return dist_;
    }

  private:
    using C = std::complex<double>;

    void apply1(State& psi, const Matrix& m, int q) const {
        const size_t bit = size_t{1} << q;
        for (size_t i = 0; i < psi.size(); ++i) {
            if (i & bit) {
                continue;
            }
            const auto a = psi[i], b = psi[i | bit];
            psi[i] = m[0] * a + m[1] * b;
            psi[i | bit] = m[2] * a + m[3] * b;
        }
    }

    static Matrix pauli(char p) {
        if (p == 'X') {
            return {C(0), C(1), C(1), C(0)};
        }
        if (p == 'Y') {
            return {C(0), C(0, -1), C(0, 1), C(0)};
        }
        return {C(1), C(0), C(0), C(-1)};
    }

    State product(const State& psi, const std::vector<int>& qs, const std::string& ps) const {
        State out = psi;
        for (size_t k = 0; k < qs.size(); ++k) {
            apply1(out, pauli(ps[k]), qs[k]);
        }
        return out;
    }

    // (1 + (-1)^outcome P) / 2 applied to psi.
    State project(const State& psi, const std::vector<int>& qs, const std::string& ps,
                  int outcome) const {
        const State p_psi = product(psi, qs, ps);
        State out(psi.size());
        const double s = outcome ? -1.0 : 1.0;
        for (size_t i = 0; i < psi.size(); ++i) {
            out[i] = (psi[i] + s * p_psi[i]) / 2.0;
        }
        return out;
    }

    // exp(-i pi alpha P / 2) = cos(pi alpha / 2) - i sin(pi alpha / 2) P.
    void rotate(State& psi, const std::vector<int>& qs, const std::string& ps, double alpha) const {
        const State p_psi = product(psi, qs, ps);
        const double h = std::numbers::pi * alpha / 2;
        for (size_t i = 0; i < psi.size(); ++i) {
            psi[i] = std::cos(h) * psi[i] - C(0, std::sin(h)) * p_psi[i];
        }
    }

    static bool normalize(State& psi, double& p) {
        p = 0.0;
        for (const auto& a : psi) {
            p += std::norm(a);
        }
        if (p < 1e-15) {
            return false;
        }
        for (auto& a : psi) {
            a /= std::sqrt(p);
        }
        return true;
    }

    // `chain_fired` tracks an E / ELSE_CORRELATED_ERROR chain in progress.
    void branch(const std::vector<Op>& ops, size_t at, State psi, double prob, uint64_t recs,
                int num_recs, bool chain_fired) {
        if (prob < 1e-15) {
            return;
        }
        const double s2 = 1.0 / std::sqrt(2.0);
        for (; at < ops.size(); ++at) {
            const Op& op = ops[at];
            const int q = op.targets.empty() ? 0 : op.targets[0];
            if (op.name != "ELSE_CORRELATED_ERROR") {
                chain_fired = false;
            }
            if (op.name == "H") {
                apply1(psi, {C(s2), C(s2), C(s2), C(-s2)}, q);
            } else if (op.name == "S" || op.name == "S_DAG") {
                apply1(psi, {C(1), C(0), C(0), C(0, op.name == "S" ? 1 : -1)}, q);
            } else if (op.name == "T" || op.name == "T_DAG") {
                const double angle = (op.name == "T" ? 1 : -1) * std::numbers::pi / 4;
                apply1(psi, {C(1), C(0), C(0), std::polar(1.0, angle)}, q);
            } else if (op.name == "X" || op.name == "Y" || op.name == "Z") {
                apply1(psi, pauli(op.name[0]), q);
            } else if (op.name == "SQRT_X") {
                apply1(psi, {C(0.5, 0.5), C(0.5, -0.5), C(0.5, -0.5), C(0.5, 0.5)}, q);
            } else if (op.name == "R_Z") {
                rotate(psi, {q}, "Z", op.args[0]);
            } else if (op.name == "R_PAULI") {
                rotate(psi, op.targets, op.paulis, op.args[0]);
            } else if (op.name == "CX" || op.name == "CZ") {
                const size_t c = size_t{1} << op.targets[0], t = size_t{1} << op.targets[1];
                for (size_t i = 0; i < psi.size(); ++i) {
                    if (!(i & c)) {
                        continue;
                    }
                    if (op.name == "CZ") {
                        if (i & t) {
                            psi[i] = -psi[i];
                        }
                    } else if (!(i & t)) {
                        std::swap(psi[i], psi[i | t]);
                    }
                }
            } else if (op.name == "X_ERROR" || op.name == "Y_ERROR" || op.name == "Z_ERROR" ||
                       op.name == "DEPOLARIZE1") {
                const std::string kinds = op.name == "DEPOLARIZE1" ? "XYZ" : op.name.substr(0, 1);
                const double p = op.args[0] / static_cast<double>(kinds.size());
                for (char l : kinds) {
                    State hit = psi;
                    apply1(hit, pauli(l), q);
                    branch(ops, at + 1, std::move(hit), prob * p, recs, num_recs, false);
                }
                prob *= 1.0 - op.args[0];
            } else if (op.name == "DEPOLARIZE2") {
                for (int k = 1; k < 16; ++k) {
                    State hit = psi;
                    if (k & 3) {
                        apply1(hit, pauli(" XYZ"[k & 3]), op.targets[0]);
                    }
                    if (k >> 2) {
                        apply1(hit, pauli(" XYZ"[k >> 2]), op.targets[1]);
                    }
                    branch(ops, at + 1, std::move(hit), prob * op.args[0] / 15, recs, num_recs,
                           false);
                }
                prob *= 1.0 - op.args[0];
            } else if (op.name == "E" || op.name == "ELSE_CORRELATED_ERROR") {
                if (chain_fired) {
                    continue;
                }
                branch(ops, at + 1, product(psi, op.targets, op.paulis), prob * op.args[0], recs,
                       num_recs, true);
                prob *= 1.0 - op.args[0];
            } else if (op.name == "M" || op.name == "MX" || op.name == "MY" || op.name == "MR" ||
                       op.name == "MRX" || op.name == "MRY" || op.name == "MPP") {
                std::string ps = op.paulis;
                if (op.name != "MPP") {
                    const char last = op.name.back();
                    ps = std::string(1, last == 'X' || last == 'Y' ? last : 'Z');
                }
                const bool resets = op.name.size() >= 2 && op.name[1] == 'R';
                for (int outcome = 0; outcome < 2; ++outcome) {
                    State post = project(psi, op.targets, ps, outcome);
                    double p = 0.0;
                    if (!normalize(post, p)) {
                        continue;
                    }
                    if (resets && outcome) {
                        apply1(post, pauli(ps[0] == 'Z' ? 'X' : 'Z'), q);
                    }
                    const int recorded = outcome ^ static_cast<int>(op.inverted);
                    const uint64_t with = recs | (uint64_t(recorded) << num_recs);
                    const double flip = op.args.empty() ? 0.0 : op.args[0];
                    branch(ops, at + 1, post, prob * p * (1 - flip), with, num_recs + 1, false);
                    if (flip > 0) {
                        branch(ops, at + 1, std::move(post), prob * p * flip,
                               with ^ (uint64_t{1} << num_recs), num_recs + 1, false);
                    }
                }
                return;
            } else if (op.name == "MPAD") {
                recs |= uint64_t(op.targets[0]) << num_recs;
                ++num_recs;
            } else if (op.name == "R" || op.name == "RX" || op.name == "RY") {
                const char basis = op.name == "R" ? 'Z' : op.name[1];
                for (int outcome = 0; outcome < 2; ++outcome) {
                    State post = project(psi, {q}, std::string(1, basis), outcome);
                    double p = 0.0;
                    if (!normalize(post, p)) {
                        continue;
                    }
                    if (outcome) {
                        apply1(post, pauli(basis == 'Z' ? 'X' : 'Z'), q);
                    }
                    branch(ops, at + 1, std::move(post), prob * p, recs, num_recs, false);
                }
                return;
            } else if (op.name == "FEEDBACK_CX" || op.name == "FEEDBACK_CZ") {
                if ((recs >> (num_recs + op.targets[0])) & 1) {
                    apply1(psi, pauli(op.name == "FEEDBACK_CX" ? 'X' : 'Z'), op.targets[1]);
                }
            } else if (op.name == "READOUT_NOISE") {
                const int index = num_recs + op.targets[0];
                const bool bit = ((recs >> index) & 1) != 0;
                const double flip = op.args.size() == 2 && bit ? op.args[1] : op.args[0];
                branch(ops, at + 1, psi, prob * flip, recs ^ (uint64_t{1} << index), num_recs,
                       false);
                prob *= 1.0 - flip;
            } else {
                FAIL("reference does not know " << op.name);
            }
        }
        dist_[recs] += prob;
    }

    int n_;
    Distribution dist_;
};

// A Pauli on up to 64 qubits as bit masks: X where x is set, Z where z is,
// Y where both.
struct PauliMasks {
    uint64_t x = 0;
    uint64_t z = 0;
    bool sign = false;
};

// P|i> = (-1)^sign i^{#Y} (-1)^{z . i} |i ^ x>, with Y = i X Z.
inline State apply_pauli(const State& psi, const PauliMasks& m) {
    State out(psi.size());
    const std::complex<double> phase =
        std::pow(std::complex<double>(0, 1), std::popcount(m.x & m.z)) * (m.sign ? -1.0 : 1.0);
    for (uint64_t i = 0; i < psi.size(); ++i) {
        const double sign = (std::popcount(m.z & i) & 1) ? -1.0 : 1.0;
        out[i ^ m.x] += phase * sign * psi[i];
    }
    return out;
}

// The distribution with every classical readout flip applied.
inline Distribution apply_classical_readout(Distribution dist, const clifft::HirModule& hir,
                                            const std::vector<uint32_t>& entries) {
    for (uint32_t k : entries) {
        const clifft::ReadoutNoiseEntry& e = hir.readout_noise[k];
        Distribution next;
        for (const auto& [recs, p] : dist) {
            const bool bit = ((recs >> e.meas_idx) & 1) != 0;
            const double flip = bit ? e.prob_one_to_zero : e.prob_zero_to_one;
            next[recs] += p * (1 - flip);
            next[recs ^ (uint64_t{1} << e.meas_idx)] += p * flip;
        }
        dist = std::move(next);
    }
    return dist;
}

}  // namespace zx_test
