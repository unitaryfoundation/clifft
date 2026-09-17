#pragma once

// A standalone external-Stim baseline. Per-history preprocessing constructs
// a Clifford program; none of this is installed into Clifft's hot executor.
#include "stim/simulators/tableau_simulator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pauli_proxy {
using Mask = __uint128_t;
using Simulator = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>;
using Pauli = stim::PauliString<stim::MAX_BITWORD_WIDTH>;
enum class Gate { R, H, CX, CZ, S, S_DAG, T, T_DAG, CCZ, M };
struct Operation {
    Gate gate;
    std::array<unsigned, 3> targets;
    unsigned arity, noise;
    int previous_flag;
};
struct Fault {
    unsigned boundary;
    Mask x, z;
};
struct Fixture {
    std::span<const Fault> faults;
    std::array<double, 4> expected;
};
struct Protocol {
    unsigned distance, width, ancilla;
    Mask logical_z;
    std::span<const Operation> operations, plus, minus;
    std::span<const Fixture> fixtures;
};
struct Command {
    stim::GateType gate;
    unsigned a, b, arity, desired, kind;
};

inline bool bit(Mask mask, unsigned q) {
    return (mask >> q) & 1;
}
inline unsigned parity(Mask mask) {
    return (std::popcount(static_cast<uint64_t>(mask)) +
            std::popcount(static_cast<uint64_t>(mask >> 64))) &
           1;
}

struct Compiler {
    Mask flips = 0;
    std::vector<Command> commands;

    void emit(stim::GateType gate, unsigned a, unsigned b = 0, unsigned arity = 1) {
        commands.push_back({gate, a, b, arity, 0, 0});
    }
    void fault(Mask x, Mask z, unsigned width) {
        flips ^= x;
        for (unsigned q = 0; q < width; ++q)
            if (bit(z, q))
                emit(stim::GateType::Z, q);
    }
    void append(const Operation& op) {
        using S = stim::GateType;
        auto [a, b, c] = op.targets;
        bool x = bit(flips, a);
        switch (op.gate) {
            case Gate::R:
                flips &= ~(Mask(1) << a);
                emit(S::R, a);
                break;
            case Gate::H:
                emit(S::H, a);
                if (x) {
                    emit(S::Z, a);
                    flips ^= Mask(1) << a;
                }
                break;
            case Gate::CX:
                if (x)
                    flips ^= Mask(1) << b;
                emit(S::CX, a, b, 2);
                break;
            case Gate::CZ:
                emit(S::CZ, a, b, 2);
                if (x)
                    emit(S::Z, b);
                if (bit(flips, b))
                    emit(S::Z, a);
                break;
            case Gate::S:
            case Gate::S_DAG:
                emit(op.gate == Gate::S ? S::S : S::S_DAG, a);
                if (x)
                    emit(S::Z, a);
                break;
            case Gate::T:
            case Gate::T_DAG:
                if (x)
                    emit(op.gate == Gate::T ? S::S_DAG : S::S, a);
                break;
            case Gate::CCZ:
                if (x)
                    emit(S::CZ, b, c, 2);
                if (bit(flips, b))
                    emit(S::CZ, a, c, 2);
                if (bit(flips, c))
                    emit(S::CZ, a, b, 2);
                if (x && bit(flips, b))
                    emit(S::Z, c);
                if (x && bit(flips, c))
                    emit(S::Z, b);
                if (bit(flips, b) && bit(flips, c))
                    emit(S::Z, a);
                break;
            case Gate::M:
                if (op.previous_flag == -1)
                    commands.push_back({S::M, a, 0, 1, static_cast<unsigned>(x), 1});
                else if (op.previous_flag >= 0) {
                    auto previous = static_cast<unsigned>(op.previous_flag);
                    commands.push_back(
                        {S::M, previous, a, 2, static_cast<unsigned>(x ^ bit(flips, previous)), 2});
                }
                break;
        }
    }
};

inline Compiler compile(const Protocol& plan, std::span<const Fault> faults) {
    Compiler result;
    result.commands.reserve(plan.operations.size() * 2);
    size_t f = 0;
    for (size_t j = 0; j <= plan.operations.size(); ++j) {
        while (f < faults.size() && faults[f].boundary == j) {
            result.fault(faults[f].x, faults[f].z, plan.width);
            ++f;
        }
        if (j < plan.operations.size())
            result.append(plan.operations[j]);
    }
    if (f != faults.size())
        throw std::invalid_argument("unordered or out-of-range fault boundary");
    return result;
}

inline double execute(const Compiler& program, Simulator& simulator, unsigned width) {
    double probability = 1;
    for (const auto& op : program.commands) {
        std::array<stim::GateTarget, 2> targets = {stim::GateTarget::qubit(op.a),
                                                   stim::GateTarget::qubit(op.b)};
        if (op.kind == 0) {
            simulator.do_gate({op.gate, {}, {targets.data(), targets.data() + op.arity}, ""});
        } else if (op.kind == 1) {
            int expectation = simulator.peek_z(op.a);
            probability *= (1 + (op.desired ? -1 : 1) * expectation) * .5;
            if (!probability)
                return 0;
            if (!expectation)
                simulator.postselect_z({targets.data(), targets.data() + 1}, op.desired);
        } else {
            Pauli observable(width);
            observable.zs[op.a] = observable.zs[op.b] = true;
            int expectation = simulator.peek_observable_expectation(observable);
            probability *= (1 + (op.desired ? -1 : 1) * expectation) * .5;
            if (!probability)
                return 0;
            if (!expectation)
                simulator.postselect_observable(observable, op.desired);
        }
    }
    return probability;
}

inline std::array<double, 4> evaluate(const Protocol& plan, const Compiler& program) {
    Simulator simulator(std::mt19937_64(0), plan.width);
    double probability = execute(program, simulator, plan.width);
    if (!probability)
        return {};
    Pauli z(plan.width);
    for (unsigned q = 0; q < plan.width; ++q)
        z.zs[q] = bit(plan.logical_z, q);
    double ze = simulator.peek_observable_expectation(z) *
                (parity(plan.logical_z & program.flips) ? -1 : 1);
    std::array<double, 2> probes;
    std::array<std::span<const Operation>, 2> operations{plan.plus, plan.minus};
    for (unsigned j = 0; j < 2; ++j) {
        Compiler probe;
        probe.flips = program.flips;
        probe.commands.reserve(operations[j].size() * 2);
        for (const auto& op : operations[j])
            probe.append(op);
        Simulator state(simulator, std::mt19937_64(0));
        execute(probe, state, plan.width);
        probes[j] = state.peek_z(plan.ancilla) * (bit(probe.flips, plan.ancilla) ? -1 : 1);
    }
    return {probability, probability * (probes[0] + probes[1]) * std::sqrt(.5),
            probability * (probes[0] - probes[1]) * std::sqrt(.5), probability * ze};
}

inline void sample(const Protocol& plan, std::mt19937_64& rng, double probability,
                   std::vector<Fault>& faults) {
    faults.clear();
    for (unsigned j = 0; j < plan.operations.size(); ++j) {
        const auto& op = plan.operations[j];
        if (!op.noise || (rng() >> 11) * 0x1.0p-53 >= probability)
            continue;
        Mask x = 0, z = 0;
        if (op.noise == 1)
            x = Mask(1) << op.targets[0];
        else {
            uint64_t n = (uint64_t(1) << (2 * op.arity)) - 1, threshold = -n % n, draw;
            do {
                draw = rng();
            } while (draw < threshold);
            draw = draw % n + 1;
            for (unsigned k = 0; k < op.arity; ++k) {
                unsigned p = draw & 3;
                draw >>= 2;
                if (p == 1 || p == 2)
                    x ^= Mask(1) << op.targets[k];
                if (p == 2 || p == 3)
                    z ^= Mask(1) << op.targets[k];
            }
        }
        faults.push_back({j + (op.gate != Gate::M), x, z});
    }
}

inline int benchmark(std::span<const Protocol> plans, int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: profile_proxy DISTANCE SHOTS_PER_TRIAL");
        size_t a, b;
        unsigned d = std::stoul(argv[1], &a), shots = std::stoul(argv[2], &b);
        if (a != std::string(argv[1]).size() || b != std::string(argv[2]).size() || !shots ||
            shots > 100000000)
            throw std::invalid_argument("invalid arguments");
        auto found =
            std::find_if(plans.begin(), plans.end(), [d](auto& p) { return p.distance == d; });
        if (found == plans.end())
            throw std::invalid_argument("unsupported distance");
        const auto& plan = *found;
        double error = 0;
        for (const auto& fixture : plan.fixtures) {
            auto result = evaluate(plan, compile(plan, fixture.faults));
            for (unsigned j = 0; j < 4; ++j)
                error = std::max(error, std::abs(result[j] - fixture.expected[j]));
        }
        if (!std::isfinite(error) || error > 2e-12)
            throw std::runtime_error("native proxy/reference mismatch");
        std::array<double, 5> trials{}, preprocessing{};
        std::array<double, 3> probes{};
        unsigned accepted = 0;
        std::mt19937_64 rng(19331);
        std::vector<Fault> faults;
        faults.reserve(plan.operations.size());
        using Clock = std::chrono::steady_clock;
        for (unsigned trial = 0; trial < 5; ++trial) {
            auto start = Clock::now();
            double construction_us = 0;
            for (unsigned shot = 0; shot < shots; ++shot) {
                sample(plan, rng, .001, faults);
                auto before = Clock::now();
                auto compiled = compile(plan, faults);
                construction_us +=
                    std::chrono::duration<double, std::micro>(Clock::now() - before).count();
                auto result = evaluate(plan, compiled);
                if ((rng() >> 11) * 0x1.0p-53 < result[0]) {
                    ++accepted;
                    for (unsigned j = 0; j < 3; ++j)
                        probes[j] += result[j + 1] / result[0];
                }
            }
            trials[trial] =
                std::chrono::duration<double, std::micro>(Clock::now() - start).count() / shots;
            preprocessing[trial] = construction_us / shots;
        }
        auto ordered = trials;
        std::sort(ordered.begin(), ordered.end());
        std::cout << std::setprecision(17) << "{\"distance\":" << d
                  << ",\"fixtures\":" << plan.fixtures.size() << ",\"max_error\":" << error
                  << ",\"sampled_attempt_median_us\":" << ordered[2]
                  << ",\"total_attempts\":" << 5ULL * shots << ",\"accepted\":" << accepted
                  << ",\"trials_us\":[";
        for (unsigned j = 0; j < 5; ++j)
            std::cout << (j ? "," : "") << trials[j];
        std::cout << "],\"preprocessing_trials_us\":[";
        for (unsigned j = 0; j < 5; ++j)
            std::cout << (j ? "," : "") << preprocessing[j];
        std::cout << "],\"accepted_mean_xyz\":[";
        for (unsigned j = 0; j < 3; ++j)
            std::cout << (j ? "," : "") << (accepted ? probes[j] / accepted : 0);
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
}  // namespace pauli_proxy
