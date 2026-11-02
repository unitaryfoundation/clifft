#pragma once

#include "clifft/noncomp/online.h"
#include "clifft/noncomp/online_state.h"
#include "clifft/noncomp/trajectory_driver.h"

#include "noncomp_replay_hooks.h"
#include "sampling_equivalence_helpers.h"

#include <array>
#include <map>
#include <numeric>
#include <tuple>

namespace clifft::test {

// A history records no-fire as zero and a destination Level as one plus its
// enum value. Source outcomes and internal reset records are hidden. Statuses,
// visible records, detectors and observables are retained in the comparison.
struct PhysicalHistory {
    std::vector<unsigned> events;
    std::vector<QubitStatus> status;
    std::vector<uint8_t> records;
    uint64_t detectors = 0, observables = 0;
    auto operator<=>(const PhysicalHistory&) const = default;
};

struct WeightedPhysicalState {
    double weight = 0;
    std::vector<double> weighted_paulis;
};

using PhysicalEnsemble = std::map<PhysicalHistory, WeightedPhysicalState>;

inline uint64_t pack_bits(std::span<const uint8_t> bits) {
    uint64_t result = 0;
    for (size_t i = 0; i < bits.size(); ++i)
        result |= uint64_t{bits[i]} << i;
    return result;
}

// All 4^n Pauli moments determine the reduced density operator. Moments with
// nonidentity support on a vacated carrier are erased, equivalent to tracing
// it out and padding it with I/2. Its definite physical level stays in status.
inline void trace_vacated_factors(std::vector<double>& paulis,
                                  std::span<const QubitStatus> status) {
    for (size_t code = 0; code < paulis.size(); ++code)
        for (size_t q = 0; q < status.size(); ++q)
            if (!is_computational(status[q]) && ((code >> (2 * q)) & 3))
                paulis[code] = 0;
}

inline void add_physical_branch(PhysicalEnsemble& ensemble, PhysicalHistory history, double weight,
                                std::vector<double> paulis) {
    if (weight == 0)
        return;
    trace_vacated_factors(paulis, history.status);
    auto& entry = ensemble[std::move(history)];
    if (entry.weighted_paulis.empty())
        entry.weighted_paulis.resize(paulis.size());
    entry.weight += weight;
    for (size_t p = 0; p < paulis.size(); ++p)
        entry.weighted_paulis[p] += weight * paulis[p];
}

inline ExactPlanDistribution physical_distribution(const PhysicalEnsemble& ensemble,
                                                   const Circuit& circuit, unsigned instruments) {
    ExactPlanDistribution result;
    result.num_visible_records =
        3 * instruments + 2 * circuit.num_qubits + circuit.num_measurements;
    REQUIRE(result.num_visible_records <= 64);
    result.num_detectors = circuit.num_detectors;
    result.num_observables = circuit.num_observables;
    result.num_exp_vals = 1U << (2 * circuit.num_qubits);
    result.kept_exp_vals.resize(result.num_exp_vals);
    for (const auto& [history, state] : ensemble) {
        REQUIRE(history.events.size() == instruments);
        uint64_t key = pack_bits(history.records);
        for (auto status : history.status)
            key = (key << 2) | static_cast<unsigned>(status);
        for (auto event : history.events)
            key = (key << 3) | event;
        ExactShotOutcome outcome{key, history.detectors, history.observables, {}};
        for (size_t p = 0; p < state.weighted_paulis.size(); ++p) {
            outcome.exp_vals.push_back(state.weighted_paulis[p] / state.weight);
            result.kept_exp_vals[p] += state.weighted_paulis[p];
        }
        result.kept_outcomes.push_back({std::move(outcome), state.weight});
    }
    canonicalize_outcome_entries(result.kept_outcomes);
    // No positive branch is pruned. The shared rare-probability comparator
    // checks relative weights and normalized moments even for tiny histories.
    result.omitted_probability_bound = 0;
    return result;
}

class WeightedReplay final : public NoncompReplayHooks {
  public:
    static constexpr size_t capacity = 64;
    struct Decision {
        ReplayChoice kind{};
        unsigned selected = 0, count = 0;
        std::array<double, 5> masses{};
    };
    std::array<unsigned, capacity> tape{};
    std::array<Decision, capacity> decisions{};
    size_t supplied = 0, used = 0, event_count = 0;
    std::array<unsigned, capacity> events{};
    double weight = 1;
    bool invalid = false;
    std::vector<double> paulis;
    std::vector<QubitStatus> status;

    unsigned choose(ReplayChoice kind, std::span<const double> masses) noexcept override {
        // Invalid requests fail after dispatch, preserving the hot-loop contract.
        double total = 0;
        unsigned fallback = 0;
        for (unsigned i = 0; i < masses.size(); ++i) {
            invalid |= !is_finite_non_negative(masses[i]);
            total += masses[i];
            if (masses[i] > 0)
                fallback = i;
        }
        if (used == capacity || masses.size() > 5 || !(total > 0)) {
            invalid = true;
            return fallback;
        }
        unsigned selected = used < supplied ? tape[used] : fallback;
        if (selected >= masses.size() || !(masses[selected] > 0)) {
            invalid = true;
            selected = fallback;
        }
        auto& decision = decisions[used++];
        decision.kind = kind;
        decision.selected = selected;
        decision.count = static_cast<unsigned>(masses.size());
        std::copy(masses.begin(), masses.end(), decision.masses.begin());
        const double next = weight * (masses[selected] / total);
        invalid |= !(next > 0) || !is_finite_non_negative(next);
        weight = next;
        if (kind == ReplayChoice::Source) {
            events[event_count++] = selected == 2 ? 0 : 99;
        } else if (kind == ReplayChoice::Destination ||
                   (kind == ReplayChoice::DestinationPartition && selected < 2)) {
            if (event_count == 0)
                invalid = true;
            else
                events[event_count - 1] = selected + 1;
        }
        return selected;
    }

    void append_probes(Circuit& circuit) override {
        REQUIRE(circuit.num_qubits <= 3);
        REQUIRE(circuit.num_exp_vals == 0);
        for (unsigned code = 1; code < (1U << (2 * circuit.num_qubits)); ++code) {
            AstNode node{GateType::EXP_VAL, {}, {}};
            for (unsigned q = 0; q < circuit.num_qubits; ++q) {
                const unsigned p = (code >> (2 * q)) & 3;
                if (p)
                    node.targets.push_back(Target::pauli(q, p << Target::kPauliShift));
            }
            circuit.nodes.push_back(std::move(node));
            ++circuit.num_exp_vals;
        }
    }

    void inspect(std::span<const double> probes, std::span<const QubitStatus> levels) override {
        paulis.assign(1, 1);
        paulis.insert(paulis.end(), probes.begin(), probes.end());
        status.assign(levels.begin(), levels.end());
    }

    void inspect_online(const noncomp_detail::OnlineState& quantum,
                        std::span<const QubitStatus> levels) override {
        const unsigned n = quantum.frame.n;
        PauliString axis(n), factor(n);
        paulis.assign(1U << (2 * n), 0);
        for (unsigned code = 0; code < paulis.size(); ++code) {
            axis.mut_x().zero_out();
            axis.mut_z().zero_out();
            axis.set_phase(0);
            for (unsigned q = 0; q < n; ++q) {
                const unsigned p = (code >> (2 * q)) & 3;
                if (p) {
                    quantum.frame.axis(q, "IXYZ"[p], factor);
                    axis.right_multiply(factor.view());
                }
            }
            const auto populations = quantum.populations(axis.view());
            paulis[code] = (populations.zero - populations.one) / populations.total();
        }
        status.assign(levels.begin(), levels.end());
    }
};

// The first corpus deliberately excludes stochastic initial levels, noise,
// readout confusion and transitions on already noncomputational sources. Fail
// closed until their draws have replay seams instead of silently sampling them.
inline void validate_weighted_replay(const PreparedNonComputationalRequest& request) {
    const auto& model = request.model();
    const auto& resolved = request.resolved_circuit();
    if (!request.requirements().final_status)
        throw std::invalid_argument("weighted replay requires final statuses");
    if (resolved.circuit().num_qubits > 3 || model.initial_probability(Level::G) != 1 ||
        model.policy().damping != DampingPolicy::Exact)
        throw std::invalid_argument(
            "weighted replay requires at most three qubits and exact G initialization");
    if (const auto* classifier = model.classifier()) {
        if (classifier->has_herald())
            throw std::invalid_argument("weighted replay does not support stochastic readout");
        for (Level level : kAllLevels)
            if (classifier->prob(0, level) != 0 && classifier->prob(0, level) != 1)
                throw std::invalid_argument("weighted replay does not support stochastic readout");
    }
    const auto& circuit = resolved.circuit();
    unsigned instruments = 0;
    std::vector<bool> instrument_seen(circuit.num_qubits);
    for (const auto& node : resolved.circuit().nodes) {
        switch (node.gate) {
            case GateType::H:
            case GateType::T:
            case GateType::R_Y:
            case GateType::CX:
            case GateType::CZ:
            case GateType::M:
            case GateType::HERALD_LOSS_EVENT:
            case GateType::DETECTOR:
            case GateType::OBSERVABLE_INCLUDE:
                break;
            case GateType::LEVEL_TRANSITION:
                if (const auto* channel = model.transition_named(node.tag);
                    !channel || channel->column_sum(Level::G) + channel->column_sum(Level::E) == 0)
                    throw std::invalid_argument(
                        "weighted replay requires a live computational instrument");
                for (auto target : node.targets) {
                    if (instrument_seen[target.value()])
                        throw std::invalid_argument(
                            "weighted replay requires distinct instrument operands");
                    instrument_seen[target.value()] = true;
                    ++instruments;
                }
                break;
            default:
                throw std::invalid_argument("weighted replay does not support this operation");
        }
    }
    if (uint64_t{circuit.num_measurements} + 2 * circuit.num_qubits + 3 * instruments > 64 ||
        circuit.num_detectors > 64 || circuit.num_observables > 64)
        throw std::invalid_argument("weighted replay exceeds the physical-history key capacity");
}

struct ReplayEnsemble {
    PhysicalEnsemble states;
    size_t leaves = 0;
    uint64_t continuations = 0, growths = 0;
};

inline ReplayEnsemble replay_physical_histories(const PreparedNonComputationalRequest& request,
                                                bool online) {
    validate_weighted_replay(request);
    ReplayEnsemble result;
    std::vector<std::vector<unsigned>> pending(1);
    const auto root = make_seed_root(1, 17);
    const OnlineNonComputationalProgram online_program(request);
    size_t runs = 0;
    while (!pending.empty()) {
        REQUIRE(++runs <= 16384);
        auto tape = std::move(pending.back());
        pending.pop_back();
        WeightedReplay replay;
        REQUIRE(tape.size() <= replay.capacity);
        replay.supplied = tape.size();
        std::copy(tape.begin(), tape.end(), replay.tape.begin());
        struct Scope {
            NoncompReplayHooks* previous = noncomp_replay;
            explicit Scope(NoncompReplayHooks& hooks) { noncomp_replay = &hooks; }
            ~Scope() { noncomp_replay = previous; }
        } scope(replay);
        auto sample = online ? online_program.sample(1, 17, 1, true)
                             : run_trajectory_driver(request, 1, root, 1, true);
        REQUIRE_FALSE(replay.invalid);
        REQUIRE(replay.used >= replay.supplied);
        REQUIRE(replay.status == sample.final_status);
        REQUIRE(replay.paulis.size() == (1U << (2 * sample.num_qubits)));
        // Expand the first unsupplied decision only. Completed runs with a
        // partial tape are discovery probes and contribute no probability mass.
        if (replay.used > replay.supplied) {
            const auto& decision = replay.decisions[replay.supplied];
            for (unsigned value = 0; value < decision.count; ++value)
                if (decision.masses[value] > 0) {
                    auto child = tape;
                    child.push_back(value);
                    pending.push_back(std::move(child));
                }
            continue;
        }
        ++result.leaves;
        REQUIRE(result.leaves <= 1024);
        PhysicalHistory history;
        history.events.assign(replay.events.begin(), replay.events.begin() + replay.event_count);
        for (auto event : history.events)
            REQUIRE(event <= kNumLevels);
        history.status = replay.status;
        history.records = sample.measurements;
        history.detectors = pack_bits(sample.detectors);
        history.observables = pack_bits(sample.observables);
        add_physical_branch(result.states, std::move(history), replay.weight,
                            std::move(replay.paulis));
        result.continuations += sample.stats->continuation_compilations;
        result.growths += sample.stats->online_state_growths;
    }
    return result;
}

}  // namespace clifft::test
