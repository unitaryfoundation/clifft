#include "clifft/circuit/parser.h"

#include "noncomp_replay_helpers.h"
#include "noncomp_test_helpers.h"

#include <catch2/generators/catch_generators.hpp>
#include <complex>
#include <numbers>

using namespace clifft;
using namespace clifft::test;

namespace {

constexpr double rare_rate = 0x1p-40;
constexpr ExactComparisonBudget precision{2e-10, 2e-10, 2e-10, 0};
constexpr std::array<const char*, 6> boundaries = {
    "prepared",          "first instrument",          "between instruments",
    "second instrument", "feedback and interference", "visible outputs"};

NonComputationalModel replay_model() {
    auto first = zero_transition_matrix();
    first[level_index(Level::Lost)][level_index(Level::G)] = 0.2;
    first[level_index(Level::Lost)][level_index(Level::E)] = 0.6;
    auto second = zero_transition_matrix();
    second[level_index(Level::E)][level_index(Level::G)] = rare_rate;
    second[level_index(Level::LeakE)][level_index(Level::E)] = 0.4;
    return NonComputationalModel::from_spec(
        pure_initial_state(Level::G), {{"first", first}, {"second", second}},
        classifier_matrix_with_column(Level::LeakE, {0, 1}), {});
}

Circuit prefix(unsigned boundary) {
    std::string circuit = "H 0\nM 0\nR_Y(0.333333333333333333) 0\nCX 0 1\nT 1\n";
    if (boundary >= 1)
        circuit += "LEVEL_TRANSITION[first] 0\n";
    if (boundary >= 2)
        circuit += "HERALD_LOSS_EVENT 0\nH 1\n";
    if (boundary >= 3)
        circuit += "LEVEL_TRANSITION[second] 1\n";
    if (boundary >= 4)
        circuit += "CZ rec[-2] 1\nCZ rec[-1] 1\nH 0 1\n";
    if (boundary >= 5)
        circuit +=
            "M 0 1\nDETECTOR rec[-4] rec[-3] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1]\n";
    return parse(circuit);
}

unsigned instrument_count(unsigned boundary) {
    return static_cast<unsigned>(boundary >= 1) + static_cast<unsigned>(boundary >= 3);
}

// Independent physical-basis Kraus calculation: no tableau, compiler, production
// kernels, or transition helper is used. Unnormalized amplitudes carry joint
// history weights; source-resolved loss branches are mixed only at comparison.
using Complex = std::complex<long double>;
struct ReferenceBranch {
    std::array<Complex, 4> amplitudes{};
    PhysicalHistory history{{}, {QubitStatus::Computational, QubitStatus::Computational}, {}};
};

long double norm(const ReferenceBranch& branch) {
    long double result = 0;
    for (auto value : branch.amplitudes)
        result += std::norm(value);
    return result;
}

void hadamard(ReferenceBranch& branch, unsigned qubit) {
    if (branch.history.status[qubit] != QubitStatus::Computational)
        return;
    for (unsigned i = 0; i < 4; ++i)
        if (!(i & (1U << qubit))) {
            const auto zero = branch.amplitudes[i];
            const auto one = branch.amplitudes[i | (1U << qubit)];
            branch.amplitudes[i] = (zero + one) / std::sqrt(2.L);
            branch.amplitudes[i | (1U << qubit)] = (zero - one) / std::sqrt(2.L);
        }
}

ReferenceBranch jump(const ReferenceBranch& input, unsigned qubit, unsigned source,
                     unsigned destination, long double rate) {
    auto result = input;
    result.amplitudes.fill(0);
    for (unsigned i = 0; i < 4; ++i)
        if (((i >> qubit) & 1) == source)
            result.amplitudes[(i & ~(1U << qubit)) | (destination << qubit)] =
                input.amplitudes[i] * std::sqrt(rate);
    return result;
}

std::vector<ReferenceBranch> instrument(const std::vector<ReferenceBranch>& input, bool second) {
    std::vector<ReferenceBranch> result;
    const unsigned qubit = second ? 1 : 0;
    const long double rates[] = {second ? static_cast<long double>(rare_rate) : 0.2L,
                                 second ? 0.4L : 0.6L};
    for (const auto& before : input) {
        auto no_fire = before;
        for (unsigned i = 0; i < 4; ++i)
            no_fire.amplitudes[i] *= std::sqrt(1 - rates[(i >> qubit) & 1]);
        no_fire.history.events.push_back(0);
        result.push_back(no_fire);
        for (unsigned source = 0; source < 2; ++source) {
            const unsigned destination = second && source == 0 ? 1 : 0;
            auto fire = jump(before, qubit, source, destination, rates[source]);
            fire.history.events.push_back(second ? (source ? 4 : 2) : 5);
            fire.history.status[qubit] =
                second ? (source ? QubitStatus::LeakE : QubitStatus::Computational)
                       : QubitStatus::Lost;
            result.push_back(fire);
        }
    }
    return result;
}

std::vector<double> reference_moments(const ReferenceBranch& branch) {
    std::vector<double> result(16);
    const long double weight = norm(branch);
    for (unsigned code = 0; code < 16; ++code) {
        Complex value = 0;
        for (unsigned i = 0; i < 4; ++i) {
            unsigned output = i;
            Complex phase = 1;
            for (unsigned q = 0; q < 2; ++q) {
                const unsigned p = (code >> (2 * q)) & 3;
                const bool bit = (i >> q) & 1;
                if (p == 1 || p == 2)
                    output ^= 1U << q;
                if (p == 2)
                    phase *= bit ? Complex(0, -1) : Complex(0, 1);
                if (p == 3 && bit)
                    phase = -phase;
            }
            value += std::conj(branch.amplitudes[output]) * phase * branch.amplitudes[i];
        }
        REQUIRE(std::abs(value.imag()) < 1e-18L);
        result[code] = static_cast<double>(value.real() / weight);
    }
    return result;
}

PhysicalEnsemble reference(unsigned boundary) {
    ReferenceBranch prepared;
    // Unequal populations expose incorrect source weights. The prefix bit
    // affects both the quantum state and feedback after continuations.
    prepared.amplitudes[0] = std::sqrt(3.L / 8);
    prepared.amplitudes[3] = std::polar(std::sqrt(1.L / 8), std::numbers::pi_v<long double> / 4);
    std::vector<ReferenceBranch> branches(2, prepared);
    branches[0].history.records.push_back(0);
    branches[1].history.records.push_back(1);
    branches[1].amplitudes[0] = -std::sqrt(1.L / 8);
    branches[1].amplitudes[3] = std::polar(std::sqrt(3.L / 8), std::numbers::pi_v<long double> / 4);
    if (boundary >= 1)
        branches = instrument(branches, false);
    if (boundary >= 2)
        for (auto& branch : branches) {
            branch.history.records.push_back(branch.history.status[0] == QubitStatus::Lost);
            hadamard(branch, 1);
        }
    if (boundary >= 3)
        branches = instrument(branches, true);
    if (boundary >= 4)
        for (auto& branch : branches) {
            if ((branch.history.records[0] ^ branch.history.records[1]) &&
                branch.history.status[1] == QubitStatus::Computational)
                for (unsigned i = 2; i < 4; ++i)
                    branch.amplitudes[i] = -branch.amplitudes[i];
            hadamard(branch, 0);
            hadamard(branch, 1);
        }
    if (boundary >= 5) {
        for (unsigned q = 0; q < 2; ++q) {
            std::vector<ReferenceBranch> measured;
            for (auto branch : branches) {
                if (branch.history.status[q] == QubitStatus::Computational) {
                    for (unsigned bit = 0; bit < 2; ++bit) {
                        auto projected = jump(branch, q, bit, bit, 1);
                        if (norm(projected) > 0) {
                            projected.history.records.push_back(static_cast<uint8_t>(bit));
                            measured.push_back(projected);
                        }
                    }
                } else {
                    branch.history.records.push_back(branch.history.status[q] ==
                                                     QubitStatus::LeakE);
                    measured.push_back(branch);
                }
            }
            branches = std::move(measured);
        }
        for (auto& branch : branches) {
            const auto& records = branch.history.records;
            branch.history.detectors = records[0] ^ records[1] ^ records[3];
            branch.history.observables = records[2] ^ records[3];
        }
    }
    PhysicalEnsemble result;
    for (const auto& branch : branches) {
        const long double probability = norm(branch);
        if (probability == 0)
            continue;
        const double weight = static_cast<double>(probability);
        REQUIRE(weight > 0);
        add_physical_branch(result, branch.history, weight, reference_moments(branch));
    }
    return result;
}

std::vector<ExactPlanDistribution> reference_boundaries() {
    std::vector<ExactPlanDistribution> result;
    for (unsigned boundary = 0; boundary < boundaries.size(); ++boundary)
        result.push_back(physical_distribution(reference(boundary), prefix(boundary),
                                               instrument_count(boundary)));
    return result;
}

std::string first_difference(const std::vector<ExactPlanDistribution>& expected,
                             const std::vector<ExactPlanDistribution>& actual) {
    for (size_t boundary = 0; boundary < actual.size(); ++boundary) {
        std::string detail;
        if (!distributions_equal(expected[boundary], actual[boundary], precision, &detail)) {
            // The shared comparator permits hidden probe clusters. Here each
            // key already denotes one mixed state, so identify the differing
            // conditional moment instead of reporting its cluster as missing.
            for (const auto& a : expected[boundary].kept_outcomes) {
                for (const auto& b : actual[boundary].kept_outcomes) {
                    if (compare_classical_outcomes(a.outcome, b.outcome) != 0 ||
                        a.outcome.exp_vals.size() != b.outcome.exp_vals.size())
                        continue;
                    if (std::abs(a.probability - b.probability) >
                        precision.probability_relative * std::max(a.probability, b.probability))
                        break;
                    for (size_t p = 0; p < a.outcome.exp_vals.size(); ++p) {
                        if (std::abs(a.outcome.exp_vals[p] - b.outcome.exp_vals[p]) >
                            precision.expectation_absolute) {
                            detail = "conditional Pauli[" + std::to_string(p) +
                                     "] reference=" + precise_probability(a.outcome.exp_vals[p]) +
                                     " actual=" + precise_probability(b.outcome.exp_vals[p]) +
                                     "; " + detail;
                            return std::string(boundaries[boundary]) + ": " + detail;
                        }
                    }
                }
            }
            return std::string(boundaries[boundary]) + ": " + detail;
        }
    }
    return {};
}

}  // namespace

TEST_CASE("Weighted histories preserve physical conditional states across two instruments",
          "[noncomp][weighted_replay]") {
    const bool online = GENERATE(false, true);
    CAPTURE(online);
    const auto model = replay_model();
    const auto expected = reference_boundaries();
    std::vector<ExactPlanDistribution> actual;
    uint64_t continuations = 0, growths = 0;
    for (unsigned boundary = 0; boundary < boundaries.size(); ++boundary) {
        CAPTURE(boundaries[boundary]);
        const auto circuit = prefix(boundary);
        const auto replay =
            replay_physical_histories(PreparedNonComputationalRequest(circuit, model), online);
        continuations += replay.continuations;
        growths += replay.growths;
        if (boundary == 3) {
            REQUIRE(replay.states.size() == 12);
            REQUIRE(replay.leaves == 18);
            double rare_mass = 0;
            unsigned rare_histories = 0;
            for (const auto& [history, state] : replay.states)
                if (history.events[1] == 2) {
                    ++rare_histories;
                    rare_mass += state.weight;
                    REQUIRE(state.weight > 0);
                    REQUIRE(state.weight < 1e-12);
                }
            REQUIRE(rare_histories == 4);
            REQUIRE(std::abs(rare_mass / (rare_rate / 2) - 1) < 2e-10);
        }
        if (boundary == 5) {
            REQUIRE(replay.states.size() == 30);
            REQUIRE(replay.leaves == 40);
        }
        actual.push_back(physical_distribution(replay.states, circuit, instrument_count(boundary)));
        const auto difference = first_difference(expected, actual);
        INFO(difference);
        REQUIRE(difference.empty());
    }
    if (online) {
        REQUIRE(growths > 0);
        REQUIRE(continuations == 0);
    } else {
        REQUIRE(continuations > 0);
    }
}

TEST_CASE("Weighted physical comparisons identify the first corrupt boundary",
          "[noncomp][weighted_replay]") {
    const auto expected = reference_boundaries();
    auto actual = expected;
    unsigned boundary = 0;
    bool conditional_state_corruption = false;
    SECTION("incorrect rare history weight") {
        boundary = 3;
        for (auto& entry : actual[boundary].kept_outcomes)
            if (entry.probability < 1e-12)
                entry.probability *= 1.5;
    }
    SECTION("omitted rare history") {
        boundary = 3;
        std::erase_if(actual[boundary].kept_outcomes,
                      [](const auto& entry) { return entry.probability < 1e-12; });
    }
    SECTION("omitted no-fire evolution") {
        boundary = 1;
        conditional_state_corruption = true;
        // The prefix-zero no-fire branch has mass 0.35 and populations 6/7 and 1/7.
        // Replacing its state with the prepared state preserves its weight.
        for (auto& entry : actual[boundary].kept_outcomes)
            if (entry.probability > 0.3)
                entry.outcome.exp_vals = expected[0].kept_outcomes[0].outcome.exp_vals;
    }
    SECTION("lost correlations") {
        boundary = 1;
        conditional_state_corruption = true;
        for (auto& entry : actual[boundary].kept_outcomes)
            for (unsigned p = 1; p < 4; ++p)
                for (unsigned q = 1; q < 4; ++q)
                    entry.outcome.exp_vals[p + 4 * q] =
                        entry.outcome.exp_vals[p] * entry.outcome.exp_vals[4 * q];
    }
    SECTION("stale continuation state") {
        boundary = 3;
        conditional_state_corruption = true;
        for (auto& entry : actual[boundary].kept_outcomes)
            if (entry.probability < 1e-12)
                entry.outcome.exp_vals[12] = 1;  // A g-to-e jump must have Z1 = -1.
    }
    SECTION("stale classical dependency") {
        boundary = 4;
        conditional_state_corruption = true;
        // Dropping the saved random bit omits Z before the final H on q1.
        // The resulting X conjugation reverses Y1 and Z1 moments.
        for (auto& entry : actual[boundary].kept_outcomes)
            if ((entry.outcome.visible_records >> 10) & 1)
                for (unsigned p = 0; p < 16; ++p)
                    if ((p >> 2) >= 2)
                        entry.outcome.exp_vals[p] *= -1;
    }
    SECTION("stale detector dependency") {
        boundary = 5;
        for (auto& entry : actual[boundary].kept_outcomes)
            entry.outcome.detectors ^= 1;
    }
    SECTION("inconclusive omission bound") {
        boundary = 3;
        actual[boundary].omitted_probability_bound = rare_rate;
    }
    const auto difference = first_difference(expected, actual);
    INFO(difference);
    REQUIRE(difference.starts_with(std::string(boundaries[boundary]) + ":"));
    REQUIRE(difference.find("branch-relative") != std::string::npos);
    REQUIRE((difference.find("conditional Pauli[") != std::string::npos) ==
            conditional_state_corruption);
}

TEST_CASE("Weighted replay rejects uninstrumented randomness", "[noncomp][weighted_replay]") {
    auto circuit = parse("H 0\nX_ERROR(0.1) 0\nM 0\n");
    REQUIRE_THROWS_AS(
        replay_physical_histories(PreparedNonComputationalRequest(circuit, replay_model()), false),
        std::invalid_argument);
}

TEST_CASE("Weighted replay refuses impossible selections and underflowed weights",
          "[noncomp][weighted_replay]") {
    WeightedReplay replay;
    replay.supplied = 2;
    SECTION("zero support") {
        const double masses[] = {0, 1};
        REQUIRE(replay.choose(ReplayChoice::Measurement, masses) == 1);
    }
    SECTION("unrepresentable positive weight") {
        const double masses[] = {1e-200, 1};
        REQUIRE(replay.choose(ReplayChoice::Measurement, masses) == 0);
        REQUIRE_FALSE(replay.invalid);
        REQUIRE(replay.choose(ReplayChoice::Measurement, masses) == 0);
    }
    REQUIRE(replay.invalid);
}
