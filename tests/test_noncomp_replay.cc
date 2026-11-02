#include "clifft/circuit/parser.h"

#include "noncomp_replay_helpers.h"
#include "noncomp_test_helpers.h"

#include <catch2/generators/catch_generators.hpp>
#include <complex>
#include <limits>
#include <numbers>

using namespace clifft;
using namespace clifft::test;

std::optional<unsigned> clifft::detail::replay_branch(BranchChoice kind,
                                                      std::span<const double> masses) noexcept {
    if (test::noncomp_replay)
        return test::noncomp_replay->choose(kind, masses);
    return std::nullopt;
}

void clifft::detail::replay_append_probes(Circuit& circuit) {
    if (test::noncomp_replay)
        test::noncomp_replay->append_probes(circuit);
}

void clifft::detail::replay_inspect(std::span<const double> probes,
                                    std::span<const QubitStatus> status) {
    if (test::noncomp_replay)
        test::noncomp_replay->inspect(probes, status);
}

void clifft::detail::replay_inspect_online(const noncomp_detail::OnlineState& quantum,
                                           std::span<const QubitStatus> status) {
    if (test::noncomp_replay)
        test::noncomp_replay->inspect_online(quantum, status);
}

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
    PhysicalHistory history{{}, {QubitStatus::Computational, QubitStatus::Computational}, {}, 0, 0,
                            {}};
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
    // Some platforms give long double only double precision. Scaling roundoff
    // by branch mass keeps this check equally strict for rare histories.
    const long double roundoff = 32 * std::numeric_limits<long double>::epsilon() * weight;
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
        REQUIRE(std::abs(value.imag()) <= roundoff);
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
                             const std::vector<ExactPlanDistribution>& actual,
                             std::span<const char* const> labels = boundaries) {
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
                            return std::string(labels[boundary]) + ": " + detail;
                        }
                    }
                }
            }
            return std::string(labels[boundary]) + ": " + detail;
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

TEST_CASE("Weighted marginalized histories retain only visible key bits",
          "[noncomp][weighted_replay]") {
    const bool online = GENERATE(false, true);
    std::string text;
    for (unsigned i = 0; i < 22; ++i)
        text += "LEVEL_TRANSITION[first] 0\n";
    const auto circuit = parse(text);
    const auto replay = replay_physical_histories(
        PreparedNonComputationalRequest(circuit, replay_model()), online, false);
    REQUIRE(replay.states.size() == 2);
    const double survival = std::pow(0.8, 22);
    for (const auto& [history, state] : replay.states) {
        REQUIRE(history.events.empty());
        const bool computational = is_computational(history.status[0]);
        const double expected = computational ? survival : 1 - survival;
        REQUIRE(std::abs(state.weight / expected - 1) < precision.probability_relative);
        REQUIRE(std::abs(state.weighted_paulis[3] / state.weight - computational) <
                precision.expectation_absolute);
    }
    REQUIRE(physical_distribution(replay.states, circuit, 0).num_visible_records == 2);

    for (unsigned i = 0; i < 32; ++i)
        text += "M 0\n";
    REQUIRE_THROWS_AS(validate_weighted_replay(
                          PreparedNonComputationalRequest(parse(text), replay_model()), false),
                      std::invalid_argument);
}

TEST_CASE("Weighted measurement replay follows production dust reachability",
          "[noncomp][weighted_replay]") {
    const bool online = GENERATE(false, true);
    const bool dust = GENERATE(false, true);
    const bool excited = GENERATE(false, true);
    const bool inverted = GENERATE(false, true);
    CAPTURE(online, dust, excited, inverted);
    const double rotation = dust ? 1e-10 : 1e-8;
    const auto circuit = parse("R_Y(" + precise_probability(excited ? 1 - rotation : rotation) +
                               ") 0\nM " + (inverted ? "!0\n" : "0\n"));
    const auto replay =
        replay_physical_histories(PreparedNonComputationalRequest(circuit, replay_model()), online);
    REQUIRE(replay.leaves == (dust ? 1 : 2));
    REQUIRE(replay.states.size() == replay.leaves);
    const double rare_probability = std::pow(std::sin(rotation * std::numbers::pi / 2), 2);
    double total = 0;
    for (const auto& [history, state] : replay.states) {
        REQUIRE(history.records.size() == 1);
        const bool physical_bit = (history.records[0] != 0) ^ inverted;
        if (dust) {
            REQUIRE(physical_bit == excited);
            REQUIRE(state.weight == 1);
        } else if (physical_bit != excited) {
            REQUIRE(state.weight > kMeasurementDustEpsilon);
            // Subtracting the angle from one loses relative precision near pi.
            REQUIRE(std::abs(state.weight / rare_probability - 1) < 1e-6);
        }
        REQUIRE(std::abs(state.weighted_paulis[3] / state.weight - (physical_bit ? -1 : 1)) <
                precision.expectation_absolute);
        total += state.weight;
    }
    REQUIRE(std::abs(total - 1) < precision.normalization_absolute);
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

namespace {

constexpr std::array<const char*, 6> lifecycle_boundaries = {
    "entangled preparation", "loss and classified readout", "recapture",
    "reset and feedback",    "compaction and reactivation", "joint outputs"};

Circuit lifecycle_prefix(unsigned boundary) {
    std::string text = "H 0\nT 0\nCX 0 1\nR_Y(0.25) 1\n";
    if (boundary >= 1)
        text += "LEVEL_TRANSITION[lose] 0\nHERALD_LOSS_EVENT 0\nM 0\n";
    if (boundary >= 2)
        text += "LEVEL_TRANSITION[recapture] 0\nHERALD_LOSS_EVENT 0\n";
    if (boundary >= 3)
        text += "R 0\nR_Y(0.25) 0\nS 0\nCZ rec[-2] 1\n";
    if (boundary >= 4)
        text += "MX 1\nR_Y(0.375) 1\nCX rec[-1] 0\nH 0\n";
    if (boundary >= 5)
        text +=
            "M 0 1\nDETECTOR rec[-6] rec[-5] rec[-3] rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-4] rec[-2] rec[-1]\n";
    return parse(text);
}

NonComputationalModel lifecycle_model(bool restore_lost) {
    auto lose = zero_transition_matrix();
    lose[4][0] = 0.2;
    lose[4][1] = 0.6;
    auto recapture = zero_transition_matrix();
    recapture[1][4] = 0.3;
    recapture[3][4] = 0.2;
    return NonComputationalModel::from_spec(
        pure_initial_state(Level::G), {{"lose", lose}, {"recapture", recapture}},
        classifier_matrix_with_column(Level::Lost, {0.25, 0.25, 0.5}),
        {.reset_restores_lost = restore_lost});
}

void rotate_y(ReferenceBranch& branch, unsigned q, long double turns) {
    if (!is_computational(branch.history.status[q]))
        return;
    const long double c = std::cos(turns * std::numbers::pi_v<long double> / 2);
    const long double s = std::sin(turns * std::numbers::pi_v<long double> / 2);
    for (unsigned i = 0; i < 4; ++i)
        if (!(i & (1U << q))) {
            const auto zero = branch.amplitudes[i], one = branch.amplitudes[i | (1U << q)];
            branch.amplitudes[i] = c * zero - s * one;
            branch.amplitudes[i | (1U << q)] = s * zero + c * one;
        }
}

void record(ReferenceBranch& branch, bool bit, bool herald = false) {
    branch.history.records.push_back(bit);
    branch.history.heralds.push_back(herald);
}

std::vector<ReferenceBranch> read_z(const std::vector<ReferenceBranch>& input, unsigned q,
                                    bool reset = false, bool restore_lost = false) {
    std::vector<ReferenceBranch> result;
    for (auto before : input) {
        const auto status = before.history.status[q];
        if (is_computational(status)) {
            for (unsigned bit = 0; bit < 2; ++bit) {
                auto branch = jump(before, q, bit, reset ? 0 : bit, 1);
                if (!reset)
                    record(branch, bit);
                if (norm(branch) > 0)
                    result.push_back(branch);
            }
        } else if (reset) {
            if (status != QubitStatus::Lost || restore_lost)
                before.history.status[q] = QubitStatus::Computational;
            result.push_back(before);
        } else if (status == QubitStatus::Lost) {
            // The herald symbol has a fair placeholder bit. Its sidecar must
            // remain correlated with the physical level and every later output.
            for (unsigned symbol = 0; symbol < 4; ++symbol) {
                auto branch = before;
                for (auto& value : branch.amplitudes)
                    value *= 0.5L;
                record(branch, symbol & 1, symbol >= 2);
                result.push_back(branch);
            }
        } else {
            record(before, false);
            result.push_back(before);
        }
    }
    return result;
}

PhysicalEnsemble lifecycle_reference(unsigned boundary, bool restore_lost) {
    ReferenceBranch initial;
    initial.amplitudes[0] = 1 / std::sqrt(2.L);
    initial.amplitudes[3] = std::polar(1 / std::sqrt(2.L), std::numbers::pi_v<long double> / 4);
    rotate_y(initial, 1, 0.25L);
    std::vector<ReferenceBranch> branches{initial};
    if (boundary >= 1) {
        branches = instrument(branches, false);
        for (auto& branch : branches)
            record(branch, branch.history.status[0] == QubitStatus::Lost);
        branches = read_z(branches, 0);
    }
    if (boundary >= 2) {
        std::vector<ReferenceBranch> recaptured;
        for (auto before : branches) {
            if (before.history.status[0] == QubitStatus::Lost) {
                for (unsigned outcome = 0; outcome < 3; ++outcome) {
                    auto branch = before;
                    const long double rates[] = {0.5L, 0.3L, 0.2L};
                    if (outcome == 1) {
                        branch = jump(branch, 0, 0, 1, 1);
                        branch.history.status[0] = QubitStatus::Computational;
                    } else if (outcome == 2) {
                        branch.history.status[0] = QubitStatus::LeakE;
                    }
                    for (auto& value : branch.amplitudes)
                        value *= std::sqrt(rates[outcome]);
                    record(branch, outcome == 0);
                    recaptured.push_back(branch);
                }
            } else {
                record(before, false);
                recaptured.push_back(before);
            }
        }
        branches = std::move(recaptured);
    }
    if (boundary >= 3) {
        branches = read_z(branches, 0, true, restore_lost);
        for (auto& branch : branches) {
            rotate_y(branch, 0, 0.25L);
            for (unsigned i = 0; i < 4; ++i) {
                if ((i & 1) && is_computational(branch.history.status[0]))
                    branch.amplitudes[i] *= Complex(0, 1);
                if ((i & 2) && branch.history.records[1])
                    branch.amplitudes[i] = -branch.amplitudes[i];
            }
        }
    }
    if (boundary >= 4) {
        for (auto& branch : branches)
            hadamard(branch, 1);
        branches = read_z(branches, 1);
        for (auto& branch : branches) {
            hadamard(branch, 1);
            rotate_y(branch, 1, 0.375L);
            if (branch.history.records.back() && is_computational(branch.history.status[0])) {
                std::swap(branch.amplitudes[0], branch.amplitudes[1]);
                std::swap(branch.amplitudes[2], branch.amplitudes[3]);
            }
            hadamard(branch, 0);
        }
    }
    if (boundary >= 5) {
        branches = read_z(read_z(branches, 0), 1);
        for (auto& branch : branches) {
            const auto& r = branch.history.records;
            branch.history.detectors = r[0] ^ r[1] ^ r[3] ^ r[5];
            branch.history.observables = r[2] ^ r[4] ^ r[5];
        }
    }
    PhysicalEnsemble result;
    for (auto& branch : branches) {
        if (norm(branch) == 0)
            continue;
        // Event decisions and reset outcomes are hidden; physical heralds and
        // records define the conditioning. Mix hidden sources before comparing.
        branch.history.events.clear();
        add_physical_branch(result, branch.history, static_cast<double>(norm(branch)),
                            reference_moments(branch));
    }
    return result;
}

}  // namespace

TEST_CASE("Weighted lifecycle histories retain heralds and coherences after recapture and reset",
          "[noncomp][weighted_replay]") {
    const bool online = GENERATE(false, true);
    const bool restore_lost = GENERATE(false, true);
    CAPTURE(online, restore_lost);
    std::vector<ExactPlanDistribution> expected, actual;
    uint64_t growths = 0, continuations = 0;
    for (unsigned boundary = 0; boundary < lifecycle_boundaries.size(); ++boundary) {
        CAPTURE(lifecycle_boundaries[boundary]);
        const auto circuit = lifecycle_prefix(boundary);
        expected.push_back(
            physical_distribution(lifecycle_reference(boundary, restore_lost), circuit, 0));
        const auto replay = replay_physical_histories(
            PreparedNonComputationalRequest(circuit, lifecycle_model(restore_lost)), online, false);
        actual.push_back(physical_distribution(replay.states, circuit, 0));
        growths += replay.growths;
        continuations += replay.continuations;
        const auto difference = first_difference(expected, actual, lifecycle_boundaries);
        INFO(difference);
        REQUIRE(difference.empty());
    }
    REQUIRE((continuations == 0) == online);
    REQUIRE((growths > 0) == online);
}

TEST_CASE("Weighted repeated no fire filters preserve normalized conditional states",
          "[noncomp][weighted_replay]") {
    const bool online = GENERATE(false, true);
    const bool entangled = GENERATE(false, true);
    CAPTURE(online, entangled);
    ReferenceBranch prepared;
    prepared.amplitudes.fill(0.5L);
    std::string text = "H 0 1\n";
    if (entangled) {
        prepared.amplitudes = {};
        prepared.amplitudes[0] = 1 / std::sqrt(2.L);
        prepared.amplitudes[3] =
            std::polar(1 / std::sqrt(2.L), std::numbers::pi_v<long double> / 4);
        text = "H 0\nT 0\nCX 0 1\n";
    }
    std::vector<ReferenceBranch> branches{prepared};
    for (unsigned count = 1; count <= 4; ++count) {
        CAPTURE(count);
        text += "LEVEL_TRANSITION[first] 0\nHERALD_LOSS_EVENT 0\n";
        std::vector<ReferenceBranch> next;
        for (auto branch : branches) {
            if (is_computational(branch.history.status[0])) {
                const auto split = instrument({branch}, false);
                next.insert(next.end(), split.begin(), split.end());
            } else {
                next.push_back(branch);
            }
        }
        branches = std::move(next);
        PhysicalEnsemble expected;
        for (auto& branch : branches) {
            record(branch, branch.history.status[0] == QubitStatus::Lost);
            branch.history.events.clear();
            add_physical_branch(expected, branch.history, static_cast<double>(norm(branch)),
                                reference_moments(branch));
        }
        const auto circuit = parse(text);
        const auto replay = replay_physical_histories(
            PreparedNonComputationalRequest(circuit, replay_model()), online, false);
        REQUIRE((replay.growths > 0) == online);
        std::string difference;
        const bool equal = distributions_equal(physical_distribution(expected, circuit, 0),
                                               physical_distribution(replay.states, circuit, 0),
                                               precision, &difference);
        INFO(difference);
        REQUIRE(equal);
        // A conditional no-fire state is renormalized at every step. Checking
        // its closed-form weight and Z moment distinguishes damping from skips.
        for (const auto& [history, state] : replay.states)
            if (is_computational(history.status[0])) {
                const double g = std::pow(0.8, count), e = std::pow(0.4, count);
                REQUIRE(std::abs(state.weight / ((g + e) / 2) - 1) < 2e-10);
                REQUIRE(std::abs(state.weighted_paulis[3] / state.weight - (g - e) / (g + e)) <
                        precision.expectation_absolute);
                if (!entangled)
                    REQUIRE(std::abs(state.weighted_paulis[1] / state.weight -
                                     2 * std::sqrt(g * e) / (g + e)) <
                            precision.expectation_absolute);
            }
    }
}

TEST_CASE("Weighted lifecycle diagnostics locate herald phase and reset corruption",
          "[noncomp][weighted_replay]") {
    const unsigned fault = GENERATE(0U, 1U, 2U, 3U);
    const unsigned boundary = fault + 1;
    std::vector<ExactPlanDistribution> expected;
    for (unsigned b = 0; b < lifecycle_boundaries.size(); ++b)
        expected.push_back(
            physical_distribution(lifecycle_reference(b, true), lifecycle_prefix(b), 0));
    auto actual = expected;
    if (fault == 0) {
        // Move mass between herald-conditioned outputs without changing the
        // sum. A record-only comparison would miss this sidecar corruption.
        actual[boundary].kept_outcomes[0].outcome.visible_records ^=
            uint64_t{1} << (lifecycle_prefix(boundary).num_measurements + 4);
    } else {
        // Pauli moments expose a wrong restoration state, a stale reset, or a
        // compaction sign even when the final computational populations agree.
        actual[boundary].kept_outcomes[0].outcome.exp_vals[fault == 1 ? 3 : 2] += 0.125;
    }
    const auto difference = first_difference(expected, actual, lifecycle_boundaries);
    INFO(difference);
    REQUIRE(difference.starts_with(std::string(lifecycle_boundaries[boundary]) + ":"));
}

TEST_CASE("Weighted repeated activation distinguishes zero dust and tiny joint support",
          "[noncomp][weighted_replay]") {
    const bool online = GENERATE(false, true);
    const double angle = GENERATE(0.0, 1e-10, 1e-8);
    CAPTURE(online, angle);
    const auto rotation = "R_Y(" + precise_probability(angle) + ") 0\nM 0\n";
    const auto circuit = parse(rotation + "R 0\n" + rotation + "R 0\n" + rotation);
    const auto replay =
        replay_physical_histories(PreparedNonComputationalRequest(circuit, replay_model()), online);
    const double p = angle == 1e-8 ? std::pow(std::sin(angle * std::numbers::pi / 2), 2) : 0;
    REQUIRE(replay.states.size() == (p > 0 ? 8 : 1));
    double total = 0;
    for (const auto& [history, state] : replay.states) {
        double expected = 1;
        for (auto bit : history.records)
            expected *= bit ? p : 1 - p;
        REQUIRE(expected > 0);
        REQUIRE(std::abs(state.weight / expected - 1) < precision.probability_relative);
        REQUIRE(std::abs(state.weighted_paulis[3] / state.weight -
                         (history.records.back() ? -1 : 1)) < precision.expectation_absolute);
        total += state.weight;
    }
    REQUIRE(std::abs(total - 1) < precision.normalization_absolute);
    if (online) {
        REQUIRE(replay.continuations == 0);
        REQUIRE((replay.growths > 0) == (angle > 0));
    }
}
