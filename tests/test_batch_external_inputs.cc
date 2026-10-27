#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace clifft::sampling;

namespace {

ExecutablePlan compile(std::string_view circuit) {
    return ExecutablePlan(plan_sampling(clifft::trace(clifft::parse(circuit))));
}

void set(std::span<uint64_t> column, uint32_t lane, bool value) {
    const uint64_t mask = uint64_t{1} << (lane & 63);
    column[lane >> 6] = (column[lane >> 6] & ~mask) | (value ? mask : 0);
}

// Enumerate the four deterministic binary channels, independently of the
// executor's conditional-flip expression.
bool reported(bool source, uint32_t channel) {
    switch (channel) {
        case 0:
            return source;
        case 1:
            return !source;
        case 2:
            return true;
        default:
            return false;
    }
}

void set_channel(BatchExternalInputs& inputs, uint32_t site, uint32_t lane, uint32_t channel) {
    set(inputs.readout_zero(site), lane, reported(false, channel));
    set(inputs.readout_one(site), lane, !reported(true, channel));
}

void compare_scalar(const BatchExecutor& batch, const Executor& scalar, uint32_t lane,
                    const ExecutablePlan& plan) {
    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
        REQUIRE(batch.measurement(lane, record) == (scalar.visible_records()[record] != 0));
    }
    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
        REQUIRE(batch.detector(lane, detector) == (scalar.detectors()[detector] != 0));
    }
    for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
        REQUIRE(batch.observable(lane, observable) == (scalar.observables()[observable] != 0));
    }
}

}  // namespace

TEST_CASE("Packed external noise preserves linear effects and reusable partial batches") {
    const uint32_t sites = GENERATE(2u, 40u);
    const uint32_t unbound = GENERATE(0u, 1u, 2u);
    const uint32_t capacity = GENERATE(1u, 64u, 65u, 129u);
    CAPTURE(sites, unbound, capacity);
    std::string circuit;
    for (uint32_t site = 0; site < sites; ++site) {
        circuit += "DEPOLARIZE1(0.75) 0\nM 0\n";
    }
    circuit +=
        "CX rec[-1] 1\nM 1\nDETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n";
    auto source = plan_sampling(clifft::trace(clifft::parse(circuit)));
    REQUIRE(source.presampled_noise_sites.size() == sites);
    for (const auto& site : source.presampled_noise_sites) {
        REQUIRE(site.outcomes.size() == 3);
    }
    // The noncomputational driver supplies symbols without a distribution.
    // Remove bindings while retaining the parsed circuit's quantum behavior.
    if (unbound == 1) {
        source.presampled_noise_sites.erase(source.presampled_noise_sites.begin(),
                                            source.presampled_noise_sites.begin() + sites / 2);
    } else if (unbound == 2) {
        source.presampled_noise_sites.clear();
    }
    const ExecutablePlan plan(source);
    REQUIRE(plan.num_presampled_symbols() == 3 * sites);
    REQUIRE((plan.num_batch_noise_carriers() != 0) == (sites == 40));
    REQUIRE((plan.num_unbound_presampled_symbols() != 0) == (unbound != 0));
    REQUIRE(plan.num_readout_noise_sites() == 0);
    BatchExternalInputs inputs(plan, capacity);
    BatchExecutor batch(plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    Executor scalar(plan);
    std::vector<uint8_t> values(plan.num_presampled_symbols());
    const auto root = clifft::make_seed_root(1024, uint64_t{314});
    uint32_t iteration = 0;
    for (uint32_t shots : {capacity, capacity - 1, 0u, 1u, capacity}) {
        CAPTURE(shots, iteration);
        inputs.clear();
        for (uint32_t input = 0; input < values.size(); ++input) {
            // Deliberately poison both padding and inactive lanes.
            std::ranges::fill(inputs.noise(input), ~uint64_t{0});
            for (uint32_t lane = 0; lane < shots; ++lane) {
                set(inputs.noise(input), lane, (((lane + iteration) >> (input % 7)) & 1) != 0);
            }
        }
        batch.run_batch(root, 17, shots, inputs);
        REQUIRE(batch.surviving_shots() == shots);
        for (uint32_t lane = 0; lane < shots; ++lane) {
            CAPTURE(lane);
            REQUIRE(batch.shot_index(lane) == 17 + lane);
            bool expected = false;
            for (uint32_t input = 0; input < values.size(); ++input) {
                values[input] = ((lane + iteration) >> (input % 7)) & 1;
            }
            for (uint32_t site = 0; site < sites; ++site) {
                // X and Y each flip a Z-basis measurement; Z has no effect.
                // Supplying both X and Y must cancel, even in a shared carrier.
                expected ^= (values[3 * site] != 0) ^ (values[3 * site + 1] != 0);
                REQUIRE(batch.measurement(lane, site) == expected);
            }
            REQUIRE(batch.measurement(lane, sites) == expected);
            REQUIRE_FALSE(batch.detector(lane, 0));
            REQUIRE(batch.observable(lane, 0) == expected);
            scalar.run_shot(values, {});
            compare_scalar(batch, scalar, lane, plan);
        }
        ++iteration;
    }
}

TEST_CASE("Packed external Pauli pairs populate every factored carrier") {
    const uint32_t sites = GENERATE(2u, 16u);
    const bool unbound = GENERATE(false, true);
    CAPTURE(sites, unbound);
    std::string circuit = "H 0\nCX 0 1\n";
    for (uint32_t site = 0; site < sites; ++site) {
        circuit += "DEPOLARIZE2(0.75) 0 1\nCX 0 1\nH 0\nM 0 1\nH 0\nCX 0 1\n";
    }
    auto source = plan_sampling(clifft::trace(clifft::parse(circuit)));
    if (unbound) {
        source.presampled_noise_sites.clear();
    }
    const ExecutablePlan plan(source);
    REQUIRE(plan.num_presampled_symbols() == 15 * sites);
    REQUIRE((plan.num_batch_noise_carriers() != 0) == (sites == 16));
    constexpr uint32_t shots = 129;
    BatchExternalInputs inputs(plan, shots);
    BatchExecutor batch(plan, shots, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    const auto supplied = [](uint32_t input, uint32_t lane) {
        return lane < 16 ? input + 1 == lane : (((lane * 17 + input / 15) >> (input % 7)) & 1) != 0;
    };
    for (uint32_t input = 0; input < plan.num_presampled_symbols(); ++input) {
        for (uint32_t lane = 0; lane < shots; ++lane) {
            set(inputs.noise(input), lane, supplied(input, lane));
        }
    }
    batch.run_batch(clifft::make_seed_root(shots, uint64_t{81}), 0, shots, inputs);
    for (uint32_t lane = 0; lane < shots; ++lane) {
        CAPTURE(lane);
        bool phase = false;
        bool bit = false;
        for (uint32_t site = 0; site < sites; ++site) {
            // Decoding a Bell pair measures ZZ and XX syndromes. Each local
            // X or Y flips ZZ; each local Z or Y flips XX. Pauli pairs use
            // the public IX IY IZ XI XX ... ZZ channel order.
            for (uint32_t pair = 1; pair < 16; ++pair) {
                if (supplied(15 * site + pair - 1, lane)) {
                    for (uint32_t pauli : {pair / 4, pair % 4}) {
                        phase ^= pauli == 2 || pauli == 3;
                        bit ^= pauli == 1 || pauli == 2;
                    }
                }
            }
            REQUIRE(batch.measurement(lane, 2 * site) == phase);
            REQUIRE(batch.measurement(lane, 2 * site + 1) == bit);
        }
    }
}

TEST_CASE("Packed external readout preserves hidden reset outcomes and ordered feedback") {
    const std::string basis = GENERATE("", "X", "Y");
    const bool inverted = GENERATE(false, true);
    CAPTURE(basis, inverted);
    const std::string prepare = basis.empty() ? "" : basis == "X" ? "H 0\n" : "H 0\nS 0\n";
    const std::string error = basis.empty() ? "X_ERROR" : "Z_ERROR";
    const auto plan =
        compile(prepare + error + "(0.1) 0\nMR" + basis + (inverted ? " !0\n" : " 0\n") +
                "READOUT_NOISE(0.2,0.3) rec[-1]\nM" + basis + " 0\nM" + basis +
                " 0\nREADOUT_NOISE(0.4,0.5) rec[-1]\n"
                "READOUT_NOISE(0.6,0.7) rec[-1]\nCX rec[-1] 1\nM 1\n"
                "DETECTOR rec[-4] rec[-3] rec[-2] rec[-1]\n"
                "OBSERVABLE_INCLUDE(0) rec[-1]\n");
    REQUIRE(plan.num_presampled_symbols() == 1);
    REQUIRE(plan.num_readout_noise_sites() == 3 + static_cast<uint32_t>(inverted));
    REQUIRE(plan.num_visible_records() == 4);
    constexpr uint32_t shots = 129;
    BatchExternalInputs inputs(plan, shots);
    BatchExecutor batch(plan, shots, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    Executor scalar(plan);
    for (uint32_t lane = 0; lane < shots; ++lane) {
        set(inputs.noise(0), lane, (lane & 1) != 0);
        uint32_t offset = 0;
        if (inverted) {
            set_channel(inputs, offset++, lane, 1);
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            set_channel(inputs, offset + channel, lane, (lane >> (1 + 2 * channel)) & 3);
        }
    }
    batch.run_batch(clifft::make_seed_root(shots, uint64_t{71}), 0, shots, inputs);
    REQUIRE(batch.surviving_shots() == shots);
    for (uint32_t lane = 0; lane < shots; ++lane) {
        CAPTURE(lane);
        const bool first = reported(((lane & 1) != 0) ^ inverted, (lane >> 1) & 3);
        const bool last = reported(reported(false, (lane >> 3) & 3), (lane >> 5) & 3);
        REQUIRE(batch.measurement(lane, 0) == first);
        REQUIRE_FALSE(batch.measurement(lane, 1));
        REQUIRE(batch.measurement(lane, 2) == last);
        REQUIRE(batch.measurement(lane, 3) == last);
        REQUIRE(batch.detector(lane, 0) == first);
        REQUIRE(batch.observable(lane, 0) == last);
        std::vector<ReadoutNoiseInput> readouts(plan.num_readout_noise_sites());
        for (uint32_t site = 0; site < readouts.size(); ++site) {
            readouts[site] = {((inputs.readout_zero(site)[lane >> 6] >> (lane & 63)) & 1) != 0,
                              ((inputs.readout_one(site)[lane >> 6] >> (lane & 63)) & 1) != 0};
        }
        const std::array<uint8_t, 1> noise{static_cast<uint8_t>(lane & 1)};
        scalar.run_shot(noise, readouts);
        compare_scalar(batch, scalar, lane, plan);
    }
}

TEST_CASE("Packed external inputs retain non Clifford probabilities and feedback correlations") {
    const auto plan = compile(
        "H 0\nT 0\nZ_ERROR(0.3) 0\nH 0\nM(0.2) 0\n"
        "CX rec[-1] 1\nM 1\nDETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n");
    REQUIRE(plan.peak_active_width() > 0);
    REQUIRE(plan.num_presampled_symbols() == 1);
    REQUIRE(plan.num_readout_noise_sites() == 1);
    constexpr uint32_t capacity = 129;
    constexpr uint32_t shots = 8192;
    BatchExternalInputs inputs(plan, capacity);
    BatchExecutor batch(plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    const auto root = clifft::make_seed_root(shots, uint64_t{2341});
    std::array<uint32_t, 8> ones{};
    for (uint32_t first = 0; first < shots; first += capacity) {
        const uint32_t count = std::min(capacity, shots - first);
        inputs.clear();
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint32_t group = (first + lane) % 8;
            set(inputs.noise(0), lane, (group & 1) != 0);
            set_channel(inputs, 0, lane, group >> 1);
        }
        batch.run_batch(root, first, count, inputs);
        REQUIRE(batch.surviving_shots() == count);
        for (uint32_t lane = 0; lane < count; ++lane) {
            const bool bit = batch.measurement(lane, 0);
            REQUIRE(batch.measurement(lane, 1) == bit);
            REQUIRE_FALSE(batch.detector(lane, 0));
            REQUIRE(batch.observable(lane, 0) == bit);
            ones[(first + lane) % 8] += bit;
        }
    }
    const double p = (1.0 - std::sqrt(0.5)) / 2;
    for (uint32_t group = 0; group < 8; ++group) {
        CAPTURE(group);
        const uint32_t channel = group >> 1;
        if (channel >= 2) {
            REQUIRE(ones[group] == (channel == 2 ? shots / 8 : 0));
        } else {
            const double expected = ((group & 1) != channel) ? 1 - p : p;
            REQUIRE_THAT(static_cast<double>(ones[group]) / (shots / 8),
                         Catch::Matchers::WithinAbs(expected, 0.06));
        }
    }
}

TEST_CASE("Packed external carriers compose with readout reset and non Clifford feedback") {
    const bool mixed_bindings = GENERATE(false, true);
    CAPTURE(mixed_bindings);
    constexpr uint32_t rounds = 40;
    std::string circuit;
    for (uint32_t round = 0; round < rounds; ++round) {
        circuit += "DEPOLARIZE1(0.75) 0\nM(0.2) 0\nCX rec[-1] 1\n";
    }
    circuit +=
        "MR(0.2) 0\nM 0\nH 1\nT 1\nH 1\nEXP_VAL Z1\n"
        "M(0.2) 1\nCX rec[-1] 2\nM 2\nDETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n";
    auto source = plan_sampling(clifft::trace(clifft::parse(circuit)));
    const auto first_readout = std::ranges::find(source.symbols, SymbolKind::Readout);
    REQUIRE(first_readout != source.symbols.end());
    REQUIRE(static_cast<size_t>(first_readout - source.symbols.begin()) <
            index(source.presampled_noise_sites.back().outcomes.front().symbol));
    if (mixed_bindings) {
        std::erase_if(source.presampled_noise_sites,
                      [](const auto& site) { return *site.source_site % 2 == 0; });
    }
    const ExecutablePlan plan(source);
    REQUIRE(plan.num_batch_noise_carriers() > 0);
    REQUIRE(plan.peak_active_width() > 0);
    REQUIRE(plan.num_presampled_symbols() == 3 * rounds);
    REQUIRE(plan.num_unbound_presampled_symbols() == (mixed_bindings ? 3 * rounds / 2 : 0));
    REQUIRE(plan.num_readout_noise_sites() == rounds + 2);
    REQUIRE(plan.num_visible_records() == rounds + 4);
    REQUIRE(plan.num_exp_vals() == 1);

    constexpr uint32_t capacity = 129;
    constexpr uint32_t shots = 4096;
    BatchExternalInputs inputs(plan, capacity);
    BatchExecutor batch(plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    const auto root = clifft::make_seed_root(shots, uint64_t{627});
    const auto noise = [](uint32_t input, uint32_t shot) {
        return (((shot * 37 + input * 11) >> (input % 8)) & 1) != 0;
    };
    const auto channel = [](uint32_t site, uint32_t shot) {
        return ((shot >> (site % 7)) + site) % 4;
    };
    uint32_t random_flips = 0;
    uint32_t random_shots = 0;
    std::array<uint32_t, 2> feedback_counts{};
    for (uint32_t first = 0; first < shots; first += capacity) {
        const uint32_t count = std::min(capacity, shots - first);
        inputs.clear();
        for (uint32_t lane = 0; lane < count; ++lane) {
            for (uint32_t input = 0; input < plan.num_presampled_symbols(); ++input) {
                set(inputs.noise(input), lane, noise(input, first + lane));
            }
            for (uint32_t site = 0; site < plan.num_readout_noise_sites(); ++site) {
                set_channel(inputs, site, lane, channel(site, first + lane));
            }
        }
        batch.run_batch(root, first, count, inputs);
        REQUIRE(batch.surviving_shots() == count);
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint32_t shot = first + lane;
            CAPTURE(shot);
            bool quantum = false;
            bool feedback = false;
            for (uint32_t round = 0; round < rounds; ++round) {
                quantum ^= noise(3 * round, shot) ^ noise(3 * round + 1, shot);
                const bool record = reported(quantum, channel(round, shot));
                REQUIRE(batch.measurement(lane, round) == record);
                feedback ^= record;
            }
            REQUIRE(batch.measurement(lane, rounds) == reported(quantum, channel(rounds, shot)));
            REQUIRE_FALSE(batch.measurement(lane, rounds + 1));
            // Feedback prepares |0> or |1>. H T H rotates its Bloch vector
            // by pi/4 about X, leaving Z expectation +/- 1/sqrt(2).
            ++feedback_counts[feedback];
            REQUIRE_THAT(batch.exp_val(lane, 0),
                         Catch::Matchers::WithinAbs((feedback ? -1 : 1) * std::sqrt(0.5), 1e-12));
            const uint32_t final_channel = channel(rounds + 1, shot);
            const bool record = batch.measurement(lane, rounds + 2);
            if (final_channel < 2) {
                random_flips += record != reported(feedback, final_channel);
                ++random_shots;
            } else {
                REQUIRE(record == reported(false, final_channel));
            }
            REQUIRE(batch.measurement(lane, rounds + 3) == record);
            REQUIRE_FALSE(batch.detector(lane, 0));
            REQUIRE(batch.observable(lane, 0) == record);
        }
    }
    REQUIRE(feedback_counts[0] > shots / 4);
    REQUIRE(feedback_counts[1] > shots / 4);
    REQUIRE(random_shots == shots / 2);
    REQUIRE_THAT(static_cast<double>(random_flips) / random_shots,
                 Catch::Matchers::WithinAbs((1 - std::sqrt(0.5)) / 2, 0.04));
}

TEST_CASE("Packed external noise and readout consume no measurement randomness") {
    const auto noisy = compile("DEPOLARIZE1(0.75) 0\nM(0.4) 0\nH 1\nM 1\n");
    const auto clean = compile("M 0\nH 1\nM 1\n");
    constexpr uint32_t shots = 129;
    BatchExternalInputs inputs(noisy, shots);
    BatchExecutor external(noisy, shots, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    BatchExecutor reference(clean, shots);
    const auto root = clifft::make_seed_root(shots, uint64_t{914});
    for (uint32_t input = 0; input < noisy.num_presampled_symbols(); ++input) {
        std::ranges::fill(inputs.noise(input), ~uint64_t{0});
    }
    std::ranges::fill(inputs.readout_zero(0), ~uint64_t{0});
    external.run_batch(root, 0, shots, inputs);
    reference.run_batch(root, 0, shots);
    for (uint32_t lane = 0; lane < shots; ++lane) {
        REQUIRE(external.measurement(lane, 0));
        REQUIRE(external.measurement(lane, 1) == reference.measurement(lane, 1));
    }
}

TEST_CASE("Packed external readout follows survivors through intermediate compaction") {
    std::string circuit = "X_ERROR(0.8) 0\nM 0\nDETECTOR rec[-1]\nH 1\n";
    for (uint32_t probe = 0; probe < 96; ++probe) {
        circuit += "T 1\nEXP_VAL X1\nH 1\n";
    }
    circuit += "X_ERROR(0.1) 2\nM(0.1) 2\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    const auto hir = clifft::trace(clifft::parse(circuit));
    const std::array<uint8_t, 1> mask{1};
    const ExecutablePlan plan(plan_sampling(
        hir, {.postselection_mask = mask, .expected_detectors = {}, .expected_observables = {}}));
    REQUIRE(plan.peak_active_width() > 0);
    REQUIRE(plan.num_presampled_symbols() == 2);
    REQUIRE(plan.num_readout_noise_sites() == 1);
    constexpr uint32_t capacity = 129;
    BatchExternalInputs inputs(plan, capacity);
    BatchExecutor batch(plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    const auto root = clifft::make_seed_root(1024, uint64_t{61});
    for (uint32_t shots : {capacity, 65u, 0u, capacity}) {
        CAPTURE(shots);
        inputs.clear();
        // Inactive lanes and word padding must not enter the compacted outputs.
        std::ranges::fill(inputs.readout_zero(0), ~uint64_t{0});
        std::ranges::fill(inputs.readout_one(0), ~uint64_t{0});
        for (uint32_t lane = 0; lane < shots; ++lane) {
            set(inputs.noise(0), lane, lane % 8 != 0);
            set(inputs.noise(1), lane, ((lane / 8) & 1) != 0);
            set_channel(inputs, 0, lane, (lane / 16) % 4);
        }
        const std::vector<uint64_t> before_zero(inputs.readout_zero(0).begin(),
                                                inputs.readout_zero(0).end());
        const std::vector<uint64_t> before_one(inputs.readout_one(0).begin(),
                                               inputs.readout_one(0).end());
        batch.run_batch(root, 100, shots, inputs);
        REQUIRE(batch.compacted_during_execution() == (shots != 0));
        REQUIRE(batch.surviving_shots() == (shots + 7) / 8);
        for (uint32_t lane = 0; lane < batch.surviving_shots(); ++lane) {
            const uint32_t original = 8 * lane;
            REQUIRE(batch.shot_index(lane) == 100 + original);
            REQUIRE_FALSE(batch.measurement(lane, 0));
            REQUIRE_FALSE(batch.detector(lane, 0));
            const bool expected = reported((lane & 1) != 0, (original / 16) % 4);
            REQUIRE(batch.measurement(lane, 1) == expected);
            REQUIRE(batch.observable(lane, 0) == expected);
        }
        REQUIRE(std::ranges::equal(inputs.readout_zero(0), before_zero));
        REQUIRE(std::ranges::equal(inputs.readout_one(0), before_one));
    }
}

TEST_CASE("Packed external aggregate counts match analytic survivors and row outputs") {
    const uint32_t reject_every = GENERATE(0u, 3u, 1u);
    CAPTURE(reject_every);
    const auto hir =
        clifft::trace(clifft::parse("X_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\n"
                                    "X_ERROR(0.2) 1\nM(0.3) 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                                    "X_ERROR(0.4) 2\nM(0.5) 2\nOBSERVABLE_INCLUDE(1) rec[-1]\n"));
    const std::array<uint8_t, 1> mask{1};
    const ExecutablePlan plan(plan_sampling(hir, {.postselection_mask = mask}));
    REQUIRE(plan.peak_active_width() == 0);
    REQUIRE(plan.num_presampled_symbols() == 3);
    REQUIRE(plan.num_readout_noise_sites() == 2);
    REQUIRE(plan.num_observables() == 2);
    constexpr uint32_t capacity = 129;
    BatchExternalInputs inputs(plan, capacity);
    BatchExecutor rows(plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    BatchExecutor aggregate(plan, capacity, BatchOutputMode::AggregateSurvivors,
                            BatchSamplingMode::ExternalInputs);
    const auto root = clifft::make_seed_root(1024, uint64_t{63});
    for (uint32_t shots : {capacity, 65u, 0u, 1u, capacity}) {
        CAPTURE(shots);
        inputs.clear();
        uint32_t expected_survivors = 0;
        uint32_t expected_errors = 0;
        std::array<uint64_t, 2> expected_ones{};
        for (uint32_t lane = 0; lane < shots; ++lane) {
            const bool rejected = reject_every != 0 && lane % reject_every == 0;
            set(inputs.noise(0), lane, rejected);
            bool any_observable = false;
            for (uint32_t site = 0; site < 2; ++site) {
                const bool source = ((lane >> site) & 1) != 0;
                const uint32_t channel = ((lane >> (site + 1)) + site) % 4;
                set(inputs.noise(site + 1), lane, source);
                set_channel(inputs, site, lane, channel);
                const bool outcome = reported(source, channel);
                expected_ones[site] += !rejected && outcome;
                any_observable |= outcome;
            }
            expected_survivors += !rejected;
            expected_errors += !rejected && any_observable;
        }
        rows.run_batch(root, 100, shots, inputs);
        aggregate.run_batch(root, 100, shots, inputs);
        // Aggregation must count a sparse live mask without final row compaction.
        REQUIRE_FALSE(aggregate.compacted_during_execution());
        REQUIRE(rows.surviving_shots() == expected_survivors);
        REQUIRE(aggregate.surviving_shots() == expected_survivors);
        std::array<uint64_t, 2> row_ones{};
        uint32_t row_errors = 0;
        for (uint32_t lane = 0; lane < rows.surviving_shots(); ++lane) {
            bool any_observable = false;
            for (uint32_t observable = 0; observable < 2; ++observable) {
                const bool outcome = rows.observable(lane, observable);
                row_ones[observable] += outcome;
                any_observable |= outcome;
            }
            row_errors += any_observable;
        }
        std::array<uint64_t, 2> aggregate_ones{};
        REQUIRE(aggregate.accumulate_survivor_counts(aggregate_ones) == expected_errors);
        REQUIRE(row_errors == expected_errors);
        REQUIRE(row_ones == expected_ones);
        REQUIRE(aggregate_ones == expected_ones);
    }
}

TEST_CASE("Packed external storage participates in worker budgets") {
    const auto plan = compile("X_ERROR(0.5) 0\nM(0.5) 0\n");
    constexpr uint32_t capacity = 129;
    const auto ordinary = batch_detail::batch_worker_storage_bytes(
        plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::Ordinary);
    const auto external = batch_detail::batch_worker_storage_bytes(
        plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    REQUIRE(external == ordinary + BatchExternalInputs::storage_bytes(plan, capacity) +
                            packed_bit_columns_storage_bytes(2, capacity));
}

TEST_CASE("Packed external readout buffers constrain automatic capacity and workers") {
    const auto plan = compile("REPEAT 6000 {\nM(0.5) 0\n}\n");
    REQUIRE(plan.num_readout_noise_sites() == 6000);
    const auto ordinary = resolve_batch_execution_policy(plan, 65536, 64, 1, BatchOutputMode::Rows,
                                                         std::nullopt, BatchSamplingMode::Ordinary);
    const auto external = resolve_batch_execution_policy(
        plan, 65536, 64, 1, BatchOutputMode::Rows, std::nullopt, BatchSamplingMode::ExternalInputs);
    REQUIRE(external.lane_capacity > 1);
    REQUIRE(external.lane_capacity < ordinary.lane_capacity);
    const uint64_t bytes = batch_detail::batch_worker_storage_bytes(
        plan, external.lane_capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    REQUIRE(bytes <= kDefaultBatchWorkerBudget);
    REQUIRE(bytes * external.worker_count <= kDefaultBatchTotalWorkerBudget);
    REQUIRE(bytes * (external.worker_count + 1) > kDefaultBatchTotalWorkerBudget);
}

TEST_CASE("Packed external mode accepts empty inputs and readout only plans") {
    const bool with_readout = GENERATE(false, true);
    const auto plan = compile(with_readout ? "M(0.5) 0\n" : "H 0\nM 0\n");
    REQUIRE(plan.num_presampled_symbols() == 0);
    constexpr uint32_t shots = 65;
    BatchExternalInputs inputs(plan, shots);
    BatchExecutor batch(plan, shots, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    if (with_readout) {
        for (uint32_t lane = 0; lane < shots; ++lane) {
            set_channel(inputs, 0, lane, lane % 4);
        }
    }
    const auto root = clifft::make_seed_root(shots, uint64_t{90});
    batch.run_batch(root, 0, shots, inputs);
    REQUIRE(batch.surviving_shots() == shots);
    BatchExecutor reference(plan, shots);
    reference.run_batch(root, 0, shots);
    for (uint32_t lane = 0; lane < shots; ++lane) {
        REQUIRE(batch.measurement(lane, 0) ==
                (with_readout ? reported(false, lane % 4) : reference.measurement(lane, 0)));
    }
}

TEST_CASE("Packed unbound symbols require external input mode") {
    auto source = plan_sampling(clifft::trace(clifft::parse("X_ERROR(0.5) 0\nM 0\n")));
    source.presampled_noise_sites.clear();
    const ExecutablePlan plan(source);
    REQUIRE(plan.num_unbound_presampled_symbols() == 1);
    REQUIRE_THROWS_AS(BatchExecutor(plan, 65), std::invalid_argument);
    REQUIRE_THROWS_AS(
        BatchExecutor(plan, 65, BatchOutputMode::Rows, BatchSamplingMode::FixedFaults),
        std::invalid_argument);
    REQUIRE_NOTHROW(
        BatchExecutor(plan, 65, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs));
}

TEST_CASE("Packed driver bit columns participate in capacity and worker budgets") {
    const auto plan = compile("X_ERROR(0.5) 0\nM(0.5) 0\n");
    constexpr uint64_t columns = 40001;
    const auto policy =
        resolve_batch_execution_policy(plan, 65536, 64, 1, BatchOutputMode::Rows, std::nullopt,
                                       BatchSamplingMode::ExternalInputs, 17, columns);
    REQUIRE(policy.lane_capacity == 1024);
    const auto bytes =
        batch_detail::batch_worker_storage_bytes(plan, policy.lane_capacity, BatchOutputMode::Rows,
                                                 BatchSamplingMode::ExternalInputs) +
        17 + packed_bit_columns_storage_bytes(columns, policy.lane_capacity);
    CHECK(bytes <= kDefaultBatchWorkerBudget);
    CHECK(bytes * policy.worker_count <= kDefaultBatchTotalWorkerBudget);
    CHECK(bytes * (policy.worker_count + 1) > kDefaultBatchTotalWorkerBudget);
    const auto serial =
        resolve_batch_execution_policy(plan, 65536, 1, 1, BatchOutputMode::Rows, std::nullopt,
                                       BatchSamplingMode::ExternalInputs, 17, columns);
    CHECK(serial.lane_capacity == policy.lane_capacity);
}

TEST_CASE("Packed driver page rounding changes capacity and worker budget decisions") {
    const auto plan = compile("X_ERROR(0.5) 0\nM(0.5) 0\n");
    constexpr uint32_t capacity = 2048;
    constexpr uint64_t columns = 7;
    const uint64_t logical = columns * packed_word_count(capacity) * sizeof(uint64_t);
    const uint64_t allocated = packed_bit_columns_storage_bytes(columns, capacity);
    REQUIRE(allocated > logical);
    const auto executor_bytes = batch_detail::batch_worker_storage_bytes(
        plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::ExternalInputs);
    const bool aggregate_limit = GENERATE(false, true);
    CAPTURE(aggregate_limit);
    const uint64_t extra =
        (aggregate_limit ? kDefaultBatchTotalWorkerBudget / 9 : kDefaultBatchWorkerBudget) -
        executor_bytes - logical;
    const auto policy =
        resolve_batch_execution_policy(plan, 65536, 64, 1, BatchOutputMode::Rows, std::nullopt,
                                       BatchSamplingMode::ExternalInputs, extra, columns);
    REQUIRE(policy.lane_capacity > 1);
    if (!aggregate_limit) {
        CHECK(policy.lane_capacity < capacity);
    } else {
        CHECK(policy.lane_capacity == capacity);
        CHECK(policy.worker_count == 8);
    }
    const uint64_t actual =
        batch_detail::batch_worker_storage_bytes(plan, policy.lane_capacity, BatchOutputMode::Rows,
                                                 BatchSamplingMode::ExternalInputs) +
        extra + packed_bit_columns_storage_bytes(columns, policy.lane_capacity);
    CHECK(actual <= kDefaultBatchWorkerBudget);
    CHECK(actual * policy.worker_count <= kDefaultBatchTotalWorkerBudget);
}
