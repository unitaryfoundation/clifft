#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/global_reconstruction.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/sampler.h"
#include "clifft/tableau/tableau.h"

#include "dense_circuit_oracle.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace clifft::test {
namespace {

std::vector<std::string> reconstruction_law_cases() {
    return {
        "",
        "DETECTOR\nOBSERVABLE_INCLUDE(2)\n",
        "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "H 0 1\nT 0\nT_DAG 1\nCX 0 1\nS 0\nMPP Y0*X1\nM 0 1\nDETECTOR rec[-3] rec[-1]\n",
        "H 0\nCX 0 1\nT 0\nR 0\nH 0\nT_DAG 0\nM 0 1\nDETECTOR rec[-2] rec[-1]\n",
        "H 0\nT 0\nMRX 0\nRY 1\nCX 0 1\nT_DAG 1\nMRY 1\nM 0 1\n",
        "H 0\nT 0\nM(0.2) !0\nCX rec[-1] 1\nH 1\nT 1\nMX 1\nDETECTOR rec[-1] rec[-2]\n",
        "H 0\nPAULI_CHANNEL_1(0.1,0.2,0.3) 0\nT 0\nMX 0\nDETECTOR rec[-1]\n",
        "H 0 1\nT 0\nDEPOLARIZE2(0.4) 0 1\nT_DAG 1\nMPP !X0*Y1\nM 0 1\n",
        "H 0 1\nT 0\nPAULI_CHANNEL_2(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0,0,0,0,0) 0 "
        "1\nT_DAG 1\nMPP X0*Y1\nM 0 1\n",
        "H 0 1\nT 0\nE(0.2) X0 Y1\nELSE_CORRELATED_ERROR(0.3) Z0\nT_DAG 1\nMX 0 1\nDETECTOR "
        "rec[-1] rec[-2]\n",
        "H 0\nT 0\nM 0\nDETECTOR rec[-1]\nREADOUT_NOISE(0.25) rec[-1]\nCX rec[-1] 1\nM "
        "1\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "H 0\nT 0\nMX 0\nREPEAT 2 {\nCZ rec[-1] 0\nT_DAG 0\nMX 0\n}\nDETECTOR rec[-3] rec[-1]\n",
    };
}

std::map<uint64_t, double> output_law(const ExactPlanDistribution& distribution,
                                      const ReconstructionOutputLayout& layout) {
    std::map<uint64_t, double> law;
    for (const auto& e : distribution.kept_outcomes) {
        const uint64_t parities =
            e.outcome.detectors | (e.outcome.observables << layout.num_detectors);
        const uint64_t records = layout.num_records ? e.outcome.visible_records : 0;
        law[records | (parities << layout.num_records)] += e.probability;
    }
    return law;
}

void require_same_law(std::map<uint64_t, double> actual,
                      const std::map<uint64_t, double>& expected) {
    for (const auto& [key, p] : expected) {
        REQUIRE_THAT(actual[key], Catch::Matchers::WithinAbs(p, 2e-12));
        actual.erase(key);
    }
    for (const auto& [key, p] : actual)
        REQUIRE_THAT(p, Catch::Matchers::WithinAbs(0, 2e-12));
}

std::map<uint64_t, double> reconstructed_law(const GlobalReconstruction& r) {
    using namespace detail_dense_oracle;
    REQUIRE(r.num_qubits <= 10);
    REQUIRE(r.num_noise_bits < 64);
    std::map<uint64_t, double> faults{{0, 1}};
    for (const auto& channel : r.noise_channels) {
        std::map<uint64_t, double> next;
        for (const auto& [assignment, p] : faults) {
            for (const auto& outcome : channel) {
                uint64_t mask = assignment;
                for (uint32_t b : outcome.bits)
                    mask ^= uint64_t{1} << b;
                next[mask] += p * outcome.probability;
            }
        }
        faults = std::move(next);
    }
    uint32_t count = 0;
    for (const auto& op : r.operations)
        if (op.output)
            count = std::max(count, *op.output + 1);
    std::map<uint64_t, double> law;
    double omitted_probability_bound = 0.0;
    for (const auto& [fault, probability] : faults) {
        if (probability == 0)
            continue;
        TrajectoryBranch initial;
        initial.psi.resize(size_t{1} << r.num_qubits);
        initial.psi[0] = 1;
        initial.records.resize(count);
        initial.probability = probability;
        std::vector<TrajectoryBranch> branches{initial};
        for (const auto& op : r.operations) {
            const uint64_t x = r.num_qubits ? op.axis.x().words[0] : 0;
            const uint64_t z = r.num_qubits ? op.axis.z().words[0] : 0;
            const uint64_t terms = op.sign.words.empty() ? 0 : op.sign.words[0];
            const bool sign = op.sign.constant ^ ((std::popcount(terms & fault) & 1) != 0);
            if (op.half_turns != 0) {
                for (auto& b : branches)
                    apply_pauli_rotation(b.psi, r.num_qubits, {x, z, sign}, op.half_turns);
            } else {
                std::vector<TrajectoryBranch> next;
                for (auto& b : branches) {
                    b.next_meas = op.output.value_or(0);
                    branch_projective_measurement(b, r.num_qubits, {x, z, false},
                                                  op.output.has_value(), sign, nullptr, next, 1e-15,
                                                  omitted_probability_bound);
                }
                branches = std::move(next);
            }
        }
        for (const auto& b : branches) {
            uint64_t value = 0;
            for (uint32_t j = 0; j < count; ++j)
                value |= uint64_t{b.records[j]} << j;
            law[value] += b.probability;
        }
    }
    REQUIRE(omitted_probability_bound <= 1e-12);
    return law;
}

void check_joint_law(const std::string& text) {
    const auto circuit = parse(text);
    const auto expected = compute_dense_circuit_distribution(circuit);
    for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
        const auto result = reconstruct_global_sampling(circuit, mode);
        INFO(result.fallback_reason);
        REQUIRE(result.reconstruction.has_value());
        const auto& r = *result.reconstruction;
        REQUIRE(r.outputs.num_records ==
                (mode == ReconstructionOutputs::Records ? circuit.num_measurements : 0));
        REQUIRE(r.outputs.num_detectors == circuit.num_detectors);
        REQUIRE(r.outputs.num_observables == circuit.num_observables);
        require_same_law(reconstructed_law(r), output_law(expected, r.outputs));
    }
}

using SignLaw = std::map<std::vector<uint32_t>, double>;

// Enumerates categorical fault outcomes and evaluates each operation sign
// directly without using the compiler's binary basis or channel projection.
SignLaw sign_law(uint32_t num_bits, const std::vector<ReconstructionSign>& signs,
                 const std::vector<std::vector<ReconstructionNoiseOutcome>>& channels) {
    SignLaw faults{{std::vector<uint32_t>(num_bits), 1}};
    for (const auto& channel : channels) {
        SignLaw next;
        for (const auto& [assignment, probability] : faults) {
            for (const auto& outcome : channel) {
                if (outcome.probability == 0)
                    continue;
                auto value = assignment;
                for (uint32_t b : outcome.bits)
                    value.at(b) ^= 1;
                next[value] += probability * outcome.probability;
            }
        }
        faults = std::move(next);
    }
    SignLaw law;
    for (const auto& [assignment, probability] : faults) {
        std::vector<uint32_t> value;
        for (const auto& sign : signs) {
            uint32_t parity = sign.constant;
            for (uint32_t b = 0; b < num_bits; ++b)
                if (b / 64 < sign.words.size() && ((sign.words[b / 64] >> (b % 64)) & 1))
                    parity ^= assignment[b];
            value.push_back(parity);
        }
        law[value] += probability;
    }
    return law;
}

ReconstructionSign make_sign(std::initializer_list<uint32_t> bits, bool constant = false) {
    ReconstructionSign result{{}, constant};
    for (uint32_t b : bits) {
        result.words.resize(std::max(result.words.size(), size_t{b / 64 + 1}));
        result.words[b / 64] ^= uint64_t{1} << (b % 64);
    }
    return result;
}

ReconstructionNoise check_sign_law(const GlobalReconstruction& r) {
    const auto result = compress_reconstruction_noise(r);
    INFO(result.fallback_reason);
    REQUIRE(result.noise.has_value());
    std::vector<ReconstructionSign> signs;
    for (const auto& op : r.operations)
        signs.push_back(op.sign);
    const auto expected = sign_law(r.num_noise_bits, signs, r.noise_channels);
    auto actual = sign_law(result.noise->num_bits, result.noise->signs, result.noise->channels);
    for (const auto& [value, p] : expected) {
        REQUIRE_THAT(actual[value], Catch::Matchers::WithinAbs(p, 2e-12));
        actual.erase(value);
    }
    for (const auto& [value, p] : actual)
        REQUIRE_THAT(p, Catch::Matchers::WithinAbs(0, 2e-12));
    return *result.noise;
}

}  // namespace

TEST_CASE("Global reconstruction preserves complete noisy output laws", "[reconstruction]") {
    for (const auto& text : reconstruction_law_cases()) {
        INFO(text);
        check_joint_law(text);
    }
}

TEST_CASE("Global reconstruction folds constraints across machine words", "[reconstruction]") {
    std::string text;
    for (uint32_t i = 0; i < 70; ++i)
        text += "R " + std::to_string(i) + "\n";
    text += "H 69\nT 69\nMX 69\nDETECTOR rec[-1]\n";
    const auto r = reconstruct_global_sampling(parse(text), ReconstructionOutputs::Records);
    REQUIRE(r.reconstruction.has_value());
    REQUIRE(r.reconstruction->dilation_qubits > 128);
    const auto law = reconstructed_law(*r.reconstruction);
    const double zero = (1 + std::sqrt(0.5)) / 2;
    REQUIRE_THAT(law.at(0), Catch::Matchers::WithinAbs(zero, 2e-12));
    REQUIRE_THAT(law.at(3), Catch::Matchers::WithinAbs(1 - zero, 2e-12));
}

TEST_CASE("Global reconstruction preserves padded constants", "[reconstruction]") {
    for (const auto* padding : {"1 0 1", "!0 !1 !0"}) {
        CAPTURE(padding);
        const auto result = reconstruct_global_sampling(
            parse(std::string("MPAD ") + padding + "\nDETECTOR rec[-1] rec[-2]\n"),
            ReconstructionOutputs::Records);
        REQUIRE(result.reconstruction.has_value());
        const auto law = reconstructed_law(*result.reconstruction);
        REQUIRE(law.size() == 1);
        REQUIRE_THAT(law.at(13), Catch::Matchers::WithinAbs(1, 2e-12));
    }
}

TEST_CASE("Global reconstruction matches mixed small instruments", "[reconstruction]") {
    Xoshiro256PlusPlus rng(34972);
    const std::vector<std::string> single{"H", "S", "S_DAG", "T",  "T_DAG", "X",
                                          "Y", "Z", "R",     "RX", "RY"};
    const std::vector<std::string> pairs{"CX", "CY", "CZ", "SWAP"};
    const std::vector<std::string> measurements{"M", "MX", "MY", "MR", "MRX", "MRY"};
    for (uint32_t trial = 0; trial < 32; ++trial) {
        std::string text = "H 0\nT 0\n";
        uint32_t records = 0;
        for (uint32_t step = 0; step < 12; ++step) {
            const auto q = rng() % 3;
            const auto choice = rng() % 5;
            if (choice < 2) {
                text += single[rng() % single.size()] + " " + std::to_string(q) + "\n";
            } else if (choice == 2) {
                text += pairs[rng() % pairs.size()] + " " + std::to_string(q) + " " +
                        std::to_string((q + 1) % 3) + "\n";
            } else if (choice == 3 || records == 0) {
                text += measurements[rng() % measurements.size()] + " " + (rng() % 2 ? "!" : "") +
                        std::to_string(q) + "\n";
                ++records;
            } else {
                text += "CZ rec[-1] " + std::to_string(q) + "\n";
            }
            if (step == 5)
                text += "Y_ERROR(0.27) 1\n";
        }
        text += "M 0 1 2\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\n";
        INFO(text);
        check_joint_law(text);
    }
}

TEST_CASE("Global reconstruction rejects unsupported preservation requirements",
          "[reconstruction]") {
    for (const auto& text : {"H 0\nT 0\nEXP_VAL X0\n", "R_X(0.13) 0\nM 0\n",
                             "M 0\nREADOUT_NOISE(0.1,0.2) rec[-1]\n"}) {
        const auto result =
            reconstruct_global_sampling(parse(text), ReconstructionOutputs::Records);
        REQUIRE_FALSE(result.reconstruction);
        REQUIRE_FALSE(result.fallback_reason.empty());
    }
    auto limits = ReconstructionLimits{};
    limits.max_dilation_qubits = 1;
    REQUIRE_FALSE(reconstruct_global_sampling(parse("H 0\nT 0\nM 0\n"),
                                              ReconstructionOutputs::Records, limits)
                      .reconstruction);
    limits = {};
    limits.max_noise_bits = 1;
    REQUIRE_FALSE(reconstruct_global_sampling(parse("DEPOLARIZE1(0.1) 0\nM 0\n"),
                                              ReconstructionOutputs::Records, limits)
                      .reconstruction);
    const auto source = parse("H 0\nT 0\nZ_ERROR(0.2) 0\nZ_ERROR(0.3) 0\nMX 0\nDETECTOR rec[-1]\n");
    for (uint32_t limit = 0; limit < 3; ++limit) {
        limits = {};
        if (limit == 0)
            limits.max_outputs = 0;
        if (limit == 1)
            limits.max_schedule_operations = 0;
        if (limit == 2)
            limits.max_planning_steps = 0;
        const auto result =
            reconstruct_global_sampling(source, ReconstructionOutputs::Records, limits);
        REQUIRE_FALSE(result.reconstruction);
        REQUIRE_FALSE(result.fallback_reason.empty());
    }
}

TEST_CASE("Noise compression preserves correlated affine sign laws", "[reconstruction]") {
    GlobalReconstruction r;
    r.num_noise_bits = 130;
    r.operations = {{PauliString(0), make_sign({0, 65}, true), 0.25, std::nullopt},
                    {PauliString(0), make_sign({65, 129}), 0, 0U},
                    {PauliString(0), make_sign({0, 129}, true), 0, 1U}};
    r.noise_channels = {{{0.25, {}}, {0.25, {0}}, {0.5, {65}}, {0, {129}}},
                        {{0.5, {}}, {0.25, {0}}, {0.25, {65}}},
                        {{1, {10}}},
                        {{1, {129}}}};
    const auto noise = check_sign_law(r);
    REQUIRE(noise.num_bits == 2);
    REQUIRE(noise.channels.size() == 2);
}

TEST_CASE("Noise compression preserves signs across multiple output words", "[reconstruction]") {
    GlobalReconstruction r;
    r.num_noise_bits = 70;
    for (uint32_t i = 0; i < 70; ++i)
        r.operations.push_back({PauliString(0), make_sign({i}, i % 2 != 0), 0, i});
    r.noise_channels = {{{0.1, {}}, {0.2, {0, 65}}, {0.7, {69}}}, {{0.4, {}}, {0.6, {65, 69}}}};
    const auto noise = check_sign_law(r);
    REQUIRE(noise.num_bits == 70);
}

TEST_CASE("Noise compression removes unobserved channels and retains constants",
          "[reconstruction]") {
    GlobalReconstruction r;
    r.num_noise_bits = 2;
    r.operations = {{PauliString(0), make_sign({}, true), 0, 0U},
                    {PauliString(0), make_sign({}), 0, 1U}};
    r.noise_channels = {{{0.3, {}}, {0.7, {0}}}, {{1, {1}}}};
    const auto noise = check_sign_law(r);
    REQUIRE(noise.num_bits == 0);
    REQUIRE(noise.channels.empty());
    REQUIRE(noise.signs[0].constant);
    REQUIRE_FALSE(noise.signs[1].constant);
}

TEST_CASE("Noise compression reports channel work and support limits", "[reconstruction]") {
    GlobalReconstruction r;
    r.num_noise_bits = 1;
    r.operations = {{PauliString(0), make_sign({0}), 0, 0U}};
    r.noise_channels = {{{0.3, {}}, {0.7, {0}}}, {{0.6, {}}, {0.4, {0}}}};
    for (bool support : {false, true}) {
        ReconstructionLimits limits;
        if (support)
            limits.max_channel_support = 1;
        else
            limits.max_channel_products = 0;
        const auto result = compress_reconstruction_noise(r, limits);
        REQUIRE_FALSE(result.noise);
        REQUIRE(result.fallback_reason == (support ? "channel support limit exceeded"
                                                   : "channel convolution work limit exceeded"));
    }
    check_sign_law(r);

    // Wide categorical channels on 11 independent coordinates (2048 outcomes each)
    // must abort during projection when max_channel_support is small, and before
    // allocating the 2048 x 2048 product when max_channel_products is exceeded.
    GlobalReconstruction wide;
    wide.num_noise_bits = 11;
    for (uint32_t b = 0; b < 11; ++b) {
        wide.operations.push_back({PauliString(0), make_sign({b}), 0, b});
    }
    std::vector<ReconstructionNoiseOutcome> full_subspace;
    full_subspace.reserve(2048);
    for (uint32_t mask = 0; mask < 2048; ++mask) {
        std::vector<uint32_t> bits;
        for (uint32_t b = 0; b < 11; ++b) {
            if (((mask >> b) & 1U) != 0) {
                bits.push_back(b);
            }
        }
        full_subspace.push_back({1.0 / 2048.0, std::move(bits)});
    }
    wide.noise_channels = {full_subspace, full_subspace};

    {
        ReconstructionLimits limits;
        limits.max_channel_support = 16;
        const auto result = compress_reconstruction_noise(wide, limits);
        REQUIRE_FALSE(result.noise);
        REQUIRE(result.fallback_reason == "channel support limit exceeded");
    }
    {
        ReconstructionLimits limits;
        limits.max_channel_support = 4096;
        limits.max_channel_products = 100;
        const auto result = compress_reconstruction_noise(wide, limits);
        REQUIRE_FALSE(result.noise);
        REQUIRE(result.fallback_reason == "channel convolution work limit exceeded");
    }

    // Two 65-outcome singleton channels share a key of size 65 (within support
    // limit 100), while their pairwise XORs expand to 2081 distinct outcomes and
    // must stop as soon as convolved support exceeds 100.
    GlobalReconstruction expanding;
    expanding.num_noise_bits = 64;
    std::vector<ReconstructionNoiseOutcome> singletons = {{0.36, {}}};
    for (uint32_t b = 0; b < 64; ++b) {
        expanding.operations.push_back({PauliString(0), make_sign({b}), 0, b});
        singletons.push_back({0.01, {b}});
    }
    expanding.noise_channels = {singletons, singletons};
    {
        ReconstructionLimits limits;
        limits.max_channel_support = 100;
        limits.max_channel_products = 10000;
        const auto result = compress_reconstruction_noise(expanding, limits);
        REQUIRE_FALSE(result.noise);
        REQUIRE(result.fallback_reason == "channel support limit exceeded");
    }
}

TEST_CASE("Reconstructed HIR preserves complete noisy output laws", "[reconstruction]") {
    for (const auto& text : reconstruction_law_cases()) {
        INFO(text);
        const auto circuit = parse(text);
        const auto expected = compute_dense_circuit_distribution(circuit);
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto r = reconstruct_global_sampling(circuit, mode);
            REQUIRE(r.reconstruction.has_value());
            auto lowered = lower_global_reconstruction(*r.reconstruction);
            INFO(lowered.fallback_reason);
            REQUIRE(lowered.hir.has_value());
            for (bool optimize : {false, true}) {
                if (optimize)
                    default_hir_pass_manager().run(*lowered.hir);
                const auto actual =
                    compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
                require_same_law(output_law(actual, r.reconstruction->outputs),
                                 output_law(expected, r.reconstruction->outputs));
            }
        }
    }
}

TEST_CASE("Reconstructed output snapshots do not require the source circuit", "[reconstruction]") {
    for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
        const auto r =
            reconstruct_global_sampling(parse("M 0\nDETECTOR rec[-1]\nREADOUT_NOISE(0.25) "
                                              "rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-1]\n"),
                                        mode);
        REQUIRE(r.reconstruction.has_value());
        const auto& layout = r.reconstruction->outputs;
        REQUIRE(layout.num_records == (mode == ReconstructionOutputs::Records ? 1u : 0u));
        REQUIRE(layout.num_detectors == 1);
        REQUIRE(layout.num_observables == 3);
        const auto lowered = lower_global_reconstruction(*r.reconstruction);
        REQUIRE(lowered.hir.has_value());
        const auto actual = compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
        // The detector saw zero before readout noise. The final record and third
        // observable see the same flip, while the undeclared observables stay zero.
        const uint64_t flipped = mode == ReconstructionOutputs::Records ? 17 : 8;
        require_same_law(output_law(actual, layout), {{0, 0.75}, {flipped, 0.25}});
    }
}

TEST_CASE("Global reconstruction lowering preserves reference and rejection mass",
          "[reconstruction]") {
    const auto source = parse(
        "H 0\nT 0\nMX 0\nX_ERROR(0.2) 1\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] "
        "rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-2]\n");
    const auto expected_detectors = std::vector<uint8_t>{1, 0};
    const auto expected_observables = std::vector<uint8_t>{1, 1};
    for (const auto& mask :
         {std::vector<uint8_t>{0, 0}, std::vector<uint8_t>{1, 0}, std::vector<uint8_t>{1, 1}}) {
        const auto expected = compute_dense_circuit_distribution(
            source, {mask, expected_detectors, expected_observables});
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto r = reconstruct_global_sampling(source, mode);
            REQUIRE(r.reconstruction.has_value());
            PostselectionMetadata meta;
            meta.mask = mask;
            meta.expected_detectors = expected_detectors;
            meta.expected_observables = expected_observables;
            const auto lowered = lower_global_reconstruction(*r.reconstruction, meta);
            REQUIRE(lowered.hir.has_value());
            const auto actual =
                compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
            REQUIRE_THAT(actual.discarded_probability,
                         Catch::Matchers::WithinAbs(expected.discarded_probability, 2e-12));
            std::map<uint64_t, double> want, got;
            auto key = [&](const ExactShotOutcome& o) {
                return (mode == ReconstructionOutputs::Records ? o.visible_records : 0) |
                       (o.detectors << 2) | (o.observables << 4);
            };
            for (const auto& e : expected.kept_outcomes)
                want[key(e.outcome)] += e.probability;
            for (const auto& e : actual.kept_outcomes)
                got[key(e.outcome)] += e.probability;
            for (auto [k, p] : want) {
                REQUIRE_THAT(got[k], Catch::Matchers::WithinAbs(p, 2e-12));
                got.erase(k);
            }
            for (auto [k, p] : got)
                REQUIRE_THAT(p, Catch::Matchers::WithinAbs(0, 2e-12));
        }
    }
}

TEST_CASE("Reconstruction lowering reports bounded preparation failures", "[reconstruction]") {
    const auto source = parse("H 0\nT 0\nZ_ERROR(0.2) 0\nZ_ERROR(0.3) 0\nMX 0\nDETECTOR rec[-1]\n");
    ReconstructionLimits limits;
    const auto r = reconstruct_global_sampling(source, ReconstructionOutputs::Records);
    REQUIRE(r.reconstruction.has_value());
    for (uint32_t limit = 0; limit < 3; ++limit) {
        limits = {};
        if (limit == 0)
            limits.max_lowering_bytes = 0;
        if (limit == 1)
            limits.max_channel_products = 0;
        if (limit == 2)
            limits.max_channel_support = 0;
        const auto lowered = lower_global_reconstruction(*r.reconstruction, {}, limits);
        REQUIRE_FALSE(lowered.hir);
        REQUIRE_FALSE(lowered.fallback_reason.empty());
    }
}

TEST_CASE("GlobalSamplingReconstructor shares dilation across output variants",
          "[reconstruction]") {
    for (const auto& text : reconstruction_law_cases()) {
        INFO(text);
        const auto circuit = parse(text);
        const auto want_det =
            reconstruct_global_sampling(circuit, ReconstructionOutputs::Detectors);
        const auto want_rec = reconstruct_global_sampling(circuit, ReconstructionOutputs::Records);
        REQUIRE(want_det.reconstruction.has_value());
        REQUIRE(want_rec.reconstruction.has_value());

        for (bool detectors_first : {true, false}) {
            GlobalSamplingReconstructor shared(circuit);
            const auto first_mode =
                detectors_first ? ReconstructionOutputs::Detectors : ReconstructionOutputs::Records;
            const auto second_mode =
                detectors_first ? ReconstructionOutputs::Records : ReconstructionOutputs::Detectors;
            const auto first = shared.reconstruct(first_mode);
            const auto second = shared.reconstruct(second_mode);
            const auto repeat = shared.reconstruct(first_mode);

            const auto& got_det = detectors_first ? first : second;
            const auto& got_rec = detectors_first ? second : first;
            REQUIRE(got_det.reconstruction.has_value());
            REQUIRE(got_rec.reconstruction.has_value());
            REQUIRE(repeat.reconstruction.has_value());

            auto require_equal_reconstruction = [](const GlobalReconstruction& a,
                                                   const GlobalReconstruction& b) {
                REQUIRE(a.num_qubits == b.num_qubits);
                REQUIRE(a.dilation_qubits == b.dilation_qubits);
                REQUIRE(a.num_noise_bits == b.num_noise_bits);
                REQUIRE(a.peak_active_width == b.peak_active_width);
                REQUIRE(a.outputs.num_records == b.outputs.num_records);
                REQUIRE(a.outputs.num_detectors == b.outputs.num_detectors);
                REQUIRE(a.outputs.num_observables == b.outputs.num_observables);
                REQUIRE(a.operations.size() == b.operations.size());
                REQUIRE(a.noise_channels.size() == b.noise_channels.size());
                for (size_t i = 0; i < a.operations.size(); ++i) {
                    REQUIRE(a.operations[i].axis == b.operations[i].axis);
                    REQUIRE(a.operations[i].sign.constant == b.operations[i].sign.constant);
                    REQUIRE(a.operations[i].sign.words == b.operations[i].sign.words);
                    REQUIRE(a.operations[i].half_turns == b.operations[i].half_turns);
                    REQUIRE(a.operations[i].output == b.operations[i].output);
                }
            };
            require_equal_reconstruction(*got_det.reconstruction, *want_det.reconstruction);
            require_equal_reconstruction(*got_rec.reconstruction, *want_rec.reconstruction);
            require_equal_reconstruction(*repeat.reconstruction,
                                         *(detectors_first ? want_det : want_rec).reconstruction);
        }
    }

    // Preparation-stage rejections apply to both output variants, while schedule
    // budgets remain independent per output variant.
    {
        ReconstructionLimits tight;
        tight.max_dilation_qubits = 1;
        GlobalSamplingReconstructor rejected(parse("H 0\nT 0\nM 0\nDETECTOR rec[-1]\n"), tight);
        for (auto mode : {ReconstructionOutputs::Detectors, ReconstructionOutputs::Records}) {
            const auto res = rejected.reconstruct(mode);
            REQUIRE_FALSE(res.reconstruction);
            REQUIRE_FALSE(res.fallback_reason.empty());
        }
    }
    {
        const auto asymmetric = parse(
            "H 0\nT 0\nMX 0\nX_ERROR(0.1) 1\nM 1 2 3\nDETECTOR rec[-4]\nOBSERVABLE_INCLUDE(0) "
            "rec[-4]\n");
        const auto det_unbounded =
            reconstruct_global_sampling(asymmetric, ReconstructionOutputs::Detectors);
        const auto rec_unbounded =
            reconstruct_global_sampling(asymmetric, ReconstructionOutputs::Records);
        REQUIRE(det_unbounded.reconstruction.has_value());
        REQUIRE(rec_unbounded.reconstruction.has_value());
        REQUIRE(det_unbounded.reconstruction->operations.size() <
                rec_unbounded.reconstruction->operations.size());

        ReconstructionLimits limits;
        limits.max_schedule_operations = det_unbounded.reconstruction->operations.size() +
                                         det_unbounded.reconstruction->num_qubits;
        GlobalSamplingReconstructor shared(asymmetric, limits);
        const auto rec_bounded = shared.reconstruct(ReconstructionOutputs::Records);
        const auto det_bounded = shared.reconstruct(ReconstructionOutputs::Detectors);
        REQUIRE_FALSE(rec_bounded.reconstruction);
        REQUIRE_FALSE(rec_bounded.fallback_reason.empty());
        REQUIRE(det_bounded.reconstruction.has_value());
        require_same_law(reconstructed_law(*det_bounded.reconstruction),
                         reconstructed_law(*det_unbounded.reconstruction));
    }
}

TEST_CASE("Global reconstruction lowering omits identity hidden measurements", "[reconstruction]") {
    // Qubits 1..3 produce identity-axis outputs (fault-flipped M 1 and inverted
    // !2, !3), while qubit 0 exercises a genuine quantum rotation.
    const auto source = parse(
        "H 0\nT 0\nMX 0\nX_ERROR(0.25) 1\nM 1 !2 !3\n"
        "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-4] rec[-2]\n");
    const auto expected = compute_dense_circuit_distribution(source);

    auto count_identity_measures = [](const HirModule& hir, bool sign) {
        size_t count = 0;
        for (const HeisenbergOp& op : hir.ops) {
            if (op.op_type() == OpType::MEASURE && hir.destab_mask(op).is_zero() &&
                hir.stab_mask(op).is_zero() && hir.sign(op) == sign) {
                ++count;
            }
        }
        return count;
    };

    for (auto mode : {ReconstructionOutputs::Detectors, ReconstructionOutputs::Records}) {
        const auto r = reconstruct_global_sampling(source, mode);
        REQUIRE(r.reconstruction.has_value());
        const auto identity_ops = std::ranges::count_if(
            r.reconstruction->operations, [](const ReconstructionOperation& op) {
                return op.half_turns == 0.0 && op.axis.x().is_zero() && op.axis.z().is_zero();
            });
        REQUIRE(identity_ops >= 3);

        const auto noise = compress_reconstruction_noise(*r.reconstruction);
        REQUIRE(noise.noise.has_value());
        const auto lowered = lower_global_reconstruction(*r.reconstruction);
        REQUIRE(lowered.hir.has_value());

        // Neither output mode emits +I hidden measurements; inverted constants
        // share a single -I helper record, and detector-only lowering omits the
        // visible-record carrier wire.
        REQUIRE(count_identity_measures(*lowered.hir, false) == 0);
        REQUIRE(count_identity_measures(*lowered.hir, true) == 1);
        const uint32_t expected_wires = r.reconstruction->num_qubits + noise.noise->num_bits +
                                        (mode == ReconstructionOutputs::Records ? 1U : 0U);
        REQUIRE(lowered.hir->num_qubits == expected_wires);

        auto hir = *lowered.hir;
        default_hir_pass_manager().run(hir);
        const auto actual = compute_exact_plan_distribution(sampling::plan_sampling(hir));
        require_same_law(output_law(actual, r.reconstruction->outputs),
                         output_law(expected, r.reconstruction->outputs));
    }

    // Without inverted constants, zero identity-axis measurements of either sign are emitted.
    const auto uninverted = reconstruct_global_sampling(
        parse("H 0\nT 0\nMX 0\nX_ERROR(0.25) 1\nM 1 2\nDETECTOR rec[-2]\n"),
        ReconstructionOutputs::Detectors);
    REQUIRE(uninverted.reconstruction.has_value());
    const auto lowered_uninverted = lower_global_reconstruction(*uninverted.reconstruction);
    REQUIRE(lowered_uninverted.hir.has_value());
    REQUIRE(count_identity_measures(*lowered_uninverted.hir, false) == 0);
    REQUIRE(count_identity_measures(*lowered_uninverted.hir, true) == 0);
}

TEST_CASE("Noise compression conserves non-subspace convolution laws across words",
          "[reconstruction]") {
    // 66 independent outputs span two compressed coordinate words. Three channels
    // share the non-subspace support {I, b_0, b_65}: convolving the first two
    // introduces {b_0, b_65}, expanding the distribution beyond its initial key
    // while the third channel must still merge into the same group.
    GlobalReconstruction r;
    r.num_noise_bits = 132;
    for (uint32_t i = 0; i < 66; ++i)
        r.operations.push_back({PauliString(0), make_sign({2 * i}, i % 3 == 0), 0, i});
    r.noise_channels = {{{0.5, {}}, {0.3, {0}}, {0.2, {130}}},
                        {{0.6, {}}, {0.15, {0}}, {0.25, {130}}},
                        {{0.7, {}}, {0.2, {0}}, {0.1, {130}}},
                        {{0.8, {}}, {0.2, {66}}}};
    const auto noise = check_sign_law(r);
    REQUIRE(noise.num_bits == 66);
    REQUIRE(noise.channels.size() == 2);
}

TEST_CASE("Global reconstruction lowers complete noisy cultivation circuits",
          "[reconstruction][expensive]") {
    for (uint32_t distance : {5, 7}) {
        const auto source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/reconstruction_msc_d" +
                                       std::to_string(distance) + ".stim");
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto r = reconstruct_global_sampling(source, mode);
            INFO(r.fallback_reason);
            REQUIRE(r.reconstruction.has_value());
            REQUIRE(r.reconstruction->peak_active_width == (distance == 5 ? 6u : 8u));
            auto lowered = lower_global_reconstruction(*r.reconstruction);
            INFO(lowered.fallback_reason);
            REQUIRE(lowered.hir.has_value());
            default_hir_pass_manager().run(*lowered.hir);
            const sampling::ExecutablePlan executable(sampling::plan_sampling(*lowered.hir));
            REQUIRE(executable.peak_active_width() <= (distance == 5 ? 6u : 8u));
            REQUIRE(executable.num_visible_records() ==
                    (mode == ReconstructionOutputs::Records ? source.num_measurements : 0));
            REQUIRE(executable.num_detectors() == source.num_detectors);
            REQUIRE(executable.num_observables() == source.num_observables);
            REQUIRE_FALSE(executable.supports_final_state_queries());
            const auto scalar = sampling::sample(executable, 65, 9, 1, {}, 1);
            const auto packed = sampling::sample(executable, 65, 9, 1, {}, 64);
            REQUIRE(scalar.measurements.size() == 65 * executable.num_visible_records());
            REQUIRE(packed.measurements.size() == scalar.measurements.size());
            REQUIRE(packed.detectors.size() == 65 * source.num_detectors);
            REQUIRE(packed.observables.size() == 65 * source.num_observables);
        }
    }
}

TEST_CASE("Global reconstruction dilation propagates fault signs for every named Clifford gate",
          "[reconstruction]") {
    for (uint32_t raw = 0; raw < static_cast<uint32_t>(GateType::UNKNOWN); ++raw) {
        const auto gate = static_cast<GateType>(raw);
        if (!is_clifford(gate) || gate == GateType::SPP || gate == GateType::SPP_DAG) {
            continue;
        }
        const Tableau& local = Tableau::from_named_gate(gate);
        const uint32_t arity = local.num_qubits();
        for (bool z_measurement : {false, true}) {
            INFO(gate_name(gate) << " z_measurement=" << z_measurement);
            Circuit circuit;
            circuit.num_qubits = arity;
            circuit.num_measurements = arity;
            for (uint32_t j = 0; j < arity; ++j) {
                circuit.nodes.push_back({GateType::H, {Target::qubit(j)}, {}});
                circuit.nodes.push_back({GateType::T, {Target::qubit(j)}, {}});
                circuit.nodes.push_back({GateType::X_ERROR, {Target::qubit(j)}, {0.11 + 0.03 * j}});
                circuit.nodes.push_back({GateType::Z_ERROR, {Target::qubit(j)}, {0.17 + 0.05 * j}});
            }
            std::vector<Target> gate_targets;
            for (uint32_t i = 0; i < arity; ++i) {
                gate_targets.push_back(Target::qubit(i));
            }
            circuit.nodes.push_back({gate, std::move(gate_targets), {}});
            for (uint32_t i = 0; i < arity; ++i) {
                circuit.nodes.push_back(
                    {z_measurement ? GateType::M : GateType::MX, {Target::qubit(i)}, {}});
            }

            const auto res = reconstruct_global_sampling(circuit, ReconstructionOutputs::Records);
            REQUIRE(res.reconstruction.has_value());
            std::vector<uint64_t> got_masks(arity, UINT64_MAX);
            for (const ReconstructionOperation& op : res.reconstruction->operations) {
                if (op.output.has_value()) {
                    got_masks.at(*op.output) = op.sign.words.empty() ? 0 : op.sign.words[0];
                }
            }
            for (uint32_t i = 0; i < arity; ++i) {
                uint64_t want_mask = 0;
                for (uint32_t j = 0; j < arity; ++j) {
                    const bool from_x = z_measurement ? local.x_output(j).x().bit_get(i)
                                                      : local.x_output(j).z().bit_get(i);
                    const bool from_z = z_measurement ? local.z_output(j).x().bit_get(i)
                                                      : local.z_output(j).z().bit_get(i);
                    if (from_x) {
                        want_mask |= uint64_t{1} << (2 * j);
                    }
                    if (from_z) {
                        want_mask |= uint64_t{1} << (2 * j + 1);
                    }
                }
                REQUIRE(got_masks[i] == want_mask);
            }

            const auto lowered = lower_global_reconstruction(*res.reconstruction);
            REQUIRE(lowered.hir.has_value());
            const auto actual =
                compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
            const auto expected =
                compute_exact_plan_distribution(sampling::plan_sampling(trace(circuit)));
            require_same_law(output_law(actual, res.reconstruction->outputs),
                             output_law(expected, res.reconstruction->outputs));
        }
    }
}

}  // namespace clifft::test
