#include "clifft/api/program.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/global_reconstruction.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/early_postselection.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/sampler.h"
#include "clifft/tableau/tableau.h"

#include "dense_circuit_oracle.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <map>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace clifft::test {
namespace {

std::string sample_pauli_channel_3_instruction() {
    std::string line = "PAULI_CHANNEL_3(";
    for (int i = 0; i < 63; ++i) {
        if (i != 0)
            line += ",";
        if (i == 0)
            line += "0.03";  // IIX
        else if (i == 6)
            line += "0.05";  // IXZ
        else if (i == 26)
            line += "0.07";  // XYZ
        else if (i == 41)
            line += "0.04";  // YYY
        else if (i == 62)
            line += "0.06";  // ZZZ
        else
            line += "0";
    }
    line += ") 0 1 2\n";
    return line;
}

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
        "R_X(0.19) 0\nZ_ERROR(0.22) 0\nR_Y(-0.31) 1\nX_ERROR(0.18) 1\nCX 0 1\nR_Z(0.43) 1\n"
        "M(0.12) 0\nCZ rec[-1] 1\nMY 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "H 0\nX_ERROR(0.21) 0\nU3(0.31,-0.47,0.19) 0\nU3(0.5,0.13,1.0) 1\nCX 0 1\n"
        "U3(0.0,-0.5,0.29) 1\nM 0 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
        "H 0 1\nDEPOLARIZE2(0.24) 0 1\nR_XX(0.17) 0 1\nR_YY(-0.29) 0 1\nR_ZZ(0.5) 0 1\n"
        "R_XX(1.0) 0 1\nR_YY(0.0) 0 1\nMX 0\nMY 1\nDETECTOR rec[-2] rec[-1]\n",
        "H 0 1 2\nY_ERROR(0.19) 1\nSPP X0*!Y1*Z2\nTPP !Z0*X1\nTPP_DAG Y1*!Z2\n"
        "SPP_DAG !X0*!Z2\nR_PAULI(0.23) !X0*Y1*Z2\nR_PAULI(0.5) !Y0*X2\n"
        "R_PAULI(1.0) !X0*Z1*Y2\nR_PAULI(0.0) X0*Y1\nM 0 1 2\n"
        "DETECTOR rec[-3] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
        "H 0 1 2\nR_X(0.21) 0\nDEPOLARIZE3(0.315) 0 1 2\nR_PAULI(-0.27) X0*!Y1*Z2\n"
        "MR(0.15) 1\nCX rec[-1] 2\nU3(0.5,0.5,-0.19) 2\nM 2\n"
        "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "H 0 1 2\nTPP X0*Y1*Z2\n" + sample_pauli_channel_3_instruction() +
            "E(0.16) Z0*X2\nELSE_CORRELATED_ERROR(0.25) Y1*Z2\nR_ZZ(-0.37) 0 2\n"
            "SPP !Y0*X1\nMX 0\nM 2\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-1]\n",
        "H 0\nR_Z(0.23) 0\nMPAD(0.18) 1\nMPAD(0.12) !0\nCX rec[-2] 0\nMX 0\n"
        "DETECTOR rec[-3] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1]\n",
    };
}

ExactPlanDistribution project_outputs(ExactPlanDistribution distribution,
                                      ReconstructionOutputs mode) {
    if (mode == ReconstructionOutputs::Detectors) {
        distribution.num_visible_records = 0;
        for (auto& entry : distribution.kept_outcomes)
            entry.outcome.visible_records = 0;
        canonicalize_outcome_entries(distribution.kept_outcomes);
    }
    return distribution;
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
    REQUIRE(r.num_noise_bits <= 64);
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
            const uint64_t terms =
                (!op.sign.words.empty() && op.sign.words[0].index == 0) ? op.sign.words[0].mask : 0;
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
    const auto traced = compute_exact_plan_distribution(sampling::plan_sampling(trace(circuit)));
    for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
        const auto result = reconstruct_global_sampling(circuit, mode);
        INFO(result.fallback_reason);
        REQUIRE(result.reconstruction.has_value());
        const auto& r = *result.reconstruction;
        REQUIRE(r.outputs.num_records ==
                (mode == ReconstructionOutputs::Records ? circuit.num_measurements : 0));
        REQUIRE(r.outputs.num_detectors == circuit.num_detectors);
        REQUIRE(r.outputs.num_observables == circuit.num_observables);
        const auto actual = reconstructed_law(r);
        require_same_law(actual, output_law(expected, r.outputs));
        require_same_law(actual, output_law(traced, r.outputs));
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
            for (const auto& w : sign.words) {
                for (uint64_t v = w.mask; v != 0; v &= v - 1) {
                    const uint32_t b = w.index * 64 + static_cast<uint32_t>(std::countr_zero(v));
                    parity ^= assignment.at(b);
                }
            }
            value.push_back(parity);
        }
        law[value] += probability;
    }
    return law;
}

ReconstructionSign make_sign(std::initializer_list<uint32_t> bits, bool constant = false) {
    std::map<uint32_t, uint64_t> by_word;
    for (uint32_t b : bits) {
        by_word[b / 64] ^= uint64_t{1} << (b % 64);
    }
    ReconstructionSign result{{}, constant};
    for (const auto& [idx, mask] : by_word) {
        if (mask != 0) {
            result.words.push_back({idx, mask});
        }
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
    const std::vector<std::string> single{
        "H",  "S",         "S_DAG",      "T",        "T_DAG",
        "X",  "Y",         "Z",          "R",        "RX",
        "RY", "R_X(0.19)", "R_Y(-0.31)", "R_Z(0.5)", "U3(0.21,-0.37,0.5)"};
    const std::vector<std::string> pairs{"CX", "CY", "CZ", "SWAP", "R_XX(0.23)", "R_ZZ(-0.5)"};
    const std::vector<std::string> products{"SPP X0*!Y1", "SPP_DAG Y1*Z2", "TPP !X0*Z2",
                                            "R_PAULI(-0.29) !X0*Y1*Z2"};
    const std::vector<std::string> measurements{"M", "MX", "MY", "MR", "MRX", "MRY"};
    for (uint32_t trial = 0; trial < 32; ++trial) {
        std::string text = "H 0\nT 0\n";
        uint32_t records = 0;
        for (uint32_t step = 0; step < 12; ++step) {
            const auto q = rng() % 3;
            const auto choice = rng() % 6;
            if (choice < 2) {
                text += single[rng() % single.size()] + " " + std::to_string(q) + "\n";
            } else if (choice == 2) {
                text += pairs[rng() % pairs.size()] + " " + std::to_string(q) + " " +
                        std::to_string((q + 1) % 3) + "\n";
            } else if (choice == 3) {
                text += products[rng() % products.size()] + "\n";
            } else if (choice == 4 || records == 0) {
                text += measurements[rng() % measurements.size()] + " " + (rng() % 2 ? "!" : "") +
                        std::to_string(q) + "\n";
                ++records;
            } else {
                text += "CZ rec[-1] " + std::to_string(q) + "\n";
            }
            if (step == 5)
                text += (trial % 2 == 0) ? "Y_ERROR(0.27) 1\n" : "DEPOLARIZE3(0.21) 0 1 2\n";
        }
        text += "M 0 1 2\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\n";
        INFO(text);
        check_joint_law(text);
    }
}

TEST_CASE("Global reconstruction rejects unsupported preservation requirements",
          "[reconstruction]") {
    for (const auto& text : {"H 0\nT 0\nEXP_VAL X0\n", "LEAKAGE(0.1) 0\nM 0\n",
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
            const auto projected = project_outputs(expected, mode);
            for (bool optimize : {false, true}) {
                if (optimize)
                    default_hir_pass_manager().run(*lowered.hir);
                const auto actual =
                    compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
                require_distributions_equal(projected, actual, 2e-12);
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
    const std::vector<std::string> sources{
        "H 0\nT 0\nMX 0\nX_ERROR(0.2) 1\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] "
        "rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-2]\n",
        "R_X(0.21) 0\nU3(0.5,-0.5,0.17) 1\nH 2\nDEPOLARIZE3(0.252) 0 1 2\n"
        "R_PAULI(-0.29) !X0*Y1*Z2\nSPP !Y0*Z2\nM(0.12) 2\n"
        "DETECTOR rec[-1]\nDETECTOR\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-1]\n",
        "U3(0.29,0.5,-0.19) 0\nE(0.18) X0*Z1\nELSE_CORRELATED_ERROR(0.25) Y1\n"
        "MRX(0.11) 0\nCX rec[-1] 1\nR_PAULI(0.37) !Y0*Z1\nM 1\n"
        "DETECTOR rec[-2] rec[-1]\nDETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-1]\n",
    };
    const auto expected_detectors = std::vector<uint8_t>{1, 0};
    const auto expected_observables = std::vector<uint8_t>{1, 1};
    for (const auto& text : sources) {
        INFO(text);
        const auto source = parse(text);
        for (const auto& mask : {std::vector<uint8_t>{0, 0}, std::vector<uint8_t>{1, 0},
                                 std::vector<uint8_t>{0, 1}, std::vector<uint8_t>{1, 1}}) {
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
                const auto projected = project_outputs(expected, mode);
                auto check_plan = [&](const sampling::SamplingPlan& plan) {
                    require_distributions_equal(projected, compute_exact_plan_distribution(plan),
                                                2e-12);
                };
                // Check raw lowered HIR directly whenever its hidden-record count
                // fits PreparedReplayPlan's 12-record enumeration cap; wider
                // multi-channel cases also run default HIR passes to prune records.
                const auto raw_plan = sampling::plan_sampling(*lowered.hir);
                if (raw_plan.num_visible_records + raw_plan.num_hidden_records <= 12) {
                    check_plan(raw_plan);
                }
                auto hir = *lowered.hir;
                default_hir_pass_manager().run(hir);
                check_plan(sampling::plan_sampling(hir));
            }
        }
    }
}

TEST_CASE("Reconstructed postselection preserves rare accepted logical errors",
          "[reconstruction]") {
    constexpr double p = 0x1p-42;
    const std::string prepare = "H 0 1\nT 0 1\nH 0 1\n";
    const std::string measure = "M 0 1 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]\n";
    const auto source = parse(prepare + "X_ERROR(" + precise_probability(p) + ") 2\n" + measure);
    const auto without_fault = parse(prepare + measure);
    PostselectionMetadata meta;
    meta.mask = {1};
    meta.expected_detectors = {1};
    const auto dense =
        compute_dense_circuit_distribution(source, {meta.mask, meta.expected_detectors});
    // The fault on qubit 2 accepts the shot independently of the H T H measurement on qubit 0.
    const double logical_error_mass = p * (1 - 1 / std::numbers::sqrt2) / 2;
    const ExactComparisonBudget budget{.max_omitted_probability = logical_error_mass * 1e-9};

    for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
        CAPTURE(mode);
        auto optimized_law = [&](const Circuit& circuit) {
            const auto r = reconstruct_global_sampling(circuit, mode);
            REQUIRE(r.reconstruction.has_value());
            auto lowered = lower_global_reconstruction(*r.reconstruction, meta);
            REQUIRE(lowered.hir.has_value());
            default_hir_pass_manager().run(*lowered.hir);
            return compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
        };
        const auto expected = project_outputs(dense, mode);
        const auto actual = optimized_law(source);
        require_distributions_equal(expected, actual, budget);
        double accepted = 0;
        double accepted_errors = 0;
        for (const auto& entry : actual.kept_outcomes) {
            accepted += entry.probability;
            if (entry.outcome.observables == 1)
                accepted_errors += entry.probability;
        }
        REQUIRE_THAT(accepted, Catch::Matchers::WithinRel(p, budget.probability_relative));
        REQUIRE_THAT(accepted_errors,
                     Catch::Matchers::WithinRel(logical_error_mass, budget.probability_relative));

        const auto deleted = optimized_law(without_fault);
        REQUIRE(deleted.kept_outcomes.empty());
        REQUIRE_FALSE(distributions_equal(expected, deleted, budget));
    }
}

TEST_CASE("Global reconstruction absorbs Clifford angle multiples and signed axes",
          "[reconstruction]") {
    // Identity rotations on untouched qubits must not allocate dilation wires,
    // magic ancillas, or hidden projective measurements, and Clifford multiples
    // (including values within kRotationCanonicalizationTolerance) must keep
    // reconstructed active width at zero.
    const auto clifford_source = parse(
        "R_Z(0.0) 100\n"
        "R_PAULI(2.0) !X101*Y102\n"
        "U3(0.0, -2.0, 2.0) 103\n"
        "H 0 1 2\n"
        "DEPOLARIZE3(0.189) 0 1 2\n"
        "R_X(0.5000000000004) 0\n"
        "R_Y(-0.4999999999996) 1\n"
        "R_Z(1.0000000000004) 2\n"
        "U3(0.5, -1.5, 1.0) 0\n"
        "R_XX(0.5) 0 1\n"
        "R_YY(-0.5) 1 2\n"
        "R_ZZ(1.0) 0 2\n"
        "SPP !X0*Y1*Z2\n"
        "SPP_DAG !Y0*!Z1\n"
        "R_PAULI(0.5) !X0*Z2\n"
        "R_PAULI(-0.5) X1*!Y2\n"
        "R_PAULI(1.0) !X0*Y1*Z2\n"
        "R_PAULI(0.0000000000004) X0*Y1*Z2\n"
        "M 0 2\n"
        "DETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n");
    for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
        const auto res = reconstruct_global_sampling(clifford_source, mode);
        INFO(res.fallback_reason);
        REQUIRE(res.reconstruction.has_value());
        const auto& r = *res.reconstruction;
        // Pure-Clifford circuits evolve entirely in the suffix frame without dilation wires.
        REQUIRE(r.dilation_qubits == 0);
        REQUIRE(r.peak_active_width == 0);
        REQUIRE(std::ranges::none_of(
            r.operations, [](const ReconstructionOperation& op) { return op.half_turns != 0.0; }));

        const auto lowered = lower_global_reconstruction(r);
        REQUIRE(lowered.hir.has_value());
        const auto actual = compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
        const auto expected =
            compute_exact_plan_distribution(sampling::plan_sampling(trace(clifford_source)));
        require_distributions_equal(project_outputs(expected, mode), actual, 2e-12);
    }

    // Identity rotations on untouched qubits before a non-Clifford gate must also
    // stay unallocated in prefix dilation.
    const auto prefix_elision = parse(
        "R_Z(0.0) 100\n"
        "R_PAULI(2.0) !X101*Y102\n"
        "U3(0.0, -2.0, 2.0) 103\n"
        "H 0 1 2\n"
        "SPP !X0*Y1*Z2\n"
        "T 0\n"
        "MX 0 2\n"
        "DETECTOR rec[-2] rec[-1]\n");
    const auto prefix_res =
        reconstruct_global_sampling(prefix_elision, ReconstructionOutputs::Records);
    REQUIRE(prefix_res.reconstruction.has_value());
    REQUIRE(prefix_res.reconstruction->dilation_qubits == 4);

    // Angles outside kRotationCanonicalizationTolerance remain genuine rotations.
    const auto outside_tolerance = parse("H 0\nR_Z(0.500000000002) 0\nMX 0\nDETECTOR rec[-1]\n");
    const auto non_clifford =
        reconstruct_global_sampling(outside_tolerance, ReconstructionOutputs::Records);
    REQUIRE(non_clifford.reconstruction.has_value());
    REQUIRE(std::ranges::any_of(
        non_clifford.reconstruction->operations,
        [](const ReconstructionOperation& op) { return op.half_turns != 0.0; }));
    const auto lowered = lower_global_reconstruction(*non_clifford.reconstruction);
    REQUIRE(lowered.hir.has_value());
    const auto actual = compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
    const auto expected =
        compute_exact_plan_distribution(sampling::plan_sampling(trace(outside_tolerance)));
    require_distributions_equal(expected, actual, 2e-12);
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
            "H 0 1 2\nT 0 1 2\nH 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) "
            "rec[-3]\n");
        const auto det_unbounded =
            reconstruct_global_sampling(asymmetric, ReconstructionOutputs::Detectors);
        const auto rec_unbounded =
            reconstruct_global_sampling(asymmetric, ReconstructionOutputs::Records);
        REQUIRE(det_unbounded.reconstruction.has_value());
        REQUIRE(rec_unbounded.reconstruction.has_value());

        // Detectors schedules 3 rotations + 1 output measurement on 3 qubits (4 + 3 = 7),
        // while Records schedules 3 rotations + 3 distinct output measurements (6 + 3 = 9).
        ReconstructionLimits limits;
        limits.max_schedule_operations = 7;
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
        require_distributions_equal(project_outputs(expected, mode), actual, 2e-12);
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
        for (bool in_suffix : {true, false}) {
            for (bool z_measurement : {false, true}) {
                INFO(gate_name(gate)
                     << " in_suffix=" << in_suffix << " z_measurement=" << z_measurement);
                Circuit circuit;
                circuit.num_qubits = arity + (in_suffix ? 0U : 1U);
                circuit.num_measurements = arity;
                for (uint32_t j = 0; j < arity; ++j) {
                    circuit.nodes.push_back({GateType::H, {Target::qubit(j)}, {}});
                    circuit.nodes.push_back({GateType::T, {Target::qubit(j)}, {}});
                    circuit.nodes.push_back(
                        {GateType::X_ERROR, {Target::qubit(j)}, {0.11 + 0.03 * j}});
                    circuit.nodes.push_back(
                        {GateType::Z_ERROR, {Target::qubit(j)}, {0.17 + 0.05 * j}});
                }
                std::vector<Target> gate_targets;
                for (uint32_t i = 0; i < arity; ++i) {
                    gate_targets.push_back(Target::qubit(i));
                }
                circuit.nodes.push_back({gate, std::move(gate_targets), {}});
                if (!in_suffix) {
                    circuit.nodes.push_back({GateType::T, {Target::qubit(arity)}, {}});
                }
                for (uint32_t i = 0; i < arity; ++i) {
                    circuit.nodes.push_back(
                        {z_measurement ? GateType::M : GateType::MX, {Target::qubit(i)}, {}});
                }

                const auto res =
                    reconstruct_global_sampling(circuit, ReconstructionOutputs::Records);
                REQUIRE(res.reconstruction.has_value());
                std::vector<uint64_t> got_masks(arity, UINT64_MAX);
                for (const ReconstructionOperation& op : res.reconstruction->operations) {
                    if (op.output.has_value()) {
                        got_masks.at(*op.output) =
                            (!op.sign.words.empty() && op.sign.words[0].index == 0)
                                ? op.sign.words[0].mask
                                : 0;
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
                require_distributions_equal(expected, actual, 2e-12);
            }
        }
    }
}

TEST_CASE("Global reconstruction handles multi-target AST nodes and identity no-ops",
          "[reconstruction]") {
    Circuit circuit;
    circuit.num_qubits = 6;
    circuit.num_measurements = 5;
    circuit.num_detectors = 1;
    circuit.num_observables = 1;
    // Identity no-ops on untouched qubits must not allocate dilation wires.
    circuit.nodes.push_back({GateType::I, {Target::qubit(3), Target::qubit(4)}, {}});
    circuit.nodes.push_back({GateType::II, {Target::qubit(4), Target::qubit(5)}, {}});
    circuit.nodes.push_back({GateType::I_ERROR, {Target::qubit(5)}, {0.4}});
    // Broadcast multi-target nodes across Clifford, rotation, noise, padding, and measurement ops.
    circuit.nodes.push_back(
        {GateType::H, {Target::qubit(0), Target::qubit(1), Target::qubit(2)}, {}});
    circuit.nodes.push_back({GateType::T, {Target::qubit(0), Target::qubit(1)}, {}});
    circuit.nodes.push_back({GateType::R_X, {Target::qubit(1), Target::qubit(2)}, {0.19}});
    circuit.nodes.push_back(
        {GateType::U3, {Target::qubit(0), Target::qubit(2)}, {0.23, -0.5, 0.5}});
    circuit.nodes.push_back(
        {GateType::R_ZZ,
         {Target::qubit(0), Target::qubit(1), Target::qubit(1), Target::qubit(2)},
         {-0.31}});
    circuit.nodes.push_back({GateType::X_ERROR, {Target::qubit(0), Target::qubit(2)}, {0.15}});
    circuit.nodes.push_back({GateType::Z_ERROR, {Target::qubit(1), Target::qubit(2)}, {0.18}});
    circuit.nodes.push_back(
        {GateType::MPAD, {Target::qubit(0), Target{1 | Target::kInvertBit}}, {}});
    circuit.nodes.push_back(
        {GateType::M, {Target::qubit(0), Target::qubit(1), Target::qubit(2)}, {}});
    circuit.nodes.push_back(
        {GateType::READOUT_NOISE, {Target::rec(2), Target::rec(3), Target::rec(4)}, {0.1}});
    circuit.nodes.push_back(
        {GateType::DETECTOR, {Target::rec(0), Target::rec(2), Target::rec(4)}, {}});
    circuit.nodes.push_back(
        {GateType::OBSERVABLE_INCLUDE, {Target::rec(1), Target::rec(3)}, {0.0}});

    const auto reference = parse(
        "H 0 1 2\n"
        "T 0 1\n"
        "R_X(0.19) 1 2\n"
        "U3(0.23, -0.5, 0.5) 0 2\n"
        "R_ZZ(-0.31) 0 1 1 2\n"
        "X_ERROR(0.15) 0 2\n"
        "Z_ERROR(0.18) 1 2\n"
        "MPAD 0 !1\n"
        "M(0.1) 0 1 2\n"
        "DETECTOR rec[-5] rec[-3] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-4] rec[-2]\n");
    const auto traced = compute_exact_plan_distribution(sampling::plan_sampling(trace(reference)));
    const auto dense = compute_dense_circuit_distribution(circuit);
    for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
        const auto res = reconstruct_global_sampling(circuit, mode);
        const auto ref_res = reconstruct_global_sampling(reference, mode);
        INFO(res.fallback_reason);
        REQUIRE(res.reconstruction.has_value());
        REQUIRE(ref_res.reconstruction.has_value());
        // Qubits 3..5 stay unallocated and suffix MPAD/M ops do not dilate (3 source + 8 gadgets).
        REQUIRE(res.reconstruction->dilation_qubits == 11);
        REQUIRE(res.reconstruction->dilation_qubits == ref_res.reconstruction->dilation_qubits);
        const auto actual = reconstructed_law(*res.reconstruction);
        require_same_law(actual, output_law(traced, res.reconstruction->outputs));
        require_same_law(actual, output_law(dense, res.reconstruction->outputs));
    }
}

TEST_CASE("Global reconstruction tracks pure-Clifford memory suffixes without dilation",
          "[reconstruction]") {
    // Exercise dead spectator fault marginalization before resets and at circuit end,
    // entangled Bell-pair resets, non-commuting suffix measurements, repeat syndrome
    // checks, classical feedback from prefix and suffix records, and Clifford-angle
    // suffix rotations.
    const std::vector<std::string> suffix_cases{
        // Entangled Bell reset followed by partner measurement must keep uniform marginal.
        "H 0\nT 0\nH 1\nCX 1 2\nR 1\nM 0 1 2\nDETECTOR rec[-2] rec[-1]\n",
        // Non-commuting multi-qubit suffix sequence: MPP Z0*Z1 -> MX 0 -> MX 1 -> M 0 -> MX 1.
        "H 0 1\nT 0\nR_Z(0.31) 1\nCZ 0 1\nT_DAG 0\n"
        "MPP Z0*Z1\nMX 0\nMX 1\nM 0\nMX 1\n"
        "DETECTOR rec[-5] rec[-3]\nDETECTOR rec[-4] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
        // Dead spectator fault marginalization across DEPOLARIZE1/2 and correlated chains.
        "DEPOLARIZE1(0.3) 3\n"
        "H 0 1\nT 0\nR_PAULI(0.27) !X0*Y1\n"
        "DEPOLARIZE2(0.24) 1 3\n"
        "R 3\n"
        "E(0.25) X3*Z2\nELSE_CORRELATED_ERROR(0.4) Y0*X1\n"
        "R 2\n"
        "CX 0 2\nCX 1 2\nMR(0.08) 2\n"
        "CX 0 2\nCX 1 2\nMR 2\n"
        "CZ rec[-2] 0\nCX rec[-1] 1\n"
        "SPP !Y0*Z1\nR_XX(0.0) 3 0\nR_PAULI(0.5) X0*!Y1\nU3(0.5,-0.5,1.0) 1\n"
        "MPP !Y0*Z1\nM 0\n"
        "DETECTOR rec[-4] rec[-3]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
        // Dead spectator fault marginalization on DEPOLARIZE3 at circuit end and before reset.
        "H 0 1\nT 0\nDEPOLARIZE3(0.315) 0 2 3\nR 2\nCX 0 1\nMX 0\nM 1\n"
        "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        // Suffix X/Y/Z resets, inverted readouts, and prefix-to-suffix classical feedback.
        "H 0\nT 0\nMR(0.12) 0\nH 0\nT_DAG 0\n"
        "CX rec[-1] 0\nRX 1\nMX 1\nRY 1\nMY !1\nM 0\n"
        "DETECTOR rec[-4] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
    };
    for (const auto& text : suffix_cases) {
        INFO(text);
        check_joint_law(text);
        const auto circuit = parse(text);
        const auto expected = compute_dense_circuit_distribution(circuit);
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto r = reconstruct_global_sampling(circuit, mode);
            REQUIRE(r.reconstruction.has_value());
            auto lowered = lower_global_reconstruction(*r.reconstruction);
            REQUIRE(lowered.hir.has_value());
            default_hir_pass_manager().run(*lowered.hir);
            const auto actual =
                compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
            require_distributions_equal(project_outputs(expected, mode), actual, 2e-12);
        }
    }

    // Scaling pure-Clifford syndrome rounds must keep prefix dilation_qubits fixed
    // and avoid triggering max_schedule_operations on classical/duplicate syndrome checks.
    auto make_memory_tail = [](uint32_t rounds) {
        std::string text = "H 0 1\nT 0\nT_DAG 1\nCX 0 1\n";
        for (uint32_t r = 0; r < rounds; ++r) {
            text += "R 2\nCX 0 2\nCX 1 2\nX_ERROR(0.05) 2\nM 2\n";
            if (r > 0) {
                text += "DETECTOR rec[-2] rec[-1]\n";
            }
        }
        text += "MX 0 1\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1]\n";
        return parse(text);
    };
    ReconstructionLimits tight_schedule;
    tight_schedule.max_schedule_operations = 16;
    const auto tail_2 = reconstruct_global_sampling(make_memory_tail(2),
                                                    ReconstructionOutputs::Records, tight_schedule);
    const auto tail_40 = reconstruct_global_sampling(
        make_memory_tail(40), ReconstructionOutputs::Records, tight_schedule);
    REQUIRE(tail_2.reconstruction.has_value());
    REQUIRE(tail_40.reconstruction.has_value());
    REQUIRE(tail_2.reconstruction->dilation_qubits == 4);
    REQUIRE(tail_40.reconstruction->dilation_qubits == 4);
    REQUIRE(tail_40.reconstruction->peak_active_width <= 2);
}

TEST_CASE("Global reconstruction matches dense oracle on random pure-Clifford suffixes",
          "[reconstruction]") {
    Xoshiro256PlusPlus rng(918273);
    const std::vector<std::string> suffix_single{
        "H",         "S",        "S_DAG",           "X", "Y", "Z", "R", "RX", "RY", "R_X(0.5)",
        "R_Y(-0.5)", "R_Z(0.5)", "U3(0.5,-0.5,1.0)"};
    const std::vector<std::string> suffix_pairs{"CX",   "CY",        "CZ",
                                                "SWAP", "R_XX(0.5)", "R_ZZ(-0.5)"};
    const std::vector<std::string> suffix_meas{"M", "MX", "MY", "MR", "MRX", "MRY"};
    const std::vector<std::string> suffix_mpp{"MPP X0*Z1", "MPP !Y1*X2", "MPP Z0*Y2*X3"};

    size_t law_checks = 0;
    size_t plan_checks = 0;
    for (uint32_t trial = 0; trial < 24; ++trial) {
        std::string text =
            "DEPOLARIZE1(0.2) 3\nH 0 1 2\nT 0\nR_Y(0.23) 1\nCX 0 1\nTPP X1*Z2\nMR(0.1) 0\n"
            "H 0\nR_Z(-0.31) 0\n";
        uint32_t records = 1;
        for (uint32_t step = 0; step < 10; ++step) {
            const uint32_t q = static_cast<uint32_t>(rng() % 4);
            const uint32_t choice = static_cast<uint32_t>(rng() % 8);
            if (choice < 2) {
                text +=
                    suffix_single[rng() % suffix_single.size()] + " " + std::to_string(q) + "\n";
            } else if (choice < 4) {
                text += suffix_pairs[rng() % suffix_pairs.size()] + " " + std::to_string(q) + " " +
                        std::to_string((q + 1) % 4) + "\n";
            } else if (choice == 4) {
                text += suffix_meas[rng() % suffix_meas.size()] + " " + (rng() % 2 ? "!" : "") +
                        std::to_string(q) + "\n";
                ++records;
            } else if (choice == 5) {
                text += suffix_mpp[rng() % suffix_mpp.size()] + "\n";
                ++records;
            } else if (choice == 6) {
                text += (rng() % 2 ? "CX" : "CZ") + std::string(" rec[-") +
                        std::to_string(1 + (rng() % records)) + "] " + std::to_string(q) + "\n";
            } else {
                text += (step == 5) ? "Y_ERROR(0.18) " + std::to_string(q) + "\n"
                                    : "SPP !X" + std::to_string(q) + "*Y" +
                                          std::to_string((q + 1) % 4) + "\n";
            }
        }
        text +=
            "M 0 1 2 3\nDETECTOR rec[-4] rec[-2]\nDETECTOR rec[-3] rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-4] rec[-1]\n";
        INFO(text);
        const auto circuit = parse(text);
        const auto expected = compute_dense_circuit_distribution(circuit);
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto res = reconstruct_global_sampling(circuit, mode);
            INFO(res.fallback_reason);
            REQUIRE(res.reconstruction.has_value());
            if (res.reconstruction->num_noise_bits <= 8 &&
                res.reconstruction->outputs.num_outputs() < 64) {
                require_same_law(reconstructed_law(*res.reconstruction),
                                 output_law(expected, res.reconstruction->outputs));
                ++law_checks;
            }
            auto lowered = lower_global_reconstruction(*res.reconstruction);
            REQUIRE(lowered.hir.has_value());
            default_hir_pass_manager().run(*lowered.hir);
            const auto plan = sampling::plan_sampling(*lowered.hir);
            uint64_t realizations = 1;
            for (const auto& site : lowered.hir->noise_sites) {
                realizations *= (site.channels.size() + 1);
                if (realizations > 256) {
                    break;
                }
            }
            if (realizations <= 256 && plan.num_visible_records + plan.num_hidden_records <= 12) {
                const auto actual = compute_exact_plan_distribution(plan);
                require_distributions_equal(project_outputs(expected, mode), actual, 2e-12);
                ++plan_checks;
            }
        }
    }
    REQUIRE(law_checks >= 40);
    REQUIRE(plan_checks >= 20);
}

TEST_CASE("Global reconstruction keeps postselected detector fault records ahead of quantum ops",
          "[reconstruction]") {
    const auto is_quantum_action = [](std::string_view text) {
        return text.starts_with("PROMOTE ") || text.starts_with("ROTATE ") ||
               text.starts_with("FUSED_ROTATION ") || text.starts_with("DYNAMIC_FUSED_ROTATION ") ||
               text.starts_with("MEASURE_ACTIVE ") || text.starts_with("MEASURE_DORMANT ");
    };

    // Detector 0 is postselected and depends on an inverted output-only fault wire
    // (exercising both `early_fault` and the constant `one` record), while detector 1
    // is unpostselected and keeps its output-only fault wire deferred past rotations.
    {
        const auto circuit = parse(
            "H 0\nT 0\nH 0\n"
            "X_ERROR(0.2) 1\nM !1\nDETECTOR rec[-1]\n"
            "X_ERROR(0.1) 2\nM 2\nDETECTOR rec[-1]\n"
            "M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
        auto original_hir = trace(circuit);
        original_hir.postselection.mask = {1, 0};

        const auto res = reconstruct_global_sampling(circuit, ReconstructionOutputs::Detectors);
        REQUIRE(res.reconstruction.has_value());
        auto lowered = lower_global_reconstruction(*res.reconstruction, original_hir.postselection);
        REQUIRE(lowered.hir.has_value());
        default_hir_pass_manager().run(*lowered.hir);

        const sampling::ExecutablePlan plan(sampling::plan_sampling(*lowered.hir));
        const auto order = sampling::EarlyPostselection::schedule(plan, true);

        std::string d1_record_token;
        for (size_t i = 0; i < plan.num_actions(); ++i) {
            const std::string action = plan.inspect_action(i);
            if (action.starts_with("WRITE_DETECTOR detector=d1 outcome=")) {
                const size_t start = std::string_view("WRITE_DETECTOR detector=d1 outcome=").size();
                const size_t end = action.find(' ', start);
                d1_record_token = "record=" + action.substr(start, end - start);
            }
        }
        REQUIRE_FALSE(d1_record_token.empty());

        size_t first_quantum_pos = order.size();
        size_t last_quantum_pos = 0;
        size_t d0_pos = order.size();
        size_t d1_record_pos = order.size();
        for (size_t pos = 0; pos < order.size(); ++pos) {
            const std::string action = plan.inspect_action(order[pos]);
            if (is_quantum_action(action)) {
                first_quantum_pos = std::min(first_quantum_pos, pos);
                last_quantum_pos = pos;
            }
            if (action.starts_with("WRITE_DETECTOR detector=d0 ")) {
                d0_pos = pos;
            }
            if (action.starts_with("RECORD_CLASSICAL ") &&
                action.find(d1_record_token) != std::string::npos) {
                d1_record_pos = pos;
            }
        }
        REQUIRE(first_quantum_pos < order.size());
        REQUIRE(d0_pos < first_quantum_pos);
        REQUIRE(d1_record_pos < order.size());
        REQUIRE(d1_record_pos > last_quantum_pos);
    }

    // On full postselected MSC d=5, early postselection must schedule nearly all
    // detector checks in the initial prefix rather than after the quantum section.
    {
        const auto circuit =
            parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/reconstruction_msc_d5.stim");
        auto original_hir = trace(circuit);
        original_hir.postselection.mask.assign(original_hir.num_detectors, 1);

        const auto res = reconstruct_global_sampling(circuit, ReconstructionOutputs::Detectors);
        REQUIRE(res.reconstruction.has_value());
        auto lowered = lower_global_reconstruction(*res.reconstruction, original_hir.postselection);
        REQUIRE(lowered.hir.has_value());
        default_hir_pass_manager().run(*lowered.hir);

        const sampling::ExecutablePlan plan(sampling::plan_sampling(*lowered.hir));
        const auto order = sampling::EarlyPostselection::schedule(plan, true);
        size_t total_quantum = 0;
        for (size_t i = 0; i < plan.num_actions(); ++i) {
            total_quantum += is_quantum_action(plan.inspect_action(i));
        }
        REQUIRE(total_quantum > 20);

        size_t seen_quantum = 0;
        size_t seen_active = 0;
        size_t checks_before_active = 0;
        size_t checks_before_end = 0;
        for (size_t idx : order) {
            const std::string action = plan.inspect_action(idx);
            if (is_quantum_action(action)) {
                ++seen_quantum;
                if (!action.starts_with("MEASURE_DORMANT ")) {
                    ++seen_active;
                }
            } else if (action.starts_with("WRITE_DETECTOR ")) {
                if (seen_active == 0) {
                    ++checks_before_active;
                }
                if (seen_quantum < total_quantum) {
                    ++checks_before_end;
                }
            }
        }
        REQUIRE(checks_before_active >= 90);
        REQUIRE(checks_before_end >= 103);
    }
}

TEST_CASE("Global reconstruction lowering bounds dense frame memory on wide fault domains",
          "[reconstruction]") {
    // 4096 independent fault wires use < 200 KB in the sparse mask arena, while
    // downstream CoordinateFrame / DormantSubspace bit matrices require ~25 MB.
    GlobalReconstruction wide;
    wide.num_qubits = 1;
    wide.num_noise_bits = 4096;
    wide.outputs = {0, 4096, 0};
    for (uint32_t i = 0; i < 4096; ++i) {
        wide.operations.push_back({PauliString(1), make_sign({i}), 0, i});
        wide.noise_channels.push_back({{0.5, {}}, {0.5, {i}}});
    }

    ReconstructionLimits tight;
    tight.max_lowering_bytes = 4 * 1024 * 1024;
    const auto rejected = lower_global_reconstruction(wide, {}, tight);
    REQUIRE_FALSE(rejected.hir.has_value());
    REQUIRE(rejected.fallback_reason == "lowering memory limit exceeded");
}

TEST_CASE(
    "Global reconstruction discards isolated suffix reset dependencies and caps prefix rotations",
    "[reconstruction]") {
    for (const char* text : {
             "H 0\nT 0\nRY 0\nM 0\nDETECTOR rec[-1]\n",
             "H 0\nT 0\nX 0\nRY 0\nMY 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
             "H 0\nT 0\nM 0\nR 0\nMY 0\nDETECTOR rec[-1]\n",
         }) {
        INFO(text);
        const auto circuit = parse(text);
        const auto expected = compute_dense_circuit_distribution(circuit);
        const auto hir = trace(circuit);
        const auto program = Program::compile(circuit, hir, true);
        const bool pure_reset_discards_all =
            std::string_view(text).find("M 0\nR 0") == std::string_view::npos;
        for (bool keep_records : {false, true}) {
            if (pure_reset_discards_all || !keep_records) {
                REQUIRE(program.uses_reconstruction(keep_records));
                REQUIRE(program.sampling_plan(keep_records).peak_active_width() == 0);
            }
            const auto mode =
                keep_records ? ReconstructionOutputs::Records : ReconstructionOutputs::Detectors;
            const auto res = reconstruct_global_sampling(circuit, mode);
            REQUIRE(res.reconstruction.has_value());
            auto lowered = lower_global_reconstruction(*res.reconstruction);
            REQUIRE(lowered.hir.has_value());
            default_hir_pass_manager().run(*lowered.hir);
            const auto actual =
                compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
            require_distributions_equal(project_outputs(expected, mode), actual, 2e-12);
        }
    }

    // Measure-reset on an isolated suffix qubit keeps the pre-reset record in
    // Records mode while still collapsing Detectors mode to width zero.
    {
        const auto circuit = parse("H 0\nT 0\nMRY 0\nM 0\nDETECTOR rec[-1]\n");
        const auto expected = compute_dense_circuit_distribution(circuit);
        const auto hir = trace(circuit);
        const auto program = Program::compile(circuit, hir, true);
        REQUIRE(program.uses_reconstruction(false));
        REQUIRE(program.sampling_plan(false).peak_active_width() == 0);
        for (auto mode : {ReconstructionOutputs::Detectors, ReconstructionOutputs::Records}) {
            const auto res = reconstruct_global_sampling(circuit, mode);
            REQUIRE(res.reconstruction.has_value());
            auto lowered = lower_global_reconstruction(*res.reconstruction);
            REQUIRE(lowered.hir.has_value());
            default_hir_pass_manager().run(*lowered.hir);
            const auto actual =
                compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
            require_distributions_equal(project_outputs(expected, mode), actual, 2e-12);
        }
    }

    // Prefix dilation checks max_schedule_operations before allocating unbounded
    // rotation gadgets or building the prefix frame.
    {
        std::string many_rotations;
        for (int i = 0; i < 20; ++i) {
            many_rotations += "H 0\nT 0\n";
        }
        many_rotations += "MX 0\nDETECTOR rec[-1]\n";
        ReconstructionLimits limits;
        limits.max_schedule_operations = 5;
        limits.max_dilation_qubits = 10;
        const auto rejected = reconstruct_global_sampling(parse(many_rotations),
                                                          ReconstructionOutputs::Detectors, limits);
        REQUIRE_FALSE(rejected.reconstruction.has_value());
        REQUIRE(rejected.fallback_reason == "schedule operation limit exceeded");
    }
}

TEST_CASE("Global reconstruction prefers rotation-stabilizing hidden nullspace candidates",
          "[reconstruction]") {
    for (const char* text : {
             "H 0\nCX 0 1\nT 0\nMPP Z1*X2\nMX 0\n",
             "H 0\nCX 0 1\nT 0\nMPP Y1*X2\nMX 0\n",
             "H 0\nCX 0 1\nCX 0 2\nTPP X0*Z1\nMPP Z2*X3\nMX 0\n",
         }) {
        INFO(text);
        const auto circuit = parse(text);
        const auto expected = compute_dense_circuit_distribution(circuit);
        auto hir = trace(circuit);
        default_hir_pass_manager().run(hir);
        const auto program = Program::compile(circuit, hir, true);
        REQUIRE(program.uses_reconstruction(true));
        REQUIRE(program.sampling_plan(true).peak_active_width() == 0);

        const auto res = reconstruct_global_sampling(circuit, ReconstructionOutputs::Records);
        REQUIRE(res.reconstruction.has_value());
        REQUIRE(res.reconstruction->peak_active_width == 0);
        auto lowered = lower_global_reconstruction(*res.reconstruction);
        REQUIRE(lowered.hir.has_value());
        default_hir_pass_manager().run(*lowered.hir);
        const auto actual = compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
        require_distributions_equal(expected, actual, 2e-12);
    }

    // Exercise the multi-word isolated-qubit check alongside an entangled suffix
    // reset across > 64 suffix modes without blowing up oracle width.
    {
        std::string wide_resets = "H 0\nT 0\nCX 0 1\n";
        for (uint32_t q = 2; q <= 70; ++q) {
            wide_resets += "R " + std::to_string(q) + "\n";
        }
        wide_resets += "RY 0\nM 0 1\nDETECTOR rec[-2] rec[-1]\n";
        const auto circuit = parse(wide_resets);
        const auto res = reconstruct_global_sampling(circuit, ReconstructionOutputs::Records);
        REQUIRE(res.reconstruction.has_value());
        REQUIRE(res.reconstruction->peak_active_width <= 1);
    }
}

TEST_CASE("Global reconstruction elides stabilized rotations and groups duplicate rotation axes",
          "[reconstruction]") {
    // Unmeasured magic outputs stabilized by synthesized nullspace measurements
    // must be dropped from the scheduled operation list rather than inflating
    // downstream phase-polynomial variable counts.
    {
        const auto circuit = parse("H 0\nCX 0 1\nT 0\nMPP Z1*X2\nMX 0\nDETECTOR rec[-1]\n");
        const auto res = reconstruct_global_sampling(circuit, ReconstructionOutputs::Detectors);
        REQUIRE(res.reconstruction.has_value());
        REQUIRE(res.reconstruction->peak_active_width == 0);
        REQUIRE(std::ranges::none_of(
            res.reconstruction->operations,
            [](const ReconstructionOperation& op) { return op.half_turns != 0.0; }));
    }

    // Commuting rotations on identical axes share one scheduler DAG node while
    // retaining all rotation steps in the emitted operation sequence and law.
    {
        std::string text = "H 0 1\n";
        for (int i = 0; i < 6; ++i) {
            text += "X_ERROR(0.05) 0\nT 0\nT_DAG 0\n";
        }
        text += "T 1\nH 0 1\nM 0 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
        const auto circuit = parse(text);
        const auto expected = compute_dense_circuit_distribution(circuit);

        // The limit admits all 13 source rotations. Scheduling each rotation
        // separately alongside the output operations would need 18 slots.
        ReconstructionLimits limits;
        limits.max_schedule_operations = 15;
        const auto res =
            reconstruct_global_sampling(circuit, ReconstructionOutputs::Records, limits);
        INFO(res.fallback_reason);
        REQUIRE(res.reconstruction.has_value());
        REQUIRE(res.reconstruction->peak_active_width == 1);
        const auto rotation_count = std::ranges::count_if(
            res.reconstruction->operations,
            [](const ReconstructionOperation& op) { return op.half_turns != 0.0; });
        REQUIRE(rotation_count == 13);

        auto lowered = lower_global_reconstruction(*res.reconstruction);
        REQUIRE(lowered.hir.has_value());
        default_hir_pass_manager().run(*lowered.hir);
        const auto actual = compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
        require_same_law(output_law(actual, res.reconstruction->outputs),
                         output_law(expected, res.reconstruction->outputs));
    }
}

}  // namespace clifft::test
