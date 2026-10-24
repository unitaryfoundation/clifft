#include "clifft/circuit/parser.h"
#include "clifft/optimizer/global_reconstruction.h"

#include "reconstruction_test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <map>

namespace clifft::test {
namespace {

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
        if (op.output >= 0)
            count = std::max(count, uint32_t(op.output + 1));
    std::map<uint64_t, double> law;
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
                    b.next_meas = op.output >= 0 ? uint32_t(op.output) : 0;
                    branch_projective_measurement(b, r.num_qubits, {x, z, false}, op.output >= 0,
                                                  sign, nullptr, next);
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
    const auto result = reconstruct_global_sampling(parse("MPAD 1 0 1\nDETECTOR rec[-1] rec[-2]\n"),
                                                    ReconstructionOutputs::Records);
    REQUIRE(result.reconstruction.has_value());
    const auto law = reconstructed_law(*result.reconstruction);
    REQUIRE(law.size() == 1);
    REQUIRE_THAT(law.at(13), Catch::Matchers::WithinAbs(1, 2e-12));
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

}  // namespace clifft::test
