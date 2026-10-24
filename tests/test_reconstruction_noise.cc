#include "clifft/optimizer/reconstruction_noise.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <map>
#include <utility>

namespace clifft::test {
namespace {

using SignLaw = std::map<std::vector<uint32_t>, double>;

// Enumerate whole categorical outcomes, then evaluate each sign directly. This
// oracle uses neither the compiler's binary basis nor its channel projection.
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

ReconstructionSign sign(std::initializer_list<uint32_t> bits, bool constant = false) {
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

TEST_CASE("Noise compression preserves correlated affine sign laws", "[reconstruction]") {
    GlobalReconstruction r;
    r.num_noise_bits = 130;
    r.operations = {{PauliString(0), sign({0, 65}, true), 0.25, -1},
                    {PauliString(0), sign({65, 129}), 0, 0},
                    {PauliString(0), sign({0, 129}, true), 0, 1}};
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
        r.operations.push_back({PauliString(0), sign({i}, i % 2 != 0), 0, int32_t(i)});
    r.noise_channels = {{{0.1, {}}, {0.2, {0, 65}}, {0.7, {69}}}, {{0.4, {}}, {0.6, {65, 69}}}};
    const auto noise = check_sign_law(r);
    REQUIRE(noise.num_bits == 70);
}

TEST_CASE("Noise compression removes unobserved channels and retains constants",
          "[reconstruction]") {
    GlobalReconstruction r;
    r.num_noise_bits = 2;
    r.operations = {{PauliString(0), sign({}, true), 0, 0}, {PauliString(0), sign({}), 0, 1}};
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
    r.operations = {{PauliString(0), sign({0}), 0, 0}};
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
}

}  // namespace clifft::test
