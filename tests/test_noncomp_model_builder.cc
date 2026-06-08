#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::ClassifierSpec;
using clifft::GateType;
using clifft::LevelSet;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;

namespace {

// Default-set ids: g=0, e=1, leak_g=2, leak_e=3, lost=4.
constexpr uint8_t kLost = 4;

std::vector<std::vector<double>> zeros(size_t rows, size_t cols) {
    return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

// T[to][from]: g and e both jump to lost with certainty.
std::vector<std::vector<double>> lost_transition() {
    auto m = zeros(5, 5);
    m[kLost][0] = 1.0;
    m[kLost][1] = 1.0;
    return m;
}

std::map<std::string, std::vector<std::vector<double>>> lost_on(const std::string& gate) {
    std::map<std::string, std::vector<std::vector<double>>> t;
    t.emplace(gate, lost_transition());
    return t;
}

ClassifierSpec binary_lost_classifier() {
    ClassifierSpec spec;
    spec.symbols = {"0", "1"};
    spec.matrix = zeros(2, 5);
    spec.matrix[0][kLost] = 1.0;  // lost -> symbol 0
    return spec;
}

std::vector<double> all_g() {
    return {1.0, 0.0, 0.0, 0.0, 0.0};
}

}  // namespace

TEST_CASE("from_spec: builds a working model from raw matrices") {
    NonComputationalModel model =
        NonComputationalModel::from_spec(LevelSet::default_set(), all_g(), lost_on("S"),
                                         binary_lost_classifier(), NonComputationalPolicy{});

    REQUIRE(model.num_levels() == 5);
    REQUIRE(model.transition_for(GateType::S) != nullptr);
    REQUIRE(model.classifier() != nullptr);
    REQUIRE(model.initial_probability(0) == 1.0);
}

TEST_CASE("from_spec: a model without a classifier reports none") {
    NonComputationalModel model = NonComputationalModel::from_spec(
        LevelSet::default_set(), all_g(), lost_on("S"), std::nullopt, NonComputationalPolicy{});
    REQUIRE(model.classifier() == nullptr);
}

TEST_CASE("from_spec: a Stim gate alias resolves to the canonical transition key") {
    // "CNOT" canonicalizes to CX; transition_for(CX) must find it.
    NonComputationalModel model = NonComputationalModel::from_spec(
        LevelSet::default_set(), all_g(), lost_on("CNOT"), std::nullopt, NonComputationalPolicy{});
    REQUIRE(model.transition_for(GateType::CX) != nullptr);
}

TEST_CASE("from_spec: an unknown gate key raises") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(LevelSet::default_set(), all_g(), lost_on("NOTAGATE"),
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("transition key"));
}

TEST_CASE("from_spec: a classifier matrix with the wrong level count raises") {
    ClassifierSpec spec;
    spec.symbols = {"0", "1"};
    spec.matrix = zeros(2, 4);  // 4 level columns, table has 5
    REQUIRE_THROWS_AS(
        NonComputationalModel::from_spec(LevelSet::default_set(), all_g(), lost_on("S"), spec,
                                         NonComputationalPolicy{}),
        std::invalid_argument);
}

TEST_CASE("from_spec: a transition matrix with the wrong shape raises") {
    std::map<std::string, std::vector<std::vector<double>>> t;
    t.emplace("S", zeros(4, 4));  // 4x4, table has 5 levels
    REQUIRE_THROWS_AS(NonComputationalModel::from_spec(LevelSet::default_set(), all_g(), t,
                                                       std::nullopt, NonComputationalPolicy{}),
                      std::invalid_argument);
}

TEST_CASE("from_spec: an initial-state vector that does not sum to one raises") {
    REQUIRE_THROWS_AS(
        NonComputationalModel::from_spec(LevelSet::default_set(), {0.5, 0.0, 0.0, 0.0, 0.0},
                                         lost_on("S"), std::nullopt, NonComputationalPolicy{}),
        std::invalid_argument);
}
