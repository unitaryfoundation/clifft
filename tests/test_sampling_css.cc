#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/css/planner.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace clifft;
using namespace clifft::sampling;

namespace {
Circuit fixture(unsigned distance, char terminal = 'Z', bool readout = false) {
    std::ifstream stream(std::string(CLIFFT_FIXTURES_DIR) +
                         "/../../tools/bench/fixtures/css_five_check_d" + std::to_string(distance) +
                         ".stim");
    REQUIRE(stream.good());
    std::stringstream text;
    text << stream.rdbuf();
    auto circuit = parse(text.str());
    if (!readout)
        std::erase_if(circuit.nodes,
                      [](const auto& op) { return op.gate == GateType::READOUT_NOISE; });
    auto& last = *std::find_if(circuit.nodes.rbegin(), circuit.nodes.rend(),
                               [](const auto& op) { return op.gate == GateType::MPP; });
    for (auto& target : last.targets)
        target = Target::pauli(target.value(), terminal == 'X'   ? Target::kPauliX
                                               : terminal == 'Y' ? Target::kPauliY
                                                                 : Target::kPauliZ);
    return circuit;
}
SamplingPlan contracted(const Circuit& circuit, SamplingPlanOptions options = {}) {
    auto candidate = try_plan_css_blocks(circuit, options, 1);
    INFO(candidate.reason);
    REQUIRE(candidate.plan.has_value());
    return std::move(*candidate.plan);
}
ApplyCssBlock& first_block(SamplingPlan& plan) {
    for (auto& action : plan.actions)
        if (auto* block = std::get_if<ApplyCssBlock>(&action.action))
            return *block;
    throw std::logic_error("missing block");
}
}  // namespace

TEST_CASE("CSS block histories agree with ordinary physical execution", "[css]") {
    for (unsigned distance : {3u, 5u, 7u})
        for (char axis : {'X', 'Y', 'Z'}) {
            auto circuit = fixture(distance, axis);
            auto candidate = contracted(circuit);
            auto ordinary = plan_sampling(trace(circuit));
            REQUIRE(prefer_css_blocks(ordinary, candidate) == (distance >= 7));
            ExecutablePlan a(candidate), b(ordinary);
            REQUIRE(a.num_css_blocks() == 5);
            REQUIRE(a.peak_active_width() == 1);
            REQUIRE(a.num_presampled_symbols() == b.num_presampled_symbols());
            Executor x(a, 15), y(b, 91);
            Xoshiro256PlusPlus random(772);
            std::vector<uint8_t> faults(a.num_presampled_symbols());
            for (unsigned shot = 0; shot < 12; ++shot) {
                std::fill(faults.begin(), faults.end(), 0);
                for (unsigned site = 0; site < faults.size() / 3; ++site)
                    if (shot && random.next_double() < .1)
                        faults[3 * site + random() % 3] = 1;
                // Compare in both directions to catch missing branches as well
                // as incorrect weights on reachable contracted histories.
                auto& source = shot % 2 ? x : y;
                source.run_shot(faults);
                std::vector<uint8_t> records(source.visible_records().begin(),
                                             source.visible_records().end());
                auto p = x.replay_shot(records, faults), q = y.replay_shot(records, faults);
                INFO(distance << " " << axis << " " << shot);
                REQUIRE(p.reachable);
                REQUIRE(q.reachable);
                REQUIRE(std::abs(p.log_probability - q.log_probability) < 1e-9);
                REQUIRE(
                    std::equal(x.detectors().begin(), x.detectors().end(), y.detectors().begin()));
                REQUIRE(std::equal(x.observables().begin(), x.observables().end(),
                                   y.observables().begin()));
            }
        }
}

TEST_CASE("CSS certificate declines unsupported physical structure", "[css]") {
    auto good = fixture(3);
    SECTION("small circuits are outside the default selection") {
        REQUIRE_FALSE(try_plan_css_blocks(good).plan);
    }
    SECTION("prefix must prepare the positive code space") {
        good.nodes.erase(good.nodes.begin());
        REQUIRE_FALSE(try_plan_css_blocks(good, {}, 1).plan);
    }
    SECTION("suffix cannot conceal unsupported work") {
        good.nodes.push_back(parse("H 0").nodes[0]);
        REQUIRE_FALSE(try_plan_css_blocks(good, {}, 1).plan);
    }
    SECTION("partial projections cannot form a block") {
        auto it = std::find_if(good.nodes.begin(), good.nodes.end(), [](const auto& op) {
            return op.gate == GateType::MPP && op.targets[0].pauli_char() == 'X';
        });
        good.nodes.erase(it);
        REQUIRE_FALSE(try_plan_css_blocks(good, {}, 1).plan);
    }
    SECTION("retained source maps use the ordinary planner") {
        SamplingPlanOptions options;
        options.retain_source_map = true;
        REQUIRE_FALSE(try_plan_css_blocks(good, options, 1).plan);
    }
}

TEST_CASE("CSS prefix certificate preserves logical phase and Clifford preparation", "[css]") {
    for (bool dagger : {false, true}) {
        auto circuit = fixture(3, 'Y');
        auto first_t = std::find_if(circuit.nodes.begin(), circuit.nodes.end(),
                                    [](const auto& op) { return op.gate == GateType::T; });
        if (dagger)
            first_t->gate = GateType::T_DAG;
        else
            circuit.nodes.erase(first_t);
        ExecutablePlan a(contracted(circuit)), b(plan_sampling(trace(circuit)));
        Executor x(a, 167), y(b, 167);
        std::vector<uint8_t> faults(a.num_presampled_symbols());
        for (unsigned shot = 0; shot < 16; ++shot) {
            x.run_shot(faults);
            std::vector<uint8_t> records(x.visible_records().begin(), x.visible_records().end());
            auto p = x.replay_shot(records, faults), q = y.replay_shot(records, faults);
            REQUIRE(p.reachable);
            REQUIRE(q.reachable);
            REQUIRE(std::abs(p.log_probability - q.log_probability) < 1e-10);
        }
    }
}

TEST_CASE("CSS actions validate branch ownership and input dependencies", "[css]") {
    auto plan = contracted(fixture(3));
    auto& block = first_block(plan);
    SECTION("missing input") {
        block.inputs.pop_back();
    }
    SECTION("missing certificate") {
        block.code.reset();
    }
    SECTION("duplicate branch") {
        block.branches[1] = block.branches[0];
    }
    SECTION("duplicate record") {
        block.records[1] = block.records[0];
    }
    SECTION("own branch cannot be an input") {
        block.inputs[0] = AffineBool::symbol(block.branches[0]);
    }
    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
}

TEST_CASE("CSS sampling composes with readout outputs and fixed fault count", "[css]") {
    auto circuit = fixture(3, 'Z', true);
    ExecutablePlan plan(contracted(circuit));
    auto a = sample(plan, 32, 174, 1), b = sample(plan, 32, 174, 2);
    REQUIRE(a.measurements == b.measurements);
    REQUIRE(a.detectors == b.detectors);
    for (unsigned shot = 0; shot < 32; ++shot) {
        auto offset = shot * circuit.num_measurements;
        REQUIRE(a.observables[shot] == a.measurements[offset + circuit.num_measurements - 1]);
        for (unsigned round = 1; round < 5; ++round)
            for (unsigned check = 1; check < 7; ++check)
                REQUIRE(a.detectors[shot * 24 + (round - 1) * 6 + check - 1] ==
                        (a.measurements[offset + round * 7 + check] ^
                         a.measurements[offset + (round - 1) * 7 + check]));
    }
    REQUIRE_THROWS_AS(sample(plan, 1, 1, 1, std::nullopt, 8), std::invalid_argument);
    auto zero_faults = sample_k(plan, 32, 0, 81);
    REQUIRE(std::all_of(zero_faults.detectors.begin(), zero_faults.detectors.end(),
                        [](auto bit) { return bit == 0; }));
    std::vector<uint8_t> mask(circuit.num_detectors, 1);
    ExecutablePlan post(contracted(circuit, {mask, {}, {}}));
    auto survivors = sample_k_survivors(post, 32, 0, 81, true);
    REQUIRE(survivors.passed_shots == 32);
    REQUIRE(survivors.measurements == zero_faults.measurements);
}

TEST_CASE("CSS large control keeps a two coefficient logical state", "[css]") {
    auto circuit = fixture(9);
    ExecutablePlan plan(contracted(circuit));
    REQUIRE(plan.peak_active_width() == 1);
    REQUIRE(plan.num_css_blocks() == 5);
    auto result = sample(plan, 8, 512);
    REQUIRE(result.measurements.size() == 8 * circuit.num_measurements);
}

TEST_CASE("CSS fixed fault count forces the selected physical or readout site", "[css]") {
    auto circuit = fixture(3, 'Z', true);
    for (auto& op : circuit.nodes)
        if (op.gate == GateType::DEPOLARIZE1 || op.gate == GateType::READOUT_NOISE)
            op.args = {0};
    SECTION("physical fault in the last block") {
        auto site = std::find_if(circuit.nodes.rbegin(), circuit.nodes.rend(), [](const auto& op) {
            return op.gate == GateType::DEPOLARIZE1 && op.targets[0].value() == 0;
        });
        site->gate = GateType::X_ERROR;
        site->args = {.125};
    }
    SECTION("readout fault in the last CSS projection") {
        auto site = std::find_if(circuit.nodes.rbegin(), circuit.nodes.rend(), [](const auto& op) {
            return op.gate == GateType::READOUT_NOISE && op.targets[0].value() == 34;
        });
        site->args = {.125};
    }
    ExecutablePlan a(contracted(circuit)), b(plan_sampling(trace(circuit)));
    auto physical = sample_k(b, 64, 1, 167);
    auto blocks = sample_k(a, 64, 1, 167);
    REQUIRE(blocks.detectors == physical.detectors);
    REQUIRE(
        std::any_of(blocks.detectors.begin(), blocks.detectors.end(), [](auto b) { return b; }));
}

TEST_CASE("CSS readout changes reported bits without replacing the true branch", "[css]") {
    for (double reported : {0., 1.}) {
        auto circuit = fixture(3, 'Y', true);
        for (auto& op : circuit.nodes)
            if (op.gate == GateType::READOUT_NOISE)
                op.args = {reported, 1 - reported};
        ExecutablePlan plan(contracted(circuit));
        auto result = sample(plan, 32, 791);
        for (auto bit : result.measurements)
            REQUIRE(bit == reported);
    }
}

TEST_CASE("CSS replay rejects an inconsistent Z syndrome", "[css]") {
    auto circuit = fixture(3);
    ExecutablePlan plan(contracted(circuit));
    Executor executor(plan, 29);
    std::vector<uint8_t> faults(plan.num_presampled_symbols());
    executor.run_shot(faults);
    std::vector<uint8_t> records(executor.visible_records().begin(),
                                 executor.visible_records().end());
    records[2] ^= 1;
    REQUIRE_FALSE(executor.replay_shot(records, faults).reachable);
}

TEST_CASE("CSS factor planning enforces independence and storage bounds", "[css]") {
    REQUIRE_THROWS_AS(css::Code(3, {{true, 3}, {false, 5}}), std::invalid_argument);
    REQUIRE_THROWS_AS(css::Code(5, {{true, 3}, {true, 3}, {false, 3}, {false, 3}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(css::Code(3, {{true, 3}, {false, 3}}, 1), std::invalid_argument);
}

TEST_CASE("CSS replay agrees with independent reference trajectories", "[css]") {
    for (unsigned distance : {3u, 9u}) {
        const char* path = distance == 3 ? "/css_aer_histories.txt" : "/css_ch_histories.txt";
        std::ifstream stream(std::string(CLIFFT_FIXTURES_DIR) + path);
        REQUIRE(stream.good());
        std::string line;
        unsigned count = 0;
        while (std::getline(stream, line)) {
            if (line.starts_with('#'))
                continue;
            std::istringstream row(line);
            char axis;
            std::string fault_word, record_word;
            double expected;
            REQUIRE(bool(row >> axis >> fault_word >> record_word >> expected));
            std::vector<uint8_t> faults, records;
            for (char c : fault_word)
                faults.push_back(c - '0');
            for (char c : record_word)
                records.push_back(c - '0');
            ExecutablePlan plan(contracted(fixture(distance, axis)));
            REQUIRE(faults.size() == plan.num_presampled_symbols());
            Executor executor(plan);
            auto result = executor.replay_shot(records, faults);
            REQUIRE(result.reachable);
            REQUIRE(std::abs(result.log_probability - expected) < 1e-10);
            ++count;
        }
        REQUIRE(count == (distance == 3 ? 6 : 3));
    }
}
