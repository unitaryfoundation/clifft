// Verifies native executor selection and jointly retained sampling outputs.

#include "clifft/api/sampler.h"
#include "clifft/api/sampling_preparation.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"

#include "dense_circuit_oracle.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <future>

using namespace clifft;
using namespace clifft::sampling;

namespace {

// Feedback, asymmetric readout, reference offsets, and byte-tail padding share
// one fixture so all retained modes exercise the same joint distribution.
Program row_program(bool postselect) {
    auto hir = trace(parse(R"(
        H 0
        T 0
        CX 0 1
        PAULI_CHANNEL_1(0.1, 0.05, 0.2) 0
        M(0.1) 0
        CX rec[-1] 1
        H 1
        T 1
        H 1
        M 1
        H 2
        T 2
        H 2
        M 2
        X_ERROR(0.2) 3
        M 3
        DETECTOR rec[-1]
        DETECTOR rec[-2] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-4]
        OBSERVABLE_INCLUDE(1) rec[-3]
        OBSERVABLE_INCLUDE(2) rec[-2]
        M 4 5 6 7 8 9 10 11 12
        OBSERVABLE_INCLUDE(8) rec[-1]
        DETECTOR rec[-1]
    )"));
    hir.postselection.mask = {static_cast<uint8_t>(postselect), static_cast<uint8_t>(postselect),
                              0};
    hir.postselection.expected_detectors = {0, 1, 0};
    hir.readout_noise.front().prob_one_to_zero = 0.3;
    default_hir_pass_manager().run(hir);
    return Program(ExecutablePlan(plan_sampling(hir)));
}

void require_same(const SamplingSurvivorResult& a, const SamplingSurvivorResult& b) {
    REQUIRE(a.execution == b.execution);
    REQUIRE(a.total_shots == b.total_shots);
    REQUIRE(a.passed_shots == b.passed_shots);
    REQUIRE(a.logical_errors == b.logical_errors);
    REQUIRE(a.observable_ones == b.observable_ones);
    REQUIRE(a.measurements == b.measurements);
    REQUIRE(a.detectors == b.detectors);
    REQUIRE(a.observables == b.observables);
    REQUIRE(a.exp_vals == b.exp_vals);
}

void require_packed(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& bytes,
                    uint32_t columns, uint32_t rows) {
    const size_t stride = (columns + 7) / 8;
    REQUIRE(packed.size() == rows * stride);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < columns; ++col) {
            REQUIRE(((packed[row * stride + col / 8] >> (col % 8)) & 1) ==
                    bytes[row * columns + col]);
        }
        if (columns % 8 != 0) {
            REQUIRE((packed[row * stride + stride - 1] >> (columns % 8)) == 0);
        }
    }
}

}  // namespace

TEST_CASE("Native shared rows preserve selective outputs and bit packing") {
    const auto mode =
        GENERATE(RecordOutputs::All, RecordOutputs::Syndromes, RecordOutputs::Measurements);
    const bool postselect = GENERATE(false, true);
    const auto noise =
        GENERATE(BatchNoiseMode::Dense, BatchNoiseMode::SiteSparse, BatchNoiseMode::SiteRows);
    const Program program = row_program(postselect);
    const SamplingOptions options{.threads = 3,
                                  .batch_size = 129,
                                  .sharing = true,
                                  .group_by_noise = noise == BatchNoiseMode::SiteRows,
                                  .noise_preparation = noise};
    const auto unpacked = clifft::sample_survivors(program, 1301, 7, {mode}, options);
    const auto packed = clifft::sample_survivors(program, 1301, 7, {mode, true}, options);
    REQUIRE(unpacked.execution.executor == SamplingExecutor::Shared);
    REQUIRE(unpacked.passed_shots > 0);
    REQUIRE((unpacked.passed_shots < 1301) == postselect);
    REQUIRE(packed.passed_shots == unpacked.passed_shots);
    REQUIRE(packed.observable_ones == unpacked.observable_ones);
    if (SamplingOutputs(mode).measurements()) {
        require_packed(packed.measurements, unpacked.measurements, 13, packed.passed_shots);
    } else {
        REQUIRE(packed.measurements.empty());
    }
    if (SamplingOutputs(mode).syndromes()) {
        require_packed(packed.detectors, unpacked.detectors, 3, packed.passed_shots);
        require_packed(packed.observables, unpacked.observables, 9, packed.passed_shots);
    } else {
        REQUIRE(packed.detectors.empty());
        REQUIRE(packed.observables.empty());
    }
    if (mode == RecordOutputs::All) {
        for (uint32_t row = 0; row < unpacked.passed_shots; ++row) {
            const auto* m = unpacked.measurements.data() + row * 13;
            const auto* d = unpacked.detectors.data() + row * 3;
            const auto* o = unpacked.observables.data() + row * 9;
            REQUIRE(d[0] == m[3]);
            REQUIRE(d[1] == (m[1] ^ m[2] ^ 1));
            REQUIRE(o[0] == m[0]);
            REQUIRE(o[1] == m[1]);
            REQUIRE(o[2] == m[2]);
            if (postselect) {
                REQUIRE(d[0] == 0);
                REQUIRE(d[1] == 0);
            }
        }
    }
    if (!postselect) {
        const auto ordinary = clifft::sample(program, 1301, 7, {mode, true}, options);
        REQUIRE(ordinary.measurements == packed.measurements);
        REQUIRE(ordinary.detectors == packed.detectors);
        REQUIRE(ordinary.observables == packed.observables);
    }
}

TEST_CASE("Native shared calls replay across preparation history and concurrency") {
    const auto mode = GENERATE(RecordOutputs::None, RecordOutputs::All, RecordOutputs::Syndromes,
                               RecordOutputs::Measurements);
    const SamplingOutputs outputs(mode, true);
    const SamplingOptions options{.threads = 4, .batch_size = 257, .sharing = true};
    const auto program = row_program(true);
    const auto cold = clifft::sample_survivors(program, 8193, 19, outputs, options);
    (void)clifft::sample_survivors(program, 33, 8, true, {.batch_size = 1});
    (void)clifft::sample_survivors(program, 1024, 8, false,
                                   {.batch_size = 129, .sharing = true, .group_by_noise = true});
    require_same(cold, clifft::sample_survivors(program, 8193, 19, outputs, options));
    const auto fresh = row_program(true);
    std::array<std::future<SamplingSurvivorResult>, 4> calls;
    for (auto& call : calls) {
        call = std::async(std::launch::async, [&] {
            return clifft::sample_survivors(fresh, 8193, 19, outputs, options);
        });
    }
    for (auto& call : calls) {
        require_same(cold, call.get());
    }
    auto serial = options;
    serial.threads = 1;
    auto serial_result = clifft::sample_survivors(program, 8193, 19, outputs, serial);
    serial_result.execution = cold.execution;
    require_same(cold, serial_result);
}

TEST_CASE("Native shared joint distribution agrees with scalar sampling") {
    const Program program = row_program(true);
    constexpr uint32_t shots = 32768;
    const auto reference = clifft::sample_survivors(program, shots, 919, true, {.batch_size = 1});
    const auto shared = clifft::sample_survivors(program, shots, 32, true, {.sharing = true});
    std::array<uint32_t, 9> expected{}, actual{};
    const auto count = [](const auto& result, auto& histogram) {
        histogram[8] = result.total_shots - result.passed_shots;
        for (uint32_t row = 0; row < result.passed_shots; ++row) {
            const auto* m = result.measurements.data() + row * 13;
            ++histogram[m[0] | (m[1] << 1) | (m[2] << 2)];
        }
    };
    count(reference, expected);
    count(shared, actual);
    for (size_t i = 0; i < actual.size(); ++i) {
        const double p = double(expected[i] + actual[i]) / (2 * shots);
        REQUIRE(std::abs(double(expected[i]) - actual[i]) <
                8 * std::sqrt(2 * shots * p * (1 - p)) + 8);
    }
}

TEST_CASE("Native sampling preserves unsupported outputs and numeric ordinary batches") {
    auto hir = trace(parse("H 0\nT 0\nEXP_VAL X0\nH 0\nM 0\n"));
    const Program probes{ExecutablePlan(plan_sampling(hir))};
    const auto rows = clifft::sample(probes, 4096, 9);
    REQUIRE(rows.execution.executor != SamplingExecutor::Shared);
    REQUIRE(rows.exp_vals.size() == 4096);
    REQUIRE_THROWS_AS(clifft::sample(probes, 17, 9, {}, {.sharing = true}), std::invalid_argument);
    const auto omitted =
        clifft::sample(probes, 65, 9, {RecordOutputs::Measurements}, {.sharing = true});
    REQUIRE(omitted.execution.executor == SamplingExecutor::Shared);
    REQUIRE(omitted.exp_vals.empty());
    const auto program = row_program(true);
    const auto native =
        clifft::sample_survivors(program, 8193, 19, true, {.threads = 2, .batch_size = 65});
    const auto& preparation = program.sampling_preparation(true, {.early_postselection = false});
    const auto ordinary = sampling::sample_survivors(preparation.plan(), 8193, 19, true, 2, {}, 65,
                                                     {}, &preparation.noise(BatchNoiseMode::Dense));
    require_same(native, ordinary);
}

TEST_CASE("Native shared sampling validates budgets and handles empty output") {
    const auto program = row_program(true);
    REQUIRE_THROWS_AS(clifft::sample_survivors(program, 65, 1, false,
                                               {.sharing = true, .max_worker_state_bytes = 1}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        clifft::sample_survivors(program, 65, 1, false, {.batch_size = 1, .sharing = true}),
        std::invalid_argument);
    auto hir = trace(parse("H 0\nT 0\nH 0\nM 0\nM 1\nDETECTOR rec[-1]\n"));
    hir.postselection.mask = {1};
    hir.postselection.expected_detectors = {1};
    const Program impossible{ExecutablePlan(plan_sampling(hir))};
    for (uint32_t shots : {0, 1, 65, 1301}) {
        for (const SamplingOptions options :
             {SamplingOptions{.batch_size = 1}, SamplingOptions{.threads = 3, .batch_size = 65},
              SamplingOptions{.threads = 3, .sharing = true}}) {
            const auto result = clifft::sample_survivors(impossible, shots, 1, true, options);
            REQUIRE(result.total_shots == shots);
            REQUIRE(result.passed_shots == 0);
            REQUIRE(result.measurements.capacity() == 0);
        }
    }
}

TEST_CASE("Native automatic selection depends on the call and meaningful sharing opportunities") {
    std::string source = "H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    for (uint32_t q = 1; q <= 8; ++q) {
        source += "X_ERROR(0.01) " + std::to_string(q) + "\nM " + std::to_string(q) +
                  "\nDETECTOR rec[-1]\n";
    }
    auto hir = trace(parse(source));
    hir.postselection.mask.assign(8, 1);
    const Program program{ExecutablePlan(plan_sampling(hir))};
    const auto cold = clifft::sample_survivors(program, 16384, 42, true, {.threads = 2});
    REQUIRE(cold.execution.executor == SamplingExecutor::Shared);
    REQUIRE(cold.execution.batch_size == 8192);
    REQUIRE(cold.execution.shot_workers == 2);
    REQUIRE(cold.execution.early_postselection);
    const auto short_call = clifft::sample_survivors(program, 1024, 42, true, {.threads = 2});
    REQUIRE(short_call.execution.executor != SamplingExecutor::Shared);
    const auto more_workers = clifft::sample_survivors(program, 16384, 42, true, {.threads = 4});
    REQUIRE(more_workers.execution.executor != SamplingExecutor::Shared);
    const auto small_budget = clifft::sample_survivors(
        program, 16384, 42, true, {.max_worker_state_bytes = State::storage_bytes(1)});
    REQUIRE(small_budget.execution.executor != SamplingExecutor::Shared);
    require_same(cold, clifft::sample_survivors(program, 16384, 42, true, {.threads = 2}));
    const Program fresh{ExecutablePlan(plan_sampling(hir))};
    auto concurrent = std::async(std::launch::async, [&] {
        return clifft::sample_survivors(fresh, 16384, 42, true, {.threads = 2});
    });
    require_same(cold, clifft::sample_survivors(fresh, 16384, 42, true, {.threads = 2}));
    require_same(cold, concurrent.get());
    auto unselected = hir;
    unselected.postselection.mask.assign(8, 0);
    const Program noisy{ExecutablePlan(plan_sampling(unselected))};
    REQUIRE(clifft::sample(noisy, 16384, 42).execution.executor != SamplingExecutor::Shared);

    auto unitary = trace(parse("H 0 1 2 3 4 5 6 7\nT 0 1 2 3 4 5 6 7\nM 0 1 2 3 4 5 6 7"));
    const Program terminal{ExecutablePlan(plan_sampling(unitary))};
    REQUIRE(terminal.original().peak_active_width() == 8);
    REQUIRE_FALSE(terminal.original().has_promotions_after_measurement());
    REQUIRE(clifft::sample(terminal, 256, 42).execution.executor == SamplingExecutor::Shared);
    REQUIRE(clifft::sample(terminal, 16, 42).execution.executor != SamplingExecutor::Shared);
    const auto branching_hir =
        trace(parse("H 0 1 2 3 4 5 6 7\nT 0 1 2 3 4 5 6 7\nM 0\nH 0\nT 0\nM 0 1 2 3 4 5 6 7"));
    const Program branching{ExecutablePlan(plan_sampling(branching_hir))};
    REQUIRE(branching.original().has_promotions_after_measurement());
    REQUIRE(clifft::sample(branching, 256, 42).execution.executor != SamplingExecutor::Shared);

    const auto qv_source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/qv10.stim");
    auto qv_hir = trace(qv_source);
    default_hir_pass_manager().run(qv_hir);
    const auto qv = Program::compile(qv_source, qv_hir, true);
    REQUIRE_FALSE(qv.sampling_plan(true).has_promotions_after_measurement());
    REQUIRE(clifft::sample(qv, 256, 42).execution.executor == SamplingExecutor::Shared);
}

TEST_CASE("Native shared calibration replays with the selected settings") {
    const Program program = row_program(true);
    const SamplingOptions options{.threads = 2, .tuning_budget_seconds = 0.01, .sharing = true};
    const auto tuned = clifft::sample_survivors(program, 4097, 42, true, options);
    REQUIRE(tuned.batch_tuning.has_value());
    REQUIRE(tuned.execution.executor == SamplingExecutor::Shared);
    REQUIRE(tuned.execution.batch_size == tuned.batch_tuning->batch_size);
    REQUIRE(tuned.execution.shot_workers == tuned.batch_tuning->shot_workers);
    for (const auto& trial : tuned.batch_tuning->trials) {
        REQUIRE(trial.batch_size > 1);
    }
    const auto replay =
        clifft::sample_survivors(program, 4097, 42, true,
                                 {.threads = tuned.execution.shot_workers,
                                  .batch_size = tuned.execution.batch_size,
                                  .sharing = true,
                                  .group_by_noise = tuned.execution.group_by_noise,
                                  .noise_preparation = tuned.execution.noise_preparation});
    require_same(tuned, replay);
    REQUIRE_THROWS_AS(clifft::sample_survivors(program, 65, 1, false,
                                               {.batch_size = 65,
                                                .sharing = true,
                                                .max_worker_state_bytes = State::storage_bytes(
                                                    program.original().peak_active_width())}),
                      std::runtime_error);
    REQUIRE(clifft::sample_survivors(program, 65, 1, false, {.sharing = true}).total_shots == 65);
}

TEST_CASE("Native shared joint outputs match the dense circuit oracle") {
    const bool postselect = GENERATE(false, true);
    const SamplingOutputs outputs(
        GENERATE(RecordOutputs::All, RecordOutputs::Syndromes, RecordOutputs::Measurements));
    const auto circuit = parse(R"(
        H 0
        T 0
        CX 0 1
        PAULI_CHANNEL_1(0.1, 0.05, 0.2) 0
        M(0.1) 0
        CX rec[-1] 1
        H 1
        T 1
        H 1
        M 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )");
    const std::array<uint8_t, 1> mask{static_cast<uint8_t>(postselect)}, reference{1};
    const SamplingPlanOptions plan_options{.postselection_mask = mask,
                                           .expected_detectors = reference,
                                           .expected_observables = reference};
    const auto oracle = clifft::test::compute_dense_circuit_distribution(circuit, plan_options);
    const Program program{ExecutablePlan(plan_sampling(trace(circuit), plan_options))};
    constexpr uint32_t shots = 32768;
    const auto result = clifft::sample_survivors(program, shots, 81, outputs, {.sharing = true});
    REQUIRE(result.execution.executor == SamplingExecutor::Shared);
    std::array<double, 17> expected{};
    std::array<uint32_t, 17> actual{};
    expected[16] = oracle.discarded_probability;
    actual[16] = shots - result.passed_shots;
    for (const auto& entry : oracle.kept_outcomes) {
        const auto& outcome = entry.outcome;
        expected[(outputs.measurements() ? outcome.visible_records : 0) |
                 (outputs.syndromes() ? (outcome.detectors << 2) | (outcome.observables << 3)
                                      : 0)] += entry.probability;
    }
    for (uint32_t row = 0; row < result.passed_shots; ++row) {
        ++actual[(outputs.measurements()
                      ? result.measurements[2 * row] | (result.measurements[2 * row + 1] << 1)
                      : 0) |
                 (outputs.syndromes()
                      ? (result.detectors[row] << 2) | (result.observables[row] << 3)
                      : 0)];
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] == 0) {
            REQUIRE(actual[i] == 0);
        } else {
            REQUIRE(std::abs(actual[i] - shots * expected[i]) <
                    8 * std::sqrt(shots * expected[i] * (1 - expected[i])) + 8);
        }
    }
}
