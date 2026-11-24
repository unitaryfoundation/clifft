// Verifies native executor selection and jointly retained sampling outputs.

#include "clifft/api/sampler.h"
#include "clifft/api/sampling_preparation.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/batch/output.h"
#include "clifft/sampling/planner.h"

#include "dense_circuit_oracle.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <future>
#include <string>
#include <utility>

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

// Width-zero circuit with three noise sites that expect 4.5 * p faults per
// shot, plus correlated faults, readout noise, and measurement feedback.
Circuit clifford_noise_circuit(double p) {
    const std::string site = std::to_string(p);
    return parse("H 0\nCX 0 1\nDEPOLARIZE2(" + site + ") 0 1\nX_ERROR(" + site +
                 ") 2\nCX 1 2\nM(0.05) 0 1\nCX rec[-1] 2\nPAULI_CHANNEL_1(" + site + ", " +
                 std::to_string(p / 2) + ", " + site +
                 ") 2\nM 2\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\n"
                 "OBSERVABLE_INCLUDE(0) rec[-1]\n");
}

// Postselects the first detector so the second still varies among survivors.
Program clifford_noise_program(double p, bool postselect) {
    const std::array<uint8_t, 2> mask{static_cast<uint8_t>(postselect), 0};
    return Program(ExecutablePlan(
        plan_sampling(trace(clifford_noise_circuit(p)), {.postselection_mask = mask})));
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
        const auto fixed = clifft::sample(program, 1301, 7, {mode, true}, options);
        REQUIRE(fixed.measurements == packed.measurements);
        REQUIRE(fixed.detectors == packed.detectors);
        REQUIRE(fixed.observables == packed.observables);
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
    REQUIRE(cold.execution.sharing);
    REQUIRE(cold.execution.batch_size == 8192);
    REQUIRE(cold.execution.shot_workers == 2);
    REQUIRE(cold.execution.outputs == SamplingOutputs(true));
    REQUIRE(cold.execution.max_worker_state_bytes == kDefaultSharedWorkerStateBudget);
    const auto& plan = program.sampling_plan(true);
    REQUIRE_FALSE(should_share_sampling(plan, 63, true, {1, 1}, kDefaultSharedWorkerStateBudget));
    REQUIRE(should_share_sampling(plan, 64, true, {1, 1}, kDefaultSharedWorkerStateBudget));
    REQUIRE(should_share_sampling(plan, 16384, true, {4, 1}, kDefaultSharedWorkerStateBudget));
    REQUIRE_FALSE(should_share_sampling(plan, 16384, true, {1, 1}, State::storage_bytes(1)));
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
    const auto ordinary = clifft::sample(noisy, 16384, 42);
    REQUIRE(ordinary.execution.executor != SamplingExecutor::Shared);
    REQUIRE_FALSE(ordinary.execution.sharing);

    auto unitary = trace(parse("H 0 1 2 3 4 5 6 7\nT 0 1 2 3 4 5 6 7\nM 0 1 2 3 4 5 6 7"));
    const Program terminal{ExecutablePlan(plan_sampling(unitary))};
    REQUIRE(terminal.original().peak_active_width() == 8);
    REQUIRE(terminal.original().has_reusable_unitary_prefix());
    REQUIRE(clifft::sample(terminal, 32, 42).execution.executor == SamplingExecutor::Shared);
    REQUIRE(clifft::sample(terminal, 15, 42).execution.executor != SamplingExecutor::Shared);
    const auto branching_hir =
        trace(parse("H 0 1 2 3 4 5 6 7\nT 0 1 2 3 4 5 6 7\nM 0\nH 0\nT 0\nM 0 1 2 3 4 5 6 7"));
    const Program branching{ExecutablePlan(plan_sampling(branching_hir))};
    REQUIRE_FALSE(branching.original().has_reusable_unitary_prefix());
    REQUIRE(clifft::sample(branching, 32, 42).execution.executor != SamplingExecutor::Shared);
    const auto dormant_independent_hir = trace(parse(
        "H 0 1 2 3 4 5 6 7 8\nT 0 1 2 3 4 5 6 7\nH 0 1\nT 0 1\nM 8\nT 0\nH 0\nM 0 1 2 3 4 5 6 7"));
    const Program dormant_independent{ExecutablePlan(plan_sampling(dormant_independent_hir))};
    REQUIRE(dormant_independent.original().has_reusable_unitary_prefix());
    REQUIRE(clifft::sample(dormant_independent, 256, 42).execution.executor ==
            SamplingExecutor::Shared);
    const auto dormant_feedback_hir = trace(
        parse("H 0 1 2 3 4 5 6 7 8\nT 0 1 2 3 4 5 6 7\nH 0 1\nT 0 1\nM 8\nCX rec[-1] 0\nT 0\nH 0\n"
              "M 0 1 2 3 4 5 6 7"));
    const Program dormant_feedback{ExecutablePlan(plan_sampling(dormant_feedback_hir))};
    REQUIRE_FALSE(dormant_feedback.original().has_reusable_unitary_prefix());
    REQUIRE(clifft::sample(dormant_feedback, 32, 42).execution.executor !=
            SamplingExecutor::Shared);
    const auto heavy_suffix_hir =
        trace(parse("H 0 1 2 3 4 5 6 7\nT 0 1 2 3 4 5 6 7\nM 0 1\n"
                    "H 2 3 4 5 6 7\nT 2 3 4 5 6 7\nH 2 3 4 5 6 7\nT 2 3 4 5 6 7\nM 2 3 4 5 6 7"));
    const Program heavy_suffix{ExecutablePlan(plan_sampling(heavy_suffix_hir))};
    REQUIRE_FALSE(heavy_suffix.original().has_reusable_unitary_prefix());
    REQUIRE(clifft::sample(heavy_suffix, 32, 42).execution.executor != SamplingExecutor::Shared);
    // A few measurement branches keep the shots of a longer call on few
    // trajectories, so these plans share without a reusable prefix.
    for (const Program* few_branches : {&branching, &dormant_feedback, &heavy_suffix}) {
        REQUIRE(clifft::sample(*few_branches, 256, 42).execution.executor ==
                SamplingExecutor::Shared);
    }

    const auto qv_source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/qv10.stim");
    auto qv_hir = trace(qv_source);
    default_hir_pass_manager().run(qv_hir);
    const auto qv = Program::compile(qv_source, qv_hir, true);
    REQUIRE(qv.sampling_plan(true).has_reusable_unitary_prefix());
    REQUIRE(clifft::sample(qv, 256, 42).execution.executor == SamplingExecutor::Shared);
}

TEST_CASE("Native shared calibration replays with the selected settings") {
    const Program program = row_program(true);
    const SamplingOptions options{.threads = 2, .tuning_budget_seconds = 0.01, .sharing = true};
    const auto tuned = clifft::sample_survivors(program, 4097, 42, true, options);
    REQUIRE(tuned.batch_tuning.has_value());
    REQUIRE(tuned.execution.executor == SamplingExecutor::Shared);
    REQUIRE(tuned.execution.sharing);
    REQUIRE(tuned.execution.batch_size == tuned.batch_tuning->batch_size);
    REQUIRE(tuned.execution.shot_workers == tuned.batch_tuning->shot_workers);
    for (const auto& trial : tuned.batch_tuning->trials) {
        REQUIRE(trial.batch_size > 1);
    }
    const auto replay =
        clifft::sample_survivors(program, 4097, 42, true,
                                 {.threads = tuned.execution.shot_workers,
                                  .batch_size = tuned.execution.batch_size,
                                  .sharing = tuned.execution.sharing,
                                  .group_by_noise = tuned.execution.group_by_noise,
                                  .noise_preparation = tuned.execution.noise_preparation});
    require_same(tuned, replay);
    REQUIRE_THROWS_AS(clifft::sample_survivors(
                          program, 65, 1, false,
                          {.batch_size = 65,
                           .sharing = true,
                           .max_worker_state_bytes =
                               2 * State::storage_bytes(program.original().peak_active_width())}),
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

TEST_CASE("Native shared calibration skips a single candidate") {
    const auto result = clifft::sample_survivors(row_program(true), 8, 42, true,
                                                 {.tuning_budget_seconds = 0.01, .sharing = true});
    REQUIRE(result.batch_tuning.has_value());
    REQUIRE(result.batch_tuning->stop_reason == "single_candidate");
    REQUIRE(result.batch_tuning->trials.empty());
    REQUIRE(result.batch_tuning->trial_shots == 0);
    REQUIRE(result.execution.batch_size == 8);
}

TEST_CASE(
    "Native automatic sharing depends on the plan rather than worker count or detector "
    "thresholds") {
    const uint32_t width = GENERATE(7u, 8u);
    const uint32_t selected = GENERATE(0u, 1u, 8u);
    std::string source;
    for (const auto* gate : {"H", "T", "M"}) {
        source += gate;
        for (uint32_t q = 0; q < width; ++q) {
            source += " " + std::to_string(q);
        }
        source += "\n";
    }
    for (uint32_t q = width; q < width + 8; ++q) {
        source += "X_ERROR(0.01) " + std::to_string(q) + "\nM " + std::to_string(q) +
                  "\nDETECTOR rec[-1]\n";
    }
    auto hir = trace(parse(source));
    hir.postselection.mask.assign(8, 0);
    std::fill_n(hir.postselection.mask.begin(), selected, 1);
    const ExecutablePlan plan(plan_sampling(hir));
    REQUIRE(plan.peak_active_width() == width);
    REQUIRE(plan.num_postselected_detectors() == selected);
    // The noise acts on classical qubits only, so every shot follows one
    // trajectory up to the terminal measurements with or without selection.
    REQUIRE(plan.trajectory_divergence().expected_sign_faults == 0.0);
    REQUIRE(should_share_sampling(plan, 64, false, {1, 1}, kDefaultSharedWorkerStateBudget));
    REQUIRE(should_share_sampling(plan, 64, false, {8, 1}, kDefaultSharedWorkerStateBudget));
    REQUIRE_FALSE(should_share_sampling(plan, 63, false, {1, 1}, kDefaultSharedWorkerStateBudget));
}

TEST_CASE("Native automatic sharing without postselection bounds trajectory divergence") {
    const auto compile = [](const std::string& source) {
        return ExecutablePlan(plan_sampling(trace(parse(source))));
    };
    const auto shares = [](const ExecutablePlan& plan, uint32_t shots = 65536,
                           uint64_t state_bytes = kDefaultSharedWorkerStateBudget) {
        return should_share_sampling(plan, shots, false, {1, 1}, state_bytes);
    };
    // Four qubits with `layers` rounds of an X error followed by T, then
    // `record_layers` rounds of X errors before the measurements. An X error
    // before T flips the rotation sign; one after the last T only flips a record.
    const auto noisy_layers = [](double probability, uint32_t layers, uint32_t record_layers) {
        const std::string error = "X_ERROR(" + std::to_string(probability) + ") 0 1 2 3\n";
        std::string source = "H 0 1 2 3\n";
        for (uint32_t layer = 0; layer < layers; ++layer) {
            source += error + "T 0 1 2 3\nH 0 1 2 3\n";
        }
        for (uint32_t layer = 0; layer < record_layers; ++layer) {
            source += error;
        }
        return source + "M 0 1 2 3\n";
    };
    // Three data qubits. Each ancilla measurement controls an X before a T
    // layer: `prepare` makes its outcome uniformly random, and `measure` can
    // add readout noise.
    const auto feedback = [](uint32_t ancillas, const std::string& prepare,
                             const std::string& measure) {
        std::string source = "H 0 1 2\n";
        for (uint32_t a = 0; a < ancillas; ++a) {
            const auto ancilla = std::to_string(3 + a);
            source += prepare.empty() ? "" : prepare + " " + ancilla + "\n";
            source += measure + " " + ancilla + "\nCX rec[-1] " + std::to_string(a % 3) +
                      "\nT 0 1 2\nH 0 1 2\n";
        }
        return source + "M 0 1 2\n";
    };
    // Every round measures a qubit of the active state and rotates it again.
    const auto remeasured = [](uint32_t rounds) {
        std::string source = "H 0 1 2\nT 0 1 2\nH 0 1 2\n";
        for (uint32_t round = 0; round < rounds; ++round) {
            source += "M 0\nH 0\nT 0\nH 0\n";
        }
        return source + "M 0 1 2\n";
    };

    SECTION("sign faults") {
        const auto rare = compile(noisy_layers(0.01, 5, 0));
        REQUIRE(rare.peak_active_width() == 4);
        REQUIRE(std::abs(rare.trajectory_divergence().expected_sign_faults - 0.2) < 1e-9);
        REQUIRE(shares(rare));
        REQUIRE(shares(rare, 64));
        REQUIRE_FALSE(shares(rare, 63));

        const auto frequent = compile(noisy_layers(0.1, 5, 0));
        REQUIRE(std::abs(frequent.trajectory_divergence().expected_sign_faults - 2.0) < 1e-9);
        REQUIRE_FALSE(shares(frequent));

        // Frequent faults that only flip records leave the estimate unchanged.
        const auto record_noise = compile(noisy_layers(0.01, 5, 40));
        REQUIRE(record_noise.expected_noise_faults() > 1.0);
        REQUIRE(std::abs(record_noise.trajectory_divergence().expected_sign_faults - 0.2) < 1e-9);
        REQUIRE(shares(record_noise));
    }

    SECTION("readout flips") {
        // A noisy readout that controls a later gate is a sign fault, like an
        // X error before an ideal measurement of the same ancilla.
        const auto readout = compile(feedback(30, "", "M(0.05)"));
        const auto bit_flip = compile(feedback(30, "X_ERROR(0.05)", "M"));
        REQUIRE(readout.peak_active_width() == 3);
        REQUIRE(readout.num_noise_sites() == 0);
        REQUIRE(std::abs(readout.trajectory_divergence().expected_sign_faults - 1.5) < 1e-9);
        REQUIRE(std::abs(bit_flip.trajectory_divergence().expected_sign_faults - 1.5) < 1e-9);
        REQUIRE_FALSE(shares(readout));
        REQUIRE_FALSE(shares(bit_flip));
        REQUIRE(shares(compile(feedback(10, "", "M(0.05)"))));

        // Readout flips that reach only records do not count.
        const auto output_only =
            compile("H 0 1 2\nT 0 1 2\nH 0 1 2\nM(0.4) 3 4 5 6 7 8\nM(0.4) 0 1 2\n");
        REQUIRE(output_only.has_readout_noise());
        REQUIRE(output_only.trajectory_divergence().expected_sign_faults == 0.0);
        REQUIRE(shares(output_only));
    }

    SECTION("width") {
        // Two qubits leave too little coefficient work to share.
        const auto narrow = compile("H 0 1\nX_ERROR(0.01) 0 1\nT 0 1\nH 0 1\nM 0 1\n");
        REQUIRE(narrow.peak_active_width() == 2);
        REQUIRE_FALSE(shares(narrow));
    }

    SECTION("dormant measurement feedback needs lanes for every branch") {
        const auto plan = compile(feedback(8, "H", "M"));
        REQUIRE(plan.peak_active_width() == 3);
        REQUIRE(plan.trajectory_divergence().sign_dormant_measurements == 8);
        // Eight random bits give 256 trajectories, which share at 16 lanes each.
        REQUIRE(shares(plan, 4096));
        REQUIRE_FALSE(shares(plan, 4095));
        REQUIRE_FALSE(shares(plan, 256));
        // A state budget below the call size limits the lanes of a batch.
        REQUIRE(shares(plan, 65536, 4096 * State::storage_bytes(3)));
        REQUIRE_FALSE(shares(plan, 65536, 4095 * State::storage_bytes(3)));

        const auto few = compile(feedback(2, "H", "M"));
        REQUIRE(few.trajectory_divergence().sign_dormant_measurements == 2);
        REQUIRE(shares(few, 64));
    }

    SECTION("mid-circuit active measurements") {
        const auto few = compile(remeasured(12));
        REQUIRE(few.peak_active_width() == 3);
        REQUIRE(few.trajectory_divergence().mid_circuit_active_measurements == 12);
        REQUIRE(shares(few));
        const auto many = compile(remeasured(13));
        REQUIRE(many.trajectory_divergence().mid_circuit_active_measurements == 13);
        REQUIRE_FALSE(shares(many));
    }

    SECTION("terminal active measurements") {
        // Measurements after the last rotation split no coefficient work, so a
        // wide noisy plan with terminal readout still shares.
        std::string qubits;
        for (uint32_t q = 0; q < 14; ++q) {
            qubits += " " + std::to_string(q);
        }
        const auto wide = compile("H" + qubits + "\nX_ERROR(0.001)" + qubits + "\nT" + qubits +
                                  "\nH" + qubits + "\nM" + qubits + "\n");
        REQUIRE(wide.peak_active_width() == 14);
        REQUIRE(wide.num_noise_sites() != 0);
        REQUIRE(wide.trajectory_divergence().mid_circuit_active_measurements == 0);
        REQUIRE(shares(wide));
    }

    SECTION("dormant and active measurements share one limit") {
        const auto combined = [&](uint32_t ancillas, uint32_t rounds) {
            auto source = feedback(ancillas, "H", "M");
            source.resize(source.size() - std::string("M 0 1 2\n").size());
            for (uint32_t round = 0; round < rounds; ++round) {
                source += "M 0\nH 0\nT 0\nH 0\n";
            }
            return compile(source + "M 0 1 2\n");
        };
        const auto within = combined(4, 8);
        REQUIRE(within.trajectory_divergence().sign_dormant_measurements == 4);
        REQUIRE(within.trajectory_divergence().mid_circuit_active_measurements == 8);
        REQUIRE(shares(within));
        REQUIRE_FALSE(shares(combined(4, 9)));
    }
}

TEST_CASE("Survivor row collection transfers a single batch and preserves batch order") {
    const auto batches = GENERATE(1u, 3u);
    batch_detail::SurvivorRowCollector collector(batches * 64, 64, true);
    for (uint32_t i = 0; i < batches; ++i) {
        auto& rows = collector.batch(i * 64);
        rows.measurements = {static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1)};
        rows.detectors = {static_cast<uint8_t>(i + 2)};
        rows.observables = {static_cast<uint8_t>(i + 3)};
        rows.exp_vals = {static_cast<double>(i) / 2};
    }
    const auto* original = collector.batch(0).measurements.data();
    SamplingSurvivorResult result;
    collector.finish(result);
    REQUIRE(result.measurements.size() == 2 * batches);
    REQUIRE(result.detectors.size() == batches);
    REQUIRE(result.observables.size() == batches);
    REQUIRE(result.exp_vals.size() == batches);
    for (uint32_t i = 0; i < batches; ++i) {
        REQUIRE(result.measurements[2 * i] == i);
        REQUIRE(result.measurements[2 * i + 1] == i + 1);
        REQUIRE(result.detectors[i] == i + 2);
        REQUIRE(result.observables[i] == i + 3);
        REQUIRE(result.exp_vals[i] == static_cast<double>(i) / 2);
    }
    if (batches == 1) {
        REQUIRE(result.measurements.data() == original);
    }
}

TEST_CASE("Native shared counts calibrate a baseline with more than eight workers") {
    const Program program{ExecutablePlan(
        plan_sampling(trace(parse("H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"))))};
    constexpr uint32_t shots = 131073;
    const auto result = clifft::sample_survivors(
        program, shots, 42, false, {.threads = 9, .tuning_budget_seconds = 0.1, .sharing = true});
    REQUIRE(result.total_shots == shots);
    REQUIRE(result.passed_shots == shots);
    REQUIRE(result.batch_tuning.has_value());
    REQUIRE_FALSE(result.batch_tuning->trials.empty());
    const auto& baseline = result.batch_tuning->trials.front();
    REQUIRE(baseline.batch_size == 8192);
    REQUIRE(baseline.shot_workers == 9);
    REQUIRE(result.batch_tuning->stop_reason != "single_candidate");
}

TEST_CASE("Native forced sharing on width zero plans uses packed batches under shared policy") {
    auto hir =
        trace(parse("H 0\nCX 0 1\nDEPOLARIZE2(0.07) 0 1\nM(0.13) 0 1\n"
                    "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"));
    hir.postselection.mask = {1};
    hir.postselection.expected_detectors = {1};
    const Program program{ExecutablePlan(plan_sampling(hir))};
    REQUIRE(program.sampling_plan(false).peak_active_width() == 0);

    const auto short_shared =
        clifft::sample_survivors(program, 17, 19, false, {.sharing = true, .group_by_noise = true});
    REQUIRE(short_shared.execution.executor == SamplingExecutor::Packed);
    REQUIRE(short_shared.execution.sharing);
    REQUIRE(short_shared.execution.batch_size == 17);
    REQUIRE(short_shared.execution.noise_preparation == BatchNoiseMode::SiteSparse);
    REQUIRE_FALSE(short_shared.execution.group_by_noise);
    REQUIRE_FALSE(short_shared.execution.max_worker_state_bytes.has_value());

    const auto wide_shared = clifft::sample_survivors(
        program, 20000, 19, {RecordOutputs::Syndromes, true}, {.sharing = true});
    REQUIRE(wide_shared.execution.executor == SamplingExecutor::Packed);
    REQUIRE(wide_shared.execution.sharing);
    REQUIRE(wide_shared.execution.batch_size == 8192);
    REQUIRE(wide_shared.execution.outputs == SamplingOutputs(RecordOutputs::Syndromes, true));
    require_same(wide_shared, clifft::sample_survivors(
                                  program, 20000, 19, wide_shared.execution.outputs,
                                  {.batch_size = wide_shared.execution.batch_size,
                                   .sharing = wide_shared.execution.sharing,
                                   .group_by_noise = wide_shared.execution.group_by_noise,
                                   .noise_preparation = wide_shared.execution.noise_preparation}));

    const auto tuned = clifft::sample_survivors(program, 4097, 19, false,
                                                {.tuning_budget_seconds = 0.01, .sharing = true});
    REQUIRE(tuned.execution.executor == SamplingExecutor::Packed);
    REQUIRE(tuned.execution.sharing);
    REQUIRE(tuned.batch_tuning.has_value());
    REQUIRE(tuned.execution.batch_size == tuned.batch_tuning->batch_size);
    require_same(
        tuned, clifft::sample_survivors(program, 4097, 19, false,
                                        {.batch_size = tuned.execution.batch_size,
                                         .sharing = tuned.execution.sharing,
                                         .group_by_noise = tuned.execution.group_by_noise,
                                         .noise_preparation = tuned.execution.noise_preparation}));

    const Program active = row_program(true);
    for (const auto* candidate : {&program, &active}) {
        for (const uint32_t count : {0u, 1u}) {
            const auto tiny =
                clifft::sample_survivors(*candidate, count, 11, false, {.sharing = true});
            REQUIRE(tiny.execution.sharing);
            REQUIRE(tiny.execution.batch_size == 2);
            require_same(tiny, clifft::sample_survivors(
                                   *candidate, count, 11, false,
                                   {.batch_size = tiny.execution.batch_size,
                                    .sharing = tiny.execution.sharing,
                                    .group_by_noise = tiny.execution.group_by_noise,
                                    .noise_preparation = tiny.execution.noise_preparation}));
        }
        REQUIRE_THROWS_AS(
            clifft::sample_survivors(*candidate, 65, 1, false, {.group_by_noise = true}),
            std::invalid_argument);
        REQUIRE_THROWS_AS(clifft::sample_survivors(*candidate, 65, 1, false,
                                                   {.sharing = false, .group_by_noise = true}),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(
            clifft::sample_survivors(*candidate, 65, 1, false, {.batch_size = 1, .sharing = true}),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            clifft::sample_survivors(*candidate, 65, 1, false,
                                     {.batch_size = 1, .noise_preparation = BatchNoiseMode::Dense}),
            std::invalid_argument);
        const auto scalar = clifft::sample_survivors(
            *candidate, 65, 1, false, {.batch_size = 1, .sharing = false, .group_by_noise = false});
        REQUIRE_FALSE(scalar.execution.sharing);
        REQUIRE_FALSE(scalar.execution.group_by_noise);
    }

    const auto dense = clifft::sample_survivors(
        active, 1301, 7, true,
        {.batch_size = 65, .sharing = true, .noise_preparation = BatchNoiseMode::Dense});
    auto sparse = clifft::sample_survivors(
        active, 1301, 7, true,
        {.batch_size = 65, .sharing = true, .noise_preparation = BatchNoiseMode::Sparse});
    REQUIRE(sparse.execution.noise_preparation == BatchNoiseMode::Sparse);
    sparse.execution.noise_preparation = BatchNoiseMode::Dense;
    require_same(dense, sparse);
}

TEST_CASE("Native width zero sampling selects noise preparation from expected faults per batch") {
    const bool postselect = GENERATE(false, true);
    const Program program = clifford_noise_program(0.004, postselect);
    const auto& plan = program.sampling_plan(true);
    REQUIRE(plan.peak_active_width() == 0);
    REQUIRE(plan.has_postselection() == postselect);
    REQUIRE(plan.num_noise_sites() == 3);
    REQUIRE(std::abs(plan.expected_noise_faults() - 0.018) < 1e-12);

    // A 64-lane batch expects 1.152 faults across the three sites, so drawing
    // once per fault is cheaper. A 2048-lane batch expects 36.9.
    const auto brief = clifft::sample_survivors(program, 64, 5, true);
    REQUIRE(brief.execution.executor == SamplingExecutor::Packed);
    REQUIRE(brief.execution.batch_size == 64);
    REQUIRE(brief.execution.noise_preparation == BatchNoiseMode::Sparse);
    const auto bulk = clifft::sample_survivors(program, 8193, 5, true);
    REQUIRE(bulk.execution.executor == SamplingExecutor::Packed);
    REQUIRE(bulk.execution.batch_size == 2048);
    REQUIRE(bulk.execution.noise_preparation == BatchNoiseMode::SiteSparse);
    REQUIRE_FALSE(bulk.execution.sharing);

    // An explicit capacity keeps the seeded rows of Dense preparation under
    // either choice, including the single-shot tail batch.
    for (const uint32_t batch : {64u, 2048u}) {
        CAPTURE(batch);
        const SamplingOptions options{.threads = 2, .batch_size = batch};
        auto selected = clifft::sample_survivors(program, 8193, 5, true, options);
        REQUIRE(selected.execution.noise_preparation ==
                (batch == 64 ? BatchNoiseMode::Sparse : BatchNoiseMode::Dense));
        auto dense = options;
        dense.noise_preparation = BatchNoiseMode::Dense;
        selected.execution.noise_preparation = BatchNoiseMode::Dense;
        require_same(selected, clifft::sample_survivors(program, 8193, 5, true, dense));
    }

    const auto rows = clifft::sample_survivors(program, 8193, 5, true,
                                               {.noise_preparation = BatchNoiseMode::SiteRows});
    REQUIRE(rows.execution.batch_size == 2048);
    REQUIRE(rows.execution.noise_preparation == BatchNoiseMode::SiteRows);

    if (!postselect) {
        const auto fixed = clifft::sample(program, 8193, 5);
        REQUIRE(fixed.execution.batch_size == 2048);
        REQUIRE(fixed.execution.noise_preparation == BatchNoiseMode::SiteSparse);
        REQUIRE(clifft::sample(program, 64, 5).execution.noise_preparation ==
                BatchNoiseMode::Sparse);
        REQUIRE(clifft::sample(program, 8193, 5, {}, {.batch_size = 2048})
                    .execution.noise_preparation == BatchNoiseMode::Dense);
    }

    // Plans with coefficient work also select noise preparation from expected faults.
    const auto active =
        clifft::sample_survivors(row_program(true), 8193, 5, true, {.sharing = false});
    REQUIRE(active.execution.executor == SamplingExecutor::Packed);
    REQUIRE(active.execution.batch_size == 256);
    REQUIRE(active.execution.noise_preparation == BatchNoiseMode::SiteSparse);
    const auto active_explicit = clifft::sample_survivors(row_program(true), 8193, 5, true,
                                                          {.batch_size = 256, .sharing = false});
    REQUIRE(active_explicit.execution.noise_preparation == BatchNoiseMode::Dense);
}

TEST_CASE("Native width zero selection replays across preparation history and concurrency") {
    const SamplingOutputs outputs(GENERATE(false, true));
    const SamplingOptions options{.threads = 4};
    const auto program = clifford_noise_program(0.1, true);
    const auto cold = clifft::sample_survivors(program, 8193, 19, outputs, options);
    REQUIRE(cold.execution.noise_preparation == BatchNoiseMode::SiteSparse);
    (void)clifft::sample_survivors(program, 64, 8, true);
    (void)clifft::sample_survivors(program, 1024, 8, false, {.batch_size = 65});
    (void)clifft::sample_survivors(program, 1024, 8, outputs,
                                   {.noise_preparation = BatchNoiseMode::SiteRows});
    require_same(cold, clifft::sample_survivors(program, 8193, 19, outputs, options));
    const auto fresh = clifford_noise_program(0.1, true);
    std::array<std::future<SamplingSurvivorResult>, 4> calls;
    for (auto& call : calls) {
        call = std::async(std::launch::async, [&] {
            return clifft::sample_survivors(fresh, 8193, 19, outputs, options);
        });
    }
    for (auto& call : calls) {
        require_same(cold, call.get());
    }
    auto serial = clifft::sample_survivors(program, 8193, 19, outputs, {.threads = 1});
    serial.execution = cold.execution;
    require_same(cold, serial);
}

TEST_CASE("Native width zero joint outputs match the dense circuit oracle") {
    const bool postselect = GENERATE(false, true);
    // Four lanes expect 1.8 faults across three sites; 2048 lanes expect 921.6.
    const auto [options, noise] =
        GENERATE(std::pair{SamplingOptions{}, BatchNoiseMode::SiteSparse},
                 std::pair{SamplingOptions{.batch_size = 4}, BatchNoiseMode::Sparse});
    const auto circuit = clifford_noise_circuit(0.1);
    const std::array<uint8_t, 2> mask{static_cast<uint8_t>(postselect), 0};
    const auto oracle =
        clifft::test::compute_dense_circuit_distribution(circuit, {.postselection_mask = mask});
    const Program program = clifford_noise_program(0.1, postselect);
    constexpr uint32_t shots = 32768;
    const auto result = clifft::sample_survivors(program, shots, 81, true, options);
    REQUIRE(result.execution.executor == SamplingExecutor::Packed);
    REQUIRE(result.execution.noise_preparation == noise);
    // Three records, two detectors, one observable, then the rejected shots.
    std::array<double, 65> expected{};
    std::array<uint32_t, 65> actual{};
    expected[64] = oracle.discarded_probability;
    actual[64] = shots - result.passed_shots;
    for (const auto& entry : oracle.kept_outcomes) {
        const auto& outcome = entry.outcome;
        expected[outcome.visible_records | (outcome.detectors << 3) | (outcome.observables << 5)] +=
            entry.probability;
    }
    for (uint32_t row = 0; row < result.passed_shots; ++row) {
        const auto* m = result.measurements.data() + 3 * row;
        const auto* d = result.detectors.data() + 2 * row;
        ++actual[m[0] | (m[1] << 1) | (m[2] << 2) | (d[0] << 3) | (d[1] << 4) |
                 (result.observables[row] << 5)];
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        if (expected[i] == 0) {
            REQUIRE(actual[i] == 0);
        } else {
            REQUIRE(std::abs(actual[i] - shots * expected[i]) <
                    8 * std::sqrt(shots * expected[i] * (1 - expected[i])) + 8);
        }
    }
}

TEST_CASE("Native site noise preparation skips zero-probability sites") {
    // A zero-probability channel keeps its noise site but no outcomes.
    const bool leading = GENERATE(false, true);
    const Program program{ExecutablePlan(
        plan_sampling(trace(parse(leading ? "X_ERROR(0) 0\nX_ERROR(0.25) 1\nM 0 1\n"
                                          : "X_ERROR(0.25) 1\nX_ERROR(0) 0\nM 0 1\n"))))};
    REQUIRE(program.original().num_noise_sites() == 2);
    const auto mode = GENERATE(BatchNoiseMode::SiteSparse, BatchNoiseMode::SiteRows);
    constexpr uint32_t shots = 4096;
    const auto automatic = clifft::sample(program, shots, 3);
    REQUIRE(automatic.execution.noise_preparation == BatchNoiseMode::SiteSparse);
    const auto result = clifft::sample(program, shots, 3, {}, {.noise_preparation = mode});
    REQUIRE(result.measurements == automatic.measurements);
    uint32_t flips = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        REQUIRE(result.measurements[2 * shot] == 0);
        flips += result.measurements[2 * shot + 1];
    }
    // Eight standard deviations around the 1,024 expected flips.
    REQUIRE(std::abs(static_cast<int>(flips) - 1024) < 222);
}

TEST_CASE("Shared sampling reuses and bounds idle workers across repeated and concurrent calls") {
    const Program program = row_program(true);
    const SamplingOutputs outputs(true);
    const auto& prep = program.sampling_preparation(
        outputs, {.early_postselection = true, .survivor_counts = true});
    REQUIRE(prep.retained_shared_worker_count() == 0);
    REQUIRE(prep.retained_shared_worker_bytes() == 0);

    const SamplingOptions one_thread{.threads = 1, .batch_size = 512, .sharing = true};
    const auto cold = clifft::sample_survivors(program, 2048, 42, outputs, one_thread);
    REQUIRE(cold.execution.executor == SamplingExecutor::Shared);
    REQUIRE(prep.retained_shared_worker_count() == 1);
    const uint64_t retained_one = prep.retained_shared_worker_bytes();
    REQUIRE(retained_one > 0);
    REQUIRE(retained_one <= clifft::kMaxRetainedSharedWorkerBytes);

    // Repeating the same call reuses the idle worker without growing the pool.
    const auto warm = clifft::sample_survivors(program, 2048, 42, outputs, one_thread);
    require_same(cold, warm);
    REQUIRE(prep.retained_shared_worker_count() == 1);
    REQUIRE(prep.retained_shared_worker_bytes() == retained_one);

    // Borrowing with a mismatched configuration releases the old pool before dispatch.
    const clifft::SharedWorkerKey dense_key{
        .output_mode = BatchOutputMode::Rows,
        .noise_mode = BatchNoiseMode::Dense,
        .lane_capacity = 512,
        .max_worker_state_bytes = kDefaultSharedWorkerStateBudget,
        .group_by_noise = false,
    };
    REQUIRE(prep.borrow_shared_workers(dense_key, 1).empty());
    REQUIRE(prep.retained_shared_worker_count() == 0);
    REQUIRE(prep.retained_shared_worker_bytes() == 0);

    // Concurrent multi-threaded calls check out exclusive workers and stay within the byte cap.
    const SamplingOptions two_threads{.threads = 2, .batch_size = 512, .sharing = true};
    const auto expected_two = clifft::sample_survivors(program, 2048, 42, outputs, two_threads);
    std::array<std::future<SamplingSurvivorResult>, 4> calls;
    for (auto& call : calls) {
        call = std::async(std::launch::async, [&] {
            return clifft::sample_survivors(program, 2048, 42, outputs, two_threads);
        });
    }
    for (auto& call : calls) {
        require_same(expected_two, call.get());
    }
    REQUIRE(prep.retained_shared_worker_count() >= 1);
    REQUIRE(prep.retained_shared_worker_bytes() <= clifft::kMaxRetainedSharedWorkerBytes);

    // Switching execution configuration replaces the retained pool cleanly.
    const SamplingOptions one_thread_alt{
        .threads = 1, .batch_size = 256, .sharing = true, .group_by_noise = true};
    const auto alt = clifft::sample_survivors(program, 1024, 42, outputs, one_thread_alt);
    REQUIRE(alt.execution.group_by_noise);
    REQUIRE(prep.retained_shared_worker_count() == 1);

    // Isolate max_worker_state_bytes in SharedWorkerKey: warm an identical
    // ungrouped 65-lane worker at the default budget, then lower only the budget.
    const SamplingOptions wide_budget{.threads = 1, .batch_size = 65, .sharing = true};
    const auto grown = clifft::sample_survivors(program, 65, 1, outputs, wide_budget);
    REQUIRE(grown.execution.batch_size == 65);
    REQUIRE_FALSE(grown.execution.group_by_noise);
    REQUIRE(prep.retained_shared_worker_count() == 1);
    const uint32_t width = program.sampling_plan(true).peak_active_width();
    REQUIRE_THROWS_AS(
        clifft::sample_survivors(program, 65, 1, outputs,
                                 {.threads = 1,
                                  .batch_size = 65,
                                  .sharing = true,
                                  .max_worker_state_bytes = 2 * State::storage_bytes(width)}),
        std::runtime_error);
    require_same(cold, clifft::sample_survivors(program, 2048, 42, outputs, one_thread));
    REQUIRE(prep.retained_shared_worker_count() == 1);
}

TEST_CASE("Process-wide shared worker cache enforces its budget and LRU eviction") {
    const SamplingOutputs outputs(true);
    const SamplingOptions options{.threads = 1, .batch_size = 512, .sharing = true};
    std::optional<Program> first = row_program(true);
    const auto& first_prep = first->sampling_preparation(
        outputs, {.early_postselection = true, .survivor_counts = true});
    const auto ref = clifft::sample_survivors(*first, 2048, 7, outputs, options);
    const uint64_t worker_bytes = first_prep.retained_shared_worker_bytes();
    REQUIRE(worker_bytes > 0);

    // A budget below one worker rejects retention completely.
    {
        const clifft::detail::ScopedSharedWorkerCacheBudget tiny(worker_bytes - 1);
        REQUIRE(clifft::total_retained_shared_worker_bytes() == 0);
        require_same(ref, clifft::sample_survivors(*first, 2048, 7, outputs, options));
        REQUIRE(first_prep.retained_shared_worker_count() == 0);
        REQUIRE(clifft::total_retained_shared_worker_bytes() == 0);
    }

    // A budget holding two workers evicts the least-recently-used Program when a third returns.
    {
        const uint64_t two_budget = 2 * worker_bytes + worker_bytes / 2;
        const clifft::detail::ScopedSharedWorkerCacheBudget two_workers(two_budget);
        std::optional<Program> second = row_program(true);
        std::optional<Program> third = row_program(true);
        const auto& second_prep = second->sampling_preparation(
            outputs, {.early_postselection = true, .survivor_counts = true});
        const auto& third_prep = third->sampling_preparation(
            outputs, {.early_postselection = true, .survivor_counts = true});
        require_same(ref, clifft::sample_survivors(*first, 2048, 7, outputs, options));
        require_same(ref, clifft::sample_survivors(*second, 2048, 7, outputs, options));
        // Touch first so second becomes the least-recently-used entry.
        require_same(ref, clifft::sample_survivors(*first, 2048, 7, outputs, options));
        REQUIRE(first_prep.retained_shared_worker_count() == 1);
        REQUIRE(second_prep.retained_shared_worker_count() == 1);

        require_same(ref, clifft::sample_survivors(*third, 2048, 7, outputs, options));
        REQUIRE(first_prep.retained_shared_worker_count() == 1);
        REQUIRE(second_prep.retained_shared_worker_count() == 0);
        REQUIRE(third_prep.retained_shared_worker_count() == 1);
        REQUIRE(clifft::total_retained_shared_worker_bytes() <= two_budget);

        // Destroying cached Programs releases their workers immediately.
        third.reset();
        REQUIRE(clifft::total_retained_shared_worker_bytes() == worker_bytes);
    }

    // Returning an extra worker that cannot fit alongside the caller's own
    // retained worker must not evict another Program's cached worker.
    {
        const SamplingOptions small_options{.threads = 1, .batch_size = 64, .sharing = true};
        const SamplingOptions two_threads{.threads = 2, .batch_size = 512, .sharing = true};
        std::optional<Program> small = row_program(true);
        const auto& small_prep = small->sampling_preparation(
            outputs, {.early_postselection = true, .survivor_counts = true});
        (void)clifft::sample_survivors(*small, 2048, 7, outputs, small_options);
        const uint64_t small_bytes = small_prep.retained_shared_worker_bytes();
        REQUIRE(small_bytes > 0);
        REQUIRE(small_bytes < worker_bytes);

        const uint64_t pair_budget = worker_bytes + small_bytes;
        const clifft::detail::ScopedSharedWorkerCacheBudget pair(pair_budget);
        REQUIRE(first_prep.retained_shared_worker_count() == 1);
        REQUIRE(small_prep.retained_shared_worker_count() == 1);
        (void)clifft::sample_survivors(*first, 2048, 7, outputs, two_threads);
        REQUIRE(first_prep.retained_shared_worker_count() == 1);
        REQUIRE(small_prep.retained_shared_worker_count() == 1);
        REQUIRE(clifft::total_retained_shared_worker_bytes() == pair_budget);

        small.reset();
        REQUIRE(clifft::total_retained_shared_worker_bytes() == worker_bytes);
    }
    first.reset();
    REQUIRE(clifft::total_retained_shared_worker_bytes() == 0);
}
