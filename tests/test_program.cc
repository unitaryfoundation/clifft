#include "clifft/api/program.h"
#include "clifft/api/reference_syndrome.h"
#include "clifft/api/sampling_preparation.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace clifft::test {

namespace {

using sampling::BatchNoiseMode;
using sampling::BatchOutputMode;

constexpr std::array kNoiseModes = {BatchNoiseMode::Dense, BatchNoiseMode::Sparse,
                                    BatchNoiseMode::SiteSparse, BatchNoiseMode::SiteRows};

// The first detector is postselected when its mask entry is set. The expectation
// probe and the record outputs exist only in the compiled plan.
Program compile_preparation_program(std::vector<uint8_t> postselection_mask) {
    const auto source = parse(R"(
        H 0
        T 0
        EXP_VAL X0
        DEPOLARIZE1(0.05) 0
        H 0
        M 0
        DETECTOR rec[-1]
        H 1
        T 1
        H 1
        M(0.1) 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
    )");
    auto hir = trace(source);
    hir.postselection.mask = std::move(postselection_mask);
    return Program::compile(source, hir, false);
}

}  // namespace

TEST_CASE("Program preserves the original plan across bounded reconstruction failures",
          "[reconstruction]") {
    const auto source = parse("H 0 1\nT 0 1\nH 0 1\nM 0 1\nDETECTOR rec[-1]\n");
    const auto hir = trace(source);
    const auto original = Program::compile(source, hir, false);
    REQUIRE(original.original().peak_active_width() == 2);
    for (bool keep_records : {false, true}) {
        REQUIRE_FALSE(original.uses_reconstruction(keep_records));
        REQUIRE(&original.sampling_plan(keep_records) == &original.original());
    }
    const auto selected = Program::compile(source, hir, true);
    for (bool keep_records : {false, true}) {
        REQUIRE(selected.uses_reconstruction(keep_records));
        REQUIRE(selected.sampling_plan(keep_records).peak_active_width() == 1);
    }
    REQUIRE(selected.original().inspect() == original.original().inspect());
    auto limits = ReconstructionLimits{};
    limits.max_operations = source.nodes.size();
    const auto at_source_limit = Program::compile(source, hir, true, true, limits);
    --limits.max_operations;
    const auto oversized = Program::compile(source, hir, true, true, limits);
    for (bool keep_records : {false, true}) {
        REQUIRE(at_source_limit.uses_reconstruction(keep_records));
        REQUIRE(at_source_limit.sampling_plan(keep_records).inspect() ==
                selected.sampling_plan(keep_records).inspect());
    }
    limits = {};
    limits.max_dilation_qubits = 1;
    const auto bounded = Program::compile(source, hir, true, true, limits);
    const auto transformed = Program::compile(source, hir, true, false);
    for (const auto* fallback : {&bounded, &transformed, &oversized}) {
        for (bool keep_records : {false, true}) {
            REQUIRE_FALSE(fallback->uses_reconstruction(keep_records));
            REQUIRE_FALSE(fallback->reconstruction_fallback_reason(keep_records).empty());
            REQUIRE(&fallback->sampling_plan(keep_records) == &fallback->original());
            REQUIRE(fallback->original().inspect() == original.original().inspect());
        }
    }
}

// `pairs` Bell pairs whose T and T_DAG cancel unless an X error flips one sign,
// and `plain` qubits with one T gate. Planned in this order, a pair needs an
// active qubit in the ordinary plan and none on a runtime Clifford frame; a
// plain qubit needs one either way.
Circuit signed_pairs(uint32_t pairs, uint32_t plain) {
    std::string prepare;
    std::string rotate;
    std::string measure;
    for (uint32_t p = 0; p < pairs; ++p) {
        const std::string a = std::to_string(2 * p);
        const std::string b = std::to_string(2 * p + 1);
        prepare += "H " + a + "\nCX " + a + " " + b + "\nX_ERROR(0.05) " + a + "\n";
        rotate += "T " + a + "\nT_DAG " + b + "\n";
        measure += "MPP X" + a + "*X" + b + "\n";
    }
    for (uint32_t q = 2 * pairs; q < 2 * pairs + plain; ++q) {
        prepare += "H " + std::to_string(q) + "\n";
        rotate += "T " + std::to_string(q) + "\n";
        measure += "H " + std::to_string(q) + "\nM " + std::to_string(q) + "\n";
    }
    return parse(prepare + rotate + measure);
}

TEST_CASE("Program keeps a runtime Clifford frame plan on request") {
    const Circuit source = signed_pairs(4, 1);
    const HirModule hir = trace(source);
    const auto compile = [&](bool runtime_frame) {
        return Program::compile(source, hir, false, true, {}, runtime_frame);
    };
    const Program ordinary = compile(false);
    REQUIRE_FALSE(ordinary.original().uses_runtime_frame());
    REQUIRE(ordinary.original().peak_active_width() == 5);
    REQUIRE_FALSE(ordinary.runtime_frame_fallback_reason().empty());

    const Program frame = compile(true);
    REQUIRE(frame.original().uses_runtime_frame());
    REQUIRE(frame.original().peak_active_width() == 1);
    REQUIRE(frame.runtime_frame_fallback_reason().empty());
    REQUIRE(frame.num_visible_records() == ordinary.num_visible_records());
    REQUIRE(&frame.sampling_plan(true) == &frame.original());

    // The frame plan starts from the source circuit, whatever passes ran on
    // the given HIR, unless they change the sampling distribution.
    HirModule optimized = hir;
    default_hir_pass_manager().run(optimized);
    const std::string after_passes =
        sampling::ExecutablePlan(sampling::plan_sampling(optimized, {.runtime_frame = true}))
            .inspect();
    REQUIRE(after_passes != frame.original().inspect());
    REQUIRE(Program::compile(source, optimized, false, true, {}, true).original().inspect() ==
            frame.original().inspect());
    REQUIRE(Program::compile(source, optimized, false, false, {}, true).original().inspect() ==
            after_passes);

    // A circuit without a sign that varies keeps its ordinary plan.
    const Circuit fixed = parse("H 0 1\nT 0 1\nCX 0 1\nT 1\nH 0 1\nM 0 1\n");
    const HirModule fixed_hir = trace(fixed);
    const Program unchanged = Program::compile(fixed, fixed_hir, false, true, {}, true);
    REQUIRE_FALSE(unchanged.original().uses_runtime_frame());
    REQUIRE_FALSE(unchanged.runtime_frame_fallback_reason().empty());
    REQUIRE(unchanged.original().inspect() ==
            Program::compile(fixed, fixed_hir, false).original().inspect());
}

TEST_CASE("Frame selection keeps a narrower plan that needs no frame", "[reconstruction]") {
    // T T is an S gate. The ordinary planner activates every qubit for the
    // first T; the frame planner reduces each pair and leaves no frame action.
    const auto circuit = [](uint32_t qubits) {
        std::string all;
        for (uint32_t q = 0; q < qubits; ++q) {
            all += " " + std::to_string(q);
        }
        return parse("H" + all + "\nT" + all + "\nT" + all + "\nH" + all + "\nM" + all + "\n");
    };
    const Circuit source = circuit(16);
    const HirModule hir = trace(source);
    const Program ordinary = Program::compile(source, hir, false);
    REQUIRE(ordinary.original().peak_active_width() == 16);
    const Program reduced = Program::compile(source, hir, false, true, {}, true);
    REQUIRE_FALSE(reduced.original().uses_runtime_frame());
    REQUIRE(reduced.original().peak_active_width() == 0);
    REQUIRE_FALSE(reduced.runtime_frame_fallback_reason().empty());
    REQUIRE(reduced.runtime_frame_fallback_reason() !=
            Program::compile(parse("H 0\nM 0\n"), trace(parse("H 0\nM 0\n")), false, true, {}, true)
                .runtime_frame_fallback_reason());

    // The reduced plan is the only one a dense state can hold.
    const Circuit wide = circuit(kDenseActiveWidthLimit + 5);
    const HirModule wide_hir = trace(wide);
    REQUIRE_THROWS_AS(Program::compile(wide, wide_hir, false), std::overflow_error);
    const Program fits = Program::compile(wide, wide_hir, false, true, {}, true);
    REQUIRE(fits.original().peak_active_width() == 0);
}

TEST_CASE("Frame selection reports what a frame does not support", "[reconstruction]") {
    const std::string pair = "H 0\nCX 0 1\nX_ERROR(0.1) 0\nT 0\nT_DAG 1\n";
    const Circuit source = parse(pair + "EXP_VAL Z0\nM 0 1\n");
    const HirModule hir = trace(source);
    const std::string_view unsupported = sampling::runtime_frame_unsupported_reason(hir);
    REQUIRE_FALSE(unsupported.empty());
    const Program probed = Program::compile(source, hir, false, true, {}, true);
    REQUIRE_FALSE(probed.original().uses_runtime_frame());
    REQUIRE(probed.runtime_frame_fallback_reason() == unsupported);

    const Circuit plain = parse(pair + "M 0 1\n");
    REQUIRE(sampling::runtime_frame_unsupported_reason(trace(plain)).empty());
    sampling::ExecutablePlan frame(sampling::plan_sampling(trace(plain), {.runtime_frame = true}));
    REQUIRE(frame.uses_runtime_frame());
    REQUIRE(Program(std::move(frame)).runtime_frame_fallback_reason().empty());
}

TEST_CASE("A reconstruction replaces a runtime frame plan only up to its static width",
          "[reconstruction]") {
    // Each loose qubit gets two T gates with a fault between them and, after a
    // basis change, a third. On a frame the pair needs no active qubit and the
    // third T needs one only in shots where the fault fired, which the plan
    // cannot tell: its width bound counts every loose qubit. A plain qubit
    // needs one active qubit in every shot. Reconstruction measures the qubits
    // one at a time and runs at width one.
    const auto circuit = [](uint32_t loose, uint32_t plain) {
        std::string prepare;
        std::string rotate;
        std::string turn;
        std::string measure;
        for (uint32_t q = 0; q < loose + plain; ++q) {
            const std::string s = std::to_string(q);
            prepare += "H " + s + "\n";
            if (q < loose) {
                rotate += "T " + s + "\nX_ERROR(0.05) " + s + "\nT " + s + "\n";
                turn += "H_YZ " + s + "\nT " + s + "\n";
                measure += "M " + s + "\nDETECTOR rec[-1]\n";
            } else {
                rotate += "T " + s + "\n";
                measure += "H " + s + "\nM " + s + "\nDETECTOR rec[-1]\n";
            }
        }
        return parse(prepare + rotate + turn + measure);
    };
    struct Case {
        uint32_t plain;
        bool reconstructed;
    };
    for (const Case& c : {Case{0, false}, Case{1, true}}) {
        CAPTURE(c.plain);
        const Circuit source = circuit(4, c.plain);
        const HirModule hir = trace(source);
        const Program ordinary = Program::compile(source, hir);
        const Program frame = Program::compile(source, hir, std::nullopt, true, {}, true);
        REQUIRE(frame.original().uses_runtime_frame());
        REQUIRE(frame.original().peak_active_width() == 4 + c.plain);
        REQUIRE(frame.original().initial_capacity_width() == c.plain);
        for (bool keep_records : {false, true}) {
            CAPTURE(keep_records);
            REQUIRE(ordinary.sampling_plan(keep_records).peak_active_width() == 1);
            const auto& sampled = frame.sampling_plan(keep_records);
            REQUIRE((&sampled != &frame.original()) == c.reconstructed);
            REQUIRE(sampled.uses_runtime_frame() != c.reconstructed);
            REQUIRE(frame.reconstruction_fallback_reason(keep_records).empty() == c.reconstructed);
        }
    }
}

TEST_CASE("Program optimizes tied reconstruction schedules after lowering", "[reconstruction]") {
    const auto source = parse("H 0\nT 0\nT_DAG 0\nH 0\nM 0\nDETECTOR rec[-1]\n");
    const auto hir = trace(source);
    const auto original = Program::compile(source, hir, false);
    REQUIRE(original.original().peak_active_width() == 1);

    // A tied reconstruction schedule can collapse to width zero after post-lowering passes.
    const auto selected = Program::compile(source, hir);
    for (bool keep_records : {false, true}) {
        REQUIRE(selected.uses_reconstruction(keep_records));
        REQUIRE(selected.sampling_plan(keep_records).peak_active_width() == 0);
    }

    auto limits = ReconstructionLimits{};
    limits.max_lowering_bytes = 0;
    const auto guarded = Program::compile(source, hir, true, true, limits);
    for (bool keep_records : {false, true}) {
        REQUIRE_FALSE(guarded.uses_reconstruction(keep_records));
        REQUIRE_FALSE(guarded.reconstruction_fallback_reason(keep_records).empty());
        REQUIRE(&guarded.sampling_plan(keep_records) == &guarded.original());
    }
    REQUIRE(guarded.original().inspect() == original.original().inspect());
}

TEST_CASE("Program shares reconstruction preparation across asymmetric output variants",
          "[reconstruction]") {
    // With measurements only and zero detectors/observables, the detector-only
    // variant skips reconstruction while the record-preserving variant still
    // reconstructs from the shared preparation.
    {
        const auto source = parse("H 0 1\nT 0 1\nH 0 1\nM 0 1\n");
        const auto hir = trace(source);
        const auto program = Program::compile(source, hir, true);
        REQUIRE_FALSE(program.uses_reconstruction(false));
        REQUIRE(program.uses_reconstruction(true));
        REQUIRE(program.sampling_plan(true).peak_active_width() == 1);
    }

    // When only the record-preserving schedule exceeds max_schedule_operations,
    // detector-only sampling retains reconstruction while record sampling falls back.
    {
        const auto source = parse(
            "H 0 1 2\nT 0 1 2\nH 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\n"
            "OBSERVABLE_INCLUDE(0) rec[-3]\n");
        const auto hir = trace(source);

        // Detectors schedules 3 rotations + 1 output measurement on 3 qubits (4 + 3 = 7),
        // while Records schedules 3 rotations + 3 distinct output measurements (6 + 3 = 9).
        ReconstructionLimits limits;
        limits.max_schedule_operations = 7;
        const auto program = Program::compile(source, hir, true, true, limits);
        REQUIRE(program.uses_reconstruction(false));
        REQUIRE_FALSE(program.uses_reconstruction(true));
        REQUIRE_FALSE(program.reconstruction_fallback_reason(true).empty());
        REQUIRE(&program.sampling_plan(true) == &program.original());
    }
}

TEST_CASE("Selective preparation removes unrequested storage and work") {
    using sampling::RecordOutputs;
    const auto program = compile_preparation_program({});
    const auto& original = program.original();
    const auto& syndromes = program.sampling_preparation(RecordOutputs::Syndromes).plan();
    REQUIRE(syndromes.num_visible_records() == 0);
    REQUIRE(syndromes.num_hidden_records() == 1);
    const sampling::BatchExecutor compact_worker(syndromes, 65);
    const sampling::BatchExecutor full_worker(original, 65);
    REQUIRE(compact_worker.record_columns().num_columns() <
            full_worker.record_columns().num_columns());
    REQUIRE(&program.sampling_preparation({RecordOutputs::Syndromes, true}).plan() == &syndromes);
    REQUIRE(syndromes.num_detectors() == original.num_detectors());
    REQUIRE(syndromes.num_observables() == original.num_observables());
    REQUIRE(syndromes.num_exp_vals() == 0);
    REQUIRE(syndromes.estimated_batch_lane_work(BatchOutputMode::Rows) <
            original.estimated_batch_lane_work(BatchOutputMode::Rows));
    const auto& measurements = program.sampling_preparation(RecordOutputs::Measurements).plan();
    REQUIRE(measurements.num_visible_records() == original.num_visible_records());
    REQUIRE(measurements.num_detectors() == 0);
    REQUIRE(measurements.num_observables() == 0);
    REQUIRE(measurements.num_exp_vals() == 0);
    REQUIRE(measurements.num_actions() < original.num_actions());
    const auto& summaries =
        program
            .sampling_preparation(RecordOutputs::Measurements,
                                  {.early_postselection = false, .survivor_counts = true})
            .plan();
    REQUIRE(summaries.num_observables() == original.num_observables());
    REQUIRE(summaries.num_detectors() == 0);
    REQUIRE(&program.sampling_preparation(RecordOutputs::Syndromes) ==
            &Program(program).sampling_preparation(RecordOutputs::Syndromes));
}

TEST_CASE("Program prepares each sampling output once and shares it across copies") {
    const Program program = compile_preparation_program({1, 0});
    const std::string inspected = program.sampling_plan(false).inspect();

    const auto& counts = program.sampling_preparation(false);
    const auto& rows = program.sampling_preparation(true);
    REQUIRE(&program.sampling_preparation(false) == &counts);
    REQUIRE(&program.sampling_preparation(true) == &rows);
    REQUIRE(&rows.plan() == &program.sampling_plan(true));
    REQUIRE(&counts.plan() != &program.sampling_plan(false));

    const auto& compiled = program.sampling_plan(false);
    REQUIRE(counts.plan().num_visible_records() == 0);
    REQUIRE(counts.plan().num_actions() < compiled.num_actions());
    REQUIRE(counts.plan().estimated_batch_lane_work(BatchOutputMode::Rows) ==
            compiled.estimated_batch_lane_work(BatchOutputMode::AggregateSurvivors));
    REQUIRE(rows.plan().num_exp_vals() == 1);

    std::vector<const void*> noise_addresses;
    for (BatchNoiseMode mode : kNoiseModes) {
        const auto& noise = counts.noise(mode);
        REQUIRE(&counts.noise(mode) == &noise);
        REQUIRE(noise.mode() == mode);
        noise_addresses.push_back(&noise);
    }
    std::ranges::sort(noise_addresses);
    REQUIRE(std::ranges::adjacent_find(noise_addresses) == noise_addresses.end());
    REQUIRE(&counts.checks() == &counts.checks());
    REQUIRE(&counts.sharing_key() == &counts.sharing_key());

    REQUIRE(&program.sampling_preparation(false, {.early_postselection = false}) != &counts);

    const Program copy = program;
    REQUIRE(&copy.sampling_preparation(false) == &counts);
    REQUIRE(&copy.original() == &program.original());

    REQUIRE(program.sampling_plan(false).inspect() == inspected);
    REQUIRE(&program.sampling_plan(false) == &program.original());

    const Program unselected = compile_preparation_program({0, 0});
    REQUIRE(&unselected.sampling_preparation(false, {.early_postselection = true}) ==
            &unselected.sampling_preparation(false, {.early_postselection = false}));
}

TEST_CASE("Concurrent first use prepares one shared sampling preparation") {
    const Program program = compile_preparation_program({1, 0});
    constexpr size_t kThreads = 8;
    std::array<std::array<const void*, 4>, kThreads> seen{};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (size_t thread = 0; thread < kThreads; ++thread) {
        threads.emplace_back([&, thread] {
            while (!start.load()) {
                std::this_thread::yield();
            }
            const auto& preparation = program.sampling_preparation(false);
            seen[thread] = {&preparation, &preparation.noise(BatchNoiseMode::SiteRows),
                            &preparation.checks(), &preparation.sharing_key()};
        });
    }
    start.store(true);
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto& addresses : seen) {
        REQUIRE(addresses == seen.front());
        for (const void* address : addresses) {
            REQUIRE(address != nullptr);
        }
    }
}

TEST_CASE("Count only sampling reuses dense noise with bounded packed storage") {
    using namespace sampling;
    const auto source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/msc_d3.stim");
    auto hir = trace(source);
    hir.postselection.mask.assign(hir.num_detectors, 1);
    default_hir_pass_manager().run(hir);
    const auto program = Program::compile(source, hir, false);
    const auto& preparation = program.sampling_preparation(false, {.early_postselection = false});
    const auto& plan = preparation.plan();
    const auto& dense = preparation.noise(BatchNoiseMode::Dense);
    REQUIRE(dense.num_carriers() > 0);
    REQUIRE(preparation.noise(BatchNoiseMode::SiteSparse).num_carriers() == 0);
    REQUIRE(&preparation.noise(BatchNoiseMode::Dense) == &dense);

    constexpr uint32_t capacity = 256;
    constexpr auto output = BatchOutputMode::AggregateSurvivors;
    for (auto mode : {BatchSamplingMode::Ordinary, BatchSamplingMode::FixedFaults}) {
        CAPTURE(mode);
        const auto dense_bytes =
            batch_detail::batch_worker_storage_bytes(plan, capacity, output, mode, &dense);
        REQUIRE(dense_bytes <
                batch_detail::batch_worker_storage_bytes(plan, capacity, output, mode));
        const uint64_t extra = kDefaultBatchWorkerBudget - dense_bytes;
        const auto policy = resolve_batch_execution_policy(plan, 4096, 1, 1, output, std::nullopt,
                                                           mode, extra, 0, &dense);
        REQUIRE(policy.lane_capacity == capacity);
        REQUIRE(resolve_batch_execution_policy(plan, 4096, 1, 1, output, std::nullopt, mode, extra)
                    .lane_capacity < capacity);
        const auto candidates = batch_detail::batch_tuning_candidates(
            plan, 4096, 1, 1, output, mode, extra, {1, 1}, 0, 0, &dense);
        REQUIRE(std::ranges::any_of(candidates, [](const auto& candidate) {
            return candidate.policy.lane_capacity == capacity;
        }));

        const auto sample_counts = [&](const PreparedBatchNoise* noise) {
            return mode == BatchSamplingMode::Ordinary
                       ? sample_survivors(plan, 4096, 73, false, 1, std::nullopt, capacity,
                                          std::nullopt, noise)
                       : sample_k_survivors(plan, 4096, 3, 73, false, 1, std::nullopt, capacity,
                                            std::nullopt, noise);
        };
        const auto expected = sample_counts(nullptr);
        const auto actual = sample_counts(&dense);
        REQUIRE(actual.passed_shots > 0);
        REQUIRE(actual.passed_shots < actual.total_shots);
        REQUIRE(actual.passed_shots == expected.passed_shots);
        REQUIRE(actual.logical_errors == expected.logical_errors);
        REQUIRE(actual.observable_ones == expected.observable_ones);
    }
}

TEST_CASE(
    "Program compiles reconstructed plans when the original HIR exceeds the dense width limit",
    "[reconstruction]") {
    using sampling::RecordOutputs;
    const auto make_wide = [](bool include_syndromes) {
        std::string ancillas;
        std::string parity_cx;
        for (uint32_t q = 2; q < 64; ++q) {
            ancillas += " " + std::to_string(q);
            parity_cx += " " + std::to_string(q) + " 64";
        }
        std::string text =
            "H 0 1" + ancillas + "\nT 0 1" + ancillas + "\nH 0 1\nCX" + parity_cx + "\nM 0 1 64\n";
        if (include_syndromes) {
            text += "DETECTOR rec[-3] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\n";
        }
        return parse(text);
    };

    const auto source = make_wide(true);
    auto hir = trace(source);
    default_hir_pass_manager().run(hir);
    REQUIRE_THROWS_AS(Program::compile(source, hir, false), std::overflow_error);

    // Both output variants reconstruct below kDenseActiveWidthLimit without an original plan.
    const auto both = Program::compile(source, hir, true);
    REQUIRE_FALSE(both.has_original());
    REQUIRE_THROWS_AS(both.original(), std::overflow_error);
    REQUIRE(both.num_qubits() == 65);
    REQUIRE(both.num_visible_records() == 3);
    REQUIRE(both.num_detectors() == 1);
    REQUIRE(both.num_observables() == 1);
    REQUIRE(both.num_exp_vals() == 0);
    REQUIRE_FALSE(both.has_postselection());
    for (bool keep_records : {false, true}) {
        REQUIRE(both.has_sampling_plan(keep_records));
        REQUIRE(both.uses_reconstruction(keep_records));
        REQUIRE(both.reconstruction_fallback_reason(keep_records).empty());
        REQUIRE(both.sampling_plan(keep_records).peak_active_width() == 1);
        REQUIRE(both.sampling_preparation(keep_records).plan().peak_active_width() == 1);
        REQUIRE_THROWS_AS(both.sampling_preparation(keep_records, {.fixed_faults = true}),
                          std::overflow_error);
    }

    // Asymmetric detector-only success: unreferenced graph-state measurements keep
    // retained-record active width >= kDenseActiveWidthLimit while syndromes fit.
    {
        std::string graph_qubits;
        std::string cz_edges;
        constexpr uint32_t kGraphCount = 160;
        for (uint32_t i = 0; i < kGraphCount; ++i) {
            graph_qubits += " " + std::to_string(2 + i);
        }
        for (uint32_t shift : {1U, 17U, 43U, 71U}) {
            for (uint32_t i = 0; i < kGraphCount; ++i) {
                cz_edges += " " + std::to_string(2 + i) + " " +
                            std::to_string(2 + ((i + shift) % kGraphCount));
            }
        }
        const auto detector_only_source = parse(
            "H 0 1" + graph_qubits + "\nT 0 1" + graph_qubits + "\nH 0 1\nCZ" + cz_edges + "\nMX" +
            graph_qubits + "\nM 0 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n");
        auto detector_only_hir = trace(detector_only_source);
        default_hir_pass_manager().run(detector_only_hir);
        const auto detector_only = Program::compile(detector_only_source, detector_only_hir, true);
        REQUIRE_FALSE(detector_only.has_original());
        REQUIRE(detector_only.has_sampling_plan(false));
        REQUIRE_FALSE(detector_only.has_sampling_plan(true));
        REQUIRE(detector_only.uses_reconstruction(false));
        REQUIRE_FALSE(detector_only.uses_reconstruction(true));
        REQUIRE(detector_only.reconstruction_fallback_reason(true) ==
                "reconstruction reaches the dense active-width limit");
        REQUIRE(detector_only.sampling_plan(false).peak_active_width() == 1);
        REQUIRE(detector_only.sampling_preparation(RecordOutputs::Syndromes)
                    .plan()
                    .peak_active_width() == 1);
        REQUIRE_THROWS_AS(detector_only.sampling_plan(true), std::overflow_error);
        REQUIRE_THROWS_AS(detector_only.sampling_preparation(true), std::overflow_error);
        REQUIRE_THROWS_AS(detector_only.sampling_preparation(RecordOutputs::Measurements),
                          std::overflow_error);
    }

    // Reconstructed noiseless reference shot succeeds when sequential reference planning overflows.
    {
        std::string ancillas;
        std::string parity_cx;
        for (uint32_t q = 2; q < 64; ++q) {
            ancillas += " " + std::to_string(q);
            parity_cx += " " + std::to_string(q) + " 64";
        }
        const auto noisy_det_source = parse(
            "H 0 1" + ancillas + "\nT 0 1" + ancillas + "\nCX" + parity_cx +
            "\nX 65\nX_ERROR(0.1) 65\nM 64 65\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
        const auto noisy_det_hir = trace(noisy_det_source);
        REQUIRE_THROWS_AS(compute_reference_syndrome(noisy_det_hir), std::overflow_error);
        const auto ref = compute_reference_syndrome(noisy_det_hir, &noisy_det_source);
        REQUIRE(ref.detectors == std::vector<uint8_t>{1});
        REQUIRE(ref.observables == std::vector<uint8_t>{1});
    }

    // Random parities sign a T pair on each of 40 Bell pairs, so the sequential
    // reference plan needs 40 active qubits: it plans, and no state holds it.
    // A frame plans the pairs at width zero.
    {
        std::string qubits;
        std::string evens;
        std::string odds;
        std::string parities;
        for (uint32_t p = 0; p < 40; ++p) {
            const std::string a = std::to_string(2 * p);
            const std::string b = std::to_string(2 * p + 1);
            qubits += " " + a + " " + b;
            evens += " " + a;
            odds += " " + b;
            parities += "MPP Z" + a + "*Z" + b + "\n";
        }
        const auto signed_hir = trace(
            parse("H" + qubits + "\n" + parities + "T" + evens + "\nT_DAG" + odds + "\nH" + qubits +
                  "\nM" + qubits + "\nMPAD 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"));
        const auto ref = compute_reference_syndrome(signed_hir, nullptr, {}, true);
        REQUIRE(ref.detectors == std::vector<uint8_t>{1});
        REQUIRE(ref.observables == std::vector<uint8_t>{1});
    }

    // 65 such pairs, chained by CX gates and checked one by one. The default
    // passes split the regions a frame reduces, so the frame reference needs
    // the circuit traced again; from the optimized HIR it stays too wide.
    {
        constexpr uint32_t kPairs = 65;
        std::string text;
        for (uint32_t p = 0; p < kPairs; ++p) {
            const std::string a = std::to_string(2 * p);
            const std::string b = std::to_string(2 * p + 1);
            text +=
                "H " + a + " " + b + "\nMPP Z" + a + "*Z" + b + "\nT " + a + "\nT_DAG " + b + "\n";
        }
        for (uint32_t p = 0; p + 1 < kPairs; ++p) {
            text += "CX " + std::to_string(2 * p) + " " + std::to_string(2 * p + 2) + "\n";
        }
        for (uint32_t p = 0; p < kPairs; ++p) {
            text += "MPP X" + std::to_string(2 * p) + "*X" + std::to_string(2 * p + 1) +
                    "\nDETECTOR rec[-1]\n";
        }
        const auto chained_source = parse(text);
        auto chained_hir = trace(chained_source);
        default_hir_pass_manager().run(chained_hir);
        const auto ref =
            compute_reference_syndrome(chained_hir, nullptr, {}, true, &chained_source);
        REQUIRE(ref.detectors.size() == kPairs);
        // It is the reference of the frame planned from the circuit's own trace.
        REQUIRE(ref.detectors ==
                compute_reference_syndrome(trace(chained_source), nullptr, {}, true).detectors);
    }

    // The frame plan of a reference states a bound that its one shot can
    // reach. Here 44 rotations each activate a qubit: the frame plan starts at
    // width zero and the shot grows to 44, while reconstruction needs none of
    // them for the detector. Reconstruction is preferred to the frame plan.
    {
        std::string rotated;
        std::string parity = "MPP Z8";
        for (uint32_t q = 8; q < 52; ++q) {
            rotated += " " + std::to_string(q);
            if (q != 8) {
                parity += "*Z" + std::to_string(q);
            }
        }
        const auto widening_source = parse(
            "H 0 1 2 3 4 5 6 7\nMPP Z0*Z1 Z2*Z3 Z4*Z5 Z6*Z7\nT 0 2 4 6\nT_DAG 1 3 5 7\nR_X(0.2)" +
            rotated + "\n" + parity +
            "\nMPAD 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
        const auto ref = compute_reference_syndrome(trace(widening_source), &widening_source, {},
                                                    true, &widening_source);
        REQUIRE(ref.detectors == std::vector<uint8_t>{1});
        REQUIRE(ref.observables == std::vector<uint8_t>{1});
    }

    // Forty rotations about Z on one qubit raise the frame plan's bound to
    // 40, the width of the sequential plan, although the shot keeps one
    // active qubit. With no reconstruction to turn to, the frame plan is the
    // reference a state can hold.
    {
        constexpr uint32_t kPairs = 40;
        std::string qubits;
        std::string evens;
        std::string odds;
        std::string parities;
        std::string rotations;
        for (uint32_t p = 0; p < kPairs; ++p) {
            const std::string a = std::to_string(2 * p);
            const std::string b = std::to_string(2 * p + 1);
            qubits += " " + a + " " + b;
            evens += " " + a;
            odds += " " + b;
            parities += "MPP Z" + a + "*Z" + b + "\n";
            rotations += "R_Z(0.2) 0\n";
        }
        const auto tied_hir =
            trace(parse("H" + qubits + "\n" + parities + "T" + evens + "\nT_DAG" + odds + "\n" +
                        rotations + "H" + qubits + "\nM" + qubits +
                        "\nMPAD 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"));
        const auto ref = compute_reference_syndrome(tied_hir, nullptr, {}, true);
        REQUIRE(ref.detectors == std::vector<uint8_t>{1});
    }

    // Asymmetric record-only success: zero detectors and observables skip detector reconstruction.
    const auto record_source = make_wide(false);
    auto record_hir = trace(record_source);
    default_hir_pass_manager().run(record_hir);
    const auto record_only = Program::compile(record_source, record_hir, true);
    REQUIRE_FALSE(record_only.has_original());
    REQUIRE_FALSE(record_only.has_sampling_plan(false));
    REQUIRE(record_only.has_sampling_plan(true));
    REQUIRE_FALSE(record_only.uses_reconstruction(false));
    REQUIRE(record_only.uses_reconstruction(true));
    REQUIRE(record_only.sampling_plan(true).peak_active_width() == 1);
    REQUIRE(record_only.sampling_preparation(true).plan().peak_active_width() == 1);
    REQUIRE_THROWS_AS(record_only.sampling_plan(false), std::overflow_error);
    REQUIRE_THROWS_AS(record_only.sampling_preparation(false), std::overflow_error);

    // Compilation fails when neither variant can reconstruct below kDenseActiveWidthLimit.
    REQUIRE_THROWS_AS(Program::compile(source, hir, true, false), std::overflow_error);
    ReconstructionLimits tight_dilation;
    tight_dilation.max_dilation_qubits = 1;
    REQUIRE_THROWS_AS(Program::compile(source, hir, true, true, tight_dilation),
                      std::overflow_error);
}

TEST_CASE("Automatic reconstruction guards small postselected drops and breaks ties by work",
          "[reconstruction]") {
    const auto d5_source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d5_r5.stim");
    auto d5_hir = trace(d5_source);
    d5_hir.postselection.mask.assign(d5_hir.num_detectors, 1);
    default_hir_pass_manager().run(d5_hir);

    const auto auto_postselected = Program::compile(d5_source, d5_hir);
    const auto forced_postselected = Program::compile(d5_source, d5_hir, true);
    for (bool keep_records : {false, true}) {
        REQUIRE_FALSE(auto_postselected.uses_reconstruction(keep_records));
        REQUIRE(forced_postselected.uses_reconstruction(keep_records));
        REQUIRE(forced_postselected.sampling_plan(keep_records).peak_active_width() + 2 ==
                auto_postselected.sampling_plan(keep_records).peak_active_width());
    }

    auto unselected_hir = trace(d5_source);
    default_hir_pass_manager().run(unselected_hir);
    const auto auto_unselected = Program::compile(d5_source, unselected_hir);
    for (bool keep_records : {false, true}) {
        REQUIRE(auto_unselected.uses_reconstruction(keep_records));
        REQUIRE(auto_unselected.sampling_plan(keep_records).peak_active_width() ==
                forced_postselected.sampling_plan(keep_records).peak_active_width());
    }

    // Small postselected width drops that do not inflate actions keep reconstruction.
    const auto compact_source = parse(R"(
        H 0 1 2
        T 0 1 2
        CX 0 1
        CX 1 2
        T 0 1 2
        H 0 1 2
        M 0 1 2
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )");
    auto compact_hir = trace(compact_source);
    compact_hir.postselection.mask.assign(compact_hir.num_detectors, 1);
    const auto compact_auto = Program::compile(compact_source, compact_hir);
    const auto compact_orig = Program::compile(compact_source, compact_hir, false);
    REQUIRE(compact_auto.uses_reconstruction(false));
    REQUIRE(compact_auto.sampling_plan(false).peak_active_width() + 1 ==
            compact_orig.sampling_plan(false).peak_active_width());
    REQUIRE(compact_auto.sampling_plan(false).num_actions() <=
            compact_orig.sampling_plan(false).num_actions());

    // Equal-width ties select reconstruction when actions do not grow and work is lower,
    // and keep the original otherwise.
    const auto msc3_source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/msc_d3.stim");
    auto msc3_hir = trace(msc3_source);
    msc3_hir.postselection.mask.assign(msc3_hir.num_detectors, 1);
    default_hir_pass_manager().run(msc3_hir);
    const auto msc3_auto = Program::compile(msc3_source, msc3_hir);
    const auto msc3_orig = Program::compile(msc3_source, msc3_hir, false);
    REQUIRE(msc3_auto.uses_reconstruction(false));
    REQUIRE(msc3_auto.sampling_plan(false).peak_active_width() ==
            msc3_orig.sampling_plan(false).peak_active_width());

    const auto d3_source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/coherent_d3_r3.stim");
    auto d3_hir = trace(d3_source);
    d3_hir.postselection.mask.assign(d3_hir.num_detectors, 1);
    default_hir_pass_manager().run(d3_hir);
    const auto d3_auto = Program::compile(d3_source, d3_hir);
    const auto d3_orig = Program::compile(d3_source, d3_hir, false);
    REQUIRE_FALSE(d3_auto.uses_reconstruction(false));
    REQUIRE(d3_auto.sampling_plan(false).peak_active_width() ==
            d3_orig.sampling_plan(false).peak_active_width());
    REQUIRE(d3_auto.reconstruction_fallback_reason(false) ==
            "original plan has lower or equal estimated work");
}

}  // namespace clifft::test
