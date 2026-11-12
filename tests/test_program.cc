#include "clifft/api/program.h"
#include "clifft/api/sampling_preparation.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>
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
    return Program::compile(source, hir);
}

}  // namespace

TEST_CASE("Program preserves the original plan across bounded reconstruction failures",
          "[reconstruction]") {
    const auto source = parse("H 0 1\nT 0 1\nH 0 1\nM 0 1\nDETECTOR rec[-1]\n");
    const auto hir = trace(source);
    const auto original = Program::compile(source, hir);
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

TEST_CASE("Program skips reconstruction lowering without a scheduled width reduction",
          "[reconstruction]") {
    const auto source = parse("H 0\nT 0\nT_DAG 0\nH 0\nM 0\nDETECTOR rec[-1]\n");
    const auto hir = trace(source);
    const auto original = Program::compile(source, hir);
    REQUIRE(original.original().peak_active_width() == 1);

    // Later passes can improve this candidate, but the scheduled width does
    // not justify their compilation cost under the default selection policy.
    for (auto outputs : {ReconstructionOutputs::Detectors, ReconstructionOutputs::Records}) {
        const auto result = reconstruct_global_sampling(source, outputs);
        REQUIRE(result.reconstruction);
        REQUIRE(result.reconstruction->peak_active_width == 1);
        auto lowered = lower_global_reconstruction(*result.reconstruction);
        REQUIRE(lowered.hir);
        default_hir_pass_manager().run(*lowered.hir);
        REQUIRE(analyze_active_width(*lowered.hir).peak_width == 0);
    }

    auto limits = ReconstructionLimits{};
    // A zero lowering budget makes an attempted lowering fail, so the fallback
    // reason also checks that the width guard runs before that work.
    limits.max_lowering_bytes = 0;
    const auto guarded = Program::compile(source, hir, true, true, limits);
    for (bool keep_records : {false, true}) {
        REQUIRE_FALSE(guarded.uses_reconstruction(keep_records));
        REQUIRE(guarded.reconstruction_fallback_reason(keep_records) ==
                "reconstruction schedule does not reduce peak active width");
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
            "H 0 1\nT 0 1\nH 0 1\nM 0 1 2 3\nDETECTOR rec[-4]\nOBSERVABLE_INCLUDE(0) "
            "rec[-3]\n");
        const auto hir = trace(source);
        const auto det = reconstruct_global_sampling(source, ReconstructionOutputs::Detectors);
        const auto rec = reconstruct_global_sampling(source, ReconstructionOutputs::Records);
        REQUIRE(det.reconstruction.has_value());
        REQUIRE(rec.reconstruction.has_value());
        REQUIRE(det.reconstruction->operations.size() < rec.reconstruction->operations.size());

        ReconstructionLimits limits;
        limits.max_schedule_operations =
            det.reconstruction->operations.size() + det.reconstruction->num_qubits;
        const auto program = Program::compile(source, hir, true, true, limits);
        REQUIRE(program.uses_reconstruction(false));
        REQUIRE_FALSE(program.uses_reconstruction(true));
        REQUIRE_FALSE(program.reconstruction_fallback_reason(true).empty());
        REQUIRE(&program.sampling_plan(true) == &program.original());
    }
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
    REQUIRE(counts.plan().num_expression_registers() < compiled.num_expression_registers());
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

    REQUIRE(&program.sampling_preparation(false, false) != &counts);

    const Program copy = program;
    REQUIRE(&copy.sampling_preparation(false) == &counts);
    REQUIRE(&copy.original() == &program.original());

    REQUIRE(program.sampling_plan(false).inspect() == inspected);
    REQUIRE(&program.sampling_plan(false) == &program.original());

    const Program unselected = compile_preparation_program({0, 0});
    REQUIRE(&unselected.sampling_preparation(false, true) ==
            &unselected.sampling_preparation(false, false));
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

}  // namespace clifft::test
