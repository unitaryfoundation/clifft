#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/compiled_sampler.h"
#include "clifft/sampling/planner.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using clifft::sampling::CompiledSampler;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::SamplingBitOutput;
using clifft::sampling::SamplingBitPacking;
using clifft::sampling::SamplingBitSource;
using clifft::sampling::SamplingOutputSelection;
using clifft::sampling::SamplingResult;

namespace {

std::shared_ptr<const ExecutablePlan> compiled_test_plan() {
    return std::make_shared<const ExecutablePlan>(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
            X_ERROR(0.25) 0
            H 1
            M 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
        )"))));
}

SamplingResult sample_all(CompiledSampler& sampler, uint32_t shots) {
    const ExecutablePlan& plan = sampler.plan();
    const SamplingOutputSelection available = sampler.available_outputs();
    SamplingResult result;
    std::vector<SamplingBitOutput> destinations;
    if (available.measurements) {
        result.measurements.resize(static_cast<size_t>(shots) * plan.num_visible_records());
        destinations.push_back(SamplingBitOutput{
            .source = SamplingBitSource::Measurements,
            .data = result.measurements,
            .row_stride = plan.num_visible_records(),
        });
    }
    if (available.detectors) {
        result.detectors.resize(static_cast<size_t>(shots) * plan.num_detectors());
        destinations.push_back(SamplingBitOutput{
            .source = SamplingBitSource::Detectors,
            .data = result.detectors,
            .row_stride = plan.num_detectors(),
        });
    }
    if (available.observables) {
        result.observables.resize(static_cast<size_t>(shots) * plan.num_observables());
        destinations.push_back(SamplingBitOutput{
            .source = SamplingBitSource::Observables,
            .data = result.observables,
            .row_stride = plan.num_observables(),
        });
    }
    sampler.sample(shots, {.bits = destinations});
    return result;
}

}  // namespace

TEST_CASE("Compiled sampler retains a reproducible advancing stream") {
    const auto plan = compiled_test_plan();
    constexpr SamplingOutputSelection outputs{
        .measurements = true,
        .detectors = true,
        .observables = true,
    };
    CompiledSampler first(plan, outputs, uint64_t{81234}, 1, uint32_t{65});
    CompiledSampler replay(plan, outputs, uint64_t{81234}, 1, uint32_t{65});

    const SamplingResult first_call = sample_all(first, 129);
    const SamplingResult replay_first_call = sample_all(replay, 129);
    REQUIRE(first_call.measurements == replay_first_call.measurements);
    REQUIRE(first_call.detectors == replay_first_call.detectors);
    REQUIRE(first_call.observables == replay_first_call.observables);

    const SamplingResult second_call = sample_all(first, 129);
    const SamplingResult replay_second_call = sample_all(replay, 129);
    REQUIRE(second_call.measurements == replay_second_call.measurements);
    REQUIRE(second_call.detectors == replay_second_call.detectors);
    REQUIRE(second_call.observables == replay_second_call.observables);
    REQUIRE(second_call.measurements != first_call.measurements);
    REQUIRE(first.calls_completed() == 2);
    REQUIRE(replay.calls_completed() == 2);
    REQUIRE(first.lane_capacity() == 65);
}

TEST_CASE("Compiled sampler shot streams ignore retained worker count") {
    const auto plan = compiled_test_plan();
    constexpr SamplingOutputSelection outputs{
        .measurements = true,
        .detectors = true,
        .observables = true,
    };
    CompiledSampler serial(plan, outputs, uint64_t{81235}, 1, uint32_t{65});
    CompiledSampler threaded(plan, outputs, uint64_t{81235}, 4, uint32_t{65});
    const SamplingResult serial_result = sample_all(serial, 521);
    const SamplingResult threaded_result = sample_all(threaded, 521);
    REQUIRE(serial_result.measurements == threaded_result.measurements);
    REQUIRE(serial_result.detectors == threaded_result.detectors);
    REQUIRE(serial_result.observables == threaded_result.observables);
    REQUIRE(serial.worker_count() == 1);
    REQUIRE(threaded.worker_count() == 4);
}

TEST_CASE("Compiled sampler writes packed composed outputs directly") {
    const auto plan = compiled_test_plan();
    constexpr SamplingOutputSelection outputs{
        .detectors = true,
        .observables = true,
    };
    CompiledSampler sampler(plan, outputs, uint64_t{81236}, 1, uint32_t{65});
    CompiledSampler unpacked_sampler(plan, outputs, uint64_t{81236}, 1, uint32_t{65});
    constexpr uint32_t shots = 129;
    const size_t combined_columns = plan->num_observables() + plan->num_detectors();
    const size_t combined_stride = (combined_columns + 7) / 8;
    std::vector<uint8_t> combined(static_cast<size_t>(shots) * combined_stride, 0xFF);
    const std::array<SamplingBitOutput, 2> packed_destinations{
        SamplingBitOutput{
            .source = SamplingBitSource::Observables,
            .packing = SamplingBitPacking::BitPacked,
            .data = combined,
            .row_stride = combined_stride,
        },
        SamplingBitOutput{
            .source = SamplingBitSource::Detectors,
            .packing = SamplingBitPacking::BitPacked,
            .data = combined,
            .row_stride = combined_stride,
            .column_offset = plan->num_observables(),
        },
    };
    sampler.sample(shots, {.bits = packed_destinations});

    SamplingResult unpacked;
    unpacked.detectors.resize(static_cast<size_t>(shots) * plan->num_detectors());
    unpacked.observables.resize(static_cast<size_t>(shots) * plan->num_observables());
    const std::array<SamplingBitOutput, 2> unpacked_destinations{
        SamplingBitOutput{.source = SamplingBitSource::Detectors,
                          .data = unpacked.detectors,
                          .row_stride = plan->num_detectors()},
        SamplingBitOutput{.source = SamplingBitSource::Observables,
                          .data = unpacked.observables,
                          .row_stride = plan->num_observables()},
    };
    unpacked_sampler.sample(shots, {.bits = unpacked_destinations});

    for (uint32_t shot = 0; shot < shots; ++shot) {
        for (uint32_t observable = 0; observable < plan->num_observables(); ++observable) {
            const uint8_t bit =
                (combined[static_cast<size_t>(shot) * combined_stride + (observable >> 3)] >>
                 (observable & 7)) &
                1;
            REQUIRE(bit ==
                    unpacked.observables[static_cast<size_t>(shot) * plan->num_observables() +
                                         observable]);
        }
        for (uint32_t detector = 0; detector < plan->num_detectors(); ++detector) {
            const size_t column = plan->num_observables() + detector;
            const uint8_t bit =
                (combined[static_cast<size_t>(shot) * combined_stride + (column >> 3)] >>
                 (column & 7)) &
                1;
            REQUIRE(
                bit ==
                unpacked.detectors[static_cast<size_t>(shot) * plan->num_detectors() + detector]);
        }
    }
}

TEST_CASE("Compiled sampler streams composed files from one advancing call") {
    const auto plan = compiled_test_plan();
    constexpr SamplingOutputSelection outputs{
        .detectors = true,
        .observables = true,
    };
    CompiledSampler sampler(plan, outputs, uint64_t{81237}, 2, uint32_t{65});
    CompiledSampler expected_sampler(plan, outputs, uint64_t{81237}, 2, uint32_t{65});
    constexpr uint32_t shots = 129;
    const SamplingResult expected = sample_all(expected_sampler, shots);

    std::ostringstream main_output(std::ios::binary);
    std::ostringstream observable_output;
    constexpr std::array main_sources{
        SamplingBitSource::Observables,
        SamplingBitSource::Detectors,
        SamplingBitSource::Observables,
    };
    constexpr std::array observable_sources{SamplingBitSource::Observables};
    const std::array files{
        clifft::sampling::SamplingFileOutput{
            .output = &main_output,
            .format = clifft::sampling::SamplingFileFormat::B8,
            .sources = main_sources,
        },
        clifft::sampling::SamplingFileOutput{
            .output = &observable_output,
            .format = clifft::sampling::SamplingFileFormat::Format01,
            .sources = observable_sources,
        },
    };
    sampler.sample_write(shots, files);

    const std::string packed = main_output.str();
    REQUIRE(packed.size() == shots);
    std::string observable_text;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        const uint8_t observable = expected.observables[shot];
        const uint8_t detector0 = expected.detectors[static_cast<size_t>(shot) * 2];
        const uint8_t detector1 = expected.detectors[static_cast<size_t>(shot) * 2 + 1];
        const uint8_t expected_byte =
            observable | (detector0 << 1) | (detector1 << 2) | (observable << 3);
        REQUIRE(static_cast<uint8_t>(packed[shot]) == expected_byte);
        observable_text += static_cast<char>('0' + observable);
        observable_text += '\n';
    }
    REQUIRE(observable_output.str() == observable_text);
    REQUIRE(sampler.calls_completed() == 1);
}

TEST_CASE("Compiled sampler validates retained sources before advancing") {
    const auto plan = compiled_test_plan();
    constexpr SamplingOutputSelection measurements_only{.measurements = true};
    CompiledSampler sampler(plan, measurements_only, uint64_t{81238}, 1, uint32_t{65});
    CompiledSampler replay(plan, measurements_only, uint64_t{81238}, 1, uint32_t{65});

    std::vector<uint8_t> unavailable(plan->num_detectors());
    const SamplingBitOutput unavailable_destination{
        .source = SamplingBitSource::Detectors,
        .data = unavailable,
        .row_stride = plan->num_detectors(),
    };
    REQUIRE_THROWS_AS(sampler.sample(1, {.bits = std::span(&unavailable_destination, 1)}),
                      std::invalid_argument);
    sampler.sample(0, {});
    REQUIRE(sampler.calls_completed() == 0);
    REQUIRE(sample_all(sampler, 65).measurements == sample_all(replay, 65).measurements);
}

TEST_CASE("Compiled sampler retains packed rows from surviving shots") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        X_ERROR(0.5) 0
        M 0
        DETECTOR rec[-1]
        X_ERROR(1) 1
        M 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    constexpr std::array<uint8_t, 2> postselection{1, 0};
    const auto plan = std::make_shared<const ExecutablePlan>(
        clifft::sampling::plan_sampling(hir, {.postselection_mask = postselection}));
    constexpr SamplingOutputSelection outputs{.detectors = true, .observables = true};
    CompiledSampler sampler(plan, outputs, uint64_t{81239}, 4, uint32_t{65});

    constexpr uint32_t shots = 513;
    std::vector<uint8_t> detectors(shots, 0xFF);
    std::vector<uint8_t> observables(shots, 0xFF);
    const std::array destinations{
        SamplingBitOutput{.source = SamplingBitSource::Detectors,
                          .packing = SamplingBitPacking::BitPacked,
                          .data = detectors,
                          .row_stride = 1},
        SamplingBitOutput{.source = SamplingBitSource::Observables,
                          .packing = SamplingBitPacking::BitPacked,
                          .data = observables,
                          .row_stride = 1},
    };
    const uint32_t survivors = sampler.sample_survivors(shots, {.bits = destinations});

    REQUIRE(survivors > 200);
    REQUIRE(survivors < 310);
    for (uint32_t shot = 0; shot < survivors; ++shot) {
        REQUIRE(detectors[shot] == 0b10);
        REQUIRE(observables[shot] == 0b1);
    }
    REQUIRE(sampler.calls_completed() == 1);
    REQUIRE_THROWS_AS(sampler.sample(shots, {.bits = destinations}), std::invalid_argument);
}
