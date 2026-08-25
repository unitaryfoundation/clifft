// Compares the current shot-major rotation path with the experimental
// basis-major, shot-interleaved path. Allocation and descriptor preparation
// stay outside the measured interval; state reset and sign materialization do
// not.

#include "clifft/sampling/interleaved_batch_kernels.h"
#include "clifft/sampling/kernel_dispatch.h"
#include "clifft/sampling/kernels.h"
#include "clifft/util/runtime_isa.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using clifft::sampling::DirectRotationKernel;
using clifft::sampling::ExecutorBackend;
using clifft::sampling::PreparedRotation;

int get_env_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::stoi(value);
}

void apply_production_rotation(clifft::sampling::State& state, const PreparedRotation& rotation,
                               bool sign, ExecutorBackend backend) noexcept {
    const DirectRotationKernel kernel =
        clifft::sampling::resolve_direct_rotation_kernel(rotation, backend);
    if (kernel == DirectRotationKernel::Scalar) {
        clifft::sampling::apply_rotation(state, rotation, sign);
    } else if (backend == ExecutorBackend::Avx2) {
        clifft::sampling::apply_direct_rotation_avx2(state, rotation, kernel, sign);
    } else {
        clifft::sampling::apply_direct_rotation_avx512(state, rotation, kernel, sign);
    }
}

uint64_t checksum_scalar(const clifft::sampling::State& state) {
    return std::bit_cast<uint64_t>(state.real_data()[0]) ^
           std::rotl(std::bit_cast<uint64_t>(state.imag_data()[0]), 17);
}

uint64_t checksum_interleaved(const clifft::sampling::InterleavedBatchState& state) {
    uint64_t result = 0;
    for (uint32_t lane = 0; lane < state.active_lanes(); ++lane) {
        result ^= std::rotl(std::bit_cast<uint64_t>(state.real_basis(0)[lane]), lane & 63U);
        result ^= std::rotl(std::bit_cast<uint64_t>(state.imag_basis(0)[lane]), (lane + 17U) & 63U);
    }
    return result;
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main() {
    const int width = get_env_int("CLIFFT_INTERLEAVED_WIDTH", 4);
    const int lanes = get_env_int("CLIFFT_INTERLEAVED_LANES", 2048);
    const int actions = get_env_int("CLIFFT_INTERLEAVED_ACTIONS", 128);
    const int warmups = get_env_int("CLIFFT_INTERLEAVED_WARMUPS", 2);
    const int repetitions = get_env_int("CLIFFT_INTERLEAVED_REPETITIONS", 9);
    const int fused_mode = get_env_int("CLIFFT_INTERLEAVED_FUSED", 0);
    const bool fused = fused_mode != 0;
    if (width < 1 || width >= 32 || lanes < 1 || actions < 1 || warmups < 0 || repetitions < 1) {
        std::cerr << "Error: invalid interleaved profiler configuration\n";
        return 1;
    }

    const auto runtime_isa = clifft::internal::runtime_isa();
    clifft::internal::validate_runtime_isa(runtime_isa);
    const ExecutorBackend backend = clifft::sampling::resolve_executor_backend(runtime_isa);
    const uint64_t mask = (uint64_t{1} << width) - 1;
    std::vector<PreparedRotation> rotations;
    std::vector<clifft::sampling::PlannedAction> planned_rotations;
    rotations.reserve(static_cast<size_t>(actions));
    planned_rotations.reserve(static_cast<size_t>(actions));
    for (int action = 0; action < actions; ++action) {
        uint64_t x =
            fused ? uint64_t{1} << (action & 1) : (static_cast<uint64_t>(action) * 13U + 5U) & mask;
        uint64_t z = fused ? ((action & 2) != 0 ? uint64_t{5} & mask : uint64_t{1})
                           : (static_cast<uint64_t>(action) * 7U + 3U) & mask;
        if (x == 0 && z == 0) {
            x = 1;
        }
        const double half_turns = 0.071 + 0.001 * static_cast<double>(action % 23);
        rotations.push_back(clifft::sampling::prepare_rotation({x, z}, width, half_turns));
        planned_rotations.push_back(clifft::sampling::PlannedAction{
            static_cast<uint32_t>(width), static_cast<uint32_t>(width),
            clifft::sampling::RotateActivePauli{
                {x, z}, half_turns, clifft::sampling::AffineBool{}}});
    }
    const clifft::sampling::FusedRotationRun fused_run =
        clifft::sampling::prepare_fused_rotation_run(planned_rotations);
    if (fused &&
        (!fused_run.rotation.has_value() || fused_run.action_count != planned_rotations.size())) {
        std::cerr << "Error: profiler rotation run did not fuse completely\n";
        return 1;
    }
    std::vector<clifft::sampling::PreparedFusedRotation> dynamic_variants;
    std::vector<const clifft::sampling::PreparedFusedRotation*> dynamic_variant_pointers;
    std::vector<uint8_t> lane_variants(static_cast<size_t>(lanes));
    if (fused_mode == 2) {
        dynamic_variants.assign(4, *fused_run.rotation);
        for (size_t variant = 1; variant < dynamic_variants.size(); ++variant) {
            for (std::complex<double>& value : dynamic_variants[variant].matrices) {
                value *= std::complex<double>{1.0, 0.001 * static_cast<double>(variant)};
            }
        }
        for (int lane = 0; lane < lanes; ++lane) {
            lane_variants[static_cast<size_t>(lane)] =
                static_cast<uint8_t>((lane * 3 + lane / 7) % dynamic_variants.size());
        }
        for (const auto& variant : dynamic_variants) {
            dynamic_variant_pointers.push_back(&variant);
        }
    }

    std::vector<uint8_t> signs(static_cast<size_t>(actions) * lanes);
    for (int action = 0; action < actions; ++action) {
        for (int lane = 0; lane < lanes; ++lane) {
            signs[static_cast<size_t>(action) * lanes + lane] =
                static_cast<uint8_t>(((action * 17 + lane * 11) % 7) < 3);
        }
    }
    std::vector<double> signed_sines(static_cast<size_t>(lanes));
    clifft::sampling::State scalar(static_cast<uint32_t>(width), static_cast<uint32_t>(width));
    clifft::sampling::InterleavedBatchState interleaved(
        static_cast<uint32_t>(width), static_cast<uint32_t>(width), static_cast<uint32_t>(lanes));

    uint64_t checksum = 0;
    std::vector<double> scalar_samples;
    std::vector<double> interleaved_samples;
    scalar_samples.reserve(static_cast<size_t>(repetitions));
    interleaved_samples.reserve(static_cast<size_t>(repetitions));
    const int total_iterations = warmups + repetitions;
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        for (int lane = 0; lane < lanes; ++lane) {
            scalar.reset();
            if (fused_mode == 2) {
                clifft::sampling::apply_fused_rotation(
                    scalar, dynamic_variants[lane_variants[static_cast<size_t>(lane)]]);
            } else if (fused) {
                clifft::sampling::apply_fused_rotation(scalar, *fused_run.rotation);
            } else {
                for (int action = 0; action < actions; ++action) {
                    apply_production_rotation(
                        scalar, rotations[static_cast<size_t>(action)],
                        signs[static_cast<size_t>(action) * lanes + lane] != 0, backend);
                }
            }
            checksum ^= std::rotl(checksum_scalar(scalar), lane & 63);
        }
        auto end = std::chrono::steady_clock::now();
        if (iteration >= warmups) {
            scalar_samples.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
        }

        start = std::chrono::steady_clock::now();
        interleaved.reset(static_cast<uint32_t>(lanes));
        if (fused_mode == 2) {
            clifft::sampling::apply_interleaved_dynamic_fused_rotation(
                interleaved, dynamic_variant_pointers, lane_variants);
        } else if (fused) {
            clifft::sampling::apply_interleaved_fused_rotation(interleaved, *fused_run.rotation);
        } else {
            for (int action = 0; action < actions; ++action) {
                const std::span<const uint8_t> action_signs(
                    signs.data() + static_cast<size_t>(action) * lanes, static_cast<size_t>(lanes));
                clifft::sampling::prepare_interleaved_rotation_sines(
                    signed_sines, rotations[static_cast<size_t>(action)].sine, action_signs);
                clifft::sampling::apply_interleaved_rotation(
                    interleaved, rotations[static_cast<size_t>(action)], signed_sines);
            }
        }
        checksum ^= checksum_interleaved(interleaved);
        end = std::chrono::steady_clock::now();
        if (iteration >= warmups) {
            interleaved_samples.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
        }
    }

    const double scalar_ms = median(std::move(scalar_samples));
    const double interleaved_ms = median(std::move(interleaved_samples));
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "width=" << width << " lanes=" << lanes << " actions=" << actions
              << " fused=" << fused_mode
              << " isa=" << clifft::internal::runtime_isa_name(runtime_isa) << '\n';
    std::cout << "scalar_ms=" << scalar_ms << '\n';
    std::cout << "interleaved_ms=" << interleaved_ms << '\n';
    std::cout << "speedup=" << scalar_ms / interleaved_ms << '\n';
    std::cout << "checksum=" << checksum << '\n';
    return 0;
}
