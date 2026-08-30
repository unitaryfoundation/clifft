#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling {

// Selects which row-major output matrices a sampling request materializes.
// Executors may still retain internal records needed to evaluate requested
// detector or observable parities.
struct SamplingOutputSelection {
    bool measurements = false;
    bool detectors = false;
    bool observables = false;
    bool exp_vals = false;

    [[nodiscard]] constexpr bool any() const noexcept {
        return measurements || detectors || observables || exp_vals;
    }

    [[nodiscard]] static constexpr SamplingOutputSelection all() noexcept {
        return {.measurements = true, .detectors = true, .observables = true, .exp_vals = true};
    }
};

enum class SamplingBitSource : uint8_t {
    Measurements,
    Detectors,
    Observables,
};

enum class SamplingBitPacking : uint8_t {
    Unpacked,
    BitPacked,
};

// Describes one destination for a Boolean output matrix. row_stride is in
// bytes for both formats. column_offset is a byte offset for unpacked output
// and a bit offset for packed output, which permits native detector/observable
// composition without a temporary matrix.
struct SamplingBitOutput {
    SamplingBitSource source = SamplingBitSource::Measurements;
    SamplingBitPacking packing = SamplingBitPacking::Unpacked;
    std::span<uint8_t> data;
    size_t row_stride = 0;
    size_t column_offset = 0;
};

// Caller-owned destinations for fixed-row sampling. A source may appear more
// than once, allowing observables to be emitted separately and composed into
// a detector matrix in the same pass. Destination storage is validated and
// cleared before execution; hot dispatch never allocates it.
struct SamplingOutputBuffer {
    std::span<const SamplingBitOutput> bits;
    std::span<double> exp_vals;
    size_t exp_val_row_stride = 0;
};

// Backend-neutral row-major outputs from ordinary sampling.
struct SamplingResult {
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
};

// Little-endian packed Boolean rows plus ordinary floating-point expectation
// rows. Each Boolean vector has shots * ceil(columns / 8) bytes when selected.
struct PackedSamplingResult {
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
};

// Backend-neutral outputs from postselected survivor sampling.
struct SamplingSurvivorResult {
    uint32_t total_shots = 0;
    uint32_t passed_shots = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
};

}  // namespace clifft::sampling
