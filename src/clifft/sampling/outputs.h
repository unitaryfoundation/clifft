#pragma once

// Requested sampling records and their row representation. Survivor summaries
// still consume observables even when their per-shot rows are omitted.

#include <cstddef>

namespace clifft::sampling {

enum class RecordOutputs { None, All, Syndromes, Measurements };

struct SamplingOutputs {
    RecordOutputs records = RecordOutputs::All;
    bool bit_packed = false;

    constexpr SamplingOutputs() = default;
    constexpr SamplingOutputs(RecordOutputs records, bool bit_packed = false)
        : records(records), bit_packed(bit_packed) {}
    constexpr SamplingOutputs(bool keep_records)
        : records(keep_records ? RecordOutputs::All : RecordOutputs::None) {}

    [[nodiscard]] constexpr bool rows() const { return records != RecordOutputs::None; }
    [[nodiscard]] constexpr bool measurements() const {
        return records == RecordOutputs::All || records == RecordOutputs::Measurements;
    }
    [[nodiscard]] constexpr bool syndromes() const {
        return records == RecordOutputs::All || records == RecordOutputs::Syndromes;
    }
    [[nodiscard]] constexpr bool expectations() const { return records == RecordOutputs::All; }
    [[nodiscard]] constexpr size_t stride(size_t columns) const {
        return bit_packed ? columns / 8 + (columns % 8 != 0) : columns;
    }
};

}  // namespace clifft::sampling
