#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <vector>

namespace clifft::sampling {

enum class SamplingFileFormat : uint8_t {
    Format01,
    B8,
};

// Encodes completed little-endian packed batches into the two common Stim
// result formats. Format01 scratch is retained at construction so each write
// is an allocation-free batch-boundary operation.
class SamplingRowWriter {
  public:
    SamplingRowWriter(std::ostream& output, SamplingFileFormat format, size_t columns,
                      uint32_t max_batch_shots);

    void write_packed_rows(std::span<const uint8_t> rows, uint32_t shots, size_t row_stride);

    [[nodiscard]] size_t columns() const noexcept { return columns_; }
    [[nodiscard]] size_t packed_row_bytes() const noexcept { return packed_row_bytes_; }

  private:
    std::ostream* output_;
    SamplingFileFormat format_;
    size_t columns_;
    size_t packed_row_bytes_;
    uint32_t max_batch_shots_;
    std::vector<char> scratch_;
};

}  // namespace clifft::sampling
