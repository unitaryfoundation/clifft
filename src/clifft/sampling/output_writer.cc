#include "clifft/sampling/output_writer.h"

#include <limits>
#include <ostream>
#include <stdexcept>

namespace clifft::sampling {

namespace {

size_t checked_product(size_t left, size_t right, const char* message) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::length_error(message);
    }
    return left * right;
}

void write_bytes(std::ostream& output, const char* data, size_t size) {
    if (size == 0) {
        return;
    }
    if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("sampling output batch exceeds streamsize range");
    }
    output.write(data, static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("failed to write sampling output");
    }
}

}  // namespace

SamplingRowWriter::SamplingRowWriter(std::ostream& output, SamplingFileFormat format,
                                     size_t columns, uint32_t max_batch_shots)
    : output_(&output),
      format_(format),
      columns_(columns),
      packed_row_bytes_(columns / 8 + static_cast<size_t>((columns & 7) != 0)),
      max_batch_shots_(max_batch_shots) {
    if (max_batch_shots == 0) {
        throw std::invalid_argument("sampling writer batch capacity must be positive");
    }
    if (format_ != SamplingFileFormat::Format01 && format_ != SamplingFileFormat::B8) {
        throw std::invalid_argument("unsupported sampling file format");
    }
    if (format_ == SamplingFileFormat::Format01) {
        if (columns_ == std::numeric_limits<size_t>::max()) {
            throw std::length_error("sampling 01 row width exceeds size_t range");
        }
        scratch_.resize(checked_product(max_batch_shots_, columns_ + 1,
                                        "sampling 01 batch exceeds size_t range"));
    }
}

void SamplingRowWriter::write_packed_rows(std::span<const uint8_t> rows, uint32_t shots,
                                          size_t row_stride) {
    if (shots > max_batch_shots_) {
        throw std::invalid_argument("sampling writer batch exceeds retained capacity");
    }
    if (row_stride < packed_row_bytes_) {
        throw std::invalid_argument("sampling writer row stride is too small");
    }
    const size_t required =
        checked_product(shots, row_stride, "sampling writer input exceeds size_t range");
    if (rows.size() < required) {
        throw std::invalid_argument("sampling writer input buffer is too small");
    }
    if (shots == 0) {
        return;
    }

    if (format_ == SamplingFileFormat::B8) {
        if (row_stride == packed_row_bytes_) {
            write_bytes(*output_, reinterpret_cast<const char*>(rows.data()),
                        static_cast<size_t>(shots) * packed_row_bytes_);
            return;
        }
        for (uint32_t shot = 0; shot < shots; ++shot) {
            write_bytes(
                *output_,
                reinterpret_cast<const char*>(rows.data() + static_cast<size_t>(shot) * row_stride),
                packed_row_bytes_);
        }
        return;
    }

    const size_t encoded_stride = columns_ + 1;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        const uint8_t* input = rows.data() + static_cast<size_t>(shot) * row_stride;
        char* encoded = scratch_.data() + static_cast<size_t>(shot) * encoded_stride;
        for (size_t column = 0; column < columns_; ++column) {
            encoded[column] = static_cast<char>('0' + ((input[column >> 3] >> (column & 7)) & 1));
        }
        encoded[columns_] = '\n';
    }
    write_bytes(*output_, scratch_.data(), static_cast<size_t>(shots) * encoded_stride);
}

}  // namespace clifft::sampling
