#pragma once

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace ucc {

// =============================================================================
// SourceMap
// =============================================================================
//
// Encapsulates the CSR (Compressed Sparse Row) source line mapping and
// per-instruction active_k history that threads through the compiler
// pipeline. Each entry corresponds to one bytecode instruction.
//
// Internal layout:
//   data_[offsets_[i] .. offsets_[i+1])  = source lines for instruction i
//   active_k_[i]                         = active dimension k at instruction i
//
// The class provides semantic operations (append, copy_entry, merge_entries)
// that replace the manual CSR pointer arithmetic previously duplicated
// across every optimizer pass.

class SourceMap {
  public:
    SourceMap() = default;

    // True if no entries have been recorded.
    [[nodiscard]] bool empty() const { return offsets_.empty(); }

    // Number of instruction entries.
    [[nodiscard]] size_t size() const { return active_k_.size(); }

    // --- Builder API ---

    // Append source lines and active_k for one new output instruction.
    void append(std::span<const uint32_t> lines, uint32_t active_k) {
        if (offsets_.empty())
            offsets_.push_back(0);
        data_.insert(data_.end(), lines.begin(), lines.end());
        offsets_.push_back(static_cast<uint32_t>(data_.size()));
        active_k_.push_back(active_k);
    }

    // Copy one entry from another SourceMap (1:1 pass-through).
    void copy_entry(const SourceMap& src, size_t i) {
        assert(i < src.size());
        auto lines = src.lines_for(i);
        append(lines, src.active_k_at(i));
    }

    // Merge entries [begin, end) from another SourceMap into one output entry.
    // The active_k comes from the last instruction in the merged range.
    // Exploits CSR contiguity: the data for entries [begin, end) is the
    // contiguous range [offsets_[begin], offsets_[end]) in the source.
    void merge_entries(const SourceMap& src, size_t begin, size_t end) {
        assert(begin < end && end <= src.size());
        if (offsets_.empty())
            offsets_.push_back(0);
        uint32_t b = src.offsets_[begin];
        uint32_t e = src.offsets_[end];
        data_.insert(data_.end(), src.data_.data() + b, src.data_.data() + e);
        offsets_.push_back(static_cast<uint32_t>(data_.size()));
        active_k_.push_back(src.active_k_at(end - 1));
    }

    // Reserve capacity hints for the internal vectors.
    void reserve(size_t num_entries, size_t data_capacity = 0) {
        active_k_.reserve(num_entries);
        offsets_.reserve(num_entries + 1);
        if (data_capacity > 0)
            data_.reserve(data_capacity);
    }

    // --- Query API ---

    // Get source lines for instruction i.
    [[nodiscard]] std::span<const uint32_t> lines_for(size_t i) const {
        assert(i < size());
        uint32_t b = offsets_[i];
        uint32_t e = offsets_[i + 1];
        return {data_.data() + b, e - b};
    }

    // Get active_k for instruction i.
    [[nodiscard]] uint32_t active_k_at(size_t i) const {
        assert(i < size());
        return active_k_[i];
    }

    // Bulk access for serialization/bindings.
    [[nodiscard]] const std::vector<uint32_t>& data() const { return data_; }
    [[nodiscard]] const std::vector<uint32_t>& offsets() const { return offsets_; }
    [[nodiscard]] const std::vector<uint32_t>& active_k_history() const { return active_k_; }

  private:
    std::vector<uint32_t> data_;
    std::vector<uint32_t> offsets_;   // size = num_entries + 1 (CSR format)
    std::vector<uint32_t> active_k_;  // size = num_entries
};

}  // namespace ucc
