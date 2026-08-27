#pragma once

#include <cstddef>

namespace clifft {

// Owns page-aligned storage suitable for dense coefficient sweeps. Large
// Linux allocations first try explicit huge pages, then fall back to aligned
// storage with a transparent-huge-page hint.
class PageAlignedAllocation {
  public:
    static constexpr size_t kBaseAlignment = 4096;
    static constexpr size_t kHugePageSize = 2 * 1024 * 1024;

    enum class Alignment {
        BasePage,
        HugePageEligible,
    };

    PageAlignedAllocation() = default;
    explicit PageAlignedAllocation(size_t requested_bytes,
                                   Alignment alignment = Alignment::HugePageEligible);
    ~PageAlignedAllocation();

    PageAlignedAllocation(const PageAlignedAllocation&) = delete;
    PageAlignedAllocation& operator=(const PageAlignedAllocation&) = delete;
    PageAlignedAllocation(PageAlignedAllocation&& other) noexcept;
    PageAlignedAllocation& operator=(PageAlignedAllocation&& other) noexcept;

    // Exact retained size selected for a request, including platform alignment.
    [[nodiscard]] static size_t allocation_size(size_t requested_bytes,
                                                Alignment alignment = Alignment::HugePageEligible);

    void reset() noexcept;

    [[nodiscard]] void* data() { return data_; }
    [[nodiscard]] const void* data() const { return data_; }
    [[nodiscard]] size_t size() const { return allocated_bytes_; }
    [[nodiscard]] bool empty() const { return data_ == nullptr; }

    // Explicit anonymous mappings are zero-filled by the kernel. Portable
    // aligned-allocation fallbacks leave their contents uninitialized.
    [[nodiscard]] bool zero_initialized() const { return memory_mapped_; }

  private:
    void* data_ = nullptr;
    size_t allocated_bytes_ = 0;
    bool memory_mapped_ = false;
};

}  // namespace clifft
