#include "clifft/util/page_allocation.h"

#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#include <malloc.h>
#endif

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace clifft {

namespace {

size_t round_up(size_t value, size_t alignment) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::length_error("page-aligned allocation size overflow");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

void* aligned_alloc_portable(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void aligned_free_portable(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

}  // namespace

PageAlignedAllocation::PageAlignedAllocation(size_t requested_bytes) {
    if (requested_bytes == 0) {
        return;
    }
    const size_t page_aligned = round_up(requested_bytes, kBaseAlignment);

#if defined(__linux__)
    if (page_aligned >= kHugePageSize) {
        const size_t huge_aligned = round_up(page_aligned, kHugePageSize);
        void* mapped = mmap(nullptr, huge_aligned, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (mapped != MAP_FAILED) {
            data_ = mapped;
            allocated_bytes_ = huge_aligned;
            memory_mapped_ = true;
            return;
        }
    }

    const size_t allocation_alignment =
        page_aligned >= kHugePageSize ? kHugePageSize : kBaseAlignment;
    const size_t allocation_bytes = round_up(page_aligned, allocation_alignment);
    data_ = aligned_alloc_portable(allocation_alignment, allocation_bytes);
    if (data_ == nullptr) {
        throw std::bad_alloc();
    }
    allocated_bytes_ = allocation_bytes;
    if (page_aligned >= kHugePageSize) {
        madvise(data_, allocation_bytes, MADV_HUGEPAGE);
    }
#else
    data_ = aligned_alloc_portable(kBaseAlignment, page_aligned);
    if (data_ == nullptr) {
        throw std::bad_alloc();
    }
    allocated_bytes_ = page_aligned;
#endif
}

PageAlignedAllocation::~PageAlignedAllocation() {
    reset();
}

PageAlignedAllocation::PageAlignedAllocation(PageAlignedAllocation&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      allocated_bytes_(std::exchange(other.allocated_bytes_, 0)),
      memory_mapped_(std::exchange(other.memory_mapped_, false)) {}

PageAlignedAllocation& PageAlignedAllocation::operator=(PageAlignedAllocation&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        allocated_bytes_ = std::exchange(other.allocated_bytes_, 0);
        memory_mapped_ = std::exchange(other.memory_mapped_, false);
    }
    return *this;
}

void PageAlignedAllocation::reset() noexcept {
    if (data_ == nullptr) {
        return;
    }
#if defined(__linux__)
    if (memory_mapped_) {
        munmap(data_, allocated_bytes_);
    } else {
        aligned_free_portable(data_);
    }
#else
    aligned_free_portable(data_);
#endif
    data_ = nullptr;
    allocated_bytes_ = 0;
    memory_mapped_ = false;
}

}  // namespace clifft
