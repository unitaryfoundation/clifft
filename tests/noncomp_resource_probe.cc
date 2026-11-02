#include "noncomp_resource_probe.h"

#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#endif

void clifft::detail::replay_hot_path(bool enter) noexcept {
    auto& counters = test::hot_path_counters;
    if (enter) {
        ++counters.depth;
        ++counters.intervals;
    } else {
        --counters.depth;
    }
}

void clifft::detail::replay_allocation() noexcept {
    if (test::hot_path_counters.depth)
        ++test::hot_path_counters.allocations;
}

void clifft::detail::replay_online_boundary(ReplayOnlineBoundary boundary,
                                            const noncomp_detail::OnlineState& state) {
    if (test::online_boundary_probe)
        test::online_boundary_probe(boundary, state);
}

// Only the diagnostic executable replaces allocation. Coefficient storage uses
// the separate PageAlignedAllocation seam, including its huge-page mmap route.
void* operator new(std::size_t size) {
    clifft::detail::replay_allocation();
    if (void* pointer = std::malloc(size ? size : 1))
        return pointer;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    return ::operator new(size);
}
void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
    ::operator delete(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    clifft::detail::replay_allocation();
    const auto align = static_cast<std::size_t>(alignment);
    void* pointer = nullptr;
#ifdef _WIN32
    pointer = _aligned_malloc(size ? size : 1, align);
#else
    if (posix_memalign(&pointer, align, size ? size : 1) != 0)
        pointer = nullptr;
#endif
    if (pointer)
        return pointer;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
#ifdef _WIN32
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}
void operator delete[](void* pointer, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}
void operator delete(void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}
