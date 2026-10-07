// Standalone allocation probe for the single-threaded HIR optimizer. Link only
// into this executable: replacement new/delete must never enter the library.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
#include <string>

namespace {
struct Header {
    void* allocation;
    size_t size;
    bool measured;
};
bool measuring = false;
size_t allocations = 0;
size_t allocated_bytes = 0;
size_t live_bytes = 0;
size_t peak_bytes = 0;

void* allocate(size_t size, size_t alignment) {
    alignment = std::max(alignment, alignof(Header));
    if (size > std::numeric_limits<size_t>::max() - sizeof(Header) - alignment) {
        throw std::bad_alloc();
    }
    void* raw = std::malloc(size + sizeof(Header) + alignment);
    if (!raw) {
        throw std::bad_alloc();
    }
    const auto address =
        (reinterpret_cast<uintptr_t>(raw) + sizeof(Header) + alignment - 1) & ~(alignment - 1);
    auto* header = reinterpret_cast<Header*>(address) - 1;
    *header = {raw, size, measuring};
    if (measuring) {
        ++allocations;
        allocated_bytes += size;
        live_bytes += size;
        peak_bytes = std::max(peak_bytes, live_bytes);
    }
    return reinterpret_cast<void*>(address);
}

void release(void* pointer) noexcept {
    if (!pointer) {
        return;
    }
    auto* header = static_cast<Header*>(pointer) - 1;
    if (header->measured) {
        live_bytes -= header->size;
    }
    std::free(header->allocation);
}
}  // namespace

void* operator new(size_t size) {
    return allocate(size, alignof(std::max_align_t));
}
void* operator new[](size_t size) {
    return allocate(size, alignof(std::max_align_t));
}
void* operator new(size_t size, std::align_val_t alignment) {
    return allocate(size, static_cast<size_t>(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment) {
    return allocate(size, static_cast<size_t>(alignment));
}
void operator delete(void* p) noexcept {
    release(p);
}
void operator delete[](void* p) noexcept {
    release(p);
}
void operator delete(void* p, size_t) noexcept {
    release(p);
}
void operator delete[](void* p, size_t) noexcept {
    release(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    release(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    release(p);
}
void operator delete(void* p, size_t, std::align_val_t) noexcept {
    release(p);
}
void operator delete[](void* p, size_t, std::align_val_t) noexcept {
    release(p);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: profile_optimizer_allocations circuit.stim\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "Cannot open circuit\n";
        return 2;
    }
    const std::string source{std::istreambuf_iterator<char>(input), {}};
    auto hir = clifft::trace(clifft::parse(source));
    auto manager = clifft::default_hir_pass_manager();
    // Exclude parsing, tracing, pass construction, and preexisting HIR storage.
    // Include optimizer temporaries and any output allocations retained in HIR.
    measuring = true;
    manager.run(hir);
    measuring = false;
    std::cout << "{\"allocations\":" << allocations << ",\"allocated_bytes\":" << allocated_bytes
              << ",\"peak_live_bytes\":" << peak_bytes << ",\"retained_bytes\":" << live_bytes
              << "}\n";
}
