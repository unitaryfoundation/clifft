#include "clifft/util/page_allocation.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

using clifft::PageAlignedAllocation;

TEST_CASE("Page-aligned allocation owns rounded movable storage") {
    PageAlignedAllocation allocation(17);
    REQUIRE_FALSE(allocation.empty());
    REQUIRE(allocation.size() >= 17);
    REQUIRE(allocation.size() % PageAlignedAllocation::kBaseAlignment == 0);
    REQUIRE(reinterpret_cast<uintptr_t>(allocation.data()) %
                PageAlignedAllocation::kBaseAlignment ==
            0);

    auto* bytes = static_cast<uint8_t*>(allocation.data());
    bytes[0] = 0x12;
    bytes[16] = 0x34;
    void* const original = allocation.data();

    PageAlignedAllocation moved(std::move(allocation));
    REQUIRE(allocation.empty());
    REQUIRE(moved.data() == original);
    REQUIRE(static_cast<uint8_t*>(moved.data())[0] == 0x12);
    REQUIRE(static_cast<uint8_t*>(moved.data())[16] == 0x34);

    PageAlignedAllocation assigned(1);
    assigned = std::move(moved);
    REQUIRE(moved.empty());
    REQUIRE(assigned.data() == original);
    assigned.reset();
    REQUIRE(assigned.empty());
    REQUIRE(assigned.size() == 0);
}

TEST_CASE("Page-aligned allocation validates edge sizes") {
    PageAlignedAllocation empty(0);
    REQUIRE(empty.empty());
    REQUIRE_THROWS_AS(PageAlignedAllocation(std::numeric_limits<size_t>::max()), std::length_error);
}

#if defined(__linux__)
TEST_CASE("Large page-aligned allocation is huge-page ready on Linux") {
    PageAlignedAllocation allocation(PageAlignedAllocation::kHugePageSize);
    REQUIRE(reinterpret_cast<uintptr_t>(allocation.data()) % PageAlignedAllocation::kHugePageSize ==
            0);
    REQUIRE(allocation.size() % PageAlignedAllocation::kHugePageSize == 0);
}
#endif
