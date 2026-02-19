# UCC Test Suite
# Uses Catch2 v3 for unit testing

include(CTest)
include(Catch)

add_executable(ucc_tests
    test_placeholder.cc
)

target_link_libraries(ucc_tests PRIVATE
    ucc_core
    Catch2::Catch2WithMain
)

# Auto-discover Catch2 tests for CTest integration
catch_discover_tests(ucc_tests)
