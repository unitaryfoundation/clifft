# UCC Core Library
# This library contains the compiler pipeline and VM runtime.

add_library(ucc_core STATIC
    stub.cc  # Placeholder until real sources are added
)

target_include_directories(ucc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(ucc_core PUBLIC
    libstim
)

target_compile_features(ucc_core PUBLIC cxx_std_20)
