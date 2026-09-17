cmake_minimum_required(VERSION 3.24)
project(clifft_proxy_study LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG 42e0b9e099180e8570407c33f87b4683cac00d81
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(stim)

if(NOT PROXY_SOURCE)
    message(FATAL_ERROR "Set PROXY_SOURCE to the exported research translation unit")
endif()
add_executable(profile_proxy "${PROXY_SOURCE}")
target_link_libraries(profile_proxy PRIVATE libstim)
target_compile_features(profile_proxy PRIVATE cxx_std_20)
target_compile_options(profile_proxy PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
