cmake_minimum_required(VERSION 3.20)
project(clifft_gpu_microbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive, native CPU optimization to match clifft's release flags.
# (-march=native -> NEON on ARM / Apple Silicon / Grace, AVX on x86.)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
else()
  check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
  if(HAS_MCPU_NATIVE)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mcpu=native")
  endif()
endif()

find_package(OpenMP QUIET)

# ---- CPU benchmark (always built; runs on the local Mac) -------------------
add_executable(bench_cpu src/bench_cpu.cpp src/bench_common.cpp)
target_include_directories(bench_cpu PRIVATE include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bench_cpu PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found: CPU baseline will be multithreaded")
else()
  message(STATUS "OpenMP NOT found: CPU baseline will be single-threaded "
                 "(brew install libomp on macOS for a fair multicore baseline)")
endif()

# ---- GPU benchmark (built only if a CUDA compiler is present) --------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  # GH200 = Hopper -> sm_90. Override with -DCUDA_ARCH=80 etc. if needed.
  if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH 90)
  endif()
  add_executable(bench_gpu src/bench_gpu.cu src/bench_common.cpp)
  target_include_directories(bench_gpu PRIVATE include)
  set_target_properties(bench_gpu PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
  target_compile_options(bench_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
  message(STATUS "CUDA found: building bench_gpu for sm_${CUDA_ARCH}")
else()
  message(STATUS "CUDA NOT found: skipping bench_gpu (build this on the GH200)")
endif()

# ---- HIP benchmark (built only if a HIP compiler is present) ----------------
# Same source as bench_gpu, compiled as HIP through include/gpu_runtime.hpp
# (a CUDA<->HIP runtime-API name shim). CMake's HIP language support needs
# >= 3.21; ROCm installs ship a newer CMake.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
  check_language(HIP)
endif()
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  # MI300X = gfx942. Override with -DHIP_ARCH=gfx90a etc. if needed.
  if(NOT DEFINED HIP_ARCH)
    set(HIP_ARCH gfx942)
  endif()
  add_executable(bench_gpu_hip src/bench_gpu_hip.hip src/bench_common.cpp)
  target_include_directories(bench_gpu_hip PRIVATE include)
  set_target_properties(bench_gpu_hip PROPERTIES HIP_ARCHITECTURES "${HIP_ARCH}")
  # Mirror the CUDA target's fast-math setting so the two GPU builds are
  # comparable; validation is tolerance-based (1e-9), not bit-exact.
  target_compile_options(bench_gpu_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math>)
  message(STATUS "HIP found: building bench_gpu_hip for ${HIP_ARCH}")
else()
  message(STATUS "HIP NOT found: skipping bench_gpu_hip (build this on the MI300X)")
endif()
