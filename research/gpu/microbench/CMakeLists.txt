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
