// HIP build of the GPU microbenchmark: exactly the CUDA source, with the
// runtime-API names mapped by include/gpu_runtime.hpp. The .hip extension
// routes this translation unit through CMake's HIP language (hipcc/clang).
#include "bench_gpu.cu"
