#pragma once

// Vendor shim: lets src/bench_gpu.cu compile unmodified as CUDA (nvcc) or
// HIP (hipcc). Under HIP, the CUDA runtime-API names the benchmark uses are
// mapped to their 1:1 HIP equivalents. Everything kernel-side (__global__,
// __shared__, <<<...>>> launches, __syncthreads) is already source-compatible,
// and the kernels use no warp-level intrinsics, so AMD's 64-wide wavefront
// requires no code changes.

#if defined(__HIP__) || defined(__HIPCC__)

#include <hip/hip_runtime.h>

#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaSuccess hipSuccess

#else  // CUDA: nvcc auto-includes the runtime; keep it explicit anyway.

#include <cuda_runtime.h>

#endif
