#pragma once

// Minimal CUDA runtime stub for clang -fsyntax-only checks on machines
// without a CUDA toolkit. Never used by a real nvcc/clang-cuda build: it is
// only reachable through the extra include path the syntax-check script adds.
// Declarations cover exactly what src/clifft/sampling/cuda uses.

#include <cstddef>

#if defined(__clang__) && defined(__CUDA__)
// Clang's CUDA mode defines the execution-space attributes in its wrapper
// headers, which -nocudainc skips.
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline
#endif

struct dim3 {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
    constexpr dim3() = default;
    constexpr dim3(unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

struct __stub_dim {
    unsigned int x, y, z;
};
#if defined(__clang__) && defined(__CUDA__)
extern __device__ const __stub_dim threadIdx;
extern __device__ const __stub_dim blockIdx;
extern __device__ const __stub_dim blockDim;
extern __device__ const __stub_dim gridDim;
#else
extern const __stub_dim threadIdx;
extern const __stub_dim blockIdx;
extern const __stub_dim blockDim;
extern const __stub_dim gridDim;
#endif

extern "C" {
__device__ void __syncthreads();
__device__ int __popcll(unsigned long long value);
__device__ double log(double value);
__device__ double sqrt(double value);
}

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorStub = 1,
};

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
};

enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
};

struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    int multiProcessorCount;
    size_t sharedMemPerBlockOptin;
};

cudaError_t cudaMalloc(void** pointer, size_t bytes);
cudaError_t cudaFree(void* pointer);
cudaError_t cudaMemcpy(void* destination, const void* source, size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaGetLastError();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device);
cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes);
const char* cudaGetErrorString(cudaError_t error);

template <typename T>
cudaError_t cudaFuncSetAttribute(T* entry, cudaFuncAttribute attribute, int value);

// Kernel-launch configuration hooks clang emits for <<<...>>>.
extern "C" unsigned int __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared_bytes = 0,
                                                    void* stream = nullptr);
extern "C" cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t shared_bytes = 0,
                                         void* stream = nullptr);
