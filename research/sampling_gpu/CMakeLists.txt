# Research benchmark for the CUDA sampling backend. Built only when the root
# configure sets CLIFFT_BUILD_GPU_BENCH=ON; the GPU path additionally needs
# CLIFFT_ENABLE_CUDA=ON and a CUDA toolchain.

add_executable(bench_sampling bench_main.cc)
target_link_libraries(bench_sampling PRIVATE clifft_cuda_plan)
target_compile_features(bench_sampling PRIVATE cxx_std_20)

if(CLIFFT_ENABLE_CUDA)
    target_link_libraries(bench_sampling PRIVATE clifft_cuda_backend)
    target_compile_definitions(bench_sampling PRIVATE CLIFFT_BENCH_HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(bench_sampling PRIVATE CUDA::cudart)
endif()
