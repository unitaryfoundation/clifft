// Portable build of the interleaved batch kernels and the table lookup that
// matches a plan's executor backend to its instruction-set build.

#define CLIFFT_INTERLEAVED_KERNELS_NAMESPACE interleaved_portable
#include "clifft/sampling/batch/interleaved_kernels_impl.h"
#undef CLIFFT_INTERLEAVED_KERNELS_NAMESPACE

namespace clifft::sampling {

#if defined(CLIFFT_ENABLE_X86_RUNTIME_DISPATCH)
namespace interleaved_avx2 {
const InterleavedKernels& kernels() noexcept;
}
namespace interleaved_avx512 {
const InterleavedKernels& kernels() noexcept;
}
#endif

const InterleavedKernels& interleaved_kernels(ExecutorBackend backend) noexcept {
    switch (backend) {
#if defined(CLIFFT_ENABLE_X86_RUNTIME_DISPATCH)
        case ExecutorBackend::Avx2:
            return interleaved_avx2::kernels();
        case ExecutorBackend::Avx512:
            return interleaved_avx512::kernels();
#else
        case ExecutorBackend::Avx2:
        case ExecutorBackend::Avx512:
#endif
        case ExecutorBackend::Scalar:
        case ExecutorBackend::Neon:
            break;
    }
    return interleaved_portable::kernels();
}

}  // namespace clifft::sampling
