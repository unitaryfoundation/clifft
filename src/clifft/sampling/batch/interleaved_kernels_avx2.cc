// Interleaved batch kernels compiled with AVX2, BMI2, and FMA enabled, so
// portable builds can select them at runtime.

#define CLIFFT_INTERLEAVED_KERNELS_NAMESPACE interleaved_avx2
#include "clifft/sampling/batch/interleaved_kernels_impl.h"
