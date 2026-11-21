// Interleaved batch kernels compiled with AVX2, BMI2, FMA, AVX-512F, and AVX-512DQ enabled, so
// portable builds can select them at runtime.

#define CLIFFT_INTERLEAVED_KERNELS_NAMESPACE interleaved_avx512
#include "clifft/sampling/batch/interleaved_kernels_impl.h"
