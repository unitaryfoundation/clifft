// AVX-512F+AVX-512DQ translation unit for SVM kernels.
// Compiled with -mavx2 -mbmi2 -mfma -mavx512f -mavx512dq on x86-64 (GCC/Clang).
// On other platforms this file is not included in the build.
//
// The AVX-512 paths handle wide strides (min_axis >= 2) using 512-bit
// vectors. For narrower strides (min_axis < 2), execution falls through
// to the AVX2 paths which are also available in this TU.

#define CLIFFT_SIMD_NAMESPACE avx512

#include "clifft/svm/svm.h"
#include "clifft/svm/svm_internal.h"
#include "clifft/svm/svm_math.h"
#include "clifft/util/constants.h"

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__x86_64__) || defined(_M_X64) || \
    defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "svm_kernels.inl"  // NOLINT(bugprone-suspicious-include)
