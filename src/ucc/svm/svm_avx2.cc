// AVX2+BMI2+FMA translation unit for SVM kernels.
// Compiled with -mavx2 -mbmi2 -mfma on x86-64 (GCC/Clang).
// On other platforms this file is not included in the build.

#define UCC_SIMD_NAMESPACE avx2

#include "ucc/svm/svm.h"
#include "ucc/svm/svm_internal.h"
#include "ucc/svm/svm_math.h"
#include "ucc/util/constants.h"

#if defined(__AVX2__) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#include <immintrin.h>
#endif

#include "svm_kernels.inl"  // NOLINT(bugprone-suspicious-include)
