// Scalar (non-SIMD) translation unit for SVM kernels.
// Compiles svm_kernels.inl without any ISA-specific compiler flags.

#define CLIFFT_SIMD_NAMESPACE scalar

#include "clifft/svm/svm.h"
#include "clifft/svm/svm_internal.h"
#include "clifft/svm/svm_math.h"
#include "clifft/util/constants.h"

#include "svm_kernels.inl"  // NOLINT(bugprone-suspicious-include)
