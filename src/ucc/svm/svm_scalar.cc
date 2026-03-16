// Scalar (non-SIMD) translation unit for SVM kernels.
// Compiles svm_kernels.inl without any ISA-specific compiler flags.

#define UCC_SIMD_NAMESPACE scalar

#include "ucc/svm/svm.h"
#include "ucc/svm/svm_internal.h"
#include "ucc/svm/svm_math.h"
#include "ucc/util/constants.h"

#include "svm_kernels.inl"  // NOLINT(bugprone-suspicious-include)
