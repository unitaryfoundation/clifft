#include "clifft/sampling/instrument_activation_dispatch.h"

#include "clifft/sampling/instrument_activation_simd.h"
#include "clifft/sampling/simd_width.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>

namespace clifft::sampling {

namespace {

// One four-coefficient block was neutral in the direct microbenchmark, while
// every wider measured width won; smaller states stay on the scalar path.
constexpr uint32_t kMinAvx2ActiveWidth = 2;
static_assert(uint64_t{1} << kMinAvx2ActiveWidth == kAvx2DoubleLanes);

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)

const internal::RuntimeIsa kResolvedInstrumentActivationIsa = internal::runtime_isa();

#endif

}  // namespace

NewXInstrumentKernel resolve_new_x_instrument_kernel(uint32_t active_width,
                                                     internal::RuntimeIsa runtime_isa) noexcept {
    // The AVX-512 tier includes every feature required by the AVX2 kernel, and
    // reusing it avoids falling back to baseline scalar code in portable wheels.
    if ((runtime_isa == internal::RuntimeIsa::Avx2 ||
         runtime_isa == internal::RuntimeIsa::Avx512) &&
        active_width >= kMinAvx2ActiveWidth) {
        return NewXInstrumentKernel::Avx2;
    }
    return NewXInstrumentKernel::Scalar;
}

void apply_new_x_instrument_no_fire_dispatched(State& state, double factor_zero, double factor_one,
                                               double no_fire_probability,
                                               NewXInstrumentKernel kernel) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    assert(kernel == resolve_new_x_instrument_kernel(state.active_width(),
                                                     kResolvedInstrumentActivationIsa) &&
           "new-X instrument kernel must match the process ISA");
    if (kernel == NewXInstrumentKernel::Avx2) {
        if (kResolvedInstrumentActivationIsa == internal::RuntimeIsa::Avx2 ||
            kResolvedInstrumentActivationIsa == internal::RuntimeIsa::Avx512) {
            apply_new_x_instrument_no_fire_avx2(state, factor_zero, factor_one,
                                                no_fire_probability);
            return;
        }
        assert(false && "AVX2 new-X instrument kernel requires an AVX2-capable process ISA");
    }
#else
    assert(kernel == NewXInstrumentKernel::Scalar &&
           "portable new-X instrument dispatch requires the scalar kernel");
    static_cast<void>(kernel);
#endif
    apply_new_x_instrument_no_fire(state, factor_zero, factor_one, no_fire_probability);
}

}  // namespace clifft::sampling
