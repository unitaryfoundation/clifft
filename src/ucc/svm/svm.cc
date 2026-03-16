#include "ucc/svm/svm.h"

#include "ucc/svm/svm_internal.h"
#include "ucc/svm/svm_math.h"
#include "ucc/util/constants.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <stdexcept>

namespace ucc {

// =============================================================================
// Forward declarations for per-ISA execute_internal() implementations
// =============================================================================

namespace scalar {
void execute_internal(const CompiledModule& program, SchrodingerState& state);
}  // namespace scalar

#if defined(UCC_ENABLE_RUNTIME_DISPATCH)
namespace avx2 {
void execute_internal(const CompiledModule& program, SchrodingerState& state);
}  // namespace avx2
#endif

// =============================================================================
// CPUID Runtime Dispatcher
// =============================================================================

using DispatchFn = void (*)(const CompiledModule&, SchrodingerState&);

#if defined(UCC_ENABLE_RUNTIME_DISPATCH)

static DispatchFn resolve_dispatcher() {
    // Allow environment override for testing.
    if (const char* env = std::getenv("UCC_FORCE_ISA")) {
        if (env[0] == 'a' || env[0] == 'A') {
            return avx2::execute_internal;
        }
        return scalar::execute_internal;
    }

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2")) {
        return avx2::execute_internal;
    }
#endif
    return scalar::execute_internal;
}

#endif  // UCC_ENABLE_RUNTIME_DISPATCH

// =============================================================================
// Public execute() wrapper
// =============================================================================

void execute(const CompiledModule& program, SchrodingerState& state) {
#if defined(UCC_ENABLE_RUNTIME_DISPATCH)
    static DispatchFn fn = resolve_dispatcher();
    fn(program, state);
#else
    scalar::execute_internal(program, state);
#endif
}

// =============================================================================
// Multi-Shot Sampling
// =============================================================================

SampleResult sample(const CompiledModule& program, uint32_t shots, std::optional<uint64_t> seed) {
    SampleResult result;
    if (shots == 0) {
        return result;
    }

    uint32_t num_vis = program.num_measurements;    // Visible measurements for output
    uint32_t num_total = program.total_meas_slots;  // Total slots for VM execution
    uint32_t num_det = program.num_detectors;
    uint32_t num_obs = program.num_observables;

    result.measurements.resize(static_cast<size_t>(shots) * num_vis);
    result.detectors.resize(static_cast<size_t>(shots) * num_det);
    result.observables.resize(static_cast<size_t>(shots) * num_obs);

    SchrodingerState state(program.peak_rank, num_total, num_det, num_obs, seed);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0) {
            state.reset();
        }

        state.next_noise_idx = 0;
        state.draw_next_noise(program.constant_pool.noise_hazards);

        execute(program, state);

        // Copy only visible measurements (first num_vis entries)
        std::copy(state.meas_record.begin(), state.meas_record.begin() + num_vis,
                  result.measurements.begin() +
                      static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_vis));
        std::copy(
            state.det_record.begin(), state.det_record.end(),
            result.detectors.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_det));

        // Normalize observables against noiseless reference before output
        auto obs_out = result.observables.begin() +
                       static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_obs);
        for (uint32_t i = 0; i < num_obs; ++i) {
            uint8_t val = state.obs_record[i];
            if (i < program.expected_observables.size() && program.expected_observables[i] != 0) {
                val ^= 1;
            }
            obs_out[static_cast<ptrdiff_t>(i)] = val;
        }
    }

    return result;
}

// =============================================================================
// Survivor-Only Sampling
// =============================================================================
//
// Returns results only for shots that pass all OP_POSTSELECT checks.
// With keep_records=false, zero arrays are allocated -- only shot/discard
// counts and per-observable error counts are tracked. This is the fast path
// for Sinter integration where decoders are not needed.

SurvivorResult sample_survivors(const CompiledModule& program, uint32_t shots,
                                std::optional<uint64_t> seed, bool keep_records) {
    SurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }

    uint32_t num_total = program.total_meas_slots;
    uint32_t num_det = program.num_detectors;
    uint32_t num_obs = program.num_observables;

    result.observable_ones.resize(num_obs, 0);

    if (keep_records) {
        result.detectors.reserve(static_cast<size_t>(shots) * num_det);
        result.observables.reserve(static_cast<size_t>(shots) * num_obs);
    }

    SchrodingerState state(program.peak_rank, num_total, num_det, num_obs, seed);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0) {
            state.reset();
        }

        state.next_noise_idx = 0;
        state.draw_next_noise(program.constant_pool.noise_hazards);

        execute(program, state);

        if (state.discarded) {
            continue;
        }

        result.passed_shots++;

        bool any_obs_flipped = false;
        for (uint32_t i = 0; i < num_obs; ++i) {
            uint8_t val = state.obs_record[i];
            if (i < program.expected_observables.size() && program.expected_observables[i] != 0) {
                val ^= 1;
            }
            if (val) {
                result.observable_ones[i]++;
                any_obs_flipped = true;
            }
            if (keep_records) {
                result.observables.push_back(val);
            }
        }
        if (any_obs_flipped) {
            result.logical_errors++;
        }

        if (keep_records) {
            result.detectors.insert(result.detectors.end(), state.det_record.begin(),
                                    state.det_record.end());
        }
    }

    return result;
}

// =============================================================================
// Statevector Expansion
// =============================================================================
//
// Expands the factored state |psi> = gamma * U_C * P * (|phi>_A (x) |0>_D)
// into a dense 2^n statevector. Used as a validation oracle for small circuits.
//
// The active statevector |phi>_A lives on virtual axes 0..k-1 (contiguous
// lowest bits). Dormant qubits occupy axes k..n-1 and are strictly |0>.
// The Clifford frame U_C (final_tableau) rotates from the virtual basis
// to the physical laboratory basis.

std::vector<std::complex<double>> get_statevector(const CompiledModule& program,
                                                  const SchrodingerState& state) {
    uint32_t n = program.num_qubits;
    if (n > 10) {
        throw std::runtime_error(
            "Statevector expansion limited to 10 qubits (dense U_C matrix is 4^n)");
    }
    uint64_t dim = 1ULL << n;

    // Step 1: Embed |phi>_A into dense 2^n virtual-basis state.
    // Active axes are strictly 0..k-1, so the 2^k active amplitudes align
    // perfectly with the first 2^k entries of the dense array.
    assert(state.active_k <= n && "More active qubits than physical qubits");
    uint64_t active_size = state.v_size();
    assert(active_size <= dim && "Active array exceeds statevector dimension");
    std::vector<std::complex<double>> dense(dim, {0.0, 0.0});
    for (uint64_t i = 0; i < active_size; ++i) {
        dense[i] = state.v()[i];
    }

    // Step 2: Apply Pauli frame P = X^{p_x} Z^{p_z}.
    // P|i> = (-1)^popcount(i & p_z) * |i XOR p_x>
    // Extract p_x and p_z as uint64_t masks (portable: reconstruct from bits).
    uint64_t px_mask = 0;
    uint64_t pz_mask = 0;
    for (uint32_t q = 0; q < n; ++q) {
        if (bit_get(state.p_x, q)) {
            px_mask |= (uint64_t{1} << q);
        }
        if (bit_get(state.p_z, q)) {
            pz_mask |= (uint64_t{1} << q);
        }
    }

    std::vector<std::complex<double>> framed(dim, {0.0, 0.0});
    for (uint64_t i = 0; i < dim; ++i) {
        uint64_t target = i ^ px_mask;
        double sign = (std::popcount(i & pz_mask) % 2 == 1) ? -1.0 : 1.0;
        framed[target] += dense[i] * sign;
    }

    // Step 3: Apply U_C (final_tableau) via dense matrix multiplication.
    // to_flat_unitary_matrix(true) returns row-major complex<float> in
    // little-endian qubit order. If no tableau, treat as identity.
    std::vector<std::complex<double>> physical(dim, {0.0, 0.0});
    if (program.constant_pool.final_tableau.has_value()) {
        auto flat_uc = program.constant_pool.final_tableau->to_flat_unitary_matrix(true);
        // flat_uc is dim x dim row-major: flat_uc[row * dim + col]
        for (uint64_t row = 0; row < dim; ++row) {
            std::complex<double> sum{0.0, 0.0};
            for (uint64_t col = 0; col < dim; ++col) {
                auto u = flat_uc[row * dim + col];
                sum += std::complex<double>(u.real(), u.imag()) * framed[col];
            }
            physical[row] = sum;
        }
    } else {
        physical = framed;
    }

    // Step 4: Scale by gamma * global_weight.
    std::complex<double> scale = state.gamma() * program.constant_pool.global_weight;
    for (uint64_t i = 0; i < dim; ++i) {
        physical[i] *= scale;
    }

    return physical;
}

}  // namespace ucc
