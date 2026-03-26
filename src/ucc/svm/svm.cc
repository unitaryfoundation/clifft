#include "ucc/svm/svm.h"

#include "ucc/svm/svm_internal.h"
#include "ucc/svm/svm_math.h"
#include "ucc/util/constants.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <numeric>
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
namespace avx512 {
void execute_internal(const CompiledModule& program, SchrodingerState& state);
}  // namespace avx512
#endif

// =============================================================================
// CPUID Runtime Dispatcher
// =============================================================================

using DispatchFn = void (*)(const CompiledModule&, SchrodingerState&);

#if defined(UCC_ENABLE_RUNTIME_DISPATCH)

static DispatchFn resolve_dispatcher() {
    // Allow environment override for testing.
    if (const char* env = std::getenv("UCC_FORCE_ISA")) {
        if (env[0] == '5' || (env[0] == 'a' && env[3] == '5')) {
            // "avx512" or "512"
            return avx512::execute_internal;
        }
        if (env[0] == 'a' || env[0] == 'A') {
            return avx2::execute_internal;
        }
        return scalar::execute_internal;
    }

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")) {
        return avx512::execute_internal;
    }
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

// Resolved once on first use; shared by execute() and svm_backend().
#if defined(UCC_ENABLE_RUNTIME_DISPATCH)
static DispatchFn resolved_fn = resolve_dispatcher();
#endif

void execute(const CompiledModule& program, SchrodingerState& state) {
#if defined(UCC_ENABLE_RUNTIME_DISPATCH)
    resolved_fn(program, state);
#else
    scalar::execute_internal(program, state);
#endif
}

const char* svm_backend() {
#if defined(UCC_ENABLE_RUNTIME_DISPATCH)
    if (resolved_fn == avx512::execute_internal)
        return "avx512";
    if (resolved_fn == avx2::execute_internal)
        return "avx2";
#endif
    return "scalar";
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
    uint32_t num_ev = program.num_exp_vals;

    result.measurements.resize(static_cast<size_t>(shots) * num_vis);
    result.detectors.resize(static_cast<size_t>(shots) * num_det);
    result.observables.resize(static_cast<size_t>(shots) * num_obs);
    result.exp_vals.resize(static_cast<size_t>(shots) * num_ev);

    SchrodingerState state({.peak_rank = program.peak_rank,
                            .num_measurements = num_total,
                            .num_detectors = num_det,
                            .num_observables = num_obs,
                            .num_exp_vals = num_ev,
                            .seed = seed});

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

        // Copy expectation values
        std::copy(
            state.exp_vals.begin(), state.exp_vals.end(),
            result.exp_vals.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_ev));
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

    uint32_t num_vis = program.num_measurements;
    uint32_t num_total = program.total_meas_slots;
    uint32_t num_det = program.num_detectors;
    uint32_t num_obs = program.num_observables;
    uint32_t num_ev = program.num_exp_vals;

    result.observable_ones.resize(num_obs, 0);

    if (keep_records) {
        result.measurements.reserve(static_cast<size_t>(shots) * num_vis);
        result.detectors.reserve(static_cast<size_t>(shots) * num_det);
        result.observables.reserve(static_cast<size_t>(shots) * num_obs);
        result.exp_vals.reserve(static_cast<size_t>(shots) * num_ev);
    }

    SchrodingerState state({.peak_rank = program.peak_rank,
                            .num_measurements = num_total,
                            .num_detectors = num_det,
                            .num_observables = num_obs,
                            .num_exp_vals = num_ev,
                            .seed = seed});

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
            result.measurements.insert(result.measurements.end(), state.meas_record.begin(),
                                       state.meas_record.begin() + num_vis);
            result.detectors.insert(result.detectors.end(), state.det_record.begin(),
                                    state.det_record.end());
            result.exp_vals.insert(result.exp_vals.end(), state.exp_vals.begin(),
                                   state.exp_vals.end());
        }
    }

    return result;
}

// =============================================================================
// Importance Sampling: Forced k-Fault Sampling
// =============================================================================
//
// Conditions each shot on exactly k fault events (quantum noise + readout)
// firing. Sites are drawn from the exact conditional Poisson-Binomial
// distribution via a DP table sweep. When all probabilities are equal,
// a persistent Fisher-Yates pool gives O(k) per shot.

std::vector<double> noise_site_probabilities(const CompiledModule& program) {
    const auto& pool = program.constant_pool;
    std::vector<double> probs;
    probs.reserve(pool.noise_sites.size() + pool.readout_noise.size());
    for (const auto& site : pool.noise_sites) {
        double p = 0.0;
        for (const auto& ch : site.channels)
            p += ch.prob;
        probs.push_back(p);
    }
    for (const auto& entry : pool.readout_noise) {
        probs.push_back(entry.prob);
    }
    return probs;
}

namespace {

// Build odds-ratio vector w[i] = p_i / (1 - p_i), clamping p to [0, 1-eps].
// After computing raw odds ratios, rescale so the mean weight is 1.0.
// This prevents overflow (p ~ 1 => huge w) and underflow (p ~ 0, large k)
// without affecting sampling correctness -- the constant factor cancels in
// the conditional inclusion probability w_i * DP[i+1][j-1] / DP[i][j].
std::vector<double> build_odds_ratios(const std::vector<double>& probs) {
    std::vector<double> w(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        double p = std::clamp(probs[i], 0.0, 1.0 - 1e-15);
        w[i] = p / (1.0 - p);
    }
    // Normalize to mean 1.0 to keep DP table values in a stable range.
    double sum_w = std::accumulate(w.begin(), w.end(), 0.0);
    if (sum_w > 0.0) {
        double scale = static_cast<double>(w.size()) / sum_w;
        for (double& weight : w)
            weight *= scale;
    }
    return w;
}

// Check if all probabilities are exactly equal. Exact equality is safe here
// because circuit noise probabilities come from floating-point literals that
// round-trip identically. A tolerance-based check would misfire at extreme
// noise scales (e.g. p ~ 1e-10 with heterogeneous noise).
bool all_probs_equal(const std::vector<double>& probs) {
    if (probs.empty())
        return true;
    double p0 = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] != p0)
            return false;
    }
    return true;
}

// Build flat DP table: dp[i * stride + j] = sum of products of odds ratios
// over all size-j subsets drawn from suffix [i, N).
// Returns the flat vector; stride = k + 1.
std::vector<double> build_dp_table(const std::vector<double>& w, uint32_t k) {
    uint32_t n = static_cast<uint32_t>(w.size());
    uint32_t stride = k + 1;
    std::vector<double> dp(static_cast<size_t>(n + 1) * stride, 0.0);

    // Base case: empty subset
    for (uint32_t i = 0; i <= n; ++i)
        dp[static_cast<size_t>(i) * stride + 0] = 1.0;

    // Fill bottom-up
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        uint32_t remaining = n - static_cast<uint32_t>(i);
        uint32_t max_j = std::min(remaining, k);
        for (uint32_t j = 1; j <= max_j; ++j) {
            dp[static_cast<size_t>(i) * stride + j] =
                dp[static_cast<size_t>(i + 1) * stride + j] +
                w[static_cast<size_t>(i)] * dp[static_cast<size_t>(i + 1) * stride + (j - 1)];
        }
    }
    return dp;
}

// Sample exactly k indices from [0, N) using the DP table.
// Appends to noise_indices and readout_indices (pre-cleared by caller).
void dp_sample_indices(SchrodingerState& state, const std::vector<double>& w,
                       const std::vector<double>& dp, uint32_t k, uint32_t n_q,
                       std::vector<uint32_t>& noise_indices,
                       std::vector<uint32_t>& readout_indices) {
    uint32_t n = static_cast<uint32_t>(w.size());
    uint32_t stride = k + 1;
    uint32_t needed = k;

    for (uint32_t i = 0; i < n && needed > 0; ++i) {
        double prob_include;
        uint32_t remaining = n - i;
        if (needed == remaining) {
            prob_include = 1.0;  // Must include all remaining
        } else {
            double denom = dp[static_cast<size_t>(i) * stride + needed];
            if (denom > 0.0) {
                prob_include =
                    (w[i] * dp[static_cast<size_t>(i + 1) * stride + (needed - 1)]) / denom;
            } else {
                prob_include = 0.0;
            }
        }
        if (state.random_double() < prob_include) {
            if (i < n_q)
                noise_indices.push_back(i);
            else
                readout_indices.push_back(i - n_q);
            needed--;
        }
    }
}

// Uniform sampling: partial Fisher-Yates on persistent pool.
void uniform_sample_indices(SchrodingerState& state, std::vector<uint32_t>& pool, uint32_t k,
                            uint32_t n_q, std::vector<uint32_t>& noise_indices,
                            std::vector<uint32_t>& readout_indices) {
    uint32_t n = static_cast<uint32_t>(pool.size());
    for (uint32_t j = 0; j < k; ++j) {
        uint32_t remaining = n - j;
        uint32_t pick = j + static_cast<uint32_t>(state.random_double() * remaining);
        std::swap(pool[j], pool[pick]);
    }

    // Sort the first k elements in-place, then partition into noise/readout.
    // Sorting the selected prefix is safe: the pool is a permutation, and any
    // permutation is valid input for the next Fisher-Yates run. We sort
    // in-place rather than copying to a temporary to avoid a heap allocation
    // per shot (the whole point of the persistent pool is O(k) amortized).
    std::sort(pool.begin(), pool.begin() + k);
    for (uint32_t j = 0; j < k; ++j) {
        uint32_t idx = pool[j];
        if (idx < n_q)
            noise_indices.push_back(idx);
        else
            readout_indices.push_back(idx - n_q);
    }
}

// Prepare forced faults for one shot: fills state.forced_faults with
// the sampled indices and sets next_noise_idx to the first forced site.
void prepare_forced_shot(SchrodingerState& state, const std::vector<double>& w,
                         const std::vector<double>& dp, uint32_t k, uint32_t n_q, bool uniform_mode,
                         std::vector<uint32_t>& uniform_pool) {
    auto& ff = state.forced_faults;
    ff.noise_indices.clear();
    ff.readout_indices.clear();
    ff.noise_pos = 0;
    ff.readout_pos = 0;

    if (uniform_mode) {
        uniform_sample_indices(state, uniform_pool, k, n_q, ff.noise_indices, ff.readout_indices);
    } else {
        dp_sample_indices(state, w, dp, k, n_q, ff.noise_indices, ff.readout_indices);
    }

    // Set next_noise_idx to the first forced noise site (or sentinel).
    ff.active = true;
    if (!ff.noise_indices.empty()) {
        state.next_noise_idx = ff.noise_indices[0];
        ff.noise_pos = 1;
    } else {
        state.next_noise_idx = static_cast<uint32_t>(-1);
    }
}

// Check that the k-fault stratum has nonzero probability mass.
// Sites with p==0 can never fire; sites with p==1 always fire.
// Feasible range: n_certain <= k <= n_total - n_impossible.
void validate_stratum(const std::vector<double>& probs, uint32_t k) {
    uint32_t n_total = static_cast<uint32_t>(probs.size());
    if (k > n_total) {
        throw std::invalid_argument("k (" + std::to_string(k) + ") exceeds total fault sites (" +
                                    std::to_string(n_total) + ")");
    }
    uint32_t n_certain = 0;     // Sites that always fire (p >= 1.0)
    uint32_t n_impossible = 0;  // Sites that never fire (p <= 0.0)
    for (double p : probs) {
        if (p <= 0.0)
            n_impossible++;
        else if (p >= 1.0)
            n_certain++;
    }
    if (k < n_certain || k > n_total - n_impossible) {
        throw std::invalid_argument("k-fault stratum k=" + std::to_string(k) +
                                    " has zero probability mass (" + std::to_string(n_certain) +
                                    " sites have p=1, " + std::to_string(n_impossible) +
                                    " sites have p=0)");
    }
}

}  // namespace

SampleResult sample_k(const CompiledModule& program, uint32_t shots, uint32_t k,
                      std::optional<uint64_t> seed) {
    SampleResult result;
    if (shots == 0)
        return result;

    uint32_t num_vis = program.num_measurements;
    uint32_t num_total = program.total_meas_slots;
    uint32_t num_det = program.num_detectors;
    uint32_t num_obs = program.num_observables;
    uint32_t num_ev = program.num_exp_vals;

    result.measurements.resize(static_cast<size_t>(shots) * num_vis);
    result.detectors.resize(static_cast<size_t>(shots) * num_det);
    result.observables.resize(static_cast<size_t>(shots) * num_obs);
    result.exp_vals.resize(static_cast<size_t>(shots) * num_ev);

    // Build fault site probabilities and precompute DP table.
    auto probs = noise_site_probabilities(program);
    validate_stratum(probs, k);
    uint32_t n_total = static_cast<uint32_t>(probs.size());
    uint32_t n_q = static_cast<uint32_t>(program.constant_pool.noise_sites.size());

    bool uniform_mode = all_probs_equal(probs);
    auto w = uniform_mode ? std::vector<double>{} : build_odds_ratios(probs);
    auto dp = uniform_mode ? std::vector<double>{} : build_dp_table(w, k);
    std::vector<uint32_t> uniform_pool;
    if (uniform_mode) {
        uniform_pool.resize(n_total);
        std::iota(uniform_pool.begin(), uniform_pool.end(), 0);
    }

    SchrodingerState state({.peak_rank = program.peak_rank,
                            .num_measurements = num_total,
                            .num_detectors = num_det,
                            .num_observables = num_obs,
                            .num_exp_vals = num_ev,
                            .seed = seed});

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0)
            state.reset();

        prepare_forced_shot(state, w, dp, k, n_q, uniform_mode, uniform_pool);
        execute(program, state);

        std::copy(state.meas_record.begin(), state.meas_record.begin() + num_vis,
                  result.measurements.begin() +
                      static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_vis));
        std::copy(
            state.det_record.begin(), state.det_record.end(),
            result.detectors.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_det));

        auto obs_out = result.observables.begin() +
                       static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_obs);
        for (uint32_t i = 0; i < num_obs; ++i) {
            uint8_t val = state.obs_record[i];
            if (i < program.expected_observables.size() && program.expected_observables[i] != 0) {
                val ^= 1;
            }
            obs_out[static_cast<ptrdiff_t>(i)] = val;
        }

        std::copy(
            state.exp_vals.begin(), state.exp_vals.end(),
            result.exp_vals.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_ev));
    }

    return result;
}

SurvivorResult sample_k_survivors(const CompiledModule& program, uint32_t shots, uint32_t k,
                                  std::optional<uint64_t> seed, bool keep_records) {
    SurvivorResult result;
    result.total_shots = shots;
    if (shots == 0)
        return result;

    uint32_t num_vis = program.num_measurements;
    uint32_t num_total = program.total_meas_slots;
    uint32_t num_det = program.num_detectors;
    uint32_t num_obs = program.num_observables;
    uint32_t num_ev = program.num_exp_vals;

    result.observable_ones.resize(num_obs, 0);

    if (keep_records) {
        result.measurements.reserve(static_cast<size_t>(shots) * num_vis);
        result.detectors.reserve(static_cast<size_t>(shots) * num_det);
        result.observables.reserve(static_cast<size_t>(shots) * num_obs);
        result.exp_vals.reserve(static_cast<size_t>(shots) * num_ev);
    }

    auto probs = noise_site_probabilities(program);
    validate_stratum(probs, k);
    uint32_t n_total = static_cast<uint32_t>(probs.size());
    uint32_t n_q = static_cast<uint32_t>(program.constant_pool.noise_sites.size());

    bool uniform_mode = all_probs_equal(probs);
    auto w = uniform_mode ? std::vector<double>{} : build_odds_ratios(probs);
    auto dp = uniform_mode ? std::vector<double>{} : build_dp_table(w, k);
    std::vector<uint32_t> uniform_pool;
    if (uniform_mode) {
        uniform_pool.resize(n_total);
        std::iota(uniform_pool.begin(), uniform_pool.end(), 0);
    }

    SchrodingerState state({.peak_rank = program.peak_rank,
                            .num_measurements = num_total,
                            .num_detectors = num_det,
                            .num_observables = num_obs,
                            .num_exp_vals = num_ev,
                            .seed = seed});

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0)
            state.reset();

        prepare_forced_shot(state, w, dp, k, n_q, uniform_mode, uniform_pool);
        execute(program, state);

        if (state.discarded)
            continue;

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
        if (any_obs_flipped)
            result.logical_errors++;

        if (keep_records) {
            result.measurements.insert(result.measurements.end(), state.meas_record.begin(),
                                       state.meas_record.begin() + num_vis);
            result.detectors.insert(result.detectors.end(), state.det_record.begin(),
                                    state.det_record.end());
            result.exp_vals.insert(result.exp_vals.end(), state.exp_vals.begin(),
                                   state.exp_vals.end());
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
