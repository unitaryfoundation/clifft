#include "clifft/svm/svm.h"

#include "clifft/svm/svm_internal.h"
#include "clifft/svm/svm_math.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace clifft {

// =============================================================================
// Forward declarations for per-ISA execute_internal() implementations
// =============================================================================

namespace scalar {
void execute_internal(const CompiledModule& program, SchrodingerState& state, size_t start_offset);
}  // namespace scalar

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
namespace avx2 {
void execute_internal(const CompiledModule& program, SchrodingerState& state, size_t start_offset);
}  // namespace avx2
namespace avx512 {
void execute_internal(const CompiledModule& program, SchrodingerState& state, size_t start_offset);
}  // namespace avx512
#endif

// =============================================================================
// CPUID Runtime Dispatcher
// =============================================================================

using DispatchFn = void (*)(const CompiledModule&, SchrodingerState&, size_t);

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)

// Each per-ISA kernel TU is compiled with a specific set of -m flags
// (see src/clifft/CMakeLists.txt). The runtime dispatcher must match
// that exact set or risk emitting an instruction the host CPU cannot
// execute. The helpers below are the one place where each kernel's
// compile-time feature requirements are declared, so the auto-detect
// path and the CLIFFT_FORCE_ISA override path stay in sync.

static bool host_supports_avx2_kernel() {
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64))
    // svm_avx2.cc is compiled with -mavx2 -mbmi2 -mfma. All three must
    // be present at runtime, not just avx2+bmi2 -- a CPU with AVX2+BMI2
    // but no FMA (Excavator-gen AMD, some VIA Eden-X4) will SIGILL on
    // an FMA instruction otherwise.
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

static bool host_supports_avx512_kernel() {
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64))
    // svm_avx512.cc is compiled with -mavx2 -mbmi2 -mfma -mavx512f
    // -mavx512dq, so all five features must be present at runtime.
    return host_supports_avx2_kernel() && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

// Trap functions installed when the user explicitly requests an ISA via
// CLIFFT_FORCE_ISA that the host CPU cannot actually execute. We install
// these at static-initialization time and let the throw fire on the
// first execute() call -- throwing during static init would terminate
// the entire process (no Python catch is in scope yet) and turn a clear
// runtime error into a hard crash at import.
[[noreturn]] static void trap_force_isa_avx2(const CompiledModule&, SchrodingerState&, size_t) {
    throw std::runtime_error(
        "CLIFFT_FORCE_ISA=avx2 requested but host CPU lacks one or more required "
        "features (avx2, bmi2, fma). Unset CLIFFT_FORCE_ISA to use the auto-detected "
        "fallback, or set it to 'scalar' explicitly.");
}

[[noreturn]] static void trap_force_isa_avx512(const CompiledModule&, SchrodingerState&, size_t) {
    throw std::runtime_error(
        "CLIFFT_FORCE_ISA=avx512 requested but host CPU lacks one or more required "
        "features (avx2, bmi2, fma, avx512f, avx512dq). Unset CLIFFT_FORCE_ISA to use "
        "the auto-detected fallback, or set it to 'avx2' or 'scalar' explicitly.");
}

[[noreturn]] static void trap_force_isa_unknown(const CompiledModule&, SchrodingerState&, size_t) {
    throw std::runtime_error(
        "CLIFFT_FORCE_ISA is set to an unrecognized value. Accepted values are "
        "'avx512', 'avx2', 'scalar' (case-insensitive). Unset CLIFFT_FORCE_ISA to "
        "use the auto-detected fallback.");
}

// Lowercase a copy of the env var and strip surrounding whitespace so the
// parser stays case-insensitive without mutating the host environment.
static std::string normalize_force_isa(const char* env) {
    std::string s(env);
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static DispatchFn resolve_dispatcher() {
    // Allow environment override for testing.
    if (const char* env = std::getenv("CLIFFT_FORCE_ISA")) {
        std::string name = normalize_force_isa(env);
        if (name == "avx512") {
            return host_supports_avx512_kernel() ? avx512::execute_internal : trap_force_isa_avx512;
        }
        if (name == "avx2") {
            return host_supports_avx2_kernel() ? avx2::execute_internal : trap_force_isa_avx2;
        }
        if (name == "scalar") {
            return scalar::execute_internal;
        }
        // An empty CLIFFT_FORCE_ISA (e.g. `CLIFFT_FORCE_ISA= python ...`)
        // falls back to auto-detect rather than reporting an error: bash
        // sets the var to empty when written that way, and the previous
        // dispatcher treated empty as scalar. Allow it to behave like
        // "unset" so callers don't have to special-case shell quirks.
        if (name.empty()) {
            // Fall through to auto-detect below.
        } else {
            return trap_force_isa_unknown;
        }
    }

    // Auto-detect: walk down to the highest kernel the host can run.
    if (host_supports_avx512_kernel()) {
        return avx512::execute_internal;
    }
    if (host_supports_avx2_kernel()) {
        return avx2::execute_internal;
    }
    return scalar::execute_internal;
}

#endif  // CLIFFT_ENABLE_RUNTIME_DISPATCH

// =============================================================================
// Public execute() wrapper
// =============================================================================

// Resolved once on first use; shared by execute() and svm_backend().
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
static DispatchFn resolved_fn = resolve_dispatcher();
#endif

void execute(const CompiledModule& program, SchrodingerState& state) {
    if (state.pending_trap.has_value()) {
        throw std::invalid_argument(
            "execute(): the state has a pending trap; continue the shot with resume() or reset "
            "the state before starting a new shot");
    }
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    resolved_fn(program, state, 0);
#else
    scalar::execute_internal(program, state, 0);
#endif
}

void resume(const CompiledModule& program, SchrodingerState& state, uint32_t offset) {
    assert_arena_widths_match(program.num_qubits, program.constant_pool);
    if (!state.pending_trap.has_value()) {
        throw std::invalid_argument(
            "resume(): the state has no pending trap; resume() only continues a shot that "
            "execute() halted at an instrument trap");
    }
    if (program.num_qubits != state.num_qubits) {
        throw std::invalid_argument(
            "resume(): continuation module declares " + std::to_string(program.num_qubits) +
            " qubits but the state was built for " + std::to_string(state.num_qubits));
    }
    if (program.num_detectors != state.det_record.size() ||
        program.num_observables != state.obs_record.size() ||
        program.num_exp_vals != state.exp_vals.size()) {
        throw std::invalid_argument(
            "resume(): continuation module's detector/observable/exp-val counts do not match "
            "the state; the visible structure of a continuation must equal the original's");
    }

    // Resume at the instruction immediately after the trapped site. Validate
    // the caller's offset so bytecode is neither skipped nor executed twice.
    const uint32_t site_id = state.pending_trap->site_id;
    if (site_id >= program.instrument_offsets.size() ||
        program.instrument_offsets[site_id] == std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            "resume(): the continuation module has no instrument at the trapped site id " +
            std::to_string(site_id) + "; its prefix does not match the executed module");
    }
    if (offset != program.instrument_offsets[site_id] + 1) {
        throw std::invalid_argument("resume(): offset " + std::to_string(offset) +
                                    " does not follow the trapped site (expected " +
                                    std::to_string(program.instrument_offsets[site_id] + 1) +
                                    " for site " + std::to_string(site_id) + ")");
    }

    // A rewritten continuation may need more amplitudes or hidden measurement
    // slots than the original module. Visible slots keep their indices while
    // these buffers grow.
    state.grow_for_continuation(program.peak_rank);
    if (state.meas_record.size() < program.total_meas_slots) {
        state.meas_record.resize(program.total_meas_slots, 0);
    }

    // Restart noise sampling at the first noise site at or after `offset`.
    // Drawing a new exponential gap is exact because that distribution is
    // memoryless.
    state.next_noise_idx = static_cast<uint32_t>(program.constant_pool.noise_sites.size());
    for (size_t i = offset; i < program.bytecode.size(); ++i) {
        const Instruction& instr = program.bytecode[i];
        if (instr.opcode == Opcode::OP_NOISE || instr.opcode == Opcode::OP_NOISE_BLOCK) {
            state.next_noise_idx = instr.pauli.cp_mask_idx;
            break;
        }
    }
    state.draw_next_noise(program.constant_pool.noise_hazards);

    state.pending_trap.reset();

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    resolved_fn(program, state, offset);
#else
    scalar::execute_internal(program, state, offset);
#endif
}

const char* svm_backend() {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    if (resolved_fn == avx512::execute_internal)
        return "avx512";
    if (resolved_fn == avx2::execute_internal)
        return "avx2";
    // Trap states surface a CLIFFT_FORCE_ISA misconfig: the user asked
    // for an ISA the host can't run, or set the variable to a value the
    // dispatcher doesn't recognize. Reporting these distinctly (rather
    // than masquerading as "scalar") lets tests and tooling notice the
    // misconfig before execute() actually throws.
    if (resolved_fn == trap_force_isa_avx512)
        return "trap:avx512";
    if (resolved_fn == trap_force_isa_avx2)
        return "trap:avx2";
    if (resolved_fn == trap_force_isa_unknown)
        return "trap:unknown";
#endif
    return "scalar";
}

// Plain sampling cannot return a partial shot. Instrument programs that stop
// at a trap must be run by the trajectory driver, which handles resume().
static void throw_on_pending_trap(const SchrodingerState& state) {
    if (state.pending_trap.has_value()) {
        throw std::runtime_error(
            "a shot halted at a resumable instrument trap; instrument programs require the "
            "trajectory driver (execute/resume), not the plain sampling entry points");
    }
}

// =============================================================================
// Multi-Shot Sampling
// =============================================================================

SampleResult sample(const CompiledModule& program, uint32_t shots, std::optional<uint64_t> seed) {
    SampleResult result;
    if (shots == 0) {
        return result;
    }

    assert_arena_widths_match(program.num_qubits, program.constant_pool);

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
                            .num_qubits = program.num_qubits,
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
        throw_on_pending_trap(state);

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
        if (num_ev > 0)
            std::copy(state.exp_vals.begin(), state.exp_vals.end(),
                      result.exp_vals.begin() +
                          static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_ev));
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

    assert_arena_widths_match(program.num_qubits, program.constant_pool);

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
                            .num_qubits = program.num_qubits,
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
        throw_on_pending_trap(state);

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
            if (num_ev > 0)
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
        // k-fault conditioning requires one fixed probability per site. An
        // asymmetric flip instead depends on whether the current record bit is
        // 0 or 1, so it cannot be included.
        if (!entry.is_symmetric()) {
            throw std::invalid_argument(
                "k-fault conditioning does not support asymmetric readout noise; "
                "measurement record index " +
                std::to_string(entry.meas_idx) + " has probabilities (" +
                std::to_string(entry.prob_zero_to_one) + ", " +
                std::to_string(entry.prob_one_to_zero) + ")");
        }
        probs.push_back(entry.prob_zero_to_one);
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

    assert_arena_widths_match(program.num_qubits, program.constant_pool);

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
                            .num_qubits = program.num_qubits,
                            .num_detectors = num_det,
                            .num_observables = num_obs,
                            .num_exp_vals = num_ev,
                            .seed = seed});

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0)
            state.reset();

        prepare_forced_shot(state, w, dp, k, n_q, uniform_mode, uniform_pool);
        execute(program, state);
        throw_on_pending_trap(state);

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

        if (num_ev > 0)
            std::copy(state.exp_vals.begin(), state.exp_vals.end(),
                      result.exp_vals.begin() +
                          static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_ev));
    }

    return result;
}

SurvivorResult sample_k_survivors(const CompiledModule& program, uint32_t shots, uint32_t k,
                                  std::optional<uint64_t> seed, bool keep_records) {
    SurvivorResult result;
    result.total_shots = shots;
    if (shots == 0)
        return result;

    assert_arena_widths_match(program.num_qubits, program.constant_pool);

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
                            .num_qubits = program.num_qubits,
                            .num_detectors = num_det,
                            .num_observables = num_obs,
                            .num_exp_vals = num_ev,
                            .seed = seed});

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0)
            state.reset();

        prepare_forced_shot(state, w, dp, k, n_q, uniform_mode, uniform_pool);
        execute(program, state);
        throw_on_pending_trap(state);

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
            if (num_ev > 0)
                result.exp_vals.insert(result.exp_vals.end(), state.exp_vals.begin(),
                                       state.exp_vals.end());
        }
    }

    return result;
}

}  // namespace clifft
