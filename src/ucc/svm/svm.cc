#include "ucc/svm/svm.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

// Cross-platform restrict qualifier to unblock SIMD auto-vectorization
#if defined(__GNUC__) || defined(__clang__)
#define UCC_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define UCC_RESTRICT __restrict
#else
#define UCC_RESTRICT
#endif

namespace ucc {

// =============================================================================
// SchrodingerState Implementation
// =============================================================================

SchrodingerState::SchrodingerState(uint32_t peak_rank, uint32_t num_measurements,
                                   uint32_t num_detectors, uint32_t num_observables, uint64_t seed)
    : peak_rank_(peak_rank), rng_(seed) {
    // Pre-allocate records first (may throw, but no cleanup needed)
    meas_record.resize(num_measurements, 0);
    det_record.resize(num_detectors, 0);
    obs_record.resize(num_observables, 0);

    // Allocate 2^peak_rank complex numbers, 64-byte aligned for AVX
    array_size_ = 1ULL << peak_rank;
    size_t bytes = array_size_ * sizeof(std::complex<double>);

    // Use aligned_alloc (C++17). Size must be multiple of alignment.
    size_t aligned_bytes = (bytes + 63) & ~63ULL;
    v_ = static_cast<std::complex<double>*>(std::aligned_alloc(64, aligned_bytes));
    if (!v_) {
        throw std::bad_alloc();
    }

    // Initialize to |0...0>: coefficient 1 at index 0
    // Note: If this loop could throw (it can't for POD), we'd leak v_.
    // For complex<double>, value initialization is noexcept.
    for (uint64_t i = 0; i < array_size_; ++i) {
        v_[i] = {0.0, 0.0};
    }
    v_[0] = {1.0, 0.0};
}

SchrodingerState::~SchrodingerState() {
    std::free(v_);
}

SchrodingerState::SchrodingerState(SchrodingerState&& other) noexcept
    : destab_signs(other.destab_signs),
      stab_signs(other.stab_signs),
      meas_record(std::move(other.meas_record)),
      det_record(std::move(other.det_record)),
      obs_record(std::move(other.obs_record)),
      v_(other.v_),
      array_size_(other.array_size_),
      peak_rank_(other.peak_rank_),
      rng_(std::move(other.rng_)) {
    other.v_ = nullptr;
    other.array_size_ = 0;
}

SchrodingerState& SchrodingerState::operator=(SchrodingerState&& other) noexcept {
    if (this != &other) {
        std::free(v_);
        v_ = other.v_;
        array_size_ = other.array_size_;
        peak_rank_ = other.peak_rank_;
        rng_ = std::move(other.rng_);
        destab_signs = other.destab_signs;
        stab_signs = other.stab_signs;
        meas_record = std::move(other.meas_record);
        det_record = std::move(other.det_record);
        obs_record = std::move(other.obs_record);
        other.v_ = nullptr;
        other.array_size_ = 0;
    }
    return *this;
}

void SchrodingerState::reset(uint64_t seed) {
    // O(1) reset: OP_BRANCH always writes the spawned half, so memory beyond
    // current rank is never read. We only need to reset the vacuum state.
    v_[0] = {1.0, 0.0};
    destab_signs = 0;
    stab_signs = 0;

    // meas_record and det_record are sequentially overwritten each shot
    // (via meas_idx++ and det_idx++), so zeroing wastes memory bandwidth.
    // obs_record uses XOR accumulation (^=) and must be cleared.
    if (!obs_record.empty()) {
        std::fill(obs_record.begin(), obs_record.end(), 0);
    }

    rng_.seed(seed);
}

// =============================================================================
// T-Gate Math Constants
// =============================================================================
//
// T = diag(1, e^{ipi/4}) acts on computational basis.
// In Heisenberg picture with GF(2) indexing, we apply butterflies:
//
// For BRANCH/COLLIDE, the transformation mixes pairs of amplitudes:
//   |0> -> |0>
//   |1> -> e^{ipi/4}|1> = (1/sqrt(2))(1 + i)|1>... but we use tan(pi/8) form
//
// The LCU decomposition: T = cos(pi/8) I + i sin(pi/8) Z
// Weight w = tan(pi/8) ~ 0.4142...

static constexpr double kTanPi8 = 0.4142135623730950488;  // tan(pi/8)
static constexpr double kCosPi8 = 0.9238795325112867561;  // cos(pi/8)
static constexpr double kSinPi8 = 0.3826834323650897717;  // sin(pi/8)

// Phase factors for base_phase_idx: {1, i, -1, -i}
static const std::complex<double> kPhases[4] = {
    {1.0, 0.0},   // 0: +1
    {0.0, 1.0},   // 1: +i
    {-1.0, 0.0},  // 2: -1
    {0.0, -1.0}   // 3: -i
};

// =============================================================================
// SVM Opcode Handlers
// =============================================================================

namespace {

// Compute sign contribution from commutation with basis vectors.
// For each bit set in alpha that also appears in commutation_mask,
// we get a factor of -1 (odd parity = negative).
inline int compute_sign_parity(uint32_t alpha, uint32_t commutation_mask) {
    return std::popcount(alpha & commutation_mask) & 1;
}

// Evaluate the Pauli against the dynamic error frame and resolve sign.
// This is CRITICAL: T-gates must respect the stochastic Pauli frame
// accumulated from measurements and noise.
inline std::complex<double> resolve_sign(const SchrodingerState& state, const Instruction& instr) {
    // Symplectic inner product: (frame_X & obs_Z) XOR (frame_Z & obs_X)
    uint64_t anti_comm = (state.destab_signs & instr.branch.stab_mask) ^
                         (state.stab_signs & instr.branch.destab_mask);
    int frame_parity = std::popcount(anti_comm) & 1;
    // Shift base_phase_idx by 2 (which is equivalent to *-1) if frame flips sign
    return kPhases[(instr.base_phase_idx + 2 * frame_parity) & 3];
}

// OP_BRANCH: New dimension, array conceptually doubles.
// Uses dominant term factoring: states are unnormalized, with the cos^k(pi/8)
// factor implicit. This saves FLOPs since v[alpha] is untouched (0 writes).
// The spawned branch gets weight i*tan(pi/8)*Z for T, or -i*tan(pi/8)*Z for T_dag.
void op_branch(SchrodingerState& state, const Instruction& instr, uint32_t& current_rank) {
    uint32_t new_bit = current_rank++;
    uint64_t old_size = 1ULL << new_bit;

    std::complex<double> base_phase = resolve_sign(state, instr);
    bool is_dagger = (instr.flags & Instruction::FLAG_IS_DAGGER) != 0;
    std::complex<double> rel_weight =
        is_dagger ? std::complex<double>(0.0, kTanPi8) : std::complex<double>(0.0, -kTanPi8);

    std::complex<double> f0 = rel_weight * base_phase;
    std::complex<double> f1 = -f0;

    // Strict pointer aliasing: src (lower half) and dst (upper half) are disjoint
    const std::complex<double>* UCC_RESTRICT src = state.v();
    std::complex<double>* UCC_RESTRICT dst = state.v() + old_size;
    uint32_t c_mask = instr.commutation_mask;

    if (c_mask == 0) {
        // Fast path: all parities are zero, pure scalar multiply vectorizes to vmulpd
        for (uint64_t i = 0; i < old_size; ++i) {
            dst[i] = src[i] * f0;
        }
    } else {
        for (uint64_t i = 0; i < old_size; ++i) {
            int source_parity = compute_sign_parity(static_cast<uint32_t>(i), c_mask);
            dst[i] = src[i] * (source_parity ? f1 : f0);  // Compiles to vblendvpd
        }
    }
}

// OP_COLLIDE: In-place butterfly on existing dimension.
// Uses dominant term factoring: v' = v + i*tan(pi/8)*Z*partner.
// Block-based iteration for cache-friendly access.
void op_collide(SchrodingerState& state, const Instruction& instr, uint32_t current_rank) {
    assert(instr.branch.x_mask != 0 && "OP_COLLIDE requires nonzero x_mask");

    std::complex<double> base_phase = resolve_sign(state, instr);
    bool is_dagger = (instr.flags & Instruction::FLAG_IS_DAGGER) != 0;
    std::complex<double> rel_weight =
        is_dagger ? std::complex<double>(0.0, kTanPi8) : std::complex<double>(0.0, -kTanPi8);

    std::complex<double> f0 = rel_weight * base_phase;
    std::complex<double> f1 = -f0;

    uint32_t x_mask = instr.branch.x_mask;
    uint32_t pivot_bit = std::countr_zero(x_mask);

    uint64_t block_size = 1ULL << pivot_bit;
    uint64_t size = 1ULL << current_rank;
    uint64_t num_blocks = (size >> 1) / block_size;
    std::complex<double>* v = state.v();
    uint32_t c_mask = instr.commutation_mask;

    if (c_mask == 0) {
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t src_start = b * (block_size * 2);
            uint64_t beta_start = src_start ^ x_mask;

            // Disjoint pointers enable SIMD auto-vectorization
            std::complex<double>* UCC_RESTRICT pA = v + src_start;
            std::complex<double>* UCC_RESTRICT pB = v + beta_start;

            for (uint64_t i = 0; i < block_size; ++i) {
                std::complex<double> va = pA[i];
                std::complex<double> vb = pB[i];
                pA[i] = va + vb * f0;
                pB[i] = vb + va * f0;
            }
        }
    } else {
        // Algebraic parity reduction: parity(beta) = parity(alpha) ^ parity(x_mask)
        int x_mask_parity = compute_sign_parity(x_mask, c_mask);

        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t src_start = b * (block_size * 2);
            uint64_t beta_start = src_start ^ x_mask;

            std::complex<double>* UCC_RESTRICT pA = v + src_start;
            std::complex<double>* UCC_RESTRICT pB = v + beta_start;

            for (uint64_t i = 0; i < block_size; ++i) {
                uint64_t alpha = src_start + i;
                int parity_a = compute_sign_parity(static_cast<uint32_t>(alpha), c_mask);
                int parity_b = parity_a ^ x_mask_parity;  // Saves one popcount per iteration

                std::complex<double> va = pA[i];
                std::complex<double> vb = pB[i];

                pA[i] = va + vb * (parity_b ? f1 : f0);
                pB[i] = vb + va * (parity_a ? f1 : f0);
            }
        }
    }
}

// OP_SCALAR_PHASE: Diagonal phase when beta=0.
// Even though beta=0 means no new dimension, the Z components can still
// anti-commute with active basis vectors. Different amplitudes may receive
// different phases based on commutation_mask.
// Helper to check is_dagger flag
inline bool instr_is_dagger(const Instruction& instr) {
    return (instr.flags & Instruction::FLAG_IS_DAGGER) != 0;
}

void op_scalar_phase(SchrodingerState& state, const Instruction& instr, uint32_t current_rank) {
    std::complex<double> base_phase = resolve_sign(state, instr);
    std::complex<double> i_tan = instr_is_dagger(instr) ? std::complex<double>(0.0, kTanPi8)
                                                        : std::complex<double>(0.0, -kTanPi8);

    std::complex<double> f0 = 1.0 + i_tan * base_phase;
    std::complex<double> f1 = 1.0 - i_tan * base_phase;

    uint64_t size = 1ULL << current_rank;
    auto* v = state.v();
    uint32_t c_mask = instr.commutation_mask;

    if (c_mask == 0) {
        // Fast path: uniform phase, pure scalar multiply vectorizes cleanly
        for (uint64_t i = 0; i < size; ++i) {
            v[i] *= f0;
        }
    } else {
        for (uint64_t i = 0; i < size; ++i) {
            int parity = compute_sign_parity(static_cast<uint32_t>(i), c_mask);
            v[i] *= (parity ? f1 : f0);
        }
    }
}

// OP_MEASURE_MERGE: Anti-commuting measurement, samples and shrinks array.
// Uses projective butterfly: (I +/- M)/2 requires interference v[alpha] +/- phase*v[alpha XOR
// beta]. The phase includes base_phase (Y-count and sign) and commutation parity. Block-based
// iteration with single-pass butterfly+compaction. Returns the sampled outcome.
uint8_t op_measure_merge(SchrodingerState& state, const Instruction& instr,
                         uint32_t& current_rank) {
    assert(current_rank >= 1 && "MEASURE_MERGE requires at least one dimension");
    assert(instr.branch.x_mask != 0 && "MEASURE_MERGE requires nonzero x_mask");

    uint32_t bit_index = instr.branch.bit_index;
    uint32_t x_mask = instr.branch.x_mask;

    uint64_t block_size = 1ULL << bit_index;
    uint64_t new_size = 1ULL << (current_rank - 1);
    uint64_t num_blocks = new_size / block_size;
    std::complex<double>* v = state.v();

    std::complex<double> base_phase = resolve_sign(state, instr);
    std::complex<double> pf0 = base_phase;
    std::complex<double> pf1 = -base_phase;
    uint32_t c_mask = instr.commutation_mask;
    double prob_0 = 0.0, prob_1 = 0.0;

    // Pass 1: Gather L2 norms with phase-aware interference
    if (c_mask == 0) {
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t src_start = b * (block_size * 2);
            uint64_t target_start = src_start ^ x_mask;

            const std::complex<double>* UCC_RESTRICT pA = v + src_start;
            const std::complex<double>* UCC_RESTRICT pB = v + target_start;

            for (uint64_t i = 0; i < block_size; ++i) {
                std::complex<double> term = pB[i] * pf0;
                prob_0 += std::norm(pA[i] + term);
                prob_1 += std::norm(pA[i] - term);
            }
        }
    } else {
        int x_mask_parity = compute_sign_parity(x_mask, c_mask);
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t src_start = b * (block_size * 2);
            uint64_t target_start = src_start ^ x_mask;

            const std::complex<double>* UCC_RESTRICT pA = v + src_start;
            const std::complex<double>* UCC_RESTRICT pB = v + target_start;

            for (uint64_t i = 0; i < block_size; ++i) {
                uint64_t beta = target_start + i;
                int parity_b = compute_sign_parity(static_cast<uint32_t>(beta), c_mask);
                std::complex<double> term = pB[i] * (parity_b ? pf1 : pf0);
                prob_0 += std::norm(pA[i] + term);
                prob_1 += std::norm(pA[i] - term);
            }
        }
    }

    double total = prob_0 + prob_1;
    if (total < 1e-30) {
        return 0;
    }

    double r = state.random_double();
    uint8_t internal_outcome = (r < prob_0 / total) ? 0 : 1;
    double norm = 1.0 / std::sqrt(internal_outcome ? prob_1 : prob_0);

    // Pass 2: Butterfly and compaction.
    // Unify outcome==0 and outcome==1 by folding the sign into the weight.
    std::complex<double> w0 = (internal_outcome == 0) ? pf0 * norm : -pf0 * norm;
    std::complex<double> w1 = -w0;

    if (c_mask == 0) {
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t dst_start = b * block_size;
            uint64_t src_start = b * (block_size * 2);
            uint64_t target_start = src_start ^ x_mask;

            for (uint64_t i = 0; i < block_size; ++i) {
                v[dst_start + i] = v[src_start + i] * norm + v[target_start + i] * w0;
            }
        }
    } else {
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t dst_start = b * block_size;
            uint64_t src_start = b * (block_size * 2);
            uint64_t target_start = src_start ^ x_mask;

            for (uint64_t i = 0; i < block_size; ++i) {
                uint64_t beta = target_start + i;
                int parity_b = compute_sign_parity(static_cast<uint32_t>(beta), c_mask);
                v[dst_start + i] =
                    v[src_start + i] * norm + v[target_start + i] * (parity_b ? w1 : w0);
            }
        }
    }

    current_rank--;
    return internal_outcome;
}

// OP_MEASURE_FILTER: Commuting measurement with beta=0 but nonzero commutation.
// Observable commutes with tableau but has sign dependence on GF(2) index.
// Returns the sampled outcome.
uint8_t op_measure_filter(SchrodingerState& state, const Instruction& instr,
                          uint32_t current_rank) {
    uint64_t size = 1ULL << current_rank;
    auto* v = state.v();
    uint32_t c_mask = instr.commutation_mask;
    double prob_0 = 0.0, prob_1 = 0.0;

    for (uint64_t alpha = 0; alpha < size; ++alpha) {
        double norm_sq = std::norm(v[alpha]);
        if (compute_sign_parity(static_cast<uint32_t>(alpha), c_mask)) {
            prob_1 += norm_sq;
        } else {
            prob_0 += norm_sq;
        }
    }

    double total = prob_0 + prob_1;
    if (total < 1e-30) {
        return 0;
    }

    double r = state.random_double();
    uint8_t internal_outcome = (r < prob_0 / total) ? 0 : 1;

    uint64_t anti_comm = (state.destab_signs & instr.branch.stab_mask) ^
                         (state.stab_signs & instr.branch.destab_mask);
    int frame_parity = std::popcount(anti_comm) & 1;

    uint8_t outcome = internal_outcome ^ static_cast<uint8_t>(frame_parity) ^ instr.ag_ref_outcome;

    double norm_factor = 1.0 / std::sqrt(internal_outcome ? prob_1 : prob_0);

    // Branchless conditional multiplier: keep matching-parity amplitudes, zero others
    for (uint64_t alpha = 0; alpha < size; ++alpha) {
        int parity = compute_sign_parity(static_cast<uint32_t>(alpha), c_mask);
        v[alpha] *= (parity == internal_outcome) ? norm_factor : 0.0;
    }
    return outcome;
}

// OP_MEASURE_DETERMINISTIC: Fully deterministic measurement (beta=0, comm=0).
// Outcome determined purely by Pauli frame signs.
// Returns the computed outcome.
uint8_t op_measure_deterministic(SchrodingerState& state, const Instruction& instr) {
    // The measurement observable is P = X^destab_mask * Z^stab_mask.
    // The Pauli frame has accumulated X^destab_signs * Z^stab_signs.
    // Anti-commutation: X anti-commutes with Z, so frame X bits that overlap
    // with observable Z bits (and vice versa) contribute to the sign.
    //
    // Commutator parity: (frame_X & obs_Z) XOR (frame_Z & obs_X)
    uint64_t anti_comm = (state.destab_signs & instr.branch.stab_mask) ^
                         (state.stab_signs & instr.branch.destab_mask);
    int parity = std::popcount(anti_comm) & 1;

    return static_cast<uint8_t>(parity) ^ instr.ag_ref_outcome;
}

// OP_AG_PIVOT: Aaronson-Gottesman pivot for anti-commuting measurements.
// Uses last_outcome when FLAG_REUSE_OUTCOME is set (follows MEASURE_MERGE).
// Otherwise samples fresh 50/50 outcome.
// Returns the outcome for use by subsequent operations.
//
// Transforms the error frame through the AG pivot matrix (mapping old logical
// coordinates to new logical coordinates), then injects the measurement
// divergence by XORing the pivot slot into destab_signs.
uint8_t op_ag_pivot(SchrodingerState& state, const Instruction& instr, const ConstantPool& pool,
                    uint8_t last_outcome) {
    // Compute error anti-commutation for both branches.
    // The error frame anti-commutes with the measured observable if:
    //   (error_X & obs_Z) XOR (error_Z & obs_X) has odd popcount
    // This affects how we interpret/record the measurement outcome.
    uint64_t anti_comm =
        (state.destab_signs & instr.meta.stab_mask) ^ (state.stab_signs & instr.meta.destab_mask);
    int error_parity = std::popcount(anti_comm) & 1;

    uint8_t divergence;
    uint8_t outcome;
    bool reuse_outcome = (instr.flags & Instruction::FLAG_REUSE_OUTCOME) != 0;

    if (reuse_outcome) {
        // Extract clean divergence from the provided outcome.
        //
        // When MEASURE_MERGE precedes AG_PIVOT, the interference-based
        // measurement naturally incorporates error_parity via the phase
        // accumulation in resolve_sign(). The outcome 'm' satisfies:
        //   m = d ^ error_parity ^ r   (where d=divergence, r=reference)
        //
        // We need the clean divergence 'd' to correctly update the error
        // frame, so we solve for it:
        //   d = m ^ error_parity ^ r
        divergence = last_outcome ^ static_cast<uint8_t>(error_parity) ^ instr.ag_ref_outcome;
        outcome = last_outcome;
    } else {
        // Standalone AG_PIVOT: sample random 50/50.
        // The outcome is: (random divergence) XOR (error parity) XOR (reference)
        // This correctly accounts for errors flipping the measurement outcome.
        double r = state.random_double();
        divergence = (r < 0.5) ? 0 : 1;
        outcome = divergence ^ static_cast<uint8_t>(error_parity) ^ instr.ag_ref_outcome;
    }

    // Sparse GF(2) Matrix Transformation of the Error Frame
    // Uses pre-extracted boolean columns, avoiding Stim PauliString allocations.
    const auto& mat = pool.ag_matrices[instr.meta.payload_idx];
    mat.apply(state.destab_signs, state.stab_signs);

    // Inject Measurement Divergence
    // D'_p is an X-type Pauli, so we XOR into destab_signs to preserve any
    // existing error that was transformed through the AG pivot matrix.
    uint32_t p = instr.meta.ag_stab_slot;
    state.destab_signs ^= (static_cast<uint64_t>(divergence) << p);

    return outcome;
}

// OP_CONDITIONAL: Apply Pauli correction based on measurement outcome.
// Uses last_outcome when FLAG_USE_LAST_OUTCOME is set (for reset decomposition).
void op_conditional(SchrodingerState& state, const Instruction& instr, uint8_t last_outcome) {
    uint8_t ctrl_val;
    if ((instr.flags & Instruction::FLAG_USE_LAST_OUTCOME) != 0) {
        ctrl_val = last_outcome;
    } else {
        ctrl_val = state.meas_record[instr.meta.controlling_meas];
    }

    if (ctrl_val == 1) {
        // XOR the Pauli into the frame
        state.destab_signs ^= instr.meta.destab_mask;
        state.stab_signs ^= instr.meta.stab_mask;
    }
}

}  // namespace

// =============================================================================
// SVM Execution Entry Points
// =============================================================================

// Internal execute implementation. When trace is non-null, captures state snapshots.
static void execute_impl(const CompiledModule& program, SchrodingerState& state,
                         std::vector<TraceEntry>* trace) {
    uint32_t current_rank = 0;  // Start with rank 0 (single amplitude)
    uint32_t meas_idx = 0;
    uint32_t det_idx = 0;
    uint8_t last_outcome = 0;  // Tracks outcome for hidden measurements / reset

    const auto& schedule = program.constant_pool.noise_schedule;
    const auto& hazards = program.constant_pool.cumulative_hazards;
    const auto& det_targets = program.constant_pool.detector_targets;
    const auto& obs_targets = program.constant_pool.observable_targets;

    // Geometric gap sampling: jump directly to the next noise site that fires.
    // Instead of drawing one RNG per noise site (O(N)), we draw one Exponential
    // variate and binary-search the cumulative hazard array to find where it
    // lands (O(log N) per error, O(E log N) total where E = expected errors).
    size_t next_noise_idx = schedule.size();  // default: no errors
    double current_hazard = 0.0;

    auto sample_next_error = [&]() {
        if (next_noise_idx >= schedule.size())
            return;
        // Draw Exp(1): -ln(U) where U ~ Uniform(0,1].
        // 1.0 - random_double() maps [0,1) to (0,1], avoiding log(0).
        double u = 1.0 - state.random_double();
        double target_hazard = current_hazard - std::log(u);

        auto it = std::upper_bound(hazards.begin() + static_cast<ptrdiff_t>(next_noise_idx),
                                   hazards.end(), target_hazard);

        if (it == hazards.end()) {
            next_noise_idx = schedule.size();
        } else {
            next_noise_idx = static_cast<size_t>(std::distance(hazards.begin(), it));
        }
    };

    if (!schedule.empty()) {
        next_noise_idx = 0;
        sample_next_error();
    }

    // Loop to <= bytecode.size() to process trailing noise at circuit end.
    // Noise scheduled at pc == bytecode.size() applies after all instructions.
    for (uint32_t pc = 0; pc <= program.bytecode.size(); ++pc) {
        // 1. Process noise only at sites where geometric sampling landed
        while (next_noise_idx < schedule.size() && pc == schedule[next_noise_idx].pc) {
            const auto& site = schedule[next_noise_idx];

            // An error fires here. Pick which Pauli channel.
            double r = state.random_double() * site.total_probability;
            double cum_p = 0.0;
            for (size_t i = 0; i < site.channels.size(); ++i) {
                cum_p += site.channels[i].prob;
                if (r < cum_p || i == site.channels.size() - 1) {
                    state.destab_signs ^= site.channels[i].destab_mask;
                    state.stab_signs ^= site.channels[i].stab_mask;
                    break;
                }
            }

            current_hazard = hazards[next_noise_idx];
            next_noise_idx++;
            sample_next_error();
        }

        // 2. Break if we've processed all bytecode
        if (pc == program.bytecode.size()) {
            break;
        }

        // 3. Execute the instruction at this PC
        const auto& instr = program.bytecode[pc];
        std::string detail;  // Only populated when trace != nullptr

        switch (instr.opcode) {
            case Opcode::OP_BRANCH:
                op_branch(state, instr, current_rank);
                if (trace) {
                    detail = "rank=" + std::to_string(current_rank);
                }
                break;

            case Opcode::OP_COLLIDE:
                op_collide(state, instr, current_rank);
                break;

            case Opcode::OP_SCALAR_PHASE:
                op_scalar_phase(state, instr, current_rank);
                break;

            case Opcode::OP_MEASURE_MERGE: {
                last_outcome = op_measure_merge(state, instr, current_rank);
                bool is_hidden = (instr.flags & Instruction::FLAG_HIDDEN) != 0;
                if (!is_hidden) {
                    state.meas_record[meas_idx++] = last_outcome;
                }
                if (trace) {
                    std::ostringstream ss;
                    ss << "outcome=" << (int)last_outcome << " rank=" << current_rank;
                    if (is_hidden)
                        ss << " [hidden]";
                    detail = ss.str();
                }
                break;
            }

            case Opcode::OP_MEASURE_FILTER: {
                last_outcome = op_measure_filter(state, instr, current_rank);
                bool is_hidden = (instr.flags & Instruction::FLAG_HIDDEN) != 0;
                if (!is_hidden) {
                    state.meas_record[meas_idx++] = last_outcome;
                }
                if (trace) {
                    std::ostringstream ss;
                    ss << "outcome=" << (int)last_outcome;
                    if (is_hidden)
                        ss << " [hidden]";
                    detail = ss.str();
                }
                break;
            }

            case Opcode::OP_MEASURE_DETERMINISTIC: {
                last_outcome = op_measure_deterministic(state, instr);
                bool is_hidden = (instr.flags & Instruction::FLAG_HIDDEN) != 0;
                if (!is_hidden) {
                    state.meas_record[meas_idx++] = last_outcome;
                }
                if (trace) {
                    std::ostringstream ss;
                    ss << "outcome=" << (int)last_outcome << " [det]";
                    if (is_hidden)
                        ss << " [hidden]";
                    detail = ss.str();
                }
                break;
            }

            case Opcode::OP_AG_PIVOT: {
                last_outcome = op_ag_pivot(state, instr, program.constant_pool, last_outcome);
                bool is_hidden = (instr.flags & Instruction::FLAG_HIDDEN) != 0;
                bool reuse_outcome = (instr.flags & Instruction::FLAG_REUSE_OUTCOME) != 0;
                if (!is_hidden && !reuse_outcome) {
                    state.meas_record[meas_idx++] = last_outcome;
                }
                if (trace) {
                    std::ostringstream ss;
                    ss << "outcome=" << (int)last_outcome;
                    if (is_hidden)
                        ss << " [hidden]";
                    if (reuse_outcome)
                        ss << " [reuse]";
                    detail = ss.str();
                }
                break;
            }

            case Opcode::OP_CONDITIONAL: {
                bool use_last = (instr.flags & Instruction::FLAG_USE_LAST_OUTCOME) != 0;
                uint8_t ctrl_val_before =
                    use_last ? last_outcome : state.meas_record[instr.meta.controlling_meas];
                op_conditional(state, instr, last_outcome);
                if (trace) {
                    std::ostringstream ss;
                    ss << "ctrl=" << (int)ctrl_val_before
                       << (ctrl_val_before ? " [applied]" : " [skipped]");
                    detail = ss.str();
                }
                break;
            }

            case Opcode::OP_READOUT_NOISE:
                if (state.random_double() < instr.readout.prob) {
                    state.meas_record[instr.readout.meas_idx] ^= 1;
                }
                break;

            case Opcode::OP_DETECTOR: {
                const auto& targets = det_targets[instr.detector.target_idx];
                uint8_t parity = 0;
                for (uint32_t idx : targets) {
                    parity ^= state.meas_record[idx];
                }
                state.det_record[det_idx++] = parity;
                if (trace) {
                    detail = "parity=" + std::to_string(parity);
                }
                break;
            }

            case Opcode::OP_OBSERVABLE: {
                const auto& targets = obs_targets[instr.observable.target_idx];
                uint8_t parity = 0;
                for (uint32_t idx : targets) {
                    parity ^= state.meas_record[idx];
                }
                state.obs_record[instr.observable.obs_idx] ^= parity;
                if (trace) {
                    detail = "obs[" + std::to_string(instr.observable.obs_idx) +
                             "]=" + std::to_string(parity);
                }
                break;
            }

            // Future opcodes - not implemented in MVP
            case Opcode::OP_BRANCH_LCU:
            case Opcode::OP_COLLIDE_LCU:
            case Opcode::OP_SCALAR_PHASE_LCU:
            case Opcode::OP_INDEX_CNOT:
            case Opcode::OP_POSTSELECT:
                break;
        }

        // Capture trace entry if tracing is enabled
        if (trace) {
            TraceEntry entry;
            entry.pc = pc;
            entry.opcode = instr.opcode;
            entry.rank_after = current_rank;
            entry.destab_signs = state.destab_signs;
            entry.stab_signs = state.stab_signs;
            uint64_t size = 1ULL << current_rank;
            entry.v.assign(state.v(), state.v() + size);
            entry.detail = std::move(detail);
            trace->push_back(std::move(entry));
        }
    }
}

void execute(const CompiledModule& program, SchrodingerState& state) {
    execute_impl(program, state, nullptr);
}

void execute_traced(const CompiledModule& program, SchrodingerState& state,
                    std::vector<TraceEntry>& trace) {
    execute_impl(program, state, &trace);
}

SampleResult sample(const CompiledModule& program, uint32_t shots, uint64_t seed) {
    SampleResult result;
    if (shots == 0) {
        return result;
    }

    uint32_t num_meas = program.num_measurements;
    uint32_t num_det = program.num_detectors;
    uint32_t num_obs = program.num_observables;

    result.measurements.resize(static_cast<size_t>(shots) * num_meas);
    result.detectors.resize(static_cast<size_t>(shots) * num_det);
    result.observables.resize(static_cast<size_t>(shots) * num_obs);

    // Create state once, reuse for all shots
    SchrodingerState state(program.peak_rank, num_meas, num_det, num_obs, seed);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0) {
            // Reset for next shot with deterministic seed progression
            state.reset(seed + shot);
        }

        execute(program, state);

        // Copy all records to results
        // Cast to size_t before multiplication to avoid uint32_t overflow
        std::copy(state.meas_record.begin(), state.meas_record.end(),
                  result.measurements.begin() +
                      static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_meas));
        std::copy(
            state.det_record.begin(), state.det_record.end(),
            result.detectors.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_det));
        std::copy(state.obs_record.begin(), state.obs_record.end(),
                  result.observables.begin() +
                      static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_obs));
    }

    return result;
}

// Internal structure for sparse reference state
struct NonZeroAmp {
    uint64_t k;
    std::complex<double> amp;
};

std::vector<std::complex<double>> get_statevector(
    const SchrodingerState& state, const std::vector<stim::bitword<kStimWidth>>& gf2_basis,
    const stim::Tableau<kStimWidth>& final_tableau, std::complex<double> global_weight) {
    uint32_t num_qubits = static_cast<uint32_t>(final_tableau.num_qubits);

    // OOM safety guard: dense statevector for >20 qubits would exceed reasonable memory
    if (num_qubits > 20) {
        throw std::invalid_argument(
            "Statevector too large (exceeds 20 qubits). Use sampling instead.");
    }

    uint64_t dim = 1ULL << num_qubits;

    // Phase-safe statevector expansion algorithm:
    // Stim's state_vector_from_stabilizers assigns an arbitrary global phase
    // per branch. To preserve exact relative phases between superposition branches,
    // we compute the reference state U|0> ONCE and then apply Pauli operators
    // to map each branch's initial X-string through the final tableau.

    // 1. Compute reference statevector U|0> exactly once to lock the global phase frame
    std::vector<stim::PauliString<kStimWidth>> base_stabs;
    std::vector<stim::PauliStringRef<kStimWidth>> stab_refs;
    base_stabs.reserve(num_qubits);
    for (uint32_t q = 0; q < num_qubits; ++q) {
        base_stabs.emplace_back(final_tableau.zs[q]);
        stab_refs.push_back(base_stabs.back().ref());
    }

    auto ref_sv_f =
        stim::VectorSimulator::state_vector_from_stabilizers<kStimWidth>(stab_refs, 1.0f);

    // OPTIMIZATION: Extract only non-zero amplitudes from reference state.
    // Stabilizer states are maximally sparse (exactly 2^M non-zeros of equal magnitude).
    // This accelerates the inner loop from O(2^rank * 2^N) to O(2^rank * 2^M).
    std::vector<NonZeroAmp> sparse_ref;
    sparse_ref.reserve(dim);  // Upper bound; actual count is much smaller
    for (uint64_t i = 0; i < dim; ++i) {
        std::complex<double> val(ref_sv_f[i].real(), ref_sv_f[i].imag());
        // Use std::norm to avoid expensive sqrt in abs
        if (std::norm(val) >= 1e-30) {
            sparse_ref.push_back({i, val});
        }
    }

    std::vector<std::complex<double>> sv(dim, {0.0, 0.0});
    uint32_t rank = static_cast<uint32_t>(gf2_basis.size());
    // Use active rank, not peak_rank - measurements compact the array
    uint64_t active_size = 1ULL << rank;

    stim::PauliString<kStimWidth> P(num_qubits);

    // 2. Accumulate each branch by applying P_alpha to the reference state
    // For each branch alpha, the state is U|x_initial> = U X^x U_dag U|0> = P' |ref>
    // where P' = final_tableau(X^x) is the Pauli propagated through the circuit
    for (uint64_t alpha = 0; alpha < active_size; ++alpha) {
        std::complex<double> coeff = state.v()[alpha];
        // Use std::norm to avoid expensive sqrt
        if (std::norm(coeff) < 1e-30) {
            continue;
        }

        // Compute the X-part (spatial shift) from GF(2) basis
        uint64_t x = 0;
        for (uint32_t i = 0; i < rank; ++i) {
            if ((alpha >> i) & 1) {
                x ^= static_cast<uint64_t>(gf2_basis[i]);
            }
        }

        // Apply Pauli frame: E = X^{destab_signs} Z^{stab_signs}
        // The branch state X^x |0> becomes E X^x |0> = X^{x XOR destab_signs} Z^{stab_signs} |0>
        // The Z^{stab_signs} anti-commutes with X^x, giving (-1)^{popcount(x & stab_signs)}
        //
        // NOTE: The observable's sign is already incorporated in v[alpha] via base_phase_idx
        // in the VM opcodes. We do NOT apply gf2_signs here to avoid double-counting.
        uint64_t initial_x = x ^ state.destab_signs;
        int frame_parity = std::popcount(x & state.stab_signs) % 2;
        double frame_phase = (frame_parity == 0) ? 1.0 : -1.0;

        // Reuse preallocated Pauli string P = X^{initial_x}
        P.xs.u64[0] = initial_x;
        P.zs.u64[0] = 0;
        P.sign = false;

        // Map P through the final tableau to get P' = U P U_dag
        stim::PauliString<kStimWidth> P_prime = final_tableau(P);

        uint64_t px = P_prime.xs.u64[0];
        uint64_t pz = P_prime.zs.u64[0];

        // Y = iXZ in Stim's convention, so Y operators contribute i per Y
        int y_count = std::popcount(px & pz);
        std::complex<double> i_fac = {1.0, 0.0};
        switch (y_count & 3) {
            case 1:
                i_fac = {0.0, 1.0};
                break;
            case 2:
                i_fac = {-1.0, 0.0};
                break;
            case 3:
                i_fac = {0.0, -1.0};
                break;
        }

        // Fold all global scalars and signs into a single constant
        std::complex<double> p_scale = coeff * frame_phase * global_weight * i_fac;
        if (P_prime.sign) {
            p_scale = -p_scale;
        }

        // Precompute both scale factors for branchless inner loop
        std::complex<double> scales[2] = {p_scale, -p_scale};

        // Apply P' to the SPARSE reference state and accumulate
        // P'|k> = (+/-1)|k XOR px> where the sign depends on k*pz parity
        for (const auto& item : sparse_ref) {
            uint64_t k_out = item.k ^ px;
            int parity = std::popcount(item.k & pz) & 1;
            sv[k_out] += item.amp * scales[parity];  // Branchless
        }
    }

    // 3. Normalize
    double norm_sq = 0.0;
    for (const auto& amp : sv) {
        norm_sq += std::norm(amp);
    }
    if (norm_sq > 1e-15) {
        double norm = std::sqrt(norm_sq);
        for (auto& amp : sv) {
            amp /= norm;
        }
    }

    return sv;
}

}  // namespace ucc
