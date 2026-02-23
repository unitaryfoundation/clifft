#include "ucc/svm/svm.h"

#include <bit>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

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

    // Initialize to |0...0⟩: coefficient 1 at index 0
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
    std::fill(meas_record.begin(), meas_record.end(), 0);
    std::fill(det_record.begin(), det_record.end(), 0);
    std::fill(obs_record.begin(), obs_record.end(), 0);
    rng_.seed(seed);
}

// =============================================================================
// T-Gate Math Constants
// =============================================================================
//
// T = diag(1, e^{iπ/4}) acts on computational basis.
// In Heisenberg picture with GF(2) indexing, we apply butterflies:
//
// For BRANCH/COLLIDE, the transformation mixes pairs of amplitudes:
//   |0⟩ → |0⟩
//   |1⟩ → e^{iπ/4}|1⟩ = (1/√2)(1 + i)|1⟩... but we use tan(π/8) form
//
// The LCU decomposition: T = cos(π/8) I + i sin(π/8) Z
// Weight w = tan(π/8) ≈ 0.4142...

static constexpr double kTanPi8 = 0.4142135623730950488;  // tan(π/8)
static constexpr double kCosPi8 = 0.9238795325112867561;  // cos(π/8)
static constexpr double kSinPi8 = 0.3826834323650897717;  // sin(π/8)

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
// Uses dominant term factoring: states are unnormalized, with the cos^k(π/8)
// factor implicit. This saves FLOPs since v[alpha] is untouched (0 writes).
// The spawned branch gets weight i*tan(π/8)*Z for T, or -i*tan(π/8)*Z for T†.
void op_branch(SchrodingerState& state, const Instruction& instr, uint32_t& current_rank) {
    uint32_t new_bit = current_rank++;
    uint64_t old_size = 1ULL << new_bit;  // Size before expansion

    std::complex<double> base_phase = resolve_sign(state, instr);
    // T = I - i*tan(π/8)*Z, T† = I + i*tan(π/8)*Z
    bool is_dagger = (instr.flags & Instruction::FLAG_IS_DAGGER) != 0;
    std::complex<double> rel_weight =
        is_dagger ? std::complex<double>(0.0, kTanPi8) : std::complex<double>(0.0, -kTanPi8);

    // Precompute the two possible phase factors (branchless array lookup)
    std::complex<double> factors[2] = {rel_weight * base_phase, -rel_weight * base_phase};

    auto* v = state.v();
    for (uint64_t alpha = 0; alpha < old_size; ++alpha) {
        uint64_t new_idx = alpha + old_size;  // Equivalent to alpha | (1 << new_bit)
        // Use parity of source index (alpha): commutation_mask encodes
        // anticommutation with existing basis vectors, not the new dimension
        int source_parity =
            compute_sign_parity(static_cast<uint32_t>(alpha), instr.commutation_mask);

        // Zero FLOPs on identity branch - just write spawned branch
        v[new_idx] = v[alpha] * factors[source_parity];
    }
}

// OP_COLLIDE: In-place butterfly on existing dimension.
// Uses dominant term factoring: v' = v + i*tan(π/8)*Z*partner.
// Block-based iteration for cache-friendly access.
void op_collide(SchrodingerState& state, const Instruction& instr, uint32_t current_rank) {
    assert(instr.branch.x_mask != 0 && "OP_COLLIDE requires nonzero x_mask");

    std::complex<double> base_phase = resolve_sign(state, instr);
    // T = I - i*tan(π/8)*Z, T† = I + i*tan(π/8)*Z
    bool is_dagger = (instr.flags & Instruction::FLAG_IS_DAGGER) != 0;
    std::complex<double> rel_weight =
        is_dagger ? std::complex<double>(0.0, kTanPi8) : std::complex<double>(0.0, -kTanPi8);

    std::complex<double> factors[2] = {rel_weight * base_phase, -rel_weight * base_phase};

    uint32_t x_mask = instr.branch.x_mask;
    uint32_t pivot_bit = std::countr_zero(x_mask);

    uint64_t block_size = 1ULL << pivot_bit;
    uint64_t size = 1ULL << current_rank;
    uint64_t num_blocks = (size >> 1) / block_size;
    auto* v = state.v();

    for (uint64_t b = 0; b < num_blocks; ++b) {
        uint64_t src_start = b * (block_size * 2);
        for (uint64_t i = 0; i < block_size; ++i) {
            uint64_t alpha = src_start + i;
            uint64_t beta = alpha ^ x_mask;

            int parity_a =
                compute_sign_parity(static_cast<uint32_t>(alpha), instr.commutation_mask);
            int parity_b = compute_sign_parity(static_cast<uint32_t>(beta), instr.commutation_mask);

            std::complex<double> va = v[alpha];
            std::complex<double> vb = v[beta];

            // Dominant term factoring: self + w*Z*partner
            v[alpha] = va + vb * factors[parity_b];
            v[beta] = vb + va * factors[parity_a];
        }
    }
}

// OP_SCALAR_PHASE: Diagonal phase when β=0.
// Even though β=0 means no new dimension, the Z components can still
// anti-commute with active basis vectors. Different amplitudes may receive
// different phases based on commutation_mask.
// Helper to check is_dagger flag
inline bool instr_is_dagger(const Instruction& instr) {
    return (instr.flags & Instruction::FLAG_IS_DAGGER) != 0;
}

void op_scalar_phase(SchrodingerState& state, const Instruction& instr, uint32_t current_rank) {
    std::complex<double> base_phase = resolve_sign(state, instr);
    // T = I - i*tan(π/8)*Z, T† = I + i*tan(π/8)*Z
    std::complex<double> i_tan = instr_is_dagger(instr) ? std::complex<double>(0.0, kTanPi8)
                                                        : std::complex<double>(0.0, -kTanPi8);

    // Phase depends on whether each basis index anti-commutes with observable
    std::complex<double> factors[2] = {1.0 + i_tan * base_phase, 1.0 - i_tan * base_phase};

    uint64_t size = 1ULL << current_rank;
    auto* v = state.v();
    for (uint64_t i = 0; i < size; ++i) {
        int parity = compute_sign_parity(static_cast<uint32_t>(i), instr.commutation_mask);
        v[i] *= factors[parity];  // Branchless phase application
    }
}

// OP_MEASURE_MERGE: Anti-commuting measurement, samples and shrinks array.
// Uses projective butterfly: (I ± M)/2 requires interference v[α] ± phase*v[α ⊕ β].
// The phase includes base_phase (Y-count and sign) and commutation parity.
// Block-based iteration with single-pass butterfly+compaction.
// Returns the sampled outcome.
uint8_t op_measure_merge(SchrodingerState& state, const Instruction& instr,
                         uint32_t& current_rank) {
    // Defensive assertions to catch backend logic errors
    assert(current_rank >= 1 && "MEASURE_MERGE requires at least one dimension");
    assert(instr.branch.x_mask != 0 && "MEASURE_MERGE requires nonzero x_mask");

    uint32_t bit_index = instr.branch.bit_index;
    uint32_t x_mask = instr.branch.x_mask;

    uint64_t block_size = 1ULL << bit_index;
    uint64_t new_size = 1ULL << (current_rank - 1);
    uint64_t num_blocks = new_size / block_size;
    auto* v = state.v();

    // Resolve the exact complex phase of the observable (includes Y-count)
    std::complex<double> base_phase = resolve_sign(state, instr);

    // Precompute phase factors for branchless inner loop (AVX-friendly)
    std::complex<double> phase_factors[2] = {base_phase, -base_phase};

    // Pass 1: Gather L2 norms with correct phase-aware interference
    double prob_0 = 0.0, prob_1 = 0.0;

    for (uint64_t b = 0; b < num_blocks; ++b) {
        uint64_t src_start = b * (block_size * 2);
        for (uint64_t i = 0; i < block_size; ++i) {
            uint64_t alpha = src_start + i;
            uint64_t beta = alpha ^ x_mask;

            // Commutation parity for beta index - branchless lookup
            int parity_b = compute_sign_parity(static_cast<uint32_t>(beta), instr.commutation_mask);
            std::complex<double> term = v[beta] * phase_factors[parity_b];

            prob_0 += std::norm(v[alpha] + term);
            prob_1 += std::norm(v[alpha] - term);
        }
    }

    double total = prob_0 + prob_1;
    if (total < 1e-30) {
        return 0;
    }

    double r = state.random_double();
    uint8_t internal_outcome = (r < prob_0 / total) ? 0 : 1;

    double norm = 1.0 / std::sqrt(internal_outcome ? prob_1 : prob_0);

    // Pass 2: Single-pass butterfly and compaction with correct phase.
    // Memory safety: dst_start ≤ src_start, so we never overwrite before reading.
    // Hoist outcome branch outside loop entirely for CPU branch predictor.
    if (internal_outcome == 0) {
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t dst_start = b * block_size;
            uint64_t src_start = b * (block_size * 2);
            for (uint64_t i = 0; i < block_size; ++i) {
                uint64_t alpha = src_start + i;
                uint64_t beta = alpha ^ x_mask;
                int parity_b =
                    compute_sign_parity(static_cast<uint32_t>(beta), instr.commutation_mask);
                v[dst_start + i] = (v[alpha] + v[beta] * phase_factors[parity_b]) * norm;
            }
        }
    } else {
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t dst_start = b * block_size;
            uint64_t src_start = b * (block_size * 2);
            for (uint64_t i = 0; i < block_size; ++i) {
                uint64_t alpha = src_start + i;
                uint64_t beta = alpha ^ x_mask;
                int parity_b =
                    compute_sign_parity(static_cast<uint32_t>(beta), instr.commutation_mask);
                v[dst_start + i] = (v[alpha] - v[beta] * phase_factors[parity_b]) * norm;
            }
        }
    }
    current_rank--;
    return internal_outcome;
}

// OP_MEASURE_FILTER: Commuting measurement with β=0 but nonzero commutation.
// Observable commutes with tableau but has sign dependence on GF(2) index.
// Returns the sampled outcome.
uint8_t op_measure_filter(SchrodingerState& state, const Instruction& instr,
                          uint32_t current_rank) {
    uint64_t size = 1ULL << current_rank;  // 64-bit to avoid UB
    auto* v = state.v();
    double prob_0 = 0.0, prob_1 = 0.0;

    for (uint64_t alpha = 0; alpha < size; ++alpha) {
        double norm_sq = std::norm(v[alpha]);
        if (compute_sign_parity(static_cast<uint32_t>(alpha), instr.commutation_mask)) {
            prob_1 += norm_sq;
        } else {
            prob_0 += norm_sq;
        }
    }

    double total = prob_0 + prob_1;
    if (total < 1e-30) {
        return 0;
    }

    // Sample internal outcome
    double r = state.random_double();
    uint8_t internal_outcome = (r < prob_0 / total) ? 0 : 1;

    // Apply Pauli frame parity to get the physical outcome
    uint64_t anti_comm = (state.destab_signs & instr.branch.stab_mask) ^
                         (state.stab_signs & instr.branch.destab_mask);
    int frame_parity = std::popcount(anti_comm) & 1;

    uint8_t outcome = internal_outcome ^ static_cast<uint8_t>(frame_parity) ^ instr.ag_ref_outcome;

    // Zero out amplitudes with wrong parity and renormalize
    // Use branchless multiplier: parity XOR outcome == 0 means keep, == 1 means zero
    double norm_factor = 1.0 / std::sqrt(internal_outcome ? prob_1 : prob_0);
    double multipliers[2] = {norm_factor, 0.0};
    for (uint64_t alpha = 0; alpha < size; ++alpha) {
        int parity = compute_sign_parity(static_cast<uint32_t>(alpha), instr.commutation_mask);
        v[alpha] *= multipliers[parity ^ internal_outcome];  // Branchless
    }
    return outcome;
}

// OP_MEASURE_DETERMINISTIC: Fully deterministic measurement (β=0, comm=0).
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

    // Full Matrix Transformation of the Error Frame
    // The AG matrix maps old error coefficients -> new error coefficients.
    // We encode the error frame as a PauliString where xs = destab_signs
    // and zs = stab_signs, then apply the tableau transformation.
    const auto& mat = pool.ag_matrices[instr.meta.payload_idx];

    stim::PauliString<kStimWidth> err_frame(mat.num_qubits);
    err_frame.xs.u64[0] = state.destab_signs;
    err_frame.zs.u64[0] = state.stab_signs;

    // Apply the exact change of basis
    err_frame = mat(err_frame);

    uint64_t new_destab = err_frame.xs.u64[0];
    uint64_t new_stab = err_frame.zs.u64[0];

    // Inject Measurement Divergence
    // When outcome diverges from reference, we apply the new destabilizer D'_p.
    // The destabilizer anti-commutes with the measured observable (now stabilizer S'_p),
    // effectively flipping the measurement outcome.
    // D'_p is an X-type Pauli, so we XOR into destab_signs to preserve any
    // existing error that was transformed through the AG pivot matrix.
    uint32_t p = instr.meta.ag_stab_slot;
    new_destab ^= (static_cast<uint64_t>(divergence) << p);

    state.destab_signs = new_destab;
    state.stab_signs = new_stab;

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

void execute(const CompiledModule& program, SchrodingerState& state) {
    uint32_t current_rank = 0;  // Start with rank 0 (single amplitude)
    uint32_t meas_idx = 0;
    uint32_t det_idx = 0;
    uint8_t last_outcome = 0;  // Tracks outcome for hidden measurements / reset

    const auto& schedule = program.constant_pool.noise_schedule;
    const auto& det_targets = program.constant_pool.detector_targets;
    const auto& obs_targets = program.constant_pool.observable_targets;
    size_t next_noise_idx = 0;

    // Loop to <= bytecode.size() to process trailing noise at circuit end.
    // Noise scheduled at pc == bytecode.size() applies after all instructions.
    for (uint32_t pc = 0; pc <= program.bytecode.size(); ++pc) {
        // 1. Process all noise scheduled for this PC (gap sampling)
        while (next_noise_idx < schedule.size() && pc == schedule[next_noise_idx].pc) {
            const auto& site = schedule[next_noise_idx];
            double r = state.random_double();
            if (r < site.total_probability) {
                // Single-draw roulette wheel: use same random number for channel selection
                double cum_p = 0.0;
                for (const auto& ch : site.channels) {
                    cum_p += ch.prob;
                    if (r < cum_p) {
                        state.destab_signs ^= ch.destab_mask;
                        state.stab_signs ^= ch.stab_mask;
                        break;
                    }
                }
            }
            next_noise_idx++;
        }

        // 2. Break if we've processed all bytecode
        if (pc == program.bytecode.size()) {
            break;
        }

        // 3. Execute the instruction at this PC
        const auto& instr = program.bytecode[pc];
        switch (instr.opcode) {
            case Opcode::OP_BRANCH:
                op_branch(state, instr, current_rank);
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
                break;
            }

            case Opcode::OP_MEASURE_FILTER: {
                last_outcome = op_measure_filter(state, instr, current_rank);
                bool is_hidden = (instr.flags & Instruction::FLAG_HIDDEN) != 0;
                if (!is_hidden) {
                    state.meas_record[meas_idx++] = last_outcome;
                }
                break;
            }

            case Opcode::OP_MEASURE_DETERMINISTIC: {
                last_outcome = op_measure_deterministic(state, instr);
                bool is_hidden = (instr.flags & Instruction::FLAG_HIDDEN) != 0;
                if (!is_hidden) {
                    state.meas_record[meas_idx++] = last_outcome;
                }
                break;
            }

            case Opcode::OP_AG_PIVOT: {
                // Use last_outcome when FLAG_REUSE_OUTCOME is set (follows MEASURE_MERGE)
                last_outcome = op_ag_pivot(state, instr, program.constant_pool, last_outcome);
                bool is_hidden = (instr.flags & Instruction::FLAG_HIDDEN) != 0;
                bool reuse_outcome = (instr.flags & Instruction::FLAG_REUSE_OUTCOME) != 0;
                // Only write to meas_record if not hidden and not reusing (standalone pivot)
                if (!is_hidden && !reuse_outcome) {
                    state.meas_record[meas_idx++] = last_outcome;
                }
                break;
            }

            case Opcode::OP_CONDITIONAL:
                op_conditional(state, instr, last_outcome);
                break;

            case Opcode::OP_READOUT_NOISE:
                // Classical bit-flip on measurement result
                if (state.random_double() < instr.readout.prob) {
                    state.meas_record[instr.readout.meas_idx] ^= 1;
                }
                break;

            case Opcode::OP_DETECTOR: {
                // Compute parity of all referenced measurement bits
                const auto& targets = det_targets[instr.detector.target_idx];
                uint8_t parity = 0;
                for (uint32_t idx : targets) {
                    parity ^= state.meas_record[idx];
                }
                state.det_record[det_idx++] = parity;
                break;
            }

            case Opcode::OP_OBSERVABLE: {
                // XOR parity into the observable accumulator
                const auto& targets = obs_targets[instr.observable.target_idx];
                uint8_t parity = 0;
                for (uint32_t idx : targets) {
                    parity ^= state.meas_record[idx];
                }
                state.obs_record[instr.observable.obs_idx] ^= parity;
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
    }
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
    // we compute the reference state U|0⟩ ONCE and then apply Pauli operators
    // to map each branch's initial X-string through the final tableau.

    // 1. Compute reference statevector U|0⟩ exactly once to lock the global phase frame
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
    // For each branch alpha, the state is U|x_initial⟩ = U X^x U† U|0⟩ = P' |ref⟩
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
        // The branch state X^x |0⟩ becomes E X^x |0⟩ = X^{x XOR destab_signs} Z^{stab_signs} |0⟩
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

        // Map P through the final tableau to get P' = U P U†
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
        // P'|k⟩ = (±1)|k XOR px⟩ where the sign depends on k·pz parity
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
