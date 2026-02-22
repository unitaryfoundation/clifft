#include "ucc/svm/svm.h"

#include <bit>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

namespace ucc {

// =============================================================================
// SchrodingerState Implementation
// =============================================================================

SchrodingerState::SchrodingerState(uint32_t peak_rank, uint32_t num_measurements, uint64_t seed)
    : peak_rank_(peak_rank), rng_(seed) {
    // Pre-allocate measurement record first (may throw, but no cleanup needed)
    meas_record.resize(num_measurements, 0);

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
    std::complex<double> rel_weight =
        instr.is_dagger ? std::complex<double>(0.0, -kTanPi8) : std::complex<double>(0.0, kTanPi8);

    // Precompute the two possible phase factors (branchless array lookup)
    std::complex<double> factors[2] = {rel_weight * base_phase, -rel_weight * base_phase};

    auto* v = state.v();
    for (uint64_t alpha = 0; alpha < old_size; ++alpha) {
        uint64_t new_idx = alpha + old_size;  // Equivalent to alpha | (1 << new_bit)
        int branch_parity =
            compute_sign_parity(static_cast<uint32_t>(new_idx), instr.commutation_mask);

        // Zero FLOPs on identity branch - just write spawned branch
        v[new_idx] = v[alpha] * factors[branch_parity];
    }
}

// OP_COLLIDE: In-place butterfly on existing dimension.
// Uses dominant term factoring: v' = v + i*tan(π/8)*Z*partner.
// Block-based iteration for cache-friendly access.
void op_collide(SchrodingerState& state, const Instruction& instr, uint32_t current_rank) {
    assert(instr.branch.x_mask != 0 && "OP_COLLIDE requires nonzero x_mask");

    std::complex<double> base_phase = resolve_sign(state, instr);
    std::complex<double> rel_weight =
        instr.is_dagger ? std::complex<double>(0.0, -kTanPi8) : std::complex<double>(0.0, kTanPi8);

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
void op_scalar_phase(SchrodingerState& state, const Instruction& instr, uint32_t current_rank) {
    std::complex<double> base_phase = resolve_sign(state, instr);
    std::complex<double> i_tan =
        instr.is_dagger ? std::complex<double>(0.0, -kTanPi8) : std::complex<double>(0.0, kTanPi8);

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
// Uses projective butterfly: (I ± M)/2 requires interference v[α] ± v[α ⊕ β].
// Block-based iteration with single-pass butterfly+compaction.
void op_measure_merge(SchrodingerState& state, const Instruction& instr, uint32_t& current_rank,
                      uint32_t meas_idx) {
    // Defensive assertions to catch backend logic errors
    assert(current_rank >= 1 && "MEASURE_MERGE requires at least one dimension");
    assert(instr.branch.x_mask != 0 && "MEASURE_MERGE requires nonzero x_mask");

    uint32_t bit_index = instr.branch.bit_index;
    uint32_t x_mask = instr.branch.x_mask;

    uint64_t block_size = 1ULL << bit_index;
    uint64_t new_size = 1ULL << (current_rank - 1);
    uint64_t num_blocks = new_size / block_size;
    auto* v = state.v();

    // Pass 1: Gather L2 norms branchlessly
    double prob_0 = 0.0, prob_1 = 0.0;

    for (uint64_t b = 0; b < num_blocks; ++b) {
        uint64_t src_start = b * (block_size * 2);
        for (uint64_t i = 0; i < block_size; ++i) {
            uint64_t alpha = src_start + i;
            uint64_t beta = alpha ^ x_mask;
            prob_0 += std::norm(v[alpha] + v[beta]);
            prob_1 += std::norm(v[alpha] - v[beta]);
        }
    }

    double total = prob_0 + prob_1;
    if (total < 1e-30) {
        state.meas_record[meas_idx] = 0;
        return;
    }

    double r = state.random_double();
    uint8_t internal_outcome = (r < prob_0 / total) ? 0 : 1;

    uint64_t anti_comm = (state.destab_signs & instr.branch.stab_mask) ^
                         (state.stab_signs & instr.branch.destab_mask);
    int frame_parity = std::popcount(anti_comm) & 1;
    state.meas_record[meas_idx] =
        internal_outcome ^ static_cast<uint8_t>(frame_parity) ^ instr.ag_ref_outcome;

    double norm = 1.0 / std::sqrt(internal_outcome ? prob_1 : prob_0);

    // Pass 2: Single-pass butterfly and compaction.
    // Memory safety: dst_start ≤ src_start, so we never overwrite before reading.
    for (uint64_t b = 0; b < num_blocks; ++b) {
        uint64_t dst_start = b * block_size;
        uint64_t src_start = b * (block_size * 2);

        for (uint64_t i = 0; i < block_size; ++i) {
            uint64_t alpha = src_start + i;
            uint64_t beta = alpha ^ x_mask;

            std::complex<double> merged =
                internal_outcome ? (v[alpha] - v[beta]) : (v[alpha] + v[beta]);
            v[dst_start + i] = merged * norm;
        }
    }
    current_rank--;
}

// OP_MEASURE_FILTER: Commuting measurement with β=0 but nonzero commutation.
// Observable commutes with tableau but has sign dependence on GF(2) index.
void op_measure_filter(SchrodingerState& state, const Instruction& instr, uint32_t current_rank,
                       uint32_t meas_idx) {
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
        state.meas_record[meas_idx] = 0;
        return;
    }

    // Sample internal outcome
    double r = state.random_double();
    uint8_t internal_outcome = (r < prob_0 / total) ? 0 : 1;

    // Apply Pauli frame parity
    uint64_t anti_comm = (state.destab_signs & instr.branch.stab_mask) ^
                         (state.stab_signs & instr.branch.destab_mask);
    int frame_parity = std::popcount(anti_comm) & 1;

    state.meas_record[meas_idx] =
        internal_outcome ^ static_cast<uint8_t>(frame_parity) ^ instr.ag_ref_outcome;

    // Zero out amplitudes with wrong parity and renormalize
    // Use branchless multiplier: parity XOR outcome == 0 means keep, == 1 means zero
    double norm_factor = 1.0 / std::sqrt(internal_outcome ? prob_1 : prob_0);
    double multipliers[2] = {norm_factor, 0.0};
    for (uint64_t alpha = 0; alpha < size; ++alpha) {
        int parity = compute_sign_parity(static_cast<uint32_t>(alpha), instr.commutation_mask);
        v[alpha] *= multipliers[parity ^ internal_outcome];  // Branchless
    }
}

// OP_MEASURE_DETERMINISTIC: Fully deterministic measurement (β=0, comm=0).
// Outcome determined purely by Pauli frame signs.
void op_measure_deterministic(SchrodingerState& state, const Instruction& instr,
                              uint32_t meas_idx) {
    // The measurement observable is P = X^destab_mask * Z^stab_mask.
    // The Pauli frame has accumulated X^destab_signs * Z^stab_signs.
    // Anti-commutation: X anti-commutes with Z, so frame X bits that overlap
    // with observable Z bits (and vice versa) contribute to the sign.
    //
    // Commutator parity: (frame_X & obs_Z) XOR (frame_Z & obs_X)
    uint64_t anti_comm = (state.destab_signs & instr.branch.stab_mask) ^
                         (state.stab_signs & instr.branch.destab_mask);
    int parity = std::popcount(anti_comm) & 1;

    uint8_t outcome = static_cast<uint8_t>(parity) ^ instr.ag_ref_outcome;
    state.meas_record[meas_idx] = outcome;
}

// OP_AG_PIVOT: Aaronson-Gottesman pivot for anti-commuting measurements.
// When reuse_outcome=true, reuse outcome from preceding MERGE.
// Otherwise, sample fresh 50/50 outcome.
//
// Transforms the error frame through the AG pivot matrix (mapping old logical
// coordinates to new logical coordinates), then injects the measurement
// divergence by XORing the pivot slot into destab_signs.
void op_ag_pivot(SchrodingerState& state, const Instruction& instr, const ConstantPool& pool,
                 uint32_t meas_idx, bool use_prev_outcome) {
    uint8_t divergence;
    if (use_prev_outcome) {
        divergence = state.meas_record[meas_idx] ^ instr.ag_ref_outcome;
    } else {
        double r = state.random_double();
        divergence = (r < 0.5) ? 0 : 1;
        state.meas_record[meas_idx] = divergence ^ instr.ag_ref_outcome;
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
}

// OP_CONDITIONAL: Apply Pauli correction if measurement was 1.
void op_conditional(SchrodingerState& state, const Instruction& instr) {
    uint32_t meas_idx = instr.meta.controlling_meas;
    if (state.meas_record[meas_idx] == 1) {
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

    for (const auto& instr : program.bytecode) {
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

            case Opcode::OP_MEASURE_MERGE:
                op_measure_merge(state, instr, current_rank, meas_idx++);
                break;

            case Opcode::OP_MEASURE_FILTER:
                op_measure_filter(state, instr, current_rank, meas_idx++);
                break;

            case Opcode::OP_MEASURE_DETERMINISTIC:
                op_measure_deterministic(state, instr, meas_idx++);
                break;

            case Opcode::OP_AG_PIVOT:
                // Double-sample fix: reuse_outcome=true means reuse previous outcome
                if (instr.reuse_outcome) {
                    // Follows a MERGE: do not increment meas_idx, use previous outcome.
                    // Backend should never emit reuse_outcome=true before any measurement.
                    assert(meas_idx > 0 && "OP_AG_PIVOT reuse_outcome requires prior measurement");
                    op_ag_pivot(state, instr, program.constant_pool, meas_idx - 1, true);
                } else {
                    // Standalone AG_PIVOT: sample fresh and increment
                    op_ag_pivot(state, instr, program.constant_pool, meas_idx++, false);
                }
                break;

            case Opcode::OP_CONDITIONAL:
                op_conditional(state, instr);
                break;

            // Future opcodes - not implemented in MVP
            case Opcode::OP_BRANCH_LCU:
            case Opcode::OP_COLLIDE_LCU:
            case Opcode::OP_SCALAR_PHASE_LCU:
            case Opcode::OP_INDEX_CNOT:
            case Opcode::OP_DETECTOR:
            case Opcode::OP_POSTSELECT:
                break;
        }
    }
}

std::vector<uint8_t> sample(const CompiledModule& program, uint32_t shots, uint64_t seed) {
    if (shots == 0) {
        return {};
    }

    uint32_t num_meas = program.num_measurements;
    std::vector<uint8_t> results(static_cast<size_t>(shots) * num_meas);

    // Create state once, reuse for all shots
    SchrodingerState state(program.peak_rank, num_meas, seed);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0) {
            // Reset for next shot with deterministic seed progression
            state.reset(seed + shot);
        }

        execute(program, state);

        // Copy measurement record to results
        std::copy(state.meas_record.begin(), state.meas_record.end(),
                  results.begin() + static_cast<ptrdiff_t>(shot * num_meas));
    }

    return results;
}

}  // namespace ucc
