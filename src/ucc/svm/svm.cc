#include "ucc/svm/svm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ucc {

// =============================================================================
// Bit helpers for stim::bitword<kStimWidth>
//
// These use only the portable bitword API (shift, bitwise ops, popcount) so
// they work at any kStimWidth (64, 128, 256) without touching internal fields.
// =============================================================================

namespace {

using Bitword = stim::bitword<kStimWidth>;

inline Bitword single_bit(uint16_t idx) {
    return Bitword(uint64_t{1}) << idx;
}

inline bool bit_get(Bitword w, uint16_t idx) {
    return bool((w >> idx) & uint64_t{1});
}

inline void bit_set(Bitword& w, uint16_t idx, bool v) {
    if (bit_get(w, idx) != v) {
        w ^= single_bit(idx);
    }
}

inline void bit_xor(Bitword& w, uint16_t idx, bool v) {
    if (v) {
        w ^= single_bit(idx);
    }
}

inline void bit_swap(Bitword& w1, uint16_t i1, Bitword& w2, uint16_t i2) {
    bool tmp = bit_get(w1, i1);
    bit_set(w1, i1, bit_get(w2, i2));
    bit_set(w2, i2, tmp);
}

// Phase constants
// Use explicit constant instead of non-standard M_SQRT1_2 (POSIX, not in C++ standard).
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr std::complex<double> kI{0.0, 1.0};
constexpr std::complex<double> kMinusI{0.0, -1.0};
constexpr std::complex<double> kExpIPiOver4{kInvSqrt2, kInvSqrt2};        // e^{i*pi/4}
constexpr std::complex<double> kExpMinusIPiOver4{kInvSqrt2, -kInvSqrt2};  // e^{-i*pi/4}

}  // namespace

// =============================================================================
// SchrodingerState Implementation
// =============================================================================

SchrodingerState::SchrodingerState(uint32_t peak_rank, uint32_t num_measurements,
                                   uint32_t num_detectors, uint32_t num_observables, uint64_t seed)
    : peak_rank_(peak_rank), rng_(seed) {
    meas_record.resize(num_measurements, 0);
    det_record.resize(num_detectors, 0);
    obs_record.resize(num_observables, 0);

    // Allocate 2^peak_rank complex numbers, 64-byte aligned for AVX
    array_size_ = 1ULL << peak_rank;
    size_t bytes = array_size_ * sizeof(std::complex<double>);
    size_t aligned_bytes = (bytes + 63) & ~63ULL;
    v_ = static_cast<std::complex<double>*>(std::aligned_alloc(64, aligned_bytes));
    if (!v_) {
        throw std::bad_alloc();
    }

    // Initialize to |0...0>: coefficient 1 at index 0
    for (uint64_t i = 0; i < array_size_; ++i) {
        v_[i] = {0.0, 0.0};
    }
    v_[0] = {1.0, 0.0};
}

SchrodingerState::~SchrodingerState() {
    std::free(v_);
}

SchrodingerState::SchrodingerState(SchrodingerState&& other) noexcept
    : p_x(other.p_x),
      p_z(other.p_z),
      gamma(other.gamma),
      active_k(other.active_k),
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
        p_x = other.p_x;
        p_z = other.p_z;
        gamma = other.gamma;
        active_k = other.active_k;
        meas_record = std::move(other.meas_record);
        det_record = std::move(other.det_record);
        obs_record = std::move(other.obs_record);
        other.v_ = nullptr;
        other.array_size_ = 0;
    }
    return *this;
}

void SchrodingerState::reset(uint64_t seed) {
    // Only zero the previously-active portion of the buffer, not the full
    // 2^peak_rank allocation — avoids O(2^peak_rank) work when active_k is small.
    uint64_t active_size = (active_k > 0) ? (uint64_t{1} << active_k) : 1;
    for (uint64_t i = 0; i < active_size; ++i) {
        v_[i] = {0.0, 0.0};
    }
    v_[0] = {1.0, 0.0};
    p_x = 0;
    p_z = 0;
    gamma = {1.0, 0.0};
    active_k = 0;

    std::fill(meas_record.begin(), meas_record.end(), 0);
    std::fill(det_record.begin(), det_record.end(), 0);
    std::fill(obs_record.begin(), obs_record.end(), 0);

    rng_.seed(seed);
}

// =============================================================================
// Frame Opcode Handlers (Zero-cost: update P only, no array touch)
// =============================================================================

// CNOT on virtual axes (c=control, t=target): conjugates the Pauli frame.
// Heisenberg rules: X_c spreads to X_t (p_x[t] ^= p_x[c]),
// Z_t spreads back to Z_c (p_z[c] ^= p_z[t]). No phase.
static inline void exec_frame_cnot(SchrodingerState& state, uint16_t c, uint16_t t) {
    bool px_c = bit_get(state.p_x, c);
    bool pz_t = bit_get(state.p_z, t);
    bit_xor(state.p_x, t, px_c);
    bit_xor(state.p_z, c, pz_t);
}

// CZ on virtual axes (c, t): conjugates the Pauli frame.
// Heisenberg rules: X_c picks up Z_t (p_z[t] ^= p_x[c]) and vice versa.
// When both X bits are set, CZ anticommutes: gamma *= -1.
static inline void exec_frame_cz(SchrodingerState& state, uint16_t c, uint16_t t) {
    bool px_c = bit_get(state.p_x, c);
    bool px_t = bit_get(state.p_x, t);
    if (px_c && px_t) {
        state.gamma = -state.gamma;
    }
    bit_xor(state.p_z, t, px_c);
    bit_xor(state.p_z, c, px_t);
}

// Hadamard on virtual axis v: conjugates X <-> Z in the Pauli frame.
// Swaps p_x[v] and p_z[v]. If both bits were set (Y Pauli), H*Y*H = -Y,
// so gamma *= -1.
static inline void exec_frame_h(SchrodingerState& state, uint16_t v) {
    bool px = bit_get(state.p_x, v);
    bool pz = bit_get(state.p_z, v);
    if (px && pz) {
        state.gamma = -state.gamma;
    }
    bit_set(state.p_x, v, pz);
    bit_set(state.p_z, v, px);
}

// S gate on virtual axis v: conjugates X -> Y in the Pauli frame.
// S*X*S_dag = Y = iXZ, so when p_x[v] is set: gamma *= i, p_z[v] ^= 1.
// Z commutes with S, so no change when only p_z is set.
static inline void exec_frame_s(SchrodingerState& state, uint16_t v) {
    bool px = bit_get(state.p_x, v);
    if (px) {
        state.gamma *= kI;
    }
    bit_xor(state.p_z, v, px);
}

// SWAP on virtual axes (a, b): exchanges Pauli frame bits for both axes.
// Swaps p_x[a] <-> p_x[b] and p_z[a] <-> p_z[b]. No phase.
static inline void exec_frame_swap(SchrodingerState& state, uint16_t a, uint16_t b) {
    bit_swap(state.p_x, a, state.p_x, b);
    bit_swap(state.p_z, a, state.p_z, b);
}

// =============================================================================
// Array Opcode Handlers (update P AND loop over v[])
// =============================================================================

// CNOT on active axes (c=control, t=target): permutes the amplitude array
// by swapping v[i] <-> v[j] for each pair where bit c=1 and bits c,t differ,
// then updates the Pauli frame identically to FRAME_CNOT.
static void exec_array_cnot(SchrodingerState& state, uint16_t c, uint16_t t) {
    assert(c < state.active_k && c < 64 && "ARRAY_CNOT: control axis out of range");
    assert(t < state.active_k && t < 64 && "ARRAY_CNOT: target axis out of range");
    uint64_t size = 1ULL << state.active_k;
    uint64_t c_bit = 1ULL << c;
    uint64_t t_bit = 1ULL << t;
    auto* v = state.v();

    for (uint64_t i = 0; i < size; ++i) {
        // Only process indices where control bit is 1 and target bit is 0
        if ((i & c_bit) && !(i & t_bit)) {
            uint64_t j = i | t_bit;  // Same index but with target bit set
            std::swap(v[i], v[j]);
        }
    }

    // Frame update (same as OP_FRAME_CNOT)
    exec_frame_cnot(state, c, t);
}

// CZ on active axes (c, t): applies diag(1,1,1,-1) in the computational basis.
// Negates v[i] for every index where both bit c and bit t are 1,
// then updates the Pauli frame identically to FRAME_CZ.
static void exec_array_cz(SchrodingerState& state, uint16_t c, uint16_t t) {
    assert(c < state.active_k && c < 64 && "ARRAY_CZ: control axis out of range");
    assert(t < state.active_k && t < 64 && "ARRAY_CZ: target axis out of range");
    uint64_t size = 1ULL << state.active_k;
    uint64_t c_bit = 1ULL << c;
    uint64_t t_bit = 1ULL << t;
    auto* v = state.v();

    for (uint64_t i = 0; i < size; ++i) {
        if ((i & c_bit) && (i & t_bit)) {
            v[i] = -v[i];
        }
    }

    // Frame update (same as OP_FRAME_CZ)
    exec_frame_cz(state, c, t);
}

// SWAP on active axes (a, b): permutes the amplitude array by exchanging
// v[i] <-> v[j] for all index pairs that differ only in bits a and b,
// then updates the Pauli frame identically to FRAME_SWAP.
static void exec_array_swap(SchrodingerState& state, uint16_t a, uint16_t b) {
    assert(a < state.active_k && a < 64 && "ARRAY_SWAP: axis a out of range");
    assert(b < state.active_k && b < 64 && "ARRAY_SWAP: axis b out of range");
    uint64_t size = 1ULL << state.active_k;
    uint64_t a_bit = 1ULL << a;
    uint64_t b_bit = 1ULL << b;
    auto* v = state.v();

    for (uint64_t i = 0; i < size; ++i) {
        // Only process where bit a=0, bit b=1 (to avoid double-swapping)
        if (!(i & a_bit) && (i & b_bit)) {
            uint64_t j = (i & ~b_bit) | a_bit;  // Flip a on, b off
            std::swap(v[i], v[j]);
        }
    }

    // Frame update (same as OP_FRAME_SWAP)
    exec_frame_swap(state, a, b);
}

// =============================================================================
// Expansion & Phase Opcodes
// =============================================================================

// EXPAND axis v: promotes a dormant qubit to active by doubling the array.
// v must equal active_k (the next available axis). Copies v[i] to v[i + 2^k]
// for all i < 2^k, producing |phi> tensor |+> on the new axis.
// gamma /= sqrt(2) to maintain normalization.
static void exec_expand(SchrodingerState& state, uint16_t v) {
    assert(v == state.active_k && "EXPAND must target the next dormant axis");
    uint64_t old_size = 1ULL << state.active_k;
    auto* arr = state.v();

    // Duplicate: v[i | (1 << k)] = v[i] for all i < 2^k
    uint64_t half = old_size;
    for (uint64_t i = 0; i < half; ++i) {
        arr[i + half] = arr[i];
    }

    state.gamma /= std::sqrt(2.0);
    state.active_k++;
}

// T gate (pi/4 Z-rotation) on active axis v: applies diag(1, e^{i*pi/4})
// to array indices where bit v is set. If p_x[v]=1 (T anticommutes with X),
// the array gets T_dag instead and gamma absorbs e^{i*pi/4} to preserve
// the factored state identity.
static void exec_phase_t(SchrodingerState& state, uint16_t v) {
    assert(v < state.active_k && v < 64 && "PHASE_T: axis out of range");
    uint64_t size = 1ULL << state.active_k;
    uint64_t v_bit = 1ULL << v;
    auto* arr = state.v();
    bool px = bit_get(state.p_x, v);

    if (px) {
        // Anti-commutes: apply T_dag to array, multiply gamma by e^{i*pi/4}
        for (uint64_t i = 0; i < size; ++i) {
            if (i & v_bit) {
                arr[i] *= kExpMinusIPiOver4;
            }
        }
        state.gamma *= kExpIPiOver4;
    } else {
        // Commutes: apply T to array
        for (uint64_t i = 0; i < size; ++i) {
            if (i & v_bit) {
                arr[i] *= kExpIPiOver4;
            }
        }
    }
}

// T_dag gate (-pi/4 Z-rotation) on active axis v: applies diag(1, e^{-i*pi/4}).
// Mirror of T: if p_x[v]=1, the array gets T instead and gamma absorbs
// e^{-i*pi/4}.
static void exec_phase_t_dag(SchrodingerState& state, uint16_t v) {
    assert(v < state.active_k && v < 64 && "PHASE_T_DAG: axis out of range");
    uint64_t size = 1ULL << state.active_k;
    uint64_t v_bit = 1ULL << v;
    auto* arr = state.v();
    bool px = bit_get(state.p_x, v);

    if (px) {
        // Anti-commutes: apply T to array, multiply gamma by e^{-i*pi/4}
        for (uint64_t i = 0; i < size; ++i) {
            if (i & v_bit) {
                arr[i] *= kExpIPiOver4;
            }
        }
        state.gamma *= kExpMinusIPiOver4;
    } else {
        // Commutes: apply T_dag to array
        for (uint64_t i = 0; i < size; ++i) {
            if (i & v_bit) {
                arr[i] *= kExpMinusIPiOver4;
            }
        }
    }
}

// =============================================================================
// Measurement Opcodes
// =============================================================================

// Dormant-static measurement on axis v: the qubit is not in the amplitude
// array, and its Z-basis eigenvalue is deterministic. The physical outcome
// is simply p_x[v] (the X bit of the Pauli frame).
static void exec_meas_dormant_static(SchrodingerState& state, uint16_t v, uint32_t classical_idx) {
    uint8_t outcome = bit_get(state.p_x, v) ? 1 : 0;
    state.meas_record[classical_idx] = outcome;
}

// Dormant-random measurement on axis v: the qubit is dormant but its
// outcome is uniformly random (e.g. X-basis eigenstate measured in Z).
// Samples m in {0,1}, extracts phase (-1)^(p_x[v]*m), and resets the
// frame to anchor the post-measurement computational state.
static void exec_meas_dormant_random(SchrodingerState& state, uint16_t v, uint32_t classical_idx) {
    uint8_t m = state.random_double() < 0.5 ? 0 : 1;

    // Phase extraction: (-1)^(p_x[v] * m)
    if (bit_get(state.p_x, v) && m) {
        state.gamma = -state.gamma;
    }

    // Frame reset: anchor the computational zero state
    bit_set(state.p_x, v, m);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = m;
}

// Active-diagonal measurement on axis v (must be active_k-1): the qubit is
// in the amplitude array and the measurement basis is diagonal (Z-like).
// Samples branch b from probabilities of the upper/lower halves of v[],
// compacts the array by discarding the unchosen half, and extracts phase
// (-1)^(p_z[v]*b). Physical outcome m = b XOR p_x[v].
static void exec_meas_active_diagonal(SchrodingerState& state, uint16_t v, uint32_t classical_idx) {
    assert(v == state.active_k - 1 && "Active diagonal measurement must target axis k-1");

    uint64_t half = 1ULL << (state.active_k - 1);
    auto* arr = state.v();
    bool px_v = bit_get(state.p_x, v);
    bool pz_v = bit_get(state.p_z, v);

    // Compute probability of array branch b=0 (bit v = 0) and b=1 (bit v = 1)
    double prob_b0 = 0.0;
    double prob_b1 = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        prob_b0 += std::norm(arr[i]);         // bit v = 0
        prob_b1 += std::norm(arr[i + half]);  // bit v = 1
    }
    double total = prob_b0 + prob_b1;
    assert(total > 0.0 && "Active diagonal measurement on zero-norm state");

    // Sample abstract branch b. Handle deterministic cases explicitly to avoid
    // dividing by a zero-probability branch due to floating-point edge cases.
    uint8_t b;
    if (prob_b1 <= 0.0) {
        b = 0;
    } else if (prob_b0 <= 0.0) {
        b = 1;
    } else {
        double rand = state.random_double();
        b = (rand * total < prob_b0) ? 0 : 1;
    }

    // Physical outcome m = b XOR p_x[v]
    uint8_t m = b ^ static_cast<uint8_t>(px_v);

    // Deferred normalization
    double prob_b = (b == 0) ? prob_b0 : prob_b1;
    assert(prob_b > 0.0);
    state.gamma /= std::sqrt(prob_b);

    // Phase extraction: (-1)^(p_z[v] * b) when b=1
    if (b == 1 && pz_v) {
        state.gamma = -state.gamma;
    }

    // Keep chosen branch, compact array
    if (b == 0) {
        // Keep lower half (already in place)
    } else {
        // Move upper half down
        for (uint64_t i = 0; i < half; ++i) {
            arr[i] = arr[i + half];
        }
    }

    // Zero out upper half (for safety)
    for (uint64_t i = half; i < 2 * half; ++i) {
        arr[i] = {0.0, 0.0};
    }

    state.active_k--;

    // Frame reset
    bit_set(state.p_x, v, m);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = m;
}

// Active-interfere measurement on axis v (must be active_k-1): the qubit is
// in the amplitude array and the measurement basis is off-diagonal (X-like).
// Computes |+> and |-> branch probabilities by summing |v[i] +/- v[i+half]|^2,
// folds the array (add or subtract), and normalizes. Physical outcome
// m = b_x XOR p_z[v], with phase extraction (-1)^(p_x[v]*m).
static void exec_meas_active_interfere(SchrodingerState& state, uint16_t v,
                                       uint32_t classical_idx) {
    assert(v == state.active_k - 1 && "Active interfere measurement must target axis k-1");

    uint64_t half = 1ULL << (state.active_k - 1);
    auto* arr = state.v();
    bool px_v = bit_get(state.p_x, v);
    bool pz_v = bit_get(state.p_z, v);

    // Compute X-basis probabilities:
    // b_x=0 (|+> branch): sum |v[i] + v[i+half]|^2
    // b_x=1 (|-> branch): sum |v[i] - v[i+half]|^2
    double prob_plus = 0.0;
    double prob_minus = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        auto sum = arr[i] + arr[i + half];
        auto diff = arr[i] - arr[i + half];
        prob_plus += std::norm(sum);
        prob_minus += std::norm(diff);
    }
    double total = prob_plus + prob_minus;
    assert(total > 0.0 && "Active interfere measurement on zero-norm state");

    // Sample X-basis branch. Handle deterministic cases explicitly to avoid
    // dividing by a zero-probability branch due to floating-point edge cases.
    uint8_t b_x;
    if (prob_minus <= 0.0) {
        b_x = 0;
    } else if (prob_plus <= 0.0) {
        b_x = 1;
    } else {
        double rand = state.random_double();
        b_x = (rand * total < prob_plus) ? 0 : 1;
    }

    // Physical outcome m = b_x XOR p_z[v]
    uint8_t m = b_x ^ static_cast<uint8_t>(pz_v);

    // Fold array: v'[i] = v[i] +/- v[i+half]
    // b_x=0 -> add, b_x=1 -> subtract
    for (uint64_t i = 0; i < half; ++i) {
        if (b_x == 0) {
            arr[i] = arr[i] + arr[i + half];
        } else {
            arr[i] = arr[i] - arr[i + half];
        }
    }

    // Deferred normalization (extra factor of 2 for X-basis)
    double prob_bx = (b_x == 0) ? prob_plus : prob_minus;
    assert(prob_bx > 0.0);
    state.gamma /= std::sqrt(2.0 * prob_bx);

    // Phase extraction: (-1)^(p_x[v] * m)
    if (px_v && m) {
        state.gamma = -state.gamma;
    }

    // Zero out upper half
    for (uint64_t i = half; i < 2 * half; ++i) {
        arr[i] = {0.0, 0.0};
    }

    state.active_k--;

    // Frame reset
    bit_set(state.p_x, v, m);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = m;
}

// =============================================================================
// Classical / Error Opcodes
// =============================================================================

// APPLY_PAULI: composes a PauliString error into the Pauli frame via XOR.
// Pauli frame composition P_current * P_err picks up a sign of
// (-1)^popcount(p_z & err_x) from Z*X = -X*Z anticommutation on
// overlapping qubits. The PauliString's own sign is also applied.
static void exec_apply_pauli(SchrodingerState& state, const ConstantPool& pool,
                             uint32_t cp_mask_idx) {
    assert(cp_mask_idx < pool.pauli_masks.size());
    const auto& ps = pool.pauli_masks[cp_mask_idx];
    assert(ps.num_qubits <= kStimWidth && "PauliString exceeds single bitword lane");

    // Extract x and z bitmasks from the PauliString (simd_bits -> bitword)
    Bitword err_x = ps.xs.ptr_simd[0];
    Bitword err_z = ps.zs.ptr_simd[0];

    // Phase: (-1)^popcount(current_z & err_x)
    if ((state.p_z & err_x).popcount() & 1) {
        state.gamma = -state.gamma;
    }

    // XOR into frame
    state.p_x ^= err_x;
    state.p_z ^= err_z;

    // Include the sign from the PauliString itself
    if (ps.sign) {
        state.gamma = -state.gamma;
    }
}

// DETECTOR: computes the XOR parity of a list of measurement record entries.
// The target list is stored in the constant pool; the result goes into det_record.
static void exec_detector(SchrodingerState& state, const ConstantPool& pool, uint32_t det_list_idx,
                          uint32_t classical_idx) {
    assert(det_list_idx < pool.detector_targets.size());
    const auto& targets = pool.detector_targets[det_list_idx];

    uint8_t parity = 0;
    for (uint32_t meas_idx : targets) {
        assert(meas_idx < state.meas_record.size());
        parity ^= state.meas_record[meas_idx];
    }

    assert(classical_idx < state.det_record.size());
    state.det_record[classical_idx] = parity;
}

// =============================================================================
// SVM Execution
// =============================================================================

void execute(const CompiledModule& program, SchrodingerState& state) {
    for (const auto& instr : program.bytecode) {
        switch (instr.opcode) {
            // Frame opcodes
            case Opcode::OP_FRAME_CNOT:
                exec_frame_cnot(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_FRAME_CZ:
                exec_frame_cz(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_FRAME_H:
                exec_frame_h(state, instr.axis_1);
                break;
            case Opcode::OP_FRAME_S:
                exec_frame_s(state, instr.axis_1);
                break;
            case Opcode::OP_FRAME_SWAP:
                exec_frame_swap(state, instr.axis_1, instr.axis_2);
                break;

            // Array opcodes
            case Opcode::OP_ARRAY_CNOT:
                exec_array_cnot(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_ARRAY_CZ:
                exec_array_cz(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_ARRAY_SWAP:
                exec_array_swap(state, instr.axis_1, instr.axis_2);
                break;

            // Expansion & Phase
            case Opcode::OP_EXPAND:
                exec_expand(state, instr.axis_1);
                break;
            case Opcode::OP_PHASE_T:
                exec_phase_t(state, instr.axis_1);
                break;
            case Opcode::OP_PHASE_T_DAG:
                exec_phase_t_dag(state, instr.axis_1);
                break;

            // Measurements
            case Opcode::OP_MEAS_DORMANT_STATIC:
                exec_meas_dormant_static(state, instr.axis_1, instr.classical.classical_idx);
                break;
            case Opcode::OP_MEAS_DORMANT_RANDOM:
                exec_meas_dormant_random(state, instr.axis_1, instr.classical.classical_idx);
                break;
            case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
                exec_meas_active_diagonal(state, instr.axis_1, instr.classical.classical_idx);
                break;
            case Opcode::OP_MEAS_ACTIVE_INTERFERE:
                exec_meas_active_interfere(state, instr.axis_1, instr.classical.classical_idx);
                break;

            // Classical / Errors
            case Opcode::OP_APPLY_PAULI:
                exec_apply_pauli(state, program.constant_pool, instr.pauli.cp_mask_idx);
                break;
            case Opcode::OP_DETECTOR:
                exec_detector(state, program.constant_pool, instr.pauli.cp_mask_idx,
                              instr.classical.classical_idx);
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

    SchrodingerState state(program.peak_rank, num_meas, num_det, num_obs, seed);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0) {
            state.reset(seed + shot);
        }

        execute(program, state);

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

// =============================================================================
// Statevector Expansion
// =============================================================================

std::vector<std::complex<double>> get_statevector(const SchrodingerState& state,
                                                  const ConstantPool& pool) {
    // TODO: Implement factored state expansion (Phase 2)
    // 1. Expand 2^k elements of |phi>_A into dense 2^n array
    // 2. Apply Pauli frame P (using p_x, p_z)
    // 3. Apply U_C (final_tableau) via Stim VectorSimulator
    // 4. Multiply by gamma
    (void)state;
    (void)pool;
    return {};
}

}  // namespace ucc
