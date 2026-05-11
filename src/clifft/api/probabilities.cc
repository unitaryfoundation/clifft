// Exact basis-state probability queries. The Gaussian elimination over the
// inverse Clifford tableau and the per-bitstring amplitude lookup are the
// implementation of the algorithm derived in docs/theory/probabilities.md.

#include "clifft/svm/svm.h"
#include "clifft/svm/svm_math.h"
#include "clifft/util/mask_view.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace clifft {
namespace {

using BasisMask = std::vector<uint64_t>;
using StabilizerRow = stim::PauliString<kStimWidth>;

[[nodiscard]] MaskView basis_mask_view(const BasisMask& mask) {
    return MaskView{std::span<const uint64_t>(mask)};
}

[[nodiscard]] MutableMaskView mutable_basis_mask_view(BasisMask& mask) {
    return MutableMaskView{std::span<uint64_t>(mask)};
}

void mask_copy_to(MutableMaskView dst, MaskView src) {
    if (dst.num_words() != src.num_words()) {
        throw std::invalid_argument("internal probability mask width mismatch");
    }
    std::copy(src.words.begin(), src.words.end(), dst.words.begin());
}

// During Gaussian elimination the sign of a stabilizer row separates into a
// static part plus a linear function of the queried physical bitstring x. The
// mask stores that linear function; binding a particular x substitutes it.
struct DynamicSignTerm {
    uint32_t bit = 0;
    bool static_sign = false;
    BasisMask sign_mask;
};

struct IdentityConstraint {
    bool static_sign = false;
    BasisMask sign_mask;
};

[[nodiscard]] size_t basis_word_count(uint32_t n) {
    return (static_cast<size_t>(n) + 63U) / 64U;
}

[[nodiscard]] BasisMask zero_basis_mask(uint32_t n) {
    return BasisMask(basis_word_count(n), 0);
}

void mask_xor_with(BasisMask& dst, const BasisMask& src) {
    mutable_basis_mask_view(dst).xor_with(basis_mask_view(src));
}

void mask_xor_with(MutableMaskView dst, MaskView src) {
    dst.xor_with(src);
}

[[nodiscard]] bool mask_is_zero(MaskView mask) {
    return mask.is_zero();
}

[[nodiscard]] bool mask_has_only_qubit_bits(MaskView mask, uint32_t n) {
    const uint32_t used_bits = n % 64;
    if (mask.words.empty() || used_bits == 0) {
        return true;
    }
    const uint64_t used_mask = (uint64_t{1} << used_bits) - 1U;
    return (mask.words.back() & ~used_mask) == 0;
}

[[nodiscard]] bool mask_parity(MaskView lhs, MaskView rhs) {
    bool parity = false;
    for (size_t w = 0; w < lhs.words.size(); ++w) {
        parity ^= (std::popcount(lhs.words[w] & rhs.words[w]) & 1U) != 0;
    }
    return parity;
}

[[nodiscard]] bool dynamic_sign(bool static_sign, MaskView sign_mask, MaskView physical_basis) {
    return static_sign ^ mask_parity(sign_mask, physical_basis);
}

[[nodiscard]] uint64_t valid_word_mask(uint32_t n, size_t words, size_t word) {
    // Stim simd_bits may contain padding above num_qubits. Word-level scans must
    // mask those bits or they can create spurious Y phases and Z parities.
    const uint32_t used_bits = n % 64;
    if (word + 1 != words || used_bits == 0) {
        return ~uint64_t{0};
    }
    return (uint64_t{1} << used_bits) - 1U;
}

[[nodiscard]] BasisMask pauli_x_mask(const StabilizerRow& p, uint32_t n) {
    BasisMask mask = zero_basis_mask(n);
    const size_t words = mask.size();
    for (size_t w = 0; w < words; ++w) {
        mask[w] = p.xs.u64[w] & valid_word_mask(n, words, w);
    }
    return mask;
}

[[nodiscard]] std::complex<double> i_pow(uint32_t phase_idx) {
    switch (phase_idx & 3U) {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, 1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, -1.0};
    }
}

[[nodiscard]] uint32_t pauli_y_count(const StabilizerRow& p, uint32_t n) {
    uint32_t count = 0;
    const size_t words = basis_word_count(n);
    for (size_t w = 0; w < words; ++w) {
        count += std::popcount(p.xs.u64[w] & p.zs.u64[w] & valid_word_mask(n, words, w));
    }
    return count;
}

[[nodiscard]] std::complex<double> pauli_action_phase(const StabilizerRow& p, bool sign, uint32_t n,
                                                      uint32_t y_count, MaskView basis) {
    // For P = (-1)^sign X^a Z^b, applying P to |basis> flips by a and
    // contributes i for each Y plus the Z eigenvalue on the pre-flip basis.
    uint32_t phase_idx = (sign ? 2U : 0U) + y_count;
    bool z_basis_parity = false;
    const size_t words = basis_word_count(n);
    for (size_t w = 0; w < words; ++w) {
        const uint64_t valid = valid_word_mask(n, words, w);
        z_basis_parity ^= (std::popcount(p.zs.u64[w] & basis.words[w] & valid) & 1U) != 0;
    }
    if (z_basis_parity) {
        phase_idx += 2;
    }
    return i_pow(phase_idx);
}

void multiply_row_by(StabilizerRow& dst, BasisMask& dst_sign_mask, const StabilizerRow& src,
                     const BasisMask& src_sign_mask) {
    dst.ref() *= src;
    mask_xor_with(dst_sign_mask, src_sign_mask);
}

[[nodiscard]] bool row_has_any_z(const StabilizerRow& p, uint32_t n) {
    const size_t words = basis_word_count(n);
    for (size_t w = 0; w < words; ++w) {
        if ((p.zs.u64[w] & valid_word_mask(n, words, w)) != 0) {
            return true;
        }
    }
    return false;
}

struct StabilizerAmplitudeStructure;

struct BoundStabilizerAmplitudeQuery {
    const StabilizerAmplitudeStructure* structure = nullptr;
    BasisMask base;
    std::vector<uint8_t> x_signs;

    [[nodiscard]] std::complex<double> amplitude(MaskView basis, MutableMaskView residual,
                                                 MutableMaskView current) const;
};

struct StabilizerAmplitudeStructure {
    uint32_t n = 0;
    // X-eliminate dormant columns first so the first num_dormant_generators
    // pivots land in [active_k, n). When every dormant column is a pivot
    // (can_use_gray_code), active generators have zero support on dormant
    // bits and the inner sum can be walked as a Gray code over active
    // generators instead of an O(2^k * r_X) per-active-index loop.
    uint32_t num_dormant_generators = 0;
    bool can_use_gray_code = false;
    double magnitude = 1.0;
    std::vector<StabilizerRow> x_rows;
    std::vector<BasisMask> x_sign_masks;
    std::vector<uint32_t> pivot_cols;
    std::vector<BasisMask> x_masks;
    std::vector<uint32_t> x_y_counts;
    std::vector<DynamicSignTerm> base_terms;
    std::vector<IdentityConstraint> identity_constraints;

    [[nodiscard]] BoundStabilizerAmplitudeQuery bind(MaskView physical_basis) const {
        // The row-reduction structure is shared across the whole batch. Binding
        // substitutes the queried physical bitstring x into the stored sign
        // masks, producing the actual X-generator signs and affine base string
        // for U_C^\dagger |x>.
        BoundStabilizerAmplitudeQuery query;
        query.structure = this;
        query.base = zero_basis_mask(n);
        query.x_signs.reserve(x_rows.size());

        for (size_t i = 0; i < x_rows.size(); ++i) {
            query.x_signs.push_back(dynamic_sign(static_cast<bool>(x_rows[i].sign),
                                                 basis_mask_view(x_sign_masks[i]), physical_basis)
                                        ? 1U
                                        : 0U);
        }

        for (const auto& term : base_terms) {
            if (dynamic_sign(term.static_sign, basis_mask_view(term.sign_mask), physical_basis)) {
                mutable_basis_mask_view(query.base).bit_set(term.bit, true);
            }
        }

        for (const auto& constraint : identity_constraints) {
            if (dynamic_sign(constraint.static_sign, basis_mask_view(constraint.sign_mask),
                             physical_basis)) {
                throw std::runtime_error(
                    "invalid stabilizer constraints while evaluating probability");
            }
        }

        return query;
    }
};

std::complex<double> BoundStabilizerAmplitudeQuery::amplitude(MaskView basis,
                                                              MutableMaskView residual,
                                                              MutableMaskView current) const {
    // The X-pivot rows generate the support of the stabilizer state. Walk from
    // the base string toward the requested virtual basis by applying the unique
    // generator for each still-set pivot bit, accumulating the Pauli phases on
    // the way. Any residual bit means the basis state is outside the support.
    mask_copy_to(residual, basis);
    mask_xor_with(residual, basis_mask_view(base));
    mask_copy_to(current, basis_mask_view(base));
    std::complex<double> amp{structure->magnitude, 0.0};

    for (size_t i = 0; i < structure->x_rows.size(); ++i) {
        if (!residual.bit_get(structure->pivot_cols[i])) {
            continue;
        }
        amp *= pauli_action_phase(structure->x_rows[i], x_signs[i] != 0, structure->n,
                                  structure->x_y_counts[i], current);
        mask_xor_with(current, basis_mask_view(structure->x_masks[i]));
        mask_xor_with(residual, basis_mask_view(structure->x_masks[i]));
    }

    if (!mask_is_zero(residual)) {
        return {0.0, 0.0};
    }
    return amp;
}

[[nodiscard]] StabilizerAmplitudeStructure make_stabilizer_amplitude_structure(
    const CompiledModule& program, const stim::Tableau<kStimWidth>& inv_tableau,
    uint32_t active_k) {
    const uint32_t n = program.num_qubits;
    std::vector<StabilizerRow> rows;
    std::vector<BasisMask> sign_masks;
    rows.reserve(n);
    sign_masks.reserve(n);

    // Start from stabilizers of U_C^\dagger |x>. Only the signs depend on the
    // queried physical bitstring x: the q-th stabilizer has sign x_q. Store that
    // dependence as a sign mask so the row-reduction structure can be reused.
    for (uint32_t q = 0; q < n; ++q) {
        rows.emplace_back(inv_tableau.zs[q]);
        sign_masks.push_back(zero_basis_mask(n));
        mutable_basis_mask_view(sign_masks.back()).bit_set(q, true);
    }

    // Pivot dormant columns first so dormant pivots collect at the front of
    // pivot_cols/x_rows. With Gauss-Jordan elimination this RREF order ensures
    // that, when every dormant column ends up pivoted, the active generators
    // have strictly zero X support on dormant columns.
    std::vector<uint32_t> x_col_order;
    x_col_order.reserve(n);
    for (uint32_t col = active_k; col < n; ++col) {
        x_col_order.push_back(col);
    }
    for (uint32_t col = 0; col < active_k; ++col) {
        x_col_order.push_back(col);
    }

    size_t rank_x = 0;
    std::vector<uint32_t> pivot_cols;
    // First eliminate the X block. Pivot rows with X support become generators
    // that move around the nonzero support of U_C^\dagger |x>; non-pivot rows
    // are purely Z-type constraints after this pass.
    for (uint32_t col : x_col_order) {
        auto pivot = rows.end();
        auto pivot_sign = sign_masks.end();
        auto rank_row = rows.begin() + static_cast<std::ptrdiff_t>(rank_x);
        auto rank_sign = sign_masks.begin() + static_cast<std::ptrdiff_t>(rank_x);
        for (auto it = rank_row; it != rows.end(); ++it) {
            if (it->xs[col]) {
                pivot = it;
                pivot_sign = sign_masks.begin() + (it - rows.begin());
                break;
            }
        }
        if (pivot == rows.end()) {
            continue;
        }

        std::iter_swap(rank_row, pivot);
        std::iter_swap(rank_sign, pivot_sign);
        for (size_t r = 0; r < rows.size(); ++r) {
            if (r != rank_x && rows[r].xs[col]) {
                multiply_row_by(rows[r], sign_masks[r], rows[rank_x], sign_masks[rank_x]);
            }
        }
        pivot_cols.push_back(col);
        ++rank_x;
    }

    std::vector<StabilizerRow> z_rows(rows.begin() + static_cast<std::ptrdiff_t>(rank_x),
                                      rows.end());
    std::vector<BasisMask> z_sign_masks(sign_masks.begin() + static_cast<std::ptrdiff_t>(rank_x),
                                        sign_masks.end());
    size_t rank_z = 0;
    std::vector<DynamicSignTerm> base_terms;
    // Pure Z constraints fix one base bit at a time. After binding x, each
    // pivot equation says whether the corresponding bit of the affine base
    // string is 0 or 1.
    for (uint32_t col = 0; col < n; ++col) {
        auto pivot = z_rows.end();
        auto pivot_sign = z_sign_masks.end();
        auto rank_row = z_rows.begin() + static_cast<std::ptrdiff_t>(rank_z);
        auto rank_sign = z_sign_masks.begin() + static_cast<std::ptrdiff_t>(rank_z);
        for (auto it = rank_row; it != z_rows.end(); ++it) {
            if (it->zs[col]) {
                pivot = it;
                pivot_sign = z_sign_masks.begin() + (it - z_rows.begin());
                break;
            }
        }
        if (pivot == z_rows.end()) {
            continue;
        }

        std::iter_swap(rank_row, pivot);
        std::iter_swap(rank_sign, pivot_sign);
        for (size_t r = 0; r < z_rows.size(); ++r) {
            if (r != rank_z && z_rows[r].zs[col]) {
                multiply_row_by(z_rows[r], z_sign_masks[r], z_rows[rank_z], z_sign_masks[rank_z]);
            }
        }
        base_terms.push_back(DynamicSignTerm{.bit = col,
                                             .static_sign = static_cast<bool>(z_rows[rank_z].sign),
                                             .sign_mask = z_sign_masks[rank_z]});
        ++rank_z;
    }

    std::vector<IdentityConstraint> identity_constraints;
    // A remaining row with no Z support is an identity constraint. For
    // U_C^\dagger |x> it should always bind to +I; binding to -I would point to
    // inconsistent tableau metadata or a reduction bug.
    for (size_t r = rank_z; r < z_rows.size(); ++r) {
        if (!row_has_any_z(z_rows[r], n)) {
            identity_constraints.push_back(
                IdentityConstraint{.static_sign = static_cast<bool>(z_rows[r].sign),
                                   .sign_mask = std::move(z_sign_masks[r])});
        }
    }

    StabilizerAmplitudeStructure structure;
    structure.n = n;
    structure.magnitude = std::pow(2.0, -0.5 * static_cast<double>(rank_x));
    structure.pivot_cols = std::move(pivot_cols);
    structure.x_rows.assign(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(rank_x));
    structure.x_sign_masks.assign(sign_masks.begin(),
                                  sign_masks.begin() + static_cast<std::ptrdiff_t>(rank_x));
    structure.x_masks.reserve(rank_x);
    structure.x_y_counts.reserve(rank_x);
    // Cache each X generator's bit flip separately from its full Pauli row. The
    // amplitude walk needs both: the mask updates the basis string, while the
    // row computes the phase contributed by that generator.
    for (const auto& row : structure.x_rows) {
        structure.x_masks.push_back(pauli_x_mask(row, n));
        structure.x_y_counts.push_back(pauli_y_count(row, n));
    }
    structure.base_terms = std::move(base_terms);
    structure.identity_constraints = std::move(identity_constraints);

    // Count how many pivots landed in the dormant range [active_k, n). With
    // the dormant-first column order these are the first pivots of pivot_cols.
    uint32_t dormant_pivots = 0;
    for (uint32_t c : structure.pivot_cols) {
        if (c >= active_k) {
            ++dormant_pivots;
        }
    }
    structure.num_dormant_generators = dormant_pivots;
    // Fast path requires every dormant column to be a pivot. Otherwise some
    // dormant column is free and active generators may have X support there,
    // which would silently flip dormant bits during the Gray code walk.
    structure.can_use_gray_code = (dormant_pivots == (n - active_k));

    return structure;
}

[[nodiscard]] bool is_unsupported_probability_opcode(Opcode opcode) {
    switch (opcode) {
        case Opcode::OP_FRAME_CNOT:
        case Opcode::OP_FRAME_CZ:
        case Opcode::OP_FRAME_H:
        case Opcode::OP_FRAME_S:
        case Opcode::OP_FRAME_S_DAG:
        case Opcode::OP_FRAME_SWAP:
        case Opcode::OP_ARRAY_CNOT:
        case Opcode::OP_ARRAY_CZ:
        case Opcode::OP_ARRAY_SWAP:
        case Opcode::OP_ARRAY_MULTI_CNOT:
        case Opcode::OP_ARRAY_MULTI_CZ:
        case Opcode::OP_ARRAY_H:
        case Opcode::OP_ARRAY_S:
        case Opcode::OP_ARRAY_S_DAG:
        case Opcode::OP_ARRAY_T:
        case Opcode::OP_ARRAY_T_DAG:
        case Opcode::OP_ARRAY_ROT:
        case Opcode::OP_ARRAY_U2:
        case Opcode::OP_ARRAY_U4:
        case Opcode::OP_EXPAND:
        case Opcode::OP_EXPAND_T:
        case Opcode::OP_EXPAND_T_DAG:
        case Opcode::OP_EXPAND_ROT:
        case Opcode::OP_EXP_VAL:
            return false;

        case Opcode::OP_MEAS_DORMANT_STATIC:
        case Opcode::OP_MEAS_DORMANT_RANDOM:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
        case Opcode::OP_SWAP_MEAS_INTERFERE:
        case Opcode::OP_APPLY_PAULI:
        case Opcode::OP_NOISE:
        case Opcode::OP_NOISE_BLOCK:
        case Opcode::OP_READOUT_NOISE:
        case Opcode::OP_DETECTOR:
        case Opcode::OP_POSTSELECT:
        case Opcode::OP_OBSERVABLE:
        case Opcode::NUM_OPCODES:
            return true;
    }
    throw std::invalid_argument("probabilities() encountered an unknown bytecode opcode");
}

void assert_probability_program_is_supported(const CompiledModule& program) {
    // Keep this bytecode-level list conceptually aligned with DropNonUnitaryPass,
    // which performs the analogous filtering on HIR OpType values.
    for (const auto& instr : program.bytecode) {
        if (is_unsupported_probability_opcode(instr.opcode)) {
            throw std::invalid_argument(
                "probabilities() requires pure-state evolution: measurements, feedback, noise, "
                "readout noise, detectors, postselection, and observables are not supported. "
                "EXP_VAL probes are allowed but their outputs are ignored. Use "
                "DropNonUnitaryPass only if you intentionally want to query the unitary "
                "skeleton of a mixed circuit.");
        }
    }
}

}  // namespace

std::vector<double> probabilities(const CompiledModule& program,
                                  std::span<const uint64_t> basis_masks, size_t num_basis_masks,
                                  size_t words_per_basis_mask) {
    assert_probability_program_is_supported(program);
    if (!program.constant_pool.final_tableau.has_value()) {
        throw std::invalid_argument(
            "probabilities() requires final Clifford tableau metadata; compile programs through "
            "clifft.compile() or preserve ConstantPool::final_tableau.");
    }
    assert_arena_widths_match(program.num_qubits, program.constant_pool);

    SchrodingerState state({.peak_rank = program.peak_rank,
                            .num_measurements = program.total_meas_slots,
                            .num_qubits = program.num_qubits,
                            .num_detectors = program.num_detectors,
                            .num_observables = program.num_observables,
                            .num_exp_vals = program.num_exp_vals,
                            .seed = uint64_t{0}});
    execute(program, state);

    // The VM gives the factored final state gamma * U_C * P *
    // (|phi>_A x |0>_D). The inverse tableau lets the batch query evaluator use
    // stabilizers of U_C^\dagger |x> for each requested physical bitstring x.
    stim::Tableau<kStimWidth> inv_tableau = program.constant_pool.final_tableau->inverse(false);

    const uint32_t n = program.num_qubits;
    const size_t expected_words = basis_word_count(n);
    if (words_per_basis_mask != expected_words) {
        throw std::invalid_argument(
            "probability basis masks must have ceil(num_qubits / 64) words");
    }
    if (basis_masks.size() != num_basis_masks * words_per_basis_mask) {
        throw std::invalid_argument("probability basis mask buffer has inconsistent shape");
    }

    const uint64_t active_size = state.v_size();
    const std::complex<double> scale = state.gamma() * program.constant_pool.global_weight;
    const auto structure =
        make_stabilizer_amplitude_structure(program, inv_tableau, state.active_k);
    const auto state_px = MaskView{std::span<const uint64_t>(state.p_x)};
    // validate_peak_rank() caps active_k below 63, so active bits fit in word 0
    // and this shift never reaches 64.
    const uint64_t active_z_mask =
        state.active_k == 0 ? 0 : (state.p_z[0] & ((uint64_t{1} << state.active_k) - uint64_t{1}));
    BasisMask virtual_basis_storage(expected_words);
    BasisMask residual_storage(expected_words);
    BasisMask current_storage(expected_words);
    auto virtual_basis = mutable_basis_mask_view(virtual_basis_storage);
    auto residual = mutable_basis_mask_view(residual_storage);
    auto current = mutable_basis_mask_view(current_storage);

    // Active bits live in [0, active_k). validate_peak_rank() caps active_k
    // below 63, so the mask fits in one word and the shift never reaches 64.
    const uint64_t active_mask =
        state.active_k == 0 ? uint64_t{0} : (uint64_t{1} << state.active_k) - uint64_t{1};

    std::vector<double> out;
    out.reserve(num_basis_masks);
    for (size_t basis_idx = 0; basis_idx < num_basis_masks; ++basis_idx) {
        const auto basis_mask =
            MaskView{basis_masks.subspan(basis_idx * expected_words, expected_words)};
        if (!mask_has_only_qubit_bits(basis_mask, n)) {
            throw std::invalid_argument("probability basis masks must not set unused high bits");
        }

        // Bind the shared stabilizer structure to this physical bitstring. This
        // resolves row signs and the base string without redoing elimination.
        auto query = structure.bind(basis_mask);

        std::complex<double> amp{0.0, 0.0};

        if (!structure.can_use_gray_code) {
            // Fallback: some dormant column is a free column, so active
            // generators can flip dormant bits. Use the original O(2^k * r_X)
            // walk that explicitly enumerates active_indices and validates
            // each amplitude via residual mask.
            mask_copy_to(virtual_basis, state_px);
            uint64_t previous_active_index = 0;
            for (uint64_t active_index = 0; active_index < active_size; ++active_index) {
                uint64_t changed_bits = active_index ^ previous_active_index;
                while (changed_bits != 0) {
                    const auto q = static_cast<uint32_t>(std::countr_zero(changed_bits));
                    virtual_basis.bit_xor(q);
                    changed_bits &= changed_bits - 1;
                }
                previous_active_index = active_index;

                auto coeff = query.amplitude(virtual_basis, residual, current);
                if (coeff == std::complex<double>{0.0, 0.0}) {
                    continue;
                }
                // The runtime Z frame is diagonal on the active basis index.
                // Dormant |0> axes do not contribute a Z-frame sign.
                const bool sign_bit = (std::popcount(active_index & active_z_mask) & 1U) != 0;
                double sign = sign_bit ? -1.0 : 1.0;
                amp += state.v()[active_index] * sign * coeff;
            }
        } else {
            // Fast path: every dormant column is a pivot, so dormant
            // generators form the identity on the dormant block and active
            // generators have strictly zero X on dormant columns. Two
            // consequences:
            //   1. Step 1 (clearing dormant bits of `current` to match
            //      state_px) always succeeds; no prune check is needed.
            //   2. Toggling active generators does not disturb dormant bits,
            //      so the Gray code walk over the 2^r_A active subsets
            //      enumerates exactly the basis states with nonzero
            //      contribution.
            mask_copy_to(current, basis_mask_view(query.base));
            amp = std::complex<double>{structure.magnitude, 0.0};

            // Step 1: match state_px on dormant pivot columns by conditionally
            // applying each dormant generator. RREF guarantees each pivot
            // column has X only in its pivot row, so generators applied in
            // order do not disturb each other's pivot bits.
            for (size_t i = 0; i < structure.num_dormant_generators; ++i) {
                const uint32_t p = structure.pivot_cols[i];
                const bool current_bit = (current.words[p / 64] >> (p % 64)) & 1U;
                const bool target_bit = (state_px.words[p / 64] >> (p % 64)) & 1U;
                if (current_bit != target_bit) {
                    amp *= pauli_action_phase(structure.x_rows[i], query.x_signs[i] != 0, n,
                                              structure.x_y_counts[i], current);
                    mask_xor_with(current, basis_mask_view(structure.x_masks[i]));
                }
            }

            // Debug invariant: every dormant bit of `current` now matches
            // state_px. Checked only in debug builds.
            assert([&]() {
                for (uint32_t q = state.active_k; q < n; ++q) {
                    const bool c = (current.words[q / 64] >> (q % 64)) & 1U;
                    const bool t = (state_px.words[q / 64] >> (q % 64)) & 1U;
                    if (c != t) {
                        return false;
                    }
                }
                return true;
            }());

            // Step 2: Gray code walk over the active generators. Each
            // transition toggles exactly one generator (the bit that flips
            // between successive Gray codes is countr_zero(i)).
            const size_t r_a = structure.x_rows.size() - structure.num_dormant_generators;
            const uint64_t num_active_states = uint64_t{1} << r_a;
            std::complex<double> total{0.0, 0.0};

            for (uint64_t i = 0; i < num_active_states; ++i) {
                if (i > 0) {
                    const uint32_t toggle_idx = static_cast<uint32_t>(std::countr_zero(i));
                    const size_t gen_idx = structure.num_dormant_generators + toggle_idx;
                    amp *=
                        pauli_action_phase(structure.x_rows[gen_idx], query.x_signs[gen_idx] != 0,
                                           n, structure.x_y_counts[gen_idx], current);
                    mask_xor_with(current, basis_mask_view(structure.x_masks[gen_idx]));
                }

                // active_index = active bits of (current XOR state_px). All
                // active bits fit in word 0 (active_k < 63).
                const uint64_t active_index =
                    state.active_k == 0 ? uint64_t{0}
                                        : ((current.words[0] ^ state_px.words[0]) & active_mask);
                const bool sign_bit = (std::popcount(active_index & active_z_mask) & 1U) != 0;
                const double sign = sign_bit ? -1.0 : 1.0;
                total += state.v()[active_index] * sign * amp;
            }
            amp = total;
        }
        out.push_back(std::norm(scale * amp));
    }
    return out;
}

}  // namespace clifft
