#include "clifft/svm/svm.h"
#include "clifft/svm/svm_math.h"

#include <algorithm>
#include <bit>
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
    for (size_t w = 0; w < dst.size(); ++w) {
        dst[w] ^= src[w];
    }
}

[[nodiscard]] bool mask_is_zero(const BasisMask& mask) {
    return std::all_of(mask.begin(), mask.end(), [](uint64_t word) { return word == 0; });
}

[[nodiscard]] bool mask_has_only_qubit_bits(const BasisMask& mask, uint32_t n) {
    const uint32_t used_bits = n % 64;
    if (mask.empty() || used_bits == 0) {
        return true;
    }
    const uint64_t used_mask = (uint64_t{1} << used_bits) - 1U;
    return (mask.back() & ~used_mask) == 0;
}

[[nodiscard]] bool mask_parity(const BasisMask& lhs, const BasisMask& rhs) {
    bool parity = false;
    for (size_t w = 0; w < lhs.size(); ++w) {
        parity ^= (std::popcount(lhs[w] & rhs[w]) & 1U) != 0;
    }
    return parity;
}

[[nodiscard]] bool dynamic_sign(bool static_sign, const BasisMask& sign_mask,
                                const BasisMask& physical_basis) {
    return static_sign ^ mask_parity(sign_mask, physical_basis);
}

[[nodiscard]] uint64_t valid_word_mask(uint32_t n, size_t words, size_t word) {
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

[[nodiscard]] std::complex<double> pauli_action_phase(const StabilizerRow& p, bool sign, uint32_t n,
                                                      const BasisMask& basis) {
    uint32_t phase_idx = sign ? 2U : 0U;
    bool z_basis_parity = false;
    const size_t words = basis_word_count(n);
    for (size_t w = 0; w < words; ++w) {
        const uint64_t valid = valid_word_mask(n, words, w);
        phase_idx += std::popcount(p.xs.u64[w] & p.zs.u64[w] & valid);
        z_basis_parity ^= (std::popcount(p.zs.u64[w] & basis[w] & valid) & 1U) != 0;
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

    [[nodiscard]] std::complex<double> amplitude(const BasisMask& basis) const;
};

struct StabilizerAmplitudeStructure {
    uint32_t n = 0;
    double magnitude = 1.0;
    std::vector<StabilizerRow> x_rows;
    std::vector<BasisMask> x_sign_masks;
    std::vector<uint32_t> pivot_cols;
    std::vector<BasisMask> x_masks;
    std::vector<DynamicSignTerm> base_terms;
    std::vector<IdentityConstraint> identity_constraints;

    [[nodiscard]] BoundStabilizerAmplitudeQuery bind(const BasisMask& physical_basis) const {
        BoundStabilizerAmplitudeQuery query;
        query.structure = this;
        query.base = zero_basis_mask(n);
        query.x_signs.reserve(x_rows.size());

        for (size_t i = 0; i < x_rows.size(); ++i) {
            query.x_signs.push_back(
                dynamic_sign(static_cast<bool>(x_rows[i].sign), x_sign_masks[i], physical_basis)
                    ? 1U
                    : 0U);
        }

        for (const auto& term : base_terms) {
            if (dynamic_sign(term.static_sign, term.sign_mask, physical_basis)) {
                bit_set(query.base, term.bit, true);
            }
        }

        for (const auto& constraint : identity_constraints) {
            if (dynamic_sign(constraint.static_sign, constraint.sign_mask, physical_basis)) {
                throw std::runtime_error(
                    "invalid stabilizer constraints while evaluating probability");
            }
        }

        return query;
    }
};

std::complex<double> BoundStabilizerAmplitudeQuery::amplitude(const BasisMask& basis) const {
    BasisMask residual = basis;
    mask_xor_with(residual, base);
    BasisMask current = base;
    std::complex<double> amp{structure->magnitude, 0.0};

    for (size_t i = 0; i < structure->x_rows.size(); ++i) {
        if (!bit_get(residual, structure->pivot_cols[i])) {
            continue;
        }
        amp *= pauli_action_phase(structure->x_rows[i], x_signs[i] != 0, structure->n, current);
        mask_xor_with(current, structure->x_masks[i]);
        mask_xor_with(residual, structure->x_masks[i]);
    }

    if (!mask_is_zero(residual)) {
        return {0.0, 0.0};
    }
    return amp;
}

[[nodiscard]] StabilizerAmplitudeStructure make_stabilizer_amplitude_structure(
    const CompiledModule& program, const stim::Tableau<kStimWidth>& inv_tableau) {
    const uint32_t n = program.num_qubits;
    std::vector<StabilizerRow> rows;
    std::vector<BasisMask> sign_masks;
    rows.reserve(n);
    sign_masks.reserve(n);

    for (uint32_t q = 0; q < n; ++q) {
        rows.emplace_back(inv_tableau.zs[q]);
        sign_masks.push_back(zero_basis_mask(n));
        bit_set(sign_masks.back(), q, true);
    }

    size_t rank_x = 0;
    std::vector<uint32_t> pivot_cols;
    for (uint32_t col = 0; col < n; ++col) {
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
    for (const auto& row : structure.x_rows) {
        structure.x_masks.push_back(pauli_x_mask(row, n));
    }
    structure.base_terms = std::move(base_terms);
    structure.identity_constraints = std::move(identity_constraints);
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
                                  const std::vector<std::vector<uint64_t>>& basis_masks) {
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

    stim::Tableau<kStimWidth> inv_tableau = program.constant_pool.final_tableau->inverse(false);

    const uint32_t n = program.num_qubits;
    const size_t expected_words = basis_word_count(n);
    const uint64_t active_size = state.v_size();
    const std::complex<double> scale = state.gamma() * program.constant_pool.global_weight;
    const auto structure = make_stabilizer_amplitude_structure(program, inv_tableau);

    std::vector<double> out;
    out.reserve(basis_masks.size());
    for (const BasisMask& basis_mask : basis_masks) {
        if (basis_mask.size() != expected_words) {
            throw std::invalid_argument(
                "probability basis masks must have ceil(num_qubits / 64) words");
        }
        if (!mask_has_only_qubit_bits(basis_mask, n)) {
            throw std::invalid_argument("probability basis masks must not set unused high bits");
        }

        auto query = structure.bind(basis_mask);

        std::complex<double> amp{0.0, 0.0};
        for (uint64_t active_index = 0; active_index < active_size; ++active_index) {
            BasisMask virtual_basis = state.p_x;
            for (uint32_t q = 0; q < state.active_k; ++q) {
                bit_xor(virtual_basis, q, ((active_index >> q) & 1ULL) != 0);
            }
            auto coeff = query.amplitude(virtual_basis);
            if (coeff == std::complex<double>{0.0, 0.0}) {
                continue;
            }
            bool sign_bit = false;
            for (uint32_t q = 0; q < state.active_k; ++q) {
                if (((active_index >> q) & 1ULL) != 0 && bit_get(state.p_z, q)) {
                    sign_bit = !sign_bit;
                }
            }
            double sign = sign_bit ? -1.0 : 1.0;
            amp += state.v()[active_index] * sign * coeff;
        }
        out.push_back(std::norm(scale * amp));
    }
    return out;
}

}  // namespace clifft
