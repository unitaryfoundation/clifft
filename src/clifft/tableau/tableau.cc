#include "clifft/tableau/tableau.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

namespace {

uint32_t row_index(bool z, uint32_t qubit, uint32_t num_qubits) {
    return (z ? num_qubits : 0) + qubit;
}

void right_multiply_generator(PauliString& pauli, uint32_t qubit, bool z_generator) {
    const bool old_x = pauli.x().bit_get(qubit);
    const bool old_z = pauli.z().bit_get(qubit);
    if (!z_generator && old_z) {
        pauli.add_phase(2);
    }
    pauli.set_pauli(qubit, old_x ^ !z_generator, old_z ^ z_generator);
}

void right_multiply_masks(MutableMaskView x, MutableMaskView z, uint8_t& phase,
                          PauliStringView value, uint8_t phase_delta = 0) {
    assert(x.num_words() == value.x().num_words());
    assert(z.num_words() == value.z().num_words());
    bool crossing_parity = false;
    for (uint32_t w = 0; w < x.num_words(); ++w) {
        crossing_parity ^= (std::popcount(z.words[w] & value.x().words[w]) & 1U) != 0;
    }
    phase = (phase + value.phase() + phase_delta + (crossing_parity ? 2U : 0U)) & 3U;
    x.xor_with(value.x());
    z.xor_with(value.z());
}

uint8_t mask_y_phase(std::span<const uint64_t> x, std::span<const uint64_t> z) {
    uint32_t count = 0;
    for (size_t w = 0; w < x.size(); ++w) {
        count += std::popcount(x[w] & z[w]);
    }
    return static_cast<uint8_t>(count & 3U);
}

void transpose_64x64(std::array<uint64_t, 64>& block) {
    constexpr std::array<std::pair<uint32_t, uint64_t>, 6> kSteps{{
        {32, 0x00000000FFFFFFFFULL},
        {16, 0x0000FFFF0000FFFFULL},
        {8, 0x00FF00FF00FF00FFULL},
        {4, 0x0F0F0F0F0F0F0F0FULL},
        {2, 0x3333333333333333ULL},
        {1, 0x5555555555555555ULL},
    }};
    for (const auto& [shift, mask] : kSteps) {
        for (uint32_t base = 0; base < 64; base += 2 * shift) {
            for (uint32_t i = 0; i < shift; ++i) {
                const uint64_t diff = ((block[base + i] >> shift) ^ block[base + i + shift]) & mask;
                block[base + i] ^= diff << shift;
                block[base + i + shift] ^= diff;
            }
        }
    }
}

void transpose_submatrix(std::span<const uint64_t> src, uint32_t src_row_base,
                         std::span<uint64_t> dst, uint32_t dst_row_base, uint32_t num_qubits,
                         uint32_t num_words) {
    std::array<uint64_t, 64> block{};
    for (uint32_t wb_row = 0; wb_row < num_words; ++wb_row) {
        const uint32_t row_limit = std::min<uint32_t>(64, num_qubits - 64 * wb_row);
        for (uint32_t wb_col = 0; wb_col < num_words; ++wb_col) {
            const uint32_t col_limit = std::min<uint32_t>(64, num_qubits - 64 * wb_col);
            for (uint32_t r = 0; r < row_limit; ++r) {
                block[r] =
                    src[static_cast<size_t>(src_row_base + 64 * wb_row + r) * num_words + wb_col];
            }
            for (uint32_t r = row_limit; r < 64; ++r) {
                block[r] = 0;
            }
            transpose_64x64(block);
            for (uint32_t c = 0; c < col_limit; ++c) {
                dst[static_cast<size_t>(dst_row_base + 64 * wb_col + c) * num_words + wb_row] =
                    block[c];
            }
        }
    }
}

}  // namespace

Tableau::Tableau(uint32_t num_qubits)
    : num_qubits_(num_qubits),
      num_words_((num_qubits + 63) / 64),
      x_rows_(static_cast<size_t>(2) * num_qubits * num_words_, 0),
      z_rows_(x_rows_.size(), 0),
      phases_(static_cast<size_t>(2) * num_qubits, 0) {
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        x_rows_[static_cast<size_t>(q) * num_words_ + q / 64] |= uint64_t{1} << (q % 64);
        z_rows_[static_cast<size_t>(num_qubits_ + q) * num_words_ + q / 64] |= uint64_t{1}
                                                                               << (q % 64);
    }
}

Tableau Tableau::from_rows(std::initializer_list<std::string_view> rows) {
    if (rows.size() % 2 != 0) {
        throw std::invalid_argument("A tableau requires one X and one Z row per qubit");
    }
    Tableau result(static_cast<uint32_t>(rows.size() / 2));
    uint32_t q = 0;
    for (auto it = rows.begin(); it != rows.end();) {
        const PauliString x = PauliString::from_text(*it++);
        const PauliString z = PauliString::from_text(*it++);
        if (x.num_qubits() != result.num_qubits() || z.num_qubits() != result.num_qubits()) {
            throw std::invalid_argument("Tableau row width mismatch");
        }
        result.set_row(row_index(false, q, result.num_qubits()), x.view());
        result.set_row(row_index(true, q, result.num_qubits()), z.view());
        ++q;
    }
    return result;
}

const Tableau& Tableau::from_named_gate(GateType gate) {
// Parsing fixed row literals for every circuit operation needlessly makes gate lookup allocate.
#define CLIFFT_RETURN_CACHED_TABLEAU(...)                       \
    do {                                                        \
        static const Tableau result = from_rows({__VA_ARGS__}); \
        return result;                                          \
    } while (false)

    switch (gate) {
        case GateType::H:
            CLIFFT_RETURN_CACHED_TABLEAU("+Z", "+X");
        case GateType::S:
            CLIFFT_RETURN_CACHED_TABLEAU("+Y", "+Z");
        case GateType::S_DAG:
            CLIFFT_RETURN_CACHED_TABLEAU("-Y", "+Z");
        case GateType::X:
            CLIFFT_RETURN_CACHED_TABLEAU("+X", "-Z");
        case GateType::Y:
            CLIFFT_RETURN_CACHED_TABLEAU("-X", "-Z");
        case GateType::Z:
            CLIFFT_RETURN_CACHED_TABLEAU("-X", "+Z");
        case GateType::SQRT_X:
            CLIFFT_RETURN_CACHED_TABLEAU("+X", "-Y");
        case GateType::SQRT_X_DAG:
            CLIFFT_RETURN_CACHED_TABLEAU("+X", "+Y");
        case GateType::SQRT_Y:
            CLIFFT_RETURN_CACHED_TABLEAU("-Z", "+X");
        case GateType::SQRT_Y_DAG:
            CLIFFT_RETURN_CACHED_TABLEAU("+Z", "-X");
        case GateType::H_XY:
            CLIFFT_RETURN_CACHED_TABLEAU("+Y", "-Z");
        case GateType::H_YZ:
            CLIFFT_RETURN_CACHED_TABLEAU("-X", "+Y");
        case GateType::H_NXY:
            CLIFFT_RETURN_CACHED_TABLEAU("-Y", "-Z");
        case GateType::H_NXZ:
            CLIFFT_RETURN_CACHED_TABLEAU("-Z", "-X");
        case GateType::H_NYZ:
            CLIFFT_RETURN_CACHED_TABLEAU("-X", "-Y");
        case GateType::C_XYZ:
            CLIFFT_RETURN_CACHED_TABLEAU("+Y", "+X");
        case GateType::C_ZYX:
            CLIFFT_RETURN_CACHED_TABLEAU("+Z", "+Y");
        case GateType::C_NXYZ:
            CLIFFT_RETURN_CACHED_TABLEAU("-Y", "-X");
        case GateType::C_NZYX:
            CLIFFT_RETURN_CACHED_TABLEAU("-Z", "-Y");
        case GateType::C_XNYZ:
            CLIFFT_RETURN_CACHED_TABLEAU("-Y", "+X");
        case GateType::C_XYNZ:
            CLIFFT_RETURN_CACHED_TABLEAU("+Y", "-X");
        case GateType::C_ZNYX:
            CLIFFT_RETURN_CACHED_TABLEAU("+Z", "-Y");
        case GateType::C_ZYNX:
            CLIFFT_RETURN_CACHED_TABLEAU("-Z", "+Y");
        case GateType::I:
            CLIFFT_RETURN_CACHED_TABLEAU("+X", "+Z");
        case GateType::CX:
            CLIFFT_RETURN_CACHED_TABLEAU("+XX", "+Z_", "+_X", "+ZZ");
        case GateType::CY:
            CLIFFT_RETURN_CACHED_TABLEAU("+XY", "+Z_", "+ZX", "+ZZ");
        case GateType::CZ:
            CLIFFT_RETURN_CACHED_TABLEAU("+XZ", "+Z_", "+ZX", "+_Z");
        case GateType::SWAP:
            CLIFFT_RETURN_CACHED_TABLEAU("+_X", "+_Z", "+X_", "+Z_");
        case GateType::ISWAP:
            CLIFFT_RETURN_CACHED_TABLEAU("+ZY", "+_Z", "+YZ", "+Z_");
        case GateType::ISWAP_DAG:
            CLIFFT_RETURN_CACHED_TABLEAU("-ZY", "+_Z", "-YZ", "+Z_");
        case GateType::SQRT_XX:
            CLIFFT_RETURN_CACHED_TABLEAU("+X_", "-YX", "+_X", "-XY");
        case GateType::SQRT_XX_DAG:
            CLIFFT_RETURN_CACHED_TABLEAU("+X_", "+YX", "+_X", "+XY");
        case GateType::SQRT_YY:
            CLIFFT_RETURN_CACHED_TABLEAU("-ZY", "+XY", "-YZ", "+YX");
        case GateType::SQRT_YY_DAG:
            CLIFFT_RETURN_CACHED_TABLEAU("+ZY", "-XY", "+YZ", "-YX");
        case GateType::SQRT_ZZ:
            CLIFFT_RETURN_CACHED_TABLEAU("+YZ", "+Z_", "+ZY", "+_Z");
        case GateType::SQRT_ZZ_DAG:
            CLIFFT_RETURN_CACHED_TABLEAU("-YZ", "+Z_", "-ZY", "+_Z");
        case GateType::CXSWAP:
            CLIFFT_RETURN_CACHED_TABLEAU("+XX", "+_Z", "+X_", "+ZZ");
        case GateType::CZSWAP:
            CLIFFT_RETURN_CACHED_TABLEAU("+ZX", "+_Z", "+XZ", "+Z_");
        case GateType::SWAPCX:
            CLIFFT_RETURN_CACHED_TABLEAU("+_X", "+ZZ", "+XX", "+Z_");
        case GateType::XCX:
            CLIFFT_RETURN_CACHED_TABLEAU("+X_", "+ZX", "+_X", "+XZ");
        case GateType::XCY:
            CLIFFT_RETURN_CACHED_TABLEAU("+X_", "+ZY", "+XX", "+XZ");
        case GateType::XCZ:
            CLIFFT_RETURN_CACHED_TABLEAU("+X_", "+ZZ", "+XX", "+_Z");
        case GateType::YCX:
            CLIFFT_RETURN_CACHED_TABLEAU("+XX", "+ZX", "+_X", "+YZ");
        case GateType::YCY:
            CLIFFT_RETURN_CACHED_TABLEAU("+XY", "+ZY", "+YX", "+YZ");
        case GateType::YCZ:
            CLIFFT_RETURN_CACHED_TABLEAU("+XZ", "+ZZ", "+YX", "+_Z");
        case GateType::II:
            CLIFFT_RETURN_CACHED_TABLEAU("+X_", "+Z_", "+_X", "+_Z");
        default:
            throw std::invalid_argument("Gate does not have a fixed Clifford tableau: " +
                                        std::string(gate_name(gate)));
    }

#undef CLIFFT_RETURN_CACHED_TABLEAU
}

GateType Tableau::inverse_named_gate(GateType gate) {
    // Validate that the gate has a fixed Clifford tableau before mapping its inverse.
    static_cast<void>(from_named_gate(gate));
    switch (gate) {
        case GateType::S:
            return GateType::S_DAG;
        case GateType::S_DAG:
            return GateType::S;
        case GateType::SQRT_X:
            return GateType::SQRT_X_DAG;
        case GateType::SQRT_X_DAG:
            return GateType::SQRT_X;
        case GateType::SQRT_Y:
            return GateType::SQRT_Y_DAG;
        case GateType::SQRT_Y_DAG:
            return GateType::SQRT_Y;
        case GateType::C_XYZ:
            return GateType::C_ZYX;
        case GateType::C_ZYX:
            return GateType::C_XYZ;
        case GateType::C_NXYZ:
            return GateType::C_ZYNX;
        case GateType::C_NZYX:
            return GateType::C_XYNZ;
        case GateType::C_XNYZ:
            return GateType::C_ZNYX;
        case GateType::C_XYNZ:
            return GateType::C_NZYX;
        case GateType::C_ZNYX:
            return GateType::C_XNYZ;
        case GateType::C_ZYNX:
            return GateType::C_NXYZ;
        case GateType::ISWAP:
            return GateType::ISWAP_DAG;
        case GateType::ISWAP_DAG:
            return GateType::ISWAP;
        case GateType::SQRT_XX:
            return GateType::SQRT_XX_DAG;
        case GateType::SQRT_XX_DAG:
            return GateType::SQRT_XX;
        case GateType::SQRT_YY:
            return GateType::SQRT_YY_DAG;
        case GateType::SQRT_YY_DAG:
            return GateType::SQRT_YY;
        case GateType::SQRT_ZZ:
            return GateType::SQRT_ZZ_DAG;
        case GateType::SQRT_ZZ_DAG:
            return GateType::SQRT_ZZ;
        case GateType::CXSWAP:
            return GateType::SWAPCX;
        case GateType::SWAPCX:
            return GateType::CXSWAP;
        default:
            return gate;
    }
}

Tableau Tableau::from_pauli_rotation(PauliStringView axis, bool dagger) {
    if (!axis.is_hermitian()) {
        throw std::invalid_argument("Pauli rotation axis must be Hermitian");
    }
    Tableau result(axis.num_qubits());
    for (uint32_t q = 0; q < axis.num_qubits(); ++q) {
        for (bool z_generator : {false, true}) {
            const uint32_t index = row_index(z_generator, q, axis.num_qubits());
            const bool anticommutes = z_generator ? axis.x().bit_get(q) : axis.z().bit_get(q);
            if (!anticommutes) {
                continue;
            }
            PauliString mapped(axis.num_qubits());
            mapped.set_phase(axis.phase());
            mapped.mut_x().xor_with(axis.x());
            mapped.mut_z().xor_with(axis.z());
            right_multiply_generator(mapped, q, z_generator);
            mapped.add_phase(dagger ? 1 : 3);
            result.set_row(index, mapped.view());
        }
    }
    return result;
}

PauliStringView Tableau::row(uint32_t index) const {
    assert(index < 2 * num_qubits_);
    const size_t offset = static_cast<size_t>(index) * num_words_;
    return PauliStringView{MaskView{std::span<const uint64_t>(x_rows_).subspan(offset, num_words_)},
                           MaskView{std::span<const uint64_t>(z_rows_).subspan(offset, num_words_)},
                           phases_[index], num_qubits_};
}

PauliStringView Tableau::x_output(uint32_t qubit) const {
    assert(qubit < num_qubits_);
    return row(row_index(false, qubit, num_qubits_));
}

PauliStringView Tableau::z_output(uint32_t qubit) const {
    assert(qubit < num_qubits_);
    return row(row_index(true, qubit, num_qubits_));
}

void Tableau::set_x_output(uint32_t qubit, PauliStringView value) {
    assert(qubit < num_qubits_);
    set_row(row_index(false, qubit, num_qubits_), value);
}

void Tableau::set_z_output(uint32_t qubit, PauliStringView value) {
    assert(qubit < num_qubits_);
    set_row(row_index(true, qubit, num_qubits_), value);
}

void Tableau::right_multiply_x_output(uint32_t qubit, PauliStringView value) {
    assert(qubit < num_qubits_);
    right_multiply_row_by_pauli(row_index(false, qubit, num_qubits_), value);
}

void Tableau::right_multiply_z_output(uint32_t qubit, PauliStringView value) {
    assert(qubit < num_qubits_);
    right_multiply_row_by_pauli(row_index(true, qubit, num_qubits_), value);
}

bool Tableau::satisfies_invariants() const {
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (!x_output(q).is_hermitian() || !z_output(q).is_hermitian()) {
            return false;
        }
        for (uint32_t other = 0; other < num_qubits_; ++other) {
            if (!x_output(q).commutes(x_output(other)) || !z_output(q).commutes(z_output(other)) ||
                (x_output(q).commutes(z_output(other)) == (q == other))) {
                return false;
            }
        }
    }
    return true;
}

PauliString Tableau::y_output(uint32_t qubit) const {
    PauliString input(num_qubits_);
    input.set_pauli(qubit, true, true);
    input.set_sign(false);
    return apply(input.view());
}

void Tableau::set_row(uint32_t index, PauliStringView value) {
    assert(index < 2 * num_qubits_);
    assert(value.num_qubits() == num_qubits_);
    const size_t offset = static_cast<size_t>(index) * num_words_;
    for (uint32_t w = 0; w < num_words_; ++w) {
        x_rows_[offset + w] = value.x().words[w];
        z_rows_[offset + w] = value.z().words[w];
    }
    phases_[index] = value.phase();
}

void Tableau::right_multiply_row(uint32_t destination, uint32_t source, uint8_t phase_delta) {
    assert(destination < 2 * num_qubits_);
    assert(source < 2 * num_qubits_);
    assert(destination != source);
    right_multiply_row_by_pauli(destination, row(source), phase_delta);
}

void Tableau::right_multiply_row_by_pauli(uint32_t destination, PauliStringView value,
                                          uint8_t phase_delta) {
    assert(destination < 2 * num_qubits_);
    assert(value.num_qubits() == num_qubits_);
    const size_t destination_offset = static_cast<size_t>(destination) * num_words_;
    right_multiply_masks(
        MutableMaskView{std::span<uint64_t>(x_rows_).subspan(destination_offset, num_words_)},
        MutableMaskView{std::span<uint64_t>(z_rows_).subspan(destination_offset, num_words_)},
        phases_[destination], value, phase_delta);
}

PauliString Tableau::apply(PauliStringView input) const {
    assert(input.num_qubits() == num_qubits_);
    PauliString result(num_qubits_);
    result.set_phase(input.phase());
    for (uint32_t w = 0; w < num_words_; ++w) {
        uint64_t pending = input.x().words[w];
        if (w + 1 == num_words_ && num_qubits_ % 64 != 0) {
            pending &= (uint64_t{1} << (num_qubits_ % 64)) - 1;
        }
        while (pending != 0) {
            const uint32_t q = 64 * w + std::countr_zero(pending);
            result.right_multiply(x_output(q));
            pending &= pending - 1;
        }
    }
    for (uint32_t w = 0; w < num_words_; ++w) {
        uint64_t pending = input.z().words[w];
        if (w + 1 == num_words_ && num_qubits_ % 64 != 0) {
            pending &= (uint64_t{1} << (num_qubits_ % 64)) - 1;
        }
        while (pending != 0) {
            const uint32_t q = 64 * w + std::countr_zero(pending);
            result.right_multiply(z_output(q));
            pending &= pending - 1;
        }
    }
    return result;
}

Tableau Tableau::then(const Tableau& next) const {
    if (num_qubits_ != next.num_qubits_) {
        throw std::invalid_argument("Cannot compose tableaus with different widths");
    }
    Tableau result(num_qubits_);
    for (uint32_t row_i = 0; row_i < 2 * num_qubits_; ++row_i) {
        const PauliString mapped = next.apply(row(row_i));
        result.set_row(row_i, mapped.view());
    }
    return result;
}

Tableau Tableau::inverse() const {
    Tableau result(num_qubits_);
    if (num_qubits_ == 0) {
        return result;
    }

    // Symplectic inversion transposes the four n x n generator blocks:
    //   inverse_x(q).x[k] = z_output(k).z[q], inverse_x(q).z[k] = x_output(k).z[q]
    //   inverse_z(q).x[k] = z_output(k).x[q], inverse_z(q).z[k] = x_output(k).x[q]
    transpose_submatrix(z_rows_, num_qubits_, result.x_rows_, 0, num_qubits_, num_words_);
    transpose_submatrix(z_rows_, 0, result.z_rows_, 0, num_qubits_, num_words_);
    transpose_submatrix(x_rows_, num_qubits_, result.x_rows_, num_qubits_, num_qubits_, num_words_);
    transpose_submatrix(x_rows_, 0, result.z_rows_, num_qubits_, num_qubits_, num_words_);

    struct RowSpan {
        uint32_t begin = 0;
        uint32_t end = 0;
    };
    std::vector<RowSpan> spans(static_cast<size_t>(2) * num_qubits_);
    for (uint32_t r = 0; r < 2 * num_qubits_; ++r) {
        const size_t offset = static_cast<size_t>(r) * num_words_;
        uint32_t b = 0;
        while (b < num_words_ && (x_rows_[offset + b] | z_rows_[offset + b]) == 0) {
            ++b;
        }
        uint32_t e = num_words_;
        while (e > b && (x_rows_[offset + e - 1] | z_rows_[offset + e - 1]) == 0) {
            --e;
        }
        spans[r] = {b, e};
    }

    std::vector<uint64_t> acc_z(num_words_, 0);
    const uint64_t last_word_mask =
        num_qubits_ % 64 == 0 ? ~uint64_t{0} : (uint64_t{1} << (num_qubits_ % 64)) - 1;
    for (uint32_t idx = 0; idx < 2 * num_qubits_; ++idx) {
        const size_t inv_offset = static_cast<size_t>(idx) * num_words_;
        const std::span<const uint64_t> inv_x =
            std::span<const uint64_t>(result.x_rows_).subspan(inv_offset, num_words_);
        const std::span<const uint64_t> inv_z =
            std::span<const uint64_t>(result.z_rows_).subspan(inv_offset, num_words_);
        const uint8_t base_phase = mask_y_phase(inv_x, inv_z);
        uint8_t phase = base_phase;
        uint32_t dirty_begin = num_words_;
        uint32_t dirty_end = 0;

        auto multiply_forward_row = [&](uint32_t forward_row) {
            const auto [b, e] = spans[forward_row];
            const size_t f_offset = static_cast<size_t>(forward_row) * num_words_;
            bool crossing = false;
            for (uint32_t w = b; w < e; ++w) {
                crossing ^= (std::popcount(acc_z[w] & x_rows_[f_offset + w]) & 1U) != 0;
                acc_z[w] ^= z_rows_[f_offset + w];
            }
            if (b < e) {
                dirty_begin = std::min(dirty_begin, b);
                dirty_end = std::max(dirty_end, e);
            }
            phase = (phase + phases_[forward_row] + (crossing ? 2U : 0U)) & 3U;
        };

        for (uint32_t w = 0; w < num_words_; ++w) {
            uint64_t pending = inv_x[w] & (w + 1 == num_words_ ? last_word_mask : ~uint64_t{0});
            while (pending != 0) {
                multiply_forward_row(64 * w + std::countr_zero(pending));
                pending &= pending - 1;
            }
        }
        for (uint32_t w = 0; w < num_words_; ++w) {
            uint64_t pending = inv_z[w] & (w + 1 == num_words_ ? last_word_mask : ~uint64_t{0});
            while (pending != 0) {
                multiply_forward_row(num_qubits_ + 64 * w + std::countr_zero(pending));
                pending &= pending - 1;
            }
        }

        if (dirty_begin < dirty_end) {
            std::fill_n(acc_z.data() + dirty_begin, dirty_end - dirty_begin, uint64_t{0});
        }
        // Forward evaluation of an unsigned inverse generator yields +/-X_q or +/-Z_q,
        // both of which have zero Y count, so phase == 2 indicates a minus sign.
        assert(phase == 0 || phase == 2);
        result.phases_[idx] = (base_phase + (phase == 2 ? 2U : 0U)) & 3U;
    }
    return result;
}

void Tableau::append_local(const Tableau& gate, std::span<const uint32_t> targets) {
    if (gate.num_qubits() != targets.size()) {
        throw std::invalid_argument("Gate tableau width does not match target count");
    }
    for (uint32_t target : targets) {
        if (target >= num_qubits_) {
            throw std::invalid_argument("Gate target is outside the tableau");
        }
    }

    for (uint32_t row_i = 0; row_i < 2 * num_qubits_; ++row_i) {
        const PauliStringView old = row(row_i);
        PauliString local(gate.num_qubits());
        for (uint32_t local_q = 0; local_q < targets.size(); ++local_q) {
            local.set_pauli(local_q, old.x().bit_get(targets[local_q]),
                            old.z().bit_get(targets[local_q]));
        }
        local.set_sign(false);
        const PauliString mapped = gate.apply(local.view());

        PauliString replacement(num_qubits_);
        replacement.set_phase(old.phase());
        replacement.mut_x().xor_with(old.x());
        replacement.mut_z().xor_with(old.z());
        replacement.add_phase((4U - local.phase() + mapped.phase()) & 3U);
        for (uint32_t local_q = 0; local_q < targets.size(); ++local_q) {
            replacement.set_pauli(targets[local_q], mapped.x().bit_get(local_q),
                                  mapped.z().bit_get(local_q));
        }
        set_row(row_i, replacement.view());
    }
}

void Tableau::prepend_local(const Tableau& gate, std::span<const uint32_t> targets) {
    if (gate.num_qubits() != targets.size()) {
        throw std::invalid_argument("Gate tableau width does not match target count");
    }
    for (uint32_t target : targets) {
        if (target >= num_qubits_) {
            throw std::invalid_argument("Gate target is outside the tableau");
        }
    }

    const size_t replacement_count = 2 * targets.size();
    const size_t scratch_words = replacement_count * num_words_;
    if (prepend_scratch_x_.size() < scratch_words) {
        prepend_scratch_x_.resize(scratch_words);
        prepend_scratch_z_.resize(scratch_words);
    }
    if (prepend_scratch_phases_.size() < replacement_count) {
        prepend_scratch_phases_.resize(replacement_count);
    }

    size_t replacement = 0;
    for (bool z_generator : {false, true}) {
        for (uint32_t local_q = 0; local_q < targets.size(); ++local_q) {
            const PauliStringView gate_row =
                z_generator ? gate.z_output(local_q) : gate.x_output(local_q);
            const size_t offset = replacement * num_words_;
            MutableMaskView mapped_x{
                std::span<uint64_t>(prepend_scratch_x_).subspan(offset, num_words_)};
            MutableMaskView mapped_z{
                std::span<uint64_t>(prepend_scratch_z_).subspan(offset, num_words_)};
            mapped_x.zero_out();
            mapped_z.zero_out();
            prepend_scratch_phases_[replacement] = gate_row.phase();
            for (uint32_t k = 0; k < targets.size(); ++k) {
                if (gate_row.x().bit_get(k)) {
                    right_multiply_masks(mapped_x, mapped_z, prepend_scratch_phases_[replacement],
                                         x_output(targets[k]));
                }
            }
            for (uint32_t k = 0; k < targets.size(); ++k) {
                if (gate_row.z().bit_get(k)) {
                    right_multiply_masks(mapped_x, mapped_z, prepend_scratch_phases_[replacement],
                                         z_output(targets[k]));
                }
            }
            ++replacement;
        }
    }
    replacement = 0;
    for (bool z_generator : {false, true}) {
        for (uint32_t target : targets) {
            const size_t offset = replacement * num_words_;
            set_row(
                row_index(z_generator, target, num_qubits_),
                PauliStringView{
                    MaskView{
                        std::span<const uint64_t>(prepend_scratch_x_).subspan(offset, num_words_)},
                    MaskView{
                        std::span<const uint64_t>(prepend_scratch_z_).subspan(offset, num_words_)},
                    prepend_scratch_phases_[replacement], num_qubits_});
            ++replacement;
        }
    }
}

void Tableau::append_named_gate(GateType gate, std::span<const uint32_t> targets) {
    append_local(from_named_gate(gate), targets);
}

void Tableau::prepend_named_gate(GateType gate, std::span<const uint32_t> targets) {
    const Tableau& gate_tableau = from_named_gate(gate);
    if (gate_tableau.num_qubits() != targets.size()) {
        throw std::invalid_argument("Gate tableau width does not match target count");
    }
    for (uint32_t target : targets) {
        if (target >= num_qubits_) {
            throw std::invalid_argument("Gate target is outside the tableau");
        }
    }

    // Common traced gates have row-local updates that avoid generic composition scratch.
    switch (gate) {
        case GateType::I:
        case GateType::II:
            return;
        case GateType::H: {
            const uint32_t x_index = row_index(false, targets[0], num_qubits_);
            const uint32_t z_index = row_index(true, targets[0], num_qubits_);
            const size_t x_offset = static_cast<size_t>(x_index) * num_words_;
            const size_t z_offset = static_cast<size_t>(z_index) * num_words_;
            for (uint32_t w = 0; w < num_words_; ++w) {
                std::swap(x_rows_[x_offset + w], x_rows_[z_offset + w]);
                std::swap(z_rows_[x_offset + w], z_rows_[z_offset + w]);
            }
            std::swap(phases_[x_index], phases_[z_index]);
            return;
        }
        case GateType::S:
        case GateType::S_DAG:
            right_multiply_row(row_index(false, targets[0], num_qubits_),
                               row_index(true, targets[0], num_qubits_),
                               gate == GateType::S ? 1U : 3U);
            return;
        case GateType::X:
            phases_[row_index(true, targets[0], num_qubits_)] ^= 2U;
            return;
        case GateType::Y:
            phases_[row_index(false, targets[0], num_qubits_)] ^= 2U;
            phases_[row_index(true, targets[0], num_qubits_)] ^= 2U;
            return;
        case GateType::Z:
            phases_[row_index(false, targets[0], num_qubits_)] ^= 2U;
            return;
        case GateType::CX:
            if (targets[0] != targets[1]) {
                right_multiply_row(row_index(false, targets[0], num_qubits_),
                                   row_index(false, targets[1], num_qubits_));
                right_multiply_row(row_index(true, targets[1], num_qubits_),
                                   row_index(true, targets[0], num_qubits_));
                return;
            }
            prepend_local(gate_tableau, targets);
            return;
        case GateType::CZ:
            if (targets[0] != targets[1]) {
                right_multiply_row(row_index(false, targets[0], num_qubits_),
                                   row_index(true, targets[1], num_qubits_));
                right_multiply_row(row_index(false, targets[1], num_qubits_),
                                   row_index(true, targets[0], num_qubits_));
                return;
            }
            prepend_local(gate_tableau, targets);
            return;
        default:
            prepend_local(gate_tableau, targets);
    }
}

void Tableau::prepend_pauli(PauliStringView axis) {
    if (axis.num_qubits() != num_qubits_ || !axis.is_hermitian()) {
        throw std::invalid_argument("Pauli axis does not match the tableau");
    }
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (axis.z().bit_get(q)) {
            phases_[row_index(false, q, num_qubits_)] ^= 2U;
        }
        if (axis.x().bit_get(q)) {
            phases_[row_index(true, q, num_qubits_)] ^= 2U;
        }
    }
}

void Tableau::prepend_pauli_rotation(PauliStringView axis, bool dagger) {
    if (axis.num_qubits() != num_qubits_ || !axis.is_hermitian()) {
        throw std::invalid_argument("Pauli rotation axis does not match the tableau");
    }
    const PauliString mapped_axis = apply(axis);
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        for (bool z_generator : {false, true}) {
            const uint32_t index = row_index(z_generator, q, num_qubits_);
            const bool anticommutes = z_generator ? axis.x().bit_get(q) : axis.z().bit_get(q);
            if (!anticommutes) {
                continue;
            }
            PauliString mapped = mapped_axis;
            mapped.right_multiply(row(index));
            mapped.add_phase(dagger ? 1 : 3);
            set_row(index, mapped.view());
        }
    }
}

bool Tableau::operator==(const Tableau& other) const {
    return num_qubits_ == other.num_qubits_ && x_rows_ == other.x_rows_ &&
           z_rows_ == other.z_rows_ && phases_ == other.phases_;
}

}  // namespace clifft
