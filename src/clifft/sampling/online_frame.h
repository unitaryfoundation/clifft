#pragma once

#include "clifft/circuit/gate_data.h"
#include "clifft/sampling/plan.h"
#include "clifft/tableau/pauli_string.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling {

// Pauli on a gate's one or two operands, encoded as i^phase X^x Z^z. Keeping
// these local lets the same prepared gate action apply to any physical targets.
struct LocalPauli {
    uint8_t x = 0;
    uint8_t z = 0;
    uint8_t phase = 0;
};

// Images of X then Z generators under an inverse Clifford gate. Prepared once
// and composed into OnlineFrame when that gate executes during a shot.
struct LocalClifford {
    uint32_t count = 0;
    std::array<LocalPauli, 4> rows{};
};

inline void copy_pauli(PauliStringView source, PauliString& destination) noexcept {
    assert(source.num_qubits() == destination.num_qubits());
    std::copy(source.x().words.begin(), source.x().words.end(), destination.mut_x().words.begin());
    std::copy(source.z().words.begin(), source.z().words.end(), destination.mut_z().words.begin());
    destination.set_phase(source.phase());
}

// A Clifford frame that changes during a shot. It maps Paulis on n input
// coordinates into the coordinates of a dense active state followed by
// remaining |0> qubits: rows q and n + q hold the images of X_q and Z_q.
// Gates and Clifford rotations compose into this inverse frame; measurements
// and activations change its output coordinates. The owner holds the dense
// coefficients and passes their active width where a method needs it.
//
// The tableau is one array. Each row is its X words followed by its Z words,
// a Pauli i^phase X^x Z^z, with the phases in a second array. A few rows past
// the tableau are scratch space.
class OnlineFrame {
  public:
    uint32_t n;

    explicit OnlineFrame(uint32_t count)
        : n(count),
          words_((count + 63) / 64),
          storage_((size_t{2} * count + kScratchRows) * 2 * words_),
          phases_(size_t{2} * count + kScratchRows) {
        reset();
    }
    void reset() noexcept {
        std::fill_n(storage_.begin(), size_t{2} * n * 2 * words_, uint64_t{0});
        std::fill_n(phases_.begin(), size_t{2} * n, uint8_t{0});
        for (uint32_t q = 0; q < n; ++q) {
            x_words(q)[q / 64] |= uint64_t{1} << (q % 64);
            z_words(n + q)[q / 64] |= uint64_t{1} << (q % 64);
        }
    }
    // Row q is the image of X_q and row n + q the image of Z_q.
    [[nodiscard]] PauliStringView row(uint32_t index) const noexcept {
        return {MaskView{std::span<const uint64_t>(x_words(index), words_)},
                MaskView{std::span<const uint64_t>(z_words(index), words_)}, phases_[index], n};
    }
    // Takes the tableau of another frame over the same coordinates.
    void copy_tableau(const OnlineFrame& source) noexcept {
        assert(source.n == n);
        std::copy_n(source.storage_.begin(), size_t{2} * n * 2 * words_, storage_.begin());
        std::copy_n(source.phases_.begin(), size_t{2} * n, phases_.begin());
    }
    void axis(uint32_t q, char kind, PauliString& output) const noexcept {
        copy_pauli(row(kind == 'Z' ? n + q : q), output);
        if (kind == 'Y') {
            output.right_multiply(row(n + q));
            output.add_phase(1);
        }
    }
    // Maps the Hermitian input Pauli i^popcount(x & z) X^x Z^z into frame
    // coordinates. Bits at or above n must be clear.
    void map(MaskView x, MaskView z, PauliString& output) const noexcept {
        assert(output.num_qubits() == n && x.num_words() == z.num_words());
        output.mut_x().zero_out();
        output.mut_z().zero_out();
        output.set_phase(0);
        unsigned y_count = 0;
        for (uint32_t w = 0; w < x.num_words(); ++w) {
            y_count += static_cast<unsigned>(std::popcount(x.words[w] & z.words[w]));
            for (uint64_t bits = x.words[w] | z.words[w]; bits != 0; bits &= bits - 1) {
                const uint32_t q = 64 * w + static_cast<uint32_t>(std::countr_zero(bits));
                assert(q < n);
                if ((x.words[w] >> (q % 64)) & 1)
                    output.right_multiply(row(q));
                if ((z.words[w] >> (q % 64)) & 1)
                    output.right_multiply(row(n + q));
            }
        }
        output.add_phase(static_cast<uint8_t>(y_count & 3U));
    }
    // First dormant coordinate where p has X or Y support. Such a Pauli has
    // expectation zero, and measuring it gives an unbiased outcome.
    static std::optional<uint32_t> dormant_x(PauliStringView p, uint32_t active_width) noexcept {
        for (uint32_t w = active_width / 64; w < p.x().num_words(); ++w) {
            uint64_t bits = p.x().words[w];
            if (w == active_width / 64)
                bits &= ~uint64_t{0} << (active_width % 64);
            if (bits)
                return 64 * w + std::countr_zero(bits);
        }
        return std::nullopt;
    }
    // Restriction of p to the active coordinates. Dormant Z factors act as the
    // identity on |0>, so this is the complete action when dormant_x is empty.
    static ActivePauli active(PauliStringView p, uint32_t active_width) noexcept {
        const uint64_t mask = (uint64_t{1} << active_width) - 1;
        return {p.x().words.empty() ? 0 : p.x().words[0] & mask,
                p.z().words.empty() ? 0 : p.z().words[0] & mask};
    }
    // Coordinate removed by a dense measurement of an active Pauli.
    static uint32_t measurement_pivot(ActivePauli p) noexcept {
        return p.x ? 63 - std::countl_zero(p.x) : std::countr_zero(p.z);
    }
    void physical(GateType gate, uint32_t a, uint32_t b, const LocalClifford& inverse) noexcept {
        // Conjugate physical input by the inverse gate before the inverse frame.
        switch (gate) {
            case GateType::H:
                swap_rows(a, n + a);
                return;
            case GateType::S:
            case GateType::S_DAG:
                multiply(a, n + a);
                add_phase(a, gate == GateType::S ? 3 : 1);
                return;
            case GateType::X:
                negate(n + a);
                return;
            case GateType::Z:
                negate(a);
                return;
            case GateType::Y:
                negate(a);
                negate(n + a);
                return;
            case GateType::CX:
                multiply(a, b);
                multiply(n + b, n + a);
                return;
            case GateType::CZ:
                multiply(a, n + b);
                multiply(b, n + a);
                return;
            case GateType::SWAP:
                swap_rows(a, b);
                swap_rows(n + a, n + b);
                return;
            default:
                break;
        }
        const uint32_t count = inverse.count;
        assert(count == 1 || count == 2);
        const uint32_t targets[2] = {a, b};
        const uint32_t old = 2 * n + kOldRows;
        for (uint32_t i = 0; i < count; ++i) {
            copy_row(old + i, targets[i]);
            copy_row(old + count + i, n + targets[i]);
        }
        for (uint32_t i = 0; i < 2 * count; ++i) {
            const auto input = inverse.rows[i];
            const uint32_t output = (i < count ? 0 : n) + targets[i % count];
            std::fill_n(x_words(output), size_t{2} * words_, uint64_t{0});
            phases_[output] = input.phase & 3U;
            for (uint32_t j = 0; j < count; ++j)
                if ((input.x & (1U << j)) != 0)
                    multiply(output, old + j);
            for (uint32_t j = 0; j < count; ++j)
                if ((input.z & (1U << j)) != 0)
                    multiply(output, old + count + j);
        }
    }
    void coordinate_h(uint32_t q) noexcept {
        const size_t word = q / 64;
        const uint64_t bit = uint64_t{1} << (q % 64);
        for (uint32_t i = 0; i < 2 * n; ++i) {
            uint64_t& x = x_words(i)[word];
            uint64_t& z = z_words(i)[word];
            if ((x & z & bit) != 0)
                negate(i);
            const uint64_t differ = (x ^ z) & bit;
            x ^= differ;
            z ^= differ;
        }
    }
    void coordinate_x(uint32_t q) noexcept {
        const size_t word = q / 64;
        const uint64_t bit = uint64_t{1} << (q % 64);
        for (uint32_t i = 0; i < 2 * n; ++i)
            if ((z_words(i)[word] & bit) != 0)
                negate(i);
    }
    void coordinate_cx(uint32_t a, uint32_t b) noexcept {
        const size_t a_word = a / 64, b_word = b / 64;
        const uint64_t a_bit = uint64_t{1} << (a % 64), b_bit = uint64_t{1} << (b % 64);
        for (uint32_t i = 0; i < 2 * n; ++i) {
            uint64_t* x = x_words(i);
            uint64_t* z = z_words(i);
            if ((x[a_word] & a_bit) != 0)
                x[b_word] ^= b_bit;
            if ((z[b_word] & b_bit) != 0)
                z[a_word] ^= a_bit;
        }
    }
    void coordinate_swap(uint32_t a, uint32_t b) noexcept {
        if (a == b)
            return;
        const size_t a_word = a / 64, b_word = b / 64;
        const uint64_t a_bit = uint64_t{1} << (a % 64), b_bit = uint64_t{1} << (b % 64);
        // Both masks of a row follow its X words, so one pass swaps the bits
        // of every mask.
        for (uint64_t *mask = storage_.data(), *end = mask + size_t{2} * n * 2 * words_;
             mask != end; mask += words_) {
            if (((mask[a_word] & a_bit) != 0) != ((mask[b_word] & b_bit) != 0)) {
                mask[a_word] ^= a_bit;
                mask[b_word] ^= b_bit;
            }
        }
    }
    void compact(uint32_t pivot, uint32_t width) noexcept {
        // Dense measurement kernels remove the pivot while preserving other bit
        // order: the coordinates above it move down and it takes the last
        // place. Active coordinates fit one word.
        assert(pivot < width && width <= 64);
        const uint64_t below = (uint64_t{1} << pivot) - 1;
        const uint64_t through = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
        const uint64_t above = through & ~below & ~(uint64_t{1} << pivot);
        for (uint64_t *mask = storage_.data(), *end = mask + size_t{2} * n * 2 * words_;
             mask != end; mask += words_) {
            const uint64_t word = mask[0];
            mask[0] = (word & (below | ~through)) | ((word & above) >> 1) |
                      (((word >> pivot) & 1) << (width - 1));
        }
    }
    void localize_x(PauliStringView p, uint32_t pivot) noexcept {
        // U = S_p^z controlled(Q), where P = (X or Y)_p Q. U fixes
        // states with dormant p=0. Apply U^-1 to every inverse-frame row.
        const uint32_t q = load_without(p, pivot);
        const uint64_t* qx = x_words(q);
        const uint64_t* qz = z_words(q);
        const bool yp = p.z().bit_get(pivot);
        const size_t word = pivot / 64;
        const uint64_t bit = uint64_t{1} << (pivot % 64);
        for (uint32_t i = 0; i < 2 * n; ++i) {
            uint64_t* x = x_words(i);
            uint64_t* z = z_words(i);
            const bool has_x = (x[word] & bit) != 0;
            if (yp && has_x) {
                z[word] ^= bit;
                add_phase(i, 3);
            }
            const bool anti = anticommute(x, z, qx, qz);
            if (has_x)
                multiply(i, qx, qz, phases_[q]);
            if (anti) {
                z[word] ^= bit;
                if (has_x)
                    negate(i);
            }
        }
    }
    // Makes p the X generator of the next active coordinate. The owner then
    // applies the rotation or filter that needs that coordinate.
    void promote(PauliStringView p, uint32_t dormant, uint32_t active_width) noexcept {
        localize_x(p, dormant);
        coordinate_swap(dormant, active_width);
    }
    // Changes coordinates so that the active Pauli a of a state of the given
    // width becomes Z on coordinate `removed`.
    void localize_measured(ActivePauli a, uint32_t removed, uint32_t width) noexcept {
        if (a.x) {
            const uint32_t body = 2 * n + kBodyRow;
            std::fill_n(x_words(body), size_t{2} * words_, uint64_t{0});
            x_words(body)[0] = a.x;
            z_words(body)[0] = a.z;
            phases_[body] = static_cast<uint8_t>(std::popcount(a.x & a.z) & 3);
            localize_x(row(body), removed);
            coordinate_h(removed);
        } else {
            for (uint32_t q = 0; q < width; ++q)
                if (q != removed && ((a.z >> q) & 1))
                    coordinate_cx(q, removed);
        }
    }
    // Follows a dense measurement of the active Pauli a that collapsed onto
    // branch and removed coordinate `removed` from a state of the given width.
    // The branch only negates the rows with Z on the localized coordinate.
    void finish_active_measurement(ActivePauli a, uint32_t removed, uint32_t width,
                                   bool branch) noexcept {
        localize_measured(a, removed, width);
        if (branch)
            coordinate_x(removed);
        compact(removed, width);
    }
    void measure_dormant(PauliStringView p, uint32_t pivot, bool branch) noexcept {
        // Fuse U^-1, H, and the sampled X correction. The controlled-Pauli
        // commutator and product phase share the same packed-word traversal.
        const uint32_t q = load_without(p, pivot);
        const uint64_t* qx = x_words(q);
        const uint64_t* qz = z_words(q);
        const unsigned q_phase = phases_[q];
        const bool yp = p.z().bit_get(pivot);
        const size_t word = pivot / 64;
        const uint64_t bit = uint64_t{1} << (pivot % 64);
        for (uint32_t i = 0; i < 2 * n; ++i) {
            uint64_t* rx = x_words(i);
            uint64_t* rz = z_words(i);
            const bool x = (rx[word] & bit) != 0;
            const bool z = (rz[word] & bit) != 0;
            unsigned anti_count = 0, crossing_count = 0;
            for (uint32_t w = 0; w < words_; ++w) {
                const uint64_t zx = rz[w] & qx[w];
                anti_count ^= static_cast<unsigned>(std::popcount((rx[w] & qz[w]) ^ zx));
                crossing_count ^= static_cast<unsigned>(std::popcount(zx));
                if (x) {
                    rx[w] ^= qx[w];
                    rz[w] ^= qz[w];
                }
            }
            const bool anti = (anti_count & 1U) != 0;
            const bool localized_z = z ^ (yp && x) ^ anti;
            unsigned phase = phases_[i];
            if (x) {
                phase += (yp ? 3 : 0) + q_phase + ((crossing_count & 1U) != 0 ? 2 : 0);
                phase += (anti ? 2 : 0) + (localized_z ? 2 : 0) + (branch ? 2 : 0);
            }
            rx[word] = (rx[word] & ~bit) | (localized_z ? bit : 0);
            rz[word] = (rz[word] & ~bit) | (x ? bit : 0);
            phases_[i] = static_cast<uint8_t>(phase & 3U);
        }
    }
    void rotate_clifford(PauliStringView p, CliffordRotation rotation) noexcept {
        if (rotation == CliffordRotation::IDENTITY)
            return;
        assert(p.x().num_words() == words_);
        const uint64_t* px = p.x().words.data();
        const uint64_t* pz = p.z().words.data();
        for (uint32_t i = 0; i < 2 * n; ++i) {
            if (anticommute(x_words(i), z_words(i), px, pz))
                rotate_row(i, p, rotation);
        }
    }
    // rotate_clifford for an input Pauli p when the frame so far is the
    // identity composed with rotations that commute with p. The frame then
    // maps p to itself, and a row anticommutes with p exactly when its
    // generator does: the X_q image where p has Z on q, the Z_q image where p
    // has X on q. Only those rows change.
    void rotate_fixed_clifford(PauliStringView p, CliffordRotation rotation) noexcept {
        assert(rotation != CliffordRotation::IDENTITY);
        for (uint32_t w = 0; w < p.x().num_words(); ++w) {
            for (uint64_t bits = p.z().words[w]; bits != 0; bits &= bits - 1)
                rotate_row(64 * w + static_cast<uint32_t>(std::countr_zero(bits)), p, rotation);
            for (uint64_t bits = p.x().words[w]; bits != 0; bits &= bits - 1)
                rotate_row(n + 64 * w + static_cast<uint32_t>(std::countr_zero(bits)), p, rotation);
        }
    }
    // rotate_fixed_clifford for a Pauli X^x when no earlier rotation had a Z
    // factor either. The image of Z_q is then Z_q times X factors, and it
    // anticommutes with the Pauli exactly when x has q. Multiplying it by the
    // Pauli adds the X words and, for the one Z it moves past, a sign, which
    // joins the rotation's own phase: one quarter for SQRT, three for
    // SQRT_DAG, two for the Pauli.
    void rotate_fixed_x_clifford(MaskView x, CliffordRotation rotation) noexcept {
        assert(rotation != CliffordRotation::IDENTITY && x.num_words() == words_);
        const unsigned phase = rotation == CliffordRotation::PAULI  ? 2
                               : rotation == CliffordRotation::SQRT ? 1
                                                                    : 3;
        for (uint32_t w = 0; w < words_; ++w) {
            for (uint64_t bits = x.words[w]; bits != 0; bits &= bits - 1) {
                const uint32_t q = 64 * w + static_cast<uint32_t>(std::countr_zero(bits));
                uint64_t* row_x = x_words(n + q);
                assert(std::popcount(z_words(n + q)[w]) == 1 &&
                       ((z_words(n + q)[w] >> (q % 64)) & 1) != 0 &&
                       "the image of Z_q keeps Z on q alone");
                if (rotation != CliffordRotation::PAULI) {
                    for (uint32_t v = 0; v < words_; ++v)
                        row_x[v] ^= x.words[v];
                }
                add_phase(n + q, phase);
            }
        }
    }

  private:
    // Scratch rows past the tableau: the Pauli a localization works with, the
    // active Pauli of a measurement, and the rows a general gate replaces.
    static constexpr uint32_t kScratchRow = 0;
    static constexpr uint32_t kBodyRow = 1;
    static constexpr uint32_t kOldRows = 2;
    static constexpr uint32_t kScratchRows = 6;

    [[nodiscard]] uint64_t* x_words(uint32_t row) noexcept {
        return storage_.data() + size_t{row} * 2 * words_;
    }
    [[nodiscard]] const uint64_t* x_words(uint32_t row) const noexcept {
        return storage_.data() + size_t{row} * 2 * words_;
    }
    [[nodiscard]] uint64_t* z_words(uint32_t row) noexcept { return x_words(row) + words_; }
    [[nodiscard]] const uint64_t* z_words(uint32_t row) const noexcept {
        return x_words(row) + words_;
    }
    void negate(uint32_t row) noexcept { phases_[row] ^= 2; }
    void add_phase(uint32_t row, unsigned phase) noexcept {
        phases_[row] = static_cast<uint8_t>((phases_[row] + phase) & 3U);
    }
    void copy_row(uint32_t destination, uint32_t source) noexcept {
        std::copy_n(x_words(source), size_t{2} * words_, x_words(destination));
        phases_[destination] = phases_[source];
    }
    void swap_rows(uint32_t a, uint32_t b) noexcept {
        std::swap_ranges(x_words(a), x_words(a) + size_t{2} * words_, x_words(b));
        std::swap(phases_[a], phases_[b]);
    }
    [[nodiscard]] bool anticommute(const uint64_t* x, const uint64_t* z, const uint64_t* px,
                                   const uint64_t* pz) const noexcept {
        unsigned parity = 0;
        for (uint32_t w = 0; w < words_; ++w)
            parity ^= static_cast<unsigned>(std::popcount((x[w] & pz[w]) ^ (z[w] & px[w])));
        return (parity & 1U) != 0;
    }
    // Multiplies a row on the right by the Pauli i^phase X^fx Z^fz.
    void multiply(uint32_t row, const uint64_t* fx, const uint64_t* fz, unsigned phase) noexcept {
        uint64_t* x = x_words(row);
        uint64_t* z = z_words(row);
        unsigned crossing = 0;
        for (uint32_t w = 0; w < words_; ++w) {
            crossing ^= static_cast<unsigned>(std::popcount(z[w] & fx[w]));
            x[w] ^= fx[w];
            z[w] ^= fz[w];
        }
        add_phase(row, phase + ((crossing & 1U) != 0 ? 2U : 0U));
    }
    void multiply(uint32_t row, uint32_t factor) noexcept {
        assert(row != factor);
        multiply(row, x_words(factor), z_words(factor), phases_[factor]);
    }
    // Copies p into the scratch row as an unsigned Hermitian Pauli without its
    // factor on the pivot, and returns that row.
    [[nodiscard]] uint32_t load_without(PauliStringView p, uint32_t pivot) noexcept {
        assert(p.x().num_words() == words_);
        const uint32_t q = 2 * n + kScratchRow;
        uint64_t* qx = x_words(q);
        uint64_t* qz = z_words(q);
        std::copy_n(p.x().words.data(), words_, qx);
        std::copy_n(p.z().words.data(), words_, qz);
        qx[pivot / 64] &= ~(uint64_t{1} << (pivot % 64));
        qz[pivot / 64] &= ~(uint64_t{1} << (pivot % 64));
        unsigned y_count = 0;
        for (uint32_t w = 0; w < words_; ++w)
            y_count += static_cast<unsigned>(std::popcount(qx[w] & qz[w]));
        phases_[q] = static_cast<uint8_t>(y_count & 3U);
        return q;
    }
    // Conjugates a row that anticommutes with p by the rotation about p.
    void rotate_row(uint32_t row, PauliStringView p, CliffordRotation rotation) noexcept {
        if (rotation == CliffordRotation::PAULI) {
            negate(row);
        } else {
            multiply(row, p.x().words.data(), p.z().words.data(), p.phase());
            add_phase(row, rotation == CliffordRotation::SQRT ? 3 : 1);
        }
    }

    // Words per X or Z mask of a row.
    uint32_t words_;
    std::vector<uint64_t> storage_;
    std::vector<uint8_t> phases_;
};

}  // namespace clifft::sampling
