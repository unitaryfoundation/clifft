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
// remaining |0> qubits: rows[q] and rows[n + q] hold the images of X_q and Z_q.
// Gates and Clifford rotations compose into this inverse frame; measurements
// and activations change its output coordinates. The owner holds the dense
// coefficients and passes their active width where a method needs it.
class OnlineFrame {
  public:
    uint32_t n;
    std::vector<PauliString> rows;
    PauliString scratch;
    PauliString body;
    std::array<PauliString, 4> old;
    explicit OnlineFrame(uint32_t count)
        : n(count),
          scratch(count),
          body(count),
          old{PauliString(count), PauliString(count), PauliString(count), PauliString(count)} {
        rows.reserve(2 * static_cast<size_t>(n));
        for (uint32_t i = 0; i < 2 * n; ++i)
            rows.emplace_back(n);
        reset();
    }
    void reset() noexcept {
        for (uint32_t i = 0; i < 2 * n; ++i) {
            auto& r = rows[i];
            r.mut_x().zero_out();
            r.mut_z().zero_out();
            r.set_phase(0);
            r.set_pauli(i % n, i < n, i >= n);
        }
    }
    void axis(uint32_t q, char kind, PauliString& output) const noexcept {
        copy_pauli(rows[kind == 'Z' ? n + q : q].view(), output);
        if (kind == 'Y') {
            output.right_multiply(rows[n + q].view());
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
                    output.right_multiply(rows[q].view());
                if ((z.words[w] >> (q % 64)) & 1)
                    output.right_multiply(rows[n + q].view());
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
                std::swap(rows[a], rows[n + a]);
                return;
            case GateType::S:
            case GateType::S_DAG:
                rows[a].right_multiply(rows[n + a].view());
                rows[a].add_phase(gate == GateType::S ? 3 : 1);
                return;
            case GateType::X:
                rows[n + a].negate();
                return;
            case GateType::Z:
                rows[a].negate();
                return;
            case GateType::Y:
                rows[a].negate();
                rows[n + a].negate();
                return;
            case GateType::CX:
                rows[a].right_multiply(rows[b].view());
                rows[n + b].right_multiply(rows[n + a].view());
                return;
            case GateType::CZ:
                rows[a].right_multiply(rows[n + b].view());
                rows[b].right_multiply(rows[n + a].view());
                return;
            case GateType::SWAP:
                std::swap(rows[a], rows[b]);
                std::swap(rows[n + a], rows[n + b]);
                return;
            default:
                break;
        }
        const uint32_t count = inverse.count;
        assert(count == 1 || count == 2);
        const uint32_t targets[2] = {a, b};
        for (uint32_t i = 0; i < count; ++i) {
            copy_pauli(rows[targets[i]].view(), old[i]);
            copy_pauli(rows[n + targets[i]].view(), old[count + i]);
        }
        for (uint32_t i = 0; i < 2 * count; ++i) {
            const auto input = inverse.rows[i];
            auto& output = rows[(i < count ? 0 : n) + targets[i % count]];
            output.mut_x().zero_out();
            output.mut_z().zero_out();
            output.set_phase(input.phase);
            for (uint32_t j = 0; j < count; ++j)
                if ((input.x & (1U << j)) != 0)
                    output.right_multiply(old[j].view());
            for (uint32_t j = 0; j < count; ++j)
                if ((input.z & (1U << j)) != 0)
                    output.right_multiply(old[count + j].view());
        }
    }
    void coordinate_h(uint32_t q) noexcept {
        for (auto& r : rows) {
            bool x = r.x().bit_get(q), z = r.z().bit_get(q);
            if (x && z)
                r.negate();
            r.set_pauli(q, z, x);
        }
    }
    void coordinate_x(uint32_t q) noexcept {
        for (auto& r : rows)
            if (r.z().bit_get(q))
                r.negate();
    }
    void coordinate_cx(uint32_t a, uint32_t b) noexcept {
        for (auto& r : rows) {
            bool xa = r.x().bit_get(a), za = r.z().bit_get(a);
            bool xb = r.x().bit_get(b), zb = r.z().bit_get(b);
            r.set_pauli(a, xa, za ^ zb);
            r.set_pauli(b, xb ^ xa, zb);
        }
    }
    void coordinate_swap(uint32_t a, uint32_t b) noexcept {
        if (a == b)
            return;
        for (auto& r : rows) {
            bool xa = r.x().bit_get(a), za = r.z().bit_get(a);
            r.set_pauli(a, r.x().bit_get(b), r.z().bit_get(b));
            r.set_pauli(b, xa, za);
        }
    }
    void compact(uint32_t pivot, uint32_t width) noexcept {
        // Dense measurement kernels remove the pivot while preserving other bit order.
        for (auto& r : rows) {
            bool xp = r.x().bit_get(pivot), zp = r.z().bit_get(pivot);
            for (uint32_t q = pivot; q + 1 < width; ++q)
                r.set_pauli(q, r.x().bit_get(q + 1), r.z().bit_get(q + 1));
            r.set_pauli(width - 1, xp, zp);
        }
    }
    void localize_x(PauliStringView p, uint32_t pivot) noexcept {
        // U = S_p^z controlled(Q), where P = (X or Y)_p Q. U fixes
        // states with dormant p=0. Apply U^-1 to every inverse-frame row.
        copy_pauli(p, scratch);
        auto& q = scratch;
        q.set_pauli(pivot, false, false);
        q.set_sign(false);
        const bool yp = p.z().bit_get(pivot);
        for (auto& r : rows) {
            const bool x = r.x().bit_get(pivot);
            if (yp && x) {
                r.set_pauli(pivot, true, !r.z().bit_get(pivot));
                r.add_phase(3);
            }
            const bool anti = !r.view().commutes(q.view());
            if (x)
                r.right_multiply(q.view());
            if (anti) {
                r.set_pauli(pivot, x, !r.z().bit_get(pivot));
                if (x)
                    r.negate();
            }
        }
    }
    // Makes p the X generator of the next active coordinate. The owner then
    // applies the rotation or filter that needs that coordinate.
    void promote(PauliStringView p, uint32_t dormant, uint32_t active_width) noexcept {
        localize_x(p, dormant);
        coordinate_swap(dormant, active_width);
    }
    // Follows a dense measurement of the active Pauli a that collapsed onto
    // branch and removed coordinate `removed` from a state of the given width.
    void finish_active_measurement(ActivePauli a, uint32_t removed, uint32_t width,
                                   bool branch) noexcept {
        body.mut_x().zero_out();
        body.mut_z().zero_out();
        if (width) {
            body.mut_x().words[0] = a.x;
            body.mut_z().words[0] = a.z;
        }
        body.set_sign(false);
        if (a.x) {
            localize_x(body.view(), removed);
            coordinate_h(removed);
        } else {
            for (uint32_t q = 0; q < width; ++q)
                if (q != removed && ((a.z >> q) & 1))
                    coordinate_cx(q, removed);
        }
        if (branch)
            coordinate_x(removed);
        compact(removed, width);
    }
    void measure_dormant(PauliStringView p, uint32_t pivot, bool branch) noexcept {
        // Fuse U^-1, H, and the sampled X correction. The controlled-Pauli
        // commutator and product phase share the same packed-word traversal.
        copy_pauli(p, scratch);
        auto& q = scratch;
        q.set_pauli(pivot, false, false);
        q.set_sign(false);
        const bool yp = p.z().bit_get(pivot);
        for (auto& r : rows) {
            const bool x = r.x().bit_get(pivot);
            const bool z = r.z().bit_get(pivot);
            bool anti = false, crossing = false;
            auto rx = r.mut_x();
            auto rz = r.mut_z();
            for (uint32_t w = 0; w < rx.num_words(); ++w) {
                const uint64_t zx = rz.words[w] & q.x().words[w];
                anti ^= (std::popcount((rx.words[w] & q.z().words[w]) ^ zx) & 1) != 0;
                crossing ^= (std::popcount(zx) & 1) != 0;
                if (x) {
                    rx.words[w] ^= q.x().words[w];
                    rz.words[w] ^= q.z().words[w];
                }
            }
            const bool localized_z = z ^ (yp && x) ^ anti;
            unsigned phase = r.phase();
            if (x) {
                phase += (yp ? 3 : 0) + q.phase() + (crossing ? 2 : 0);
                phase += (anti ? 2 : 0) + (localized_z ? 2 : 0) + (branch ? 2 : 0);
            }
            r.set_pauli(pivot, localized_z, x);
            r.set_phase(phase);
        }
    }
    void rotate_clifford(PauliStringView p, CliffordRotation rotation) noexcept {
        if (rotation == CliffordRotation::IDENTITY)
            return;
        for (auto& row : rows) {
            if (!row.view().commutes(p))
                rotate_row(row, p, rotation);
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
                rotate_row(rows[64 * w + static_cast<uint32_t>(std::countr_zero(bits))], p,
                           rotation);
            for (uint64_t bits = p.x().words[w]; bits != 0; bits &= bits - 1)
                rotate_row(rows[n + 64 * w + static_cast<uint32_t>(std::countr_zero(bits))], p,
                           rotation);
        }
    }

  private:
    // Conjugates a row that anticommutes with p by the rotation about p.
    static void rotate_row(PauliString& row, PauliStringView p,
                           CliffordRotation rotation) noexcept {
        if (rotation == CliffordRotation::PAULI) {
            row.negate();
        } else {
            row.right_multiply(p);
            row.add_phase(rotation == CliffordRotation::SQRT ? 3 : 1);
        }
    }
};

}  // namespace clifft::sampling
