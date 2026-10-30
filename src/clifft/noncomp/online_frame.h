#pragma once

#include "clifft/circuit/gate_data.h"
#include "clifft/tableau/pauli_string.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>

namespace clifft::noncomp_detail {

struct LocalPauli {
    uint8_t x = 0;
    uint8_t z = 0;
    uint8_t phase = 0;
};

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

// Rows map physical Paulis into current active/dormant coordinates. Physical
// Cliffords combine a few rows; measurement basis changes transform the rows.
class OnlineFrame {
  public:
    uint32_t n;
    std::vector<PauliString> rows;
    PauliString scratch;
    std::array<PauliString, 4> old;
    explicit OnlineFrame(uint32_t count)
        : n(count),
          scratch(count),
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
            if (row.view().commutes(p))
                continue;
            if (rotation == CliffordRotation::PAULI) {
                row.negate();
            } else {
                row.right_multiply(p);
                row.add_phase(rotation == CliffordRotation::SQRT ? 3 : 1);
            }
        }
    }
};

}  // namespace clifft::noncomp_detail
