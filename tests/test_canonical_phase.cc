// Canonical tableau phase tracking, validated against stim's dense
// to_flat_unitary_matrix() convention: the S absorption phase used by
// PeepholeFusionPass and the frame composition phase used by lower().

#include "clifft/backend/compiler_context.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/util/canonical_phase.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

using namespace clifft;
using clifft::internal::PendingGate;
using clifft::internal::PendingGateType;
using clifft::internal::VirtualFrame;
using clifft::test::dense_axis_rotation;
using clifft::test::dense_matmul;
using clifft::test::dense_tableau_matrix;
using clifft::test::DenseMatrix;

TEST_CASE("Canonical phase: S absorption matches dense stim oracle") {
    // Fuzz s_absorption_phase against stim's to_flat_unitary_matrix: for a
    // random frame U_C and random signed Pauli P, the exact rotation R must
    // satisfy canonical(U_C) * R == phase * canonical(U_C * S_P)
    // componentwise, where phase is the value folded into global_weight.
    std::mt19937_64 rng(0x139);

    for (int trial = 0; trial < 256; ++trial) {
        CAPTURE(trial);
        const size_t n = 1 + static_cast<size_t>(trial % 4);
        const uint64_t qubit_mask = (uint64_t{1} << n) - 1;
        auto tab = stim::Tableau<kStimWidth>::random(n, rng);

        uint64_t x = 0;
        uint64_t z = 0;
        while (x == 0 && z == 0) {
            x = rng() & qubit_mask;
            z = rng() & qubit_mask;
        }
        const bool sign = (rng() & 1) != 0;
        const bool is_dagger = (rng() & 1) != 0;

        const uint64_t x_words[1] = {x};
        const uint64_t z_words[1] = {z};
        MaskView x_v{std::span<const uint64_t>(x_words)};
        MaskView z_v{std::span<const uint64_t>(z_words)};

        auto updated = tab;
        internal::apply_s_to_tableau(updated, x_v, z_v, sign, is_dagger);
        const auto phase = internal::s_absorption_phase(tab, updated, x_v, z_v, sign, is_dagger);

        const uint64_t dim = uint64_t{1} << n;
        const auto lhs =
            dense_matmul(dense_tableau_matrix(tab),
                         dense_axis_rotation(x, z, sign, is_dagger ? 1.5 : 0.5, n), dim);
        const auto m_new = dense_tableau_matrix(updated);
        for (uint64_t i = 0; i < dim * dim; ++i) {
            const auto rhs = phase * m_new[i];
            CAPTURE(i);
            REQUIRE_THAT(lhs[i].real(), Catch::Matchers::WithinAbs(rhs.real(), 1e-5));
            REQUIRE_THAT(lhs[i].imag(), Catch::Matchers::WithinAbs(rhs.imag(), 1e-5));
        }
    }
}

TEST_CASE("Canonical phase: S absorption is invariant under qubit embedding") {
    // Multi-word phase oracle. The canonical phase delta factorizes over
    // tensor products with identity: the Choi stabilizer group, its
    // minimum-integer support index, and the amplitude walk all decompose
    // over disjoint qubit sets, so embedding the frame and the Pauli at a
    // qubit offset must leave s_absorption_phase unchanged. Offsets
    // straddling bit 64 of a 72-qubit frame drive the 144-bit Choi indices
    // across word boundaries, pinning the multi-word arithmetic to the
    // dense-validated small-frame value.
    std::mt19937_64 rng(0xC0FFEE);
    constexpr size_t k = 3;
    constexpr size_t n_wide = 72;

    for (int trial = 0; trial < 64; ++trial) {
        CAPTURE(trial);
        const size_t offset = 60 + static_cast<size_t>(trial % 8);
        CAPTURE(offset);
        auto small_tab = stim::Tableau<kStimWidth>::random(k, rng);

        uint64_t x = 0;
        uint64_t z = 0;
        while (x == 0 && z == 0) {
            x = rng() & 7;
            z = rng() & 7;
        }
        const bool sign = (rng() & 1) != 0;
        const bool is_dagger = (rng() & 1) != 0;

        auto wide_tab = stim::Tableau<kStimWidth>::identity(n_wide);
        for (size_t a = 0; a < k; ++a) {
            auto wx = wide_tab.xs[offset + a];
            auto wz = wide_tab.zs[offset + a];
            wx.xs[offset + a] = false;
            wz.zs[offset + a] = false;
            for (size_t b = 0; b < k; ++b) {
                wx.xs[offset + b] = small_tab.xs[a].xs[b];
                wx.zs[offset + b] = small_tab.xs[a].zs[b];
                wz.xs[offset + b] = small_tab.zs[a].xs[b];
                wz.zs[offset + b] = small_tab.zs[a].zs[b];
            }
            wx.sign = static_cast<bool>(small_tab.xs[a].sign);
            wz.sign = static_cast<bool>(small_tab.zs[a].sign);
        }

        const uint64_t x_small_words[1] = {x};
        const uint64_t z_small_words[1] = {z};
        MaskView x_small{std::span<const uint64_t>(x_small_words)};
        MaskView z_small{std::span<const uint64_t>(z_small_words)};

        uint64_t x_wide_words[2] = {0, 0};
        uint64_t z_wide_words[2] = {0, 0};
        for (size_t b = 0; b < k; ++b) {
            const size_t bit = offset + b;
            if ((x >> b) & 1) {
                x_wide_words[bit / 64] |= uint64_t{1} << (bit % 64);
            }
            if ((z >> b) & 1) {
                z_wide_words[bit / 64] |= uint64_t{1} << (bit % 64);
            }
        }
        MaskView x_wide{std::span<const uint64_t>(x_wide_words)};
        MaskView z_wide{std::span<const uint64_t>(z_wide_words)};

        auto small_updated = small_tab;
        internal::apply_s_to_tableau(small_updated, x_small, z_small, sign, is_dagger);
        const auto phase_small = internal::s_absorption_phase(small_tab, small_updated, x_small,
                                                              z_small, sign, is_dagger);

        auto wide_updated = wide_tab;
        internal::apply_s_to_tableau(wide_updated, x_wide, z_wide, sign, is_dagger);
        const auto phase_wide =
            internal::s_absorption_phase(wide_tab, wide_updated, x_wide, z_wide, sign, is_dagger);

        REQUIRE_THAT(phase_wide.real(), Catch::Matchers::WithinAbs(phase_small.real(), 1e-9));
        REQUIRE_THAT(phase_wide.imag(), Catch::Matchers::WithinAbs(phase_small.imag(), 1e-9));
    }
}

namespace {

// Canonical matrix of a single pending frame gate, little-endian basis.
DenseMatrix dense_pending_gate(const PendingGate& g, size_t n) {
    const uint64_t dim = uint64_t{1} << n;
    constexpr std::complex<double> kI{0.0, 1.0};
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    DenseMatrix m(dim * dim, {0.0, 0.0});
    for (uint64_t c = 0; c < dim; ++c) {
        const bool b1 = ((c >> g.axis_1) & 1) != 0;
        const bool b2 = ((c >> g.axis_2) & 1) != 0;
        switch (g.type) {
            case PendingGateType::S:
                m[c * dim + c] = b1 ? kI : 1.0;
                break;
            case PendingGateType::CZ:
                m[c * dim + c] = (b1 && b2) ? -1.0 : 1.0;
                break;
            case PendingGateType::CNOT:
                m[(b1 ? c ^ (uint64_t{1} << g.axis_2) : c) * dim + c] = 1.0;
                break;
            case PendingGateType::SWAP: {
                uint64_t r = c;
                if (b1 != b2) {
                    r ^= (uint64_t{1} << g.axis_1) | (uint64_t{1} << g.axis_2);
                }
                m[r * dim + c] = 1.0;
                break;
            }
            case PendingGateType::H: {
                const uint64_t r0 = c & ~(uint64_t{1} << g.axis_1);
                const uint64_t r1 = c | (uint64_t{1} << g.axis_1);
                m[r0 * dim + c] = inv_sqrt2;
                m[r1 * dim + c] = b1 ? -inv_sqrt2 : inv_sqrt2;
                break;
            }
        }
    }
    return m;
}

}  // namespace

TEST_CASE("Canonical phase: frame composition matches dense stim oracle") {
    // For target tableau T and frame gates u_1..u_J (application order),
    // composed = tableau(T * (u_J...u_1)^{-1}) is what lower() stores; the
    // chained phase must satisfy
    //   canonical(composed) * u_J * ... * u_1 == phase * canonical(T)
    // componentwise against stim's dense matrices.
    std::mt19937_64 rng(0x141);

    for (int trial = 0; trial < 128; ++trial) {
        CAPTURE(trial);
        const size_t n = 2 + static_cast<size_t>(trial % 2);
        const auto target = stim::Tableau<kStimWidth>::random(n, rng);

        std::vector<PendingGate> log;
        const size_t num_gates = 1 + rng() % 8;
        for (size_t i = 0; i < num_gates; ++i) {
            PendingGate g{};
            g.type = static_cast<PendingGateType>(rng() % 5);
            g.axis_1 = static_cast<uint16_t>(rng() % n);
            g.axis_2 = static_cast<uint16_t>(rng() % n);
            if ((g.type == PendingGateType::CNOT || g.type == PendingGateType::CZ ||
                 g.type == PendingGateType::SWAP) &&
                g.axis_2 == g.axis_1) {
                g.axis_2 = static_cast<uint16_t>((g.axis_1 + 1) % n);
            }
            log.push_back(g);
        }

        // Reproduce lower()'s composition through the production frame.
        VirtualFrame frame(static_cast<uint32_t>(n), 4);
        for (const auto& g : log) {
            frame.append_gate(g);
        }
        const stim::Tableau<kStimWidth> v_cum_inv = frame.mutable_materialized_tableau().inverse();
        const stim::Tableau<kStimWidth> composed = v_cum_inv.then(target);

        const auto phase = internal::frame_composition_phase(composed, log, target);

        const uint64_t dim = uint64_t{1} << n;
        DenseMatrix lhs = dense_tableau_matrix(composed);
        for (auto it = log.rbegin(); it != log.rend(); ++it) {
            lhs = dense_matmul(lhs, dense_pending_gate(*it, n), dim);
        }
        const DenseMatrix rhs = dense_tableau_matrix(target);
        for (uint64_t i = 0; i < dim * dim; ++i) {
            CAPTURE(i);
            const auto expected = phase * rhs[i];
            REQUIRE_THAT(lhs[i].real(), Catch::Matchers::WithinAbs(expected.real(), 1e-5));
            REQUIRE_THAT(lhs[i].imag(), Catch::Matchers::WithinAbs(expected.imag(), 1e-5));
        }
    }
}
