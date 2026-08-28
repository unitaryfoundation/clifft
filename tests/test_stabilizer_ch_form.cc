#include "clifft/tableau/stabilizer_ch_form.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

namespace {

std::complex<double> amplitude(const clifft::StabilizerChForm& state, uint64_t basis) {
    const std::vector<uint64_t> mask{basis};
    return state.amplitude(mask);
}

using clifft::test::check_complex;
using clifft::test::kInvSqrt2;

using DenseState = std::vector<std::complex<double>>;

void apply_dense_single(DenseState& state, uint32_t qubit, std::complex<double> m00,
                        std::complex<double> m01, std::complex<double> m10,
                        std::complex<double> m11) {
    const uint64_t stride = uint64_t{1} << qubit;
    for (uint64_t base = 0; base < state.size(); base += 2 * stride) {
        for (uint64_t offset = 0; offset < stride; ++offset) {
            const uint64_t zero = base + offset;
            const uint64_t one = zero + stride;
            const auto a = state[zero];
            const auto b = state[one];
            state[zero] = m00 * a + m01 * b;
            state[one] = m10 * a + m11 * b;
        }
    }
}

void apply_dense_cx(DenseState& state, uint32_t control, uint32_t target) {
    for (uint64_t basis = 0; basis < state.size(); ++basis) {
        if (((basis >> control) & 1U) != 0 && ((basis >> target) & 1U) == 0) {
            std::swap(state[basis], state[basis ^ (uint64_t{1} << target)]);
        }
    }
}

void apply_dense_cz(DenseState& state, uint32_t q1, uint32_t q2) {
    for (uint64_t basis = 0; basis < state.size(); ++basis) {
        if (((basis >> q1) & 1U) != 0 && ((basis >> q2) & 1U) != 0) {
            state[basis] = -state[basis];
        }
    }
}

void apply_dense_swap(DenseState& state, uint32_t q1, uint32_t q2) {
    for (uint64_t basis = 0; basis < state.size(); ++basis) {
        const bool b1 = ((basis >> q1) & 1U) != 0;
        const bool b2 = ((basis >> q2) & 1U) != 0;
        if (!b1 && b2) {
            std::swap(state[basis], state[basis ^ (uint64_t{1} << q1) ^ (uint64_t{1} << q2)]);
        }
    }
}

}  // namespace

TEST_CASE("CH form retains single-qubit Clifford phases") {
    SECTION("Hadamard") {
        clifft::StabilizerChForm state(1);
        state.apply_h(0);
        check_complex(amplitude(state, 0), {kInvSqrt2, 0.0});
        check_complex(amplitude(state, 1), {kInvSqrt2, 0.0});
    }

    SECTION("Y") {
        clifft::StabilizerChForm state(1);
        state.apply_y(0);
        check_complex(amplitude(state, 0), {0.0, 0.0});
        check_complex(amplitude(state, 1), {0.0, 1.0});
    }

    SECTION("S on one") {
        clifft::StabilizerChForm state(1);
        state.apply_x(0);
        state.apply_s(0);
        check_complex(amplitude(state, 0), {0.0, 0.0});
        check_complex(amplitude(state, 1), {0.0, 1.0});
    }

    SECTION("S dagger on plus") {
        clifft::StabilizerChForm state(1);
        state.apply_h(0);
        state.apply_s_dag(0);
        check_complex(amplitude(state, 0), {kInvSqrt2, 0.0});
        check_complex(amplitude(state, 1), {0.0, -kInvSqrt2});
    }
}

TEST_CASE("CH form retains entangled Clifford amplitudes") {
    clifft::StabilizerChForm state(2);
    state.apply_h(0);
    state.apply_cx(0, 1);
    state.apply_s(1);

    check_complex(amplitude(state, 0), {kInvSqrt2, 0.0});
    check_complex(amplitude(state, 1), {0.0, 0.0});
    check_complex(amplitude(state, 2), {0.0, 0.0});
    check_complex(amplitude(state, 3), {0.0, kInvSqrt2});
}

TEST_CASE("CH form supports Clifford permutations and explicit global phase") {
    clifft::StabilizerChForm state(2);
    state.apply_x(0);
    state.apply_swap(0, 1);
    state.apply_global_phase({0.0, -1.0});

    check_complex(amplitude(state, 0), {0.0, 0.0});
    check_complex(amplitude(state, 1), {0.0, 0.0});
    check_complex(amplitude(state, 2), {0.0, -1.0});
    check_complex(amplitude(state, 3), {0.0, 0.0});
}

TEST_CASE("CH form matches dense random Clifford circuits componentwise") {
    constexpr uint32_t n = 3;
    constexpr uint32_t depth = 200;
    std::mt19937_64 rng(0x43485f464f524dULL);
    clifft::StabilizerChForm state(n);
    DenseState dense(uint64_t{1} << n, {0.0, 0.0});
    dense[0] = {1.0, 0.0};

    for (uint32_t step = 0; step < depth; ++step) {
        const uint32_t gate = static_cast<uint32_t>(rng() % 9U);
        const uint32_t q1 = static_cast<uint32_t>(rng() % n);
        uint32_t q2 = static_cast<uint32_t>(rng() % (n - 1));
        if (q2 >= q1) {
            ++q2;
        }
        switch (gate) {
            case 0:
                state.apply_h(q1);
                apply_dense_single(dense, q1, {kInvSqrt2, 0.0}, {kInvSqrt2, 0.0}, {kInvSqrt2, 0.0},
                                   {-kInvSqrt2, 0.0});
                break;
            case 1:
                state.apply_s(q1);
                apply_dense_single(dense, q1, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0});
                break;
            case 2:
                state.apply_s_dag(q1);
                apply_dense_single(dense, q1, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0});
                break;
            case 3:
                state.apply_x(q1);
                apply_dense_single(dense, q1, {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0});
                break;
            case 4:
                state.apply_y(q1);
                apply_dense_single(dense, q1, {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0});
                break;
            case 5:
                state.apply_z(q1);
                apply_dense_single(dense, q1, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0});
                break;
            case 6:
                state.apply_cx(q1, q2);
                apply_dense_cx(dense, q1, q2);
                break;
            case 7:
                state.apply_cz(q1, q2);
                apply_dense_cz(dense, q1, q2);
                break;
            default:
                state.apply_swap(q1, q2);
                apply_dense_swap(dense, q1, q2);
                break;
        }

        INFO("step " << step << " gate " << gate << " q1 " << q1 << " q2 " << q2);
        for (uint64_t basis = 0; basis < dense.size(); ++basis) {
            check_complex(amplitude(state, basis), dense[basis]);
        }
    }
}

TEST_CASE("CH form supports multiword computational basis masks") {
    clifft::StabilizerChForm state(70);
    state.apply_x(69);
    std::vector<uint64_t> basis(2, 0);
    basis[1] = uint64_t{1} << 5U;

    check_complex(state.amplitude(basis), {1.0, 0.0});
    basis[1] = 0;
    check_complex(state.amplitude(basis), {0.0, 0.0});
}
