#pragma once

#include "clifft/util/bitmask.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

// Sparse sums of XOR parities modulo eight for commuting Pauli rotations.
// The common case uses one machine word. Only the optional two-word search
// draws on the work allowance shared across a pass.
namespace clifft::phase_detail {

struct SearchLimit {};
struct WideBudget {
    size_t max_updates = 50000000;
    size_t max_terms = 200000;
    size_t updates = 0;
    bool exhausted = false;

    bool try_charge(size_t amount);
    void charge(size_t amount = 1);
    void check_terms(size_t size);
};

struct OrdinaryWork {
    bool try_charge(size_t) const { return true; }
    void charge(size_t = 1) const {}
    void check_terms(size_t) const {}
};

template <size_t N>
struct PhaseReduction {
    uint32_t core_width = 0;
    size_t t_count = 0;
    // Each mask addresses the original rotation axes, in emission order.
    std::vector<std::pair<BitMask<N>, uint8_t>> phases;
};

template <size_t N>
struct PauliDerivative {
    BitMask<N> parity;
    uint8_t constant = 0;
};

template <size_t N>
class PhasePolynomial {
  public:
    using Mask = BitMask<N>;
    using Work = std::conditional_t<N == 64, OrdinaryWork, WideBudget>;
    using Terms = std::map<Mask, uint8_t>;

    const Terms& terms() const { return terms_; }
    size_t odd_terms() const;
    void add(Mask parity, int coefficient, Work& work);
    // p(x xor flip) - p(x) = constant + 4 * parity(x) for a Pauli pullback.
    std::optional<PauliDerivative<N>> derivative(Mask flip, Work& work);
    // Independent parity coordinates for the non-Clifford core. A flip is a
    // Clifford direction exactly when it is orthogonal to every returned mask.
    std::vector<Mask> core_axes(uint32_t width, Work& work);
    std::optional<PhaseReduction<N>> linear_reduction(uint32_t width, Work& work);
    PhaseReduction<N> synthesize(uint32_t width, Work& work);

  private:
    // For each selected pair (i,j), bit k of odd_columns is the cubic
    // signature entry ijk. One row packs all possible third indices.
    struct Pair {
        uint8_t weight = 0;  // Sum of parity coefficients modulo four.
        Mask odd_columns;
    };
    struct Signature {
        std::array<uint8_t, N> linear{};
        std::map<std::pair<uint32_t, uint32_t>, Pair> pairs;
        void add(Mask parity, int coefficient, Work& work);
    };
    const Signature& signature(Work& work);
    Terms terms_;
    std::optional<Signature> signature_;
};

extern template class PhasePolynomial<64>;
extern template class PhasePolynomial<128>;

}  // namespace clifft::phase_detail
