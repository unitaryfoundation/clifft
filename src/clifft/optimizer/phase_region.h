#pragma once

#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/known_stabilizers.h"
#include "clifft/optimizer/phase_polynomial.h"

#include <span>

namespace clifft::phase_detail {

struct Observer {
    size_t index;
    PauliString pulled_axis;
};

template <size_t N>
PauliString axis_product(const std::vector<PauliString>& axes, BitMask<N> mask, uint32_t width) {
    PauliString result(width);
    for (; !mask.is_zero(); mask.clear_lowest_bit()) {
        result.right_multiply(axes[mask.lowest_bit()].view());
    }
    assert(result.is_hermitian());
    return result;
}

// Rotation coordinates use the stabilizers known at entry. Crossing an
// observer or noise channel can remove relations available to later rotations.
template <size_t N>
struct PhaseRegion {
    using Mask = BitMask<N>;
    using Work = typename PhasePolynomial<N>::Work;
    PhaseRegion(size_t start, const KnownStabilizers& fixed) : end(start), available(fixed) {}
    size_t end;
    bool capped = false;
    KnownStabilizers available, required;
    PhasePolynomial<N> polynomial;
    std::vector<PauliString> axes;
    std::vector<size_t> rotations;
    std::vector<Observer> observers;

    // Consume the operation at end if it belongs to this region.
    bool extend(const HirModule& hir, const KnownStabilizers& fixed,
                const optimizer_detail::CliffordAbsorption& frame,
                const KnownStabilizers* postselected, uint32_t variable_limit, Work& work);
    // Add one T rotation about a signed axis on num_qubits qubits, or T_DAG
    // when dagger is set, if it belongs to this region. The caller records
    // the rotation and advances end.
    bool add_rotation(const PauliString& axis, bool dagger, uint32_t num_qubits,
                      const KnownStabilizers& fixed, const KnownStabilizers* postselected,
                      uint32_t variable_limit, Work& work);
    Mask coordinates(PauliString& body) const;

  private:
    struct Row {
        PauliString axis;
        Mask coordinates;
    };
    std::map<uint32_t, Row> rows;
};

extern template struct PhaseRegion<64>;
extern template struct PhaseRegion<128>;

struct WideRewrite {
    size_t end;
    uint32_t core_width;
    size_t t_count;
    std::vector<std::pair<PauliString, uint8_t>> phases;
    std::vector<size_t> rotations;
    std::vector<Observer> observers;
    bool elide_post_observer = false;
};

std::optional<WideRewrite> try_wide_region(const HirModule& hir, size_t start,
                                           const KnownStabilizers& fixed,
                                           const optimizer_detail::CliffordAbsorption& frame,
                                           uint32_t variable_limit, WideBudget& budget,
                                           const KnownStabilizers* postselected = nullptr,
                                           std::optional<size_t> final_observer_end = std::nullopt);

// One T rotation about a signed Hermitian axis, or T_DAG when dagger is set.
struct AxisRotation {
    PauliString axis;
    bool dagger = false;
};

// Reduce the longest prefix of rotations that forms one commuting region on a
// state stabilized by `fixed`. Unlike try_wide_region this takes the axes
// directly and returns the rewrite whether or not it lowers the T count, so
// the caller can weigh it against its own alternative. The rewrite's end is
// the number of rotations it covers. No value means the prefix needs more than
// variable_limit independent axes or the budget ran out.
std::optional<WideRewrite> reduce_rotations(uint32_t num_qubits,
                                            std::span<const AxisRotation> rotations,
                                            const KnownStabilizers& fixed, uint32_t variable_limit,
                                            WideBudget& budget);
}  // namespace clifft::phase_detail
