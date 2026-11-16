#pragma once

#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/known_stabilizers.h"
#include "clifft/optimizer/wide_phase_polynomial.h"

#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace clifft::phase_detail::wide {

// A commuting run of T rotations rewritten as phases about products of its
// axes. Each entry applies exp(-i * pi * coefficient * axis / 8): an odd
// coefficient keeps one T or T_DAG and its even remainder is Clifford. The
// run ends before rotation `end`.
struct Rewrite {
    size_t end;
    uint32_t core_width;
    size_t t_count;
    std::vector<std::pair<PauliString, uint8_t>> phases;
};

// One T rotation about a signed Hermitian axis, or T_DAG when dagger is set.
struct AxisRotation {
    PauliString axis;
    bool dagger = false;
};

// Try one complete contiguous commuting region. Barriers stay on the ordinary
// path; no measurement, fault, or feedback is moved by this optional attempt.
std::optional<Rewrite> try_region(const HirModule& hir, size_t start, const KnownStabilizers& fixed,
                                  const optimizer_detail::CliffordAbsorption& frame,
                                  uint32_t variable_limit, Budget& budget);

// Reduce the longest prefix of rotations that forms one commuting region on a
// state stabilized by `fixed`. Unlike try_region this accepts any width up to
// variable_limit and returns the rewrite whether or not it lowers the T count,
// so the caller can weigh it against its own alternative. No value means the
// prefix exceeds variable_limit, its core exceeds 64 coordinates, or the
// budget ran out.
std::optional<Rewrite> reduce_rotations(uint32_t num_qubits,
                                        std::span<const AxisRotation> rotations,
                                        const KnownStabilizers& fixed, uint32_t variable_limit,
                                        Budget& budget);

}  // namespace clifft::phase_detail::wide
