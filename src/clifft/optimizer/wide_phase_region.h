#pragma once

#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/known_stabilizers.h"
#include "clifft/optimizer/wide_phase_polynomial.h"

#include <optional>
#include <utility>
#include <vector>

namespace clifft::phase_detail::wide {

struct Rewrite {
    size_t end;
    uint32_t core_width;
    size_t t_count;
    std::vector<std::pair<PauliString, uint8_t>> phases;
};

// Try one complete contiguous commuting region. Barriers stay on the ordinary
// path; no measurement, fault, or feedback is moved by this optional attempt.
std::optional<Rewrite> try_region(const HirModule& hir, size_t start, const KnownStabilizers& fixed,
                                  const optimizer_detail::CliffordAbsorption& frame,
                                  uint32_t variable_limit, Budget& budget);

}  // namespace clifft::phase_detail::wide
