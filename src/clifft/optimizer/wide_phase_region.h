#pragma once

#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/known_stabilizers.h"
#include "clifft/optimizer/wide_phase_polynomial.h"

#include <optional>
#include <utility>
#include <vector>

namespace clifft::phase_detail::wide {

struct Observer {
    size_t index;
    PauliString pulled_axis;
};

struct Rewrite {
    size_t end;
    uint32_t core_width;
    size_t t_count;
    std::vector<std::pair<PauliString, uint8_t>> phases;
    std::vector<size_t> rotations;
    std::vector<Observer> observers;
    bool elide_post_observer = false;
};

// Try one complete commuting region beyond 64 variables, pulling compatible
// Pauli observers across the phase prefix and eliding trailing post-observer
// phases when the region reaches the final observer of the module.
std::optional<Rewrite> try_region(const HirModule& hir, size_t start, const KnownStabilizers& fixed,
                                  const optimizer_detail::CliffordAbsorption& frame,
                                  uint32_t variable_limit, Budget& budget,
                                  const KnownStabilizers* postselected = nullptr,
                                  std::optional<size_t> final_observer_end = std::nullopt);

}  // namespace clifft::phase_detail::wide
