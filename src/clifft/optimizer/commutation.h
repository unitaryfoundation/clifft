#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/symplectic.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace clifft {

/// Returns true if the two HIR operations can be safely swapped in the
/// ops vector without changing program semantics or PRNG trajectory.
bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir);

/// Content fingerprint of an HirModule. Covers exactly the HirModule fields
/// can_swap reads, so a change to can_swap's inputs is a change to this
/// function, in this same file.
struct CommutationFingerprint {
    size_t op_count = 0;
    uint32_t num_qubits = 0;
    uint64_t hash = 0;

    friend bool operator==(const CommutationFingerprint&, const CommutationFingerprint&) = default;
};

[[nodiscard]] CommutationFingerprint commutation_fingerprint(const HirModule& hir);

}  // namespace clifft
