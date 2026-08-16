#pragma once

// String-formatting utilities that depend only on the HIR.

#include "clifft/frontend/hir.h"

#include <optional>
#include <string>

namespace clifft {

// Format a Pauli mask (X bits, Z bits, sign) as a human-readable string.
// Example: "+X0*Z3" for destab bit 0 and stab bit 3.
std::string format_pauli_mask(PauliMaskView mask);

std::string op_type_to_str(OpType type);

// Format a HIR op as a human-readable string. For mask-carrying ops the
// caller passes the op's mask via `hir.mask_view(op)`; for non-mask ops
// (NOISE, READOUT_NOISE, DETECTOR, OBSERVABLE) the caller passes
// std::nullopt and the mask argument is unused.
std::string format_hir_op(const HeisenbergOp& op, std::optional<PauliMaskView> mask);

}  // namespace clifft
