#pragma once

// Lowering a scheduled PBC program back to HIR.
//
// Each group gets its own lines, all starting in |0>. A sign that depends on
// noise bits is carried by dormant sign lines: one line per vector of a GF(2)
// basis of the noise-dependent parts of the signs, so that line l holds the
// parity v_l . e of the noise bits e. A noise channel flips the lines whose
// parity it changes, and an op whose sign is a combination of basis vectors
// carries Z on those lines. Sign lines see only X noise and Z products, so
// they never widen the active state.
//
// Noise stays site for site, so fault counting and importance sampling see
// the same sites with the same probabilities; only the channel masks change.
// Readout noise entries stay classical flips of their records, except where
// feedback reads the flipped record: the flip then changes later signs, which
// a classical flip cannot express, so it becomes a one-channel noise site on
// the sign lines, appended after the original sites, and its entry is no
// longer referenced.
//
// Visible records keep their indices, so detectors and observables are copied
// unchanged; measurements whose outcomes the reader dropped become hidden.
// The rewrite has no op-to-source mapping: every source_map entry is empty.
//
// The output preserves the record distribution, not the state: it has no
// final tableau.

#include "clifft/frontend/hir.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/pbc.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace clifft::zx {

// `orders[g]` is a valid order of `program.groups[g].ops`. Returns nullopt with
// a reason when the input cannot be rebuilt one to one: readout noise on a
// hidden record, or a noise site applied twice.
[[nodiscard]] std::optional<HirModule> emit_hir(const HirModule& input,
                                                const SingleCopyDiagram& diagram,
                                                const PbcProgram& program,
                                                const std::vector<std::vector<uint32_t>>& orders,
                                                std::string* reason);

}  // namespace clifft::zx
