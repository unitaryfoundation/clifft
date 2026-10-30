#pragma once

// A Pauli-based computation (PBC) read off a reduced single-copy diagram.
//
// The reading follows Tsim's statevector reader (tsim/statevector/pbc.py,
// https://github.com/QuEraComputing/tsim), adapted to single-copy diagrams.
// The reduced diagram is graph-like: a graph state on its Z spiders, then a
// rotation and an X or Y measurement per spider. Each spider's outcome bit is
// tied to the diagram's parameters by its phase, and the scalar's constraints
// tie outcome parameters together. Eliminating the unrecorded outcomes over
// GF(2) leaves postselections on spider outcomes and each record as a parity
// of spider outcomes, fair coins and noise bits.
//
// The postselections are moved to the start, where they act on the graph
// state: each rotation axis is multiplied by graph-state stabilizers so that
// it commutes with every postselection, a search like gflow finding. A
// Clifford frame then maps the postselected graph state to |0...0>, so the
// result is rotations and measurements of Pauli products on |0...0> whose
// signs are affine in the noise bits.

#include "clifft/zx/bits.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/graph.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace clifft::zx {

struct PbcOp {
    enum class Kind : uint8_t { Rotation, Measurement };

    Kind kind = Kind::Rotation;
    // The Pauli on the group's lines: X where x is set, Z where z is, Y where both.
    Bits x;
    Bits z;
    // A rotation is exp(-i theta (-1)^s P); a measurement measures (-1)^s P.
    double theta = 0.0;
    // s as an affine form: bit 0 the constant, bit k + 1 the group's noise bit k.
    Bits sign;
    // The record a measurement writes, or -1 for one whose outcome is dropped.
    int64_t record = -1;
};

// One independent part of the computation, on its own lines.
struct PbcGroup {
    uint32_t num_lines = 0;
    // The diagram parameter ID of each of the group's noise bits.
    std::vector<uint32_t> noise;
    // In a valid order: commutant measurements, then rotations, then the
    // record measurements.
    std::vector<PbcOp> ops;
};

struct PbcProgram {
    std::vector<PbcGroup> groups;
};

// Read `reduced`, the full_reduce result of `diagram.graph`. Returns nullopt
// with a reason when the diagram is not graph-like, its scalar weights
// trajectories, its constraints are inconsistent, no rotation order exists,
// or it postselects an event of probability zero.
[[nodiscard]] std::optional<PbcProgram> read_pbc(const SingleCopyDiagram& diagram,
                                                 const Graph& reduced, std::string* reason);

}  // namespace clifft::zx
