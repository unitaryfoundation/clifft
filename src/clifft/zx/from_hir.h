#pragma once

// Single-copy parametric ZX diagrams of traced HIR modules.
//
// HIR operations are Pauli products in the frame of the initial |0...0>, so the
// HIR reads as a Clifford-free circuit on |0...0>: Pauli rotations, Pauli
// measurements, classically controlled Paulis and Pauli noise. The diagram is
// the amplitude <o|U_e|0> of one trajectory:
//
//   - every measurement outcome (visible or hidden) and every noise bit is a
//     Boolean parameter; a measurement is a projector onto the eigenspace its
//     outcome selects; a readout flip is a parameter only where feedback reads
//     the flipped record, and otherwise stays a classical flip of the record;
//   - a rotation exp(-i pi alpha P / 2) is a phase gadget;
//   - feedback and noise are Pauli spiders with phase pi times a parity of
//     parameters;
//   - a reset of a used qubit (a measurement followed by a correction that
//     anticommutes with it) continues on a fresh qubit: later Paulis are
//     rewritten so the reset qubit is never touched again, which makes the
//     reset a cut wire instead of an outcome spread over later spiders;
//   - every qubit is projected in the Z basis at the end, onto an unrecorded
//     final outcome, so nothing is traced out.
//
// The record distribution is then P(records | noise) = sum over hidden and
// final outcomes of |<o|U_e|0>|^2, normalized for each noise assignment.

#include "clifft/frontend/hir.h"
#include "clifft/zx/graph.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace clifft::zx {

struct DiagramParam {
    enum class Kind : uint8_t {
        Record,  // a measurement outcome as recorded, readout flips included
        Hidden,  // the outcome of a hidden measurement
        Final,   // the unrecorded end-of-circuit Z outcome of a qubit
        Noise,   // a noise bit
    };

    Kind kind = Kind::Record;
    // The record index of a record or hidden outcome, the qubit of a final
    // outcome, and the owning noise group of a noise bit.
    uint32_t index = 0;
};

// The distribution of one noise group's bits: a HIR noise site, or a readout
// noise entry. Outcome 0 is "nothing fired"; the others follow the source's
// channels one to one, so a later lowering can rebuild the site exactly.
struct NoiseGroup {
    enum class Source : uint8_t { NoiseSite, Readout };

    struct Outcome {
        double probability = 0.0;
        // Bit k set means bits[k] is 1.
        uint64_t pattern = 0;
    };

    Source source = Source::NoiseSite;
    uint32_t source_index = 0;
    std::vector<uint32_t> bits;
    std::vector<Outcome> outcomes;
};

struct SingleCopyDiagram {
    Graph graph;
    // Indexed by parameter ID.
    std::vector<DiagramParam> params;
    std::vector<NoiseGroup> noise;
    // Readout noise entries that no feedback reads: classical flips of their
    // records, applied to the sampled record outside the diagram.
    std::vector<uint32_t> classical_readout;
};

// Build the diagram, or return nullopt with a reason when the HIR holds
// operations the single-copy form does not cover: expectation-value probes,
// leakage and loss instruments, asymmetric readout noise that feedback reads,
// and noise rescheduled across other operations. Malformed indices also
// reject. So does a circuit that needs more than `max_qubits` qubits once every
// reset continues on a fresh qubit: the diagram's per-op cost grows with that
// count, and a long memory experiment would otherwise grow it without bound.
[[nodiscard]] std::optional<SingleCopyDiagram> build_single_copy_diagram(
    const HirModule& hir, std::string* reason,
    uint32_t max_qubits = std::numeric_limits<uint32_t>::max());

}  // namespace clifft::zx
