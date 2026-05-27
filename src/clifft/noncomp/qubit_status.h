#pragma once

// Per-qubit runtime status carried by a sampled trajectory.
//
// QubitStatusKind tags how much we know about the qubit's state:
//
//   ComputationalUnknown : in an arbitrary quantum state on H_C
//                          (a superposition of |0> and |1>). The
//                          level_id field is not meaningful.
//   ComputationalKnown   : in a known computational basis state
//                          (|0> or |1>). level_id identifies which
//                          Computational level the qubit holds.
//   Leaked               : outside the computational subspace.
//                          level_id identifies which Leaked level
//                          the qubit holds.
//   Lost                 : absent / vacuum. level_id identifies
//                          the Lost level (typically a single
//                          "lost" level).
//
// QubitStatus is a plain aggregate so callers may construct it via
// designated initializers when convenient, but the canonical builders
// live on LevelSet (see level.h):
//
//   LevelSet::computational_known(level_id)
//   LevelSet::leaked(level_id)
//   LevelSet::lost(level_id)
//
// Those factories validate that the level id refers to a level of
// the matching category in the LevelSet's table. Direct aggregate
// construction bypasses that check and should be reserved for tests
// or interior code where the invariant is already established.

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace clifft {

enum class QubitStatusKind : uint8_t {
    ComputationalUnknown = 0,
    ComputationalKnown = 1,
    Leaked = 2,
    Lost = 3,
};

// Sentinel for the level_id field when no specific level is resolved.
// Valid only with kind == ComputationalUnknown.
constexpr uint8_t kInvalidLevel = 0xFF;

struct QubitStatus {
    QubitStatusKind kind;
    uint8_t level_id;

    // The no-resolved-source case has no level id to record.
    static QubitStatus computational_unknown() {
        return QubitStatus{QubitStatusKind::ComputationalUnknown, kInvalidLevel};
    }

    bool is_unknown_computational() const { return kind == QubitStatusKind::ComputationalUnknown; }

    // Returns the resolved level id when the qubit has a known
    // source (ComputationalKnown, Leaked, or Lost); nullopt when the
    // source is unresolved (ComputationalUnknown).
    std::optional<uint8_t> known_source_level() const {
        if (kind == QubitStatusKind::ComputationalUnknown) {
            return std::nullopt;
        }
        return level_id;
    }

    // Like known_source_level but throws on ComputationalUnknown. Use
    // in code paths where the caller must consult a definite level id
    // (classical Markov transitions; selecting a transition-matrix
    // column for a known-coherent source).
    uint8_t require_classical_source_level() const {
        if (kind == QubitStatusKind::ComputationalUnknown) {
            throw std::invalid_argument(
                "require_classical_source_level: qubit kind is "
                "ComputationalUnknown; no resolved level id");
        }
        return level_id;
    }
};

}  // namespace clifft
