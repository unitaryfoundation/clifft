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
// QubitStatus is non-aggregate; the only construction paths are the
// public static factories. The canonical validated builders live on
// LevelSet (see level.h), which calls the _unchecked factories here
// after checking the level id matches the requested kind in its
// table. The _unchecked factories are public for use by tests and
// by interior code where the invariant is already established; their
// name is the warning.

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

class QubitStatus {
  public:
    static QubitStatus computational_unknown() {
        return QubitStatus(QubitStatusKind::ComputationalUnknown, kInvalidLevel);
    }

    // Construct a known-source status without checking level_id against
    // a level table. Tests and interior code may call these; user code
    // should prefer LevelSet::computational_known / leaked / lost.
    static QubitStatus computational_known_unchecked(uint8_t level_id) {
        return QubitStatus(QubitStatusKind::ComputationalKnown, level_id);
    }
    static QubitStatus leaked_unchecked(uint8_t level_id) {
        return QubitStatus(QubitStatusKind::Leaked, level_id);
    }
    static QubitStatus lost_unchecked(uint8_t level_id) {
        return QubitStatus(QubitStatusKind::Lost, level_id);
    }

    QubitStatusKind kind() const { return kind_; }
    uint8_t level_id() const { return level_id_; }

    bool is_unknown_computational() const { return kind_ == QubitStatusKind::ComputationalUnknown; }

    // Returns the resolved level id when the qubit has a known source
    // (ComputationalKnown, Leaked, or Lost); nullopt when the source
    // is unresolved (ComputationalUnknown).
    std::optional<uint8_t> known_source_level() const {
        if (kind_ == QubitStatusKind::ComputationalUnknown) {
            return std::nullopt;
        }
        return level_id_;
    }

    // Like known_source_level but throws on ComputationalUnknown. Use
    // in code paths where the caller must consult a definite level id
    // (classical Markov transitions; selecting a transition-matrix
    // column for a known-coherent source).
    uint8_t require_classical_source_level() const {
        if (kind_ == QubitStatusKind::ComputationalUnknown) {
            throw std::invalid_argument(
                "require_classical_source_level: qubit kind is "
                "ComputationalUnknown; no resolved level id");
        }
        return level_id_;
    }

  private:
    QubitStatus(QubitStatusKind kind, uint8_t level_id) : kind_(kind), level_id_(level_id) {}

    QubitStatusKind kind_;
    uint8_t level_id_;
};

}  // namespace clifft
