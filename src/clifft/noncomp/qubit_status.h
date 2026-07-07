#pragma once

// Per-qubit runtime status carried by a sampled trajectory.
//
// QubitStatusKind tags the category a qubit occupies:
//
//   Computational : in H_C (any state on the |0>/|1> subspace). The
//                   level_id field is not meaningful: which basis
//                   state -- if either definitely -- the qubit holds
//                   is runtime information living in the SVM, never
//                   in this classical ledger.
//   Leaked        : outside the computational subspace.
//                   level_id identifies which Leaked level
//                   the qubit holds.
//   Lost          : absent / vacuum. level_id identifies
//                   the Lost level (typically a single
//                   "lost" level).
//
// The enum values are the sidecar code the Python surface exposes
// ({0 computational, 1 leaked, 2 lost}); keep them aligned.
//
// QubitStatus is non-aggregate; the only construction paths are the
// public static factories. The canonical validated builders live on
// LevelSet (see level.h), which calls the _unchecked factories here
// after checking the level id matches the requested kind in its
// table. The _unchecked factories are public for use by tests and
// by interior code where the invariant is already established; their
// name is the warning.

#include <cstdint>

namespace clifft {

enum class QubitStatusKind : uint8_t {
    Computational = 0,
    Leaked = 1,
    Lost = 2,
};

// Human-readable name of a status kind, for diagnostics.
inline const char* kind_name(QubitStatusKind kind) {
    switch (kind) {
        case QubitStatusKind::Computational:
            return "Computational";
        case QubitStatusKind::Leaked:
            return "Leaked";
        case QubitStatusKind::Lost:
            return "Lost";
    }
    return "unknown";
}

// Sentinel for the level_id field when no specific level is carried.
// Every Computational status holds it.
constexpr uint8_t kInvalidLevel = 0xFF;

class QubitStatus {
  public:
    static QubitStatus computational() {
        return QubitStatus(QubitStatusKind::Computational, kInvalidLevel);
    }

    // Construct a noncomputational status without checking level_id
    // against a level table. Tests and interior code may call these;
    // user code should prefer LevelSet::leaked / lost.
    static QubitStatus leaked_unchecked(uint8_t level_id) {
        return QubitStatus(QubitStatusKind::Leaked, level_id);
    }
    static QubitStatus lost_unchecked(uint8_t level_id) {
        return QubitStatus(QubitStatusKind::Lost, level_id);
    }

    QubitStatusKind kind() const { return kind_; }
    uint8_t level_id() const { return level_id_; }

    bool is_computational() const { return kind_ == QubitStatusKind::Computational; }

  private:
    QubitStatus(QubitStatusKind kind, uint8_t level_id) : kind_(kind), level_id_(level_id) {}

    QubitStatusKind kind_;
    uint8_t level_id_;
};

}  // namespace clifft
