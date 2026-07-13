#pragma once

// The noncomputational model's level structure, fixed at compile time.
//
// clifft models one noncomputational scenario: a two-level computational
// qubit (g = |0>, e = |1>), two leaked levels (a carrier present but
// outside the computational subspace, one reading out like each
// computational level), and one lost level (empty trap / vacuum).
// Transition matrices are T[to][from] over the five levels in enum
// order; classifier columns are indexed the same way. The structure is
// deliberately not configurable: the known use cases fit it, every
// layer above and below assumes it, and a future level would be added
// here explicitly -- one enum value and one matrix row/column -- rather
// than through a runtime table.
//
// QubitStatus is the per-qubit ledger entry: computational, or the
// definite noncomputational level the qubit holds. Which basis state a
// computational qubit holds -- if either definitely -- is runtime
// information living in the SVM, never in this classical ledger, so g
// and e collapse to the one Computational status and a status needs no
// auxiliary level field.

#include <array>
#include <cstdint>
#include <stdexcept>

namespace clifft {

// The five levels, in matrix index order.
enum class Level : uint8_t {
    G = 0,      // computational |0>
    E = 1,      // computational |1>
    LeakG = 2,  // leaked, reads out like g
    LeakE = 3,  // leaked, reads out like e
    Lost = 4,   // empty trap / vacuum
};

inline constexpr uint8_t kNumLevels = 5;

inline constexpr std::array<Level, kNumLevels> kAllLevels = {
    Level::G, Level::E, Level::LeakG, Level::LeakE, Level::Lost,
};

constexpr bool is_computational(Level level) {
    return level == Level::G || level == Level::E;
}

constexpr bool is_leaked(Level level) {
    return level == Level::LeakG || level == Level::LeakE;
}

constexpr bool is_lost(Level level) {
    return level == Level::Lost;
}

// Human-readable level name, for diagnostics.
constexpr const char* level_name(Level level) {
    switch (level) {
        case Level::G:
            return "g";
        case Level::E:
            return "e";
        case Level::LeakG:
            return "leak_g";
        case Level::LeakE:
            return "leak_e";
        case Level::Lost:
            return "lost";
    }
    return "?";
}

// Per-qubit ledger status: computational, or the definite
// noncomputational level the qubit holds.
enum class QubitStatus : uint8_t {
    Computational = 0,
    LeakG = 1,
    LeakE = 2,
    Lost = 3,
};

// Human-readable status name, for diagnostics.
constexpr const char* status_name(QubitStatus status) {
    switch (status) {
        case QubitStatus::Computational:
            return "Computational";
        case QubitStatus::LeakG:
            return "LeakG";
        case QubitStatus::LeakE:
            return "LeakE";
        case QubitStatus::Lost:
            return "Lost";
    }
    return "unknown";
}

constexpr bool is_computational(QubitStatus status) {
    return status == QubitStatus::Computational;
}

constexpr bool is_leaked(QubitStatus status) {
    return status == QubitStatus::LeakG || status == QubitStatus::LeakE;
}

constexpr bool is_lost(QubitStatus status) {
    return status == QubitStatus::Lost;
}

// The status a qubit holds after landing on `level`: the computational
// levels collapse to the one Computational status (which basis state
// the qubit holds is SVM runtime information), the noncomputational
// levels carry through.
constexpr QubitStatus status_for(Level level) {
    switch (level) {
        case Level::G:
        case Level::E:
            return QubitStatus::Computational;
        case Level::LeakG:
            return QubitStatus::LeakG;
        case Level::LeakE:
            return QubitStatus::LeakE;
        case Level::Lost:
            return QubitStatus::Lost;
    }
    return QubitStatus::Lost;  // unreachable for a valid Level
}

// The definite level of a noncomputational status -- the transition
// source column and classifier column a classical consult reads.
// A computational status has no definite level; asking for one is a
// logic error in the caller, which must branch on the category first.
inline Level noncomp_level(QubitStatus status) {
    switch (status) {
        case QubitStatus::LeakG:
            return Level::LeakG;
        case QubitStatus::LeakE:
            return Level::LeakE;
        case QubitStatus::Lost:
            return Level::Lost;
        case QubitStatus::Computational:
            break;
    }
    throw std::logic_error(
        "noncomp_level: a computational status holds no definite level; the caller must branch "
        "on the status category first");
}

}  // namespace clifft
