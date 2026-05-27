#pragma once

// Per-qubit runtime status for the noncomputational trajectory MVP.
// See design/noncomputational-mvp.md section 2.1 (enums) and 2.3
// (tagged QubitStatus + invariants).
//
// The tag QubitStatusKind splits the computational case by whether
// the energy-basis value has been resolved. The level_id field has a
// meaning that depends on the tag:
//
//   ComputationalUnknown : level_id == kInvalidLevel (no resolved source)
//   ComputationalKnown   : level_id is a Computational-category level
//   Leaked               : level_id is a Leaked-category level
//   Lost                 : level_id is a Lost-category level
//
// Factories enforce these invariants at construction time so that an
// "unknown coherent" qubit cannot accidentally expose a stale level
// id to a source-dependent transition. Accessors enforce the
// invariants on read.

#include "clifft/noncomp/level.h"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

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
};

// Factories ---------------------------------------------------------------

inline QubitStatus make_computational_unknown() {
    return QubitStatus{QubitStatusKind::ComputationalUnknown, kInvalidLevel};
}

namespace detail {
inline void check_level_for_kind(uint8_t level_id, LevelCategory expected,
                                 std::span<const Level> levels, const char* factory_name) {
    if (level_id >= levels.size()) {
        throw std::invalid_argument(std::string(factory_name) + ": level_id " +
                                    std::to_string(level_id) + " out of range (level set has " +
                                    std::to_string(levels.size()) + " entries)");
    }
    if (levels[level_id].category != expected) {
        throw std::invalid_argument(std::string(factory_name) + ": level '" +
                                    levels[level_id].label +
                                    "' has category that does not match the requested kind");
    }
}
}  // namespace detail

inline QubitStatus make_computational_known(uint8_t level_id, std::span<const Level> levels) {
    detail::check_level_for_kind(level_id, LevelCategory::Computational, levels,
                                 "make_computational_known");
    return QubitStatus{QubitStatusKind::ComputationalKnown, level_id};
}

inline QubitStatus make_leaked(uint8_t level_id, std::span<const Level> levels) {
    detail::check_level_for_kind(level_id, LevelCategory::Leaked, levels, "make_leaked");
    return QubitStatus{QubitStatusKind::Leaked, level_id};
}

inline QubitStatus make_lost(uint8_t level_id, std::span<const Level> levels) {
    detail::check_level_for_kind(level_id, LevelCategory::Lost, levels, "make_lost");
    return QubitStatus{QubitStatusKind::Lost, level_id};
}

// Accessors ---------------------------------------------------------------

inline bool is_unknown_computational(const QubitStatus& s) {
    return s.kind == QubitStatusKind::ComputationalUnknown;
}

// Returns the resolved level id when the qubit has a known source
// (ComputationalKnown, Leaked, or Lost); nullopt when the source is
// unresolved (ComputationalUnknown).
inline std::optional<uint8_t> known_source_level(const QubitStatus& s) {
    if (s.kind == QubitStatusKind::ComputationalUnknown) {
        return std::nullopt;
    }
    return s.level_id;
}

// Like known_source_level but throws std::invalid_argument on
// ComputationalUnknown. Use in code paths where the caller must consult
// a definite level id (classical Markov transitions; selecting a
// transition-matrix column for a known-coherent source).
inline uint8_t require_classical_source_level(const QubitStatus& s) {
    if (s.kind == QubitStatusKind::ComputationalUnknown) {
        throw std::invalid_argument(
            "require_classical_source_level: qubit kind is "
            "ComputationalUnknown; no resolved level id");
    }
    return s.level_id;
}

}  // namespace clifft
