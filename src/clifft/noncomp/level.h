#pragma once

// Level definitions and the validated level table.
//
// A Level is a model-defined tag with a stable integer id (its index
// into the level table), a human-readable label, a LevelCategory, and
// an optional basis_bit:
//
//   - For LevelCategory::Computational, basis_bit is REQUIRED and
//     identifies which computational basis state (|0> or |1>) the
//     level represents. The rewriter uses it to prepend an X prep
//     for basis_bit == One initial samples.
//   - For LevelCategory::Leaked, basis_bit is optional metadata
//     (origin / classifier hint). No current code path consumes it.
//   - For LevelCategory::Lost, basis_bit MUST be empty. "Lost from
//     |1>" provenance, if ever needed, belongs in an event record or
//     in distinct lost levels — not the default lost level.
//
// LevelSet validation also enforces that the level table contains
// exactly one Computational level with basis_bit == Zero and exactly
// one with basis_bit == One. Downstream paths (visible Z-basis
// measurement, Z-basis reset, initial prep, classifier defaults)
// need unambiguous g/e ids; duplicates or missing canonical levels
// reject at LevelSet construction.
//
// LevelSet wraps a std::vector<Level>, runs validation in its ctor,
// and owns the QubitStatus factories that bind a level id to this
// specific table. Construction of QubitStatus values for non-Unknown
// kinds should go through LevelSet so the (kind, level_id) pair is
// guaranteed to refer to a real level of the matching category in a
// known table.

#include "clifft/noncomp/qubit_status.h"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clifft {

enum class LevelCategory : uint8_t {
    Computational = 0,
    Leaked = 1,
    Lost = 2,
};

enum class BasisBit : uint8_t {
    Zero = 0,
    One = 1,
};

struct Level {
    std::string label;
    LevelCategory category;
    std::optional<BasisBit> basis_bit;
};

class LevelSet {
  public:
    // Validates the level table; throws std::invalid_argument with a
    // named field on failure.
    explicit LevelSet(std::vector<Level> levels) : levels_(std::move(levels)) { validate(); }

    // Default level set: g, e, leak_g, leak_e, lost.
    // Stable order; the integer ids are positional.
    static LevelSet default_set() {
        return LevelSet({
            Level{"g", LevelCategory::Computational, BasisBit::Zero},
            Level{"e", LevelCategory::Computational, BasisBit::One},
            Level{"leak_g", LevelCategory::Leaked, std::nullopt},
            Level{"leak_e", LevelCategory::Leaked, std::nullopt},
            Level{"lost", LevelCategory::Lost, std::nullopt},
        });
    }

    std::span<const Level> levels() const { return levels_; }
    size_t size() const { return levels_.size(); }

    const Level& at(uint8_t level_id) const {
        require_in_range(level_id, "at");
        return levels_[level_id];
    }

    // QubitStatus factories: validate that level_id refers to a level
    // of the matching category in this table.
    QubitStatus computational_known(uint8_t level_id) const {
        check_kind(level_id, LevelCategory::Computational, "computational_known");
        return QubitStatus::computational_known_unchecked(level_id);
    }

    QubitStatus leaked(uint8_t level_id) const {
        check_kind(level_id, LevelCategory::Leaked, "leaked");
        return QubitStatus::leaked_unchecked(level_id);
    }

    QubitStatus lost(uint8_t level_id) const {
        check_kind(level_id, LevelCategory::Lost, "lost");
        return QubitStatus::lost_unchecked(level_id);
    }

  private:
    std::vector<Level> levels_;

    void validate() const {
        if (levels_.empty()) {
            throw std::invalid_argument("LevelSet: level set is empty");
        }
        if (levels_.size() > 128) {
            throw std::invalid_argument("LevelSet: level set has " +
                                        std::to_string(levels_.size()) +
                                        " entries; max supported is 128");
        }
        size_t computational_zero_count = 0;
        size_t computational_one_count = 0;
        for (size_t i = 0; i < levels_.size(); ++i) {
            const Level& lv = levels_[i];
            switch (lv.category) {
                case LevelCategory::Computational:
                    if (!lv.basis_bit.has_value()) {
                        throw std::invalid_argument("LevelSet: level '" + lv.label + "' (id " +
                                                    std::to_string(i) +
                                                    ") is Computational but has no basis_bit");
                    }
                    if (*lv.basis_bit == BasisBit::Zero) {
                        ++computational_zero_count;
                    } else {
                        ++computational_one_count;
                    }
                    break;
                case LevelCategory::Leaked:
                    // basis_bit allowed as optional origin metadata.
                    break;
                case LevelCategory::Lost:
                    if (lv.basis_bit.has_value()) {
                        throw std::invalid_argument("LevelSet: level '" + lv.label + "' (id " +
                                                    std::to_string(i) +
                                                    ") is Lost and must not carry basis_bit");
                    }
                    break;
                default:
                    throw std::invalid_argument("LevelSet: level '" + lv.label + "' (id " +
                                                std::to_string(i) +
                                                ") has an unrecognized LevelCategory value");
            }
        }
        if (computational_zero_count != 1) {
            throw std::invalid_argument(
                "LevelSet: expected exactly one Computational level with basis_bit == Zero, "
                "got " +
                std::to_string(computational_zero_count));
        }
        if (computational_one_count != 1) {
            throw std::invalid_argument(
                "LevelSet: expected exactly one Computational level with basis_bit == One, "
                "got " +
                std::to_string(computational_one_count));
        }
    }

    void require_in_range(uint8_t level_id, const char* fn) const {
        if (level_id >= levels_.size()) {
            throw std::invalid_argument(std::string("LevelSet::") + fn + ": level_id " +
                                        std::to_string(level_id) + " out of range (size " +
                                        std::to_string(levels_.size()) + ")");
        }
    }

    void check_kind(uint8_t level_id, LevelCategory expected, const char* fn) const {
        require_in_range(level_id, fn);
        if (levels_[level_id].category != expected) {
            throw std::invalid_argument(std::string("LevelSet::") + fn + ": level '" +
                                        levels_[level_id].label +
                                        "' has category that does not match the requested kind");
        }
    }
};

}  // namespace clifft
