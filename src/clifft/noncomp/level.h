#pragma once

// Level definitions and the validated level table.
//
// A Level is a model-defined tag with a stable integer id (its index
// into the level table), a human-readable label, and a LevelCategory.
//
// A level table must contain exactly two Computational levels; in table
// order the first is the |0> state and the second is |1>. The rewriter
// uses computational_one_id() to prepend an X prep for a sampled known-|1>
// initial level, or when a transition materializes the carrier at the |1>
// level (the SVM default initialization is |0...0>). Leaked and Lost levels
// carry no basis information -- "lost from |1>" provenance, if ever needed,
// belongs in an event record or in distinct levels, not in the level tag.
//
// LevelSet wraps a std::vector<Level>, runs validation in its ctor,
// and owns the QubitStatus factories that bind a level id to this
// specific table. Construction of QubitStatus values for non-Unknown
// kinds should go through LevelSet so the (kind, level_id) pair is
// guaranteed to refer to a real level of the matching category in a
// known table.

#include "clifft/noncomp/qubit_status.h"

#include <cstdint>
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

struct Level {
    std::string label;
    LevelCategory category;
};

class LevelSet {
  public:
    // Validates the level table; throws std::invalid_argument with a
    // named field on failure.
    explicit LevelSet(std::vector<Level> levels) : levels_(std::move(levels)) {
        validate();
        fingerprint_ = compute_fingerprint();
        assign_computational_ids();
    }

    // Default level set: g, e, leak_g, leak_e, lost.
    // Stable order; the integer ids are positional (g = |0>, e = |1>).
    static LevelSet default_set() {
        return LevelSet({
            Level{"g", LevelCategory::Computational},
            Level{"e", LevelCategory::Computational},
            Level{"leak_g", LevelCategory::Leaked},
            Level{"leak_e", LevelCategory::Leaked},
            Level{"lost", LevelCategory::Lost},
        });
    }

    std::span<const Level> levels() const { return levels_; }
    size_t size() const { return levels_.size(); }

    // Deterministic schema signature over (label, category) per level, in
    // order. Two tables that agree on every level's identity share a
    // fingerprint; any difference in labels, order, or categories changes
    // it. Instruments and classifiers record the fingerprint of the table
    // they were built against so a model can reject a component bound to a
    // different table.
    uint64_t fingerprint() const { return fingerprint_; }

    const Level& at(uint8_t level_id) const {
        require_in_range(level_id, "at");
        return levels_[level_id];
    }

    // QubitStatus factories: validate that level_id refers to a level
    // of the matching category in this table.
    QubitStatus leaked(uint8_t level_id) const {
        check_kind(level_id, LevelCategory::Leaked, "leaked");
        return QubitStatus::leaked_unchecked(level_id);
    }

    QubitStatus lost(uint8_t level_id) const {
        check_kind(level_id, LevelCategory::Lost, "lost");
        return QubitStatus::lost_unchecked(level_id);
    }

    // Ids of the two Computational levels in table order: the first is the
    // |0> state (g), the second is |1> (e). Validation guarantees exactly
    // two Computational levels, so these always refer to real levels.
    uint8_t computational_zero_id() const { return computational_zero_id_; }
    uint8_t computational_one_id() const { return computational_one_id_; }

    // QubitStatus for a level id, dispatched on the level's category:
    // Computational -> Computational (the level is not carried: which
    // basis state a computational qubit holds is SVM runtime
    // information), Leaked -> Leaked, Lost -> Lost. Used by the status
    // walks to turn a transition destination into the resulting qubit
    // status.
    QubitStatus status_for(uint8_t level_id) const {
        require_in_range(level_id, "status_for");
        switch (levels_[level_id].category) {
            case LevelCategory::Computational:
                return QubitStatus::computational();
            case LevelCategory::Leaked:
                return QubitStatus::leaked_unchecked(level_id);
            case LevelCategory::Lost:
                return QubitStatus::lost_unchecked(level_id);
        }
        throw std::invalid_argument("LevelSet::status_for: unrecognized category");
    }

  private:
    std::vector<Level> levels_;
    uint64_t fingerprint_ = 0;
    uint8_t computational_zero_id_ = kInvalidLevel;
    uint8_t computational_one_id_ = kInvalidLevel;

    // validate() guarantees exactly two Computational levels; the first in
    // table order is |0>, the second is |1>.
    void assign_computational_ids() {
        bool have_zero = false;
        for (size_t i = 0; i < levels_.size(); ++i) {
            if (levels_[i].category != LevelCategory::Computational) {
                continue;
            }
            if (!have_zero) {
                computational_zero_id_ = static_cast<uint8_t>(i);
                have_zero = true;
            } else {
                computational_one_id_ = static_cast<uint8_t>(i);
                return;
            }
        }
    }

    // FNV-1a over a canonical byte serialization of each level. The
    // label is length-prefixed so that, e.g., {"ab", "c"} and {"a",
    // "bc"} do not collide. Hand-rolled (not std::hash) so the value is
    // stable across platforms and runs.
    uint64_t compute_fingerprint() const {
        constexpr uint64_t kOffsetBasis = 1469598103934665603ULL;
        constexpr uint64_t kPrime = 1099511628211ULL;
        uint64_t h = kOffsetBasis;
        const auto mix = [&h](uint8_t byte) {
            h ^= byte;
            h *= kPrime;
        };
        for (const Level& lv : levels_) {
            const uint64_t len = lv.label.size();
            for (int i = 0; i < 8; ++i) {
                mix(static_cast<uint8_t>((len >> (8 * i)) & 0xFFu));
            }
            for (const char c : lv.label) {
                mix(static_cast<uint8_t>(c));
            }
            mix(static_cast<uint8_t>(lv.category));
        }
        return h;
    }

    void validate() const {
        if (levels_.empty()) {
            throw std::invalid_argument("LevelSet: level set is empty");
        }
        if (levels_.size() > 128) {
            throw std::invalid_argument("LevelSet: level set has " +
                                        std::to_string(levels_.size()) +
                                        " entries; max supported is 128");
        }
        size_t computational_count = 0;
        for (size_t i = 0; i < levels_.size(); ++i) {
            const Level& lv = levels_[i];
            switch (lv.category) {
                case LevelCategory::Computational:
                    ++computational_count;
                    break;
                case LevelCategory::Leaked:
                case LevelCategory::Lost:
                    break;
                default:
                    throw std::invalid_argument("LevelSet: level '" + lv.label + "' (id " +
                                                std::to_string(i) +
                                                ") has an unrecognized LevelCategory value");
            }
        }
        if (computational_count != 2) {
            throw std::invalid_argument(
                "LevelSet: expected exactly two Computational levels (the first is |0>, the "
                "second |1>), got " +
                std::to_string(computational_count));
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
