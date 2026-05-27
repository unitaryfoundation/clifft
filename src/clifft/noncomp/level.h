#pragma once

// Level categories and the per-level model record for the
// noncomputational trajectory MVP. See design/noncomputational-mvp.md.
//
// A Level is a model-defined tag with a stable integer id (its index
// into the level table), a human-readable label, a LevelCategory, and
// an optional basis_bit identifying which computational basis state a
// Computational-category level represents.

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
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
    // Required when category == Computational; must be omitted otherwise.
    // Identifies the computational basis state (0 or 1) this level
    // represents. Used by the rewriter to prepend an X prep for
    // basis_bit == 1 initial samples.
    std::optional<uint8_t> basis_bit;
};

// Sqale-aligned default level set: g, e, leak_g, leak_e, lost.
// Stable order; the integer ids are positional.
inline std::vector<Level> default_levels() {
    return {
        Level{"g", LevelCategory::Computational, uint8_t{0}},
        Level{"e", LevelCategory::Computational, uint8_t{1}},
        Level{"leak_g", LevelCategory::Leaked, std::nullopt},
        Level{"leak_e", LevelCategory::Leaked, std::nullopt},
        Level{"lost", LevelCategory::Lost, std::nullopt},
    };
}

// Validates the structural invariants of a level set. Throws
// std::invalid_argument naming the offending entry on failure.
//
// Checks:
//   - level set is non-empty and at most 128 entries (so a level_id
//     fits in 7 bits if a future implementation packs the byte);
//   - every Computational-category level has a basis_bit in {0, 1};
//   - every Leaked/Lost-category level has basis_bit unset.
inline void validate_levels(std::span<const Level> levels) {
    if (levels.empty()) {
        throw std::invalid_argument("validate_levels: level set is empty");
    }
    if (levels.size() > 128) {
        throw std::invalid_argument("validate_levels: level set has " +
                                    std::to_string(levels.size()) +
                                    " entries; max supported is 128");
    }
    for (size_t i = 0; i < levels.size(); ++i) {
        const Level& lv = levels[i];
        if (lv.category == LevelCategory::Computational) {
            if (!lv.basis_bit.has_value()) {
                throw std::invalid_argument("validate_levels: level '" + lv.label + "' (id " +
                                            std::to_string(i) +
                                            ") is Computational but has no basis_bit");
            }
            if (*lv.basis_bit > 1) {
                throw std::invalid_argument("validate_levels: level '" + lv.label +
                                            "' has basis_bit = " + std::to_string(*lv.basis_bit) +
                                            "; must be 0 or 1");
            }
        } else {
            if (lv.basis_bit.has_value()) {
                throw std::invalid_argument("validate_levels: level '" + lv.label + "' (id " +
                                            std::to_string(i) +
                                            ") is non-Computational but has basis_bit set");
            }
        }
    }
}

}  // namespace clifft
