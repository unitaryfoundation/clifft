#include "clifft/optimizer/tohpe.h"

#include <algorithm>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>

namespace clifft::phase_detail {

namespace {

// TOHPE has cubic worst-case cost in the number of columns. A fixed search
// bound keeps the default compiler inexpensive; projection still works above it.
constexpr size_t kMaxTerms = 128;
using Selection = std::bitset<kMaxTerms>;

struct Row {
    std::vector<uint64_t> features;
    Selection selection;
};

Selection dependency(const std::vector<uint64_t>& columns, uint32_t width) {
    const size_t feature_count = width * (width + 1) / 2;
    std::vector<std::optional<Row>> pivots(feature_count);
    for (size_t i = 0; i < columns.size(); ++i) {
        Row row{std::vector<uint64_t>((feature_count + 63) / 64), {}};
        row.selection.set(i);
        row.features[0] = columns[i];
        for (uint64_t bits = columns[i]; bits; bits &= bits - 1) {
            const auto a = std::countr_zero(bits);
            for (uint64_t lower = columns[i] & ((uint64_t{1} << a) - 1); lower;
                 lower &= lower - 1) {
                const size_t feature = width + a * (a - 1) / 2 + std::countr_zero(lower);
                row.features[feature / 64] |= uint64_t{1} << (feature % 64);
            }
        }
        bool independent = false;
        for (size_t word = row.features.size(); word-- > 0;) {
            while (row.features[word]) {
                const size_t pivot = word * 64 + 63 - std::countl_zero(row.features[word]);
                if (!pivots[pivot]) {
                    pivots[pivot] = std::move(row);
                    independent = true;
                    break;
                }
                for (size_t j = 0; j <= word; ++j) {
                    row.features[j] ^= pivots[pivot]->features[j];
                }
                row.selection ^= pivots[pivot]->selection;
            }
            if (independent) {
                break;
            }
        }
        if (!independent) {
            // An odd selection of every column only adds back the removed
            // column. As the first dependency it also spans the whole kernel.
            if (row.selection.count() == columns.size() && columns.size() % 2) {
                return {};
            }
            return row.selection;
        }
    }
    return {};
}

}  // namespace

std::vector<uint64_t> tohpe(std::vector<uint64_t> columns, uint32_t core_width) {
    assert(core_width <= 64);
    if (columns.size() <= core_width || columns.size() > kMaxTerms) {
        return columns;
    }
    std::ranges::sort(columns);
    // Algorithm 2 of Vandaele, arXiv:2407.08695. The kernel constraints on
    // linear and pair features preserve all cubic signature moments when the
    // selected columns are translated together. Odd selections add the shift.
    while (columns.size() > core_width) {
        const auto selected = dependency(columns, core_width);
        if (selected.none()) {
            break;
        }
        const bool odd = selected.count() % 2;
        std::map<uint64_t, size_t> scores;
        for (size_t i = 0; i < columns.size(); ++i) {
            // A translated column can vanish, or cancel the extra odd column.
            scores[columns[i]] += selected[i] ? 1 : 2 * odd;
            for (size_t j = 0; j < i; ++j) {
                if (selected[i] != selected[j]) {
                    scores[columns[i] ^ columns[j]] += 2;
                }
            }
        }
        uint64_t shift = 0;
        size_t best = odd;
        for (const auto& [candidate, score] : scores) {
            if (score > best) {
                best = score;
                shift = candidate;
            }
        }
        if (!shift) {
            break;
        }
        [[maybe_unused]] const size_t before = columns.size();
        for (size_t i = 0; i < columns.size(); ++i) {
            if (selected[i]) {
                columns[i] ^= shift;
            }
        }
        if (odd) {
            columns.push_back(shift);
        }
        std::ranges::sort(columns);
        size_t written = 0;
        for (size_t i = 0; i < columns.size();) {
            size_t end = i + 1;
            while (end < columns.size() && columns[end] == columns[i]) {
                ++end;
            }
            if (columns[i] && (end - i) % 2) {
                columns[written++] = columns[i];
            }
            i = end;
        }
        columns.resize(written);
        assert(columns.size() < before);
    }
    return columns;
}

}  // namespace clifft::phase_detail
