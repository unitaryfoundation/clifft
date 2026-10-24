#include "clifft/zx/phase.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace clifft::zx {

Params::Params(std::vector<uint32_t> ids) : ids_(std::move(ids)) {
    std::ranges::sort(ids_);
    // Repeated IDs cancel in pairs, as XOR of the same bit does.
    std::vector<uint32_t> reduced;
    reduced.reserve(ids_.size());
    for (uint32_t id : ids_) {
        if (!reduced.empty() && reduced.back() == id) {
            reduced.pop_back();
        } else {
            reduced.push_back(id);
        }
    }
    ids_ = std::move(reduced);
}

void Params::xor_with(const Params& other) {
    if (other.ids_.empty()) {
        return;
    }
    std::vector<uint32_t> merged;
    merged.reserve(ids_.size() + other.ids_.size());
    std::ranges::set_symmetric_difference(ids_, other.ids_, std::back_inserter(merged));
    ids_ = std::move(merged);
}

}  // namespace clifft::zx
