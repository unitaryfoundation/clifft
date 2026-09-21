#include "clifft/sampling/folded/region.h"

namespace clifft::sampling::folded {

void RegionPlan::certify(std::span<const ActiveExpectation> probes) const {
    if (probes.size() != 2 * rank_ + 3 + ancilla_count_)
        throw std::invalid_argument("incorrect folded boundary probe set");
    unsigned axes = 0;
    for (size_t i = 0; i < probes.size(); ++i) {
        const auto p = probes[i].projection;
        if (i < 2 * rank_ || i >= 2 * rank_ + 3) {
            if (!p.is_identity())
                throw std::invalid_argument(
                    "folded boundary is not a signed CSS code with classical ancillas");
        } else {
            const auto label = p.x | (p.z << 1);
            if (!label || label > 3 || (axes & (1 << label)))
                throw std::invalid_argument("folded logical probes do not span one active qubit");
            axes |= 1 << label;
        }
    }
}

std::vector<RecordSlot> RegionPlan::output_records() const {
    std::vector<RecordSlot> result;
    for (const auto* block : {&body_, &final_})
        for (const auto& event : block->events)
            if (event.kind == 0 || event.kind == 3)
                result.push_back(RecordSlot{static_cast<uint32_t>(event.b)});
    for (const auto& event : post_)
        result.push_back(RecordSlot{static_cast<uint32_t>(event.b)});
    return result;
}

Workspace::Workspace(const RegionPlan& plan)
    : plan_(plan),
      kernel_(plan.kernel),
      verification_values_(plan.verification_rules_.size()),
      fault_choices(plan.sites_.size()),
      local_(4 * plan.characters_),
      record_flips_(plan.visible_ + plan.hidden_) {
    for (auto& record : choice_records_)
        record.resize(plan.visible_ + plan.hidden_);
    for (auto& choice : inputs_)
        for (auto& terms : choice)
            for (auto& term : terms)
                term.local.resize(4 * plan.leaves_);
    for (size_t i = 0; i < roots_.size(); ++i)
        roots_[i] = std::polar(1.0, std::numbers::pi * double(i) / 4);
}

void Workspace::run(std::span<const double> probes, std::span<const uint8_t> symbols,
                    const std::vector<std::vector<uint32_t>>& faults, std::span<uint8_t> output,
                    clifft::Xoshiro256PlusPlus& rng) noexcept {
    noise_rng_ = &rng;
    kernel_.use_rng(rng);
    assert(probes.size() == 2 * plan_.rank_ + 3 + plan_.ancilla_count_);
    assert(output.size() == plan_.visible_ + plan_.hidden_);
    records = output;
    std::fill(record_flips_.begin(), record_flips_.end(), 0);
    post_x_ = post_z_ = post_ancillas_ = 0;
    for (size_t i = 0; i < faults.size(); ++i) {
        fault_choices[i] = 0;
        for (size_t j = 0; j < faults[i].size(); ++j) {
            if (!symbols[faults[i][j]])
                continue;
            fault_choices[i] = j + 1;
            if (!plan_.sites_[i].effects.empty()) {
                const auto& effect = plan_.sites_[i].effects[j];
                post_x_ ^= effect.x;
                post_z_ ^= effect.z;
                post_ancillas_ ^= effect.ancillas;
                for (auto slot : effect.records)
                    record_flips_[slot] ^= 1;
            }
            break;
        }
    }
    Boundary boundary;
    for (size_t i = 0; i < plan_.rank_; ++i) {
        boundary.syndrome |= uint64_t(probes[i] < 0) << i;
        if (probes[plan_.rank_ + i] < 0)
            boundary.offset ^= plan_.z_duals_[i];
    }
    for (size_t i = 0; i < plan_.ancilla_count_; ++i)
        boundary.ancillas |= uint64_t(probes[2 * plan_.rank_ + 3 + i] < 0) << i;
    const double x = probes[2 * plan_.rank_], y = probes[2 * plan_.rank_ + 1];
    const double z = std::clamp(probes[2 * plan_.rank_ + 2], -1.0, 1.0);
    if (z >= 0) {
        boundary.logical[0] = std::sqrt((1 + z) / 2);
        boundary.logical[1] = Complex(x, y) / (2 * boundary.logical[0].real());
    } else {
        boundary.logical[1] = std::sqrt((1 - z) / 2);
        boundary.logical[0] = Complex(x, -y) / (2 * boundary.logical[1].real());
    }
    sample_body(boundary);
    finish(boundary);
}

}  // namespace clifft::sampling::folded
