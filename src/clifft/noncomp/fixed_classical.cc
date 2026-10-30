#include "clifft/noncomp/fixed_classical.h"

#include "clifft/noncomp/sample.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace clifft::detail {
namespace {

uint32_t table_index(size_t size) {
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("fixed classical program exceeds its table index range");
    }
    return static_cast<uint32_t>(size);
}

bool same_channel(const ReadoutChannel& a, const ReadoutChannel& b) {
    return a.p01 == b.p01 && a.p10 == b.p10 && a.herald == b.herald;
}

template <typename Visit>
void for_set_lanes(std::span<const uint64_t> words, Visit&& visit) noexcept {
    for (size_t word = 0; word < words.size(); ++word) {
        uint64_t bits = words[word];
        while (bits != 0) {
            visit(static_cast<uint32_t>(word * 64 + std::countr_zero(bits)));
            bits &= bits - 1;
        }
    }
}

}  // namespace

FixedClassicalWorkspace::FixedClassicalWorkspace(uint32_t qubits, uint32_t capacity)
    : qubits_(qubits), capacity_(capacity), bits_(bit_columns(qubits), capacity) {}

FixedClassicalProgram::FixedClassicalProgram(std::span<const InputStep> inputs, uint32_t qubits,
                                             uint32_t noise_inputs, uint32_t readout_inputs,
                                             uint32_t records)
    : qubits_(qubits) {
    const auto check_qubit = [&](uint32_t qubit) {
        if (qubit >= qubits) {
            throw std::logic_error("fixed classical qubit is out of range");
        }
    };
    const auto check_readout = [&](uint32_t input) {
        if (input >= readout_inputs) {
            throw std::logic_error("fixed classical readout input is out of range");
        }
    };
    const NoiseInput* condition = nullptr;
    for (const auto& input : inputs) {
        std::visit(
            [&](const auto& step) {
                using T = std::decay_t<decltype(step)>;
                if constexpr (std::is_same_v<T, TransitionInput>) {
                    check_qubit(step.qubit);
                    operations_.emplace_back(Transition{
                        step.qubit, sampling::PreparedBernoulli(step.probability), step.loss});
                    condition = nullptr;
                } else if constexpr (std::is_same_v<T, ResetInput>) {
                    check_qubit(step.qubit);
                    for (auto destination : step.destinations) {
                        if (destination != kComputational && destination != kLeaked &&
                            destination != kLost) {
                            throw std::logic_error("fixed classical reset destination is invalid");
                        }
                    }
                    operations_.emplace_back(step);
                    condition = nullptr;
                } else if constexpr (std::is_same_v<T, NoiseInput>) {
                    for (auto qubit : step.active_targets) {
                        check_qubit(qubit);
                    }
                    if (step.inactive_source) {
                        check_qubit(*step.inactive_source);
                        if (step.source_status != kLeaked && step.source_status != kLost) {
                            throw std::logic_error("fixed classical interaction status is invalid");
                        }
                    }
                    for (const auto& outcome : step.outcomes) {
                        if (outcome.input >= noise_inputs) {
                            throw std::logic_error("fixed classical noise input is out of range");
                        }
                    }
                    if (step.outcomes.empty() ||
                        step.outcomes.back().cumulative_probability == 0.0) {
                        return;
                    }
                    // Only status writes invalidate this mask. Preserve the full
                    // original operand set, including grouped ordinary noise.
                    if (!condition || condition->active_targets != step.active_targets ||
                        condition->inactive_source != step.inactive_source ||
                        condition->source_status != step.source_status) {
                        const auto begin = table_index(operands_.size());
                        operands_.insert(operands_.end(), step.active_targets.begin(),
                                         step.active_targets.end());
                        table_index(operands_.size());
                        operations_.emplace_back(
                            Condition{begin, table_index(step.active_targets.size()),
                                      step.inactive_source, step.source_status});
                        condition = &step;
                    }
                    const auto begin = table_index(outcomes_.size());
                    outcomes_.insert(outcomes_.end(), step.outcomes.begin(), step.outcomes.end());
                    table_index(outcomes_.size());
                    operations_.emplace_back(Noise{
                        begin, table_index(step.outcomes.size()),
                        sampling::PreparedBernoulli(step.outcomes.back().cumulative_probability)});
                } else if constexpr (std::is_same_v<T, ReadoutInput>) {
                    check_readout(step.input);
                    if (step.record >= records) {
                        throw std::logic_error("fixed classical record is out of range");
                    }
                    if (step.qubit) {
                        check_qubit(*step.qubit);
                    }
                    uint8_t used = 0;
                    for (size_t category = 0; category < kStatusBits.size(); ++category) {
                        if (used & kStatusBits[category]) {
                            continue;
                        }
                        const auto& channel = step.channels[category];
                        uint8_t categories = 0;
                        for (size_t other = category; other < kStatusBits.size(); ++other) {
                            if (same_channel(channel, step.channels[other])) {
                                categories |= kStatusBits[other];
                            }
                        }
                        used |= categories;
                        // A shared uniform draw gives nested flips: P(both) is
                        // min(p01, p10). Select the high event, then both within it.
                        const double high = std::max(channel.p01, channel.p10);
                        const double low = std::min(channel.p01, channel.p10);
                        if (high == 0.0 && channel.herald == 0.0) {
                            continue;
                        }
                        operations_.emplace_back(Readout{
                            step.input, step.record, step.qubit, categories,
                            channel.p01 >= channel.p10, sampling::PreparedBernoulli(channel.herald),
                            sampling::PreparedBernoulli(high),
                            sampling::PreparedBernoulli(high == 0.0 ? 0.0 : low / high)});
                    }
                } else if constexpr (std::is_same_v<T, ProbeInput>) {
                    check_readout(step.input);
                    check_qubit(step.qubit);
                    if (step.status != kLeaked && step.status != kLost) {
                        throw std::logic_error("fixed classical probe status is invalid");
                    }
                    operations_.emplace_back(step);
                }
            },
            input);
    }
}

void FixedClassicalProgram::prepare(const SeedRoot& root, uint32_t first, uint32_t count,
                                    FixedClassicalWorkspace& workspace,
                                    sampling::BatchExternalInputs& inputs,
                                    NonComputationalSample& output) const noexcept {
    assert(workspace.qubits_ == qubits_ && count <= workspace.capacity_);
    assert(uint64_t{first} + count <= output.shots);
    assert(output.final_status.empty() || output.num_qubits == qubits_);
    workspace.bits_.clear();
    inputs.clear();
    const auto bits = workspace.bits_.mutable_view();
    const size_t scratch = size_t{2} * qubits_;
    const auto live = bits.column(scratch);
    const auto condition = bits.column(scratch + 1);
    const auto eligible = bits.column(scratch + 2);
    const auto events = bits.column(scratch + 3);
    const auto secondary = bits.column(scratch + 4);
    sampling::fill_low_lane_mask(live, count);
    const auto destination = inputs.mutable_view();
    const auto seed = derive_state(root, first, kFixedNonComputationalBatchInputDomain);
    Xoshiro256PlusPlus rng;
    rng.seed_full(seed[0], seed[1], seed[2], seed[3]);
    const auto status_mask = [&](std::optional<uint32_t> qubit, uint8_t categories) noexcept {
        if (!qubit) {
            for (size_t word = 0; word < live.size(); ++word) {
                eligible[word] = (categories & kComputational) ? live[word] : 0;
            }
            return;
        }
        const auto leaked = bits.column(size_t{2} * *qubit);
        const auto lost = bits.column(size_t{2} * *qubit + 1);
        for (size_t word = 0; word < live.size(); ++word) {
            eligible[word] =
                ((categories & kComputational) ? live[word] & ~(leaked[word] | lost[word]) : 0) |
                ((categories & kLeaked) ? leaked[word] : 0) |
                ((categories & kLost) ? lost[word] : 0);
        }
    };
    for (const auto& operation : operations_) {
        std::visit(
            [&](const auto& step) noexcept {
                using T = std::decay_t<decltype(step)>;
                if constexpr (std::is_same_v<T, Transition>) {
                    status_mask(step.qubit, step.loss ? kComputational | kLeaked : kComputational);
                    sampling::sample_bernoulli_bits(step.event, eligible, events, count, rng);
                    const auto leaked = bits.column(size_t{2} * step.qubit);
                    const auto lost = bits.column(size_t{2} * step.qubit + 1);
                    for (size_t word = 0; word < live.size(); ++word) {
                        if (step.loss) {
                            lost[word] |= events[word];
                            leaked[word] &= ~events[word];
                        } else {
                            leaked[word] |= events[word];
                        }
                    }
                } else if constexpr (std::is_same_v<T, ResetInput>) {
                    const auto leaked = bits.column(size_t{2} * step.qubit);
                    const auto lost = bits.column(size_t{2} * step.qubit + 1);
                    for (size_t word = 0; word < live.size(); ++word) {
                        const std::array<uint64_t, 3> before{
                            live[word] & ~(leaked[word] | lost[word]), leaked[word], lost[word]};
                        uint64_t next_leaked = 0, next_lost = 0;
                        for (size_t category = 0; category < before.size(); ++category) {
                            if (step.destinations[category] == kLeaked) {
                                next_leaked |= before[category];
                            }
                            if (step.destinations[category] == kLost) {
                                next_lost |= before[category];
                            }
                        }
                        leaked[word] = next_leaked;
                        lost[word] = next_lost;
                    }
                } else if constexpr (std::is_same_v<T, Condition>) {
                    if (step.source) {
                        const auto source =
                            bits.column(size_t{2} * *step.source + (step.source_status == kLost));
                        std::ranges::copy(source, condition.begin());
                    } else {
                        std::ranges::copy(live, condition.begin());
                    }
                    for (uint32_t operand = step.begin; operand < step.begin + step.count;
                         ++operand) {
                        const auto leaked = bits.column(size_t{2} * operands_[operand]);
                        const auto lost = bits.column(size_t{2} * operands_[operand] + 1);
                        for (size_t word = 0; word < live.size(); ++word) {
                            condition[word] &= ~(leaked[word] | lost[word]);
                        }
                    }
                } else if constexpr (std::is_same_v<T, Noise>) {
                    sampling::sample_bernoulli_bits(step.event, condition, events, count, rng);
                    if (step.count == 1) {
                        const auto column = destination.noise(outcomes_[step.begin].input);
                        for (size_t word = 0; word < live.size(); ++word) {
                            column[word] |= events[word];
                        }
                        return;
                    }
                    for_set_lanes(events, [&](uint32_t lane) noexcept {
                        uint32_t outcome = step.begin;
                        const double draw = rng.next_double() * step.event.probability();
                        while (outcome + 1 < step.begin + step.count &&
                               draw >= outcomes_[outcome].cumulative_probability) {
                            ++outcome;
                        }
                        destination.set_noise(outcomes_[outcome].input, lane);
                    });
                } else if constexpr (std::is_same_v<T, Readout>) {
                    status_mask(step.qubit, step.categories);
                    const auto zero = destination.readout_zero(step.input);
                    const auto one = destination.readout_one(step.input);
                    if (step.herald.probability() > 0.0) {
                        sampling::sample_bernoulli_bits(step.herald, eligible, events, count, rng);
                        for_set_lanes(events, [&](uint32_t lane) noexcept {
                            output.heralds[static_cast<size_t>(first + lane) *
                                               output.num_measurements +
                                           step.record] = 1;
                        });
                        sampling::sample_bernoulli_bits(fair_, events, secondary, count, rng);
                        for (size_t word = 0; word < live.size(); ++word) {
                            eligible[word] &= ~events[word];
                            zero[word] |= secondary[word];
                            one[word] |= secondary[word];
                        }
                    }
                    sampling::sample_bernoulli_bits(step.high, eligible, events, count, rng);
                    sampling::sample_bernoulli_bits(step.both_given_high, events, secondary, count,
                                                    rng);
                    for (size_t word = 0; word < live.size(); ++word) {
                        zero[word] |= step.high_is_zero ? events[word] : secondary[word];
                        one[word] |= step.high_is_zero ? secondary[word] : events[word];
                    }
                } else if constexpr (std::is_same_v<T, ProbeInput>) {
                    const auto source =
                        bits.column(size_t{2} * step.qubit + (step.status == kLost));
                    std::ranges::copy(source, destination.readout_zero(step.input).begin());
                    std::ranges::copy(source, destination.readout_one(step.input).begin());
                }
            },
            operation);
    }
    if (!output.final_status.empty()) {
        for (uint32_t qubit = 0; qubit < qubits_; ++qubit) {
            assert(std::ranges::all_of(bits.column(size_t{2} * qubit),
                                       [](uint64_t word) { return word == 0; }));
            for_set_lanes(bits.column(size_t{2} * qubit + 1), [&](uint32_t lane) noexcept {
                output.final_status[static_cast<size_t>(first + lane) * output.num_qubits + qubit] =
                    QubitStatus::Lost;
            });
        }
    }
}

}  // namespace clifft::detail
