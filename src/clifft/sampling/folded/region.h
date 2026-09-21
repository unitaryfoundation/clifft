#pragma once

#include "clifft/sampling/folded/kernel.h"
#include "clifft/sampling/folded/mask.h"
#include "clifft/sampling/plan.h"

#include <bit>
#include <numbers>
#include <span>

namespace clifft::sampling::folded {

inline bool parity(Mask value) noexcept {
    return (std::popcount(uint64_t(value)) + std::popcount(uint64_t(value >> 64))) & 1;
}

inline Mask read_mask(Reader& reader) {
    std::string token;
    if (!(reader.input >> token) || token.empty())
        throw std::invalid_argument("missing physical mask");
    Mask value = 0;
    for (char c : token) {
        constexpr Mask limit{11068046444225730969ULL, 1844674407370955161ULL};
        if (c < '0' || c > '9' || value > limit || (value == limit && c > '5'))
            throw std::invalid_argument("invalid physical mask");
        value = (value << 3) + (value << 1) + Mask{unsigned(c - '0')};
    }
    return value;
}

struct Event {
    size_t kind, a, b;
    Mask c;
};
struct Effect {
    Mask x, z;
    uint64_t ancillas;
    std::vector<size_t> records;
};
struct Noise {
    double probability;
    std::vector<size_t> targets;
    std::vector<std::vector<size_t>> channels;
    std::vector<Effect> effects;
};
struct Block {
    std::vector<uint64_t> shifts;
    std::vector<Event> events;
};
struct Boundary {
    Mask offset = 0;
    uint64_t syndrome = 0, ancillas = 0;
    // Amplitudes use the physical logical X and Z axes at every block boundary.
    std::array<Complex, 2> logical{};
};
struct RegionPlan {
    size_t n_, ancilla_count_, visible_, hidden_, prefix_visible_, prefix_hidden_;
    size_t rank_, leaves_, characters_;
    Mask logical_x_, logical_z_;
    std::vector<Mask> x_rows_, z_duals_;
    std::vector<std::array<size_t, 2>> pairs_;
    std::vector<Noise> sites_;
    Block body_, final_;
    std::vector<Event> post_;
    std::vector<std::vector<std::array<size_t, 2>>> verification_rules_;
    KernelPlan kernel;
    RegionPlan(const std::string& text, const std::vector<std::string>& tables, size_t rank,
               size_t leaves, size_t characters)
        : rank_(rank),
          leaves_(leaves),
          characters_(characters),
          kernel(tables, rank, leaves, characters) {
        std::istringstream input(text);
        Reader reader{input};
        if (reader.size() != 1)
            throw std::invalid_argument("unsupported folded block version");
        n_ = reader.size();
        ancilla_count_ = reader.size();
        visible_ = reader.size();
        hidden_ = reader.size();
        prefix_visible_ = reader.size();
        prefix_hidden_ = reader.size();
        logical_x_ = read_mask(reader);
        logical_z_ = read_mask(reader);
        if (!n_ || n_ > 127 || !ancilla_count_ || ancilla_count_ > 63 || n_ != 2 * rank_ + 1 ||
            prefix_visible_ > visible_ || prefix_hidden_ > hidden_)
            throw std::invalid_argument("invalid protocol dimensions");
        const auto data_mask = (Mask{1} << n_) - 1;
        for (auto* rows : {&x_rows_, &z_duals_}) {
            rows->resize(rank_);
            for (auto& row : *rows) {
                row = read_mask(reader);
                if (row & ~data_mask)
                    throw std::invalid_argument("invalid code mask");
            }
        }
        pairs_.resize(reader.size());
        if (characters_ != n_ + pairs_.size() || ((logical_x_ | logical_z_) & ~data_mask))
            throw std::invalid_argument("invalid local factor geometry");
        for (auto& pair : pairs_) {
            pair = {reader.size(), reader.size()};
            if (pair[0] >= n_ || pair[1] >= n_ || pair[0] == pair[1])
                throw std::invalid_argument("invalid mirror pair");
        }
        sites_.resize(reader.size());
        for (auto& site : sites_) {
            site.probability = reader.number();
            if (site.probability < 0 || site.probability > 1)
                throw std::invalid_argument("invalid physical noise probability");
            site.targets.resize(reader.size());
            if (site.targets.empty() || site.targets.size() > 3)
                throw std::invalid_argument("invalid physical noise width");
            for (auto& target : site.targets) {
                target = reader.size();
                if (target >= n_ + ancilla_count_)
                    throw std::invalid_argument("invalid noise target");
            }
            site.channels.resize(reader.size());
            if (site.channels.empty() || site.channels.size() > 63)
                throw std::invalid_argument("invalid physical channel count");
            for (auto& channel : site.channels) {
                channel.resize(site.targets.size());
                for (auto& axis : channel) {
                    axis = reader.size();
                    if (axis > 3)
                        throw std::invalid_argument("invalid Pauli axis");
                }
            }
            site.effects.resize(reader.size());
            if (!site.effects.empty() && site.effects.size() != site.channels.size())
                throw std::invalid_argument("invalid continuation channel effects");
            for (auto& effect : site.effects) {
                effect.x = read_mask(reader);
                effect.z = read_mask(reader);
                effect.ancillas = reader.word();
                if (((effect.x | effect.z) & ~data_mask) || (effect.ancillas >> ancilla_count_))
                    throw std::invalid_argument("invalid continuation frame");
                effect.records.resize(reader.size());
                for (auto& slot : effect.records) {
                    slot = reader.size();
                    if (slot >= visible_ + hidden_)
                        throw std::invalid_argument("invalid record effect");
                }
            }
        }
        for (auto* block : {&body_, &final_}) {
            const size_t count = block == &body_ ? 4 : 2;
            if (reader.size() != count)
                throw std::invalid_argument("invalid branch count");
            block->shifts.resize(count);
            for (auto& shift : block->shifts) {
                shift = reader.word();
                if (shift >> (rank_ + 1))
                    throw std::invalid_argument("invalid code translation");
            }
            block->events.resize(reader.size());
            for (auto& e : block->events) {
                e = {reader.size(), reader.size(), reader.size(), read_mask(reader)};
                if (e.kind > 10 || (e.kind != 10 && e.c))
                    throw std::invalid_argument("invalid folded event");
                if (e.kind == 9) {
                    if (e.a >= sites_.size() || !sites_[e.a].effects.empty())
                        throw std::invalid_argument("invalid monomial fault site");
                } else if (e.kind == 4 || e.kind == 5) {
                    if (e.a >= n_)
                        throw std::invalid_argument("invalid T target");
                } else {
                    if (e.a < n_ || e.a >= n_ + ancilla_count_)
                        throw std::invalid_argument("invalid cat target");
                    if ((e.kind == 0 || e.kind == 3) && e.b >= visible_ + hidden_)
                        throw std::invalid_argument("invalid cat record slot");
                    if ((e.kind == 1 || e.kind == 2 || e.kind == 10) && e.b >= (count == 4 ? 2 : 1))
                        throw std::invalid_argument("invalid cat branch bit");
                    if (e.kind == 6 && (e.b < n_ || e.b >= n_ + ancilla_count_))
                        throw std::invalid_argument("invalid cat CNOT target");
                    if (e.kind == 7 && e.b >= n_)
                        throw std::invalid_argument("invalid data CNOT");
                    if (e.kind == 8 && e.b >= pairs_.size())
                        throw std::invalid_argument("invalid CZ pair");
                }
            }
        }
        post_.resize(reader.size());
        for (auto& e : post_) {
            e = {reader.size(), reader.size(), reader.size(), read_mask(reader)};
            if (e.kind > 2 || e.a >= ancilla_count_ || e.b >= visible_ + hidden_ ||
                (e.kind == 0 && e.c >= rank_) || (e.kind == 1 && (e.c & ~data_mask)))
                throw std::invalid_argument("invalid CSS record action");
        }
        {
            const auto count = reader.size();
            if (count > 2)
                throw std::invalid_argument("too many cat verification groups");
            verification_rules_.resize(count);
            for (auto& rule : verification_rules_) {
                if (reader.size() >= 2)
                    throw std::invalid_argument("invalid verification branch bit");
                rule.resize(reader.size());
                for (auto& pair : rule) {
                    pair = {reader.size(), reader.size()};
                    if (pair[0] >= sites_.size() || pair[1] >= sites_[pair[0]].targets.size())
                        throw std::invalid_argument("invalid verification fault dependency");
                }
            }
        }
        for (const auto* block : {&body_, &final_})
            for (const auto& e : block->events)
                if (e.kind == 10 && e.c >= verification_rules_.size())
                    throw std::invalid_argument("invalid verification event");
    }
    void certify(std::span<const ActiveExpectation> probes) const;
    std::vector<RecordSlot> output_records() const;
};
class Workspace {
    const RegionPlan& plan_;
    Kernel kernel_;
    Case inputs_;
    std::vector<uint8_t> verification_values_;
    std::span<uint8_t> records;
    std::vector<size_t> fault_choices;
    clifft::Xoshiro256PlusPlus* noise_rng_ = nullptr;
    std::array<Complex, 8> roots_;
    std::vector<Complex> local_;
    std::array<std::vector<uint8_t>, 4> choice_records_;
    std::array<Mask, 4> choice_offsets_{};
    std::array<uint64_t, 4> choice_ancillas_{};
    std::vector<uint8_t> record_flips_;
    Mask post_x_ = 0, post_z_ = 0;
    uint64_t post_ancillas_ = 0;
    void bind(const Block& block, const Boundary& incoming) noexcept {
        auto& inputs = inputs_;
        for (auto& choice : inputs)
            for (auto& terms : choice)
                for (auto& term : terms) {
                    term.coefficient = 0;
                    std::fill(term.local.begin(), term.local.end(), Complex(1));
                }
        const size_t count = block.shifts.size();
        for (size_t outcome = 0; outcome < count; ++outcome) {
            std::fill(choice_records_[outcome].begin(), choice_records_[outcome].end(), 0);
            for (size_t branch = 0; branch < count; ++branch) {
                uint64_t bits = incoming.ancillas;
                Mask flips = 0;
                Complex scalar = 1;
                std::fill(local_.begin(), local_.end(), Complex(1));
                const auto phase = [&](size_t q, unsigned exponent) noexcept {
                    if (q < plan_.n_) {
                        const bool flip = uint64_t(flips >> q) & 1;
                        local_[4 * q] *= roots_[(exponent * flip) & 7];
                        local_[4 * q + 1] *= roots_[(exponent * (!flip)) & 7];
                    } else {
                        scalar *= roots_[(exponent * ((bits >> (q - plan_.n_)) & 1)) & 7];
                    }
                };
                const auto apply_pauli = [&](size_t q, size_t axis) noexcept {
                    if (axis == 2 || axis == 3)
                        phase(q, 4);
                    if (axis == 2)
                        scalar *= Complex(0, 1);
                    if (axis == 1 || axis == 2) {
                        if (q < plan_.n_)
                            flips ^= Mask{1} << q;
                        else
                            bits ^= uint64_t{1} << (q - plan_.n_);
                    }
                };
                for (const auto& event : block.events) {
                    const auto q = event.a;
                    switch (event.kind) {
                        case 0:
                        case 3: {
                            const auto value = (bits >> (q - plan_.n_)) & 1;
                            if (branch == 0)
                                choice_records_[outcome][event.b] = value;
                            else
                                assert(choice_records_[outcome][event.b] == value);
                            if (event.kind == 0)
                                bits &= ~(uint64_t{1} << (q - plan_.n_));
                            break;
                        }
                        case 1:
                        case 2: {
                            const bool value =
                                ((event.kind == 1 ? branch : outcome) >> event.b) & 1;
                            scalar *= (((bits >> (q - plan_.n_)) & value) ? -1 : 1) *
                                      0x1.6a09e667f3bcdp-1;
                            bits = (bits & ~(uint64_t{1} << (q - plan_.n_))) |
                                   (uint64_t(value) << (q - plan_.n_));
                            break;
                        }
                        case 10: {
                            // Condition on the fair verification readout. Its
                            // remaining branches are kept coherent.
                            const bool value = ((branch >> event.b) & 1) ^
                                               verification_values_[size_t(uint64_t(event.c))];
                            if ((bits >> (q - plan_.n_)) & value)
                                scalar *= -1;
                            bits = (bits & ~(uint64_t{1} << (q - plan_.n_))) |
                                   (uint64_t(value) << (q - plan_.n_));
                            break;
                        }
                        case 4:
                            phase(q, 1);
                            break;
                        case 5:
                            phase(q, 7);
                            break;
                        case 6:
                            if ((bits >> (q - plan_.n_)) & 1)
                                bits ^= uint64_t{1} << (event.b - plan_.n_);
                            break;
                        case 7:
                            if ((bits >> (q - plan_.n_)) & 1)
                                flips ^= Mask{1} << event.b;
                            break;
                        case 8:
                            if ((bits >> (q - plan_.n_)) & 1) {
                                const auto pair = plan_.pairs_[event.b];
                                const auto shift = (uint64_t(flips >> pair[0]) & 1) |
                                                   ((uint64_t(flips >> pair[1]) & 1) << 1);
                                local_[4 * (plan_.n_ + event.b) + (3 ^ shift)] *= -1;
                            }
                            break;
                        case 9: {
                            const auto& site = plan_.sites_[q];
                            const auto selected = fault_choices[q];
                            if (selected)
                                for (size_t j = 0; j < site.targets.size(); ++j)
                                    apply_pauli(site.targets[j], site.channels[selected - 1][j]);
                            break;
                        }
                        default:
                            assert(false);
                    }
                }
                if (branch == 0) {
                    choice_offsets_[outcome] = incoming.offset ^ flips;
                    choice_ancillas_[outcome] = bits;
                } else
                    assert(choice_ancillas_[outcome] == bits);
                const auto shift = block.shifts[branch];
                for (size_t l = 0; l < 2; ++l) {
                    const auto base = choice_offsets_[outcome] ^ flips ^ (l ? plan_.logical_x_ : 0);
                    auto& term = inputs[outcome][l][branch];
                    term.coefficient =
                        scalar * incoming.logical[l ^ (shift >> plan_.rank_) ^
                                                  parity(incoming.offset & plan_.logical_z_)];
                    if (parity(incoming.syndrome & shift))
                        term.coefficient *= -1;
                    for (size_t q = 0; q < plan_.n_; ++q) {
                        const size_t bit = uint64_t(base >> q) & 1;
                        term.local[4 * q] = local_[4 * q + bit];
                        term.local[4 * q + 1] = local_[4 * q + (1 ^ bit)];
                    }
                    for (size_t i = 0; i < plan_.pairs_.size(); ++i) {
                        const auto pair = plan_.pairs_[i];
                        const auto flip = (uint64_t(base >> pair[0]) & 1) |
                                          ((uint64_t(base >> pair[1]) & 1) << 1);
                        for (size_t value = 0; value < 4; ++value)
                            term.local[4 * (plan_.n_ + i) + value] =
                                local_[4 * (plan_.n_ + i) + (value ^ flip)];
                    }
                    for (size_t i = 0; i < plan_.rank_; ++i)
                        term.local[4 * (plan_.characters_ + i) + 1] =
                            ((incoming.syndrome >> i) & 1) ? -1 : 1;
                }
            }
        }
    }
    void sample_body(Boundary& boundary) noexcept {
        for (size_t i = 0; i < plan_.verification_rules_.size(); ++i) {
            bool value = ((*noise_rng_)() >> 63) != 0;
            for (auto pair : plan_.verification_rules_[i]) {
                const auto channel = fault_choices[pair[0]];
                if (channel) {
                    const auto axis = plan_.sites_[pair[0]].channels[channel - 1][pair[1]];
                    value ^= axis == 1 || axis == 2;
                }
            }
            verification_values_[i] = value;
        }
        bind(plan_.body_, boundary);
        kernel_.sample_roots(inputs_);
        const auto choice = kernel_.root_outcomes;
        for (const auto& event : plan_.body_.events)
            if (event.kind == 0 || event.kind == 3)
                records[event.b] = choice_records_[choice][event.b];
        kernel_.finish(inputs_[choice]);
        boundary.offset = choice_offsets_[choice];
        boundary.ancillas = choice_ancillas_[choice];
        boundary.syndrome = kernel_.syndrome;
        boundary.logical = kernel_.logical;
        if (parity(boundary.offset & plan_.logical_z_))
            std::swap(boundary.logical[0], boundary.logical[1]);
    }

    void finish(Boundary& boundary) noexcept {
        for (const auto& event : plan_.post_) {
            bool value;
            if (event.kind == 2) {
                value = (boundary.ancillas >> event.a) & 1;
                boundary.ancillas &= ~(uint64_t{1} << event.a);
            } else {
                value = event.kind == 0 ? ((boundary.syndrome >> size_t(uint64_t(event.c))) & 1)
                                        : parity(boundary.offset & event.c);
                boundary.ancillas =
                    (boundary.ancillas & ~(uint64_t{1} << event.a)) | (uint64_t(value) << event.a);
            }
            records[event.b] = value ^ record_flips_[event.b];
        }
        boundary.offset ^= post_x_;
        boundary.ancillas ^= post_ancillas_;
        for (size_t i = 0; i < plan_.rank_; ++i)
            boundary.syndrome ^= uint64_t(parity(post_z_ & plan_.x_rows_[i])) << i;
        if (parity(post_x_ & plan_.logical_z_))
            std::swap(boundary.logical[0], boundary.logical[1]);
        if (parity(post_z_ & plan_.logical_x_))
            boundary.logical[1] *= -1;
        bind(plan_.final_, boundary);
        const double zero = kernel_.probability(inputs_[0]);
        const double one = kernel_.probability(inputs_[1]);
        assert(std::abs(zero + one - 1) < 1e-8);
        const bool ending = kernel_.sample_binary(zero, zero + one);
        for (const auto& event : plan_.final_.events)
            if (event.kind == 0 || event.kind == 3)
                records[event.b] = choice_records_[ending][event.b];
    }

  public:
    explicit Workspace(const RegionPlan& plan);
    void run(std::span<const double> probes, std::span<const uint8_t> symbols,
             const std::vector<std::vector<uint32_t>>& faults, std::span<uint8_t> output,
             clifft::Xoshiro256PlusPlus& rng) noexcept;
};

}  // namespace clifft::sampling::folded
