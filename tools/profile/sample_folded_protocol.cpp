// Complete reconstructed folded-MSC attempts using a Clifft prefix and fixed contractions.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"

#include "folded_sampling_kernel.h"
#include "study_schedule.h"

#include <bit>
#include <memory>
#include <numbers>
#include <span>
#include <sstream>
#include <sys/resource.h>

using namespace clifft;
using namespace clifft::sampling;
using namespace folded_study;

namespace {
std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::invalid_argument("cannot read folded protocol input");
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

SamplingPlan compile(const std::string& text, const std::string& schedule, size_t detectors,
                     bool early) {
    auto hir = trace(parse(text));
    auto passes = default_hir_pass_manager();
    passes.run(hir);
    apply_study_schedule(hir, schedule);
    std::vector<uint8_t> mask(early ? detectors : 0, 1);
    SamplingPlanOptions options;
    options.postselection_mask = mask;
    return plan_sampling(hir, options);
}

__extension__ typedef unsigned __int128 Mask;

bool parity(Mask value) noexcept {
    return (std::popcount(uint64_t(value)) + std::popcount(uint64_t(value >> 64))) & 1;
}

Mask read_mask(Reader& reader) {
    std::string token;
    if (!(reader.input >> token) || token.empty())
        throw std::invalid_argument("missing physical mask");
    Mask value = 0;
    for (char c : token) {
        if (c < '0' || c > '9' || value > (~Mask{0} - unsigned(c - '0')) / 10)
            throw std::invalid_argument("invalid physical mask");
        value = value * 10 + unsigned(c - '0');
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

class GrowthBridge {
    using Packed = std::vector<uint64_t>;
    struct Site {
        double probability;
        std::vector<Packed> effects;
    };
    std::vector<size_t> record_slots_;
    size_t source_width_, target_width_, ancillas_, logical_begin_, ancilla_begin_;
    std::vector<Mask> source_z_, target_duals_;
    std::vector<uint64_t> masks_;
    Packed constants_, values_;
    std::vector<Site> sites_;
    Xoshiro256PlusPlus rng_;

    static bool bit(const Packed& value, size_t i) noexcept {
        return (value[i / 64] >> (i % 64)) & 1;
    }

  public:
    std::vector<size_t> fault_choices;

    GrowthBridge(const std::filesystem::path& path, uint64_t seed, size_t visible, size_t hidden)
        : rng_(seed ^ 0x47726f777468ULL) {
        std::ifstream input(path);
        Reader reader{input};
        if (reader.size() != 2 || reader.size() != visible || reader.size() != hidden)
            throw std::invalid_argument("invalid growth dimensions");
        source_width_ = reader.size();
        target_width_ = reader.size();
        const auto source_rank = reader.size(), target_rank = reader.size();
        ancillas_ = reader.size();
        if (!source_rank || !target_rank || source_width_ != 2 * source_rank + 1 ||
            target_width_ != 2 * target_rank + 1 || source_width_ > 127 || target_width_ > 127 ||
            !ancillas_ || 2 * source_rank + ancillas_ > 64)
            throw std::invalid_argument("unsupported growth boundary dimensions");
        source_z_.resize(source_rank);
        target_duals_.resize(target_rank);
        record_slots_.resize(reader.size());
        logical_begin_ = record_slots_.size() + 2 * target_rank;
        ancilla_begin_ = logical_begin_ + 3;
        masks_.resize(ancilla_begin_ + ancillas_);
        constants_.resize((masks_.size() + 63) / 64);
        values_.resize(constants_.size());
        std::vector<bool> seen(visible + hidden);
        for (auto& slot : record_slots_) {
            slot = reader.size();
            if (slot >= visible + hidden || seen[slot])
                throw std::invalid_argument("invalid growth record slot");
            seen[slot] = true;
        }
        for (auto& row : source_z_) {
            row = read_mask(reader);
            if (row >> source_width_)
                throw std::invalid_argument("invalid growth source check");
        }
        for (auto& row : target_duals_) {
            row = read_mask(reader);
            if (row >> target_width_)
                throw std::invalid_argument("invalid growth output dual");
        }
        if (reader.size() != masks_.size())
            throw std::invalid_argument("invalid growth relation count");
        for (size_t i = 0; i < masks_.size(); ++i) {
            masks_[i] = reader.word();
            const auto sign = reader.size();
            if ((2 * source_rank + ancillas_ < 64 && masks_[i] >> (2 * source_rank + ancillas_)) ||
                sign > 1)
                throw std::invalid_argument("invalid growth relation");
            constants_[i / 64] |= uint64_t(sign) << (i % 64);
        }
        const auto logical_frame = [&](const Packed& value) {
            return bit(value, logical_begin_ + 1) ==
                   (bit(value, logical_begin_) ^ bit(value, logical_begin_ + 2));
        };
        if (masks_[logical_begin_ + 1] != (masks_[logical_begin_] ^ masks_[logical_begin_ + 2]) ||
            !logical_frame(constants_))
            throw std::invalid_argument("growth logical relation is not a Pauli frame");
        sites_.resize(reader.size());
        for (auto& site : sites_) {
            site.probability = reader.number();
            site.effects.resize(reader.size());
            if (site.probability < 0 || site.probability > 1 || site.effects.empty() ||
                site.effects.size() > 15)
                throw std::invalid_argument("invalid growth noise site");
            for (auto& effect : site.effects) {
                effect.resize(constants_.size());
                for (auto& word : effect)
                    word = reader.word();
                if ((masks_.size() % 64 && effect.back() >> (masks_.size() % 64)) ||
                    !logical_frame(effect))
                    throw std::invalid_argument("invalid growth noise effect");
            }
        }
        fault_choices.resize(sites_.size());
    }

    void validate_boundaries(size_t source_width, size_t target_width, size_t source_ancillas,
                             size_t target_ancillas) const {
        if (source_width != source_width_ || target_width != target_width_ ||
            source_ancillas > ancillas_ || target_ancillas != ancillas_)
            throw std::invalid_argument("growth and folded block boundaries disagree");
    }

    void clear() noexcept { std::fill(fault_choices.begin(), fault_choices.end(), 0); }

    void run(Boundary& boundary, std::vector<uint8_t>& records) noexcept {
        const auto source_rank = source_z_.size(), target_rank = target_duals_.size();
        assert(!(boundary.syndrome >> source_rank) && !(boundary.ancillas >> ancillas_) &&
               !(boundary.offset >> source_width_));
        uint64_t inputs = boundary.syndrome | (boundary.ancillas << (2 * source_rank));
        for (size_t i = 0; i < source_z_.size(); ++i)
            inputs |= uint64_t(parity(boundary.offset & source_z_[i])) << (source_rank + i);
        std::copy(constants_.begin(), constants_.end(), values_.begin());
        auto& values = values_;
        for (size_t i = 0; i < masks_.size(); ++i)
            values[i / 64] ^= uint64_t(std::popcount(inputs & masks_[i]) & 1) << (i % 64);
        for (size_t i = 0; i < sites_.size(); ++i) {
            const auto& site = sites_[i];
            const double draw = (rng_() >> 11) * 0x1.0p-53;
            fault_choices[i] = 0;
            if (draw >= site.probability)
                continue;
            const size_t channel = std::min(size_t(draw / site.probability * site.effects.size()),
                                            site.effects.size() - 1);
            fault_choices[i] = channel + 1;
            for (size_t j = 0; j < values.size(); ++j)
                values[j] ^= site.effects[channel][j];
        }
        for (size_t i = 0; i < record_slots_.size(); ++i)
            records[record_slots_[i]] = bit(values, i);
        const bool flip_x = bit(values, logical_begin_), flip_z = bit(values, logical_begin_ + 2);
        assert(bit(values, logical_begin_ + 1) == (flip_x ^ flip_z));
        if (flip_z)
            std::swap(boundary.logical[0], boundary.logical[1]);
        if (flip_x)
            boundary.logical[1] *= -1;
        boundary.syndrome = boundary.ancillas = 0;
        boundary.offset = 0;
        for (size_t i = 0; i < target_rank; ++i) {
            boundary.syndrome |= uint64_t(bit(values, record_slots_.size() + i)) << i;
            if (bit(values, record_slots_.size() + target_rank + i))
                boundary.offset ^= target_duals_[i];
        }
        for (size_t i = 0; i < ancillas_; ++i)
            boundary.ancillas |= uint64_t(bit(values, ancilla_begin_ + i)) << i;
    }
};

class FoldedBlock {
    bool terminal_;
    Case inputs_;
    size_t n_, ancilla_count_, visible_, hidden_, prefix_visible_, prefix_hidden_,
        prefix_detectors_;
    size_t rank_, leaves_, characters_, logical_slot_;
    Mask logical_x_, logical_z_;
    std::vector<Mask> x_rows_, z_rows_, z_duals_;
    std::vector<std::array<size_t, 2>> pairs_;
    std::vector<Noise> sites_;
    Block body_, final_;
    std::vector<Event> post_;
    std::vector<size_t> postselect_, detector_partners_;
    std::vector<std::vector<std::array<size_t, 2>>> verification_rules_;
    std::vector<uint8_t> verification_values_;
    FoldedSampler kernel_;
    Xoshiro256PlusPlus noise_rng_;
    std::array<Complex, 8> roots_;
    std::vector<Complex> local_;
    std::array<std::vector<uint8_t>, 4> choice_records_;
    std::array<Mask, 4> choice_offsets_{};
    std::array<uint64_t, 4> choice_ancillas_{};
    std::vector<uint8_t> record_flips_;
    Mask post_x_ = 0, post_z_ = 0;
    uint64_t post_ancillas_ = 0;

  public:
    void certify(const SamplingPlan& plan) {
        if (plan.num_visible_records != prefix_visible_ ||
            plan.num_hidden_records != prefix_hidden_)
            throw std::invalid_argument("prefix record dimensions disagree");
        std::vector<const WriteExpectationValue*> probes(plan.num_exp_vals, nullptr);
        for (const auto& action : plan.actions)
            if (const auto* probe = std::get_if<WriteExpectationValue>(&action.action)) {
                if (action.active_before != 1)
                    throw std::invalid_argument(
                        "prefix must have one active coordinate at handoff");
                probes.at(index(probe->exp_val)) = probe;
            }
        if (probes.size() != 2 * rank_ + 3 + ancilla_count_ ||
            std::ranges::any_of(probes, [](auto p) { return !p; }))
            throw std::invalid_argument("incorrect signed-CSS probe set");
        unsigned axes = 0;
        for (size_t i = 0; i < probes.size(); ++i) {
            if (!probes[i]->active)
                throw std::invalid_argument("uncertified prefix probe");
            const auto p = probes[i]->active->projection;
            if (i < 2 * rank_ || i >= 2 * rank_ + 3) {
                if (!p.is_identity())
                    throw std::invalid_argument(
                        "prefix is not in one signed CSS code and Z ancillas");
            } else {
                const auto label = p.x | (p.z << 1);
                if (!label || label > 3 || (axes & (1 << label)))
                    throw std::invalid_argument("logical probes do not span one qubit");
                axes |= 1 << label;
            }
        }
    }

  private:
    Boundary handoff(const Executor& prefix) noexcept {
        Boundary result;
        const auto values = prefix.exp_vals();
        for (size_t i = 0; i < rank_; ++i) {
            result.syndrome |= uint64_t(values[i] < 0) << i;
            if (values[rank_ + i] < 0)
                result.offset ^= z_duals_[i];
        }
        for (size_t i = 0; i < ancilla_count_; ++i)
            result.ancillas |= uint64_t(values[2 * rank_ + 3 + i] < 0) << i;
        const double x = values[2 * rank_], y = values[2 * rank_ + 1];
        const double z = std::clamp(values[2 * rank_ + 2], -1.0, 1.0);
        if (z >= 0) {
            result.logical[0] = std::sqrt((1 + z) / 2);
            result.logical[1] = Complex(x, y) / (2 * result.logical[0].real());
        } else {
            result.logical[1] = std::sqrt((1 - z) / 2);
            result.logical[0] = Complex(x, -y) / (2 * result.logical[1].real());
        }
        return result;
    }

    void draw_faults() noexcept {
        std::fill(record_flips_.begin(), record_flips_.end(), 0);
        post_x_ = post_z_ = post_ancillas_ = 0;
        for (size_t i = 0; i < sites_.size(); ++i) {
            const auto& site = sites_[i];
            fault_choices[i] = 0;
            const double draw = (noise_rng_() >> 11) * 0x1.0p-53;
            if (draw >= site.probability)
                continue;
            const size_t channel = std::min(size_t(draw / site.probability * site.channels.size()),
                                            site.channels.size() - 1);
            fault_choices[i] = channel + 1;
            if (!site.effects.empty()) {
                const auto& effect = site.effects[channel];
                post_x_ ^= effect.x;
                post_z_ ^= effect.z;
                post_ancillas_ ^= effect.ancillas;
                for (auto slot : effect.records)
                    record_flips_[slot] ^= 1;
            }
        }
    }

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
                    if (q < n_) {
                        const bool flip = (flips >> q) & 1;
                        local_[4 * q] *= roots_[(exponent * flip) & 7];
                        local_[4 * q + 1] *= roots_[(exponent * (!flip)) & 7];
                    } else {
                        scalar *= roots_[(exponent * ((bits >> (q - n_)) & 1)) & 7];
                    }
                };
                const auto apply_pauli = [&](size_t q, size_t axis) noexcept {
                    if (axis == 2 || axis == 3)
                        phase(q, 4);
                    if (axis == 2)
                        scalar *= Complex(0, 1);
                    if (axis == 1 || axis == 2) {
                        if (q < n_)
                            flips ^= Mask{1} << q;
                        else
                            bits ^= uint64_t{1} << (q - n_);
                    }
                };
                for (const auto& event : block.events) {
                    const auto q = event.a;
                    switch (event.kind) {
                        case 0:
                        case 3: {
                            const auto value = (bits >> (q - n_)) & 1;
                            if (branch == 0)
                                choice_records_[outcome][event.b] = value;
                            else
                                assert(choice_records_[outcome][event.b] == value);
                            if (event.kind == 0)
                                bits &= ~(uint64_t{1} << (q - n_));
                            break;
                        }
                        case 1:
                        case 2: {
                            const bool value =
                                ((event.kind == 1 ? branch : outcome) >> event.b) & 1;
                            scalar *=
                                (((bits >> (q - n_)) & value) ? -1 : 1) * 0x1.6a09e667f3bcdp-1;
                            bits =
                                (bits & ~(uint64_t{1} << (q - n_))) | (uint64_t(value) << (q - n_));
                            break;
                        }
                        case 10: {
                            // Condition on the fair verification readout. Its
                            // probability is included once in the shot log.
                            const bool value =
                                ((branch >> event.b) & 1) ^ verification_values_[size_t(event.c)];
                            if ((bits >> (q - n_)) & value)
                                scalar *= -1;
                            bits =
                                (bits & ~(uint64_t{1} << (q - n_))) | (uint64_t(value) << (q - n_));
                            break;
                        }
                        case 4:
                            phase(q, 1);
                            break;
                        case 5:
                            phase(q, 7);
                            break;
                        case 6:
                            if ((bits >> (q - n_)) & 1)
                                bits ^= uint64_t{1} << (event.b - n_);
                            break;
                        case 7:
                            if ((bits >> (q - n_)) & 1)
                                flips ^= Mask{1} << event.b;
                            break;
                        case 8:
                            if ((bits >> (q - n_)) & 1) {
                                const auto pair = pairs_[event.b];
                                const auto shift =
                                    ((flips >> pair[0]) & 1) | (((flips >> pair[1]) & 1) << 1);
                                local_[4 * (n_ + event.b) + (3 ^ shift)] *= -1;
                            }
                            break;
                        case 9: {
                            const auto& site = sites_[q];
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
                    const auto base = choice_offsets_[outcome] ^ flips ^ (l ? logical_x_ : 0);
                    auto& term = inputs[outcome][l][branch];
                    term.coefficient =
                        scalar *
                        incoming
                            .logical[l ^ (shift >> rank_) ^ parity(incoming.offset & logical_z_)];
                    if (parity(incoming.syndrome & shift))
                        term.coefficient *= -1;
                    for (size_t q = 0; q < n_; ++q) {
                        const size_t bit = (base >> q) & 1;
                        term.local[4 * q] = local_[4 * q + bit];
                        term.local[4 * q + 1] = local_[4 * q + (1 ^ bit)];
                    }
                    for (size_t i = 0; i < pairs_.size(); ++i) {
                        const auto pair = pairs_[i];
                        const auto flip = ((base >> pair[0]) & 1) | (((base >> pair[1]) & 1) << 1);
                        for (size_t value = 0; value < 4; ++value)
                            term.local[4 * (n_ + i) + value] =
                                local_[4 * (n_ + i) + (value ^ flip)];
                    }
                    for (size_t i = 0; i < rank_; ++i)
                        term.local[4 * (characters_ + i) + 1] =
                            ((incoming.syndrome >> i) & 1) ? -1 : 1;
                }
            }
        }
    }

  public:
    bool output_bits() noexcept {
        bool accepted = true;
        for (size_t i = 0; i < postselect_.size(); ++i) {
            detectors[i] = records[postselect_[i]] ^
                           (detector_partners_[i] < visible_ ? records[detector_partners_[i]] : 0);
            accepted &= !detectors[i];
        }
        observable = terminal_ ? records[logical_slot_] : 0;
        return accepted;
    }

  public:
    std::vector<uint8_t> records, detectors;
    std::vector<size_t> fault_choices;
    uint8_t observable = 0;
    bool accepted = false;
    double conditional_log_probability = 0;

    FoldedBlock(const std::filesystem::path& directory, uint64_t seed, size_t rank, size_t leaves,
                size_t characters)
        : rank_(rank),
          leaves_(leaves),
          characters_(characters),
          kernel_(directory, seed ^ 0x476164676574ULL, rank, leaves, characters),
          noise_rng_(seed ^ 0x4e6f697365ULL) {
        std::ifstream input(directory / "block.txt");
        Reader reader{input};
        if (reader.size() != 1)
            throw std::invalid_argument("unsupported folded block version");
        const auto terminal = reader.size();
        if (terminal > 1)
            throw std::invalid_argument("invalid folded continuation kind");
        terminal_ = terminal;
        n_ = reader.size();
        ancilla_count_ = reader.size();
        visible_ = reader.size();
        hidden_ = reader.size();
        prefix_visible_ = reader.size();
        prefix_hidden_ = reader.size();
        prefix_detectors_ = reader.size();
        logical_x_ = read_mask(reader);
        logical_z_ = read_mask(reader);
        if (!n_ || n_ > 127 || !ancilla_count_ || ancilla_count_ > 63 || n_ != 2 * rank_ + 1 ||
            prefix_visible_ > visible_ || prefix_hidden_ > hidden_)
            throw std::invalid_argument("invalid protocol dimensions");
        const auto data_mask = (Mask{1} << n_) - 1;
        for (auto* rows : {&x_rows_, &z_rows_, &z_duals_}) {
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
            if (block == &final_ && !terminal_)
                break;
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
        postselect_.resize(reader.size());
        for (auto& slot : postselect_) {
            slot = reader.size();
            if (slot >= visible_)
                throw std::invalid_argument("invalid acceptance slot");
        }
        logical_slot_ = terminal_ ? reader.size() : 0;
        if (terminal_ && logical_slot_ >= visible_)
            throw std::invalid_argument("invalid logical slot");
        detector_partners_.resize(postselect_.size(), visible_);
        {
            const auto count = reader.size();
            if (count > 2)
                throw std::invalid_argument("too many cat verification groups");
            verification_rules_.resize(count);
            verification_values_.resize(count);
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
            for (size_t i = 0; i < detector_partners_.size(); ++i) {
                detector_partners_[i] = reader.size();
                if (detector_partners_[i] > visible_ || detector_partners_[i] == postselect_[i])
                    throw std::invalid_argument("invalid detector parity partner");
            }
        }
        for (const auto* block : {&body_, &final_})
            for (const auto& e : block->events)
                if (e.kind == 10 && e.c >= verification_rules_.size())
                    throw std::invalid_argument("invalid verification event");
        records.resize(visible_ + hidden_);
        detectors.resize(postselect_.size());
        fault_choices.resize(sites_.size());
        record_flips_.resize(records.size());
        for (auto& record : choice_records_)
            record.resize(records.size());
        local_.resize(4 * characters_);
        for (size_t i = 0; i < roots_.size(); ++i)
            roots_[i] = std::polar(1.0, std::numbers::pi * double(i) / 4);
        if (!terminal_ && !post_.empty())
            throw std::invalid_argument("intermediate block has a terminal continuation");
        for (auto& choice : inputs_)
            for (auto& terms : choice)
                for (auto& term : terms)
                    term.local.resize(4 * leaves_);
    }

    size_t width() const noexcept { return n_; }
    size_t ancillas() const noexcept { return ancilla_count_; }
    bool terminal() const noexcept { return terminal_; }
    size_t prefix_detectors() const noexcept { return prefix_detectors_; }
    size_t visible() const noexcept { return visible_; }
    size_t hidden() const noexcept { return hidden_; }
    double normalization_error() const noexcept { return kernel_.max_normalization_error; }
    double continuation_error() const noexcept { return kernel_.max_continuation_error; }
    size_t lookup_bytes() const noexcept { return kernel_.lookup_bytes(); }
    size_t workspace_bytes() const noexcept { return kernel_.workspace_bytes(); }

    void reset() noexcept {
        accepted = false;
        observable = 0;
        conditional_log_probability = 0;
        std::fill(records.begin(), records.end(), 0);
        std::fill(detectors.begin(), detectors.end(), 0);
        std::fill(fault_choices.begin(), fault_choices.end(), 0);
    }

    bool run_body_from_prefix(Executor& prefix, bool early, Boundary& boundary) noexcept {
        reset();
        prefix.run_shot();
        std::copy(prefix.visible_records().begin(), prefix.visible_records().end(),
                  records.begin());
        std::copy(prefix.hidden_records().begin(), prefix.hidden_records().end(),
                  records.begin() + visible_);
        if (prefix.discarded())
            return false;
        boundary = handoff(prefix);
        return sample_body(boundary, early);
    }

    bool sample_body(Boundary& boundary, bool early) noexcept {
        draw_faults();
        for (size_t i = 0; i < verification_rules_.size(); ++i) {
            bool value = (noise_rng_() >> 63) != 0;
            for (auto pair : verification_rules_[i]) {
                const auto channel = fault_choices[pair[0]];
                if (channel) {
                    const auto axis = sites_[pair[0]].channels[channel - 1][pair[1]];
                    value ^= axis == 1 || axis == 2;
                }
            }
            verification_values_[i] = value;
        }
        bind(body_, boundary);
        kernel_.sample_roots(inputs_);
        const auto choice = kernel_.root_outcomes;
        for (const auto& event : body_.events)
            if (event.kind == 0 || event.kind == 3)
                records[event.b] = choice_records_[choice][event.b];
        if (early && !output_bits())
            return false;
        kernel_.finish(inputs_[choice]);
        conditional_log_probability +=
            kernel_.log_probability - verification_rules_.size() * std::log(2.0);
        boundary.offset = choice_offsets_[choice];
        boundary.ancillas = choice_ancillas_[choice];
        boundary.syndrome = kernel_.syndrome;
        boundary.logical = kernel_.logical;
        if (parity(boundary.offset & logical_z_))
            std::swap(boundary.logical[0], boundary.logical[1]);
        return true;
    }

    void finish(Boundary& boundary, bool early) noexcept {
        assert(terminal_);
        for (const auto& event : post_) {
            bool value;
            if (event.kind == 2) {
                value = (boundary.ancillas >> event.a) & 1;
                boundary.ancillas &= ~(uint64_t{1} << event.a);
            } else {
                value = event.kind == 0 ? ((boundary.syndrome >> event.c) & 1)
                                        : parity(boundary.offset & event.c);
                boundary.ancillas =
                    (boundary.ancillas & ~(uint64_t{1} << event.a)) | (uint64_t(value) << event.a);
            }
            records[event.b] = value ^ record_flips_[event.b];
        }
        boundary.offset ^= post_x_;
        boundary.ancillas ^= post_ancillas_;
        for (size_t i = 0; i < rank_; ++i)
            boundary.syndrome ^= uint64_t(parity(post_z_ & x_rows_[i])) << i;
        if (parity(post_x_ & logical_z_))
            std::swap(boundary.logical[0], boundary.logical[1]);
        if (parity(post_z_ & logical_x_))
            boundary.logical[1] *= -1;
        if (early && !output_bits())
            return;
        bind(final_, boundary);
        const double zero = kernel_.probability(inputs_[0]);
        const double one = kernel_.probability(inputs_[1]);
        kernel_.max_normalization_error =
            std::max(kernel_.max_normalization_error, std::abs(zero + one - 1));
        assert(std::abs(zero + one - 1) < 1e-8);
        const bool ending = kernel_.sample_binary(zero, zero + one);
        for (const auto& event : final_.events)
            if (event.kind == 0 || event.kind == 3)
                records[event.b] = choice_records_[ending][event.b];
        conditional_log_probability += std::log(ending ? one : zero);
        accepted = output_bits();
    }
};

class Protocol {
    static std::unique_ptr<FoldedBlock> read_block(const std::filesystem::path& path,
                                                   uint64_t seed) {
        std::ifstream dimensions(path / "dimensions.txt");
        Reader sizes{dimensions};
        const auto rank = sizes.size(), leaves = sizes.size(), characters = sizes.size();
        return std::make_unique<FoldedBlock>(path, seed, rank, leaves, characters);
    }
    std::unique_ptr<FoldedBlock> last_, first_;
    std::unique_ptr<GrowthBridge> growth_;

    const FoldedBlock& prefix_block() const noexcept { return first_ ? *first_ : *last_; }

  public:
    Protocol(const std::filesystem::path& directory, uint64_t seed) {
        const auto sequence = read_text(directory / "sequence.txt");
        const bool growth = sequence == "1\nfolded f5\ngrowth growth.txt\nfolded .\n";
        if (!growth && sequence != "1\nfolded .\n")
            throw std::invalid_argument("unsupported folded protocol sequence");
        last_ = read_block(directory, seed);
        if (!last_->terminal())
            throw std::invalid_argument("protocol must end with a terminal block");
        if (growth) {
            first_ = read_block(directory / "f5", seed ^ 0x4635ULL);
            if (first_->terminal() || first_->visible() > last_->visible() ||
                first_->hidden() > last_->hidden())
                throw std::invalid_argument("invalid intermediate folded block");
            growth_ = std::make_unique<GrowthBridge>(directory / "growth.txt", seed,
                                                     last_->visible(), last_->hidden());
            growth_->validate_boundaries(first_->width(), last_->width(), first_->ancillas(),
                                         last_->ancillas());
        }
    }

    const FoldedBlock& result() const noexcept { return *last_; }
    size_t prefix_detectors() const noexcept { return prefix_block().prefix_detectors(); }
    size_t detectors_count() const noexcept { return last_->detectors.size(); }
    void certify_prefix(const SamplingPlan& plan) { (first_ ? first_ : last_)->certify(plan); }
    double normalization_error() const noexcept {
        return std::max(last_->normalization_error(), first_ ? first_->normalization_error() : 0);
    }
    double continuation_error() const noexcept {
        return std::max(last_->continuation_error(), first_ ? first_->continuation_error() : 0);
    }
    size_t lookup_bytes() const noexcept {
        return last_->lookup_bytes() + (first_ ? first_->lookup_bytes() : 0);
    }
    size_t workspace_bytes() const noexcept {
        return last_->workspace_bytes() + (first_ ? first_->workspace_bytes() : 0);
    }
    std::span<const size_t> earlier_faults() const noexcept {
        return first_ ? std::span<const size_t>(first_->fault_choices) : std::span<const size_t>();
    }
    std::span<const size_t> growth_faults() const noexcept {
        return growth_ ? std::span<const size_t>(growth_->fault_choices)
                       : std::span<const size_t>();
    }
    void run(Executor& prefix, bool early) noexcept {
        Boundary boundary;
        auto& last = *last_;
        if (first_) {
            last.reset();
            growth_->clear();
            const bool ready = first_->run_body_from_prefix(prefix, early, boundary);
            std::copy_n(first_->records.begin(), first_->visible(), last.records.begin());
            std::copy_n(first_->records.begin() + first_->visible(), first_->hidden(),
                        last.records.begin() + last.visible());
            if (!ready)
                return;
            last.conditional_log_probability = first_->conditional_log_probability;
            growth_->run(boundary, last.records);
            if (early && !last.output_bits())
                return;
            if (!last.sample_body(boundary, early))
                return;
        } else if (!last.run_body_from_prefix(prefix, early, boundary)) {
            return;
        }
        last.finish(boundary, early);
    }
};

void print_bits(const auto& values) {
    std::cout << '"';
    for (auto value : values)
        std::cout << unsigned(value);
    std::cout << '"';
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 8)
        throw std::invalid_argument(
            "usage: sample_folded_protocol bundle shots traces seed early schedule baseline_shots");
    const std::filesystem::path directory(argv[1]);
    const auto shots = std::stoull(argv[2]), traces = std::stoull(argv[3]),
               seed = std::stoull(argv[4]);
    const bool early = std::stoull(argv[5]);
    const std::string schedule = argv[6];
    const auto baseline_shots = std::stoull(argv[7]);
    if (!shots || (early && traces))
        throw std::invalid_argument("use full attempts for validation traces");
    const auto setup_start = std::chrono::steady_clock::now();
    Protocol sampler(directory, seed);
    const auto prefix_plan =
        compile(read_text(directory / "prefix.stim"), schedule, sampler.prefix_detectors(), early);
    sampler.certify_prefix(prefix_plan);
    ExecutablePlan prefix_executable(prefix_plan);
    Executor prefix(prefix_executable, seed);
    const double setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();
    std::cout << std::setprecision(17);
    for (size_t shot = 0; shot < traces; ++shot) {
        sampler.run(prefix, false);
        std::cout << "{\"kind\":\"sample\",\"records\":";
        print_bits(sampler.result().records);
        std::cout << ",\"detectors\":";
        print_bits(sampler.result().detectors);
        std::cout << ",\"observable\":" << unsigned(sampler.result().observable)
                  << ",\"conditional_log_probability\":"
                  << sampler.result().conditional_log_probability << ",\"prefix_faults\":[";
        bool first = true;
        for (size_t i = 0; i < prefix_plan.presampled_noise_sites.size(); ++i)
            for (size_t j = 0; j < prefix_plan.presampled_noise_sites[i].outcomes.size(); ++j)
                if (prefix.symbols()[index(
                        prefix_plan.presampled_noise_sites[i].outcomes[j].symbol)]) {
                    std::cout << (first ? "" : ",") << '[' << i << ',' << j + 1 << ']';
                    first = false;
                }
        std::cout << "],\"suffix_faults\":[";
        first = true;
        for (size_t i = 0; i < sampler.result().fault_choices.size(); ++i)
            if (sampler.result().fault_choices[i]) {
                std::cout << (first ? "" : ",") << '[' << i << ','
                          << sampler.result().fault_choices[i] << ']';
                first = false;
            }
        std::cout << ']';
        const auto extra_faults = [&](const char* name, std::span<const size_t> choices) {
            std::cout << ",\"" << name << "\":[";
            bool first_choice = true;
            for (size_t i = 0; i < choices.size(); ++i)
                if (choices[i]) {
                    std::cout << (first_choice ? "" : ",") << '[' << i << ',' << choices[i] << ']';
                    first_choice = false;
                }
            std::cout << ']';
        };
        extra_faults("earlier_faults", sampler.earlier_faults());
        extra_faults("growth_faults", sampler.growth_faults());
        std::cout << "}\n";
    }
    rusage native_usage{};
    getrusage(RUSAGE_SELF, &native_usage);
    const auto baseline_start = std::chrono::steady_clock::now();
    const auto full_plan =
        compile(read_text(directory / "full.stim"), schedule, sampler.detectors_count(), early);
    std::unique_ptr<ExecutablePlan> executable;
    std::unique_ptr<Executor> ordinary;
    if (baseline_shots && full_plan.peak_active_width > 24)
        throw std::invalid_argument("dense reference exceeds the profiling allocation limit");
    if (baseline_shots) {
        executable = std::make_unique<ExecutablePlan>(full_plan);
        ordinary = std::make_unique<Executor>(*executable, seed);
    }
    const double baseline_setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - baseline_start).count();
    for (size_t i = 0; i < 16; ++i)
        sampler.run(prefix, early);
    if (ordinary)
        ordinary->run_shot();
    std::array<double, 3> seconds{}, baseline_seconds{};
    std::array<size_t, 3> accepted{}, failures{}, baseline_accepted{}, baseline_failures{};
    for (size_t repeat = 0; repeat < 3; ++repeat) {
        auto start = std::chrono::steady_clock::now();
        for (size_t shot = 0; shot < shots; ++shot) {
            sampler.run(prefix, early);
            accepted[repeat] += sampler.result().accepted;
            failures[repeat] += sampler.result().accepted && sampler.result().observable;
        }
        seconds[repeat] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        for (size_t shot = 0; shot < baseline_shots; ++shot) {
            ordinary->run_shot();
            const bool ok = !ordinary->discarded() &&
                            std::ranges::none_of(ordinary->detectors(), [](auto b) { return b; });
            baseline_accepted[repeat] += ok;
            baseline_failures[repeat] += ok && ordinary->observables()[0];
        }
        baseline_seconds[repeat] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    std::cout << "{\"kind\":\"benchmark\",\"setup_seconds\":" << setup_seconds
              << ",\"baseline_setup_seconds\":" << baseline_setup_seconds
              << ",\"shots_per_batch\":" << shots
              << ",\"baseline_shots_per_batch\":" << baseline_shots
              << ",\"prefix_peak_width\":" << prefix_plan.peak_active_width
              << ",\"ordinary_peak_width\":" << full_plan.peak_active_width
              << ",\"native_peak_rss_kib\":" << native_usage.ru_maxrss
              << ",\"contraction_lookup_bytes\":" << sampler.lookup_bytes()
              << ",\"contraction_workspace_bytes\":" << sampler.workspace_bytes()
              << ",\"normalization_error\":" << sampler.normalization_error()
              << ",\"continuation_error\":" << sampler.continuation_error();
    const auto array = [&](const char* name, const auto& values) {
        std::cout << ",\"" << name << "\":[";
        for (size_t i = 0; i < values.size(); ++i)
            std::cout << (i ? "," : "") << values[i];
        std::cout << ']';
    };
    array("seconds", seconds);
    array("baseline_seconds", baseline_seconds);
    array("accepted", accepted);
    array("logical_failures", failures);
    array("baseline_accepted", baseline_accepted);
    array("baseline_logical_failures", baseline_failures);
    std::cout << "}\n";
}
