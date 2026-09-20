// Native experimental composition of Clifft prefix execution and terminal CSS sampling.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/xoshiro.h"

#include "gadget_contraction_kernel.h"
#include "study_schedule.h"

#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>

using namespace clifft;
using namespace clifft::sampling;
using namespace gadget_study;

namespace {
std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::invalid_argument("cannot open native gadget input");
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

SamplingPlan compile(const std::string& source, const std::string& schedule = "off") {
    auto hir = trace(parse(source));
    auto passes = default_hir_pass_manager();
    passes.run(hir);
    apply_study_schedule(hir, schedule);
    return plan_sampling(hir);
}

struct Effect {
    uint64_t qx = 0, qz = 0, wx = 0, wz = 0, spectator = 0;
    bool pre = false, middle = false;
    size_t readout = 0;
};
struct Site {
    double probability;
    std::vector<Effect> channels;
};
struct Spectator {
    size_t parity, record, probe = 0;
};
struct Term {
    Complex coefficient;
    std::vector<Complex> local;
};

class Terminal {
    size_t n_, rank_, prefix_visible_, prefix_hidden_, visible_, hidden_;
    size_t measurement_, reset_, logical_record_;
    bool parity_sign_;
    uint64_t reset_x_, reset_z_, reset_spectator_;
    std::vector<int> signs_;
    std::vector<uint64_t> x_rows_, z_rows_, duals_;
    std::vector<size_t> x_records_, z_records_;
    std::vector<Spectator> spectators_;
    std::vector<Site> sites_;
    std::vector<std::vector<size_t>> detector_rows_, observable_rows_;
    std::vector<Contraction> contractions_;
    ContractionWorkspace workspace_;
    std::array<std::array<Term, 4>, 2> terms_;
    std::array<Complex, 128> phases_;
    std::vector<Complex> paired_;
    std::vector<uint8_t> readout_;
    std::array<uint64_t, 2> offsets_, spectator_flips_;
    std::array<uint8_t, 2> resets_;
    uint64_t spectator_bits_ = 0;
    Xoshiro256PlusPlus rng_;
    size_t leaf_count_;

  public:
    std::vector<uint8_t> records, detectors, observables;
    std::vector<size_t> fault_choices;
    double conditional_log_probability = 0;
    double max_normalization_error = 0;

    Terminal(const std::filesystem::path& directory, const SamplingPlan& prefix, uint64_t seed)
        : rng_(seed ^ 0x54726d4761646765ULL) {
        std::ifstream input(directory / "model.txt");
        Reader reader{input};
        n_ = reader.size();
        rank_ = reader.size();
        const auto spectators = reader.size();
        prefix_visible_ = reader.size();
        prefix_hidden_ = reader.size();
        visible_ = reader.size();
        hidden_ = reader.size();
        if (n_ != 2 * rank_ + 1 || n_ > 63 || spectators > 63 ||
            prefix_visible_ != prefix.num_visible_records ||
            prefix_hidden_ != prefix.num_hidden_records || hidden_ != prefix_hidden_ + 1 ||
            prefix_visible_ >= visible_)
            throw std::invalid_argument("invalid native terminal dimensions");
        measurement_ = reader.size();
        reset_ = reader.size();
        logical_record_ = reader.size();
        parity_sign_ = reader.size();
        reset_x_ = reader.word();
        reset_z_ = reader.word();
        reset_spectator_ = reader.word();
        const uint64_t data_mask = (uint64_t{1} << n_) - 1;
        if ((reset_x_ | reset_z_) & ~data_mask || reset_spectator_ >> spectators ||
            measurement_ < prefix_visible_ || measurement_ >= visible_ ||
            logical_record_ < prefix_visible_ || logical_record_ >= visible_ ||
            reset_ != visible_ + prefix_hidden_)
            throw std::invalid_argument("invalid terminal record or reset map");
        signs_.resize(n_);
        for (auto& sign : signs_) {
            const auto bit = reader.size();
            if (bit > 1)
                throw std::invalid_argument("invalid T sign");
            sign = bit ? -1 : 1;
        }
        for (auto* rows : {&x_rows_, &z_rows_, &duals_}) {
            rows->resize(rank_);
            for (auto& row : *rows) {
                row = reader.word();
                if (row & ~data_mask)
                    throw std::invalid_argument("code row exceeds data width");
            }
        }
        for (auto* slots : {&x_records_, &z_records_}) {
            slots->resize(rank_);
            for (auto& slot : *slots) {
                slot = reader.size();
                if (slot < prefix_visible_ || slot >= visible_)
                    throw std::invalid_argument("invalid CSS record slot");
            }
        }
        spectators_.resize(spectators);
        for (auto& spectator : spectators_) {
            spectator.parity = reader.size();
            spectator.record = reader.size();
            if (spectator.parity > 3 || spectator.record > visible_ ||
                (spectator.record && spectator.record <= prefix_visible_))
                throw std::invalid_argument("invalid spectator map");
        }
        sites_.resize(reader.size());
        for (auto& site : sites_) {
            site.probability = reader.number();
            site.channels.resize(reader.size());
            if (site.probability < 0 || site.probability > 1 || site.channels.empty())
                throw std::invalid_argument("invalid terminal noise channel");
            for (auto& effect : site.channels) {
                effect.qx = reader.word();
                effect.qz = reader.word();
                effect.wx = reader.word();
                effect.wz = reader.word();
                effect.spectator = reader.word();
                const auto pre = reader.size(), middle = reader.size();
                effect.pre = pre;
                effect.middle = middle;
                effect.readout = reader.size();
                if (((effect.qx | effect.qz | effect.wx | effect.wz) & ~data_mask) ||
                    effect.spectator >> spectators || pre > 1 || middle > 1 ||
                    effect.readout > visible_ ||
                    (effect.readout && effect.readout <= prefix_visible_))
                    throw std::invalid_argument("invalid terminal fault effect");
            }
        }
        for (auto* parities : {&detector_rows_, &observable_rows_}) {
            parities->resize(reader.size());
            for (auto& row : *parities) {
                row.resize(reader.size());
                for (auto& slot : row) {
                    slot = reader.size();
                    if (slot >= visible_)
                        throw std::invalid_argument("output parity reads invalid record");
                }
            }
        }
        certify(prefix);
        static constexpr std::array<Complex, 4> roots{Complex(1), Complex(0, 1), Complex(-1),
                                                      Complex(0, -1)};
        for (size_t negative = 0; negative < 2; ++negative)
            for (size_t q = 0; q < 4; ++q)
                for (size_t w = 0; w < 4; ++w)
                    for (size_t branch = 0; branch < 2; ++branch)
                        for (size_t bit = 0; bit < 2; ++bit) {
                            const auto a = q & 1, b = q >> 1, c = w & 1, d = w >> 1;
                            const auto intermediate = bit ^ branch, output = intermediate ^ a;
                            const int exponent = (negative ? -1 : 1) * (int(bit) - int(output));
                            phases_[negative * 64 + q * 16 + w * 4 + branch * 2 + bit] =
                                std::polar(1.0, exponent * std::numbers::pi / 4) *
                                roots[(a * b + c * d) % 4] *
                                double(((b * intermediate) ^ (d * output)) ? -1 : 1);
                        }
        leaf_count_ = n_ + rank_;
        for (size_t measured = 0; measured <= rank_; ++measured) {
            std::ifstream plan_file(directory / ("marginal_" + std::to_string(measured) + ".txt"));
            Reader plan_reader{plan_file};
            contractions_.emplace_back(plan_reader, rank_ + measured, 2 * leaf_count_);
        }
        for (auto& choice : terms_)
            for (auto& term : choice)
                term.local.resize(2 * leaf_count_);
        size_t scratch_size = 0, product_size = 0;
        for (const auto& plan : contractions_) {
            scratch_size = std::max(scratch_size, plan.scratch_size());
            product_size = std::max(product_size, plan.product_size());
        }
        workspace_.prepare(scratch_size, product_size);
        paired_.resize(4 * leaf_count_);
        records.resize(visible_ + hidden_);
        readout_.resize(visible_);
        detectors.resize(detector_rows_.size());
        observables.resize(observable_rows_.size());
        fault_choices.resize(sites_.size());
    }

    void certify(const SamplingPlan& prefix) {
        std::vector<const WriteExpectationValue*> probes(prefix.num_exp_vals);
        for (const auto& action : prefix.actions) {
            if (const auto* probe = std::get_if<WriteExpectationValue>(&action.action)) {
                if (action.active_before != 1)
                    throw std::invalid_argument("terminal boundary must have active width one");
                probes.at(index(probe->exp_val)) = probe;
            }
        }
        if (probes.size() != 2 * rank_ + 3 + 2 * spectators_.size() ||
            std::ranges::any_of(probes, [](auto p) { return !p; }))
            throw std::invalid_argument("incorrect compiler handoff probe set");
        const auto fixed = [&](size_t i) {
            return probes[i]->active && probes[i]->active->projection.is_identity();
        };
        for (size_t i = 0; i < 2 * rank_; ++i)
            if (!fixed(i))
                throw std::invalid_argument("compiler cannot certify the signed CSS code space");
        uint8_t seen = 0;
        for (size_t i = 2 * rank_; i < 2 * rank_ + 3; ++i) {
            if (!probes[i]->active)
                throw std::invalid_argument("logical handoff is not a one-qubit Pauli map");
            const auto p = probes[i]->active->projection;
            const auto label = p.x | (p.z << 1);
            if (!label || label > 3 || (seen & (1 << label)))
                throw std::invalid_argument("logical handoff does not span one qubit");
            seen |= 1 << label;
        }
        for (size_t i = 0; i < spectators_.size(); ++i) {
            auto& spectator = spectators_[i];
            const auto x = 2 * rank_ + 3 + 2 * i, z = x + 1;
            if (fixed(x) == fixed(z))
                throw std::invalid_argument("compiler cannot certify a product spectator");
            const bool is_x = fixed(x);
            spectator.probe = is_x ? x : z;
            if ((spectator.record && !is_x) ||
                (spectator.parity && spectator.parity != (is_x ? 1 : 3)))
                throw std::invalid_argument(
                    "spectator branches or measurements are not deterministic");
        }
    }

    void prepare(const Executor& prefix) noexcept {
        const auto probes = prefix.exp_vals();
        uint64_t offset = 0, syndrome = 0;
        for (size_t i = 0; i < rank_; ++i) {
            syndrome |= uint64_t(probes[i] < 0) << i;
            if (probes[rank_ + i] < 0)
                offset ^= duals_[i];
        }
        double x = probes[2 * rank_], y = probes[2 * rank_ + 1], z = probes[2 * rank_ + 2];
        if (n_ % 4 == 3)
            y = -y;
        z = std::clamp(z, -1.0, 1.0);
        std::array<Complex, 2> logical;
        if (z >= 0) {
            logical[0] = std::sqrt((1 + z) / 2);
            logical[1] = Complex(x, y) / (2.0 * logical[0].real());
        } else {
            logical[1] = std::sqrt((1 - z) / 2);
            logical[0] = Complex(x, -y) / (2.0 * logical[1].real());
        }
        bool spectator_parity = parity_sign_;
        spectator_bits_ = 0;
        for (size_t i = 0; i < spectators_.size(); ++i) {
            const bool bit = probes[spectators_[i].probe] < 0;
            spectator_bits_ |= uint64_t(bit) << i;
            spectator_parity ^= bit && spectators_[i].parity;
        }
        Effect total;
        std::fill(readout_.begin(), readout_.end(), 0);
        for (size_t i = 0; i < sites_.size(); ++i) {
            const auto& site = sites_[i];
            fault_choices[i] = 0;
            const auto draw = rng_.next_double();
            if (draw >= site.probability)
                continue;
            const auto channel = std::min(size_t(draw / site.probability * site.channels.size()),
                                          site.channels.size() - 1);
            fault_choices[i] = channel + 1;
            const auto& effect = site.channels[channel];
            total.qx ^= effect.qx;
            total.qz ^= effect.qz;
            total.wx ^= effect.wx;
            total.wz ^= effect.wz;
            total.spectator ^= effect.spectator;
            total.pre ^= effect.pre;
            total.middle ^= effect.middle;
            if (effect.readout)
                readout_[effect.readout - 1] ^= 1;
        }
        static constexpr std::array<Complex, 4> roots{Complex(1), Complex(0, 1), Complex(-1),
                                                      Complex(0, -1)};
        for (size_t m = 0; m < 2; ++m) {
            const bool reset = m ^ total.middle;
            resets_[m] = reset;
            const auto qx = total.qx ^ (reset ? reset_x_ : 0);
            const auto qz = total.qz ^ (reset ? reset_z_ : 0);
            offsets_[m] = offset ^ qx ^ total.wx;
            spectator_flips_[m] = total.spectator ^ (reset ? reset_spectator_ : 0);
            const auto y_bra =
                roots[(4 - n_ % 4) % 4] * double(std::popcount(offsets_[m]) % 2 ? -1 : 1);
            for (size_t branch = 0; branch < 2; ++branch) {
                const auto sign = branch && (m ^ total.pre ^ spectator_parity) ? -1 : 1;
                for (size_t l = 0; l < 2; ++l) {
                    auto& term = terms_[m][2 * branch + l];
                    term.coefficient = logical[l] * (sign * 0.5 / std::sqrt(2.0));
                    if (l ^ branch)
                        term.coefficient *= y_bra;
                    for (size_t q = 0; q < n_; ++q) {
                        const auto a = (qx >> q) & 1, b = (qz >> q) & 1;
                        const auto c = (total.wx >> q) & 1, d = (total.wz >> q) & 1;
                        for (size_t value = 0; value < 2; ++value) {
                            const auto input = value ^ ((offset >> q) & 1) ^ l;
                            term.local[2 * q + value] =
                                phases_[size_t(signs_[q] < 0) * 64 + (a | (b << 1)) * 16 +
                                        (c | (d << 1)) * 4 + branch * 2 + input];
                        }
                    }
                    for (size_t i = 0; i < rank_; ++i) {
                        term.local[2 * (n_ + i)] = 1;
                        term.local[2 * (n_ + i) + 1] = (syndrome >> i) & 1 ? -1 : 1;
                    }
                }
            }
        }
    }

    double probability(size_t m, size_t measured, uint64_t bits, bool y) noexcept {
        double result = 0;
        for (size_t i = 0; i < 4; ++i) {
            const auto& left = terms_[m][i];
            const auto a = y && (i == 1 || i == 2) ? -left.coefficient : left.coefficient;
            for (size_t j = 0; j <= i; ++j) {
                const auto& right = terms_[m][j];
                const auto b = y && (j == 1 || j == 2) ? -right.coefficient : right.coefficient;
                std::copy(left.local.begin(), left.local.end(), paired_.begin());
                for (size_t k = 0; k < right.local.size(); ++k)
                    paired_[2 * leaf_count_ + k] = std::conj(right.local[k]);
                for (size_t k = 0; k < measured; ++k) {
                    if ((bits >> k) & 1) {
                        paired_[2 * (n_ + k) + 1] *= -1;
                        paired_[2 * leaf_count_ + 2 * (n_ + k) + 1] *= -1;
                    }
                }
                result += (i == j ? 1 : 2) *
                          (a * std::conj(b) *
                           contractions_[measured].evaluate(paired_.data(), workspace_))
                              .real();
            }
        }
        assert(result > -1e-9);
        return std::max(0.0, result);
    }

    bool draw(double zero, double parent) noexcept {
        assert(parent > 0 && zero >= -1e-9 && zero <= parent + 1e-9);
        return rng_.next_double() * parent >= std::clamp(zero, 0.0, parent);
    }

    void run(Executor& prefix) noexcept {
        prefix.run_shot();
        prepare(prefix);
        const double p00 = probability(0, 0, 0, false), p01 = probability(0, 0, 0, true);
        const double p10 = probability(1, 0, 0, false), p11 = probability(1, 0, 0, true);
        const double p0 = p00 + p01, p1 = p10 + p11;
        max_normalization_error = std::max(max_normalization_error, std::abs(p0 + p1 - 1));
        assert(std::abs(p0 + p1 - 1) < 1e-8);
        const bool m = draw(p0, p0 + p1);
        double parent = m ? p1 : p0;
        const auto y0 = m ? p10 : p00;
        const bool y = draw(y0, parent);
        parent = y ? parent - y0 : y0;
        uint64_t bits = 0;
        for (size_t measured = 1; measured <= rank_; ++measured) {
            const auto zero = probability(m, measured, bits, y);
            const bool bit = draw(zero, parent);
            parent = bit ? parent - zero : zero;
            bits |= uint64_t(bit) << (measured - 1);
        }
        conditional_log_probability = std::log(parent);
        std::fill(records.begin(), records.end(), 0);
        std::copy(prefix.visible_records().begin(), prefix.visible_records().end(),
                  records.begin());
        std::copy(prefix.hidden_records().begin(), prefix.hidden_records().end(),
                  records.begin() + visible_);
        records[measurement_] = m ^ readout_[measurement_];
        records[reset_] = resets_[m];
        records[logical_record_] = y ^ readout_[logical_record_];
        for (size_t i = 0; i < rank_; ++i) {
            records[x_records_[i]] = ((bits >> i) & 1) ^ readout_[x_records_[i]];
            records[z_records_[i]] =
                (std::popcount(offsets_[m] & z_rows_[i]) % 2) ^ readout_[z_records_[i]];
        }
        for (size_t i = 0; i < spectators_.size(); ++i) {
            if (const auto slot = spectators_[i].record) {
                records[slot - 1] =
                    (((spectator_bits_ ^ spectator_flips_[m]) >> i) & 1) ^ readout_[slot - 1];
            }
        }
        const auto write_parities = [&](const auto& rows, auto& output) noexcept {
            for (size_t i = 0; i < rows.size(); ++i) {
                uint8_t parity = 0;
                for (auto slot : rows[i])
                    parity ^= records[slot];
                output[i] = parity;
            }
        };
        write_parities(detector_rows_, detectors);
        write_parities(observable_rows_, observables);
    }
};

void print_bits(const auto& bits) {
    std::cout << '"';
    for (auto bit : bits)
        std::cout << unsigned(bit);
    std::cout << '"';
}
}  // namespace

int run_cli(int argc, char** argv) {
    if (argc < 5 || argc > 7)
        throw std::invalid_argument(
            "usage: sample_terminal_gadget bundle shots traces seed [baseline_width_limit] "
            "[off|budgeted|unbounded]");
    const std::filesystem::path directory(argv[1]);
    const auto shots = std::stoull(argv[2]), traces = std::stoull(argv[3]),
               seed = std::stoull(argv[4]);
    if (!shots)
        throw std::invalid_argument("shots must be positive");
    const auto baseline_width_limit = argc >= 6 ? std::stoull(argv[5]) : 24;
    const std::string schedule = argc == 7 ? argv[6] : "off";
    auto start = std::chrono::steady_clock::now();
    const auto prefix_plan = compile(read_text(directory / "prefix.stim"));
    ExecutablePlan prefix_executable(prefix_plan);
    Executor prefix(prefix_executable, seed);
    Terminal terminal(directory, prefix_plan, seed);
    const auto compile_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::setprecision(17);
    for (size_t shot = 0; shot < traces; ++shot) {
        terminal.run(prefix);
        std::cout << "{\"kind\":\"sample\",\"shot\":" << shot << ",\"records\":";
        print_bits(terminal.records);
        std::cout << ",\"detectors\":";
        print_bits(terminal.detectors);
        std::cout << ",\"observables\":";
        print_bits(terminal.observables);
        std::cout << ",\"conditional_log_probability\":" << terminal.conditional_log_probability;
        std::cout << ",\"prefix_faults\":[";
        bool first = true;
        for (size_t i = 0; i < prefix_plan.presampled_noise_sites.size(); ++i) {
            const auto& site = prefix_plan.presampled_noise_sites[i];
            for (size_t j = 0; j < site.outcomes.size(); ++j) {
                if (prefix.symbols()[index(site.outcomes[j].symbol)]) {
                    std::cout << (first ? "" : ",") << '[' << i << ',' << j + 1 << ']';
                    first = false;
                }
            }
        }
        std::cout << "],\"prefix_readout\":[";
        first = true;
        for (const auto& action : prefix_plan.actions) {
            if (const auto* readout = std::get_if<ApplyReadoutNoise>(&action.action)) {
                if (prefix.symbols()[index(readout->flip)]) {
                    std::cout << (first ? "" : ",") << index(readout->record);
                    first = false;
                }
            }
        }
        std::cout << "],\"suffix_faults\":[";
        first = true;
        for (size_t i = 0; i < terminal.fault_choices.size(); ++i) {
            if (terminal.fault_choices[i]) {
                std::cout << (first ? "" : ",") << '[' << i << ',' << terminal.fault_choices[i]
                          << ']';
                first = false;
            }
        }
        std::cout << "]}\n";
    }
    start = std::chrono::steady_clock::now();
    const auto full_plan = compile(read_text(directory / "full.stim"), schedule);
    // Inspect the symbolic plan before allocating its exponential state.
    const bool baseline_enabled = full_plan.peak_active_width <= baseline_width_limit;
    std::unique_ptr<ExecutablePlan> full_executable;
    std::unique_ptr<Executor> ordinary;
    if (baseline_enabled) {
        full_executable = std::make_unique<ExecutablePlan>(full_plan);
        ordinary = std::make_unique<Executor>(*full_executable, seed);
    }
    const auto ordinary_compile_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (size_t i = 0; i < 16; ++i) {
        terminal.run(prefix);
        if (ordinary)
            ordinary->run_shot();
    }
    std::array<double, 3> compiled_times, ordinary_times;
    uint64_t checksum = 0;
    for (size_t repeat = 0; repeat < 3; ++repeat) {
        start = std::chrono::steady_clock::now();
        for (size_t shot = 0; shot < shots; ++shot) {
            terminal.run(prefix);
            checksum += terminal.observables.empty() ? 0 : terminal.observables[0];
        }
        compiled_times[repeat] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        if (ordinary)
            for (size_t shot = 0; shot < shots; ++shot) {
                ordinary->run_shot();
                checksum += ordinary->observables().empty() ? 0 : ordinary->observables()[0];
            }
        ordinary_times[repeat] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    std::cout << "{\"kind\":\"benchmark\",\"compile_seconds\":" << compile_seconds
              << ",\"baseline_schedule\":\"" << schedule << "\""
              << ",\"ordinary_compile_seconds\":" << ordinary_compile_seconds
              << ",\"shots_per_repeat\":" << shots << ",\"compiled_seconds\":[";
    for (size_t i = 0; i < 3; ++i)
        std::cout << (i ? "," : "") << compiled_times[i];
    std::cout << "],\"ordinary_seconds\":[";
    if (ordinary)
        for (size_t i = 0; i < 3; ++i)
            std::cout << (i ? "," : "") << ordinary_times[i];
    std::cout << "],\"prefix_peak_width\":" << prefix_plan.peak_active_width
              << ",\"ordinary_peak_width\":" << full_plan.peak_active_width
              << ",\"baseline_enabled\":" << (baseline_enabled ? "true" : "false")
              << ",\"baseline_width_limit\":" << baseline_width_limit
              << ",\"max_normalization_error\":" << terminal.max_normalization_error
              << ",\"checksum\":" << checksum << "}\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "native terminal compilation failed: " << error.what() << '\n';
        return 2;
    }
}
