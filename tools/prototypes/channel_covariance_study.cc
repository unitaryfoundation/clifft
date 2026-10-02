// Compiler-only study of channel commutation. It does not register a pass.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/active_width_schedule_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/phase_polynomial_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace clifft;
using Body = std::vector<uint64_t>;
using Distribution = std::map<Body, double>;
using Clock = std::chrono::steady_clock;

Body body(PauliMaskView mask) {
    Body out(mask.x().words.begin(), mask.x().words.end());
    out.insert(out.end(), mask.z().words.begin(), mask.z().words.end());
    return out;
}

bool anticommutes(const Body& a, const Body& b) {
    const size_t words = a.size() / 2;
    unsigned parity = 0;
    for (size_t j = 0; j < words; ++j) {
        parity ^= std::popcount(a[j] & b[j + words]);
        parity ^= std::popcount(a[j + words] & b[j]);
    }
    return parity & 1;
}

Body product(Body a, const Body& b) {
    for (size_t j = 0; j < a.size(); ++j) {
        a[j] ^= b[j];
    }
    return a;
}

Distribution distribution(const HirModule& hir, const NoiseSite& site) {
    Distribution out;
    for (const auto& channel : site.channels) {
        if (channel.prob > 0) {
            out[body(hir.noise_channel_masks.at(channel.mask))] += channel.prob;
        }
    }
    return out;
}

bool channel_commutes(const Body& axis, const Distribution& noise) {
    // On anticommuting Paulis, rotation about A mixes R with i*A*R.
    // Equal weights in each such pair cancel the off-diagonal channel terms.
    // Exact equality is conservative for externally supplied floating weights.
    for (const auto& [error, probability] : noise) {
        if (anticommutes(axis, error)) {
            const auto paired = noise.find(product(axis, error));
            if (paired == noise.end() || paired->second != probability) {
                return false;
            }
        }
    }
    return true;
}

bool components_commute(const Body& axis, const Distribution& noise) {
    return std::ranges::all_of(noise,
                               [&](const auto& term) { return !anticommutes(axis, term.first); });
}

struct Movement {
    uint64_t moved_rotations = 0;
    uint64_t noise_crossings = 0;
    uint64_t covariance_crossings = 0;
    uint64_t noncovariant_crossings = 0;
    uint64_t probes = 0;
};

struct Mode {
    const char* name;
    int direction;
    bool preserve_faults;
    bool require_covariance;
    bool collect = false;
};

constexpr std::array<Mode, 7> modes{{
    {"baseline", 0, false, true},
    {"channel_left", -1, false, true},
    {"channel_right", 1, false, true},
    {"labelled_covariant_left", -1, true, true},
    {"labelled_covariant_right", 1, true, true},
    {"labelled_all_left", -1, true, false},
    {"labelled_all_right", 1, true, false},
}};

constexpr std::array<Mode, 2> collection_modes{{
    {"channel_collect", -1, false, true, true},
    {"labelled_collect", -1, true, true, true},
}};

Movement collect(HirModule& hir, const Mode& mode) {
    if (!hir.logical_noise_prefix_matches_schedule()) {
        throw std::invalid_argument("study requires original logical noise positions");
    }
    if (mode.preserve_faults) {
        hir.materialize_logical_noise_prefix();
    } else {
        hir.logical_noise_prefix.clear();
    }
    std::vector<Distribution> sites;
    for (const auto& site : hir.noise_sites)
        sites.push_back(distribution(hir, site));
    std::vector<uint32_t> order;
    Movement movement;
    size_t begin = 0;
    while (begin < hir.ops.size()) {
        const auto is_rotation = [](OpType type) {
            return type == OpType::T_GATE || type == OpType::PHASE_ROTATION;
        };
        if (!is_rotation(hir.ops[begin].op_type())) {
            order.push_back(begin++);
            continue;
        }
        std::vector<Body> axes;
        std::vector<uint32_t> pending_noise;
        size_t end = begin;
        for (; end < hir.ops.size(); ++end) {
            const auto& op = hir.ops[end];
            if (op.op_type() == OpType::NOISE) {
                pending_noise.push_back(end);
            } else if (is_rotation(op.op_type())) {
                const auto axis = body(hir.mask_view(op));
                bool allowed = std::ranges::all_of(
                    axes, [&](const Body& prior) { return !anticommutes(prior, axis); });
                for (const auto index : pending_noise) {
                    ++movement.probes;
                    const auto& noise =
                        sites.at(static_cast<uint32_t>(hir.ops[index].noise_site_idx()));
                    allowed &= channel_commutes(axis, noise);
                }
                if (!allowed)
                    break;
                // Moving only across pending noise preserves rotation order and
                // keeps the complete block together before suffix faults.
                for (const auto index : pending_noise) {
                    const auto& noise =
                        sites.at(static_cast<uint32_t>(hir.ops[index].noise_site_idx()));
                    ++movement.noise_crossings;
                    movement.covariance_crossings += !components_commute(axis, noise);
                }
                movement.moved_rotations += !pending_noise.empty();
                axes.push_back(axis);
                order.push_back(end);
            } else {
                break;
            }
        }
        order.insert(order.end(), pending_noise.begin(), pending_noise.end());
        begin = end;
    }
    const bool mapped = hir.source_map.size() == hir.ops.size();
    HirModule candidate = hir;
    for (size_t position = 0; position < order.size(); ++position) {
        candidate.ops[position] = hir.ops[order[position]];
        if (mapped)
            candidate.source_map[position] = hir.source_map[order[position]];
        if (mode.preserve_faults) {
            candidate.logical_noise_prefix[position] = hir.logical_noise_prefix[order[position]];
        }
    }
    hir = std::move(candidate);
    return movement;
}

Movement reorder(HirModule& hir, const Mode& mode) {
    if (mode.collect)
        return collect(hir, mode);
    const bool leftward = mode.direction < 0;
    if (!hir.logical_noise_prefix_matches_schedule()) {
        throw std::invalid_argument("study requires original logical noise positions");
    }
    // Channel covariance preserves an average. Original prefixes retain each
    // labelled realization through the planner's existing sign corrections.
    if (mode.preserve_faults) {
        hir.materialize_logical_noise_prefix();
    } else {
        hir.logical_noise_prefix.clear();
    }
    std::vector<Distribution> sites;
    for (const auto& site : hir.noise_sites) {
        sites.push_back(distribution(hir, site));
    }
    Movement result;
    const bool mapped = hir.source_map.size() == hir.ops.size();
    const auto move = [&](size_t index) {
        auto& moving = hir.ops[index];
        if (moving.op_type() != OpType::T_GATE && moving.op_type() != OpType::PHASE_ROTATION) {
            return;
        }
        const auto axis = body(hir.mask_view(moving));
        size_t position = index;
        while (leftward ? position > 0 : position + 1 < hir.ops.size()) {
            const size_t next = leftward ? position - 1 : position + 1;
            const auto& other = hir.ops[next];
            const auto type = other.op_type();
            if (type == OpType::NOISE) {
                ++result.probes;
                const auto& noise = sites.at(static_cast<uint32_t>(other.noise_site_idx()));
                const bool covariant = channel_commutes(axis, noise);
                if (mode.require_covariance && !covariant) {
                    break;
                }
                ++result.noise_crossings;
                result.covariance_crossings += covariant && !components_commute(axis, noise);
                result.noncovariant_crossings += !covariant;
            } else if (type == OpType::T_GATE || type == OpType::PHASE_ROTATION) {
                if (anticommutes(axis, body(hir.mask_view(other)))) {
                    break;
                }
            } else {
                // Do not change instrument positions, record order or classical dependencies.
                break;
            }
            std::swap(hir.ops[position], hir.ops[next]);
            if (mapped) {
                std::swap(hir.source_map[position], hir.source_map[next]);
            }
            if (mode.preserve_faults) {
                std::swap(hir.logical_noise_prefix[position], hir.logical_noise_prefix[next]);
            }
            position = next;
        }
        result.moved_rotations += position != index;
    };
    if (leftward) {
        for (size_t i = 0; i < hir.ops.size(); ++i) {
            move(i);
        }
    } else {
        for (size_t i = hir.ops.size(); i-- > 0;) {
            move(i);
        }
    }
    return result;
}

// This reference uses Pauli transfer eigenvalues, independently of pairing.
bool transfer_commutes(const Body& axis, const Distribution& noise, uint32_t width) {
    const auto eigenvalue = [&](const Body& observable) {
        double value = 0;
        for (const auto& [error, p] : noise) {
            value += anticommutes(observable, error) ? -p : p;
        }
        return value;
    };
    for (uint64_t label = 0; label < (uint64_t{1} << (2 * width)); ++label) {
        const Body observable{label & ((uint64_t{1} << width) - 1), label >> width};
        if (anticommutes(axis, observable) &&
            eigenvalue(observable) != eigenvalue(product(axis, observable))) {
            return false;
        }
    }
    return true;
}

void self_test() {
    uint64_t checks = 0;
    for (uint32_t width : {1U, 2U}) {
        const uint64_t count = uint64_t{1} << (2 * width);
        const uint64_t low = (uint64_t{1} << width) - 1;
        for (uint64_t a = 0; a < count; ++a) {
            const Body axis{a & low, a >> width};
            for (uint64_t trial = 0; trial < 256; ++trial) {
                Distribution noise;
                for (uint64_t e = 1; e < count; ++e) {
                    const uint64_t hash = (trial * 17 + e * 13 + (trial >> (e % 8))) % 8;
                    if (hash) {
                        noise[{e & low, e >> width}] = static_cast<double>(hash) / 512;
                    }
                }
                if (trial & 1) {
                    auto paired = noise;
                    for (const auto& [error, p] : noise) {
                        if (anticommutes(axis, error)) {
                            paired[product(axis, error)] = p;
                        }
                    }
                    // Average a distribution with its A-conjugate pairing.
                    for (auto& [error, p] : paired) {
                        if (anticommutes(axis, error)) {
                            const auto partner = product(axis, error);
                            const double first = noise.contains(error) ? noise.at(error) : 0;
                            const double second = noise.contains(partner) ? noise.at(partner) : 0;
                            p = (first + second) / 2;
                        }
                    }
                    noise = std::move(paired);
                }
                if (channel_commutes(axis, noise) != transfer_commutes(axis, noise, width)) {
                    throw std::runtime_error("channel pairing disagrees with transfer reference");
                }
                ++checks;
            }
        }
    }
    const Body high_axis{0, 0, 0, 0, 0, 2};
    const Distribution high_noise{
        {{0, 0, 2, 0, 0, 0}, 0.125}, {{0, 0, 2, 0, 0, 2}, 0.125}, {{0, 0, 0, 0, 0, 2}, 0.25}};
    if (!channel_commutes(high_axis, high_noise) || components_commute(high_axis, high_noise)) {
        throw std::runtime_error("multiword covariance check failed");
    }
    std::cout << "{\"transfer_checks\":" << checks << ",\"multiword_checks\":1}\n";
}

std::vector<double> joint_by_fault_count(const HirModule& hir,
                                         std::vector<std::vector<double>>* paths = nullptr) {
    const auto semantic = sampling::plan_sampling(hir);
    const sampling::ExecutablePlan program(semantic);
    if (semantic.num_hidden_records || semantic.num_visible_records > 8 ||
        semantic.presampled_noise_sites.size() > 4 || !hir.readout_noise.empty()) {
        throw std::invalid_argument("joint oracle limited to eight records and four noise sites");
    }
    size_t patterns = 1;
    for (const auto& site : semantic.presampled_noise_sites) {
        const size_t outcomes = site.outcomes.size() + 1;
        if (outcomes > 4096 / patterns) {
            throw std::invalid_argument("joint oracle fault-pattern cap exceeded");
        }
        patterns *= outcomes;
    }
    const size_t record_count = size_t{1} << semantic.num_visible_records;
    std::vector<double> out(record_count * (semantic.presampled_noise_sites.size() + 1), 0);
    std::vector<uint32_t> slots(semantic.symbols.size(), 0);
    uint32_t assigned = 0;
    for (size_t i = 0; i < semantic.symbols.size(); ++i) {
        if (semantic.symbols[i] == sampling::SymbolKind::Presampled) {
            slots[i] = assigned++;
        }
    }
    std::vector<uint8_t> values(assigned, 0);
    std::vector<uint8_t> records(semantic.num_visible_records, 0);
    sampling::Executor executor(program);
    std::function<void(size_t, size_t, double)> visit = [&](size_t site_index, size_t k, double p) {
        if (site_index == semantic.presampled_noise_sites.size()) {
            if (paths != nullptr)
                paths->emplace_back(record_count, 0);
            for (size_t bits = 0; bits < record_count; ++bits) {
                for (size_t j = 0; j < records.size(); ++j) {
                    records[j] = (bits >> j) & 1;
                }
                const auto replay = executor.replay_shot(records, values);
                if (replay.reachable) {
                    const double probability = std::exp(replay.log_probability);
                    out[k * record_count + bits] += p * probability;
                    if (paths != nullptr)
                        paths->back()[bits] = probability;
                }
            }
            return;
        }
        const auto& site = semantic.presampled_noise_sites[site_index];
        visit(site_index + 1, k, p * (1 - site.total_probability));
        for (const auto& outcome : site.outcomes) {
            const auto slot = slots.at(sampling::index(outcome.symbol));
            values[slot] = 1;
            visit(site_index + 1, k + 1, p * outcome.probability);
            values[slot] = 0;
        }
    };
    visit(0, 0, 1);
    return out;
}

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void print_array(const std::vector<double>& values) {
    std::cout << '[';
    for (size_t j = 0; j < values.size(); ++j) {
        if (j)
            std::cout << ',';
        std::cout << values[j];
    }
    std::cout << ']';
}

struct Arm {
    std::unique_ptr<sampling::ExecutablePlan> program;
    Movement movement;
    uint64_t output_t = 0;
    uint64_t phase_reductions = 0;
    uint64_t phase_oversized = 0;
    uint64_t phase_examined = 0;
    uint64_t reordered_ops = 0;
    uint64_t final_hir_ops = 0;
    uint64_t semantic_actions = 0;
    uint64_t lowered_lane_work = 0;
    double compile_seconds = 0;
    uint32_t reordered_peak = 0;
    uint32_t squeezed_peak = 0;
    double final_dense_work = 0;
    bool schedule_applied = false;
    std::vector<uint32_t> schedule_peaks;
};

Arm prepare(HirModule hir, uint32_t mask_count, const Mode& mode,
            ActiveWidthScheduleOptions schedule_options = {}, uint32_t schedule_runs = 1,
            bool extra_fusion = false, uint32_t phase_variables = 32) {
    const auto start = Clock::now();
    PeepholeFusionPass fusion;
    fusion.run(hir);
    Arm arm;
    if (mode.direction) {
        arm.movement = reorder(hir, mode);
        fusion.run(hir);
    }
    PhasePolynomialPass phase({.max_variables = phase_variables});
    phase.run(hir);
    arm.output_t = hir.num_t_gates();
    arm.phase_reductions = phase.blocks_reduced();
    arm.phase_oversized = phase.oversized_blocks();
    arm.phase_examined = phase.blocks_examined();
    arm.reordered_ops = hir.ops.size();
    arm.reordered_peak = analyze_active_width(hir).peak_width;
    StatevectorSqueezePass().run(hir);
    arm.squeezed_peak = analyze_active_width(hir).peak_width;
    if (extra_fusion)
        fusion.run(hir);
    ActiveWidthSchedulePass schedule(schedule_options);
    for (uint32_t run = 0; run < schedule_runs; ++run) {
        schedule.run(hir);
        arm.schedule_peaks.push_back(schedule.result_peak());
        arm.schedule_applied |= schedule.applied();
        arm.final_dense_work = schedule.result_dense_work();
    }
    if (mask_count > hir.num_detectors) {
        throw std::invalid_argument("postselection detector count out of range");
    }
    std::vector<uint8_t> mask(hir.num_detectors, 0);
    std::fill_n(mask.begin(), mask_count, 1);
    arm.final_hir_ops = hir.ops.size();
    const auto semantic = sampling::plan_sampling(hir, {mask});
    arm.semantic_actions = semantic.actions.size();
    arm.program = std::make_unique<sampling::ExecutablePlan>(semantic);
    arm.lowered_lane_work =
        arm.program->estimated_batch_lane_work(sampling::BatchOutputMode::AggregateSurvivors);
    arm.compile_seconds = seconds(start);
    return arm;
}

void print_joint(const HirModule& source) {
    std::vector<std::vector<double>> baseline_paths;
    const auto baseline = joint_by_fault_count(source, &baseline_paths);
    std::cout << "{\"joint_by_k\":[";
    print_array(baseline);
    std::vector<uint64_t> crossings;
    std::vector<double> path_differences;
    std::vector<Mode> oracle_modes(modes.begin() + 1, modes.end());
    oracle_modes.insert(oracle_modes.end(), collection_modes.begin(), collection_modes.end());
    for (const auto& mode : oracle_modes) {
        auto candidate = source;
        PeepholeFusionPass().run(candidate);
        const auto movement = reorder(candidate, mode);
        crossings.push_back(movement.covariance_crossings);
        PeepholeFusionPass().run(candidate);
        PhasePolynomialPass().run(candidate);
        std::cout << ',';
        std::vector<std::vector<double>> candidate_paths;
        print_array(joint_by_fault_count(candidate, &candidate_paths));
        if (candidate_paths.size() != baseline_paths.size()) {
            throw std::runtime_error("noise outcome count changed");
        }
        double maximum = 0;
        for (size_t pattern = 0; pattern < baseline_paths.size(); ++pattern) {
            for (size_t record = 0; record < baseline_paths[pattern].size(); ++record) {
                maximum = std::max(maximum, std::abs(baseline_paths[pattern][record] -
                                                     candidate_paths[pattern][record]));
            }
        }
        path_differences.push_back(maximum);
    }
    std::cout << "],\"covariance_crossings\":[";
    for (size_t j = 0; j < crossings.size(); ++j) {
        if (j)
            std::cout << ',';
        std::cout << crossings[j];
    }
    std::cout << "],\"maximum_labelled_fault_difference\":";
    print_array(path_differences);
    std::cout << "}\n";
}

void print_collection(const HirModule& source, uint32_t mask_count) {
    const std::array<Mode, 4> variants{
        {modes[0], collection_modes[0], collection_modes[1], collection_modes[0]}};
    std::cout << "{\"input_t\":" << source.num_t_gates() << ",\"arms\":[";
    for (size_t j = 0; j < variants.size(); ++j) {
        if (j)
            std::cout << ',';
        const auto arm = prepare(source, mask_count, variants[j], {}, 1, false, j == 3 ? 64 : 32);
        std::cout << "{\"mode\":\"" << (j == 3 ? "channel_collect_64" : variants[j].name)
                  << "\",\"peak\":" << arm.program->peak_active_width()
                  << ",\"output_t\":" << arm.output_t
                  << ",\"phase_examined\":" << arm.phase_examined
                  << ",\"phase_oversized\":" << arm.phase_oversized
                  << ",\"phase_reductions\":" << arm.phase_reductions
                  << ",\"noise_crossings\":" << arm.movement.noise_crossings
                  << ",\"covariance_crossings\":" << arm.movement.covariance_crossings
                  << ",\"compile_seconds\":" << arm.compile_seconds
                  << ",\"lowered_lane_work\":" << arm.lowered_lane_work << '}';
    }
    std::cout << "]}\n";
}

void print_settings(const HirModule& source, uint32_t mask_count) {
    struct Setting {
        const char* name;
        ActiveWidthScheduleOptions options;
        uint32_t runs = 1;
        bool extra_fusion = false;
    };
    const std::array<Setting, 7> settings{{
        {"baseline", {}},
        {"fusion_after_squeeze", {}, 1, true},
        {"schedule_twice", {}, 2},
        {"schedule_three_times", {}, 3},
        {"greedy", {.beam_width = 1, .search_budget = 0.0}},
        {"budget64", {.search_budget = 64.0}},
        {"beam32_budget64", {.beam_width = 32, .search_budget = 64.0}},
    }};
    std::cout << "{\"input_t\":" << source.num_t_gates() << ",\"arms\":[";
    for (size_t j = 0; j < settings.size(); ++j) {
        if (j)
            std::cout << ',';
        const auto& setting = settings[j];
        const auto arm = prepare(source, mask_count, modes[0], setting.options, setting.runs,
                                 setting.extra_fusion);
        std::cout << "{\"mode\":\"" << setting.name
                  << "\",\"peak\":" << arm.program->peak_active_width()
                  << ",\"actions\":" << arm.program->num_actions()
                  << ",\"reordered_ops\":" << arm.reordered_ops
                  << ",\"final_hir_ops\":" << arm.final_hir_ops
                  << ",\"semantic_actions\":" << arm.semantic_actions
                  << ",\"lowered_lane_work\":" << arm.lowered_lane_work
                  << ",\"output_t\":" << arm.output_t
                  << ",\"compile_seconds\":" << arm.compile_seconds
                  << ",\"final_dense_work\":" << arm.final_dense_work << ",\"schedule_peaks\":[";
        for (size_t run = 0; run < arm.schedule_peaks.size(); ++run) {
            if (run)
                std::cout << ',';
            std::cout << arm.schedule_peaks[run];
        }
        std::cout << "]}";
    }
    std::cout << "]}\n";
}

void print_comparison(const HirModule& source, uint32_t mask_count, bool benchmark,
                      double target_seconds, size_t repeats) {
    std::vector<Arm> arms;
    for (const auto& mode : modes) {
        arms.push_back(prepare(source, mask_count, mode));
    }
    std::vector<uint32_t> shots;
    std::vector<std::vector<double>> times(arms.size());
    if (benchmark) {
        for (const auto& arm : arms) {
            auto start = Clock::now();
            const auto warmup =
                sampling::sample_survivors(*arm.program, 1, 20261001, false, 1, std::nullopt, 1);
            (void)warmup;
            auto count = static_cast<uint32_t>(
                std::clamp(std::ceil(target_seconds / seconds(start)), 1.0, 200000.0));
            start = Clock::now();
            const auto calibration = sampling::sample_survivors(*arm.program, count, 20261001,
                                                                false, 1, std::nullopt, 1);
            (void)calibration;
            count = static_cast<uint32_t>(
                std::clamp(std::ceil(count * target_seconds / seconds(start)), 1.0, 200000.0));
            shots.push_back(count);
        }
        for (size_t repeat = 0; repeat < repeats; ++repeat) {
            for (size_t j = 0; j < arms.size(); ++j) {
                const size_t arm = repeat & 1 ? arms.size() - 1 - j : j;
                const auto start = Clock::now();
                const auto result = sampling::sample_survivors(
                    *arms[arm].program, shots[arm], 20261001 + repeat, false, 1, std::nullopt, 1);
                (void)result;
                times[arm].push_back(seconds(start));
            }
        }
    }
    std::cout << "{\"input_t\":" << source.num_t_gates() << ",\"target_seconds\":" << target_seconds
              << ",\"repeats\":" << (benchmark ? repeats : 0) << ",\"arms\":[";
    for (size_t j = 0; j < arms.size(); ++j) {
        if (j)
            std::cout << ',';
        const auto& arm = arms[j];
        std::cout << "{\"mode\":\"" << modes[j].name
                  << "\",\"peak\":" << arm.program->peak_active_width()
                  << ",\"actions\":" << arm.program->num_actions()
                  << ",\"reordered_ops\":" << arm.reordered_ops
                  << ",\"final_hir_ops\":" << arm.final_hir_ops
                  << ",\"semantic_actions\":" << arm.semantic_actions
                  << ",\"lowered_lane_work\":" << arm.lowered_lane_work
                  << ",\"output_t\":" << arm.output_t
                  << ",\"phase_reductions\":" << arm.phase_reductions
                  << ",\"phase_oversized\":" << arm.phase_oversized
                  << ",\"phase_examined\":" << arm.phase_examined
                  << ",\"compile_seconds\":" << arm.compile_seconds
                  << ",\"moved_rotations\":" << arm.movement.moved_rotations
                  << ",\"noise_crossings\":" << arm.movement.noise_crossings
                  << ",\"covariance_crossings\":" << arm.movement.covariance_crossings
                  << ",\"noncovariant_crossings\":" << arm.movement.noncovariant_crossings
                  << ",\"reordered_peak\":" << arm.reordered_peak
                  << ",\"squeezed_peak\":" << arm.squeezed_peak
                  << ",\"schedule_applied\":" << arm.schedule_applied
                  << ",\"final_dense_work\":" << arm.final_dense_work
                  << ",\"probes\":" << arm.movement.probes;
        if (benchmark) {
            auto sorted = times[j];
            std::ranges::sort(sorted);
            std::cout << ",\"shots\":" << shots[j]
                      << ",\"seconds_per_shot\":" << sorted[sorted.size() / 2] / shots[j]
                      << ",\"timings_seconds\":";
            print_array(times[j]);
        }
        std::cout << '}';
    }
    std::cout << "]}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc < 3) {
            throw std::invalid_argument(
                "usage: channel_covariance_study CIRCUIT POSTSELECT_COUNT "
                "[--joint|--scan|--settings-scan|--collection-scan|--bench [SECONDS "
                "[ODD_REPEATS]]]");
        }
        std::ifstream file(argv[1]);
        if (!file)
            throw std::invalid_argument("cannot read input circuit");
        const std::string source((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
        const auto hir = trace(parse(source));
        const std::string mode = argc > 3 ? argv[3] : "";
        const double target_seconds = argc > 4 ? std::stod(argv[4]) : 0.03;
        const size_t repeats = argc > 5 ? std::stoul(argv[5]) : 3;
        if (!std::isfinite(target_seconds) || target_seconds <= 0 || repeats == 0 ||
            repeats % 2 == 0 || argc > 6 ||
            (mode != "" && mode != "--bench" && mode != "--joint" && mode != "--scan" &&
             mode != "--settings-scan" && mode != "--collection-scan")) {
            throw std::invalid_argument("invalid mode or benchmark calibration settings");
        }
        if (mode == "--joint") {
            print_joint(hir);
        } else if (mode == "--collection-scan") {
            print_collection(hir, static_cast<uint32_t>(std::stoul(argv[2])));
        } else if (mode == "--settings-scan") {
            print_settings(hir, static_cast<uint32_t>(std::stoul(argv[2])));
        } else {
            print_comparison(hir, static_cast<uint32_t>(std::stoul(argv[2])), mode != "--scan",
                             target_seconds, repeats);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
