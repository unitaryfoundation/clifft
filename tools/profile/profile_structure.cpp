// Export semantic actions without parsing the human-facing plan inspection.
// Research tooling stays outside the executor and its allocation-free dispatch.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using namespace clifft::sampling;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

SamplingPlan prefix_plan(const SamplingPlan& original, size_t count) {
    auto prefix = original;
    prefix.actions.clear();
    prefix.final_tableau.reset();
    prefix.source_map.reset();
    prefix.num_detectors = prefix.num_observables = prefix.num_exp_vals = 0;
    prefix.peak_active_width = prefix.initial_active_width;
    std::vector<bool> records(prefix.num_visible_records + prefix.num_hidden_records);
    for (auto& kind : prefix.symbols) {
        if (kind != SymbolKind::Presampled) {
            kind = SymbolKind::Unused;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const auto& action = original.actions[i];
        if (std::holds_alternative<WriteDetector>(action.action) ||
            std::holds_alternative<WriteObservable>(action.action) ||
            std::holds_alternative<WriteExpectationValue>(action.action)) {
            continue;
        }
        std::visit(
            [&](const auto& op) {
                using T = std::decay_t<decltype(op)>;
                auto keep_symbol = [&](SymbolId symbol) {
                    prefix.symbols[index(symbol)] = original.symbols[index(symbol)];
                };
                if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                              std::is_same_v<T, MeasureDormantRandom>) {
                    keep_symbol(op.branch);
                    records[index(op.record)] = true;
                } else if constexpr (std::is_same_v<T, DefineSymbol>) {
                    keep_symbol(op.symbol);
                } else if constexpr (std::is_same_v<T, ApplyReadoutNoise>) {
                    keep_symbol(op.flip);
                } else if constexpr (std::is_same_v<T, RecordClassical>) {
                    records[index(op.record)] = true;
                }
            },
            action.action);
        prefix.actions.push_back(action);
        prefix.peak_active_width = std::max(prefix.peak_active_width, action.active_after);
    }
    const auto width = original.actions[count - 1].active_after;
    // Preserve original record indices for feedback, filling unused suffix slots.
    // These writes cannot affect the residual or consume RNG.
    for (uint32_t record = 0; record < records.size(); ++record) {
        if (!records[record]) {
            prefix.actions.push_back(
                {width, width, RecordClassical{AffineBool(false), RecordSlot{record}}});
        }
    }
    prefix.validate();
    return prefix;
}

void write_snapshots(const SamplingPlan& plan, const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::vector<size_t> candidates;
    for (size_t i = 0; i < plan.actions.size(); ++i) {
        const auto& action = plan.actions[i];
        if (action.active_after > 0 && action.active_after <= 16 &&
            predicted_dense_passes(action.action)) {
            candidates.push_back(i);
        }
    }
    std::vector<size_t> selected;
    for (size_t j = 0; j < std::min(size_t{16}, candidates.size()); ++j) {
        selected.push_back(
            candidates[j * (candidates.size() - 1) /
                       std::max(size_t{1}, std::min(size_t{16}, candidates.size()) - 1)]);
    }
    if (!candidates.empty()) {
        // A short-lived peak can fall between the evenly spaced checkpoints.
        selected.push_back(
            *std::max_element(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
                return plan.actions[a].active_after < plan.actions[b].active_after;
            }));
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    }
    std::ofstream metadata(directory / "snapshots.json");
    metadata.exceptions(std::ios::badbit | std::ios::failbit);
    metadata << "{\"byte_order\":\""
             << (std::endian::native == std::endian::little ? "little" : "big")
             << "\",\"sampling\":\"independent unconditional prefixes\",\"snapshots\":[";
    bool first = true;
    for (size_t action : selected) {
        auto prefix = prefix_plan(plan, action + 1);
        ExecutablePlan executable(prefix);
        Executor executor(executable, 42 + action);
        for (int sample = 0; sample < 4; ++sample) {
            executor.run_shot();
            const auto filename = std::to_string(action) + "-" + std::to_string(sample) + ".bin";
            std::ofstream output(directory / filename, std::ios::binary);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            const auto& state = executor.state();
            const auto bytes = static_cast<std::streamsize>(state.size() * sizeof(double));
            output.write(reinterpret_cast<const char*>(state.real_data()), bytes);
            output.write(reinterpret_cast<const char*>(state.imag_data()), bytes);
            if (!first) {
                metadata << ',';
            }
            first = false;
            metadata << "{\"action\":" << action << ",\"sample\":" << sample
                     << ",\"width\":" << state.active_width() << ",\"file\":\"" << filename
                     << "\"}";
        }
    }
    metadata << "]}\n";
}

void write_action(const PlannedAction& action) {
    std::cout << "{\"before\":" << action.active_before << ",\"after\":" << action.active_after
              << ",\"dense_passes\":" << predicted_dense_passes(action.action);
    std::visit(
        [](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, RotateActivePauli> ||
                          std::is_same_v<T, MeasureActivePauli>) {
                std::cout << ",\"kind\":\""
                          << (std::is_same_v<T, RotateActivePauli> ? "rotate" : "measure")
                          << "\",\"support\":" << (op.pauli.x | op.pauli.z);
                if constexpr (std::is_same_v<T, MeasureActivePauli>) {
                    std::cout << ",\"pivot\":" << op.active_pivot;
                }
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                std::cout << ",\"kind\":\"promote\"";
            } else if constexpr (std::is_same_v<T, WriteExpectationValue>) {
                std::cout << ",\"kind\":\"expectation\",\"support\":"
                          << (op.active ? op.active->projection.x | op.active->projection.z : 0);
            } else if constexpr (std::is_same_v<T, ApplyInstrument> ||
                                 std::is_same_v<T, InstrumentBoundary>) {
                std::cout << ",\"kind\":\"unsupported_instrument\"";
            } else {
                static_assert(
                    std::is_same_v<T, MeasureDormantRandom> || std::is_same_v<T, RecordClassical> ||
                    std::is_same_v<T, DefineSymbol> || std::is_same_v<T, ApplyReadoutNoise> ||
                    std::is_same_v<T, WriteDetector> || std::is_same_v<T, WriteObservable>);
                std::cout << ",\"kind\":\"classical\"";
            }
        },
        action.action);
    std::cout << '}';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 6) {
            throw std::invalid_argument(
                "usage: profile_structure CIRCUIT [SHOTS=128] [MAX_WIDTH=22] [POSTSELECT=0] "
                "[SNAPSHOT_DIR]");
        }
        const int shots = argc > 2 ? std::stoi(argv[2]) : 128;
        const int max_width = argc > 3 ? std::stoi(argv[3]) : 22;
        const int postselect = argc > 4 ? std::stoi(argv[4]) : 0;
        if (shots < 1 || max_width < 0 || max_width > 24 || postselect < 0 || postselect > 1) {
            throw std::invalid_argument(
                "require SHOTS > 0, MAX_WIDTH in [0,24], POSTSELECT 0 or 1");
        }
        std::ifstream input(argv[1]);
        if (!input) {
            throw std::invalid_argument("cannot open circuit");
        }
        std::ostringstream text;
        text << input.rdbuf();
        const auto circuit_text = text.str();
        const auto start = Clock::now();
        auto circuit = clifft::parse(circuit_text);
        const auto parsed = Clock::now();
        auto hir = clifft::trace(circuit);
        auto passes = clifft::default_hir_pass_manager();
        passes.run(hir);
        const auto traced = Clock::now();
        std::vector<uint8_t> mask;
        if (postselect) {
            for (const auto& op : hir.ops) {
                if (op.op_type() == clifft::OpType::DETECTOR) {
                    mask.push_back(1);
                }
            }
        }
        SamplingPlanOptions options;
        options.postselection_mask = mask;
        auto plan = plan_sampling(hir, options);
        const auto planned = Clock::now();
        ExecutablePlan executable(plan);
        const auto prepared = Clock::now();
        bool instruments = false;
        for (const auto& action : plan.actions) {
            instruments |= std::holds_alternative<ApplyInstrument>(action.action);
        }
        std::cout << std::setprecision(17) << "{\"schema\":1,\"num_qubits\":" << plan.num_qubits
                  << ",\"initial_width\":" << plan.initial_active_width
                  << ",\"peak_width\":" << plan.peak_active_width
                  << ",\"executable_actions\":" << executable.num_actions()
                  << ",\"postselect\":" << postselect
                  << ",\"parse_ms\":" << milliseconds(start, parsed)
                  << ",\"trace_ms\":" << milliseconds(parsed, traced)
                  << ",\"plan_ms\":" << milliseconds(traced, planned)
                  << ",\"prepare_ms\":" << milliseconds(planned, prepared)
                  << ",\"compile_ms\":" << milliseconds(start, prepared);
        // Keep allocation out of hot-shot timing and report it separately.
        // Instruments need the public trajectory driver, not this bare executor.
        if (!instruments && plan.peak_active_width <= static_cast<uint32_t>(max_width)) {
            const auto init = Clock::now();
            Executor executor(executable, 42);
            const auto initialized = Clock::now();
            executor.run_shot();
            int discarded = 0;
            const auto sampled = Clock::now();
            for (int shot = 0; shot < shots; ++shot) {
                executor.run_shot();
                discarded += executor.discarded();
            }
            const auto end = Clock::now();
            std::cout << ",\"executor_init_ms\":" << milliseconds(init, initialized)
                      << ",\"shot_ms\":" << milliseconds(sampled, end) / shots
                      << ",\"shots\":" << shots << ",\"discarded\":" << discarded;
            // A diagnostic probe outside timing verifies reconstructed ideal inputs.
            // Values from an early-discarded shot are not meaningful.
            std::cout << ",\"last_shot_exp_vals\":";
            if (executor.discarded()) {
                std::cout << "null";
            } else {
                std::cout << '[';
                bool first = true;
                for (double expectation : executor.exp_vals()) {
                    if (!first) {
                        std::cout << ',';
                    }
                    first = false;
                    std::cout << expectation;
                }
                std::cout << ']';
            }
        } else {
            std::cout << ",\"shot_ms\":null,\"skip_sampling\":\""
                      << (instruments ? "instrument" : "width_limit") << '"';
        }
        std::cout << ",\"actions\":[";
        for (size_t i = 0; i < plan.actions.size(); ++i) {
            if (i) {
                std::cout << ',';
            }
            write_action(plan.actions[i]);
        }
        std::cout << "]}\n";
        if (argc == 6) {
            if (instruments || plan.peak_active_width > 16) {
                throw std::invalid_argument(
                    "snapshots require no instruments and peak width <= 16");
            }
            write_snapshots(plan, argv[5]);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
