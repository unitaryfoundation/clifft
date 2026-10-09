// Research host: compose independently traced fragments before ordinary planning.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/sampling/state_queries.h"
#include "clifft/util/hir_introspection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace clifft;
using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string read(const char* path) {
    std::ifstream file(path);
    if (!file)
        throw std::invalid_argument("Cannot open input file");
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

void validate(const HirModule& hir, bool prefix) {
    if (!hir.final_tableau || !hir.noise_sites.empty() || !hir.readout_noise.empty() ||
        !hir.instrument_sites.empty() || !hir.logical_noise_prefix.empty() ||
        hir.forced_traceout_slot || hir.neglect_instrument_damping)
        throw std::invalid_argument("Expected materialized Pauli faults and a complete frame");
    if (prefix && (hir.num_measurements || hir.num_hidden_measurements || hir.num_detectors ||
                   hir.num_observables || hir.num_exp_vals))
        throw std::invalid_argument("Reusable prefix must be unmeasured and deterministic");
    for (const auto& op : hir.ops) {
        const auto type = op.op_type();
        if (type == OpType::T_GATE || type == OpType::PHASE_ROTATION)
            continue;
        if (!prefix &&
            (type == OpType::MEASURE || type == OpType::CONDITIONAL_PAULI ||
             type == OpType::DETECTOR || type == OpType::OBSERVABLE || type == OpType::EXP_VAL))
            continue;
        throw std::invalid_argument("Unsupported fragment operation");
    }
}

void append(HirModule& dest, const HirModule& source, const HeisenbergOp& op,
            const Tableau* pullback) {
    const auto fill = [&](MutablePauliMaskView slot) {
        auto axis = optimizer_detail::copy_axis(source.mask_view(op), source.num_qubits);
        if (pullback)
            axis = pullback->apply(axis.view());
        optimizer_detail::write_axis(slot, axis);
    };
    switch (op.op_type()) {
        case OpType::T_GATE:
            dest.append_tgate(op.is_dagger(), fill);
            break;
        case OpType::PHASE_ROTATION:
            dest.append_phase_rotation(op.alpha(), fill);
            break;
        case OpType::MEASURE:
            dest.append_measure(op.meas_record_idx(), fill).set_hidden(op.is_hidden());
            break;
        case OpType::CONDITIONAL_PAULI:
            dest.append_conditional(op.controlling_meas(), fill);
            break;
        case OpType::DETECTOR:
            dest.append_detector(op.detector_idx());
            break;
        case OpType::OBSERVABLE:
            dest.append_observable(op.observable_idx(), op.observable_target_list_idx());
            break;
        case OpType::EXP_VAL:
            dest.append_exp_val(op.exp_val_idx(), fill);
            break;
        default:
            throw std::invalid_argument("Unsupported fragment operation");
    }
}

HirModule compose(const HirModule& prefix, const Tableau& pullback, const HirModule& tail) {
    HirModule result(prefix.num_qubits, prefix.ops.size() + tail.ops.size());
    result.num_measurements = tail.num_measurements;
    result.num_hidden_measurements = tail.num_hidden_measurements;
    result.num_detectors = tail.num_detectors;
    result.num_observables = tail.num_observables;
    result.num_exp_vals = tail.num_exp_vals;
    result.detector_targets = tail.detector_targets;
    result.observable_targets = tail.observable_targets;
    for (const auto& op : prefix.ops)
        append(result, prefix, op, nullptr);
    // U_prefix = F_prefix R_prefix. Pull each tail axis back through
    // F_prefix, leaving the original rotations before the transformed tail.
    // The prefix has no records, so visible and hidden tail indices stay fixed.
    for (const auto& op : tail.ops)
        append(result, tail, op, &pullback);
    result.final_tableau = prefix.final_tableau->then(*tail.final_tableau);
    // Separate fragment line numbers do not describe the assembled input.
    result.source_map.clear();
    return result;
}

bool same_hir(const HirModule& a, const HirModule& b) {
    if (a.final_tableau != b.final_tableau || a.ops.size() != b.ops.size() ||
        a.num_qubits != b.num_qubits || a.num_measurements != b.num_measurements ||
        a.num_hidden_measurements != b.num_hidden_measurements ||
        a.num_detectors != b.num_detectors || a.num_observables != b.num_observables ||
        a.num_exp_vals != b.num_exp_vals || a.detector_targets != b.detector_targets ||
        a.observable_targets != b.observable_targets)
        return false;
    for (size_t i = 0; i < a.ops.size(); ++i) {
        const auto& x = a.ops[i];
        const auto& y = b.ops[i];
        const auto mx = x.has_mask() ? std::optional{a.mask_view(x)} : std::nullopt;
        const auto my = y.has_mask() ? std::optional{b.mask_view(y)} : std::nullopt;
        if (x.flags() != y.flags() || format_hir_op(x, mx) != format_hir_op(y, my) ||
            (x.op_type() == OpType::PHASE_ROTATION && x.alpha() != y.alpha()))
            return false;
    }
    return true;
}

std::string bits(const std::vector<uint8_t>& values) {
    std::string result;
    for (const auto bit : values)
        result += '0' + bit;
    return result;
}

long resident_peak_kib() {
    std::istringstream status(read("/proc/self/status"));
    for (std::string line; std::getline(status, line);) {
        if (line.starts_with("VmHWM:")) {
            long result = 0;
            std::istringstream(line.substr(6)) >> result;
            return result;
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("Usage: profile_prefix_trace_reuse PREFIX MAX_WIDTH PHASE");
        const auto max_width = std::stoul(argv[2]);
        const bool phase = std::stoi(argv[3]) != 0;
        if (max_width > 16)
            throw std::invalid_argument("Width budget exceeds sixteen");
        auto start = Clock::now();
        auto prefix_text = read(argv[1]);
        if (!prefix_text.empty() && prefix_text.back() != '\n')
            prefix_text += '\n';
        const auto prefix_lines = std::count(prefix_text.begin(), prefix_text.end(), '\n');
        const auto prefix_ast = parse(prefix_text);
        const auto prefix_hir = trace(prefix_ast);
        validate(prefix_hir, true);
        const auto pullback = prefix_hir.final_tableau->inverse();
        std::cout << std::setprecision(17) << "{\"setup_seconds\":" << seconds(start)
                  << ",\"prefix_nodes\":" << prefix_ast.nodes.size()
                  << ",\"prefix_ops\":" << prefix_hir.ops.size() << "}" << std::endl;
        for (std::string request; std::getline(std::cin, request);) {
            std::string mode;
            uint64_t seed;
            size_t count;
            bool check;
            if (!(std::istringstream(request) >> mode >> seed >> count >> check) ||
                (mode != "fresh" && mode != "parsed" && mode != "traced") || count > 1000000)
                throw std::invalid_argument("Invalid request");
            std::string tail_text, line;
            for (size_t i = 0; i < count; ++i) {
                if (!std::getline(std::cin, line))
                    throw std::invalid_argument("Incomplete request");
                tail_text += line + "\n";
            }
            start = Clock::now();
            auto circuit = parse(mode == "fresh" ? prefix_text + tail_text : tail_text);
            if (circuit.num_qubits > prefix_hir.num_qubits)
                throw std::invalid_argument("Continuation exceeds the prepared physical width");
            circuit.num_qubits = prefix_hir.num_qubits;
            const double parse_seconds = seconds(start);
            double assemble_seconds = 0;
            if (mode == "parsed") {
                start = Clock::now();
                for (auto& node : circuit.nodes)
                    if (node.source_line)
                        node.source_line += prefix_lines;
                circuit.nodes.insert(circuit.nodes.begin(), prefix_ast.nodes.begin(),
                                     prefix_ast.nodes.end());
                assemble_seconds = seconds(start);
            }
            start = Clock::now();
            auto hir = trace(circuit);
            validate(hir, false);
            const double trace_seconds = seconds(start);
            double compose_seconds = 0;
            if (mode == "traced") {
                start = Clock::now();
                hir = compose(prefix_hir, pullback, hir);
                compose_seconds = seconds(start);
            }
            start = Clock::now();
            if (check && mode != "fresh") {
                const auto reference = trace(parse(prefix_text + tail_text));
                if (!same_hir(hir, reference))
                    throw std::runtime_error("Composed trace differs from complete tracing");
            }
            const double validation_seconds = seconds(start);
            start = Clock::now();
            for (const auto* name : {"PeepholeFusionPass", "PhasePolynomialPass",
                                     "RotationSimplificationPass", "StatevectorSqueezePass"}) {
                if (phase || std::string(name) != "PhasePolynomialPass")
                    make_hir_pass(name)->run(hir);
            }
            const double optimize_seconds = seconds(start);
            start = Clock::now();
            const auto width = analyze_active_width(hir).peak_width;
            const double width_seconds = seconds(start);
            double plan_seconds = 0, prepare_seconds = 0, sample_seconds = 0;
            sampling::SamplingResult sample;
            std::vector<double> probabilities;
            std::vector<std::complex<double>> state;
            if (width <= max_width) {
                start = Clock::now();
                auto plan = sampling::plan_sampling(hir);
                plan_seconds = seconds(start);
                if (plan.peak_active_width > max_width)
                    throw std::runtime_error("Planner width exceeds the inspection budget");
                start = Clock::now();
                sampling::ExecutablePlan executable(plan);
                prepare_seconds = seconds(start);
                start = Clock::now();
                sample = sampling::sample(executable, 1, seed, 1);
                sample_seconds = seconds(start);
                if (check && hir.num_qubits <= 5) {
                    const auto total_records = hir.num_measurements + hir.num_hidden_measurements;
                    if (total_records <= 10) {
                        probabilities.resize(size_t{1} << hir.num_measurements);
                        sampling::Executor executor(executable);
                        std::vector<uint8_t> forced(total_records);
                        for (size_t record = 0; record < (size_t{1} << total_records); ++record) {
                            for (uint32_t i = 0; i < total_records; ++i)
                                forced[i] = (record >> i) & 1;
                            const auto replay = executor.replay_shot(forced);
                            if (replay.reachable)
                                probabilities[record & (probabilities.size() - 1)] +=
                                    std::exp(replay.log_probability);
                        }
                    }
                    if (executable.supports_final_state_queries())
                        state = sampling::get_statevector(executable);
                }
            }
            std::cout << "{\"width\":" << width << ",\"t_count\":" << hir.num_t_gates()
                      << ",\"validation_seconds\":" << validation_seconds
                      << ",\"checked\":" << (check && mode != "fresh" ? "true" : "false")
                      << ",\"stage_seconds\":{\"parse\":" << parse_seconds
                      << ",\"assemble\":" << assemble_seconds << ",\"trace\":" << trace_seconds
                      << ",\"compose\":" << compose_seconds << ",\"optimize\":" << optimize_seconds
                      << ",\"width\":" << width_seconds << ",\"plan\":" << plan_seconds
                      << ",\"prepare\":" << prepare_seconds << ",\"sample\":" << sample_seconds
                      << "},\"measurements\":\"" << bits(sample.measurements)
                      << "\",\"detectors\":\"" << bits(sample.detectors) << "\",\"observables\":\""
                      << bits(sample.observables) << "\",\"exp_vals\":[";
            for (size_t i = 0; i < sample.exp_vals.size(); ++i)
                std::cout << (i ? "," : "") << sample.exp_vals[i];
            std::cout << "],\"record_probabilities\":[";
            for (size_t i = 0; i < probabilities.size(); ++i)
                std::cout << (i ? "," : "") << probabilities[i];
            std::cout << "],\"statevector\":[";
            for (size_t i = 0; i < state.size(); ++i)
                std::cout << (i ? "," : "") << '[' << state[i].real() << ',' << state[i].imag()
                          << ']';
            std::cout << "],\"native_peak_kib\":" << resident_peak_kib() << "}" << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
