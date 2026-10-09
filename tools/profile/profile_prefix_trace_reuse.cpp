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

#include "planning_reuse_audit.h"
#include "squeeze_schedule_reuse.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

struct CompositionTimes {
    double boundary = 0;
    double patch = 0;
    double axes = 0;
    double frame = 0;
};

void append_diagonal(Tableau& frame, const std::vector<uint8_t>& powers,
                     const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    std::vector<std::pair<uint32_t, uint8_t>> singles;
    for (uint32_t q = 0; q < powers.size(); ++q)
        if (powers[q])
            singles.emplace_back(q, powers[q]);
    if (singles.empty() && pairs.empty())
        return;
    for (uint32_t q = 0; q < frame.num_qubits(); ++q) {
        for (const bool z_generator : {false, true}) {
            PauliString axis(z_generator ? frame.z_output(q) : frame.x_output(q));
            // Diagonal Cliffords preserve X support. For i^p X^x Z^z,
            // S_a^k adds k*x_a to p and k*x_a mod 2 to z_a. CZ_ab
            // adds 2*x_a*x_b to p and swaps the X bits into the Z updates.
            // Keeping the raw phase also preserves signs of Y-containing rows.
            for (const auto& [a, power] : singles) {
                if (axis.x().bit_get(a)) {
                    axis.add_phase(power);
                    if (power & 1)
                        axis.set_pauli(a, true, !axis.z().bit_get(a));
                }
            }
            for (const auto& [a, b] : pairs) {
                const bool x_a = axis.x().bit_get(a), x_b = axis.x().bit_get(b);
                axis.add_phase(x_a && x_b ? 2 : 0);
                if (x_b)
                    axis.set_pauli(a, x_a, !axis.z().bit_get(a));
                if (x_a)
                    axis.set_pauli(b, x_b, !axis.z().bit_get(b));
            }
            if (z_generator)
                frame.set_z_output(q, axis.view());
            else
                frame.set_x_output(q, axis.view());
        }
    }
}

HirModule compose(const HirModule& prefix, const Tableau& pullback, const HirModule& tail,
                  CompositionTimes* times = nullptr) {
    auto start = Clock::now();
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
    if (times)
        times->axes = seconds(start);
    start = Clock::now();
    result.final_tableau = prefix.final_tableau->then(*tail.final_tableau);
    if (times)
        times->frame = seconds(start);
    // Separate fragment line numbers do not describe the assembled input.
    result.source_map.clear();
    return result;
}

class ContinuationTemplate {
  public:
    explicit ContinuationTemplate(const Circuit& circuit) {
        const auto probed = trace(circuit);
        if (!probed.readout_noise.empty() || !probed.instrument_sites.empty())
            throw std::invalid_argument("Continuation probes must be Pauli channels only");
        base_ = HirModule(probed.num_qubits, probed.ops.size());
        base_.num_measurements = probed.num_measurements;
        base_.num_hidden_measurements = probed.num_hidden_measurements;
        base_.num_detectors = probed.num_detectors;
        base_.num_observables = probed.num_observables;
        base_.num_exp_vals = probed.num_exp_vals;
        base_.detector_targets = probed.detector_targets;
        base_.observable_targets = probed.observable_targets;
        base_.final_tableau = probed.final_tableau;
        std::vector<size_t> starts;
        for (const auto& op : probed.ops) {
            if (op.op_type() == OpType::NOISE) {
                const auto& site = probed.noise_sites[static_cast<uint32_t>(op.noise_site_idx())];
                if (site.channels.size() != 1 || site.total_probability != 0.5)
                    throw std::invalid_argument("Expected one synthetic Pauli generator per probe");
                generators_.push_back(optimizer_detail::copy_axis(
                    probed.noise_channel_masks.at(site.channels[0].mask), probed.num_qubits));
                starts.push_back(base_.ops.size());
            } else {
                if (op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION)
                    throw std::invalid_argument("Reusable continuation must be Clifford");
                append(base_, probed, op, nullptr);
            }
        }
        validate(base_, false);
        words_ = (base_.ops.size() + 63) / 64;
        responses_.resize(generators_.size(), std::vector<uint64_t>(words_));
        record_ops_.resize(base_.num_measurements, base_.ops.size());
        for (size_t i = 0; i < base_.ops.size(); ++i) {
            const auto& op = base_.ops[i];
            if (op.op_type() == OpType::MEASURE && !op.is_hidden())
                record_ops_[static_cast<uint32_t>(op.meas_record_idx())] = i;
            if (!op.has_mask())
                continue;
            const auto axis = optimizer_detail::copy_axis(base_.mask_view(op), base_.num_qubits);
            for (size_t j = 0; j < generators_.size(); ++j)
                if (i >= starts[j] && !axis.view().commutes(generators_[j].view()))
                    responses_[j][i / 64] ^= uint64_t{1} << (i % 64);
        }
    }

    HirModule instantiate(const HirModule& prefix, const Tableau& prefix_inverse,
                          const Circuit& correction, const std::vector<size_t>& active,
                          const std::vector<size_t>& flips, CompositionTimes& times,
                          bool diagonal) const {
        auto start = Clock::now();
        if (prefix.num_qubits != base_.num_qubits || correction.num_qubits > base_.num_qubits)
            throw std::invalid_argument("Correction or prefix has the wrong physical width");
        auto head = prefix;
        auto inverse = prefix_inverse;
        std::vector<uint8_t> powers(diagonal ? base_.num_qubits : 0);
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        for (const auto& node : correction.nodes) {
            const auto gate = node.gate;
            if (gate != GateType::S && gate != GateType::S_DAG && gate != GateType::Z &&
                gate != GateType::CZ && gate != GateType::I)
                throw std::invalid_argument("Expected the diagonal Clifford boundary correction");
            const size_t arity = gate == GateType::CZ ? 2 : 1;
            for (size_t i = 0; i < node.targets.size(); i += arity) {
                std::vector<uint32_t> targets;
                for (size_t j = 0; j < arity; ++j) {
                    if (node.targets[i + j].is_rec())
                        throw std::invalid_argument("Boundary feedback must already be sampled");
                    targets.push_back(node.targets[i + j].value());
                }
                if (!diagonal)
                    head.final_tableau->append_named_gate(gate, targets);
                else if (gate == GateType::CZ)
                    pairs.emplace_back(targets[0], targets[1]);
                else if (gate != GateType::I) {
                    const uint8_t power = gate == GateType::S ? 1 : gate == GateType::Z ? 2 : 3;
                    powers[targets[0]] = (powers[targets[0]] + power) & 3;
                }
                inverse.prepend_named_gate(gate == GateType::S       ? GateType::S_DAG
                                           : gate == GateType::S_DAG ? GateType::S
                                                                     : gate,
                                           targets);
            }
        }
        if (diagonal)
            append_diagonal(*head.final_tableau, powers, pairs);
        times.boundary = seconds(start);
        start = Clock::now();
        auto tail = base_;
        std::vector<uint64_t> signs(words_);
        PauliString final_fault(base_.num_qubits);
        for (const auto j : active) {
            if (j >= generators_.size())
                throw std::invalid_argument("Invalid continuation generator");
            for (size_t w = 0; w < words_; ++w)
                signs[w] ^= responses_[j][w];
            final_fault.mut_x().xor_with(generators_[j].x());
            final_fault.mut_z().xor_with(generators_[j].z());
        }
        for (const auto record : flips) {
            if (record >= record_ops_.size() || record_ops_[record] >= tail.ops.size())
                throw std::invalid_argument("Invalid continuation record flip");
            const auto i = record_ops_[record];
            signs[i / 64] ^= uint64_t{1} << (i % 64);
        }
        for (size_t i = 0; i < tail.ops.size(); ++i)
            if ((signs[i / 64] >> (i % 64)) & 1)
                tail.set_sign(tail.ops[i], !tail.sign(tail.ops[i]));
        // The trace leaves measurements and feedback explicit. Its accumulated
        // Clifford frame therefore retains faults even across resets; removing
        // them at a reset would lose the signed measurement/correction identity.
        final_fault.set_sign(false);
        tail.final_tableau->prepend_pauli(final_fault.view());
        times.patch = seconds(start);
        return compose(head, inverse, tail, &times);
    }

    size_t generators() const { return generators_.size(); }
    size_t response_bytes() const { return generators_.size() * words_ * sizeof(uint64_t); }

  private:
    HirModule base_;
    size_t words_ = 0;
    std::vector<PauliString> generators_;
    std::vector<std::vector<uint64_t>> responses_;
    std::vector<size_t> record_ops_;
};

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
        if (argc != 4 && argc != 5)
            throw std::invalid_argument(
                "Usage: profile_prefix_trace_reuse PREFIX MAX_WIDTH PHASE [CONTINUATION]");
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
        std::unique_ptr<ContinuationTemplate> continuation;
        if (argc == 5) {
            auto circuit = parse(read(argv[4]));
            if (circuit.num_qubits > prefix_hir.num_qubits)
                throw std::invalid_argument("Continuation exceeds the prepared physical width");
            circuit.num_qubits = prefix_hir.num_qubits;
            continuation = std::make_unique<ContinuationTemplate>(circuit);
        }
        const bool diagonal_fixes_prefix =
            planning_reuse_audit::diagonal_boundary_fixes_prefix_axes(prefix_hir);
        const auto schedule_start = Clock::now();
        std::optional<squeeze_schedule_reuse::Schedule> schedule;
        if (continuation && diagonal_fixes_prefix) {
            CompositionTimes unused;
            schedule.emplace(
                continuation->instantiate(prefix_hir, pullback, Circuit{}, {}, {}, unused, true));
        }
        const double schedule_setup_seconds = seconds(schedule_start);
        std::cout << std::setprecision(17) << "{\"setup_seconds\":" << seconds(start)
                  << ",\"prefix_nodes\":" << prefix_ast.nodes.size()
                  << ",\"prefix_ops\":" << prefix_hir.ops.size()
                  << ",\"generators\":" << (continuation ? continuation->generators() : 0)
                  << ",\"response_bytes\":" << (continuation ? continuation->response_bytes() : 0)
                  << ",\"diagonal_boundary_fixes_prefix_axes\":"
                  << (diagonal_fixes_prefix ? "true" : "false")
                  << ",\"squeeze_setup_seconds\":" << schedule_setup_seconds
                  << ",\"squeeze_schedule_ops\":" << (schedule ? schedule->size() : 0)
                  << ",\"squeeze_schedule_moved\":" << (schedule ? schedule->moved() : 0) << "}"
                  << std::endl;
        for (std::string request; std::getline(std::cin, request);) {
            std::string mode;
            uint64_t seed;
            size_t count;
            bool check;
            if (!(std::istringstream(request) >> mode >> seed >> count >> check) ||
                (mode != "fresh" && mode != "parsed" && mode != "traced" &&
                 mode != "continuation" && mode != "diagonal" && mode != "audit" &&
                 mode != "squeeze") ||
                count > 1000000)
                throw std::invalid_argument("Invalid request");
            std::string tail_text, line;
            for (size_t i = 0; i < count; ++i) {
                if (!std::getline(std::cin, line))
                    throw std::invalid_argument("Incomplete request");
                tail_text += line + "\n";
            }
            std::vector<size_t> active, flips;
            std::string reference_text;
            const bool auditing = mode == "audit";
            const bool squeezing = mode == "squeeze";
            const bool reused =
                mode == "continuation" || mode == "diagonal" || auditing || squeezing;
            if (reused) {
                if (!continuation)
                    throw std::invalid_argument("No continuation template was supplied");
                for (auto* values : {&active, &flips}) {
                    if (!std::getline(std::cin, line))
                        throw std::invalid_argument("Missing continuation controls");
                    std::istringstream row(line);
                    for (size_t value; row >> value;)
                        values->push_back(value);
                    if (!row.eof())
                        throw std::invalid_argument("Invalid continuation control");
                }
                if (check) {
                    if (!std::getline(std::cin, line))
                        throw std::invalid_argument("Missing reference size");
                    const auto reference_lines = std::stoul(line);
                    if (reference_lines > 1000000)
                        throw std::invalid_argument("Reference exceeds diagnostic budget");
                    for (size_t i = 0; i < reference_lines; ++i) {
                        if (!std::getline(std::cin, line))
                            throw std::invalid_argument("Missing reference source");
                        reference_text += line + "\n";
                    }
                }
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
            auto hir = reused ? HirModule() : trace(circuit);
            if (!reused)
                validate(hir, false);
            const double trace_seconds = seconds(start);
            double compose_seconds = 0;
            CompositionTimes composition;
            if (mode == "traced" || reused) {
                start = Clock::now();
                hir = mode == "traced"
                          ? compose(prefix_hir, pullback, hir)
                          : continuation->instantiate(prefix_hir, pullback, circuit, active, flips,
                                                      composition, mode != "continuation");
                compose_seconds = seconds(start);
            }
            start = Clock::now();
            std::optional<HirModule> reference_hir;
            if (check && mode != "fresh") {
                reference_hir = trace(parse(prefix_text + (reused ? reference_text : tail_text)));
                if (!same_hir(hir, *reference_hir))
                    throw std::runtime_error("Composed trace differs from complete tracing");
            }
            double validation_seconds = seconds(start);
            std::map<std::string, double> optimizer_seconds;
            std::ostringstream audit;
            if (auditing) {
                hir.source_map.clear();
                for (uint32_t i = 0; i < hir.ops.size(); ++i)
                    hir.source_map.push_back({i + 1});
                audit << "{\"initial\":" << planning_reuse_audit::hir_snapshot(hir);
            }
            start = Clock::now();
            auto scheduling_input =
                squeezing && schedule
                    ? std::optional{squeeze_schedule_reuse::scheduling_snapshot(hir)}
                    : std::nullopt;
            double guard_seconds = squeezing ? seconds(start) : 0;
            std::string squeeze_status = squeezing ? "boundary_not_certified" : "not_requested";
            double optimize_seconds = 0;
            for (const auto* name : {"PeepholeFusionPass", "PhasePolynomialPass",
                                     "RotationSimplificationPass", "StatevectorSqueezePass"}) {
                if (phase || std::string(name) != "PhasePolynomialPass") {
                    const auto before = auditing ? std::optional{hir} : std::nullopt;
                    bool apply_schedule = false;
                    if (scheduling_input && std::string(name) == "StatevectorSqueezePass") {
                        start = Clock::now();
                        apply_schedule = squeeze_schedule_reuse::unchanged_scheduling_input(
                            *scheduling_input, hir);
                        // Charge both creation and destruction of the guard snapshot.
                        scheduling_input.reset();
                        guard_seconds += seconds(start);
                        squeeze_status = apply_schedule ? "reused" : "optimizer_changed";
                    }
                    const auto pass_start = Clock::now();
                    if (apply_schedule)
                        schedule->apply(hir);
                    else
                        make_hir_pass(name)->run(hir);
                    optimizer_seconds[name] = seconds(pass_start);
                    optimize_seconds += optimizer_seconds[name];
                    if (auditing)
                        audit << ",\"" << name
                              << "\":{\"changed\":" << (same_hir(*before, hir) ? "false" : "true")
                              << ",\"hir\":" << planning_reuse_audit::hir_snapshot(hir) << '}';
                }
            }
            optimize_seconds += guard_seconds;
            if (reference_hir && squeezing) {
                start = Clock::now();
                for (const auto* name : {"PeepholeFusionPass", "PhasePolynomialPass",
                                         "RotationSimplificationPass", "StatevectorSqueezePass"})
                    if (phase || std::string(name) != "PhasePolynomialPass")
                        make_hir_pass(name)->run(*reference_hir);
                if (!same_hir(hir, *reference_hir))
                    throw std::runtime_error("Reused schedule differs from fresh optimized HIR");
                validation_seconds += seconds(start);
            }
            start = Clock::now();
            const auto width_trace = analyze_active_width(hir);
            const auto width = width_trace.peak_width;
            const double width_seconds = seconds(start);
            if (auditing) {
                audit << ",\"width_trace\":[";
                for (size_t i = 0; i < width_trace.transitions.size(); ++i) {
                    const auto& t = width_trace.transitions[i];
                    audit << (i ? ",[" : "[") << t.before << ',' << t.after << ','
                          << static_cast<unsigned>(t.effect) << ']';
                }
                audit << ']';
            }
            double plan_seconds = 0, prepare_seconds = 0, sample_seconds = 0;
            sampling::SamplingResult sample;
            std::vector<double> probabilities;
            std::vector<std::complex<double>> state;
            if (width <= max_width) {
                start = Clock::now();
                auto plan = sampling::plan_sampling(hir);
                plan_seconds = seconds(start);
                if (auditing)
                    audit << ",\"plan\":" << planning_reuse_audit::plan_snapshot(plan);
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
            if (auditing)
                audit << '}';
            std::cout << "{\"width\":" << width << ",\"t_count\":" << hir.num_t_gates()
                      << ",\"validation_seconds\":" << validation_seconds
                      << ",\"checked\":" << (check && mode != "fresh" ? "true" : "false")
                      << ",\"optimized_checked\":"
                      << (reference_hir && squeezing ? "true" : "false") << ",\"squeeze_status\":\""
                      << squeeze_status << '"' << ",\"squeeze_guard_seconds\":" << guard_seconds
                      << ",\"stage_seconds\":{\"parse\":" << parse_seconds
                      << ",\"assemble\":" << assemble_seconds << ",\"trace\":" << trace_seconds
                      << ",\"compose\":" << compose_seconds << ",\"optimize\":" << optimize_seconds
                      << ",\"width\":" << width_seconds << ",\"plan\":" << plan_seconds
                      << ",\"prepare\":" << prepare_seconds << ",\"sample\":" << sample_seconds
                      << "},\"composition_seconds\":{\"boundary\":" << composition.boundary
                      << ",\"patch\":" << composition.patch << ",\"axes\":" << composition.axes
                      << ",\"frame\":" << composition.frame << "},\"measurements\":\""
                      << bits(sample.measurements) << "\",\"detectors\":\""
                      << bits(sample.detectors) << "\",\"observables\":\""
                      << bits(sample.observables) << "\",\"optimizer_seconds\":{";
            bool first_pass = true;
            for (const auto& [name, value] : optimizer_seconds) {
                std::cout << (first_pass ? "" : ",") << '"' << name << "\":" << value;
                first_pass = false;
            }
            std::cout << "},\"audit\":" << (auditing ? audit.str() : "null") << ",\"exp_vals\":[";
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
