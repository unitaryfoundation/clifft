// Research host: compose and plan each fixed history before ordinary execution.
// The second boundary applies decoder Paulis before logical target scoring.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/util/hir_introspection.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace clifft;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string read(const fs::path& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot read " + path.string());
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

void append(HirModule& dest, const HirModule& source, const HeisenbergOp& op,
            const Tableau* pullback = nullptr) {
    const auto record = [&](uint32_t index) {
        return index < source.num_measurements
                   ? index
                   : index + dest.num_measurements - source.num_measurements;
    };
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
        case OpType::MEASURE:
            dest.append_measure(
                    static_cast<MeasRecordIdx>(record(static_cast<uint32_t>(op.meas_record_idx()))),
                    fill)
                .set_hidden(op.is_hidden());
            break;
        case OpType::CONDITIONAL_PAULI:
            dest.append_conditional(static_cast<ControllingMeasIdx>(
                                        record(static_cast<uint32_t>(op.controlling_meas()))),
                                    fill);
            break;
        case OpType::DETECTOR:
            dest.append_detector(op.detector_idx());
            break;
        case OpType::OBSERVABLE:
            dest.append_observable(op.observable_idx(), op.observable_target_list_idx());
            break;
        default:
            throw std::invalid_argument("Operation outside the scored diagnostic subset");
    }
}

bool same_hir(const HirModule& a, const HirModule& b) {
    if (a.final_tableau != b.final_tableau || a.ops.size() != b.ops.size() ||
        a.num_qubits != b.num_qubits || a.num_measurements != b.num_measurements ||
        a.num_hidden_measurements != b.num_hidden_measurements ||
        a.num_detectors != b.num_detectors || a.detector_targets != b.detector_targets ||
        a.num_observables != b.num_observables || a.observable_targets != b.observable_targets)
        return false;
    for (size_t i = 0; i < a.ops.size(); ++i) {
        const auto& x = a.ops[i];
        const auto& y = b.ops[i];
        const auto mx = x.has_mask() ? std::optional{a.mask_view(x)} : std::nullopt;
        const auto my = y.has_mask() ? std::optional{b.mask_view(y)} : std::nullopt;
        if (x.flags() != y.flags() || format_hir_op(x, mx) != format_hir_op(y, my))
            return false;
    }
    return true;
}

class Boundaries {
  public:
    explicit Boundaries(const fs::path& directory)
        : raw_(trace(parse(read(directory / "prefix.stim")))),
          protocol_(trace(parse(read(directory / "protocol.stim")))),
          full_(trace(parse(read(directory / "scored.stim")))),
          optimized_(raw_) {
        for (const auto* hir : {&raw_, &protocol_, &full_}) {
            if (hir->num_qubits != 135 || !hir->final_tableau || !hir->noise_sites.empty() ||
                !hir->readout_noise.empty() || !hir->instrument_sites.empty() ||
                hir->num_exp_vals || hir->num_hidden_measurements != 63)
                throw std::invalid_argument("Expected the ideal BT27 boundary inputs");
        }
        if (raw_.num_measurements != 54 || protocol_.num_measurements != 126 ||
            full_.num_measurements != 135 || raw_.num_detectors || raw_.num_observables ||
            protocol_.num_detectors != 72 || protocol_.num_observables ||
            full_.num_detectors != 72 || full_.num_observables != 9 ||
            raw_.ops.size() > protocol_.ops.size() || protocol_.ops.size() > full_.ops.size())
            throw std::invalid_argument("Unexpected scored BT27 interface");
        for (size_t i = raw_.ops.size(); i < protocol_.ops.size(); ++i) {
            if (protocol_.ops[i].op_type() == OpType::T_GATE)
                throw std::invalid_argument("Decoder must remain Clifford");
        }
        const Tableau identity(135);
        if (!same_hir(compose(raw_, identity, identity), full_))
            throw std::runtime_error("Ideal raw boundary composition mismatch");
        for (const auto* name : {"PeepholeFusionPass", "PhasePolynomialPass",
                                 "RotationSimplificationPass", "StatevectorSqueezePass"})
            make_hir_pass(name)->run(optimized_);
    }

    HirModule compose(const HirModule& prefix, const Tableau& phase, const Tableau& decoder) const {
        HirModule result(full_.num_qubits, prefix.ops.size() + full_.ops.size());
        result.num_measurements = full_.num_measurements;
        result.num_hidden_measurements = full_.num_hidden_measurements;
        result.num_detectors = full_.num_detectors;
        result.num_observables = full_.num_observables;
        result.detector_targets = full_.detector_targets;
        result.observable_targets = full_.observable_targets;
        for (const auto& op : prefix.ops)
            append(result, prefix, op);
        const auto decoder_pullback =
            raw_.final_tableau->then(phase.inverse()).then(prefix.final_tableau->inverse());
        for (size_t i = raw_.ops.size(); i < protocol_.ops.size(); ++i)
            append(result, protocol_, protocol_.ops[i], &decoder_pullback);
        const auto decoded_frame = prefix.final_tableau->then(phase)
                                       .then(raw_.final_tableau->inverse())
                                       .then(*protocol_.final_tableau);
        // Decoder Pauli corrections precede the non-Clifford verifier. Their
        // effect is generally more than a flip of its final measurement bits.
        const auto scoring_pullback =
            protocol_.final_tableau->then(decoder.inverse()).then(decoded_frame.inverse());
        for (size_t i = protocol_.ops.size(); i < full_.ops.size(); ++i)
            append(result, full_, full_.ops[i], &scoring_pullback);
        result.final_tableau = decoded_frame.then(decoder)
                                   .then(protocol_.final_tableau->inverse())
                                   .then(*full_.final_tableau);
        return result;
    }

    void restore(sampling::SamplingResult& samples, const std::string& flips) const {
        std::vector<uint8_t> detector_flips(full_.num_detectors, 0);
        std::vector<uint8_t> observable_flips(full_.num_observables, 0);
        for (const auto& op : full_.ops) {
            if (op.op_type() == OpType::DETECTOR) {
                const auto index = static_cast<uint32_t>(op.detector_idx());
                for (auto record : full_.detector_targets[index])
                    detector_flips[index] ^= flips[record] - '0';
            } else if (op.op_type() == OpType::OBSERVABLE) {
                const auto index = static_cast<uint32_t>(op.observable_idx());
                for (auto record : full_.observable_targets[op.observable_target_list_idx()])
                    observable_flips[index] ^= flips[record] - '0';
            }
        }
        for (size_t i = 0; i < samples.measurements.size(); ++i)
            samples.measurements[i] ^= flips[i % full_.num_measurements] - '0';
        for (size_t i = 0; i < samples.detectors.size(); ++i)
            samples.detectors[i] ^= detector_flips[i % full_.num_detectors];
        for (size_t i = 0; i < samples.observables.size(); ++i)
            samples.observables[i] ^= observable_flips[i % full_.num_observables];
    }

    const HirModule& raw() const { return raw_; }
    const HirModule& optimized() const { return optimized_; }

  private:
    HirModule raw_, protocol_, full_, optimized_;
};

Tableau read_correction(size_t count, bool pauli_only) {
    if (count > 4096)
        throw std::invalid_argument("Correction exceeds diagnostic budget");
    std::string source;
    for (size_t i = 0; i < count; ++i) {
        std::string line, gate;
        if (!std::getline(std::cin, line))
            throw std::invalid_argument("Incomplete correction request");
        std::istringstream(line) >> gate;
        if (gate != "I" && gate != "X" && gate != "Y" && gate != "Z" &&
            (pauli_only || (gate != "S" && gate != "S_DAG" && gate != "CZ")))
            throw std::invalid_argument("Unexpected correction gate");
        source += line + "\n";
    }
    source += "I 134\n";
    const auto hir = trace(parse(source));
    if (hir.num_qubits != 135 || !hir.ops.empty() || !hir.final_tableau)
        throw std::invalid_argument("Expected a 135-qubit Clifford correction");
    return *hir.final_tableau;
}

std::string bit_string(const std::vector<uint8_t>& values) {
    std::string result;
    result.reserve(values.size());
    for (const auto value : values)
        result.push_back('0' + value);
    return result;
}

long peak_rss_kib() {
    // Linux getrusage can retain the Python parent's pre-exec resident peak.
    std::istringstream status(read("/proc/self/status"));
    for (std::string line; std::getline(status, line);) {
        if (line.starts_with("VmHWM:")) {
            long kib = 0;
            if (std::istringstream(line.substr(6)) >> kib)
                return kib;
        }
    }
    throw std::runtime_error("Cannot read Linux resident-memory high-water mark");
}

void export_clifford(HirModule hir, const fs::path& path) {
    for (const auto* name : {"PeepholeFusionPass", "PhasePolynomialPass",
                             "RotationSimplificationPass", "StatevectorSqueezePass"})
        make_hir_pass(name)->run(hir);
    if (hir.num_t_gates() || !hir.noise_sites.empty() || !hir.readout_noise.empty())
        throw std::runtime_error("Exact output check requires a noiseless Clifford HIR");
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Cannot write Clifford diagnostic");
    const auto reset = [&] {
        out << "R";
        for (uint32_t q = 0; q < hir.num_qubits; ++q)
            out << ' ' << q;
        out << '\n';
    };
    // The HIR starts in |0>. Discarding the final quantum state isolates the
    // classical output law; its final physical tableau cannot affect records.
    reset();
    std::vector<int64_t> order(hir.num_measurements + hir.num_hidden_measurements, -1);
    int64_t count = 0;
    for (const auto& op : hir.ops) {
        if (op.op_type() == OpType::DETECTOR || op.op_type() == OpType::OBSERVABLE)
            continue;
        const auto axis = format_pauli_mask(hir.mask_view(op));
        if (op.op_type() == OpType::MEASURE) {
            const auto record = static_cast<uint32_t>(op.meas_record_idx());
            if (record >= order.size() || order[record] != -1)
                throw std::runtime_error("Invalid exported record numbering");
            order[record] = count++;
            out << "# RECORD " << record << '\n';
            if (axis.substr(1) == "I")
                out << "MPAD " << (axis[0] == '-') << '\n';
            else
                out << "MPP " << (axis[0] == '-' ? "!" : "") << axis.substr(1) << '\n';
        } else if (op.op_type() == OpType::CONDITIONAL_PAULI) {
            const auto record = static_cast<uint32_t>(op.controlling_meas());
            if (record >= order.size() || order[record] < 0)
                throw std::runtime_error("Exported feedback precedes its measurement");
            if (axis.substr(1) == "I")
                continue;
            std::istringstream terms(axis.substr(1));
            for (std::string term; std::getline(terms, term, '*');)
                out << 'C' << term[0] << " rec[" << order[record] - count << "] " << term.substr(1)
                    << '\n';
        } else {
            throw std::runtime_error("Unsupported Clifford export operation");
        }
    }
    const auto targets = [&](const std::vector<uint32_t>& records) {
        for (auto record : records) {
            if (record >= hir.num_measurements || order[record] < 0)
                throw std::runtime_error("Invalid exported output parity");
            out << " rec[" << order[record] - count << ']';
        }
        out << '\n';
    };
    for (const auto& records : hir.detector_targets) {
        out << "DETECTOR";
        targets(records);
    }
    for (const auto& op : hir.ops) {
        if (op.op_type() == OpType::OBSERVABLE) {
            out << "OBSERVABLE_INCLUDE(" << static_cast<uint32_t>(op.observable_idx()) << ')';
            targets(hir.observable_targets[op.observable_target_list_idx()]);
        }
    }
    reset();
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--export-source") {
            export_clifford(trace(parse(read(argv[2]))), argv[3]);
            return 0;
        }
        if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--export-clifford"))
            throw std::invalid_argument(
                "Usage: profile_bt27_scored_sampling DIRECTORY [--export-clifford]");
        const bool export_enabled = argc == 3;
        const fs::path directory(argv[1]);
        auto start = Clock::now();
        Boundaries boundaries(directory);
        std::cout << "{\"setup_seconds\":" << elapsed(start)
                  << ",\"prefix_t_count\":" << boundaries.optimized().num_t_gates()
                  << ",\"num_measurements\":135,\"num_detectors\":72,\"num_observables\":9}"
                  << std::endl;
        for (std::string header; std::getline(std::cin, header);) {
            std::istringstream input(header);
            std::string name, flips, extra;
            uint32_t shots = 0, check = 0;
            uint64_t seed = 0;
            size_t phase_count = 0, decoder_count = 0;
            if (!(input >> name >> shots >> seed >> check >> phase_count >> decoder_count >>
                  flips) ||
                input >> extra || shots == 0 || shots > 65536 || check > 1 || flips.size() != 135 ||
                flips.find_first_not_of("01") != std::string::npos ||
                flips.substr(126) != std::string(9, '0') ||
                name.find_first_not_of("abcdefghijklmnopqrstuvwxyz_0123456789") !=
                    std::string::npos)
                throw std::invalid_argument("Invalid scored sampling request");
            start = Clock::now();
            const auto phase = read_correction(phase_count, false);
            const auto decoder = read_correction(decoder_count, true);
            auto hir = boundaries.compose(boundaries.optimized(), phase, decoder);
            const auto compose_seconds = elapsed(start);
            if (check && !same_hir(boundaries.compose(boundaries.raw(), phase, decoder),
                                   trace(parse(read(directory / (name + ".moved"))))))
                throw std::runtime_error("Scored raw boundary mismatch: " + name);
            if (export_enabled) {
                export_clifford(hir, directory / (name + ".shared.stim"));
                export_clifford(trace(parse(read(directory / (name + ".original")))),
                                directory / (name + ".fresh.stim"));
            }
            const std::vector<uint8_t> detectors(72, 0), observables(9, 0);
            start = Clock::now();
            auto plan = sampling::plan_sampling(
                hir, {.expected_detectors = detectors, .expected_observables = observables});
            const auto plan_seconds = elapsed(start);
            if (plan.peak_active_width > 16)
                throw std::runtime_error("Scored execution exceeds the width budget");
            start = Clock::now();
            const sampling::ExecutablePlan executable(plan);
            const auto prepare_seconds = elapsed(start);
            start = Clock::now();
            auto samples = sampling::sample(executable, shots, seed, 1, std::nullopt, 1);
            const auto sample_seconds = elapsed(start);
            start = Clock::now();
            boundaries.restore(samples, flips);
            const auto restore_seconds = elapsed(start);
            std::cout << "{\"name\":\"" << name << "\",\"shots\":" << shots
                      << ",\"checked\":" << (check ? "true" : "false")
                      << ",\"peak_width\":" << plan.peak_active_width
                      << ",\"t_count\":" << hir.num_t_gates()
                      << ",\"compose_seconds\":" << compose_seconds
                      << ",\"plan_seconds\":" << plan_seconds
                      << ",\"prepare_seconds\":" << prepare_seconds
                      << ",\"sample_seconds\":" << sample_seconds
                      << ",\"restore_seconds\":" << restore_seconds
                      << ",\"peak_rss_kib\":" << peak_rss_kib() << ",\"measurements\":\""
                      << bit_string(samples.measurements) << "\",\"detectors\":\""
                      << bit_string(samples.detectors) << "\",\"observables\":\""
                      << bit_string(samples.observables) << "\"}" << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
