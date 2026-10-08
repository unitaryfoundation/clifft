// Offline boundary composition: reuse one optimized prefix, then plan each
// fixed correction and decoder before ordinary allocation-free execution.

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

void optimize(HirModule& hir) {
    for (const auto* name : {"PeepholeFusionPass", "PhasePolynomialPass",
                             "RotationSimplificationPass", "StatevectorSqueezePass"}) {
        make_hir_pass(name)->run(hir);
    }
}

void append(HirModule& dest, const HirModule& source, const HeisenbergOp& op,
            uint32_t hidden_shift = 0, const Tableau* pullback = nullptr) {
    const auto record = [&](uint32_t index) {
        return index < source.num_measurements ? index : index + hidden_shift;
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
        case OpType::EXP_VAL:
            dest.append_exp_val(op.exp_val_idx(), fill);
            break;
        default:
            throw std::invalid_argument("Operation outside this diagnostic's supported subset");
    }
}

bool same_hir(const HirModule& a, const HirModule& b) {
    if (a.final_tableau != b.final_tableau || a.ops.size() != b.ops.size() ||
        a.num_qubits != b.num_qubits || a.num_measurements != b.num_measurements ||
        a.num_hidden_measurements != b.num_hidden_measurements ||
        a.num_exp_vals != b.num_exp_vals || a.num_detectors != b.num_detectors ||
        a.detector_targets != b.detector_targets)
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

class Boundary {
  public:
    explicit Boundary(const fs::path& directory)
        : raw_(trace(parse(read(directory / "prefix.stim")))),
          full_(trace(parse(read(directory / "full.stim")))),
          optimized_(raw_) {
        if (raw_.num_qubits != full_.num_qubits || raw_.num_detectors || raw_.num_observables ||
            full_.num_observables || !raw_.noise_sites.empty() || !full_.noise_sites.empty() ||
            !full_.readout_noise.empty() || !full_.instrument_sites.empty() ||
            raw_.num_hidden_measurements != full_.num_hidden_measurements ||
            raw_.num_measurements > full_.num_measurements || !raw_.final_tableau ||
            !full_.final_tableau) {
            throw std::invalid_argument("Unsupported boundary interface");
        }
        optimize(optimized_);
        for (size_t i = raw_.ops.size(); i < full_.ops.size(); ++i) {
            if (full_.ops[i].op_type() == OpType::T_GATE) {
                throw std::invalid_argument("Non-Clifford operation after the boundary");
            }
        }
        // The standalone prefix renumbers hidden records after its visible
        // records. Verify that restoring their full-program offsets is exact.
        if (!same_hir(compose(raw_, Tableau(raw_.num_qubits)), full_)) {
            throw std::runtime_error("Raw prefix does not match the full-program boundary");
        }
    }

    HirModule compose(const HirModule& prefix, const Tableau& correction) const {
        HirModule result(full_.num_qubits, prefix.ops.size() + full_.ops.size());
        result.num_measurements = full_.num_measurements;
        result.num_hidden_measurements = full_.num_hidden_measurements;
        result.num_detectors = full_.num_detectors;
        result.num_exp_vals = full_.num_exp_vals;
        result.detector_targets = full_.detector_targets;
        const auto shift = full_.num_measurements - raw_.num_measurements;
        for (const auto& op : prefix.ops)
            append(result, prefix, op, shift);
        // If F and G are the raw and optimized boundary frames, a suffix
        // Pauli in F coordinates pulls back by G^-1 C^-1 F. The full output
        // frame becomes F_full F^-1 C G. This uses physical C, not a frame
        // inferred from separately optimized faulty programs.
        const auto pullback =
            raw_.final_tableau->then(correction.inverse()).then(prefix.final_tableau->inverse());
        for (size_t i = raw_.ops.size(); i < full_.ops.size(); ++i) {
            append(result, full_, full_.ops[i], 0, &pullback);
        }
        result.final_tableau = prefix.final_tableau->then(correction)
                                   .then(raw_.final_tableau->inverse())
                                   .then(*full_.final_tableau);
        return result;
    }

    const HirModule& raw() const { return raw_; }
    const HirModule& optimized() const { return optimized_; }

  private:
    HirModule raw_, full_, optimized_;
};

sampling::SamplingPlan plan(const HirModule& hir) {
    const std::vector<uint8_t> reference(hir.num_detectors, 0);
    return sampling::plan_sampling(hir, {.expected_detectors = reference});
}

void write_samples(const fs::path& path, const sampling::SamplingResult& result) {
    // Same-host temporary interchange only; Python checks byte counts and
    // dimensions. No binary outputs are retained as research artifacts.
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(result.measurements.data()),
              result.measurements.size());
    out.write(reinterpret_cast<const char*>(result.detectors.data()), result.detectors.size());
    out.write(reinterpret_cast<const char*>(result.observables.data()), result.observables.size());
    out.write(reinterpret_cast<const char*>(result.exp_vals.data()),
              result.exp_vals.size() * sizeof(double));
    if (!out)
        throw std::runtime_error("Cannot write sample output");
}

std::pair<std::string, size_t> prefix_description(const sampling::SamplingPlan& plan) {
    size_t end = 0;
    for (size_t i = 0; i < plan.actions.size(); ++i) {
        const auto& action = plan.actions[i].action;
        if (std::holds_alternative<sampling::RotateActivePauli>(action) ||
            std::holds_alternative<sampling::PromoteDormantRotation>(action))
            end = i + 1;
    }
    std::string description;
    for (size_t i = 0; i < end; ++i)
        description += plan.inspect_action(i) + "\n";
    return {description, end};
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("Usage: profile_bt27_boundary_reuse DIRECTORY SHOTS");
        const fs::path directory(argv[1]);
        const auto shots = static_cast<uint32_t>(std::stoul(argv[2]));
        if (shots < 16 || shots > 65536)
            throw std::invalid_argument("Shot count outside diagnostic budget");
        auto start = Clock::now();
        Boundary boundary(directory);
        const auto setup = elapsed(start);
        const auto prefix_plan = plan(boundary.optimized());
        std::cout << "{\"setup_seconds\":" << setup
                  << ",\"prefix_t_count\":" << boundary.optimized().num_t_gates()
                  << ",\"prefix_peak_width\":" << prefix_plan.peak_active_width
                  << ",\"prefix_visible_records\":" << prefix_plan.num_visible_records << "}"
                  << std::endl;
        std::istringstream names(read(directory / "names.txt"));
        std::string first_prefix;
        size_t index = 0;
        for (std::string name; std::getline(names, name); ++index) {
            if (name.empty() ||
                name.find_first_not_of("abcdefghijklmnopqrstuvwxyzXYZ0123456789_") !=
                    std::string::npos)
                throw std::invalid_argument("Invalid case name");
            const auto stem = directory / std::to_string(index);
            start = Clock::now();
            const auto correction = trace(parse(read(stem.string() + ".correction")));
            if (!correction.ops.empty() || correction.num_qubits != boundary.raw().num_qubits)
                throw std::invalid_argument("Expected a Clifford-only physical correction");
            auto reused = boundary.compose(boundary.optimized(), *correction.final_tableau);
            const auto compose_seconds = elapsed(start);
            // Check the coordinate formula without any optimization before
            // relying on the existing prefix optimizer's state-preservation contract.
            const auto raw_reused = boundary.compose(boundary.raw(), *correction.final_tableau);
            const auto raw_moved = trace(parse(read(stem.string() + ".moved")));
            if (!same_hir(raw_reused, raw_moved))
                throw std::runtime_error("Raw boundary mismatch: " + name);
            start = Clock::now();
            auto fresh = trace(parse(read(stem.string() + ".original")));
            optimize(fresh);
            const auto fresh_seconds = elapsed(start);
            start = Clock::now();
            auto reused_plan = plan(reused);
            const auto plan_seconds = elapsed(start);
            const auto fresh_plan = plan(fresh);
            if (reused_plan.peak_active_width > 16 || fresh_plan.peak_active_width > 16)
                throw std::runtime_error("Execution width exceeds diagnostic budget");
            const auto [description, prefix_actions] = prefix_description(reused_plan);
            if (index == 0)
                first_prefix = description;
            std::ofstream(stem.string() + ".plan") << reused_plan.inspect();
            std::ofstream(stem.string() + ".fresh.plan") << fresh_plan.inspect();
            start = Clock::now();
            const sampling::ExecutablePlan executable(reused_plan);
            const auto prepare_seconds = elapsed(start);
            const sampling::ExecutablePlan reference(fresh_plan);
            start = Clock::now();
            auto samples = sampling::sample(executable, shots, 213791, 1, std::nullopt, 1);
            const auto sample_seconds = elapsed(start);
            write_samples(stem.string() + ".shared.bin", samples);
            start = Clock::now();
            const auto fresh_samples =
                sampling::sample(reference, shots, 198331, 1, std::nullopt, 1);
            const auto fresh_sample_seconds = elapsed(start);
            write_samples(stem.string() + ".fresh.bin", fresh_samples);
            std::cout << "{\"name\":\"" << name
                      << "\",\"raw_boundary_matches\":true,\"t_count\":" << reused.num_t_gates()
                      << ",\"peak_width\":" << reused_plan.peak_active_width
                      << ",\"fresh_peak_width\":" << fresh_plan.peak_active_width
                      << ",\"prefix_actions\":" << prefix_actions
                      << ",\"prefix_matches\":" << (description == first_prefix ? "true" : "false")
                      << ",\"actions\":" << reused_plan.actions.size()
                      << ",\"compose_seconds\":" << compose_seconds
                      << ",\"fresh_compile_seconds\":" << fresh_seconds
                      << ",\"plan_seconds\":" << plan_seconds
                      << ",\"prepare_seconds\":" << prepare_seconds
                      << ",\"sample_seconds\":" << sample_seconds
                      << ",\"fresh_sample_seconds\":" << fresh_sample_seconds << "}" << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
