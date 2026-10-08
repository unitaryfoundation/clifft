// Research diagnostic: reconstruct optimized variants before ordinary planning.
// Every reconstruction is checked against a fresh compiler result; inferred
// frame composition is not a production certificate or an executor operation.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/hir_introspection.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace clifft;
using Clock = std::chrono::steady_clock;
using Fault = std::pair<uint32_t, char>;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string read_file(const char* path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open input file");
    }
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

struct Case {
    std::string name;
    std::vector<Fault> faults;
};

std::vector<Case> read_cases(const char* path) {
    std::istringstream input(read_file(path));
    std::vector<Case> result;
    for (std::string line; std::getline(input, line);) {
        std::istringstream row(line);
        Case entry;
        if (!(row >> entry.name)) {
            continue;
        }
        if (entry.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzXYZ0123456789_") !=
            std::string::npos) {
            throw std::invalid_argument("Unsupported case name");
        }
        for (std::string fault; row >> fault;) {
            if (fault.size() < 2 || std::string("XYZ").find(fault[0]) == std::string::npos ||
                fault.find_first_not_of("0123456789", 1) != std::string::npos) {
                throw std::invalid_argument("Fault must be a Pauli followed by a qubit index");
            }
            const auto q = std::stoul(fault.substr(1));
            if (q >= 81) {
                throw std::invalid_argument("Fault is outside the pinned data layer");
            }
            entry.faults.emplace_back(static_cast<uint32_t>(q), fault[0]);
        }
        result.push_back(std::move(entry));
    }
    return result;
}

// T_DAG differs from T by a Clifford rotation on the same axis. Moving that
// exact correction into the frame removes a presentation choice from comparison.
void canonicalize_t(HirModule& hir) {
    optimizer_detail::CliffordAbsorption correction;
    for (auto& op : hir.ops) {
        correction.transform(hir, op);
        if (op.op_type() == OpType::T_GATE && op.is_dagger()) {
            const auto axis = optimizer_detail::copy_axis(hir.mask_view(op), hir.num_qubits);
            hir.demote_to_tgate(op, false);
            correction.absorb(axis, 6);
        }
    }
    correction.finish(hir);
}

HirModule phase_compile(const std::string& source) {
    auto hir = trace(parse(source));
    make_hir_pass("PeepholeFusionPass")->run(hir);
    make_hir_pass("PhasePolynomialPass")->run(hir);
    canonicalize_t(hir);
    if (!hir.final_tableau || !hir.noise_sites.empty() || !hir.instrument_sites.empty() ||
        !hir.readout_noise.empty()) {
        throw std::invalid_argument("The diagnostic requires fixed Pauli faults and a final frame");
    }
    return hir;
}

void finish_optimization(HirModule& hir) {
    make_hir_pass("RotationSimplificationPass")->run(hir);
    make_hir_pass("StatevectorSqueezePass")->run(hir);
}

bool same_semantics(const HirModule& a, const HirModule& b, bool include_frame = true) {
    if (a.num_qubits != b.num_qubits || a.num_measurements != b.num_measurements ||
        a.num_hidden_measurements != b.num_hidden_measurements ||
        a.num_detectors != b.num_detectors || a.num_observables != b.num_observables ||
        a.num_exp_vals != b.num_exp_vals || a.detector_targets != b.detector_targets ||
        a.observable_targets != b.observable_targets ||
        a.neglect_instrument_damping != b.neglect_instrument_damping ||
        a.forced_traceout_slot != b.forced_traceout_slot ||
        (include_frame && a.final_tableau != b.final_tableau) || a.ops.size() != b.ops.size() ||
        a.logical_noise_prefix != b.logical_noise_prefix) {
        return false;
    }
    // Both modules have no remaining noise or instruments. Compare operation
    // payloads, signed masks, and hidden flags without relying on arena handles.
    for (size_t i = 0; i < a.ops.size(); ++i) {
        const auto& x = a.ops[i];
        const auto& y = b.ops[i];
        const auto mx = x.has_mask() ? std::optional{a.mask_view(x)} : std::nullopt;
        const auto my = y.has_mask() ? std::optional{b.mask_view(y)} : std::nullopt;
        if (x.flags() != y.flags() || format_hir_op(x, mx) != format_hir_op(y, my) ||
            (x.op_type() == OpType::PHASE_ROTATION && x.alpha() != y.alpha())) {
            return false;
        }
    }
    return true;
}

class Study {
  public:
    explicit Study(std::string source) : source_(std::move(source)) {
        const auto marker = source_.find("\nT ");
        if (marker == std::string::npos) {
            throw std::invalid_argument("No T region found");
        }
        insertion_ = marker + 1;
        std::istringstream lines(source_);
        uint32_t line_number = 0;
        for (std::string line; std::getline(lines, line);) {
            ++line_number;
            if (line.starts_with("T ") || line.starts_with("T_DAG ")) {
                last_t_line_ = line_number;
            }
        }
        base_ = phase_compile(source_);
        for (size_t i = 0; i < base_.ops.size(); ++i) {
            if (!base_.ops[i].has_mask() || base_.source_map[i].empty()) {
                continue;
            }
            const auto [first, last] =
                std::minmax_element(base_.source_map[i].begin(), base_.source_map[i].end());
            if (*first <= last_t_line_ && *last > last_t_line_) {
                throw std::invalid_argument("Operation mixes phase and suffix provenance");
            }
            if (*first > last_t_line_) {
                suffix_masks_.push_back(i);
            }
        }
    }

    std::string specialized(const std::vector<Fault>& faults) const {
        std::string inserted;
        for (const auto& [q, pauli] : faults) {
            inserted += pauli + std::string(" ") + std::to_string(q) + "\n";
        }
        return source_.substr(0, insertion_) + inserted + source_.substr(insertion_);
    }

    void prepare_generators() {
        for (uint32_t q = 0; q < 81; ++q) {
            for (const char pauli : {'X', 'Z'}) {
                auto variant = phase_compile(specialized({{q, pauli}}));
                auto correction = base_.final_tableau->then(variant.final_tableau->inverse());
                if (correction.then(correction) != Tableau(base_.num_qubits)) {
                    ++non_involutions_;
                }
                generators_.emplace(Fault{q, pauli}, std::move(correction));
            }
        }
    }

    HirModule reconstruct(const std::vector<Fault>& faults) const {
        Tableau correction(base_.num_qubits);
        for (const auto& [q, pauli] : faults) {
            if (pauli == 'X' || pauli == 'Y') {
                correction = correction.then(generators_.at({q, 'X'}));
            }
            if (pauli == 'Z' || pauli == 'Y') {
                correction = correction.then(generators_.at({q, 'Z'}));
            }
        }
        auto result = base_;
        for (const auto i : suffix_masks_) {
            const auto axis =
                optimizer_detail::copy_axis(result.mask_view(result.ops[i]), result.num_qubits);
            const auto updated = correction.apply(axis.view());
            optimizer_detail::write_axis(result.mask_at(result.ops[i]), updated);
        }
        result.final_tableau = correction.inverse().then(*base_.final_tableau);
        // Provenance belongs to the original no-fault input, not the injected
        // source text. Clear it rather than report shifted lines as exact.
        result.source_map.clear();
        return result;
    }

    size_t suffix_masks() const { return suffix_masks_.size(); }
    size_t non_involutions() const { return non_involutions_; }

  private:
    std::string source_;
    size_t insertion_ = 0;
    uint32_t last_t_line_ = 0;
    HirModule base_;
    std::vector<size_t> suffix_masks_;
    std::map<Fault, Tableau> generators_;
    size_t non_involutions_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument("Usage: profile_bt27_shared_analysis FIXTURE CASES");
        }
        const auto cases = read_cases(argv[2]);
        const auto setup_start = Clock::now();
        Study study(read_file(argv[1]));
        study.prepare_generators();
        const auto setup = seconds(setup_start);
        std::cout << "{\"generator_setup_seconds\":" << setup
                  << ",\"generator_count\":162,\"suffix_masks\":" << study.suffix_masks()
                  << ",\"non_involutive_generators\":" << study.non_involutions()
                  << ",\"cases\":[\n";
        size_t index = 0;
        for (const auto& entry : cases) {
            auto start = Clock::now();
            auto rebuilt = study.reconstruct(entry.faults);
            const auto reconstruct_time = seconds(start);
            start = Clock::now();
            auto fresh = phase_compile(study.specialized(entry.faults));
            const auto fresh_time = seconds(start);
            if (!same_semantics(rebuilt, fresh)) {
                if (index++) {
                    std::cout << ",\n";
                }
                std::cout << "{\"name\":\"" << entry.name
                          << "\",\"matches\":false,\"frame_matches\":"
                          << (rebuilt.final_tableau == fresh.final_tableau ? "true" : "false")
                          << ",\"operations_match\":"
                          << (same_semantics(rebuilt, fresh, false) ? "true" : "false")
                          << ",\"reconstruct_seconds\":" << reconstruct_time
                          << ",\"fresh_phase_seconds\":" << fresh_time << "}";
                continue;
            }
            start = Clock::now();
            finish_optimization(rebuilt);
            const auto tail_time = seconds(start);
            finish_optimization(fresh);
            if (!same_semantics(rebuilt, fresh)) {
                throw std::runtime_error("Completed HIR differs for " + entry.name);
            }
            start = Clock::now();
            auto plan = sampling::plan_sampling(rebuilt);
            const auto plan_time = seconds(start);
            const auto reference = sampling::plan_sampling(fresh);
            if (plan.inspect() != reference.inspect()) {
                throw std::runtime_error("Semantic plans differ for " + entry.name);
            }
            start = Clock::now();
            const sampling::ExecutablePlan executable(plan);
            const auto prepare_time = seconds(start);
            if (index++) {
                std::cout << ",\n";
            }
            std::cout << "{\"name\":\"" << entry.name
                      << "\",\"peak_width\":" << plan.peak_active_width
                      << ",\"matches\":true,\"reconstruct_seconds\":" << reconstruct_time
                      << ",\"fresh_phase_seconds\":" << fresh_time
                      << ",\"remaining_passes_seconds\":" << tail_time
                      << ",\"plan_seconds\":" << plan_time
                      << ",\"prepare_seconds\":" << prepare_time << "}";
            if (index % 32 == 0) {
                std::cerr << "Checked " << index << " reconstructed variants\n";
            }
        }
        std::cout << "\n]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
