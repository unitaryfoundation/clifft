// Exact small-circuit checks for runtime Clifford frames and for T regions
// whose rotation signs depend on the shot. A Pauli fault, a readout flip on
// feedback, or a random stabilizer outcome that crosses a T-family rotation
// changes the trajectory by a Clifford:
// exp(-i pi/8 (-1)^t P) = exp(+i pi/4 t P) exp(-i pi/8 P). Each case compares
// the complete joint distribution of a sampling plan against an independent
// dense calculation.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/numeric.h"
#include "clifft/util/xoshiro.h"

#include "dense_circuit_oracle.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace clifft::test {
namespace {

using sampling::AffineBool;
using sampling::FramePauli;
using sampling::PlannedAction;
using sampling::RecordSlot;
using sampling::SamplingPlan;
using sampling::SymbolId;
using sampling::SymbolKind;

// One step of a small program written directly on frame coordinates. A
// rotation applies exp(-i * pi * half_turns * P / 2). Its parity is the XOR of
// the listed earlier records and noise bits: a Clifford angle with a parity is
// applied only when the parity is one, and any other angle is negated by it.
struct FrameStep {
    enum class Kind : uint8_t { Rotation, Measurement };

    Kind kind = Kind::Rotation;
    uint64_t x = 0;
    uint64_t z = 0;
    double half_turns = 0.0;
    std::vector<uint32_t> records{};
    std::vector<uint32_t> noise{};

    [[nodiscard]] bool has_parity() const { return !records.empty() || !noise.empty(); }
};

struct FrameProgram {
    uint32_t num_qubits = 0;
    // Each noise bit is an independent Bernoulli variable with this probability.
    std::vector<double> noise_probabilities{};
    std::vector<FrameStep> steps{};
};

FrameStep rotate(uint64_t x, uint64_t z, double half_turns, std::vector<uint32_t> records = {},
                 std::vector<uint32_t> noise = {}) {
    return {FrameStep::Kind::Rotation, x, z, half_turns, std::move(records), std::move(noise)};
}

FrameStep measure(uint64_t x, uint64_t z) {
    return {FrameStep::Kind::Measurement, x, z, 0.0, {}, {}};
}

// Lowers a program to a plan whose frame covers every coordinate from the
// first action, so all quantum work goes through frame actions.
SamplingPlan build_frame_plan(const FrameProgram& program) {
    SamplingPlan plan;
    plan.num_qubits = program.num_qubits;
    plan.frame_width = program.num_qubits;
    for (double probability : program.noise_probabilities) {
        const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
        plan.symbols.push_back(SymbolKind::Presampled);
        plan.presampled_noise_sites.push_back({probability, {{symbol, probability}}});
    }
    std::vector<AffineBool> record_values;
    uint32_t width = 0;
    for (const FrameStep& step : program.steps) {
        AffineBool parity;
        for (uint32_t record : step.records) {
            parity ^= record_values.at(record);
        }
        for (uint32_t bit : step.noise) {
            parity ^= AffineBool::symbol(SymbolId{bit});
        }
        FramePauli pauli{{step.x}, {step.z}};
        if (step.kind == FrameStep::Kind::Measurement) {
            const SymbolId branch{static_cast<uint32_t>(plan.symbols.size())};
            plan.symbols.push_back(SymbolKind::Branch);
            const RecordSlot record{static_cast<uint32_t>(record_values.size())};
            record_values.push_back(AffineBool::symbol(branch));
            plan.actions.push_back({width, width,
                                    sampling::MeasureFramePauli{std::move(pauli), branch,
                                                                record_values.back(), record}});
        } else if (const auto clifford = classify_clifford_rotation(step.half_turns)) {
            if (*clifford == CliffordRotation::IDENTITY) {
                continue;
            }
            plan.actions.push_back(
                {width, width,
                 sampling::ApplyConditionalClifford{
                     std::move(pauli), *clifford, step.has_parity() ? parity : AffineBool(true)}});
        } else {
            const uint32_t after = std::min(width + 1, program.num_qubits);
            plan.actions.push_back(
                {width, after,
                 sampling::RotateFramePauli{std::move(pauli), step.half_turns, std::move(parity)}});
            width = after;
        }
    }
    plan.num_visible_records = static_cast<uint32_t>(record_values.size());
    plan.peak_active_width = width;
    return plan;
}

// Evaluates the same program with a 2^n statevector, branching on each
// measurement and summing over the noise bits.
ExactPlanDistribution dense_frame_distribution(const FrameProgram& program) {
    using namespace detail_dense_oracle;
    uint32_t num_records = 0;
    for (const FrameStep& step : program.steps) {
        num_records += step.kind == FrameStep::Kind::Measurement;
    }
    ExactPlanDistribution distribution;
    distribution.num_visible_records = num_records;
    const size_t num_noise = program.noise_probabilities.size();
    for (size_t noise = 0; noise < (size_t{1} << num_noise); ++noise) {
        double weight = 1.0;
        for (size_t bit = 0; bit < num_noise; ++bit) {
            const double p = program.noise_probabilities[bit];
            weight *= ((noise >> bit) & 1) != 0 ? p : 1.0 - p;
        }
        TrajectoryBranch initial;
        initial.psi.assign(size_t{1} << program.num_qubits, {0.0, 0.0});
        initial.psi[0] = {1.0, 0.0};
        initial.probability = weight;
        initial.records.assign(num_records, 0);
        std::vector<TrajectoryBranch> branches{std::move(initial)};
        for (const FrameStep& step : program.steps) {
            const PauliSpec spec{step.x, step.z, false};
            if (step.kind == FrameStep::Kind::Measurement) {
                std::vector<TrajectoryBranch> next;
                for (const TrajectoryBranch& branch : branches) {
                    branch_projective_measurement(branch, program.num_qubits, spec, true, false,
                                                  nullptr, next, 1e-18,
                                                  distribution.omitted_probability_bound);
                }
                branches = std::move(next);
                continue;
            }
            const bool is_clifford = classify_clifford_rotation(step.half_turns).has_value();
            for (TrajectoryBranch& branch : branches) {
                bool parity = false;
                for (uint32_t record : step.records) {
                    parity ^= branch.records.at(record) != 0;
                }
                for (uint32_t bit : step.noise) {
                    parity ^= ((noise >> bit) & 1) != 0;
                }
                if (is_clifford && step.has_parity() && !parity) {
                    continue;
                }
                apply_pauli_rotation(branch.psi, program.num_qubits, spec,
                                     !is_clifford && parity ? -step.half_turns : step.half_turns);
            }
        }
        for (const TrajectoryBranch& branch : branches) {
            uint64_t bits = 0;
            for (uint32_t record = 0; record < num_records; ++record) {
                bits |= uint64_t{branch.records[record]} << record;
            }
            distribution.kept_outcomes.push_back({{bits, 0, 0, {}}, branch.probability});
        }
    }
    canonicalize_outcome_entries(distribution.kept_outcomes);
    return distribution;
}

void check_frame_program(const FrameProgram& program) {
    const SamplingPlan plan = build_frame_plan(program);
    require_distributions_equal(dense_frame_distribution(program),
                                compute_exact_plan_distribution(plan));
}

constexpr uint64_t kQ0 = 1;
constexpr uint64_t kQ1 = 2;

// CX with control c and target t, up to a global phase, as three commuting
// Clifford rotations.
void append_cx(std::vector<FrameStep>& steps, uint64_t control, uint64_t target) {
    steps.push_back(rotate(0, control, 0.5));
    steps.push_back(rotate(target, 0, 0.5));
    steps.push_back(rotate(target, control, 1.5));
}

// Gate teleportation of T onto exp(-i pi theta Y / 2)|0>: entangle with the
// magic state T|+>, measure it, and apply S when the outcome is one. Undoing T
// and the preparation then returns the data qubit to |0> on every shot.
FrameProgram t_teleportation(double theta, bool apply_correction) {
    FrameProgram program;
    program.num_qubits = 2;
    auto& steps = program.steps;
    steps.push_back(rotate(kQ0, kQ0, theta));
    steps.push_back(rotate(kQ1, kQ1, 0.5));
    steps.push_back(rotate(0, kQ1, 0.25));
    append_cx(steps, kQ0, kQ1);
    steps.push_back(measure(0, kQ1));
    if (apply_correction) {
        steps.push_back(rotate(0, kQ0, 0.5, {0}));
    }
    steps.push_back(rotate(0, kQ0, -0.25));
    steps.push_back(rotate(kQ0, kQ0, -theta));
    steps.push_back(measure(0, kQ0));
    return program;
}

FrameProgram random_frame_program(Xoshiro256PlusPlus& rng) {
    const auto below = [&](uint32_t bound) {
        return static_cast<uint32_t>(rng.next_double() * bound);
    };
    FrameProgram program;
    program.num_qubits = 1 + below(4);
    program.noise_probabilities.assign(below(3), 0.0);
    for (double& probability : program.noise_probabilities) {
        probability = 0.1 + 0.5 * rng.next_double();
    }
    const uint64_t mask = (uint64_t{1} << program.num_qubits) - 1;
    const uint32_t num_steps = 6 + below(14);
    uint32_t num_records = 0;
    for (uint32_t i = 0; i < num_steps; ++i) {
        uint64_t x = 0;
        uint64_t z = 0;
        while ((x | z) == 0) {
            x = below(16) & mask;
            z = below(16) & mask;
        }
        if (num_records < 7 && below(4) == 0) {
            program.steps.push_back(measure(x, z));
            ++num_records;
            continue;
        }
        constexpr double kAngles[] = {0.5, 1.0, 1.5, 0.5, 1.5, 0.25, -0.25, 0.137};
        FrameStep step = rotate(x, z, kAngles[below(8)]);
        if (below(2) == 0) {
            if (num_records != 0 && below(2) == 0) {
                step.records.push_back(below(num_records));
            }
            if (!program.noise_probabilities.empty() && (step.records.empty() || below(2) == 0)) {
                step.noise.push_back(
                    below(static_cast<uint32_t>(program.noise_probabilities.size())));
            }
        }
        program.steps.push_back(std::move(step));
    }
    return program;
}

struct SignedRegionCase {
    std::string name;
    std::string source;
    // One flag per detector; nonzero entries are postselected.
    std::vector<uint8_t> postselection_mask{};
};

std::vector<SignedRegionCase> signed_region_cases() {
    return {
        {"bit and phase faults before a CCZ",
         "H 0 1 2\n"
         "X_ERROR(0.1) 0\n"
         "Z_ERROR(0.2) 1\n"
         "CCZ 0 1 2\n"
         "H 0 1 2\n"
         "M 0 1 2\n"},
        {"faults between overlapping CCZ gates and a T layer",
         "H 0 1 2 3\n"
         "CCZ 0 1 2\n"
         "DEPOLARIZE1(0.06) 1\n"
         "CCZ 1 2 3\n"
         "X_ERROR(0.1) 3\n"
         "T 0 1 2 3\n"
         "H 0 1 2 3\n"
         "M 0 1 2 3\n"},
        // Z0*Z1 stabilizes the reference state, so T 0 and T_DAG 1 cancel
        // there. The fault flips that stabilizer and leaves a Clifford.
        {"fault turns a cancelling T pair into a Clifford",
         "H 0\n"
         "CX 0 1\n"
         "X_ERROR(0.3) 0\n"
         "T 0\n"
         "T_DAG 1\n"
         "MPP X0*X1\n"
         "MPP Y0*X1\n"
         "M 0 1\n"},
        // The reference rotations merge into S. A fault between them flips the
        // second sign, so they cancel instead.
        {"fault between two T gates on one axis",
         "H 0\n"
         "T 0\n"
         "X_ERROR(0.25) 0\n"
         "T 0\n"
         "H 0\n"
         "M 0\n"},
        {"random stabilizer outcome without feedback",
         "H 0 1\n"
         "MPP Z0*Z1\n"
         "T 0\n"
         "T_DAG 1\n"
         "H 0 1\n"
         "M 0 1\n"},
        {"readout flip on Pauli feedback",
         "H 0 2\n"
         "CX 0 1\n"
         "M 1\n"
         "READOUT_NOISE(0.2, 0.3) rec[-1]\n"
         "CX rec[-1] 0\n"
         "TPP Z0*Z2\n"
         "T_DAG 2\n"
         "H 2\n"
         "M 2 0\n"},
        // Qubit 3 never enters the region, X0*X4 joins a region qubit to an
        // outside one, and the repeated Z0*Z1 is fixed by its first outcome.
        {"measurements inside and outside the region",
         "H 0 1 2 3 4\n"
         "X_ERROR(0.15) 1\n"
         "CCZ 0 1 2\n"
         "M 3\n"
         "MPP X0*X4\n"
         "MPP Z0*Z1\n"
         "MX 2\n"
         "MPP Z0*Z1\n"
         "M 0 1 4\n"},
        {"second region after measurement and reset",
         "H 0 1 2\n"
         "X_ERROR(0.1) 0\n"
         "CCZ 0 1 2\n"
         "MX 0 1 2\n"
         "R 0 1 2\n"
         "H 0 1 2\n"
         "Z_ERROR(0.1) 1\n"
         "CCZ 0 1 2\n"
         "H 0 1 2\n"
         "M 0 1 2\n"},
        // Without the fault the CCZ and both T gates act trivially. With it
        // the CCZ leaves CZ on qubits 1 and 2, and the T gates after the
        // Hadamards are no longer Clifford.
        {"non-commuting second region on live coordinates",
         "H 1 2\n"
         "X_ERROR(0.3) 0\n"
         "CCZ 0 1 2\n"
         "H 1 2\n"
         "T 1 2\n"
         "H 1 2\n"
         "M 0 1 2\n"},
        {"region entered with an active coordinate",
         "H 0 1\n"
         "T 0\n"
         "H 0\n"
         "X_ERROR(0.2) 1\n"
         "CX 1 0\n"
         "T 0 1\n"
         "TPP Z0*Z1\n"
         "H 0 1\n"
         "M 0 1\n"},
        {"mixed-basis axes with a two-qubit fault",
         "H 0 1 2\n"
         "S 1\n"
         "DEPOLARIZE2(0.3) 0 1\n"
         "TPP X0*Y1\n"
         "TPP_DAG X0*Y1*Z2\n"
         "T_DAG 2\n"
         "Y_ERROR(0.1) 2\n"
         "TPP Z2\n"
         "M 0\n"
         "MX 1 2\n"},
        {"correlated fault across two regions",
         "H 0 1 2 3\n"
         "E(0.2) X0 Z2\n"
         "ELSE_CORRELATED_ERROR(0.3) Y1 X3\n"
         "CCZ 0 1 2\n"
         "T 3\n"
         "H 0 1 2 3\n"
         "M 0 1 2 3\n"},
        // The Bell pair's T and T_DAG cancel on the reference state, while the
        // T on qubit 2 does not, so the region keeps one T and one frame-only
        // coordinate.
        {"cancelling pair beside an irreducible T",
         "H 0 2\n"
         "CX 0 1\n"
         "X_ERROR(0.2) 0\n"
         "Z_ERROR(0.3) 2\n"
         "T 0\n"
         "T_DAG 1\n"
         "T 2\n"
         "X_ERROR(0.1) 2\n"
         "MPP X0*X1*X2\n"
         "MPP Y0*X1\n"
         "MX 2\n"
         "M 0 1\n"},
        // Four T rotations on a GHZ state leave S on the reference state.
        {"GHZ state with a Clifford remainder",
         "H 0\n"
         "CX 0 1 0 2 0 3\n"
         "DEPOLARIZE1(0.15) 1\n"
         "X_ERROR(0.2) 3\n"
         "T 0 1\n"
         "T_DAG 2\n"
         "T 3\n"
         "MPP X0*X1*X2*X3\n"
         "MPP Y0*X1*X2*X3\n"
         "M 0 1 2 3\n"},
        // Qubit 3 is untouched until its T, which then adds a coordinate to
        // the live frame. Qubit 1 is rotated again in a different basis.
        {"rotations after the frame is live",
         "H 0 3\n"
         "CX 0 1\n"
         "X_ERROR(0.25) 1\n"
         "T 0\n"
         "T_DAG 1\n"
         "H 1\n"
         "T 1 3\n"
         "R_Z(0.3) 0\n"
         "CX 3 0\n"
         "H 0 1\n"
         "M 0 1\n"
         "MX 3\n"},
        {"feedback from a frame measurement",
         "H 0 2\n"
         "CX 0 1\n"
         "X_ERROR(0.3) 0\n"
         "T 0\n"
         "T_DAG 1\n"
         "MX 0\n"
         "CZ rec[-1] 1\n"
         "CX rec[-1] 2\n"
         "T 2\n"
         "MX 1 2\n"
         "DETECTOR rec[-2] rec[-3]\n"},
        {"postselected detector after a signed region",
         "H 0 1 2\n"
         "X_ERROR(0.2) 0\n"
         "DEPOLARIZE1(0.1) 2\n"
         "CCZ 0 1 2\n"
         "H 0 1 2\n"
         "M 0 1 2\n"
         "DETECTOR rec[-1] rec[-2]\n"
         "DETECTOR rec[-3]\n"
         "OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]\n",
         {1, 0}},
    };
}

}  // namespace

TEST_CASE("Signed T regions match the dense oracle", "[runtime_frame][sampling]") {
    for (const SignedRegionCase& test : signed_region_cases()) {
        DYNAMIC_SECTION(test.name) {
            const Circuit circuit = clifft::parse(test.source);
            const sampling::SamplingPlanOptions options{.postselection_mask =
                                                            test.postselection_mask};
            const ExactPlanDistribution expected =
                compute_dense_circuit_distribution(circuit, options);
            for (const bool optimized : {false, true}) {
                CAPTURE(optimized);
                HirModule hir = clifft::trace(circuit);
                if (optimized) {
                    auto passes = default_hir_pass_manager();
                    passes.run(hir);
                }
                for (const bool runtime_frame : {false, true}) {
                    CAPTURE(runtime_frame);
                    sampling::SamplingPlanOptions planned = options;
                    planned.runtime_frame = runtime_frame;
                    require_distributions_equal(expected,
                                                compute_exact_hir_distribution(hir, planned));
                }
            }
        }
    }
}

TEST_CASE("Frame planning replaces signed promotions with conditional Cliffords",
          "[runtime_frame][sampling]") {
    struct Expected {
        std::string source;
        uint32_t static_peak;
        uint32_t frame_peak;
        uint32_t frame_width;
        size_t conditional_cliffords;
    };
    const std::vector<Expected> cases = {
        // The reference rotations cancel, so no coordinate becomes active.
        {"H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nM 0 1\n", 1, 0, 1, 1},
        // One T survives; the Bell pair contributes a frame-only coordinate.
        {"H 0 2\nCX 0 1\nX_ERROR(0.2) 0\nZ_ERROR(0.3) 2\nT 0\nT_DAG 1\nT 2\nM 0 1 2\n", 2, 1, 2, 1},
        // Both sign flips come from one fault, but they rotate about different axes.
        {"H 0 1\nCX 0 2 1 3\nE(0.2) X0 X1\nT 0\nT_DAG 2\nT 1\nT_DAG 3\nM 0 1 2 3\n", 2, 0, 2, 2},
    };
    for (const Expected& test : cases) {
        CAPTURE(test.source);
        const HirModule hir = clifft::trace(clifft::parse(test.source));
        const SamplingPlan fixed = sampling::plan_sampling(hir);
        const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
        REQUIRE(fixed.frame_width == 0);
        REQUIRE(fixed.peak_active_width == test.static_peak);
        REQUIRE(frame.peak_active_width == test.frame_peak);
        REQUIRE(frame.frame_width == test.frame_width);
        REQUIRE(static_cast<size_t>(std::ranges::count_if(frame.actions, [](const auto& planned) {
                    return std::holds_alternative<sampling::ApplyConditionalClifford>(
                        planned.action);
                })) == test.conditional_cliffords);
    }
}

TEST_CASE("Frame planning reduces a region without sign flips and adds no frame",
          "[runtime_frame][sampling]") {
    // Nothing flips a sign here, so the cancelling pair needs neither an
    // active coordinate nor conditional Cliffords.
    const Circuit circuit = clifft::parse("H 0\nCX 0 1\nT 0\nT_DAG 1\nMPP X0*X1\nM 0 1\n");
    const HirModule hir = clifft::trace(circuit);
    const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
    REQUIRE(sampling::plan_sampling(hir).peak_active_width == 1);
    REQUIRE(frame.peak_active_width == 0);
    REQUIRE(frame.frame_width == 0);
    require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                compute_exact_plan_distribution(frame));
}

TEST_CASE("Frame planning keeps the static plan when it cannot help", "[runtime_frame][sampling]") {
    const std::vector<std::string> sources = {
        // The reference CCZ already needs every coordinate.
        "H 0 1 2\nX_ERROR(0.1) 0\nCCZ 0 1 2\nH 0 1 2\nM 0 1 2\n",
        // Expectation probes need the static coordinates.
        "H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nEXP_VAL X0*X1\nM 0 1\n",
    };
    for (const std::string& source : sources) {
        CAPTURE(source);
        const HirModule hir = clifft::trace(clifft::parse(source));
        const SamplingPlan fixed = sampling::plan_sampling(hir);
        const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
        REQUIRE(frame.frame_width == 0);
        REQUIRE(frame.peak_active_width == fixed.peak_active_width);
        REQUIRE(frame.inspect() == fixed.inspect());
    }
}

// A stabilizer-state preparation, then layers of commuting diagonal T
// rotations with faults between them, separated by basis changes. The result
// ends with a measurement of every qubit when measure_all is set.
std::string random_signed_region_source(Xoshiro256PlusPlus& rng, uint32_t num_qubits,
                                        bool measure_all = true) {
    const auto below = [&](uint32_t bound) { return static_cast<uint32_t>(rng() % bound); };
    const auto qubit = [&] { return std::to_string(below(num_qubits)); };
    const auto pair = [&] {
        const uint32_t a = below(num_qubits);
        const uint32_t b = (a + 1 + below(num_qubits - 1)) % num_qubits;
        return std::to_string(a) + " " + std::to_string(b);
    };
    std::string source;
    uint32_t faults = 0;
    const auto fault = [&] {
        if (faults == 5 || below(2) == 0) {
            return;
        }
        ++faults;
        source += std::string(below(2) == 0 ? "X_ERROR" : "Z_ERROR") + "(0.2) " + qubit() + "\n";
    };
    const auto cliffords = [&](uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            switch (below(3)) {
                case 0:
                    source += "H " + qubit() + "\n";
                    break;
                case 1:
                    source += "S " + qubit() + "\n";
                    break;
                default:
                    source += "CX " + pair() + "\n";
                    break;
            }
        }
    };
    for (uint32_t q = 0; q < num_qubits; ++q) {
        if (below(2) == 0) {
            source += "H " + std::to_string(q) + "\n";
        }
    }
    cliffords(2 * num_qubits);
    const uint32_t layers = 1 + below(2);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const uint32_t rotations = 2 + below(2 * num_qubits);
        for (uint32_t i = 0; i < rotations; ++i) {
            fault();
            switch (below(4)) {
                case 0:
                    source += "T " + qubit() + "\n";
                    break;
                case 1:
                    source += "T_DAG " + qubit() + "\n";
                    break;
                case 2: {
                    const uint32_t a = below(num_qubits);
                    const uint32_t b = (a + 1 + below(num_qubits - 1)) % num_qubits;
                    source += std::string(below(2) == 0 ? "TPP" : "TPP_DAG") + " Z" +
                              std::to_string(a) + "*Z" + std::to_string(b) + "\n";
                    break;
                }
                default:
                    source += "CCZ " + std::to_string(below(num_qubits)) + " " +
                              std::to_string((below(num_qubits))) + " " + qubit() + "\n";
                    break;
            }
        }
        if (below(3) == 0) {
            source += "MPP Z" + std::to_string(below(num_qubits)) + "\n";
        }
        cliffords(1 + below(num_qubits));
    }
    for (uint32_t q = 0; measure_all && q < num_qubits; ++q) {
        source += std::string(below(2) == 0 ? "M " : "MX ") + std::to_string(q) + "\n";
    }
    return source;
}

// The generator can repeat a CCZ target, which the parser rejects.
Circuit parse_random_signed_region(Xoshiro256PlusPlus& rng, uint32_t num_qubits, bool measure_all,
                                   const std::function<std::string()>& suffix,
                                   std::string& source) {
    while (true) {
        source = random_signed_region_source(rng, num_qubits, measure_all) + suffix();
        try {
            return clifft::parse(source);
        } catch (const ParseError&) {}
    }
}

TEST_CASE("Random signed regions match the dense oracle on a runtime frame",
          "[runtime_frame][sampling]") {
    Xoshiro256PlusPlus rng(0x704F);
    uint32_t frame_plans = 0;
    uint32_t narrower_plans = 0;
    for (uint32_t trial = 0; trial < 250; ++trial) {
        std::string source;
        const Circuit circuit = parse_random_signed_region(
            rng, 2 + trial % 4, true, [] { return std::string(); }, source);
        CAPTURE(trial, source);
        const HirModule hir = clifft::trace(circuit);
        const SamplingPlan fixed = sampling::plan_sampling(hir);
        const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
        frame_plans += frame.frame_width != 0;
        narrower_plans += frame.peak_active_width < fixed.peak_active_width;
        require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                    compute_exact_plan_distribution(frame));
    }
    // The generator must reach the lowering often enough to test it.
    REQUIRE(frame_plans >= 60);
    REQUIRE(narrower_plans >= 30);
}

TEST_CASE("Operations after a signed region agree between static and frame plans",
          "[runtime_frame][sampling]") {
    // The suffix mixes rotations, resets, readout noise, and feedback, so
    // these run while a frame is live on wider registers than the dense
    // oracle supports.
    Xoshiro256PlusPlus rng(0x704A);
    uint32_t frame_plans = 0;
    for (uint32_t trial = 0; trial < 250; ++trial) {
        const uint32_t num_qubits = 3 + trial % 6;
        std::string source;
        const Circuit circuit = parse_random_signed_region(
            rng, num_qubits, false,
            [&] { return generate_noisy_source(rng, num_qubits, 8 + trial % 12); }, source);
        CAPTURE(trial, source);
        const HirModule hir = clifft::trace(circuit);
        // Exhaustive replay enumerates every record and readout flip.
        if (hir.num_measurements > 10 ||
            hir.num_measurements + hir.num_hidden_measurements + hir.readout_noise.size() > 13) {
            continue;
        }
        const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
        frame_plans += frame.frame_width != 0;
        check_exact_equivalent(sampling::plan_sampling(hir), frame, rng);
    }
    REQUIRE(frame_plans >= 40);
}

TEST_CASE("Frame plan teleports T with a record-conditioned S", "[runtime_frame][sampling]") {
    for (const double theta : {0.5, 0.3}) {
        CAPTURE(theta);
        const FrameProgram corrected = t_teleportation(theta, true);
        check_frame_program(corrected);
        const ExactPlanDistribution law =
            compute_exact_plan_distribution(build_frame_plan(corrected));
        // The magic-state outcome is unbiased and the data qubit always returns to |0>.
        REQUIRE(law.kept_outcomes.size() == 2);
        for (uint64_t outcome : {0, 1}) {
            REQUIRE(law.kept_outcomes[outcome].outcome.visible_records == outcome);
            REQUIRE_THAT(law.kept_outcomes[outcome].probability,
                         Catch::Matchers::WithinAbs(0.5, 1e-12));
        }

        // Without the correction the outcome-one branch keeps a leftover S_DAG.
        const FrameProgram uncorrected = t_teleportation(theta, false);
        check_frame_program(uncorrected);
        REQUIRE(
            compute_exact_plan_distribution(build_frame_plan(uncorrected)).kept_outcomes.size() ==
            3);
    }
}

TEST_CASE("Conditional Clifford equals a sign flip on a T rotation", "[runtime_frame][sampling]") {
    // exp(-i pi/8 (-1)^t P) = exp(+i pi/4 t P) exp(-i pi/8 P) for a noise bit t.
    const uint64_t x = 0b011;
    const uint64_t z = 0b110;
    FrameProgram prefix;
    prefix.num_qubits = 3;
    prefix.noise_probabilities = {0.3};
    prefix.steps = {rotate(0b001, 0b001, 0.5), rotate(0b010, 0b010, 0.2),
                    rotate(0b100, 0b001, 0.5)};
    const std::vector<FrameStep> suffix = {rotate(0b111, 0, 0.4), measure(0b001, 0),
                                           measure(0b010, 0b100), measure(0, 0b010)};

    FrameProgram flipped = prefix;
    flipped.steps.push_back(rotate(x, z, 0.25, {}, {0}));
    FrameProgram split = prefix;
    split.steps.push_back(rotate(x, z, 0.25));
    split.steps.push_back(rotate(x, z, 1.5, {}, {0}));
    for (FrameProgram* program : {&flipped, &split}) {
        program->steps.insert(program->steps.end(), suffix.begin(), suffix.end());
        check_frame_program(*program);
    }
    require_distributions_equal(compute_exact_plan_distribution(build_frame_plan(flipped)),
                                compute_exact_plan_distribution(build_frame_plan(split)));
}

TEST_CASE("Random frame programs match dense evolution", "[runtime_frame][sampling]") {
    Xoshiro256PlusPlus rng(20261113);
    for (uint32_t trial = 0; trial < 300; ++trial) {
        CAPTURE(trial);
        check_frame_program(random_frame_program(rng));
    }
}

TEST_CASE("Growing coefficient storage does not change a seeded frame shot",
          "[runtime_frame][sampling]") {
    FrameProgram program;
    program.num_qubits = 3;
    program.steps = {
        rotate(0b001, 0, 0.137),     rotate(0b010, 0, 0.3),  rotate(0b110, 0b001, 0.25),
        measure(0b011, 0b100),       rotate(0b100, 0, 0.21), measure(0b100, 0b100),
        rotate(0b001, 0b010, 0.137), measure(0, 0b001),      measure(0b010, 0)};
    const SamplingPlan lazy = build_frame_plan(program);

    // The same plan after three promotions and three measurements with fixed
    // outcomes. They draw nothing from the RNG and leave |000>, but raise the
    // storage allocated at construction to the width the frame later needs.
    SamplingPlan eager = lazy;
    eager.num_hidden_records = 3;
    eager.peak_active_width = 3;
    std::vector<PlannedAction> prefix;
    for (uint32_t width = 0; width < 3; ++width) {
        prefix.push_back({width, width + 1, sampling::PromoteDormantRotation{0.0, AffineBool{}}});
    }
    for (uint32_t width = 3; width > 0; --width) {
        const SymbolId branch{static_cast<uint32_t>(eager.symbols.size())};
        eager.symbols.push_back(SymbolKind::Branch);
        prefix.push_back({width, width - 1,
                          sampling::MeasureActivePauli{
                              sampling::ActivePauli{0, 1}, 0, branch, AffineBool::symbol(branch),
                              RecordSlot{lazy.num_visible_records + width - 1}}});
    }
    eager.actions.insert(eager.actions.begin(), prefix.begin(), prefix.end());

    const sampling::ExecutablePlan lazy_executable(lazy);
    const sampling::ExecutablePlan eager_executable(eager);
    REQUIRE(lazy_executable.initial_capacity_width() == 0);
    REQUIRE(eager_executable.initial_capacity_width() == 3);
    sampling::Executor lazy_executor(lazy_executable, 99);
    sampling::Executor eager_executor(eager_executable, 99);
    REQUIRE(lazy_executor.state().max_active_width() == 0);
    bool saw_one = false;
    for (uint32_t shot = 0; shot < 200; ++shot) {
        CAPTURE(shot);
        lazy_executor.run_shot();
        eager_executor.run_shot();
        REQUIRE(
            std::ranges::equal(lazy_executor.visible_records(), eager_executor.visible_records()));
        saw_one |= std::ranges::count(lazy_executor.visible_records(), uint8_t{1}) != 0;
    }
    REQUIRE(saw_one);
    REQUIRE(lazy_executor.state().max_active_width() > 0);
    REQUIRE(eager_executor.state().max_active_width() == 3);
}

TEST_CASE("Sampled frame plans follow their exact distribution", "[runtime_frame][sampling]") {
    Xoshiro256PlusPlus rng(704);
    constexpr uint32_t kShots = 8000;
    for (uint32_t trial = 0; trial < 8; ++trial) {
        CAPTURE(trial);
        const SamplingPlan plan = build_frame_plan(random_frame_program(rng));
        const ExactPlanDistribution exact = compute_exact_plan_distribution(plan);
        std::map<uint64_t, double> expected;
        for (const ExactOutcomeEntry& entry : exact.kept_outcomes) {
            expected[entry.outcome.visible_records] = entry.probability;
        }

        const sampling::ExecutablePlan executable(plan);
        sampling::Executor executor(executable, 1000 + trial);
        std::map<uint64_t, uint32_t> counts;
        for (uint32_t shot = 0; shot < kShots; ++shot) {
            executor.run_shot();
            uint64_t bits = 0;
            for (size_t record = 0; record < executor.visible_records().size(); ++record) {
                bits |= uint64_t{executor.visible_records()[record]} << record;
            }
            ++counts[bits];
        }
        for (const auto& [bits, count] : counts) {
            CAPTURE(bits);
            REQUIRE(expected.contains(bits));
        }
        for (const auto& [bits, probability] : expected) {
            CAPTURE(bits, probability);
            const double observed = counts.contains(bits) ? double(counts[bits]) / kShots : 0.0;
            const double sigma = std::sqrt(probability * (1.0 - probability) / kShots);
            REQUIRE(std::abs(observed - probability) <= 6.0 * sigma + 1e-9);
        }
    }
}

}  // namespace clifft::test
