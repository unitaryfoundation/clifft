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
#include "clifft/sampling/sampler.h"
#include "clifft/sampling/state_queries.h"
#include "clifft/util/numeric.h"
#include "clifft/util/xoshiro.h"

#include "dense_circuit_oracle.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
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

// The same plan after `width` promotions and `width` measurements with fixed
// outcomes. They draw nothing from the RNG and leave |0...0>, but raise the
// storage allocated at construction to the width the frame later needs.
SamplingPlan with_storage_from_the_start(const SamplingPlan& lazy, uint32_t width) {
    SamplingPlan eager = lazy;
    eager.num_hidden_records = width;
    eager.peak_active_width = width;
    std::vector<PlannedAction> prefix;
    for (uint32_t active = 0; active < width; ++active) {
        prefix.push_back({active, active + 1, sampling::PromoteDormantRotation{0.0, AffineBool{}}});
    }
    for (uint32_t active = width; active > 0; --active) {
        const SymbolId branch{static_cast<uint32_t>(eager.symbols.size())};
        eager.symbols.push_back(SymbolKind::Branch);
        prefix.push_back({active, active - 1,
                          sampling::MeasureActivePauli{
                              sampling::ActivePauli{0, 1}, 0, branch, AffineBool::symbol(branch),
                              RecordSlot{lazy.num_visible_records + active - 1}}});
    }
    eager.actions.insert(eager.actions.begin(), prefix.begin(), prefix.end());
    return eager;
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
        // On two pairs the frame Paulis X_a*X_b and Y_a*Y_b are measured as
        // Z0*Z1 and Y0*X2*Y1*X3. Their product is -Z_a*Z_b, so the stabilizers
        // that retire the frame carry a negative sign, which the X checks then
        // read back.
        {"checks whose product is negative fix a frame",
         "H 0 1\nCX 0 2\nCX 1 3\nX_ERROR(0.3) 0\nX_ERROR(0.2) 1\nT 0 1\nT_DAG 2 3\n"
         "MPP Z0*Z1\nMPP Y0*X2*Y1*X3\nMPP X0*X2*X1*X3\nM 0 1 2 3\n"},
        // The same with a third pair. Measuring X on one of its qubits replaces
        // a stabilizer outside the frame before that pair's check completes
        // the frame, and the negative stabilizer must come through it.
        {"a negative stabilizer is kept across a measurement outside the frame",
         "H 0 1 4\nCX 0 2\nCX 1 3\nCX 4 5\nX_ERROR(0.3) 0\nX_ERROR(0.2) 1\nX_ERROR(0.25) 4\n"
         "T 0 1 4\nT_DAG 2 3 5\nMPP Z0*Z1\nMPP Y0*X2*Y1*X3\nMX 4\nMPP X4*X5\n"
         "MPP X0*X2*X1*X3\nM 0 1 2 3 4 5\n"},
        // The two checks fix both of those coordinates with a negative product
        // while a third pair keeps the frame live. The T gate starts a region,
        // where the plan releases the two coordinates; the X check then reads
        // them as static coordinates.
        {"released coordinates carry a negative stabilizer",
         "H 0 1 4\nCX 0 2\nCX 1 3\nCX 4 5\nX_ERROR(0.3) 0\nX_ERROR(0.2) 1\nX_ERROR(0.25) 4\n"
         "T 0 1 4\nT_DAG 2 3 5\nMPP Z0*Z1\nMPP Y0*X2*Y1*X3\nT 4\n"
         "MPP X0*X2*X1*X3\nM 0 1 2 3 4 5\n"},
        // Measuring qubit 0 fixes its coordinate while the pair keeps the
        // frame live, and the next region releases it. A T pair about Y on
        // that qubit then cancels and disturbs nothing, so a stabilizer the
        // plan still held after the release would be released a second time
        // where the last T on qubit 0 starts a region.
        {"a released stabilizer is forgotten",
         "H 4\nCX 4 5\nX_ERROR(0.3) 4\nT 4\nT_DAG 5\nH 0\nT 0\nH 0\nM 0\n"
         "S_DAG 0\nH 0\nT 0\nT_DAG 0\nH 0\nS 0\nH 0\nT 0\nH 0\nM 0\nMPP X4*X5\nM 4 5\n"},
        // The check of the first pair is known until the T gate rotates about
        // an axis that anticommutes with it. The second check must not then
        // count as fixing the whole frame: the first check, repeated, has a
        // random outcome again.
        {"a rotation between two checks disturbs the first",
         "H 0 2\nCX 0 1\nCX 2 3\nX_ERROR(0.3) 0\nX_ERROR(0.2) 2\nT 0 2\nT_DAG 1 3\n"
         "MPP X0*X1\nT 0\nMPP X2*X3\nMPP X0*X1\nM 0 1 2 3\n"},
        // The same with a rotation that is no T gate and so joins no region.
        {"a phase rotation between two checks disturbs the first",
         "H 0 2\nCX 0 1\nCX 2 3\nX_ERROR(0.3) 0\nX_ERROR(0.2) 2\nT 0 2\nT_DAG 1 3\n"
         "MPP X0*X1\nR_Z(0.3) 0\nMPP X2*X3\nMPP X0*X1\nM 0 1 2 3\n"},
        // The same, where a measurement that reaches an idle qubit outside the
        // frame anticommutes with the first check.
        {"a check through an idle qubit disturbs an earlier one",
         "H 0 2\nCX 0 1\nCX 2 3\nX_ERROR(0.3) 0\nX_ERROR(0.2) 2\nT 0 2\nT_DAG 1 3\n"
         "MPP X0*X1\nMPP Z0*X4\nMPP X2*X3\nM 0 1 2 3 4\n"},
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
        // Qubits 2 and 3 are untouched while the first pair's frame is live,
        // so their region is reduced as well and adds no active coordinate.
        {"second factory on fresh qubits",
         "H 0 2\n"
         "CX 0 1 2 3\n"
         "X_ERROR(0.3) 0\n"
         "T 0\n"
         "T_DAG 1\n"
         "MPP X0*X1\n"
         "X_ERROR(0.2) 2\n"
         "T 2\n"
         "T_DAG 3\n"
         "MPP X2*X3\n"
         "MPP Y2*X3\n"
         "M 0 1 2 3\n"},
        // The second factory keeps one T on qubit 4 and folds two T gates on
        // qubit 5 into S, both while the first factory's frame is live.
        {"second factory with a T core and a Clifford remainder",
         "H 0 2 4 5\n"
         "CX 0 1 2 3\n"
         "X_ERROR(0.3) 0\n"
         "T 0\n"
         "T_DAG 1\n"
         "MPP Y0*X1\n"
         "X_ERROR(0.2) 3\n"
         "T 2\n"
         "T_DAG 3\n"
         "T 4\n"
         "T 5\n"
         "X_ERROR(0.25) 5\n"
         "T 5\n"
         "H 5\n"
         "MPP X2*X3*X4\n"
         "M 5\n"
         "MX 4\n"
         "M 0 2\n"},
        // The record is always one, so the feedback X is part of the reference
        // trajectory: it makes Z1 = -Z2 there, and the two T gates cancel.
        {"feedback from a record that is always one",
         "X 0\n"
         "M 0\n"
         "H 1\n"
         "CX 1 2\n"
         "CX rec[-1] 1\n"
         "X_ERROR(0.3) 2\n"
         "T 1 2\n"
         "MPP X1*X2\n"
         "MPP Y1*X2\n"
         "M 1 2\n"},
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

TEST_CASE("Sign flips about one Pauli under one condition compose", "[runtime_frame][sampling]") {
    // On a Bell pair Z0 and Z1 are the same frame Pauli. A cancelling pair of
    // T gates opens the region, and a fault on qubit 0 then flips the sign of
    // every later T gate on that qubit. Each flipped sign leaves a quarter
    // turn, and quarter turns under one condition add: two make a Pauli for
    // the symbolic frame and four make nothing.
    struct Expected {
        const char* rotations;
        size_t conditional_cliffords;
    };
    const Expected cases[] = {
        {"T 0\nT_DAG 1\n", 1},
        {"T 0\nT 0\n", 0},
        {"T 0\nT 0\nT 0\nT_DAG 1\n", 1},
        {"T 0\nT 0\nT 0\nT 0\n", 0},
        {"T 0\nT_DAG 0\n", 0},
        {"T_DAG 0\nT_DAG 0\nT 1\nT 1\n", 0},
        // An X gate adds a constant to every later sign. The half turn two
        // T gates then leave applies on the fault alone.
        {"X 0\nT 0\nT 0\n", 0},
        {"X 0\nT 0\nT 0\nT 0\nT_DAG 1\n", 1},
        // A second fault between the gates gives them different conditions.
        {"T 0\nX_ERROR(0.2) 0\nT 0\n", 2},
        {"T 0\nT 0\nX_ERROR(0.2) 1\nT 0\nT 1\n", 2},
    };
    for (const Expected& test : cases) {
        CAPTURE(test.rotations);
        const Circuit circuit =
            clifft::parse(std::string("H 0\nCX 0 1\nT 0\nT_DAG 1\nX_ERROR(0.3) 0\n") +
                          test.rotations + "MPP Y0*X1\nMPP X0*X1\nM 0 1\n");
        const SamplingPlan frame =
            sampling::plan_sampling(clifft::trace(circuit), {.runtime_frame = true});
        REQUIRE(static_cast<size_t>(std::ranges::count_if(frame.actions, [](const auto& planned) {
                    return std::holds_alternative<sampling::ApplyConditionalClifford>(
                        planned.action);
                })) == test.conditional_cliffords);
        require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                    compute_exact_plan_distribution(frame));
    }

    // Twenty-two faults before the region give its signs a long entry sign,
    // which conditional Cliffords name with a derived symbol. They act on one
    // qubit of the pair, so the two T gates there compose into a Pauli that
    // the symbolic frame takes under the written-out sign.
    std::string source = "H 0\nCX 0 1\n";
    for (uint32_t i = 0; i < 22; ++i) {
        source += "X_ERROR(0.05) 0\n";
    }
    source += "T 0\nT 0\nT 1\nT 1\nMPP Y0*X1\nMPP X0*X1\nM 0 1\n";
    const HirModule hir = clifft::trace(clifft::parse(source));
    const SamplingPlan fixed = sampling::plan_sampling(hir);
    const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
    REQUIRE(fixed.peak_active_width == 1);
    REQUIRE(frame.peak_active_width == 0);
    REQUIRE(std::ranges::none_of(frame.actions, [](const PlannedAction& planned) {
        return std::holds_alternative<sampling::ApplyConditionalClifford>(planned.action);
    }));
    Xoshiro256PlusPlus rng(0x2B);
    check_exact_equivalent(fixed, frame, rng, 96);
}

TEST_CASE("Quarter-turn phase rotations join a signed region", "[runtime_frame][sampling]") {
    // trace() leaves a parameterized quarter turn as a phase rotation. Each
    // spelling plans like the T gates it equals up to global phase.
    const std::string before = "H 0 2\nCX 0 1\nX_ERROR(0.2) 0\nZ_ERROR(0.3) 2\n";
    const std::string after = "H 0 1 2\nM 0 1 2\n";
    const SamplingPlan t_gates = sampling::plan_sampling(
        clifft::trace(clifft::parse(before + "T 0\nT_DAG 1\nTPP Z1*Z2\n" + after)),
        {.runtime_frame = true});
    REQUIRE(t_gates.frame_width != 0);
    const std::vector<std::string> spellings = {
        "R_Z(0.25) 0\nR_Z(-0.25) 1\nR_ZZ(0.25) 1 2\n",
        // The same angles modulo two half-turns.
        "R_Z(2.25) 0\nR_Z(1.75) 1\nR_PAULI(-3.75) Z1*Z2\n",
    };
    for (const std::string& rotations : spellings) {
        CAPTURE(rotations);
        const Circuit circuit = clifft::parse(before + rotations + after);
        const HirModule hir = clifft::trace(circuit);
        REQUIRE(hir.num_t_gates() == 0);
        const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
        REQUIRE(frame.inspect() == t_gates.inspect());
        require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                    compute_exact_plan_distribution(frame));
    }
}

TEST_CASE("Frame planning reduces a later region on fresh coordinates",
          "[runtime_frame][sampling]") {
    const auto count = [](const SamplingPlan& plan, auto matches) {
        return std::ranges::count_if(plan.actions, matches);
    };
    const auto is_conditional = [](const PlannedAction& planned) {
        return std::holds_alternative<sampling::ApplyConditionalClifford>(planned.action);
    };
    const auto is_frame_rotation = [](const PlannedAction& planned) {
        return std::holds_alternative<sampling::RotateFramePauli>(planned.action);
    };
    const std::string first = "H 0 2\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nMPP X0*X1\n";

    // Qubits 2 and 3 are outside the live frame: the pair cancels again.
    const SamplingPlan fresh = sampling::plan_sampling(
        clifft::trace(clifft::parse(first + "CX 2 3\nX_ERROR(0.2) 2\nT 2\nT_DAG 3\nM 0 1 2 3\n")),
        {.runtime_frame = true});
    REQUIRE(fresh.peak_active_width == 0);
    REQUIRE(count(fresh, is_conditional) == 2);
    REQUIRE(count(fresh, is_frame_rotation) == 0);

    // A second pair that no measurement resolves keeps the frame live. The
    // check of the first pair fixes its frame coordinate, which the plan
    // releases where the next region starts.
    const std::string two_pairs =
        "H 0 2 4\nCX 0 1\nCX 4 5\nX_ERROR(0.3) 0 4\n"
        "T 0 4\nT_DAG 1 5\nMPP X0*X1\n";
    const auto is_release = [](const PlannedAction& planned) {
        return std::holds_alternative<sampling::ReleaseFrameCoordinate>(planned.action);
    };
    struct Later {
        const char* entangle;
        std::ptrdiff_t conditionals;
        std::ptrdiff_t frame_rotations;
        uint32_t peak;
    };
    // Entangled with qubit 1 of the checked pair, the later pair is on
    // released coordinates and cancels as a fresh one does. Entangled with
    // qubit 5 of the other pair, it touches the live coordinate and keeps its
    // signed rotations.
    for (const Later& later : {Later{"CX 1 2\n", 3, 0, 0}, Later{"CX 5 2\n", 2, 2, 2}}) {
        CAPTURE(later.entangle);
        const Circuit circuit = clifft::parse(
            two_pairs + later.entangle + "CX 2 3\nX_ERROR(0.2) 2\nT 2\nT_DAG 3\nM 0 1 2 3 4 5\n");
        const SamplingPlan plan =
            sampling::plan_sampling(clifft::trace(circuit), {.runtime_frame = true});
        REQUIRE(count(plan, is_release) == 1);
        REQUIRE(count(plan, is_conditional) == later.conditionals);
        REQUIRE(count(plan, is_frame_rotation) == later.frame_rotations);
        REQUIRE(plan.peak_active_width == later.peak);
        require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                    compute_exact_plan_distribution(plan));
    }
}

TEST_CASE("Frame planning retires a frame once measurements fix its state",
          "[runtime_frame][sampling]") {
    const auto count = [](const SamplingPlan& plan, auto matches) {
        return std::ranges::count_if(plan.actions, matches);
    };
    const auto holds = [&]<typename Action>(const SamplingPlan& plan) {
        return count(plan, [](const PlannedAction& planned) {
            return std::holds_alternative<Action>(planned.action);
        });
    };
    // The check of the first pair fixes the only frame coordinate, so the
    // frame ends before the second pair, which reuses qubit 1 and is reduced
    // from the reference state again.
    const std::string pair = "H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nMPP X0*X1\n";
    const SamplingPlan reused = sampling::plan_sampling(
        clifft::trace(clifft::parse(pair + "H 2\nCX 1 2\nCX 2 3\nX_ERROR(0.2) 2\nT 2\nT_DAG 3\n"
                                           "M 0 1 2 3\n")),
        {.runtime_frame = true});
    REQUIRE(reused.peak_active_width == 0);
    REQUIRE(holds.operator()<sampling::ApplyConditionalClifford>(reused) == 2);
    REQUIRE(holds.operator()<sampling::RotateFramePauli>(reused) == 0);
    const auto index_of = [&](auto matches, int occurrence) {
        auto it = reused.actions.begin();
        for (int found = 0;; ++it) {
            it = std::find_if(it, reused.actions.end(), matches);
            REQUIRE(it != reused.actions.end());
            if (++found == occurrence) {
                return it - reused.actions.begin();
            }
        }
    };
    const auto first_retirement = index_of(
        [](const PlannedAction& planned) {
            return std::holds_alternative<sampling::RetireFrame>(planned.action);
        },
        1);
    const auto second_clifford = index_of(
        [](const PlannedAction& planned) {
            return std::holds_alternative<sampling::ApplyConditionalClifford>(planned.action);
        },
        2);
    REQUIRE(first_retirement < second_clifford);

    // Three rounds on the same two qubits, each ending in a reset: the frame
    // is retired before every later round.
    std::string rounds;
    for (int round = 0; round < 3; ++round) {
        rounds += "H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nCX 0 1\nH 0\nMR 0 1\n";
    }
    const SamplingPlan repeated =
        sampling::plan_sampling(clifft::trace(clifft::parse(rounds)), {.runtime_frame = true});
    REQUIRE(repeated.peak_active_width == 0);
    REQUIRE(repeated.frame_width == 1);
    REQUIRE(holds.operator()<sampling::RetireFrame>(repeated) >= 2);
    REQUIRE(holds.operator()<sampling::ApplyConditionalClifford>(repeated) == 3);
    // A round's correction reads its own fault and nothing an earlier round
    // left in the Pauli frame.
    for (const PlannedAction& planned : repeated.actions) {
        if (const auto* turn = std::get_if<sampling::ApplyConditionalClifford>(&planned.action)) {
            REQUIRE(turn->condition.terms().size() == 1);
        }
    }
    REQUIRE(holds.operator()<sampling::RotateFramePauli>(repeated) == 0);
    // The ordinary plan needs an active qubit in each round.
    REQUIRE(sampling::plan_sampling(clifft::trace(clifft::parse(rounds))).peak_active_width == 1);

    // A measurement that reaches an idle qubit outside the frame anticommutes
    // with the first pair's check. The static frame absorbs it without
    // changing the frame's state, so that check stays known, the second one
    // completes the frame, and the later T gates on the first pair are reduced.
    const Circuit through_idle = clifft::parse(
        "H 0 2\nCX 0 1\nCX 2 3\nX_ERROR(0.3) 0\nX_ERROR(0.2) 2\nT 0 2\n"
        "T_DAG 1 3\nMPP X0*X1\nMPP Z0*X4\nMPP X2*X3\n"
        "T 0\nT_DAG 1\nM 0 1 2 3 4\n");
    const SamplingPlan carried =
        sampling::plan_sampling(clifft::trace(through_idle), {.runtime_frame = true});
    REQUIRE(holds.operator()<sampling::RetireFrame>(carried) >= 1);
    REQUIRE(holds.operator()<sampling::RotateFramePauli>(carried) == 0);
    require_distributions_equal(compute_dense_circuit_distribution(through_idle),
                                compute_exact_plan_distribution(carried));

    // Static work after a retirement counts toward the coefficient storage an
    // executor allocates: the frame leaves none, and three plain T gates then
    // need three active qubits.
    const Circuit widening = clifft::parse(pair + "MR 0 1\nH 0 1 2\nT 0 1 2\nH 0 1 2\nM 0 1 2\n");
    const SamplingPlan wider =
        sampling::plan_sampling(clifft::trace(widening), {.runtime_frame = true});
    REQUIRE(holds.operator()<sampling::RetireFrame>(wider) >= 1);
    REQUIRE(wider.peak_active_width == 3);
    REQUIRE(sampling::ExecutablePlan(wider).initial_capacity_width() == 3);
    require_distributions_equal(compute_dense_circuit_distribution(widening),
                                compute_exact_plan_distribution(wider));
}

TEST_CASE("A live frame reduces the fresh rotations of a region", "[runtime_frame][sampling]") {
    // The first pair starts a frame. After a basis change, T gates on its two
    // qubits share a commuting region with a second signed pair. One of the
    // two acts on the frame's coordinate and keeps its sign. The rest of the
    // region is on coordinates that are still |0> and is reduced as before:
    // the second pair cancels into one conditional Clifford.
    const Circuit circuit = clifft::parse(
        "H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nH 0 1\n"
        "H 2\nCX 2 3\nX_ERROR(0.2) 2\nT 0 1 2\nT_DAG 3\n"
        "MPP X2*X3\nH 0 1\nM 0 1 2 3\n");
    const SamplingPlan frame =
        sampling::plan_sampling(clifft::trace(circuit), {.runtime_frame = true});
    const auto count = [&]<typename Action>() {
        return std::ranges::count_if(frame.actions, [](const PlannedAction& planned) {
            return std::holds_alternative<Action>(planned.action);
        });
    };
    REQUIRE(count.operator()<sampling::ApplyConditionalClifford>() == 2);
    REQUIRE(count.operator()<sampling::RotateFramePauli>() == 2);
    REQUIRE(frame.peak_active_width == 2);
    require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                compute_exact_plan_distribution(frame));
}

TEST_CASE("Long entry signs are defined once and shared", "[runtime_frame][sampling]") {
    // Twenty-two faults flip the stabilizer of each Bell pair before the
    // region. Every pair gets one derived symbol for that history instead of
    // repeating it in every condition, and the two product rotations reuse
    // both.
    std::string source = "H 0 2\nCX 0 1\nCX 2 3\n";
    for (const char* qubit : {"0", "2"}) {
        for (uint32_t i = 0; i < 22; ++i) {
            source += std::string("X_ERROR(0.05) ") + qubit + "\n";
        }
    }
    source +=
        "T 0 2\nT_DAG 1 3\nX_ERROR(0.2) 2\nTPP Z0*Z2\nTPP_DAG Z1*Z3\n"
        "MPP X0*X1\nMPP X2*X3\nM 0 1 2 3\n";
    const HirModule hir = clifft::trace(clifft::parse(source));
    const SamplingPlan fixed = sampling::plan_sampling(hir);
    const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
    REQUIRE(frame.peak_active_width < fixed.peak_active_width);

    size_t definitions = 0;
    size_t conditionals = 0;
    for (const PlannedAction& planned : frame.actions) {
        definitions += std::holds_alternative<sampling::DefineSymbol>(planned.action);
        if (const auto* clifford =
                std::get_if<sampling::ApplyConditionalClifford>(&planned.action)) {
            ++conditionals;
            REQUIRE(clifford->condition.terms().size() <= 4);
        }
    }
    REQUIRE(definitions == 2);
    REQUIRE(conditionals == 4);

    Xoshiro256PlusPlus rng(0x1D);
    check_exact_equivalent(fixed, frame, rng, 96);
    // Count-only compaction must keep the shared definitions usable.
    const sampling::ExecutablePlan executable(frame);
    const sampling::ExecutablePlan compact = executable.compact_outputs({}, false, true);
    REQUIRE(compact.inspect().find("DEFINE_SYMBOL") != std::string::npos);
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

TEST_CASE("Frame planning keeps the final state when a region reduces to a Clifford",
          "[runtime_frame][sampling]") {
    // Without noise no sign varies, so a region is replaced by its reduced T
    // core and a Clifford correction on the static coordinates. The final
    // tableau has to follow that correction.
    struct Case {
        const char* text;
        uint32_t ordinary_width;
        uint32_t frame_width;
    };
    for (const Case& c : {Case{"H 0\nT 0\nT 0\n", 1, 0}, Case{"H 0\nT 0\nT 0\nH 0\n", 1, 0},
                          Case{"H 0\nCX 0 1\nT 0\nT 1\nH 1\n", 1, 0},
                          Case{"H 0 1\nCX 0 1\nT 0\nT 1\nT 0\nH 0\n", 2, 1},
                          Case{"H 0 1 2\nT 0\nT 1\nT 0\nT 1\nT 2\nCZ 0 2\nH 1\n", 3, 1}}) {
        CAPTURE(c.text);
        const HirModule hir = clifft::trace(clifft::parse(c.text));
        const SamplingPlan ordinary = sampling::plan_sampling(hir);
        const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
        REQUIRE(ordinary.peak_active_width == c.ordinary_width);
        REQUIRE(frame.peak_active_width == c.frame_width);
        REQUIRE(frame.frame_width == 0);
        const auto expected = sampling::get_statevector(sampling::ExecutablePlan(ordinary));
        const auto actual = sampling::get_statevector(sampling::ExecutablePlan(frame));
        REQUIRE(expected.size() == actual.size());
        std::complex<double> overlap = 0.0;
        for (size_t i = 0; i < expected.size(); ++i) {
            overlap += std::conj(expected[i]) * actual[i];
        }
        // Statevectors are defined up to a global phase.
        REQUIRE_THAT(std::abs(overlap), Catch::Matchers::WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("Frame planning keeps the static plan when it cannot help", "[runtime_frame][sampling]") {
    const std::vector<std::string> sources = {
        // The reference CCZ already needs every coordinate.
        "H 0 1 2\nX_ERROR(0.1) 0\nCCZ 0 1 2\nH 0 1 2\nM 0 1 2\n",
        // Expectation probes need the static coordinates.
        "H 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nEXP_VAL X0*X1\nM 0 1\n",
    };
    Xoshiro256PlusPlus rng(0x2F);
    for (const std::string& source : sources) {
        CAPTURE(source);
        const HirModule hir = clifft::trace(clifft::parse(source));
        const SamplingPlan fixed = sampling::plan_sampling(hir);
        const SamplingPlan frame = sampling::plan_sampling(hir, {.runtime_frame = true});
        REQUIRE(frame.frame_width == 0);
        REQUIRE(frame.peak_active_width == fixed.peak_active_width);
        // The actions match. Their signs may be shorter: the X error on |+>
        // does nothing, and the frame planner leaves it out.
        REQUIRE(frame.actions.size() == fixed.actions.size());
        for (size_t i = 0; i < frame.actions.size(); ++i) {
            REQUIRE(frame.actions[i].action.index() == fixed.actions[i].action.index());
        }
        if (hir.num_exp_vals == 0) {
            check_exact_equivalent(fixed, frame, rng, 64);
        }
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

// Rounds of a signed T layer and measurements on the same qubits. A round
// measures a random choice of single qubits, some with a reset, and Pauli
// products, some of them through one further idle qubit, so its frame ends
// with all, some, or none of its state fixed.
std::string random_measured_rounds_source(Xoshiro256PlusPlus& rng, uint32_t num_qubits) {
    const auto below = [&](uint32_t bound) { return static_cast<uint32_t>(rng() % bound); };
    const auto qubit = [&] { return std::to_string(below(num_qubits)); };
    const auto pair = [&](const char* separator) {
        const uint32_t a = below(num_qubits);
        const uint32_t b = (a + 1 + below(num_qubits - 1)) % num_qubits;
        return std::pair{std::to_string(a) + separator, std::to_string(b)};
    };
    // The dense oracle branches on every record.
    constexpr uint32_t kMaxRecords = 9;
    std::string source;
    uint32_t faults = 0;
    uint32_t records = 0;
    const auto cliffords = [&](uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            switch (below(3)) {
                case 0:
                    source += "H " + qubit() + "\n";
                    break;
                case 1:
                    source += "S " + qubit() + "\n";
                    break;
                default: {
                    const auto [a, b] = pair(" ");
                    source += "CX " + a + b + "\n";
                    break;
                }
            }
        }
    };
    const uint32_t rounds = 2 + below(2);
    for (uint32_t round = 0; round < rounds; ++round) {
        for (uint32_t q = 0; q < num_qubits; ++q) {
            if (below(2) == 0) {
                source += "H " + std::to_string(q) + "\n";
            }
        }
        cliffords(num_qubits);
        const uint32_t rotations = 2 + below(num_qubits + 1);
        for (uint32_t i = 0; i < rotations; ++i) {
            // Faults follow a round's first rotation, where fewer of them
            // act on a stabilizer of the state and change nothing. Two a
            // round leave some for every round.
            if (i > 0 && faults < 2 * (round + 1)) {
                ++faults;
                source +=
                    std::string(below(2) == 0 ? "X_ERROR" : "Z_ERROR") + "(0.2) " + qubit() + "\n";
            }
            switch (below(3)) {
                case 0:
                    source += "T " + qubit() + "\n";
                    break;
                case 1:
                    source += "T_DAG " + qubit() + "\n";
                    break;
                default: {
                    const auto [a, b] = pair("*Z");
                    source += std::string(below(2) == 0 ? "TPP" : "TPP_DAG") + " Z" + a + b + "\n";
                    break;
                }
            }
        }
        cliffords(below(num_qubits));
        // Half the rounds end by measuring and resetting every qubit.
        if (below(2) == 0) {
            for (uint32_t q = 0; q < num_qubits && records < kMaxRecords; ++q, ++records) {
                source += "MR " + std::to_string(q) + "\n";
            }
            continue;
        }
        const uint32_t measurements = 1 + below(num_qubits + 1);
        for (uint32_t i = 0; i < measurements && records < kMaxRecords; ++i, ++records) {
            constexpr char kPaulis[] = {'X', 'Y', 'Z'};
            switch (below(6)) {
                case 0:
                    source += "M " + qubit() + "\n";
                    break;
                case 1:
                    source += "MX " + qubit() + "\n";
                    break;
                case 2:
                case 3:
                    source += "MR " + qubit() + "\n";
                    break;
                case 4: {
                    const uint32_t a = below(num_qubits);
                    const uint32_t b = (a + 1 + below(num_qubits - 1)) % num_qubits;
                    source += std::string("MPP ") + kPaulis[below(3)] + std::to_string(a) + "*" +
                              kPaulis[below(3)] + std::to_string(b) + "\n";
                    break;
                }
                default:
                    // Through the idle qubit, which no rotation touches.
                    source += std::string("MPP ") + kPaulis[below(3)] + qubit() + "*" +
                              kPaulis[below(2)] + std::to_string(num_qubits) + "\n";
                    break;
            }
        }
    }
    return source;
}

TEST_CASE("Random measured rounds match the dense oracle on a runtime frame",
          "[runtime_frame][sampling]") {
    Xoshiro256PlusPlus rng(0x706A);
    uint32_t retired = 0;
    uint32_t restarted = 0;
    uint32_t continued = 0;
    for (uint32_t trial = 0; trial < 300; ++trial) {
        const std::string source = random_measured_rounds_source(rng, 2 + trial % 3);
        CAPTURE(trial, source);
        const Circuit circuit = clifft::parse(source);
        const SamplingPlan frame =
            sampling::plan_sampling(clifft::trace(circuit), {.runtime_frame = true});
        // A plan restarts when a conditional Clifford follows a retirement, and
        // continues when a frame rotation shows a later region met a live frame.
        bool seen_retirement = false;
        bool restarts = false;
        bool continues = false;
        for (const PlannedAction& planned : frame.actions) {
            seen_retirement |= std::holds_alternative<sampling::RetireFrame>(planned.action);
            restarts |= seen_retirement &&
                        std::holds_alternative<sampling::ApplyConditionalClifford>(planned.action);
            continues |= std::holds_alternative<sampling::RotateFramePauli>(planned.action);
        }
        retired += seen_retirement;
        restarted += restarts;
        continued += continues;
        require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                    compute_exact_plan_distribution(frame));
    }
    // The generator must reach each outcome often enough to test it.
    CAPTURE(retired, restarted, continued);
    REQUIRE(retired >= 30);
    REQUIRE(restarted >= 10);
    REQUIRE(continued >= 10);
}

// Rounds of signed T gates on a data qubit, which is measured only at the end,
// and on factory qubits, some of which every round measures and resets. A
// frame then keeps part of its state while the rest is fixed. Most rounds
// leave the data qubit unentangled with the factory, so that their T gates
// meet released coordinates.
std::string random_partial_rounds_source(Xoshiro256PlusPlus& rng, uint32_t num_qubits) {
    const auto below = [&](uint32_t bound) { return static_cast<uint32_t>(rng() % bound); };
    const uint32_t factory_qubits = num_qubits - 1;
    const auto factory = [&] { return std::to_string(1 + below(factory_qubits)); };
    // The dense oracle branches on every record.
    constexpr uint32_t kMaxRecords = 9;
    std::string source = "H 0\n";
    uint32_t faults = 0;
    uint32_t records = 0;
    const uint32_t rounds = 2 + below(2);
    for (uint32_t round = 0; round < rounds; ++round) {
        for (uint32_t q = 1; q < num_qubits; ++q) {
            if (below(3) != 0) {
                source += "H " + std::to_string(q) + "\n";
            }
        }
        for (uint32_t i = 0; i + 1 < factory_qubits; ++i) {
            const uint32_t a = 1 + below(factory_qubits);
            const uint32_t b = 1 + (a + below(factory_qubits - 1)) % factory_qubits;
            source += "CX " + std::to_string(a) + " " + std::to_string(b) + "\n";
        }
        if (below(3) == 0) {
            source += "CX 0 " + factory() + "\n";
        }
        const uint32_t rotations = 2 + below(num_qubits);
        for (uint32_t i = 0; i < rotations; ++i) {
            const std::string target = below(4) == 0 ? "0" : factory();
            const auto fault = [&] {
                if (faults < 2 * (round + 1)) {
                    ++faults;
                    source += std::string(below(2) == 0 ? "X_ERROR" : "Z_ERROR") + "(0.2) " +
                              target + "\n";
                }
            };
            if (below(3) != 0) {
                // Two T gates with a fault between them: the pair is a
                // Clifford unless the fault flips one sign, which a frame
                // applies as a conditional Clifford.
                source += "T " + target + "\n";
                fault();
                source += "T " + target + "\n";
            } else {
                fault();
                source += std::string(below(2) == 0 ? "T " : "T_DAG ") + target + "\n";
            }
        }
        if (round + 1 == rounds) {
            for (uint32_t q = 0; q < num_qubits && records < kMaxRecords; ++q, ++records) {
                source += std::string(below(2) == 0 ? "M " : "MX ") + std::to_string(q) + "\n";
            }
            break;
        }
        // A check of two factory qubits leaves a stabilizer on more than one
        // frame coordinate.
        if (below(2) == 0 && records < kMaxRecords) {
            const uint32_t a = 1 + below(factory_qubits);
            const uint32_t b = 1 + (a + below(factory_qubits - 1)) % factory_qubits;
            source += std::string("MPP ") + (below(2) == 0 ? "X" : "Z") + std::to_string(a) + "*" +
                      (below(2) == 0 ? "X" : "Z") + std::to_string(b) + "\n";
            ++records;
        }
        const uint32_t forced = 1 + below(factory_qubits);
        for (uint32_t q = 1; q < num_qubits && records < kMaxRecords; ++q) {
            if (q == forced || below(3) != 0) {
                source += std::string(below(3) == 0 ? "MRX " : "MR ") + std::to_string(q) + "\n";
                ++records;
            }
        }
    }
    return source;
}

TEST_CASE("Random rounds that fix part of a frame match the dense oracle",
          "[runtime_frame][sampling]") {
    Xoshiro256PlusPlus rng(0x706B);
    uint32_t released = 0;
    uint32_t reduced_after = 0;
    uint32_t continued = 0;
    for (uint32_t trial = 0; trial < 200; ++trial) {
        const std::string source = random_partial_rounds_source(rng, 3 + trial % 3);
        CAPTURE(trial, source);
        const Circuit circuit = clifft::parse(source);
        const SamplingPlan frame =
            sampling::plan_sampling(clifft::trace(circuit), {.runtime_frame = true});
        // A plan reduces a region on released coordinates when a conditional
        // Clifford follows a release, and the frame continues through one when
        // a frame rotation does.
        bool seen_release = false;
        bool reduces = false;
        bool continues = false;
        for (const PlannedAction& planned : frame.actions) {
            seen_release |=
                std::holds_alternative<sampling::ReleaseFrameCoordinate>(planned.action);
            reduces |= seen_release &&
                       std::holds_alternative<sampling::ApplyConditionalClifford>(planned.action);
            continues |=
                seen_release && std::holds_alternative<sampling::RotateFramePauli>(planned.action);
        }
        released += seen_release;
        reduced_after += reduces;
        continued += continues;
        require_distributions_equal(compute_dense_circuit_distribution(circuit),
                                    compute_exact_plan_distribution(frame));
    }
    // The generator must reach each outcome often enough to test it.
    CAPTURE(released, reduced_after, continued);
    REQUIRE(released >= 30);
    REQUIRE(reduced_after >= 8);
    REQUIRE(continued >= 30);
}

TEST_CASE("Random signed regions match the dense oracle on a runtime frame",
          "[runtime_frame][sampling]") {
    Xoshiro256PlusPlus rng(0x704F);
    uint32_t frame_plans = 0;
    uint32_t narrower_plans = 0;
    for (uint32_t trial = 0; trial < 400; ++trial) {
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

TEST_CASE("Leading commuting conditional Cliffords skip the frame map",
          "[runtime_frame][sampling]") {
    // X0*X1 and Z0*Z1 commute, so the frame still fixes both. Z0 anticommutes
    // with X0*X1, which ends the run for it and for everything after it.
    FrameProgram program;
    program.num_qubits = 3;
    program.noise_probabilities = {0.3, 0.4};
    program.steps = {rotate(0b011, 0, 0.5, {}, {0}),
                     rotate(0, 0b011, 1.5, {}, {1}),
                     rotate(0b100, 0b100, 1.0, {}, {0, 1}),
                     rotate(0, 0b001, 0.5, {}, {0}),
                     rotate(0b011, 0, 1.5, {}, {1}),
                     measure(0b001, 0b010),
                     rotate(0b100, 0, 0.5, {}, {0}),
                     measure(0b010, 0b001),
                     measure(0b100, 0)};
    check_frame_program(program);

    const sampling::ExecutablePlan executable(build_frame_plan(program));
    const auto uses_support = [&](size_t action) {
        return executable.inspect_action(action).find("kernel=support") != std::string::npos;
    };
    REQUIRE(uses_support(0));
    REQUIRE(uses_support(1));
    REQUIRE(uses_support(2));
    // Only the first Pauli comes before any Z factor.
    const auto adds_x_words_only = [&](size_t action) {
        return executable.inspect_action(action).find("kernel=support_x") != std::string::npos;
    };
    REQUIRE(adds_x_words_only(0));
    REQUIRE_FALSE(adds_x_words_only(1));
    REQUIRE_FALSE(adds_x_words_only(2));
    REQUIRE_FALSE(uses_support(3));
    REQUIRE_FALSE(uses_support(4));
    REQUIRE_FALSE(uses_support(6));
}

TEST_CASE("Rotations about X products from the identity only add X words",
          "[runtime_frame][sampling]") {
    // A frame over 70 coordinates, so rows span two words. Random products of
    // X commute, and from the identity each one's image is itself.
    constexpr uint32_t kCoordinates = 70;
    Xoshiro256PlusPlus rng(0x5EED);
    for (uint32_t trial = 0; trial < 20; ++trial) {
        CAPTURE(trial);
        sampling::OnlineFrame general(kCoordinates);
        sampling::OnlineFrame x_only(kCoordinates);
        for (uint32_t step = 0; step < 40; ++step) {
            PauliString pauli(kCoordinates);
            const uint32_t factors = 1 + static_cast<uint32_t>(rng() % 6);
            for (uint32_t i = 0; i < factors; ++i) {
                pauli.set_pauli(static_cast<uint32_t>(rng() % kCoordinates), true, false);
            }
            constexpr CliffordRotation kRotations[] = {
                CliffordRotation::SQRT, CliffordRotation::PAULI, CliffordRotation::SQRT_DAG};
            const CliffordRotation rotation = kRotations[rng() % 3];
            general.rotate_clifford(pauli.view(), rotation);
            x_only.rotate_fixed_x_clifford(pauli.x(), rotation);
        }
        for (uint32_t row = 0; row < 2 * kCoordinates; ++row) {
            CAPTURE(row);
            REQUIRE(PauliString(x_only.row(row)) == PauliString(general.row(row)));
        }
    }
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
    const sampling::ExecutablePlan lazy_executable(lazy);
    const sampling::ExecutablePlan eager_executable(with_storage_from_the_start(lazy, 3));
    REQUIRE(lazy_executable.initial_capacity_width() == 0);
    REQUIRE(eager_executable.initial_capacity_width() == 3);
    sampling::Executor lazy_executor(lazy_executable, 99);
    sampling::Executor eager_executor(eager_executable, 99);
    REQUIRE(lazy_executor.state().max_active_width() == 0);
    bool saw_one = false;
    for (uint32_t shot = 0; shot < 200; ++shot) {
        CAPTURE(shot);
        lazy_executor.run_frame_shot();
        eager_executor.run_frame_shot();
        REQUIRE(
            std::ranges::equal(lazy_executor.visible_records(), eager_executor.visible_records()));
        saw_one |= std::ranges::count(lazy_executor.visible_records(), uint8_t{1}) != 0;
    }
    REQUIRE(saw_one);
    REQUIRE(lazy_executor.state().max_active_width() > 0);
    REQUIRE(eager_executor.state().max_active_width() == 3);
}

TEST_CASE("A frame shot repeated after storage growth counts its dust clamps once",
          "[runtime_frame][sampling]") {
    // Qubit 0 is measured with a branch of dust probability while one
    // coefficient is active. The frame on the two Bell pairs later promotes
    // two coordinates, so the first shot outgrows its storage and runs again.
    const Circuit circuit = clifft::parse(
        "H 0\nR_Z(0.0000000001) 0\nH 0\nM 0\n"
        "H 1 3\nCX 1 2 3 4\nX_ERROR(0.25) 1 3\nT 1 3\nT_DAG 2 4\n"
        "H 1 3\nT 1 3\nM 1 2 3 4\n");
    const SamplingPlan plan =
        sampling::plan_sampling(clifft::trace(circuit), {.runtime_frame = true});
    REQUIRE(plan.frame_width >= 2);
    const sampling::ExecutablePlan executable(plan);
    REQUIRE(executable.initial_capacity_width() == 1);
    sampling::Executor executor(executable, 7);
    REQUIRE(executor.state().max_active_width() == 1);
    for (uint64_t shot = 1; shot <= 4; ++shot) {
        CAPTURE(shot);
        executor.run_frame_shot();
        REQUIRE(executor.state().max_active_width() == 2);
        REQUIRE(executor.dust_clamps() == shot);
    }
}

TEST_CASE("A T gate that anticommutes with the held-back region starts a new one",
          "[runtime_frame][sampling]") {
    // The rotations about X3 and X0 are held back together. The T gate about
    // Z0 anticommutes with the second of them, so the region ends there and a
    // new one starts, which the signed pair on qubits 1 and 2 then joins and
    // is reduced in. Left in the first region, the pair would be applied one
    // rotation at a time and need an active qubit of its own.
    const auto circuit = clifft::parse(
        "H 1\nCX 1 2\nX_ERROR(0.3) 1\n"
        "H 3\nT 3\nH 3\n"
        "H 0\nT 0\nH 0\n"
        "T 0\n"
        "T 1\nT_DAG 2\n"
        "MPP X1*X2\nM 0 3\n");
    const SamplingPlan fixed = sampling::plan_sampling(clifft::trace(circuit));
    const SamplingPlan frame =
        sampling::plan_sampling(clifft::trace(circuit), {.runtime_frame = true});
    REQUIRE(fixed.peak_active_width == 3);
    REQUIRE(frame.peak_active_width == 2);
    REQUIRE(std::ranges::count_if(frame.actions, [](const PlannedAction& planned) {
                return std::holds_alternative<sampling::ApplyConditionalClifford>(planned.action);
            }) == 1);
}

TEST_CASE("A frame over more than 64 coordinates follows the fault of every pair",
          "[runtime_frame][sampling]") {
    // Seventy signed pairs give the frame one coordinate each, so six of them
    // lie in the second word of its Pauli masks. A pair's check reads one in
    // half of the shots where its fault fired and never otherwise. Before the
    // single checks, pair i and pair i + 35 are checked together, and for the
    // last six of those the measured Pauli spans both words. The joint outcome
    // is the parity of the two single ones in every shot. The fault
    // probabilities repeat every nine pairs, so pairs that are checked
    // together, neighbors, and pairs 64 apart all differ, and a sign that
    // reaches the wrong pair shows in its rate.
    constexpr uint32_t kPairs = 70;
    constexpr uint32_t kJoint = kPairs / 2;
    constexpr uint32_t kShots = 8000;
    const auto fault = [](uint32_t pair) { return 0.1 + 0.08 * (pair % 9); };
    const auto product = [](uint32_t pair) {
        return "X" + std::to_string(pair) + "*X" + std::to_string(pair + kPairs);
    };
    std::string prepare;
    std::string rotate;
    std::string joint;
    std::string single;
    for (uint32_t pair = 0; pair < kPairs; ++pair) {
        const std::string a = std::to_string(pair);
        const std::string b = std::to_string(pair + kPairs);
        prepare += "H " + a + "\nCX " + a + " " + b + "\nX_ERROR(" + std::to_string(fault(pair)) +
                   ") " + a + "\n";
        rotate += "T " + a + "\nT_DAG " + b + "\n";
        if (pair < kJoint) {
            joint += "MPP " + product(pair) + "*" + product(pair + kJoint) + "\n";
        }
        single += "MPP " + product(pair) + "\n";
    }
    const SamplingPlan plan = sampling::plan_sampling(
        clifft::trace(clifft::parse(prepare + rotate + joint + single)), {.runtime_frame = true});
    REQUIRE(plan.frame_width > 64);
    REQUIRE(plan.peak_active_width == 0);

    const sampling::ExecutablePlan executable(plan);
    sampling::Executor executor(executable, 709);
    std::vector<uint32_t> joint_ones(kJoint, 0);
    std::vector<uint32_t> single_ones(kPairs, 0);
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        executor.run_frame_shot();
        const auto records = executor.visible_records();
        for (uint32_t pair = 0; pair < kPairs; ++pair) {
            single_ones[pair] += records[kJoint + pair];
        }
        for (uint32_t pair = 0; pair < kJoint; ++pair) {
            joint_ones[pair] += records[pair];
            REQUIRE(records[pair] == (records[kJoint + pair] ^ records[kJoint + pair + kJoint]));
        }
    }
    const auto check_rate = [&](uint32_t ones, double expected) {
        const double sigma = std::sqrt(expected * (1.0 - expected) / kShots);
        REQUIRE(std::abs(double(ones) / kShots - expected) <= 6.0 * sigma);
    };
    for (uint32_t pair = 0; pair < kPairs; ++pair) {
        CAPTURE(pair);
        check_rate(single_ones[pair], 0.5 * fault(pair));
    }
    for (uint32_t pair = 0; pair < kJoint; ++pair) {
        CAPTURE(pair);
        const double first = 0.5 * fault(pair);
        const double second = 0.5 * fault(pair + kJoint);
        check_rate(joint_ones[pair], first + second - 2.0 * first * second);
    }
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
            executor.run_frame_shot();
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

TEST_CASE("Output compaction keeps its work estimate across retired frames",
          "[runtime_frame][sampling]") {
    // Each round promotes qubit 2 on a frame that the round's measurements
    // then retire, so no round is wider than the first.
    constexpr uint32_t kRounds = 70;
    std::string text;
    for (uint32_t round = 0; round < kRounds; ++round) {
        text +=
            "H 0 2\nCX 0 1\nX_ERROR(0.2) 0\nT 0 2\nT_DAG 1\nCX 0 1\nH 0 2\nMR 0 1 2\n"
            "DETECTOR rec[-3]\n";
    }
    const SamplingPlan plan =
        sampling::plan_sampling(clifft::trace(clifft::parse(text)), {.runtime_frame = true});
    REQUIRE(plan.peak_active_width == 1);
    REQUIRE(std::ranges::count_if(plan.actions, [](const PlannedAction& planned) {
                return std::holds_alternative<sampling::RetireFrame>(planned.action);
            }) >= std::ptrdiff_t{kRounds} - 1);
    const sampling::ExecutablePlan executable(plan);
    for (const sampling::SamplingOutputs outputs :
         {sampling::SamplingOutputs{sampling::RecordOutputs::Syndromes},
          sampling::SamplingOutputs{sampling::RecordOutputs::None}}) {
        const sampling::ExecutablePlan compact = executable.compact_outputs(outputs, false, true);
        for (const sampling::BatchOutputMode mode :
             {sampling::BatchOutputMode::Rows, sampling::BatchOutputMode::AggregateSurvivors}) {
            // Dropping outputs removes work and adds none.
            REQUIRE(compact.estimated_batch_lane_work(mode) <=
                    executable.estimated_batch_lane_work(mode));
        }
    }
}

TEST_CASE("Fixed-fault sampling on a frame plan follows the static plan",
          "[runtime_frame][sampling]") {
    // Faults before the T region decide its rotation signs, and readout
    // faults flip the reported records.
    const std::string body =
        "H 0 2\n"
        "CX 0 1\n"
        "X_ERROR(0.2) 0\n"
        "Z_ERROR(0.3) 2\n"
        "DEPOLARIZE1(0.1) 1\n"
        "T 0\nT_DAG 1\nT 2\n"
        "H 0 1 2\n"
        "M(0.05) 0 1 2\n";
    constexpr uint32_t kShots = 40000;
    // Two-sample bound at six standard deviations for a count out of kShots.
    const auto require_close = [&](uint64_t a, uint64_t b) {
        const double pooled = static_cast<double>(a + b) / (2.0 * kShots);
        const double sigma = std::sqrt(2.0 * pooled * (1.0 - pooled) / kShots);
        REQUIRE(std::abs(static_cast<double>(a) - static_cast<double>(b)) / kShots <=
                6.0 * sigma + 1e-12);
    };

    SECTION("record rows") {
        const HirModule hir = clifft::trace(clifft::parse(body));
        const SamplingPlan frame_plan = sampling::plan_sampling(hir, {.runtime_frame = true});
        REQUIRE(frame_plan.frame_width != 0);
        const sampling::ExecutablePlan fixed(sampling::plan_sampling(hir));
        const sampling::ExecutablePlan frame(frame_plan);
        for (uint32_t k = 0; k <= 3; ++k) {
            CAPTURE(k);
            const auto histogram = [&](const sampling::ExecutablePlan& plan, uint64_t seed) {
                const sampling::SamplingResult result = sampling::sample_k(plan, kShots, k, seed);
                std::array<uint64_t, 8> counts{};
                for (uint32_t shot = 0; shot < kShots; ++shot) {
                    const size_t row = size_t{3} * shot;
                    ++counts[result.measurements[row] | (result.measurements[row + 1] << 1) |
                             (result.measurements[row + 2] << 2)];
                }
                return counts;
            };
            const auto expected = histogram(fixed, 705 + k);
            const auto actual = histogram(frame, 805 + k);
            for (size_t pattern = 0; pattern < expected.size(); ++pattern) {
                CAPTURE(pattern);
                require_close(expected[pattern], actual[pattern]);
            }
        }
    }

    SECTION("survivor counts") {
        const HirModule hir = clifft::trace(
            clifft::parse(body + "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-3]\n"));
        const std::vector<uint8_t> mask = {1};
        const SamplingPlan frame_plan =
            sampling::plan_sampling(hir, {.postselection_mask = mask, .runtime_frame = true});
        REQUIRE(frame_plan.frame_width != 0);
        const sampling::ExecutablePlan fixed(
            sampling::plan_sampling(hir, {.postselection_mask = mask}));
        const sampling::ExecutablePlan frame(frame_plan);
        for (uint32_t k = 0; k <= 3; ++k) {
            CAPTURE(k);
            const auto expected = sampling::sample_k_survivors(fixed, kShots, k, 905 + k);
            const auto actual = sampling::sample_k_survivors(frame, kShots, k, 1005 + k);
            require_close(expected.passed_shots, actual.passed_shots);
            require_close(expected.observable_ones.at(0), actual.observable_ones.at(0));
        }
    }
}

TEST_CASE("Growing coefficient storage does not change a seeded fixed-fault frame shot",
          "[runtime_frame][sampling]") {
    // Every rotation sign reads a noise bit, so the faults a shot selects
    // decide which promotions it needs. Equal probabilities select sites from
    // a shuffled pool and unequal ones from a table.
    for (const std::vector<double>& probabilities :
         {std::vector<double>(4, 0.2), std::vector<double>{0.2, 0.3, 0.1, 0.25}}) {
        FrameProgram program;
        program.num_qubits = 3;
        program.noise_probabilities = probabilities;
        program.steps = {rotate(0b001, 0, 0.137, {}, {0}),
                         rotate(0b010, 0, 0.3, {}, {1}),
                         rotate(0b110, 0b001, 0.25, {}, {2}),
                         measure(0b011, 0b100),
                         rotate(0b100, 0, 0.21, {}, {3}),
                         measure(0b100, 0b100),
                         rotate(0b001, 0b010, 0.137, {}, {0, 1}),
                         measure(0, 0b001),
                         measure(0b010, 0)};
        const SamplingPlan lazy = build_frame_plan(program);
        const sampling::ExecutablePlan lazy_executable(lazy);
        const sampling::ExecutablePlan eager_executable(with_storage_from_the_start(lazy, 3));
        REQUIRE(lazy_executable.initial_capacity_width() == 0);
        REQUIRE(eager_executable.initial_capacity_width() == 3);

        // Each call starts new executors, so each one repeats the shots that
        // first reach a wider state.
        constexpr uint32_t kShots = 8;
        const size_t stride = lazy.num_visible_records;
        bool saw_one = false;
        for (uint32_t k = 0; k <= 3; ++k) {
            for (uint64_t seed = 0; seed < 40; ++seed) {
                CAPTURE(probabilities, k, seed);
                const sampling::SamplingResult grown =
                    sampling::sample_k(lazy_executable, kShots, k, seed);
                const sampling::SamplingResult eager =
                    sampling::sample_k(eager_executable, kShots, k, seed);
                REQUIRE(grown.measurements.size() == kShots * stride);
                REQUIRE(std::ranges::equal(grown.measurements, eager.measurements));
                saw_one |= std::ranges::count(grown.measurements, uint8_t{1}) != 0;
            }
        }
        REQUIRE(saw_one);
    }
}

}  // namespace clifft::test
