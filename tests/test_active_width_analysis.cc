// Checks structural width predictions against the planner and basis updates
// against a separate generator-list implementation.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/tableau/pauli_string.h"
#include "clifft/util/xoshiro.h"

#include "instrument_test_helpers.h"
#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace clifft;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::InstrumentMode;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::PlannedAction;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// Compare widths, transitions, and action counts; matching only the peak
// could hide misclassified operations.
void require_trace_matches_plan(const HirModule& hir, const SamplingPlan& plan) {
    const ActiveWidthTrace trace = analyze_active_width(hir);

    REQUIRE(trace.transitions.size() == hir.ops.size());
    REQUIRE(trace.initial_width == plan.initial_active_width);
    REQUIRE(trace.peak_width == plan.peak_active_width);

    std::vector<std::pair<uint32_t, uint32_t>> plan_changes;
    for (const PlannedAction& action : plan.actions) {
        if (action.active_before != action.active_after) {
            plan_changes.emplace_back(action.active_before, action.active_after);
        }
    }
    std::vector<std::pair<uint32_t, uint32_t>> trace_changes;
    for (const WidthTransition& transition : trace.transitions) {
        if (transition.before != transition.after) {
            trace_changes.emplace_back(transition.before, transition.after);
        }
    }
    REQUIRE(trace_changes == plan_changes);

    const uint32_t plan_final_width =
        plan.actions.empty() ? plan.initial_active_width : plan.actions.back().active_after;
    REQUIRE(trace.final_width == plan_final_width);

    size_t plan_promote = 0;
    size_t plan_rotate_neutral = 0;
    size_t plan_measure_active = 0;
    size_t plan_measure_dormant = 0;
    size_t plan_record_classical = 0;
    for (const PlannedAction& action : plan.actions) {
        if (std::holds_alternative<PromoteDormantRotation>(action.action)) {
            ++plan_promote;
        } else if (std::holds_alternative<RotateActivePauli>(action.action)) {
            ++plan_rotate_neutral;
        } else if (std::holds_alternative<MeasureActivePauli>(action.action)) {
            ++plan_measure_active;
        } else if (std::holds_alternative<MeasureDormantRandom>(action.action)) {
            ++plan_measure_dormant;
        } else if (std::holds_alternative<RecordClassical>(action.action)) {
            ++plan_record_classical;
        }
    }

    size_t trace_promote = 0;
    size_t trace_neutral = 0;
    size_t trace_measure_active = 0;
    size_t trace_measure_dormant = 0;
    size_t trace_classical = 0;
    for (const WidthTransition& transition : trace.transitions) {
        switch (transition.effect) {
            case WidthEffect::RotationPromote:
                ++trace_promote;
                break;
            case WidthEffect::RotationNeutral:
                ++trace_neutral;
                break;
            case WidthEffect::MeasureActive:
                ++trace_measure_active;
                break;
            case WidthEffect::MeasureDormantRandom:
                ++trace_measure_dormant;
                break;
            case WidthEffect::MeasureClassical:
                ++trace_classical;
                break;
            default:
                break;
        }
    }

    REQUIRE(plan_promote == trace_promote);
    REQUIRE(plan_rotate_neutral == trace_neutral);
    REQUIRE(plan_measure_active == trace_measure_active);
    REQUIRE(plan_measure_dormant == trace_measure_dormant);
    REQUIRE(plan_record_classical == trace_classical);
}

// Raw HIR can exceed the planner's width limit before squeezing. Skip
// only that case; other planning errors must still fail the comparison.
bool require_trace_matches_plan_if_plannable(const HirModule& hir) {
    SamplingPlan plan;
    try {
        plan = clifft::sampling::plan_sampling(hir);
    } catch (const std::overflow_error&) {
        return false;
    }
    require_trace_matches_plan(hir, plan);
    return true;
}

HirModule run_peephole_only(const HirModule& source) {
    HirModule hir = source;
    HirPassManager passes;
    passes.add_pass(std::make_unique<PeepholeFusionPass>());
    passes.run(hir);
    return hir;
}

// Name the passes explicitly so changes to the default pipeline do not
// change which HIR variants this analysis test covers.
HirModule run_production_pipeline(const HirModule& source) {
    HirModule hir = source;
    HirPassManager passes;
    passes.add_pass(std::make_unique<PeepholeFusionPass>());
    passes.add_pass(std::make_unique<StatevectorSqueezePass>());
    passes.run(hir);
    return hir;
}

// Mix Clifford gates, rotations, measurements, resets, and noise; detector
// references must point to records already produced.
std::string random_stim_circuit(std::mt19937& rng, uint32_t num_qubits, uint32_t num_ops) {
    static const double kAngles[] = {0.1, 0.125, 0.25, 0.3, 0.5, 0.7, 0.75, 1.25, 1.5, 1.75};
    static const double kNoiseProbs[] = {0.01, 0.02, 0.05, 0.1, 0.2};

    std::string source;
    uint32_t measurement_count = 0;
    for (uint32_t op = 0; op < num_ops; ++op) {
        const uint32_t q = rng() % num_qubits;
        switch (rng() % 12) {
            case 0:
                source += "H " + std::to_string(q) + "\n";
                break;
            case 1:
                source += "S " + std::to_string(q) + "\n";
                break;
            case 2: {
                uint32_t partner = rng() % num_qubits;
                if (partner == q) {
                    partner = (partner + 1) % num_qubits;
                }
                source += "CNOT " + std::to_string(q) + " " + std::to_string(partner) + "\n";
                break;
            }
            case 3:
                source += "T " + std::to_string(q) + "\n";
                break;
            case 4:
                source += "T_DAG " + std::to_string(q) + "\n";
                break;
            case 5:
                source += "R_Z(" + std::to_string(kAngles[rng() % std::size(kAngles)]) + ") " +
                          std::to_string(q) + "\n";
                break;
            case 6:
                source += "M " + std::to_string(q) + "\n";
                ++measurement_count;
                break;
            case 7:
                source += "MX " + std::to_string(q) + "\n";
                ++measurement_count;
                break;
            case 8:
                source += "MR " + std::to_string(q) + "\n";
                ++measurement_count;
                break;
            case 9:
                source += "R " + std::to_string(q) + "\n";
                break;
            case 10:
                source += "DEPOLARIZE1(" +
                          std::to_string(kNoiseProbs[rng() % std::size(kNoiseProbs)]) + ") " +
                          std::to_string(q) + "\n";
                break;
            case 11:
                source += "X_ERROR(" + std::to_string(kNoiseProbs[rng() % std::size(kNoiseProbs)]) +
                          ") " + std::to_string(q) + "\n";
                break;
            default:
                break;
        }
        if (measurement_count > 0 && (rng() % 5) == 0) {
            const uint32_t back = 1 + (rng() % measurement_count);
            source += "DETECTOR rec[-" + std::to_string(back) + "]\n";
        }
    }
    return source;
}

}  // namespace

// ---------------------------------------------------------------------------
// Differential tests against the planner
// ---------------------------------------------------------------------------

TEST_CASE("Active width analysis matches the planner on fixture circuits", "[active_width]") {
    const std::vector<std::string> mandatory_fixtures = {
        "coherent_d3_r3.stim",
        "coherent_d5_r5.stim",
        "cultivation_d5.stim",
        "surface_d7_r7_p001.stim",
    };
    const std::vector<std::string> all_fixtures = {
        "coherent_d3_r3.stim",     "coherent_d5_r5.stim", "cultivation_d5.stim",
        "surface_d7_r7_p001.stim", "qv10.stim",           "surface_d11_r11_p001.stim",
        "surface_d5_r5_p05.stim",  "target_qec.stim",
    };

    for (const std::string& fixture : all_fixtures) {
        const bool mandatory = std::find(mandatory_fixtures.begin(), mandatory_fixtures.end(),
                                         fixture) != mandatory_fixtures.end();
        DYNAMIC_SECTION(fixture) {
            const Circuit circuit = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/" + fixture);
            const HirModule raw = trace(circuit);
            const HirModule peephole_only = run_peephole_only(raw);
            const HirModule production = run_production_pipeline(raw);

            // Unsqueezed variants may exceed the planner's dense-width limit.
            if (!require_trace_matches_plan_if_plannable(raw)) {
                WARN("raw HIR exceeded the planner's dense active-width limit for " << fixture);
            }
            if (!require_trace_matches_plan_if_plannable(peephole_only)) {
                WARN("peephole-only HIR exceeded the planner's dense active-width limit for "
                     << fixture);
            }

            // Every corpus fixture must plan after fusion and squeezing.
            if (!require_trace_matches_plan_if_plannable(production)) {
                if (mandatory) {
                    FAIL("mandatory fixture's production pipeline could not be planned: "
                         << fixture);
                } else {
                    WARN("production pipeline could not be planned for " << fixture);
                }
            }
        }
    }
}

TEST_CASE("Active width analysis matches the planner on random circuits", "[active_width]") {
    constexpr uint32_t kActiveWidthFuzzSeed = 0x41770;
    constexpr int kTrials = 400;

    std::mt19937 rng(kActiveWidthFuzzSeed);
    for (int trial = 0; trial < kTrials; ++trial) {
        const uint32_t num_qubits = 4 + static_cast<uint32_t>(trial % 9);
        const uint32_t num_ops = 20 + static_cast<uint32_t>(trial % 30);
        const std::string source = random_stim_circuit(rng, num_qubits, num_ops);
        CAPTURE(trial, num_qubits, num_ops, source);

        const HirModule raw = trace(parse(source));
        const HirModule peephole_only = run_peephole_only(raw);
        const HirModule production = run_production_pipeline(raw);

        // These small circuits stay below the planner's width limit.
        require_trace_matches_plan(raw, clifft::sampling::plan_sampling(raw));
        require_trace_matches_plan(peephole_only, clifft::sampling::plan_sampling(peephole_only));
        require_trace_matches_plan(production, clifft::sampling::plan_sampling(production));
    }
}

TEST_CASE("Active width analysis matches instrument modes across all four branches",
          "[active_width]") {
    struct InstrumentCase {
        const char* name;
        const char* circuit;
        bool neglect_damping;
        InstrumentMode expected_mode;
        WidthEffect expected_effect;
    };
    const InstrumentCase cases[] = {
        {"classical", "LEVEL_TRANSITION[jump] 0", false, InstrumentMode::Classical,
         WidthEffect::InstrumentClassical},
        {"activate", "H 0\nT 0\nH 1\nLEVEL_TRANSITION[jump] 1\nM 1", false,
         InstrumentMode::Activate, WidthEffect::InstrumentActivate},
        {"active", "H 0\nT 0\nLEVEL_TRANSITION[jump] 0", false, InstrumentMode::Active,
         WidthEffect::InstrumentActive},
        {"dormant trap", "H 0\nLEVEL_TRANSITION[jump] 0", true, InstrumentMode::DormantTrap,
         WidthEffect::InstrumentDormantTrap},
    };

    for (const InstrumentCase& test_case : cases) {
        DYNAMIC_SECTION(test_case.name) {
            const InstrumentTraceOptions options =
                clifft::test::source_dependent_jump_options(test_case.neglect_damping);
            const HirModule hir = trace(parse(test_case.circuit), &options);

            const SamplingPlan plan = clifft::sampling::plan_sampling(hir);
            const PlannedAction* instrument_action = nullptr;
            const ApplyInstrument* instrument = nullptr;
            for (const PlannedAction& action : plan.actions) {
                if (const auto* candidate = std::get_if<ApplyInstrument>(&action.action)) {
                    instrument_action = &action;
                    instrument = candidate;
                    break;
                }
            }
            REQUIRE(instrument != nullptr);
            REQUIRE(instrument->mode == test_case.expected_mode);

            const auto instrument_op = std::ranges::find_if(
                hir.ops, [](const HeisenbergOp& op) { return op.op_type() == OpType::INSTRUMENT; });
            REQUIRE(instrument_op != hir.ops.end());
            const auto op_index = static_cast<size_t>(instrument_op - hir.ops.begin());

            const ActiveWidthTrace ir_trace = analyze_active_width(hir);
            REQUIRE(op_index < ir_trace.transitions.size());
            const WidthTransition& transition = ir_trace.transitions[op_index];

            REQUIRE(transition.effect == test_case.expected_effect);
            REQUIRE(transition.before == instrument_action->active_before);
            REQUIRE(transition.after == instrument_action->active_after);

            require_trace_matches_plan(hir, plan);
        }
    }
}

// ---------------------------------------------------------------------------
// DormantSubspace unit tests
// ---------------------------------------------------------------------------

TEST_CASE("DormantSubspace widths across a promote-heavy two-qubit sequence", "[active_width]") {
    DormantSubspace subspace(2);
    std::vector<uint32_t> widths{subspace.active_width()};

    REQUIRE(subspace.apply_rotation(PauliString::from_text("+XX")));
    widths.push_back(subspace.active_width());

    REQUIRE(subspace.apply_rotation(PauliString::from_text("+ZY")));
    widths.push_back(subspace.active_width());

    REQUIRE(subspace.apply_measurement(PauliString::from_text("+YY")) ==
            DormantSubspace::MeasurementEffect::Active);
    widths.push_back(subspace.active_width());

    REQUIRE(subspace.apply_measurement(PauliString::from_text("+YI")) ==
            DormantSubspace::MeasurementEffect::Active);
    widths.push_back(subspace.active_width());

    REQUIRE(widths == std::vector<uint32_t>{0, 1, 2, 1, 0});
}

TEST_CASE("DormantSubspace widths across a collapse-then-promote two-qubit sequence",
          "[active_width]") {
    DormantSubspace subspace(2);
    std::vector<uint32_t> widths{subspace.active_width()};

    REQUIRE(subspace.apply_rotation(PauliString::from_text("+ZY")));
    widths.push_back(subspace.active_width());

    REQUIRE(subspace.apply_measurement(PauliString::from_text("+YY")) ==
            DormantSubspace::MeasurementEffect::DormantRandom);
    widths.push_back(subspace.active_width());

    REQUIRE_FALSE(subspace.apply_rotation(PauliString::from_text("+XX")));
    widths.push_back(subspace.active_width());

    REQUIRE(subspace.apply_measurement(PauliString::from_text("+YI")) ==
            DormantSubspace::MeasurementEffect::Active);
    widths.push_back(subspace.active_width());

    REQUIRE(widths == std::vector<uint32_t>{0, 1, 1, 1, 0});
}

TEST_CASE("DormantSubspace treats a rotation or measurement inside S as inert", "[active_width]") {
    SECTION("a lone generator") {
        DormantSubspace subspace(2);
        const PauliString z0 = PauliString::from_text("+ZI");

        REQUIRE(subspace.commutes_with_all(z0));
        REQUIRE(subspace.contains(z0));

        REQUIRE_FALSE(subspace.apply_rotation(z0));
        REQUIRE(subspace.active_width() == 0);
        REQUIRE(subspace.contains(z0));

        REQUIRE(subspace.apply_measurement(z0) == DormantSubspace::MeasurementEffect::Classical);
        REQUIRE(subspace.active_width() == 0);
    }

    SECTION("a product of generators") {
        DormantSubspace subspace(2);
        const PauliString zz = PauliString::from_text("+ZZ");

        REQUIRE(subspace.commutes_with_all(zz));
        REQUIRE(subspace.contains(zz));

        REQUIRE_FALSE(subspace.apply_rotation(zz));
        REQUIRE(subspace.active_width() == 0);
        REQUIRE(subspace.contains(zz));

        REQUIRE(subspace.apply_measurement(zz) == DormantSubspace::MeasurementEffect::Classical);
        REQUIRE(subspace.active_width() == 0);
    }
}

// ---------------------------------------------------------------------------
// Reduced row echelon form and cross-implementation checks
// ---------------------------------------------------------------------------

namespace {

// Use two independent random bits per qubit for an unsigned Pauli.
PauliString random_pauli_string(Xoshiro256PlusPlus& rng, uint32_t num_qubits) {
    PauliString p(num_qubits);
    for (uint32_t q = 0; q < num_qubits; ++q) {
        const uint64_t bits = rng();
        p.set_pauli(q, (bits & 1U) != 0, (bits & 2U) != 0);
    }
    return p;
}

// Inspect the reduced-basis invariants through public masks, independently
// of DormantSubspace's private pivot helpers.
uint32_t combined_domain(const PauliString& p) {
    return p.x().num_words() * 64;
}

bool combined_bit(const PauliString& p, uint32_t domain, uint32_t bit) {
    return bit < domain ? p.x().bit_get(bit) : p.z().bit_get(bit - domain);
}

uint32_t combined_lowest_bit(const PauliString& p) {
    const uint32_t domain = combined_domain(p);
    const uint32_t x_bit = p.x().lowest_bit();
    if (x_bit < domain) {
        return x_bit;
    }
    return domain + p.z().lowest_bit();
}

// Reference implementation using an unreduced generator list and exhaustive
// subset sums for membership, independent of the production elimination code.
class ReferenceDormantSubspace {
  public:
    explicit ReferenceDormantSubspace(uint32_t num_qubits) : num_qubits_(num_qubits) {
        for (uint32_t q = 0; q < num_qubits; ++q) {
            PauliString z(num_qubits);
            z.set_pauli(q, false, true);
            generators_.push_back(std::move(z));
        }
    }

    [[nodiscard]] uint32_t active_width() const {
        return num_qubits_ - static_cast<uint32_t>(generators_.size());
    }

    [[nodiscard]] const std::vector<PauliString>& generators() const { return generators_; }

    [[nodiscard]] bool commutes_with_all(const PauliString& p) const {
        for (const PauliString& g : generators_) {
            if (!g.view().commutes(p.view())) {
                return false;
            }
        }
        return true;
    }

    // Exhaustive subset sums are affordable for these tests' small subspaces.
    [[nodiscard]] bool contains(const PauliString& p) const {
        const size_t dim = generators_.size();
        for (uint64_t mask = 0; mask < (uint64_t{1} << dim); ++mask) {
            PauliString sum(num_qubits_);
            for (size_t i = 0; i < dim; ++i) {
                if ((mask & (uint64_t{1} << i)) != 0) {
                    sum.mut_x().xor_with(generators_[i].x());
                    sum.mut_z().xor_with(generators_[i].z());
                }
            }
            if (sum == p) {
                return true;
            }
        }
        return false;
    }

    bool apply_rotation(const PauliString& p) {
        for (size_t i = 0; i < generators_.size(); ++i) {
            if (!generators_[i].view().commutes(p.view())) {
                for (size_t j = 0; j < generators_.size(); ++j) {
                    if (j != i && !generators_[j].view().commutes(p.view())) {
                        generators_[j].mut_x().xor_with(generators_[i].x());
                        generators_[j].mut_z().xor_with(generators_[i].z());
                    }
                }
                generators_.erase(generators_.begin() + static_cast<ptrdiff_t>(i));
                return true;
            }
        }
        return false;
    }

    DormantSubspace::MeasurementEffect apply_measurement(const PauliString& p) {
        if (apply_rotation(p)) {
            generators_.push_back(p);
            return DormantSubspace::MeasurementEffect::DormantRandom;
        }
        if (contains(p)) {
            return DormantSubspace::MeasurementEffect::Classical;
        }
        generators_.push_back(p);
        return DormantSubspace::MeasurementEffect::Active;
    }

  private:
    uint32_t num_qubits_;
    std::vector<PauliString> generators_;
};

}  // namespace

TEST_CASE("DormantSubspace generators stay in reduced row echelon form across random updates",
          "[active_width]") {
    const std::vector<uint64_t> seeds = {1, 2, 3, 42, 987654321};

    for (const uint64_t seed : seeds) {
        for (uint32_t num_qubits = 1; num_qubits <= 12; ++num_qubits) {
            DYNAMIC_SECTION("seed " << seed << " with " << num_qubits << " qubits") {
                Xoshiro256PlusPlus rng(seed * 1000003U + num_qubits);
                DormantSubspace subspace(num_qubits);

                constexpr int kOps = 300;
                for (int op = 0; op < kOps; ++op) {
                    const PauliString p = random_pauli_string(rng, num_qubits);
                    if ((rng() & 1U) != 0) {
                        subspace.apply_rotation(p);
                    } else {
                        subspace.apply_measurement(p);
                    }

                    const std::vector<PauliString> generators = subspace.generators();
                    CAPTURE(seed, num_qubits, op);
                    REQUIRE(generators.size() == num_qubits - subspace.active_width());

                    std::vector<uint32_t> pivots;
                    pivots.reserve(generators.size());
                    for (const PauliString& g : generators) {
                        pivots.push_back(combined_lowest_bit(g));
                    }

                    // Each pivot must be unique and zero in every other generator.
                    std::vector<uint32_t> sorted_pivots = pivots;
                    std::ranges::sort(sorted_pivots);
                    REQUIRE(std::adjacent_find(sorted_pivots.begin(), sorted_pivots.end()) ==
                            sorted_pivots.end());

                    for (size_t i = 0; i < generators.size(); ++i) {
                        const uint32_t domain = combined_domain(generators[i]);
                        for (size_t j = 0; j < generators.size(); ++j) {
                            if (i == j) {
                                continue;
                            }
                            REQUIRE(generators[i].view().commutes(generators[j].view()));
                            REQUIRE_FALSE(combined_bit(generators[i], domain, pivots[j]));
                        }
                    }

                    for (const PauliString& g : generators) {
                        REQUIRE(subspace.contains(g));
                        REQUIRE(subspace.commutes_with_all(g));
                    }
                }
            }
        }
    }
}

TEST_CASE("DormantSubspace matches a plain generator-list reference across random updates",
          "[active_width]") {
    const std::vector<uint64_t> seeds = {7, 13, 4242};

    for (const uint64_t seed : seeds) {
        for (uint32_t num_qubits = 1; num_qubits <= 8; ++num_qubits) {
            DYNAMIC_SECTION("seed " << seed << " with " << num_qubits << " qubits") {
                Xoshiro256PlusPlus rng(seed * 2654435761U + num_qubits);
                DormantSubspace subspace(num_qubits);
                ReferenceDormantSubspace reference(num_qubits);

                constexpr int kOps = 200;
                for (int op = 0; op < kOps; ++op) {
                    const PauliString p = random_pauli_string(rng, num_qubits);
                    CAPTURE(seed, num_qubits, op);

                    if ((rng() & 1U) != 0) {
                        const bool subspace_result = subspace.apply_rotation(p);
                        const bool reference_result = reference.apply_rotation(p);
                        REQUIRE(subspace_result == reference_result);
                    } else {
                        const DormantSubspace::MeasurementEffect subspace_result =
                            subspace.apply_measurement(p);
                        const DormantSubspace::MeasurementEffect reference_result =
                            reference.apply_measurement(p);
                        REQUIRE(subspace_result == reference_result);
                    }

                    REQUIRE(subspace.active_width() == reference.active_width());

                    for (const PauliString& g : subspace.generators()) {
                        REQUIRE(reference.contains(g));
                    }
                    for (const PauliString& g : reference.generators()) {
                        REQUIRE(subspace.contains(g));
                    }
                }
            }
        }
    }
}

TEST_CASE("Dormant commutation queries agree across word boundaries", "[active_width]") {
    for (uint32_t n : {0u, 1u, 63u, 64u, 65u, 128u}) {
        Xoshiro256PlusPlus rng(991 + n);
        DormantSubspace subspace(n);
        for (uint32_t step = 0; step < 200; ++step) {
            const PauliString p = random_pauli_string(rng, n);
            const auto generators = subspace.generators();
            const bool expected = std::ranges::all_of(
                generators, [&](const PauliString& g) { return g.view().commutes(p.view()); });
            CAPTURE(n, step);
            REQUIRE(subspace.commutes_with_all(p) == expected);
            for (const auto& g : generators) {
                REQUIRE(subspace.commutes_with_all(g));
            }
            if (step % 3 == 0) {
                subspace.apply_rotation(p);
            } else {
                subspace.apply_measurement(p);
            }
        }
    }
}

TEST_CASE("Dormant updates agree with an embedded small reference across word boundaries",
          "[active_width]") {
    for (uint32_t n : {63u, 64u, 65u, 128u}) {
        const std::vector<uint32_t> sites{0, 1, n / 2 - 1, n / 2, n - 4, n - 3, n - 2, n - 1};
        const auto embed = [&](const PauliString& p) {
            PauliString result(n);
            for (uint32_t q = 0; q < sites.size(); ++q) {
                result.set_pauli(sites[q], p.x().bit_get(q), p.z().bit_get(q));
            }
            return result;
        };
        Xoshiro256PlusPlus rng(719 + n);
        DormantSubspace subspace(n);
        ReferenceDormantSubspace reference(8);
        for (uint32_t step = 0; step < 200; ++step) {
            const PauliString p = random_pauli_string(rng, 8);
            const PauliString wide = embed(p);
            CAPTURE(n, step);
            if (step % 3 == 0) {
                REQUIRE(subspace.apply_rotation(wide) == reference.apply_rotation(p));
            } else {
                REQUIRE(subspace.apply_measurement(wide) == reference.apply_measurement(p));
            }
            REQUIRE(subspace.active_width() == reference.active_width());
            for (const auto& g : reference.generators()) {
                REQUIRE(subspace.contains(embed(g)));
            }
            const PauliString query = random_pauli_string(rng, 8);
            REQUIRE(subspace.commutes_with_all(embed(query)) == reference.commutes_with_all(query));
            REQUIRE(subspace.contains(embed(query)) == reference.contains(query));
        }
    }
}
