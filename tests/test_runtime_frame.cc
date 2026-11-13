// Exact small-circuit checks for T regions whose rotation signs depend on the
// shot. A Pauli fault, a readout flip on feedback, or a random stabilizer
// outcome that crosses a T-family rotation changes the trajectory by a
// Clifford: exp(-i pi/8 (-1)^t P) = exp(+i pi/4 t P) exp(-i pi/8 P). Each case
// compares the complete joint distribution of a sampling plan against the
// independent dense trajectory oracle.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"

#include "dense_circuit_oracle.h"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

namespace clifft::test {
namespace {

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
                require_distributions_equal(expected, compute_exact_hir_distribution(hir, options));
            }
        }
    }
}

}  // namespace clifft::test
