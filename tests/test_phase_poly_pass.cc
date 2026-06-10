// Tests for PhasePolynomialPass.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/phase_poly_pass.h"
#include "clifft/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

using namespace clifft;

static HirModule hir_from(const char* text) {
    return clifft::trace(clifft::parse(text));
}

static void run_phase_poly_pipeline(HirModule& hir) {
    PeepholeFusionPass peep;
    peep.run(hir);
    PhasePolynomialPass poly;
    poly.run(hir);
    peep.run(hir);
}

static std::vector<std::complex<double>> statevector_for(const std::string& circuit_text,
                                                         bool optimize) {
    auto circuit = clifft::parse(circuit_text);
    auto hir = clifft::trace(circuit);
    if (optimize)
        run_phase_poly_pipeline(hir);

    auto mod = clifft::lower(hir);
    SchrodingerState state({.peak_rank = mod.peak_rank,
                            .num_measurements = mod.total_meas_slots,
                            .num_detectors = mod.num_detectors,
                            .num_observables = mod.num_observables,
                            .num_exp_vals = mod.num_exp_vals,
                            .seed = 42});
    execute(mod, state);
    return get_statevector(mod, state);
}

static void check_statevectors_equal(const std::vector<std::complex<double>>& a,
                                     const std::vector<std::complex<double>>& b,
                                     double tol = 1e-9) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        CAPTURE(i);
        clifft::test::check_complex(a[i], b[i], tol);
    }
}

static void check_statevectors_equal_up_to_global_phase(
    const std::vector<std::complex<double>>& opt, const std::vector<std::complex<double>>& ref,
    double fidelity_tol = 1e-8) {
    REQUIRE(opt.size() == ref.size());
    std::complex<double> inner{0.0, 0.0};
    for (size_t i = 0; i < ref.size(); ++i)
        inner += std::conj(ref[i]) * opt[i];
    double fidelity = std::norm(inner);
    INFO("fidelity=" << fidelity);
    REQUIRE(fidelity >= 1.0 - fidelity_tol);
}

// =============================================================================
// Basic correctness: no-op cases
// =============================================================================

TEST_CASE("PhasePolyPass: empty circuit is unchanged", "[phase_poly]") {
    HirModule hir(1, 0);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.t_reductions() == 0);
}

TEST_CASE("PhasePolyPass: single T gate is unchanged", "[phase_poly]") {
    auto hir = hir_from("T 0");
    size_t before = hir.num_t_gates();
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(hir.num_t_gates() == before);
    REQUIRE(pass.t_reductions() == 0);
}

TEST_CASE("PhasePolyPass: two T gates on same axis -- peephole, not TOHPE", "[phase_poly]") {
    // Peephole should have already reduced this; TOHPE pass sees empty HIR.
    auto hir = hir_from("T 0\nT 0");
    PeepholeFusionPass pre;
    pre.run(hir);
    REQUIRE(hir.ops.empty());

    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.t_reductions() == 0);
}

TEST_CASE("PhasePolyPass: anti-commuting T gates are in separate blocks", "[phase_poly]") {
    // T(Z0) and T(X0) anti-commute -- each ends up as a singleton block,
    // which is skipped by the pass (block size < 2).
    auto hir = hir_from("T 0\nH 0\nT 0");
    size_t before = hir.num_t_gates();
    REQUIRE(before == 2);

    PhasePolynomialPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == before);
    REQUIRE(pass.t_reductions() == 0);
}

TEST_CASE("PhasePolyPass: non-T ops are preserved", "[phase_poly]") {
    auto hir = hir_from("T 0\nM 0\nT 1");
    PhasePolynomialPass pass;
    pass.run(hir);
    // T(Z0) and T(Z1) commute and form a 2-T block; T count cannot increase.
    REQUIRE(hir.num_t_gates() <= 2);
    REQUIRE(pass.t_reductions() % 2 == 0);
    // MEASURE must still be present.
    bool has_measure = false;
    for (const auto& op : hir.ops)
        if (op.op_type() == OpType::MEASURE)
            has_measure = true;
    REQUIRE(has_measure);
}

// =============================================================================
// T-count monotonicity: pass never increases T count
// =============================================================================

TEST_CASE("PhasePolyPass: T count never increases on disjoint qubits", "[phase_poly]") {
    // Three commuting Z-type T gates -- synthesis matrix is full rank for 3
    // distinct single-qubit Paulis; no TOHPE reduction expected.
    auto hir = hir_from("T 0\nT 1\nT 2");
    size_t before = hir.num_t_gates();
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(hir.num_t_gates() <= before);
}

TEST_CASE("PhasePolyPass: T count never increases on mixed dagger", "[phase_poly]") {
    auto hir = hir_from("T 0\nT_DAG 1\nT 2\nT_DAG 0");
    size_t before = hir.num_t_gates();
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(hir.num_t_gates() <= before);
    REQUIRE(pass.t_reductions() % 2 == 0);
}

TEST_CASE("PhasePolyPass: reductions always come in pairs", "[phase_poly]") {
    // Each TOHPE step removes exactly two T gates (a duplicate pair).
    auto hir = hir_from("T 0\nT 1\nT 2\nT 0");
    size_t before = hir.num_t_gates();
    PhasePolynomialPass pass;
    pass.run(hir);
    size_t after = hir.num_t_gates();
    size_t removed = before - after;
    REQUIRE(removed % 2 == 0);
    REQUIRE(pass.t_reductions() == removed);
}

// =============================================================================
// Block boundary: anti-commuting ops segment blocks correctly
// =============================================================================

TEST_CASE("PhasePolyPass: MEASURE barrier respected for T count", "[phase_poly]") {
    // M 0 measures Z0 -- anti-commutes with X-basis T gates but commutes with
    // Z-axis T gates.  The pass must not push T gates across measurements.
    auto hir = hir_from("T 0\nM 0\nT 0");
    // After peephole: T(Z0) slides past M Z0 and fuses.
    PeepholeFusionPass pre;
    pre.run(hir);
    size_t after_peephole = hir.num_t_gates();

    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(hir.num_t_gates() <= after_peephole);
}

TEST_CASE("PhasePolyPass: 32-qubit limit enforced", "[phase_poly]") {
    // Circuits with more than 32 qubits are skipped without modification.
    HirModule hir(33, 2);
    hir.append_tgate(false, [](MutablePauliMaskView slot) {
        slot.z().words[0] = 1ULL;
        slot.set_sign(false);
    });
    hir.append_tgate(false, [](MutablePauliMaskView slot) {
        slot.z().words[0] = 2ULL;
        slot.set_sign(false);
    });
    PhasePolynomialPass pass;
    pass.run(hir);
    // The pass bails early for nq > 32; both T gates must survive.
    REQUIRE(hir.num_t_gates() == 2);
    REQUIRE(pass.t_reductions() == 0);
}

TEST_CASE("PhasePolyPass: evaluation table on representative circuits",
          "[phase_poly][evaluation]") {
    struct Row {
        const char* name;
        const char* circuit;
    };
    const Row rows[] = {
        {"toggle_sandwich",
         "R_XX(0.25) 0 1\nR_PAULI(0.25) X0*Y1\nR_PAULI(0.25) Y0*X1\nR_XX(0.25) 0 1\n"
         "R_YY(0.25) 0 1\nR_PAULI(0.25) Y0*X1"},
        {"ccx_toffoli",
         "H 2\nCNOT 1 2\nT_DAG 2\nCNOT 0 2\nT 2\nCNOT 1 2\nT_DAG 2\nCNOT 0 2\nT_DAG 1\nT 2\n"
         "H 2\nCNOT 0 1\nT_DAG 1\nCNOT 0 1\nT 0\nT 1"},
        {"controlled_s", "T 0\nT 1\nCNOT 0 1\nT_DAG 1\nCNOT 0 1"},
    };

    for (const auto& row : rows) {
        auto hir = hir_from(row.circuit);
        PeepholeFusionPass peep;
        peep.run(hir);
        size_t t_peep = hir.num_t_gates();

        PhasePolynomialPass pass;
        pass.run(hir);
        size_t t_poly = hir.num_t_gates();

        INFO("name=" << row.name << " peephole_T=" << t_peep << " pass_T=" << t_poly
                     << " mcr_swaps=" << pass.mcr_stats().swaps_applied
                     << " tohpe_red=" << pass.t_reductions());
        REQUIRE(t_poly <= t_peep);
        REQUIRE(pass.t_reductions() % 2 == 0);
    }
}

TEST_CASE("PhasePolyPass: registered as opt-in in pass registry", "[phase_poly]") {
    std::string json = clifft::pass_registry_json();
    REQUIRE(json.find("McrTcountPass") != std::string::npos);
    REQUIRE(json.find("TohpePhasePass") != std::string::npos);
    REQUIRE(json.find("PhasePolynomialPass") != std::string::npos);

    REQUIRE(make_hir_pass("McrTcountPass") != nullptr);
    REQUIRE(make_hir_pass("TohpePhasePass") != nullptr);
    REQUIRE(make_hir_pass("PhasePolynomialPass") != nullptr);
}

TEST_CASE("PhasePolyPass: MCR reduces toggle sandwich", "[phase_poly][mcr]") {
    auto hir = hir_from(
        "R_XX(0.25) 0 1\n"
        "R_PAULI(0.25) X0*Y1\n"
        "R_PAULI(0.25) Y0*X1\n"
        "R_XX(0.25) 0 1\n"
        "R_YY(0.25) 0 1\n"
        "R_PAULI(0.25) Y0*X1");

    PeepholeFusionPass peep;
    peep.run(hir);
    const size_t before = hir.num_t_gates();

    PhasePolynomialPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() < before);
    REQUIRE(pass.mcr_stats().swaps_applied >= 1);
}

TEST_CASE("PhasePolyPass: stats track before and after T count", "[phase_poly]") {
    auto hir = hir_from("T 0\nT 1\nT 2");
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.t_gates_before() == 3);
    REQUIRE(pass.t_gates_after() == hir.num_t_gates());
    REQUIRE(pass.t_gates_after() <= pass.t_gates_before());
}

// =============================================================================
// Statevector equivalence -- peephole + TOHPE pipeline preserves semantics
// =============================================================================

TEST_CASE("PhasePolyPass: statevector equivalence on CCX", "[phase_poly][statevector]") {
    const char* ccx =
        "H 2\n"
        "CNOT 1 2\n"
        "T_DAG 2\n"
        "CNOT 0 2\n"
        "T 2\n"
        "CNOT 1 2\n"
        "T_DAG 2\n"
        "CNOT 0 2\n"
        "T_DAG 1\n"
        "T 2\n"
        "H 2\n"
        "CNOT 0 1\n"
        "T_DAG 1\n"
        "CNOT 0 1\n"
        "T 0\n"
        "T 1";

    auto ref = statevector_for(ccx, false);
    auto opt = statevector_for(ccx, true);
    check_statevectors_equal(opt, ref, 1e-8);
}

TEST_CASE("PhasePolyPass: statevector equivalence on small Clifford+T mix",
          "[phase_poly][statevector]") {
    const char* circuit =
        "H 0\n"
        "T 0\n"
        "H 0\n"
        "S 0\n"
        "H 0\n"
        "T 1\n"
        "H 1\n"
        "CX 0 1\n"
        "T 0\n"
        "H 0\n"
        "T 1\n"
        "H 1";

    auto ref = statevector_for(circuit, false);
    auto opt = statevector_for(circuit, true);
    check_statevectors_equal(opt, ref, 1e-8);
}

TEST_CASE("PhasePolyPass: statevector equivalence on MCR toggle sandwich",
          "[phase_poly][statevector][mcr]") {
    const char* circuit =
        "R_XX(0.25) 0 1\n"
        "R_PAULI(0.25) X0*Y1\n"
        "R_PAULI(0.25) Y0*X1\n"
        "R_XX(0.25) 0 1\n"
        "R_YY(0.25) 0 1\n"
        "R_PAULI(0.25) Y0*X1";

    auto ref = statevector_for(circuit, false);
    auto opt = statevector_for(circuit, true);
    check_statevectors_equal_up_to_global_phase(opt, ref, 1e-6);
}

TEST_CASE("PhasePolyPass: statevector equivalence on controlled-S decomposition",
          "[phase_poly][statevector]") {
    const char* cs =
        "T 0\n"
        "T 1\n"
        "CNOT 0 1\n"
        "T_DAG 1\n"
        "CNOT 0 1";

    auto ref = statevector_for(cs, false);
    auto opt = statevector_for(cs, true);
    check_statevectors_equal(opt, ref, 1e-8);
}

TEST_CASE("PhasePolyPass: statevector equivalence with phase rotation between T gates",
          "[phase_poly][statevector]") {
    const char* circuit =
        "T 0\n"
        "R_X(0.125) 0\n"
        "T 1\n"
        "T 2\n"
        "T 0\n";

    auto ref = statevector_for(circuit, false);
    auto opt = statevector_for(circuit, true);
    check_statevectors_equal(opt, ref, 1e-8);
}

TEST_CASE("PhasePolyPass: barrier edge cases preserve statevectors",
          "[phase_poly][statevector][barrier]") {
    struct Case {
        const char* name;
        const char* circuit;
    };
    const Case cases[] = {
        {"t_split_by_rx",
         "T 0\n"
         "R_X(0.125) 0\n"
         "T 1\n"
         "T 2\n"
         "T 0\n"},
        {"same_qubit_both_sides", "T 0\nR_X(0.125) 0\nT 0\n"},
        {"t_split_by_rz", "T 0\nR_Z(0.125) 0\nT 1\nT 0\n"},
        {"t_split_by_rxx", "T 0\nT 1\nR_XX(0.25) 0 1\nT 0\nT 1\n"},
        {"t_split_by_rpauli", "T 0\nR_PAULI(0.125) X0\nT 1\nT 0\n"},
        {"double_phase_barrier", "T 0\nR_X(0.125) 0\nR_Z(0.125) 0\nT 1\nT 0\n"},
        {"barrier_leading", "R_X(0.125) 0\nT 0\nT 1\nT 2\n"},
        {"barrier_trailing", "T 0\nT 1\nR_X(0.125) 0\n"},
        {"t_dag_split_by_rx", "T 0\nR_X(0.125) 0\nT_DAG 1\nT 0\nT_DAG 2\n"},
        {"alternating_single_t_blocks", "T 0\nR_Z(0.0625) 0\nT 0\nR_Z(0.0625) 0\nT 0\n"},
    };

    for (const auto& row : cases) {
        CAPTURE(row.name);
        auto ref = statevector_for(row.circuit, false);
        auto opt = statevector_for(row.circuit, true);
        check_statevectors_equal(opt, ref, 1e-8);
    }
}

TEST_CASE("PhasePolyPass: global barriers prevent spurious TOHPE T reduction",
          "[phase_poly][barrier]") {
    struct Case {
        const char* name;
        const char* circuit;
        size_t expected_t;
    };
    const Case cases[] = {
        {"t_split_by_rx",
         "T 0\n"
         "R_X(0.125) 0\n"
         "T 1\n"
         "T 2\n"
         "T 0\n",
         4},
        {"same_qubit_both_sides", "T 0\nR_X(0.125) 0\nT 0\n", 2},
        {"noise_barrier",
         "T 0\n"
         "X_ERROR(0.001) 0\n"
         "T 1\n"
         "T 2\n"
         "T 0\n",
         4},
    };

    for (const auto& row : cases) {
        CAPTURE(row.name);
        auto hir_peep = hir_from(row.circuit);
        PeepholeFusionPass peep;
        peep.run(hir_peep);
        REQUIRE(hir_peep.num_t_gates() == row.expected_t);

        auto hir_opt = hir_from(row.circuit);
        run_phase_poly_pipeline(hir_opt);
        REQUIRE(hir_opt.num_t_gates() == row.expected_t);
    }
}

TEST_CASE("PhasePolyPass: surface-code fragments preserve T count across noise barriers",
          "[phase_poly][barrier][surface]") {
    struct Case {
        const char* name;
        const char* circuit;
        size_t expected_t;
    };
    const Case cases[] = {
        {"two_t_bursts_depolarize",
         "T 0 3 7 9 10 12 13\n"
         "DEPOLARIZE1(0.001) 0 3 7 9 10 12 13 1 2 4 5 6 8 11 14\n"
         "T 0 3 7 9 10 12 13\n",
         14},
        {"tdag_t_split_by_depolarize",
         "T_DAG 0 3 7 9 10 12 13\n"
         "DEPOLARIZE1(0.001) 0 3 7 9 10 12 13 1 2 4 5 6 8 11 14\n"
         "CX 1 0 2 3 6 7 8 9 11 10 14 13\n"
         "T 0 3 7 9 10 12 13\n",
         14},
        {"tdag3_noise_cx",
         "CX 2 3\n"
         "DEPOLARIZE2(0.001) 2 3\n"
         "T_DAG 3\n"
         "DEPOLARIZE1(0.001) 3 0 1 2 4 5 6 7 8 9 10 11 12 13 14\n"
         "CX 2 3\n",
         1},
    };

    for (const auto& row : cases) {
        CAPTURE(row.name);
        auto hir_peep = hir_from(row.circuit);
        PeepholeFusionPass peep;
        peep.run(hir_peep);
        REQUIRE(hir_peep.num_t_gates() == row.expected_t);

        auto hir_opt = hir_from(row.circuit);
        run_phase_poly_pipeline(hir_opt);
        REQUIRE(hir_opt.num_t_gates() == row.expected_t);
    }
}

TEST_CASE("PhasePolyPass: randomized Clifford+T statevector oracle", "[phase_poly][statevector]") {
    constexpr int kNumQubits = 4;
    constexpr int kDepth = 30;
    const char* gates_1q[] = {"H", "S", "S_DAG", "T", "T_DAG"};
    const char* gates_2q[] = {"CX", "CY", "CZ"};

    for (uint64_t trial_seed = 200; trial_seed < 206; ++trial_seed) {
        CAPTURE(trial_seed);
        uint64_t lcg = trial_seed;
        std::string circuit;
        for (int d = 0; d < kDepth; ++d) {
            uint64_t r = clifft::test::test_lcg(lcg);
            if (r % 3 == 0 && kNumQubits > 1) {
                uint64_t r2 = clifft::test::test_lcg(lcg);
                int q1 = static_cast<int>(r2 % kNumQubits);
                uint64_t r3 = clifft::test::test_lcg(lcg);
                int q2 = static_cast<int>(r3 % (kNumQubits - 1));
                if (q2 >= q1)
                    ++q2;
                circuit += std::string(gates_2q[r2 / 3 % 3]) + " " + std::to_string(q1) + " " +
                           std::to_string(q2) + "\n";
            } else {
                uint64_t r2 = clifft::test::test_lcg(lcg);
                int q = static_cast<int>(r2 % kNumQubits);
                circuit += std::string(gates_1q[r2 / 5 % 5]) + " " + std::to_string(q) + "\n";
            }
        }

        auto ref = statevector_for(circuit, false);
        auto opt = statevector_for(circuit, true);
        check_statevectors_equal_up_to_global_phase(opt, ref, 1e-6);
    }
}
