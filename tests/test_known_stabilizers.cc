#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/symbolic_stabilizers.h"

#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <string>
#include <string_view>

using namespace clifft;

namespace {

std::optional<bool> final_eigenvalue(std::string_view source, std::string_view pauli,
                                     SymbolicStabilizerOptions options = {}) {
    const auto hir = trace(parse(source));
    SymbolicStabilizers known(hir, options);
    for (const auto& op : hir.ops) {
        known.advance(hir, op);
    }
    const auto physical = PauliString::from_text("+" + std::string(pauli));
    return known.fixed_constraints().eigenvalue(
        hir.final_tableau->inverse().apply(physical.view()));
}

}  // namespace

TEST_CASE("Measured constraints become fixed after physical outcome feedback", "[optimizer]") {
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nM 0\nCX rec[-1] 0", "Z") == false);
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nM !0\nCX rec[-1] 0", "Z") == true);
    REQUIRE_FALSE(final_eigenvalue("R_X(0.137) 0\nM 0", "Z"));
    REQUIRE_FALSE(final_eigenvalue("R_X(0.137) 0\nM(0.1) 0\nCX rec[-1] 0", "Z"));
    REQUIRE_FALSE(
        final_eigenvalue("R_X(0.137) 0\nM 0\nREADOUT_NOISE(0.1,0.3) rec[-1]\nCX rec[-1] 0", "Z"));
}

TEST_CASE("Reported record mutations leave physical constraints intact", "[optimizer]") {
    REQUIRE(final_eigenvalue("M(1) 0", "Z") == false);
    REQUIRE_FALSE(final_eigenvalue("M(1) 0\nCX rec[-1] 0", "Z"));
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nM 0\nCX rec[-1] 0\n"
                             "READOUT_NOISE(0.1) rec[-1]",
                             "Z") == false);
}

TEST_CASE("Repeated measurement relations cancel an earlier correction", "[optimizer]") {
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nM 0\nX_ERROR(0.2) 0\nM 0\n"
                             "CX rec[-1] 0",
                             "Z") == false);
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nM(0.2) 0\nM 0\nCX rec[-1] 0", "Z") == false);
    REQUIRE_FALSE(final_eigenvalue("R_X(0.137) 0\nM 0\nM(0.2) 0\nCX rec[-1] 0", "Z"));
}

TEST_CASE("Affine products preserve Pauli multiplication phases and shared faults", "[optimizer]") {
    const std::string bell = "H 0\nCX 0 1\n";
    REQUIRE(final_eigenvalue(bell + "E(0.2) Y0", "YY") == true);
    REQUIRE_FALSE(final_eigenvalue(bell + "E(0.2) Y0", "XX"));
    REQUIRE_FALSE(final_eigenvalue(bell + "E(0.2) Y0", "ZZ"));
    REQUIRE_FALSE(final_eigenvalue(bell + "X_ERROR(0.2) 0\nZ_ERROR(0.2) 0", "YY"));
    REQUIRE(final_eigenvalue("E(0.2) X0 X1", "ZZ") == false);
    REQUIRE_FALSE(final_eigenvalue("X_ERROR(1) 0", "Z"));
}

TEST_CASE("Noncommuting measurements retain the surviving subgroup", "[optimizer]") {
    REQUIRE(final_eigenvalue("H 0\nCX 0 1\nM 0\nCX rec[-1] 0\nCX rec[-1] 1", "ZZ") == false);
    REQUIRE(final_eigenvalue("H 0\nCX 0 1\nM 0\nCX rec[-1] 0\nCX rec[-1] 1", "IZ") == false);
    REQUIRE_FALSE(final_eigenvalue("H 0\nCX 0 1\nM 0\nCX rec[-1] 0", "XX"));
}

TEST_CASE("Unobserved faults preserve products without accumulating private signs", "[optimizer]") {
    SymbolicStabilizerOptions options;
    options.max_expression_terms = 1;
    const std::string source = "H 0\nCX 0 1\nREPEAT 20 {\nE(0.2) Y0\n}\n";
    REQUIRE(final_eigenvalue(source, "YY", options) == true);
    REQUIRE_FALSE(final_eigenvalue(source, "XX", options));
    REQUIRE_FALSE(final_eigenvalue(source, "ZZ", options));
    REQUIRE(final_eigenvalue(source + "MPP Z0*Z1\nCX rec[-1] 0", "ZZ", options) == false);
}

TEST_CASE("Resets recover constraints after non Clifford work", "[optimizer]") {
    REQUIRE(final_eigenvalue("H 0\nT 0\nR 0", "Z") == false);
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nRX 0", "X") == false);
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nRY 0", "Y") == false);
    REQUIRE(final_eigenvalue("H 0\nT 0\nMR(0.2) 0", "Z") == false);
    REQUIRE_FALSE(final_eigenvalue("H 0\nT 0\nR 0\nR_X(0.137) 0", "Z"));
}

TEST_CASE("Affine analysis budgets lose knowledge and allow subsequent recovery", "[optimizer]") {
    SymbolicStabilizerOptions options;
    options.max_expression_terms = 1;
    REQUIRE_FALSE(final_eigenvalue("X_ERROR(0.2) 0\nX_ERROR(0.3) 0", "Z", options));
    REQUIRE(final_eigenvalue("X_ERROR(0.2) 0\nX_ERROR(0.3) 0\nR 0", "Z", options) == false);
    options.max_record_entries = 1;
    REQUIRE_FALSE(final_eigenvalue("H 0 1\nM 0 1\nCX rec[-2] 0\nCX rec[-1] 1", "ZI", options));
    REQUIRE(final_eigenvalue("H 0 1\nM 0 1\nCX rec[-1] 1", "IZ", options) == false);
    options.max_row_products = 0;
    REQUIRE_FALSE(final_eigenvalue("E(0.2) X0 X1", "ZZ", options));
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nR 0", "Z", options) == false);
}

TEST_CASE("Fixed lookahead snapshots cannot learn constraints from future measurements",
          "[optimizer]") {
    const auto hir = trace(parse("H 0\nM 0\nCX rec[-1] 0"));
    SymbolicStabilizers known(hir);
    auto entry = known.fixed_constraints();
    for (const auto& op : hir.ops) {
        known.advance(hir, op);
        entry.advance(hir, op);
    }
    const auto axis = hir.final_tableau->inverse().apply(PauliString::from_text("+Z").view());
    REQUIRE(known.fixed_constraints().eigenvalue(axis) == false);
    REQUIRE_FALSE(entry.eigenvalue(axis));
}

TEST_CASE("Unused measurement records do not evict a pending feedback control", "[optimizer]") {
    SymbolicStabilizerOptions options;
    options.max_record_entries = 1;
    REQUIRE(final_eigenvalue("H 0\nM 0\nH 1 2\nM 1 2\nCX rec[-3] 0", "ZII", options) == false);
    REQUIRE(final_eigenvalue("H 0\nM 0\nCX rec[-1] 0\nH 1\nM 1\nCX rec[-1] 1", "IZ", options) ==
            false);
}

TEST_CASE("Live record eviction follows measurement order rather than slot numbering",
          "[optimizer]") {
    auto hir = trace(parse("H 0 1 2\nM 0 1 2\nCX rec[-3] 0\nCX rec[-2] 1\nCX rec[-1] 2"));
    for (auto& op : hir.ops) {
        if (op.op_type() == OpType::MEASURE) {
            const auto id = static_cast<uint32_t>(op.meas_record_idx());
            op = HeisenbergOp::make_measure(op.mask_handle(), MeasRecordIdx{2 - id});
        } else if (op.op_type() == OpType::CONDITIONAL_PAULI) {
            const auto id = static_cast<uint32_t>(op.controlling_meas());
            op = HeisenbergOp::make_conditional(op.mask_handle(), ControllingMeasIdx{2 - id});
        }
    }
    SymbolicStabilizerOptions options;
    options.max_record_entries = 2;
    SymbolicStabilizers known(hir, options);
    for (const auto& op : hir.ops) {
        known.advance(hir, op);
    }
    const auto inverse = hir.final_tableau->inverse();
    REQUIRE_FALSE(
        known.fixed_constraints().eigenvalue(inverse.apply(PauliString::from_text("+ZII").view())));
    REQUIRE(known.fixed_constraints().eigenvalue(
                inverse.apply(PauliString::from_text("+IZI").view())) == false);
    REQUIRE(known.fixed_constraints().eigenvalue(
                inverse.apply(PauliString::from_text("+IIZ").view())) == false);
}

TEST_CASE("Repeated feedback retains a record through intervening report mutations",
          "[optimizer]") {
    const std::string prefix = "H 0\nM 0\nCX rec[-1] 0\nREADOUT_NOISE(0.2) rec[-1]\n";
    REQUIRE_FALSE(final_eigenvalue(prefix + "CX rec[-1] 0", "Z"));
    REQUIRE(final_eigenvalue(prefix + "CX rec[-1] 0\nCX rec[-1] 0", "Z") == false);
}

TEST_CASE("Readout expression exhaustion preserves conservative fallback and recovery",
          "[optimizer]") {
    SymbolicStabilizerOptions options;
    options.max_expression_terms = 1;
    const std::string source = "H 0\nM 0\nREADOUT_NOISE(0.2) rec[-1]\nCX rec[-1] 0\n";
    REQUIRE_FALSE(final_eigenvalue(source, "Z", options));
    REQUIRE(final_eigenvalue(source + "R 0", "Z", options) == false);
}

TEST_CASE("Discarded facts retain the schedule needed for later reset recovery", "[optimizer]") {
    SymbolicStabilizerOptions options;
    options.max_row_products = 0;
    REQUIRE_FALSE(final_eigenvalue("E(0.2) X0 X1", "ZZ", options));
    REQUIRE(final_eigenvalue("E(0.2) X0 X1\nR 0 1", "ZZ", options) == false);
}

TEST_CASE("Measured feedback constraints span multiple mask words", "[optimizer]") {
    const std::string source = "R_X(0.173) 0 64 129\nMPP Z0*Z64*Z129\nCX rec[-1] 129\n";
    std::string axis(130, 'I');
    axis[0] = axis[64] = axis[129] = 'Z';
    REQUIRE(final_eigenvalue(source, axis) == false);
    REQUIRE(final_eigenvalue(source + "E(0.2) X0 X64", axis) == false);
    REQUIRE_FALSE(final_eigenvalue(source + "X_ERROR(0.2) 129", axis));
}

TEST_CASE("Rewrite proof obligations retain the complete fixed group", "[optimizer]") {
    KnownStabilizers obligations;
    obligations.insert(PauliString::from_text("+XX"));
    obligations.insert(PauliString::from_text("+ZZ"));
    REQUIRE(obligations.eigenvalue(PauliString::from_text("+YY")) == true);
    REQUIRE_FALSE(obligations.commutes(PauliString::from_text("+XI").view()));
    REQUIRE_FALSE(obligations.commutes(PauliString::from_text("+ZI").view()));
}
