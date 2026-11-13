#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/symbolic_stabilizers.h"
#include "clifft/util/xoshiro.h"

#include <catch2/catch_test_macros.hpp>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33)) && \
    !defined(__SANITIZE_ADDRESS__)
#if defined(__has_feature)
#if !__has_feature(address_sanitizer)
#include <malloc.h>
#define CLIFFT_HAVE_MALLINFO2 1
#endif
#else
#include <malloc.h>
#define CLIFFT_HAVE_MALLINFO2 1
#endif
#endif

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
    const std::string cancellation =
        "H 1 2\nM 1 2\nCX rec[-2] 0\nCX rec[-1] 0\nCX rec[-2] 0\nCX rec[-1] 0";
    REQUIRE(final_eigenvalue(cancellation, "ZII") == false);
    REQUIRE_FALSE(final_eigenvalue(cancellation, "ZII", options));
    REQUIRE(final_eigenvalue(cancellation + "\nR 0", "ZII", options) == false);
    options.max_record_entries = 1;
    REQUIRE_FALSE(final_eigenvalue("H 0 1\nM 0 1\nCX rec[-2] 0\nCX rec[-1] 1", "ZI", options));
    REQUIRE(final_eigenvalue("H 0 1\nM 0 1\nCX rec[-1] 1", "IZ", options) == false);
    options.max_row_products = 0;
    REQUIRE_FALSE(final_eigenvalue("E(0.2) X0 X1", "ZZ", options));
    REQUIRE(final_eigenvalue("R_X(0.137) 0\nR 0", "Z", options) == false);
}

TEST_CASE("Anticommuting measurements preserve spectators when reduction is capped",
          "[optimizer]") {
    SymbolicStabilizerOptions options;
    options.max_row_products = 0;
    const std::string source = "MY 0\nCX rec[-1] 0\nI 1";
    REQUIRE(final_eigenvalue(source, "IZ") == false);
    REQUIRE(final_eigenvalue(source, "IZ", options) == false);
    REQUIRE(final_eigenvalue(source, "YI", options) == false);
}

TEST_CASE("Oversized symbolic expressions do not hide later fixed products", "[optimizer]") {
    // The first two data rows have signs a+b+c and a+d+e. Eliminating a
    // overflows three terms; the later rows still share a single sign f.
    const std::string source =
        "H 4 5 6 7 8 9\nM 4 5 6 7 8 9\n"
        "CX rec[-6] 0\nCX rec[-5] 0\nCX rec[-4] 0\n"
        "CX rec[-6] 1\nCX rec[-3] 1\nCX rec[-2] 1\n"
        "CX rec[-1] 2\nCX rec[-1] 3\nX_ERROR(0.2) 4 5 6 7 8 9";
    SymbolicStabilizerOptions options;
    options.max_expression_terms = 3;
    REQUIRE(final_eigenvalue(source, "IIZZIIIIII") == false);
    REQUIRE(final_eigenvalue(source, "IIZZIIIIII", options) == false);
    options.max_row_products = 1;
    REQUIRE_FALSE(final_eigenvalue(source, "IIZZIIIIII", options));
}

TEST_CASE("Constant facts survive the symbolic elimination work limit", "[optimizer]") {
    const std::string source = "H 0\nM 0\nCX rec[-1] 1\nCX rec[-1] 2\nI 3";
    SymbolicStabilizerOptions options;
    options.max_row_products = 0;
    REQUIRE(final_eigenvalue(source, "IZZI") == false);
    REQUIRE_FALSE(final_eigenvalue(source, "IZZI", options));
    REQUIRE(final_eigenvalue(source, "IIIZ", options) == false);
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

TEST_CASE("Fault checks follow constraint products into other mask words", "[optimizer]") {
    // A fault on qubits 0 and 70 leaves the product Z0*Z70, which spans two
    // mask words. A later fault confined to the second word must remove it.
    std::string zz(71, 'I');
    zz[0] = zz[70] = 'Z';
    const std::string upward = "E(0.2) X0 X70\n";
    REQUIRE(final_eigenvalue(upward, zz) == false);
    REQUIRE_FALSE(final_eigenvalue(upward + "X_ERROR(0.2) 70\n", zz));

    // A measured X70*X71 constraint in the second word picks up Z0 from the
    // first. A later fault confined to the first word must remove it.
    std::string zxx(72, 'I');
    zxx[0] = 'Z';
    zxx[70] = zxx[71] = 'X';
    const std::string downward = "MPP X70*X71\nCZ rec[-1] 70\nE(0.2) X0 Z70\n";
    REQUIRE(final_eigenvalue(downward, zxx) == false);
    REQUIRE_FALSE(final_eigenvalue(downward + "X_ERROR(0.2) 0\n", zxx));
}

TEST_CASE("Rewrite proof obligations retain the complete fixed group", "[optimizer]") {
    KnownStabilizers obligations;
    obligations.insert(PauliString::from_text("+XX"));
    obligations.insert(PauliString::from_text("+ZZ"));
    REQUIRE(obligations.eigenvalue(PauliString::from_text("+YY")) == true);
    REQUIRE_FALSE(obligations.commutes(PauliString::from_text("+XI").view()));
    REQUIRE_FALSE(obligations.commutes(PauliString::from_text("+ZI").view()));
}

TEST_CASE("Combined feedback preserves products and ignores branch global phases", "[optimizer]") {
    REQUIRE(final_eigenvalue("H 0\nM 0\nCX rec[-1] 0\nCZ rec[-1] 0", "Z") == false);
    SymbolicStabilizerOptions options;
    options.max_record_entries = 0;
    const std::string source = "H 0\nCX 0 1\nH 2\nM 2\nCX rec[-1] 0\nCX rec[-1] 1";
    REQUIRE(final_eigenvalue(source, "ZZI", options) == false);
    REQUIRE(final_eigenvalue(source, "XXI", options) == false);
}

TEST_CASE("Fixed constraint queries flush pending feedback before later uses", "[optimizer]") {
    const auto hir = trace(parse("H 0\nM 0\nCX rec[-1] 0\nCX rec[-1] 0"));
    SymbolicStabilizers known(hir);
    const auto axis = hir.final_tableau->inverse().apply(PauliString::from_text("+Z").view());
    known.advance(hir, hir.ops[0]);
    known.advance(hir, hir.ops[1]);
    REQUIRE(known.fixed_constraints().eigenvalue(axis) == false);
    known.advance(hir, hir.ops[2]);
    REQUIRE_FALSE(known.fixed_constraints().eigenvalue(axis));
}

TEST_CASE("Cached fixed subgroups agree with fresh extraction after intersections", "[optimizer]") {
    const auto hir =
        trace(parse("H 0 1 2\nMPP Z0*Z1\nCX rec[-1] 0\n"
                    "E(0.2) X0 X1\nR_Z(0.17) 2\nM 1\n"
                    "CX rec[-1] 1\nR_X(0.21) 0\nR 0"));
    SymbolicStabilizers cached(hir);
    for (size_t end = 0; end < hir.ops.size(); ++end) {
        cached.advance(hir, hir.ops[end]);
        const auto& fixed = cached.fixed_constraints();
        SymbolicStabilizers fresh(hir);
        for (size_t i = 0; i <= end; ++i) {
            fresh.advance(hir, hir.ops[i]);
        }
        for (uint32_t body = 0; body < 64; ++body) {
            PauliString axis(3);
            for (uint32_t q = 0; q < 3; ++q) {
                axis.set_pauli(q, (body >> q) & 1, (body >> (q + 3)) & 1);
            }
            REQUIRE(fixed.eigenvalue(axis) == fresh.fixed_constraints().eigenvalue(axis));
        }
    }
}

TEST_CASE("Wide stabilizer tracking preserves multiword unit Z and entangled subgroups",
          "[optimizer]") {
    // Exercise negative unit-Z signs, resets, multi-qubit pure-Z rows in rows_
    // intersected against unit-Z qubits in lower words, and bounded reduction
    // across three 64-qubit words.
    const std::string source =
        "M !5\nCX rec[-1] 5\n"
        "E(0.2) X66 X131\n"
        "E(0.2) X2 X66\n"
        "R_X(0.19) 70\nR 70\n";
    const auto hir = trace(parse(source));
    REQUIRE(hir.num_qubits == 132);

    SymbolicStabilizers symbolic(hir);
    KnownStabilizers direct(hir.num_qubits);
    for (const auto& op : hir.ops) {
        symbolic.advance(hir, op);
        direct.advance(hir, op);
    }

    const auto& fixed = symbolic.fixed_constraints();
    auto unit_z = [&](uint32_t q) {
        PauliString p(hir.num_qubits);
        p.set_pauli(q, false, true);
        return p;
    };

    // Qubit 5 was measured with inversion and corrected: fixed -Z_5.
    REQUIRE(fixed.eigenvalue(unit_z(5)) == true);
    // Untouched spectator and reset qubits remain +Z_q.
    REQUIRE(fixed.eigenvalue(unit_z(0)) == false);
    REQUIRE(fixed.eigenvalue(unit_z(70)) == false);
    REQUIRE(fixed.eigenvalue(unit_z(129)) == false);
    REQUIRE(direct.eigenvalue(unit_z(129)) == false);
    REQUIRE_FALSE(direct.eigenvalue(unit_z(70)));

    // E(0.2) X66 X131 left Z66*Z131; E(0.2) X2 X66 then multiplied unit-Z_2 by
    // the higher-pivot multi-qubit row Z66*Z131 to leave +Z2*Z66*Z131.
    PauliString triple(hir.num_qubits);
    triple.set_pauli(2, false, true);
    triple.set_pauli(66, false, true);
    triple.set_pauli(131, false, true);
    REQUIRE(fixed.eigenvalue(triple) == false);
    REQUIRE(direct.eigenvalue(triple) == false);
    REQUIRE_FALSE(fixed.eigenvalue(unit_z(2)));
    REQUIRE_FALSE(fixed.eigenvalue(unit_z(66)));
    REQUIRE_FALSE(fixed.eigenvalue(unit_z(131)));
}

TEST_CASE("Repeated reset feedback reclaims stabilizer storage across rounds", "[optimizer]") {
    constexpr uint32_t kNumQubits = 256;
    auto build_reset_loop = [](size_t rounds) {
        HirModule hir(kNumQubits, 6);
        hir.num_hidden_measurements = 2;
        hir.ops.reserve(6 * rounds);
        hir.append_measure(MeasRecordIdx{0},
                           [](MutablePauliMaskView s) { s.x().bit_set(0, true); });
        hir.append_conditional(ControllingMeasIdx{0},
                               [](MutablePauliMaskView s) { s.z().bit_set(0, true); });
        hir.append_measure(MeasRecordIdx{1},
                           [](MutablePauliMaskView s) { s.z().bit_set(0, true); });
        hir.append_conditional(ControllingMeasIdx{1},
                               [](MutablePauliMaskView s) { s.x().bit_set(0, true); });
        hir.append_measure(MeasRecordIdx{0}, [](MutablePauliMaskView s) {
            s.z().bit_set(199, true);
            s.set_sign(true);
        });
        hir.append_conditional(ControllingMeasIdx{0},
                               [](MutablePauliMaskView s) { s.x().bit_set(199, true); });

        for (size_t r = 1; r < rounds; ++r) {
            for (size_t j = 0; j < 6; ++j) {
                hir.ops.push_back(hir.ops[j]);
            }
        }
        return hir;
    };

    auto run_and_measure_retained_bytes = [](const HirModule& hir) -> size_t {
        PauliString x0(kNumQubits), z0(kNumQubits), z129(kNumQubits), z199(kNumQubits);
        x0.set_pauli(0, true, false);
        z0.set_pauli(0, false, true);
        z129.set_pauli(129, false, true);
        z199.set_pauli(199, false, true);

#ifdef CLIFFT_HAVE_MALLINFO2
        const size_t before_bytes = mallinfo2().uordblks;
#endif
        SymbolicStabilizers symbolic(hir);
        // Check intermediate +X_0 after the first X-reset pair.
        symbolic.advance(hir, hir.ops[0]);
        symbolic.advance(hir, hir.ops[1]);
        REQUIRE(symbolic.fixed_constraints().eigenvalue(x0) == false);
        REQUIRE_FALSE(symbolic.fixed_constraints().eigenvalue(z0));

        for (size_t i = 2; i < hir.ops.size(); ++i) {
            symbolic.advance(hir, hir.ops[i]);
        }
        const auto& fixed = symbolic.fixed_constraints();
        REQUIRE(fixed.eigenvalue(z0) == false);
        REQUIRE(fixed.eigenvalue(z129) == false);
        REQUIRE(fixed.eigenvalue(z199) == true);
#ifdef CLIFFT_HAVE_MALLINFO2
        const size_t after_bytes = mallinfo2().uordblks;
        return after_bytes > before_bytes ? (after_bytes - before_bytes) : 0;
#else
        return 0;
#endif
    };

    const HirModule short_hir = build_reset_loop(16);
    const HirModule long_hir = build_reset_loop(4096);
    const size_t short_retained = run_and_measure_retained_bytes(short_hir);
    const size_t long_retained = run_and_measure_retained_bytes(long_hir);
#ifdef CLIFFT_HAVE_MALLINFO2
    // Live stabilizer rank is bounded by kNumQubits regardless of reset-loop count;
    // leaking 4096 x 3 four-word PauliString rows would retain >1.5 MiB.
    REQUIRE(long_retained <= short_retained + 64 * 1024);
#else
    (void)short_retained;
    (void)long_retained;
#endif
}

namespace {

// Dense-PauliString reference tracker with no unit-Z bitset specialization,
// used as an exact differential oracle for signed stabilizer updates.
class NaiveKnownStabilizers {
  public:
    explicit NaiveKnownStabilizers(uint32_t num_qubits) {
        for (uint32_t q = 0; q < num_qubits; ++q) {
            PauliString z(num_qubits);
            z.set_pauli(q, false, true);
            const uint32_t pivot = optimizer_detail::pivot_of(z);
            rows_.emplace(pivot, std::move(z));
        }
    }

    void insert(PauliString axis) {
        if (!axis.x().is_zero() || !axis.z().is_zero()) {
            for (auto it = rows_.lower_bound(optimizer_detail::pivot_of(axis)); it != rows_.end();
                 ++it) {
                if (optimizer_detail::body_bit(axis, it->first)) {
                    axis.right_multiply(it->second.view());
                    if (axis.x().is_zero() && axis.z().is_zero()) {
                        break;
                    }
                }
            }
        }
        if (axis.x().is_zero() && axis.z().is_zero()) {
            REQUIRE_FALSE(axis.sign());
            return;
        }
        const uint32_t pivot = optimizer_detail::pivot_of(axis);
        rows_.emplace(pivot, std::move(axis));
    }

    void intersect(PauliStringView axis) {
        const PauliCommutation comm(axis.x(), axis.z());
        std::vector<uint32_t> hits;
        for (const auto& [pivot, row] : rows_) {
            if (comm.anti_commutes(row.x(), row.z())) {
                hits.push_back(pivot);
            }
        }
        if (hits.empty()) {
            return;
        }
        const PauliString pivot_row = std::move(rows_.at(hits.back()));
        rows_.erase(hits.back());
        for (size_t i = 0; i + 1 < hits.size(); ++i) {
            rows_.at(hits[i]).right_multiply(pivot_row.view());
        }
    }

    [[nodiscard]] bool commutes(PauliStringView axis) const {
        const PauliCommutation comm(axis.x(), axis.z());
        for (const auto& [pivot, row] : rows_) {
            if (comm.anti_commutes(row.x(), row.z())) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] PauliString reduce_body(PauliString axis) const {
        for (auto it = rows_.lower_bound(optimizer_detail::pivot_of(axis));
             it != rows_.end() && (!axis.x().is_zero() || !axis.z().is_zero()); ++it) {
            if (optimizer_detail::body_bit(axis, it->first)) {
                axis.mut_x().xor_with(it->second.x());
                axis.mut_z().xor_with(it->second.z());
            }
        }
        return axis;
    }

    [[nodiscard]] std::optional<bool> eigenvalue(PauliString axis) const {
        for (auto it = rows_.lower_bound(optimizer_detail::pivot_of(axis));
             it != rows_.end() && (!axis.x().is_zero() || !axis.z().is_zero()); ++it) {
            if (optimizer_detail::body_bit(axis, it->first)) {
                axis.right_multiply(it->second.view());
            }
        }
        if (!axis.x().is_zero() || !axis.z().is_zero()) {
            return std::nullopt;
        }
        return axis.sign();
    }

    void advance(const HirModule& hir, const HeisenbergOp& op) {
        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI: {
                const auto mask = hir.mask_view(op);
                intersect(optimizer_detail::copy_axis(mask, hir.num_qubits).view());
                break;
            }
            case OpType::NOISE: {
                const auto& channels =
                    hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
                for (const auto& ch : channels) {
                    if (ch.prob != 0) {
                        const auto mask = hir.noise_channel_masks.at(ch.mask);
                        intersect(optimizer_detail::copy_axis(mask, hir.num_qubits).view());
                    }
                }
                break;
            }
            case OpType::INSTRUMENT:
                rows_.clear();
                break;
            default:
                break;
        }
    }

    [[nodiscard]] PauliString random_in_group(uint32_t num_qubits, Xoshiro256PlusPlus& rng) const {
        PauliString prod(num_qubits);
        if ((rng() & 1U) != 0) {
            prod.set_sign(true);
        }
        for (const auto& [pivot, row] : rows_) {
            if ((rng() % 5U) == 0) {
                prod.right_multiply(row.view());
            }
        }
        return prod;
    }

  private:
    std::map<uint32_t, PauliString> rows_;
};

// Reference symbolic stabilizer tracker storing every generator as a dense row
// without unit-Z bitsets.
class NaiveSymbolicStabilizers {
  public:
    explicit NaiveSymbolicStabilizers(const HirModule& hir) : num_qubits_(hir.num_qubits) {
        for (uint32_t q = 0; q < num_qubits_; ++q) {
            PauliString z(num_qubits_);
            z.set_pauli(q, false, true);
            const uint32_t pivot = optimizer_detail::pivot_of(z);
            rows_.emplace(pivot, Row{std::move(z), {}});
        }
        for (const HeisenbergOp& op : hir.ops) {
            if (op.op_type() == OpType::CONDITIONAL_PAULI) {
                ++remaining_uses_[static_cast<uint32_t>(op.controlling_meas())];
            }
        }
    }

    void advance(const HirModule& hir, const HeisenbergOp& op) {
        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION: {
                const auto mask = hir.mask_view(op);
                intersect(mask.x(), mask.z());
                break;
            }
            case OpType::MEASURE: {
                const uint32_t rec = static_cast<uint32_t>(op.meas_record_idx());
                if (!remaining_uses_.contains(rec)) {
                    const auto mask = hir.mask_view(op);
                    intersect(mask.x(), mask.z());
                    break;
                }
                PauliString axis = optimizer_detail::copy_axis(hir.mask_view(op), num_qubits_);
                auto outcome = eigenvalue_expr(axis);
                if (!outcome) {
                    outcome = sampling::AffineBool::symbol(sampling::SymbolId{next_symbol_++});
                    intersect(axis.x(), axis.z());
                    insert_row(Row{std::move(axis), *outcome});
                }
                records_.insert_or_assign(rec, *outcome);
                break;
            }
            case OpType::CONDITIONAL_PAULI: {
                const uint32_t rec = static_cast<uint32_t>(op.controlling_meas());
                const auto mask = hir.mask_view(op);
                const auto it = records_.find(rec);
                if (it != records_.end()) {
                    apply_pauli(mask.x(), mask.z(), it->second);
                } else {
                    intersect(mask.x(), mask.z());
                }
                if (--remaining_uses_.at(rec) == 0) {
                    records_.erase(rec);
                    remaining_uses_.erase(rec);
                }
                break;
            }
            case OpType::NOISE: {
                const auto& channels =
                    hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
                for (const auto& ch : channels) {
                    if (ch.prob != 0) {
                        const auto mask = hir.noise_channel_masks.at(ch.mask);
                        intersect(mask.x(), mask.z());
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    [[nodiscard]] NaiveKnownStabilizers fixed_constraints() const {
        NaiveKnownStabilizers fixed(num_qubits_);
        // Clear initial +Z_q basis cleanly by intersecting single-qubit X_q.
        for (uint32_t q = 0; q < num_qubits_; ++q) {
            PauliString x(num_qubits_);
            x.set_pauli(q, true, false);
            fixed.intersect(x.view());
        }
        std::map<sampling::SymbolId, Row> sym_basis;
        for (const auto& [pivot, orig] : rows_) {
            Row row = orig;
            bool independent = false;
            while (!row.sign.terms().empty()) {
                const auto s = row.sign.terms().front();
                const auto it = sym_basis.find(s);
                if (it == sym_basis.end()) {
                    sym_basis.emplace(s, std::move(row));
                    independent = true;
                    break;
                }
                row.sign ^= it->second.sign;
                row.axis.right_multiply(it->second.axis.view());
            }
            if (!independent) {
                row.axis.set_sign(row.axis.sign() ^ row.sign.constant());
                fixed.insert(std::move(row.axis));
            }
        }
        return fixed;
    }

  private:
    struct Row {
        PauliString axis;
        sampling::AffineBool sign;
    };

    [[nodiscard]] std::optional<sampling::AffineBool> eigenvalue_expr(PauliString axis) const {
        sampling::AffineBool sign;
        for (auto it = rows_.lower_bound(optimizer_detail::pivot_of(axis));
             it != rows_.end() && (!axis.x().is_zero() || !axis.z().is_zero()); ++it) {
            if (optimizer_detail::body_bit(axis, it->first)) {
                axis.right_multiply(it->second.axis.view());
                sign ^= it->second.sign;
            }
        }
        if (!axis.x().is_zero() || !axis.z().is_zero()) {
            return std::nullopt;
        }
        sign ^= axis.sign();
        return sign;
    }

    void insert_row(Row row) {
        for (auto it = rows_.lower_bound(optimizer_detail::pivot_of(row.axis));
             it != rows_.end() && (!row.axis.x().is_zero() || !row.axis.z().is_zero()); ++it) {
            if (optimizer_detail::body_bit(row.axis, it->first)) {
                row.axis.right_multiply(it->second.axis.view());
                row.sign ^= it->second.sign;
            }
        }
        if (!row.axis.x().is_zero() || !row.axis.z().is_zero()) {
            const uint32_t pivot = optimizer_detail::pivot_of(row.axis);
            rows_.emplace(pivot, std::move(row));
        }
    }

    void intersect(MaskView ax, MaskView az) {
        const PauliCommutation comm(ax, az);
        std::vector<uint32_t> hits;
        for (const auto& [pivot, row] : rows_) {
            if (comm.anti_commutes(row.axis.x(), row.axis.z())) {
                hits.push_back(pivot);
            }
        }
        if (hits.empty()) {
            return;
        }
        const Row pivot_row = std::move(rows_.at(hits.back()));
        rows_.erase(hits.back());
        for (size_t i = 0; i + 1 < hits.size(); ++i) {
            rows_.at(hits[i]).axis.right_multiply(pivot_row.axis.view());
            rows_.at(hits[i]).sign ^= pivot_row.sign;
        }
    }

    void apply_pauli(MaskView ax, MaskView az, const sampling::AffineBool& cond) {
        if (cond.terms().empty() && !cond.constant()) {
            return;
        }
        const PauliCommutation comm(ax, az);
        for (auto& [pivot, row] : rows_) {
            if (comm.anti_commutes(row.axis.x(), row.axis.z())) {
                row.sign ^= cond;
            }
        }
    }

    uint32_t num_qubits_;
    uint32_t next_symbol_ = 0;
    std::map<uint32_t, Row> rows_;
    std::map<uint32_t, sampling::AffineBool> records_;
    std::map<uint32_t, size_t> remaining_uses_;
};

}  // namespace

TEST_CASE("KnownStabilizers matches dense reference across signed multiword updates",
          "[optimizer]") {
    constexpr uint32_t kNumQubits = 140;
    Xoshiro256PlusPlus rng(0x57AB1123ULL);

    KnownStabilizers actual(kNumQubits);
    NaiveKnownStabilizers expected(kNumQubits);

    auto check_agreement = [&] {
        for (int p = 0; p < 12; ++p) {
            const PauliString in_group = expected.random_in_group(kNumQubits, rng);
            REQUIRE(actual.eigenvalue(in_group) == expected.eigenvalue(in_group));
            REQUIRE(actual.commutes(in_group.view()) == expected.commutes(in_group.view()));
            const PauliString reduced_a = actual.reduce_body(in_group);
            const PauliString reduced_e = expected.reduce_body(in_group);
            REQUIRE(reduced_a.x() == reduced_e.x());
            REQUIRE(reduced_a.z() == reduced_e.z());

            PauliString probe(kNumQubits);
            const uint32_t weight = 1 + static_cast<uint32_t>(rng() % 5);
            for (uint32_t k = 0; k < weight; ++k) {
                const uint32_t q = static_cast<uint32_t>(rng() % kNumQubits);
                const uint32_t kind = 1 + static_cast<uint32_t>(rng() % 3);
                probe.set_pauli(q, (kind & 1U) != 0, (kind & 2U) != 0);
            }
            probe.set_sign(false);
            REQUIRE(actual.eigenvalue(probe) == expected.eigenvalue(probe));
            REQUIRE(actual.commutes(probe.view()) == expected.commutes(probe.view()));
        }
    };

    // Flip roughly half of the initial +Z_q unit stabilizers to -Z_q across all 3 words.
    for (uint32_t q = 0; q < kNumQubits; ++q) {
        if ((rng() & 1U) != 0) {
            PauliString x(kNumQubits), neg_z(kNumQubits);
            x.set_pauli(q, true, false);
            neg_z.set_pauli(q, false, true);
            neg_z.set_sign(true);
            actual.intersect(x.view());
            expected.intersect(x.view());
            actual.insert(neg_z);
            expected.insert(neg_z);
        }
    }
    check_agreement();

    for (int step = 0; step < 80; ++step) {
        const uint64_t mode = rng() % 4;
        if (mode == 0) {
            // Pure-X multi-qubit intersection: pairs high-pivot pure-Z rows with low -Z_q
            // unit bits as well as high -Z_{q_max} unit bits with earlier rows.
            PauliString op(kNumQubits);
            const uint32_t count = 2 + static_cast<uint32_t>(rng() % 3);
            for (uint32_t k = 0; k < count; ++k) {
                op.set_pauli(static_cast<uint32_t>(rng() % kNumQubits), true, false);
            }
            actual.intersect(op.view());
            expected.intersect(op.view());
        } else if (mode == 1) {
            // General X/Y/Z intersection across words.
            PauliString op(kNumQubits);
            const uint32_t count = 1 + static_cast<uint32_t>(rng() % 4);
            for (uint32_t k = 0; k < count; ++k) {
                const uint32_t q = static_cast<uint32_t>(rng() % kNumQubits);
                const uint32_t kind = 1 + static_cast<uint32_t>(rng() % 3);
                op.set_pauli(q, (kind & 1U) != 0, (kind & 2U) != 0);
            }
            actual.intersect(op.view());
            expected.intersect(op.view());
        } else {
            // Collapse onto a commuting or anticommuting Hermitian Pauli generator
            // overlapping signed unit-Z qubits, exercising signed reduction + insertion.
            PauliString gen(kNumQubits);
            const uint32_t q_lead = static_cast<uint32_t>(rng() % kNumQubits);
            gen.set_pauli(q_lead, true, (rng() & 1U) != 0);
            for (uint32_t k = 0; k < 3; ++k) {
                const uint32_t q = static_cast<uint32_t>(rng() % kNumQubits);
                if (q != q_lead) {
                    gen.set_pauli(q, false, true);
                }
            }
            gen.set_sign((rng() & 1U) != 0);
            if (const auto eig = expected.eigenvalue(gen); eig.has_value()) {
                if (*eig) {
                    gen.add_phase(2);
                }
            } else {
                actual.intersect(gen.view());
                expected.intersect(gen.view());
            }
            actual.insert(gen);
            expected.insert(gen);
        }
        check_agreement();
    }
}

TEST_CASE("SymbolicStabilizers matches dense reference across signed unit Z and demotions",
          "[optimizer]") {
    constexpr uint32_t kNumQubits = 136;
    Xoshiro256PlusPlus rng(0x9E3779B9ULL);

    std::string source;
    // Prepare negative unit-Z stabilizers across all three 64-qubit words.
    for (uint32_t q = 0; q < kNumQubits; ++q) {
        if (q % 2 == 0 || q + 1 == kNumQubits || (rng() % 3U) != 0) {
            source += "M !" + std::to_string(q) + "\n";
            source += "CX rec[-1] " + std::to_string(q) + "\n";
        }
    }
    // Exercise symbolic promotion of negative unit-Z qubits followed by separated
    // cancelling feedback that demotes them back into unit_z_mask_/unit_z_sign_,
    // shared-symbol elimination in fixed_constraints across negative unit-Z bits,
    // and multi-word intersections hitting both Case 1 (unit pivot) and Case 2 (row pivot).
    for (int step = 0; step < 45; ++step) {
        const uint32_t slot = static_cast<uint32_t>((step / 5) % 6);
        const uint32_t a = slot * 6 + static_cast<uint32_t>(rng() % 2) * 2;
        const uint32_t b = 64 + slot * 6;
        const uint32_t c = 100 + slot * 6;
        const uint32_t d = 100 + slot * 6 + 2 + static_cast<uint32_t>((step / 5) % 2) * 2;
        switch (step % 5) {
            case 0:
                // High-qubit pure-X fault creates pure-Z pair row Z_b * Z_c (with negative signs).
                source += "E(0.2) X" + std::to_string(b) + " X" + std::to_string(c) + "\n";
                break;
            case 1:
                // Low-plus-high X fault multiplies low negative unit-Z_a by higher-pivot row on b.
                source += "E(0.2) X" + std::to_string(a) + " X" + std::to_string(b) + "\n";
                break;
            case 2:
                // Stored row on a/b plus fresh top negative unit-Z_d exercises unit-pivot s_max.
                source += "E(0.2) X" + std::to_string(a) + " X" + std::to_string(d) + "\n";
                break;
            case 3:
                // Uncorrected X measurement kicks two negative unit-Z qubits with the same
                // symbol (fixed_constraints eliminates the symbol across unit-Z bits), then
                // a later separated conditional Pauli cancels one back to unit-Z (demotion).
                source += "MX " + std::to_string(a) + "\n";
                source += "CX rec[-1] " + std::to_string(b) + "\n";
                source += "CX rec[-1] " + std::to_string(c) + "\n";
                source += "R_Z(0.125) " + std::to_string(a) + "\n";
                source += "CX rec[-1] " + std::to_string(c) + "\n";
                break;
            case 4:
                // Multi-word MPP measurement + feedback across words.
                source += "MPP X" + std::to_string(a) + "*Z" + std::to_string(b) + "*Z" +
                          std::to_string(c) + "\n";
                source += "CZ rec[-1] " + std::to_string(a) + "\n";
                break;
        }
    }

    const HirModule hir = trace(parse(source));
    REQUIRE(hir.num_qubits == kNumQubits);
    SymbolicStabilizers actual(hir);
    NaiveSymbolicStabilizers expected(hir);
    KnownStabilizers actual_fixed_direct(kNumQubits);
    NaiveKnownStabilizers expected_fixed_direct(kNumQubits);

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        actual.advance(hir, hir.ops[i]);
        expected.advance(hir, hir.ops[i]);
        actual_fixed_direct.advance(hir, hir.ops[i]);
        expected_fixed_direct.advance(hir, hir.ops[i]);

        if (i + 1 == hir.ops.size() || (i % 7 == 0)) {
            const KnownStabilizers& got = actual.fixed_constraints();
            const NaiveKnownStabilizers want = expected.fixed_constraints();
            for (int p = 0; p < 10; ++p) {
                const PauliString in_sym = want.random_in_group(kNumQubits, rng);
                REQUIRE(got.eigenvalue(in_sym) == want.eigenvalue(in_sym));
                REQUIRE(got.commutes(in_sym.view()) == want.commutes(in_sym.view()));

                const PauliString in_dir = expected_fixed_direct.random_in_group(kNumQubits, rng);
                REQUIRE(actual_fixed_direct.eigenvalue(in_dir) ==
                        expected_fixed_direct.eigenvalue(in_dir));

                PauliString probe(kNumQubits);
                for (int k = 0; k < 3; ++k) {
                    const uint32_t q = static_cast<uint32_t>(rng() % kNumQubits);
                    const uint32_t kind = 1 + static_cast<uint32_t>(rng() % 3);
                    probe.set_pauli(q, (kind & 1U) != 0, (kind & 2U) != 0);
                }
                probe.set_sign(false);
                REQUIRE(got.eigenvalue(probe) == want.eigenvalue(probe));
            }
        }
    }
}
