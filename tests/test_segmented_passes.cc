// Segmented pass execution: fences as structural optimization barriers.
//
// run_segmented presents each maximal fence-free segment to the passes as if
// it were the whole module, so no pass can observe, fuse, or move operations
// across a fence. These tests pin the barrier behavior, the no-fence
// equivalence with run(), and the semantic equivalence (exact and
// statistical) of fenced and unfenced compilations of the same circuit.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace clifft;

namespace {

bool is_noise_op(const HeisenbergOp& op) {
    return op.op_type() == OpType::NOISE;
}

bool is_noise_instr(const Instruction& in) {
    return in.opcode == Opcode::OP_NOISE;
}

bool is_measure_op(const HeisenbergOp& op) {
    return op.op_type() == OpType::MEASURE;
}

bool is_measure_instr(const Instruction& in) {
    switch (in.opcode) {
        case Opcode::OP_MEAS_DORMANT_STATIC:
        case Opcode::OP_MEAS_DORMANT_RANDOM:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
        case Opcode::OP_SWAP_MEAS_INTERFERE:
            return true;
        default:
            return false;
    }
}

enum class Fences : uint8_t { None, AtNoise, AtMeasurements };

// Compile through the default pass pipelines, optionally segmented.
CompiledModule compile_with(const Circuit& circuit, Fences fences) {
    HirModule hir = trace(circuit);
    HirPassManager pm = default_hir_pass_manager();
    switch (fences) {
        case Fences::None:
            pm.run(hir);
            break;
        case Fences::AtNoise:
            pm.run_segmented(hir, is_noise_op);
            break;
        case Fences::AtMeasurements:
            pm.run_segmented(hir, is_measure_op);
            break;
    }
    CompiledModule mod = lower(hir);
    BytecodePassManager bpm = default_bytecode_pass_manager();
    switch (fences) {
        case Fences::None:
            bpm.run(mod);
            break;
        case Fences::AtNoise:
            bpm.run_segmented(mod, is_noise_instr);
            break;
        case Fences::AtMeasurements:
            bpm.run_segmented(mod, is_measure_instr);
            break;
    }
    return mod;
}

size_t count_opcode(const CompiledModule& m, Opcode op) {
    size_t n = 0;
    for (const Instruction& in : m.bytecode) {
        if (in.opcode == op) {
            ++n;
        }
    }
    return n;
}

}  // namespace

TEST_CASE("run_segmented: with no fences it matches run exactly") {
    Circuit c = parse(
        "H 0\nT 0\nX_ERROR(0.1) 0\nCX 0 1\nT 1\nT 1\nX_ERROR(0.05) 1\nM 0\nM 1\n"
        "DETECTOR rec[-1] rec[-2]\n");

    HirModule plain_hir = trace(c);
    HirPassManager pm1 = default_hir_pass_manager();
    pm1.run(plain_hir);

    HirModule seg_hir = trace(c);
    HirPassManager pm2 = default_hir_pass_manager();
    pm2.run_segmented(seg_hir, [](const HeisenbergOp&) { return false; });

    REQUIRE(seg_hir.ops.size() == plain_hir.ops.size());
    for (size_t i = 0; i < seg_hir.ops.size(); ++i) {
        REQUIRE(seg_hir.ops[i].op_type() == plain_hir.ops[i].op_type());
    }

    CompiledModule plain = lower(plain_hir);
    BytecodePassManager bpm1 = default_bytecode_pass_manager();
    bpm1.run(plain);

    CompiledModule seg = lower(seg_hir);
    BytecodePassManager bpm2 = default_bytecode_pass_manager();
    bpm2.run_segmented(seg, [](const Instruction&) { return false; });

    REQUIRE(seg.bytecode.size() == plain.bytecode.size());
    for (size_t i = 0; i < seg.bytecode.size(); ++i) {
        REQUIRE(seg.bytecode[i].opcode == plain.bytecode[i].opcode);
    }
    // Identical instruction streams draw identically: same-seed samples match.
    SampleResult a = sample(plain, 64, 7);
    SampleResult b = sample(seg, 64, 7);
    REQUIRE(a.measurements == b.measurements);
    REQUIRE(a.detectors == b.detectors);
}

TEST_CASE("run_segmented: a fence blocks T-pair fusion across it") {
    // The two T 0 ops fuse (and are absorbed) when the peephole sees them
    // together; a fence at the intervening noise op splits them into separate
    // segments, so both survive.
    Circuit c = parse("T 0\nX_ERROR(0.01) 1\nT 0\nM 0\n");

    HirModule plain = trace(c);
    HirPassManager pm1 = default_hir_pass_manager();
    pm1.run(plain);

    HirModule fenced = trace(c);
    HirPassManager pm2 = default_hir_pass_manager();
    pm2.run_segmented(fenced, is_noise_op);

    REQUIRE(plain.num_t_gates() == 0);   // fused across the commuting noise
    REQUIRE(fenced.num_t_gates() == 2);  // fence keeps them apart
}

TEST_CASE("run_segmented: a fence blocks noise-block coalescing across it") {
    // Both noise sites on the measured qubit, so the squeeze pass cannot
    // bubble the measurement between them and they stay adjacent.
    Circuit c = parse("H 0\nX_ERROR(0.1) 0\nX_ERROR(0.2) 0\nM 0\n");

    CompiledModule plain = compile_with(c, Fences::None);
    REQUIRE(count_opcode(plain, Opcode::OP_NOISE_BLOCK) == 1);
    REQUIRE(count_opcode(plain, Opcode::OP_NOISE) == 0);

    // Fencing at every noise instruction removes them from pass visibility
    // entirely, so nothing coalesces.
    CompiledModule fenced = compile_with(c, Fences::AtNoise);
    REQUIRE(count_opcode(fenced, Opcode::OP_NOISE_BLOCK) == 0);
    REQUIRE(count_opcode(fenced, Opcode::OP_NOISE) == 2);
}

TEST_CASE("run_segmented: fenced compilation preserves exact record probabilities") {
    // Noise-free and measurement-fenced, so every record's probability is
    // exactly computable on both compilations. Fences change the instruction
    // stream (fusion and reordering are segment-local) but must not change
    // the distribution.
    Circuit c =
        parse("H 0\nT 0\nCX 0 1\nS 1\nT 1\nM 0\nH 1\nT 1\nH 1\nM 1\nCX 1 2\nH 2\nT 2\nH 2\nM 2\n");

    CompiledModule plain = compile_with(c, Fences::None);
    CompiledModule fenced = compile_with(c, Fences::AtMeasurements);

    // All 2^3 records at once.
    const size_t num_records = 8;
    std::vector<uint8_t> records(num_records * 3);
    for (size_t r = 0; r < num_records; ++r) {
        for (size_t b = 0; b < 3; ++b) {
            records[r * 3 + b] = static_cast<uint8_t>((r >> b) & 1);
        }
    }

    std::vector<double> lp_plain = record_probabilities(plain, records, num_records);
    std::vector<double> lp_fenced = record_probabilities(fenced, records, num_records);
    REQUIRE(lp_plain.size() == num_records);

    double total = 0.0;
    for (size_t r = 0; r < num_records; ++r) {
        if (std::isinf(lp_plain[r])) {
            REQUIRE(std::isinf(lp_fenced[r]));
            continue;
        }
        REQUIRE(std::abs(lp_plain[r] - lp_fenced[r]) < 1e-9);
        total += std::exp(lp_plain[r]);
    }
    REQUIRE(std::abs(total - 1.0) < 1e-9);
}

TEST_CASE("run_segmented: fenced compilation of a noisy circuit matches statistically") {
    // A two-round repetition-code-ish circuit with noise everywhere; fenced
    // at every noise site. Same physics, different instruction stream: the
    // detector-event fractions must agree within a generous band.
    Circuit c = parse(
        "X_ERROR(0.2) 0 1 2\n"
        "CX 0 1 2 1\nDEPOLARIZE2(0.1) 0 1\nMR 1\nDETECTOR rec[-1]\n"
        "CX 0 1 2 1\nDEPOLARIZE2(0.1) 2 1\nMR 1\nDETECTOR rec[-1] rec[-2]\n"
        "M 0 2\nDETECTOR rec[-1] rec[-2]\n");

    CompiledModule plain = compile_with(c, Fences::None);
    CompiledModule fenced = compile_with(c, Fences::AtNoise);
    REQUIRE(fenced.peak_rank == plain.peak_rank);  // Clifford circuit: both zero
    REQUIRE(fenced.num_measurements == plain.num_measurements);
    REQUIRE(fenced.num_detectors == plain.num_detectors);

    constexpr uint32_t kShots = 20000;
    SampleResult a = sample(plain, kShots, 3);
    SampleResult b = sample(fenced, kShots, 5);
    REQUIRE(a.detectors.size() == b.detectors.size());

    for (uint32_t d = 0; d < plain.num_detectors; ++d) {
        size_t ones_a = 0;
        size_t ones_b = 0;
        for (uint32_t s = 0; s < kShots; ++s) {
            ones_a += a.detectors[s * plain.num_detectors + d];
            ones_b += b.detectors[s * plain.num_detectors + d];
        }
        const double fa = static_cast<double>(ones_a) / kShots;
        const double fb = static_cast<double>(ones_b) / kShots;
        // ~7 sigma for p in [0.1, 0.9] at 20k shots.
        REQUIRE(std::abs(fa - fb) < 0.025);
    }
}
