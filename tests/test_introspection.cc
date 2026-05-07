#include "clifft/backend/backend.h"
#include "clifft/frontend/hir.h"
#include "clifft/util/introspection.h"
#include "clifft/util/pauli_arena.h"

#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using clifft::ControllingMeasIdx;
using clifft::DetectorIdx;
using clifft::ExpectedParity;
using clifft::ExpValIdx;
using clifft::format_hir_op;
using clifft::format_instruction;
using clifft::format_pauli_mask;
using clifft::HirModule;
using clifft::Instruction;
using clifft::make_apply_pauli;
using clifft::make_array_cnot;
using clifft::make_array_cz;
using clifft::make_array_h;
using clifft::make_array_multi_cnot;
using clifft::make_array_multi_cz;
using clifft::make_array_rot;
using clifft::make_array_s;
using clifft::make_array_s_dag;
using clifft::make_array_swap;
using clifft::make_array_t;
using clifft::make_array_t_dag;
using clifft::make_array_u2;
using clifft::make_array_u4;
using clifft::make_detector;
using clifft::make_exp_val;
using clifft::make_expand;
using clifft::make_expand_rot;
using clifft::make_expand_t;
using clifft::make_expand_t_dag;
using clifft::make_frame_cnot;
using clifft::make_frame_cz;
using clifft::make_frame_h;
using clifft::make_frame_s;
using clifft::make_frame_s_dag;
using clifft::make_frame_swap;
using clifft::make_meas;
using clifft::make_noise;
using clifft::make_noise_block;
using clifft::make_observable;
using clifft::make_postselect;
using clifft::make_readout_noise;
using clifft::make_swap_meas_interfere;
using clifft::MeasRecordIdx;
using clifft::NoiseSiteIdx;
using clifft::ObservableIdx;
using clifft::op_type_to_str;
using clifft::Opcode;
using clifft::opcode_to_str;
using clifft::OpType;
using clifft::PauliMaskArena;
using clifft::PauliMaskHandle;
using clifft::ReadoutNoiseIdx;

TEST_CASE("introspection formats Pauli masks") {
    PauliMaskArena arena(130, 3);

    REQUIRE(format_pauli_mask(arena.at(PauliMaskHandle{0})) == "+I");

    arena.mut_at(PauliMaskHandle{1}).set_sign(true);
    REQUIRE(format_pauli_mask(arena.at(PauliMaskHandle{1})) == "-I");

    auto mask = arena.mut_at(PauliMaskHandle{2});
    mask.x().bit_set(0, true);
    mask.z().bit_set(1, true);
    mask.x().bit_set(65, true);
    mask.z().bit_set(65, true);
    mask.set_sign(true);

    REQUIRE(format_pauli_mask(arena.at(PauliMaskHandle{2})) == "-X0*Z1*Y65");
}

TEST_CASE("introspection formats every HIR op kind") {
    HirModule hir(130, 6);

    hir.append_tgate(false, [](auto mask) { mask.x().bit_set(0, true); });
    hir.append_tgate(true, [](auto mask) { mask.z().bit_set(1, true); });
    auto& measure = hir.append_measure(MeasRecordIdx{7}, [](auto mask) {
        mask.x().bit_set(65, true);
        mask.z().bit_set(65, true);
    });
    measure.set_hidden(true);
    hir.append_conditional(ControllingMeasIdx{3}, [](auto mask) {
        mask.x().bit_set(2, true);
        mask.z().bit_set(5, true);
    });
    hir.append_phase_rotation(0.125, [](auto mask) { mask.z().bit_set(4, true); });
    hir.append_exp_val(ExpValIdx{2}, [](auto mask) { mask.x().bit_set(9, true); });
    hir.append_noise(NoiseSiteIdx{11});
    hir.append_readout_noise(ReadoutNoiseIdx{12});
    hir.append_detector(DetectorIdx{13});
    hir.append_observable(ObservableIdx{14}, 15);

    const std::vector<std::string> expected = {
        "T +X0",
        "T_DAG +Z1",
        "MEASURE +Y65 -> rec[7] (hidden)",
        "IF rec[3] THEN +X2*Z5",
        "PHASE_ROTATION +Z4 alpha=0.125",
        "EXP_VAL +X9 -> exp[2]",
        "NOISE site=11",
        "READOUT_NOISE entry=12",
        "DETECTOR target_list=13",
        "OBSERVABLE index=14 target_list=15",
    };

    REQUIRE(hir.ops.size() == expected.size());
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const auto& op = hir.ops[i];
        auto mask = op.has_mask() ? std::optional{hir.mask_view(op)} : std::nullopt;
        REQUIRE(format_hir_op(op, mask) == expected[i]);
        REQUIRE(op_type_to_str(op.op_type()) != "UNKNOWN");
    }

    REQUIRE(op_type_to_str(OpType::NUM_OP_TYPES) == "UNKNOWN");
}

TEST_CASE("introspection formats every VM opcode") {
    Instruction signed_meas = make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 21, 22, true);
    signed_meas.flags |= Instruction::FLAG_IDENTITY;

    const std::vector<std::pair<Instruction, std::string>> cases = {
        {make_frame_cnot(1, 2), "OP_FRAME_CNOT 1, 2"},
        {make_frame_cz(2, 3), "OP_FRAME_CZ 2, 3"},
        {make_frame_h(4), "OP_FRAME_H 4"},
        {make_frame_s(5), "OP_FRAME_S 5"},
        {make_frame_s_dag(6), "OP_FRAME_S_DAG 6"},
        {make_frame_swap(7, 8), "OP_FRAME_SWAP 7, 8"},
        {make_array_cnot(9, 10), "OP_ARRAY_CNOT 9, 10"},
        {make_array_cz(10, 11), "OP_ARRAY_CZ 10, 11"},
        {make_array_swap(11, 12), "OP_ARRAY_SWAP 11, 12"},
        {make_array_multi_cnot(13, 0xa5), "OP_ARRAY_MULTI_CNOT target=13 ctrl_mask=0xa5"},
        {make_array_multi_cz(14, 0x5a), "OP_ARRAY_MULTI_CZ ctrl=14 target_mask=0x5a"},
        {make_array_h(15), "OP_ARRAY_H 15"},
        {make_array_s(16), "OP_ARRAY_S 16"},
        {make_array_s_dag(17), "OP_ARRAY_S_DAG 17"},
        {make_array_t(18), "OP_ARRAY_T 18"},
        {make_array_t_dag(19), "OP_ARRAY_T_DAG 19"},
        {make_array_rot(20, 0.25, -0.5), "OP_ARRAY_ROT 20 z=(0.25, -0.5)"},
        {make_array_u2(21, 33), "OP_ARRAY_U2 21 cp_idx=33"},
        {make_array_u4(22, 23, 44), "OP_ARRAY_U4 22, 23 cp_idx=44"},
        {make_expand(24), "OP_EXPAND 24"},
        {make_expand_t(25), "OP_EXPAND_T 25"},
        {make_expand_t_dag(26), "OP_EXPAND_T_DAG 26"},
        {make_expand_rot(27, 0.125, 0.875), "OP_EXPAND_ROT 27 z=(0.125, 0.875)"},
        {signed_meas, "OP_MEAS_DORMANT_STATIC 21 -> rec[22] (invert) (identity)"},
        {make_meas(Opcode::OP_MEAS_DORMANT_RANDOM, 28, 29, false),
         "OP_MEAS_DORMANT_RANDOM 28 -> rec[29]"},
        {make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL, 30, 31, false),
         "OP_MEAS_ACTIVE_DIAGONAL 30 -> rec[31]"},
        {make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE, 32, 33, true),
         "OP_MEAS_ACTIVE_INTERFERE 32 -> rec[33] (invert)"},
        {make_swap_meas_interfere(34, 35, 36, true),
         "OP_SWAP_MEAS_INTERFERE swap(34,35) meas_idx=36 (sign)"},
        {make_apply_pauli(37, 38), "OP_APPLY_PAULI cp_mask=37 if rec[38]"},
        {make_noise(39), "OP_NOISE cp_site=39"},
        {make_noise_block(40, 5), "OP_NOISE_BLOCK sites=[40..45)"},
        {make_readout_noise(41), "OP_READOUT_NOISE cp_entry=41"},
        {make_detector(42, 43, ExpectedParity::One), "OP_DETECTOR cp_targets=42 -> det[43]"},
        {make_postselect(44, 45, ExpectedParity::Zero), "OP_POSTSELECT cp_targets=44 -> det[45]"},
        {make_observable(46, 47), "OP_OBSERVABLE cp_targets=46 -> obs[47]"},
        {make_exp_val(48, 49), "OP_EXP_VAL cp=48 -> exp[49]"},
    };

    for (const auto& [inst, expected] : cases) {
        REQUIRE(format_instruction(inst) == expected);
        REQUIRE(opcode_to_str(inst.opcode) != "UNKNOWN");
    }

    Instruction unknown{};
    unknown.opcode = static_cast<Opcode>(255);
    REQUIRE(format_instruction(unknown) == "UNKNOWN ");
    REQUIRE(opcode_to_str(Opcode::NUM_OPCODES) == "UNKNOWN");
}
