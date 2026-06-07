"""Experimental bounded MCR T-count pass tests."""

from __future__ import annotations

import numpy as np
from conftest import assert_statevectors_equal

import clifft


def _peephole_only_pm() -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    return pm


def _mcr_pipeline_pm(
    mcr_pass: clifft.ExperimentalMcrTCountPass | None = None,
) -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(mcr_pass if mcr_pass is not None else clifft.ExperimentalMcrTCountPass())
    pm.add(clifft.PeepholeFusionPass())
    return pm


def _statevector(text: str, pm: clifft.HirPassManager) -> np.ndarray:
    prog = clifft.compile(
        text,
        hir_passes=pm,
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    return np.asarray(clifft.get_statevector(prog, state))


def _t_count_after(text: str, pm: clifft.HirPassManager) -> int:
    hir = clifft.trace(clifft.parse(text))
    pm.run(hir)
    return int(hir.num_t_gates)


def _disjoint_pair_blocks(num_qubits: int, steps: int = 1) -> str:
    lines: list[str] = []
    for _ in range(steps):
        for i in range(0, num_qubits - 1, 2):
            lines.extend(
                [
                    f"R_XX(0.25) {i} {i + 1}",
                    f"R_Z(0.25) {i}",
                    f"R_Z(0.25) {i + 1}",
                    f"R_XX(0.25) {i} {i + 1}",
                    f"R_YY(0.25) {i} {i + 1}",
                ]
            )
    return "\n".join(lines)


def _pair_block(q0: int, q1: int) -> list[str]:
    return [
        f"R_XX(0.25) {q0} {q1}",
        f"R_Z(0.25) {q0}",
        f"R_Z(0.25) {q1}",
        f"R_XX(0.25) {q0} {q1}",
        f"R_YY(0.25) {q0} {q1}",
    ]


TOGGLE_SANDWICH = "\n".join(
    [
        "R_XX(0.25) 0 1",
        "R_PAULI(0.25) X0*Y1",
        "R_PAULI(0.25) Y0*X1",
        "R_XX(0.25) 0 1",
        "R_YY(0.25) 0 1",
        "R_PAULI(0.25) Y0*X1",
    ]
)

KICKED_XY_BLOCK = "\n".join(
    [
        "R_XX(0.25) 0 1",
        "R_Z(0.25) 0",
        "R_Z(0.25) 1",
        "R_XX(0.25) 0 1",
        "R_YY(0.25) 0 1",
    ]
)

NEGATIVE_SIGN_BLOCK = "\n".join(["X 0", KICKED_XY_BLOCK])

LATE_WINDOW_BLOCK = "\n".join([*(f"R_Z(0.25) {q}" for q in range(10, 28)), KICKED_XY_BLOCK])
DISJOINT_PAIRS_4 = _disjoint_pair_blocks(4, 1)
THREE_DISJOINT_PAIR_BLOCKS = "\n".join(_pair_block(0, 1) + _pair_block(2, 3) + _pair_block(4, 5))
TWO_DISJOINT_PAIRS_X2 = "\n".join(
    _pair_block(0, 1) + _pair_block(2, 3) + _pair_block(0, 1) + _pair_block(2, 3)
)
BLOCKED_GATHER_REGRESSION = "\n".join(
    [
        "R_XX(0.25) 1 0",
        "R_Y(0.25) 1",
        "R_Y(0.25) 0",
        "R_XX(0.25) 0 1",
        "R_X(0.25) 0",
        "R_ZZ(0.25) 1 0",
    ]
)
MIXED_DIRECTION_REGRESSION = "\n".join(
    [
        "R_XX(0.25) 0 1",
        "R_ZZ(0.25) 1 0",
        "S 1",
        "R_Y(0.25) 0",
        "R_X(0.25) 1",
        "R_ZZ(0.25) 1 0",
    ]
)


class TestExperimentalMcrTCountPass:
    def test_t_count_reduction_examples(self) -> None:
        examples = {
            "toggle_sandwich": (TOGGLE_SANDWICH, 6, 2),
            "kicked_xy_block": (KICKED_XY_BLOCK, 5, 3),
            "negative_sign_block": (NEGATIVE_SIGN_BLOCK, 5, 5),
            "late_window_block": (LATE_WINDOW_BLOCK, 23, 21),
            "two_disjoint_pair_blocks": (DISJOINT_PAIRS_4, 10, 6),
            "three_disjoint_pair_blocks": (THREE_DISJOINT_PAIR_BLOCKS, 15, 9),
            "two_disjoint_pairs_x2": (TWO_DISJOINT_PAIRS_X2, 16, 12),
        }

        for name, (text, after_peephole, after_mcr) in examples.items():
            assert _t_count_after(text, _peephole_only_pm()) == after_peephole, name
            assert _t_count_after(text, _mcr_pipeline_pm()) == after_mcr, name

    def test_stats_report_applied_swap(self) -> None:
        hir = clifft.trace(clifft.parse(KICKED_XY_BLOCK))
        _peephole_only_pm().run(hir)

        mcr = clifft.ExperimentalMcrTCountPass()
        pm = clifft.HirPassManager()
        pm.add(mcr)
        pm.run(hir)

        assert mcr.lookahead_cap == 16
        assert mcr.window_scans >= 1
        assert mcr.quadruples_found >= 1
        assert mcr.swaps_applied == 1
        assert mcr.merges == 1
        assert mcr.t_removed == 2

    def test_statevector_equivalence_on_reduction_examples(self) -> None:
        for text in (
            TOGGLE_SANDWICH,
            KICKED_XY_BLOCK,
            NEGATIVE_SIGN_BLOCK,
            DISJOINT_PAIRS_4,
            THREE_DISJOINT_PAIR_BLOCKS,
            TWO_DISJOINT_PAIRS_X2,
        ):
            base = _statevector(text, _peephole_only_pm())
            opt = _statevector(text, _mcr_pipeline_pm())
            assert_statevectors_equal(opt, base, msg=text)

    def test_blocked_gather_regression_preserves_statevector(self) -> None:
        base = _statevector(BLOCKED_GATHER_REGRESSION, _peephole_only_pm())
        opt = _statevector(BLOCKED_GATHER_REGRESSION, _mcr_pipeline_pm())
        assert_statevectors_equal(opt, base, msg=BLOCKED_GATHER_REGRESSION)
        assert _t_count_after(BLOCKED_GATHER_REGRESSION, _peephole_only_pm()) == 6
        assert _t_count_after(BLOCKED_GATHER_REGRESSION, _mcr_pipeline_pm()) == 6

    def test_mixed_direction_regression_preserves_statevector(self) -> None:
        base = _statevector(MIXED_DIRECTION_REGRESSION, _peephole_only_pm())
        opt = _statevector(MIXED_DIRECTION_REGRESSION, _mcr_pipeline_pm())
        assert_statevectors_equal(opt, base, msg=MIXED_DIRECTION_REGRESSION)
        assert _t_count_after(MIXED_DIRECTION_REGRESSION, _peephole_only_pm()) == 5
        assert _t_count_after(MIXED_DIRECTION_REGRESSION, _mcr_pipeline_pm()) == 5

    def test_measurement_distribution_preserved(self) -> None:
        circuit = KICKED_XY_BLOCK + "\nH 0\nM 0"
        prog_base = clifft.compile(
            circuit,
            hir_passes=_peephole_only_pm(),
            bytecode_passes=clifft.default_bytecode_pass_manager(),
        )
        prog_opt = clifft.compile(
            circuit,
            hir_passes=_mcr_pipeline_pm(),
            bytecode_passes=clifft.default_bytecode_pass_manager(),
        )

        probs_base = clifft.record_probabilities(prog_base, ["0", "1"])
        probs_opt = clifft.record_probabilities(prog_opt, ["0", "1"])
        np.testing.assert_allclose(probs_opt, probs_base, atol=1e-9, rtol=0.0)

    def test_experimental_pass_not_in_default_pipeline(self) -> None:
        default_hir = clifft.default_hir_pass_manager()
        explicit_hir = _peephole_only_pm()
        mcr_hir = _mcr_pipeline_pm()

        assert _t_count_after(KICKED_XY_BLOCK, default_hir) == _t_count_after(
            KICKED_XY_BLOCK, explicit_hir
        )
        assert _t_count_after(KICKED_XY_BLOCK, mcr_hir) < _t_count_after(
            KICKED_XY_BLOCK, explicit_hir
        )

    def test_sliding_anchor_finds_candidate_past_first_horizon(self) -> None:
        hir = clifft.trace(clifft.parse(LATE_WINDOW_BLOCK))
        _peephole_only_pm().run(hir)

        mcr = clifft.ExperimentalMcrTCountPass()
        pm = clifft.HirPassManager()
        pm.add(mcr)
        pm.run(hir)

        assert hir.num_t_gates == 21
        assert mcr.window_scans_over_lookahead_cap >= 1
        assert mcr.swaps_applied == 1
        assert mcr.t_removed == 2
