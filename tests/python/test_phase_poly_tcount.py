"""Experimental ExperimentalGlobalTcountPass: issue #40 evaluation."""

import numpy as np
import pytest
from conftest import assert_statevectors_equal

import clifft


def _pair_block(q0: int, q1: int) -> list[str]:
    return [
        f"R_XX(0.25) {q0} {q1}",
        f"R_Z(0.25) {q0}",
        f"R_Z(0.25) {q1}",
        f"R_XX(0.25) {q0} {q1}",
        f"R_YY(0.25) {q0} {q1}",
    ]


MCR_EXAMPLES = {
    "toggle_sandwich": "\n".join(
        [
            "R_XX(0.25) 0 1",
            "R_PAULI(0.25) X0*Y1",
            "R_PAULI(0.25) Y0*X1",
            "R_XX(0.25) 0 1",
            "R_YY(0.25) 0 1",
            "R_PAULI(0.25) Y0*X1",
        ]
    ),
    "kicked_xy_block": "\n".join(_pair_block(0, 1)),
    "two_disjoint_pair_blocks": "\n".join(_pair_block(0, 1) + _pair_block(2, 3)),
    "three_disjoint_pair_blocks": "\n".join(
        _pair_block(0, 1) + _pair_block(2, 3) + _pair_block(4, 5)
    ),
}


def _pass_manager() -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(clifft.ExperimentalGlobalTcountPass())
    pm.add(clifft.PeepholeFusionPass())
    return pm


def _statevector(circuit_str: str, *, hir_passes: clifft.HirPassManager | None) -> np.ndarray:
    prog = clifft.compile(circuit_str, hir_passes=hir_passes, bytecode_passes=None)
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    return np.array(clifft.get_statevector(prog, state))


class TestGlobalTcountStatevectorEquivalence:
    @pytest.mark.parametrize("name,circuit", list(MCR_EXAMPLES.items()))
    def test_preserves_statevector(self, name: str, circuit: str) -> None:
        base = _statevector(circuit, hir_passes=None)
        opt = _statevector(circuit, hir_passes=_pass_manager())
        assert_statevectors_equal(opt, base, msg=name)


class TestGlobalTcountReduction:
    @pytest.mark.parametrize(
        "name,circuit,expected_max_t",
        [
            ("toggle_sandwich", MCR_EXAMPLES["toggle_sandwich"], 2),
            ("kicked_xy_block", MCR_EXAMPLES["kicked_xy_block"], 3),
            ("two_disjoint_pair_blocks", MCR_EXAMPLES["two_disjoint_pair_blocks"], 6),
            ("three_disjoint_pair_blocks", MCR_EXAMPLES["three_disjoint_pair_blocks"], 9),
        ],
    )
    def test_matches_mcr_benchmark_targets(
        self, name: str, circuit: str, expected_max_t: int
    ) -> None:
        hir = clifft.trace(clifft.parse(circuit))
        pm_peep = clifft.HirPassManager()
        pm_peep.add(clifft.PeepholeFusionPass())
        pm_peep.run(hir)
        t_peep = hir.num_t_gates

        hir2 = clifft.trace(clifft.parse(circuit))
        _pass_manager().run(hir2)
        t_global = hir2.num_t_gates

        assert t_global <= t_peep, f"{name}: T increased"
        assert t_global <= expected_max_t, f"{name}: T={t_global} expected <= {expected_max_t}"
