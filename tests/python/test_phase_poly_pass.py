"""Tests for PhasePolynomialPass.

Validates that MCR reordering plus TOHPE on commuting T-gate blocks never
increases T count and preserves statevectors on small Clifford+T circuits.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import assert_statevectors_equal, random_clifford_t_circuit

import clifft

_EVAL_SCRIPTS = Path(__file__).resolve().parents[2] / "docs" / "guide" / "scripts"
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from phase_poly_evaluation import _build_real_world_examples  # noqa: E402


def _pair_block(q0: int, q1: int) -> list[str]:
    return [
        f"R_XX(0.25) {q0} {q1}",
        f"R_Z(0.25) {q0}",
        f"R_Z(0.25) {q1}",
        f"R_XX(0.25) {q0} {q1}",
        f"R_YY(0.25) {q0} {q1}",
    ]


EVAL_EXAMPLES = {
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
    "ccx_toffoli": (
        "H 2\nCNOT 1 2\nT_DAG 2\nCNOT 0 2\nT 2\nCNOT 1 2\nT_DAG 2\nCNOT 0 2\n"
        "T_DAG 1\nT 2\nH 2\nCNOT 0 1\nT_DAG 1\nCNOT 0 1\nT 0\nT 1"
    ),
}


def _pass_manager(poly: clifft.PhasePolynomialPass | None = None) -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(poly if poly is not None else clifft.PhasePolynomialPass())
    pm.add(clifft.PeepholeFusionPass())
    return pm


def _statevector(circuit_str: str, *, optimize: bool) -> np.ndarray:
    prog = clifft.compile(
        circuit_str,
        hir_passes=_pass_manager() if optimize else None,
        bytecode_passes=None,
    )
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    return np.array(clifft.get_statevector(prog, state))


class TestPhasePolyStatevectorEquivalence:
    @pytest.mark.parametrize("name,circuit", list(EVAL_EXAMPLES.items()))
    def test_preserves_statevector(self, name: str, circuit: str) -> None:
        assert_statevectors_equal(
            _statevector(circuit, optimize=True),
            _statevector(circuit, optimize=False),
            msg=name,
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_clifford_t_circuits(self, seed: int) -> None:
        circuit = random_clifford_t_circuit(4, depth=30, seed=seed)
        assert_statevectors_equal(
            _statevector(circuit, optimize=True),
            _statevector(circuit, optimize=False),
            msg=f"seed={seed}",
        )


class TestPhasePolyPerPhasePasses:
    def test_mcr_pass_reduces_toggle_sandwich(self) -> None:
        circuit = EVAL_EXAMPLES["toggle_sandwich"]
        hir = clifft.trace(clifft.parse(circuit))
        pm = clifft.HirPassManager()
        pm.add(clifft.PeepholeFusionPass())
        pm.add(clifft.McrTcountPass())
        pm.add(clifft.PeepholeFusionPass())
        pm.run(hir)
        assert int(hir.num_t_gates) == 2

    def test_tohpe_pass_reduces_surface_d3_t_gate(self) -> None:
        stim_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "guide"
            / "circuits"
            / "circuit_d3_t_gate_p0.001.stim"
        )
        circuit = stim_path.read_text()
        hir_peep = clifft.trace(clifft.parse(circuit))
        pm_peep = clifft.HirPassManager()
        pm_peep.add(clifft.PeepholeFusionPass())
        pm_peep.run(hir_peep)

        hir = clifft.trace(clifft.parse(circuit))
        pm = clifft.HirPassManager()
        pm.add(clifft.PeepholeFusionPass())
        pm.add(clifft.TohpePhasePass())
        pm.add(clifft.PeepholeFusionPass())
        pm.run(hir)
        assert int(hir.num_t_gates) < int(hir_peep.num_t_gates)


class TestPhasePolyMcrReduction:
    def test_toggle_sandwich_reduces_t_count(self) -> None:
        circuit = EVAL_EXAMPLES["toggle_sandwich"]
        hir_peep = clifft.trace(clifft.parse(circuit))
        pm_peep = clifft.HirPassManager()
        pm_peep.add(clifft.PeepholeFusionPass())
        pm_peep.run(hir_peep)

        poly = clifft.PhasePolynomialPass()
        hir_opt = clifft.trace(clifft.parse(circuit))
        _pass_manager(poly).run(hir_opt)

        assert int(hir_opt.num_t_gates) < int(hir_peep.num_t_gates)
        assert poly.mcr_stats()["swaps_applied"] >= 1


class TestPhasePolyTcountMonotonicity:
    @pytest.mark.parametrize("name,circuit", list(EVAL_EXAMPLES.items()))
    def test_never_increases_t_count(self, name: str, circuit: str) -> None:
        hir_peep = clifft.trace(clifft.parse(circuit))
        pm_peep = clifft.HirPassManager()
        pm_peep.add(clifft.PeepholeFusionPass())
        pm_peep.run(hir_peep)

        hir_opt = clifft.trace(clifft.parse(circuit))
        _pass_manager().run(hir_opt)

        assert int(hir_opt.num_t_gates) <= int(hir_peep.num_t_gates), name


REAL_WORLD_CASES = list(_build_real_world_examples())


class TestPhasePolyRealWorldMonotonicity:
    @pytest.mark.parametrize("name,category,builder", REAL_WORLD_CASES)
    def test_never_increases_t_count(
        self, name: str, category: str, builder: Callable[[], str]
    ) -> None:
        circuit = builder()
        hir_peep = clifft.trace(clifft.parse(circuit))
        pm_peep = clifft.HirPassManager()
        pm_peep.add(clifft.PeepholeFusionPass())
        pm_peep.run(hir_peep)

        hir_opt = clifft.trace(clifft.parse(circuit))
        _pass_manager().run(hir_opt)

        assert int(hir_opt.num_t_gates) <= int(hir_peep.num_t_gates), f"{name} ({category})"
