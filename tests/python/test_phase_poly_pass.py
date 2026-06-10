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

    def test_tohpe_pass_is_monotone_on_surface_d3_t_gate(self) -> None:
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
        assert int(hir.num_t_gates) <= int(hir_peep.num_t_gates)


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


T_GATES_SPLIT_BY_RX_CIRCUIT = """\
T 0
R_X(0.125) 0
T 1
T 2
T 0
"""

BARRIER_EDGE_CASES = {
    "t_split_by_rx": T_GATES_SPLIT_BY_RX_CIRCUIT,
    "same_qubit_both_sides": "T 0\nR_X(0.125) 0\nT 0",
    "t_split_by_rz": "T 0\nR_Z(0.125) 0\nT 1\nT 0",
    "t_split_by_rxx": "T 0\nT 1\nR_XX(0.25) 0 1\nT 0\nT 1",
    "t_split_by_rpauli": "T 0\nR_PAULI(0.125) X0\nT 1\nT 0",
    "double_phase_barrier": "T 0\nR_X(0.125) 0\nR_Z(0.125) 0\nT 1\nT 0",
    "barrier_leading": "R_X(0.125) 0\nT 0\nT 1\nT 2",
    "barrier_trailing": "T 0\nT 1\nR_X(0.125) 0",
    "t_dag_split_by_rx": "T 0\nR_X(0.125) 0\nT_DAG 1\nT 0\nT_DAG 2",
    "alternating_single_t_blocks": "T 0\nR_Z(0.0625) 0\nT 0\nR_Z(0.0625) 0\nT 0",
}

NOISE_BARRIER_CIRCUIT = """\
T 0
X_ERROR(0.001) 0
T 1
T 2
T 0
"""


def _peephole_t_count(circuit_str: str) -> int:
    hir = clifft.trace(clifft.parse(circuit_str))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates)


def _phase_poly_t_count(circuit_str: str) -> int:
    hir = clifft.trace(clifft.parse(circuit_str))
    _pass_manager().run(hir)
    return int(hir.num_t_gates)


class TestPhasePolyBarrierEdgeCases:
    @pytest.mark.parametrize("name,circuit", list(BARRIER_EDGE_CASES.items()))
    def test_preserves_statevector(self, name: str, circuit: str) -> None:
        assert_statevectors_equal(
            _statevector(circuit, optimize=True),
            _statevector(circuit, optimize=False),
            msg=name,
        )

    @pytest.mark.parametrize("name,circuit", list(BARRIER_EDGE_CASES.items()))
    def test_t_count_matches_peephole(self, name: str, circuit: str) -> None:
        assert _phase_poly_t_count(circuit) == _peephole_t_count(circuit), name

    def test_noise_barrier_preserves_t_count(self) -> None:
        assert _phase_poly_t_count(NOISE_BARRIER_CIRCUIT) == _peephole_t_count(
            NOISE_BARRIER_CIRCUIT
        )

    def test_t_split_by_rx_preserves_four_t_gates(self) -> None:
        assert _peephole_t_count(T_GATES_SPLIT_BY_RX_CIRCUIT) == 4
        assert _phase_poly_t_count(T_GATES_SPLIT_BY_RX_CIRCUIT) == 4


# op-T-mize / SOFT-style patterns: phase rotations or noise between T regions.
_QFT_4Q_LAYER = """\
R_Z(0.25) 1 0
R_Z(0.25) 2 0
R_Z(0.25) 3 0
H 0
R_Z(0.25) 2 1
R_Z(0.25) 3 1
H 1
R_Z(0.25) 3 2
H 2
H 3"""

REAL_WORLD_ODD_CASES = {
    "qft4_t_sandwich": f"T 0\n{_QFT_4Q_LAYER}\nT 1\nT 2",
    "toffoli_chain_rx_barrier": (
        "R_Z(0.25) 0 1 2\n" "R_XX(0.25) 0 1\n" "R_X(0.125) 1\n" "R_Z(0.25) 1 2\n" "T 0\nT 1\nT 2"
    ),
    "ccx_phase_rot_inject": (
        "H 2\nCNOT 1 2\nT_DAG 2\nCNOT 0 2\nT 2\nR_Z(0.125) 2\n"
        "CNOT 1 2\nT_DAG 2\nCNOT 0 2\nT_DAG 1\nT 2\nH 2\n"
        "CNOT 0 1\nT_DAG 1\nCNOT 0 1\nT 0\nT 1"
    ),
}

_SURFACE_D3_STIM = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "guide"
    / "circuits"
    / "circuit_d3_t_gate_p0.001.stim"
)

_SURFACE_FRAGMENT_CASES: dict[str, tuple[str, int]] = {
    "two_t_bursts_depolarize": (
        "T 0 3 7 9 10 12 13\n"
        "DEPOLARIZE1(0.001) 0 3 7 9 10 12 13 1 2 4 5 6 8 11 14\n"
        "T 0 3 7 9 10 12 13",
        14,
    ),
    "tdag_t_split_by_depolarize": (
        "T_DAG 0 3 7 9 10 12 13\n"
        "DEPOLARIZE1(0.001) 0 3 7 9 10 12 13 1 2 4 5 6 8 11 14\n"
        "CX 1 0 2 3 6 7 8 9 11 10 14 13\n"
        "T 0 3 7 9 10 12 13",
        14,
    ),
    "tdag3_noise_cx": (
        "CX 2 3\n"
        "DEPOLARIZE2(0.001) 2 3\n"
        "T_DAG 3\n"
        "DEPOLARIZE1(0.001) 3 0 1 2 4 5 6 7 8 9 10 11 12 13 14\n"
        "CX 2 3",
        1,
    ),
    "z_error_before_tdag_burst": (
        "Z_ERROR(0.001) 14 11 6 2 8 1\n"
        "DEPOLARIZE1(0.001) 0 3 4 5 7 9 10 12 13\n"
        "T_DAG 0 3 7 9 10 12 13\n"
        "DEPOLARIZE1(0.001) 0 3 7 9 10 12 13 1 2 4 5 6 8 11 14",
        7,
    ),
}


class TestPhasePolyRealWorldOddCases:
    @pytest.mark.parametrize("name,circuit", list(REAL_WORLD_ODD_CASES.items()))
    def test_preserves_statevector(self, name: str, circuit: str) -> None:
        assert_statevectors_equal(
            _statevector(circuit, optimize=True),
            _statevector(circuit, optimize=False),
            msg=name,
        )

    @pytest.mark.parametrize("name,circuit", list(REAL_WORLD_ODD_CASES.items()))
    def test_t_count_matches_peephole(self, name: str, circuit: str) -> None:
        assert _phase_poly_t_count(circuit) == _peephole_t_count(circuit), name

    def test_surface_d3_no_spurious_t_reduction(self) -> None:
        circuit = _SURFACE_D3_STIM.read_text()
        assert _peephole_t_count(circuit) == 29
        assert _phase_poly_t_count(circuit) == 29


class TestPhasePolySurfaceFragments:
    @pytest.mark.parametrize("name,payload", list(_SURFACE_FRAGMENT_CASES.items()))
    def test_t_count_matches_peephole(self, name: str, payload: tuple[str, int]) -> None:
        circuit, expected_t = payload
        assert _peephole_t_count(circuit) == expected_t, name
        assert _phase_poly_t_count(circuit) == expected_t, name
