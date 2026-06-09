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


def _measurement_frequencies(
    text: str, pm: clifft.HirPassManager, outcomes: list[str], *, shots: int = 20_000
) -> np.ndarray:
    prog = clifft.compile(
        text,
        hir_passes=pm,
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    result = clifft.sample(prog, shots, seed=123)
    bitstrings = ["".join(str(int(bit)) for bit in row) for row in np.asarray(result.measurements)]
    counts = {bits: 0 for bits in outcomes}
    for bits in bitstrings:
        counts[bits] += 1
    return np.asarray([counts[bits] / shots for bits in outcomes], dtype=np.float64)


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


def _fermionic_swap_network(num_qubits: int, layers: int) -> str:
    lines: list[str] = []
    for layer in range(layers):
        start = layer % 2
        for q in range(start, num_qubits - 1, 2):
            lines.extend(_pair_block(q, q + 1))
    return "\n".join(lines)


def _fermionic_swap_network_onsite(num_qubits: int, layers: int) -> str:
    lines: list[str] = []
    for layer in range(layers):
        for q in range(num_qubits):
            lines.append(f"R_X(0.25) {q}")
        start = layer % 2
        for q in range(start, num_qubits - 1, 2):
            lines.extend(_pair_block(q, q + 1))
    return "\n".join(lines)


def _fermionic_swap_network_hubbard(num_qubits: int, layers: int) -> str:
    lines: list[str] = []
    for layer in range(layers):
        start = layer % 2
        for q in range(start, num_qubits - 1, 2):
            lines.extend(_pair_block(q, q + 1))
            lines.append(f"R_ZZ(0.25) {q} {q + 1}")
        for q in range(num_qubits):
            lines.append(f"R_Z(0.25) {q}")
    return "\n".join(lines)


def _star_hub_entangler(num_qubits: int, rounds: int) -> str:
    lines: list[str] = []
    for _ in range(rounds):
        for q in range(1, num_qubits):
            lines.extend(_pair_block(0, q))
    return "\n".join(lines)


def _bell_pumping(rounds: int) -> str:
    lines: list[str] = []
    for _ in range(rounds):
        lines.extend(_pair_block(0, 1))
        lines.extend(_pair_block(2, 3))
        lines.append("R_ZZ(0.25) 0 2")
        lines.append("R_ZZ(0.25) 1 3")
        lines.append("M 2")
        lines.append("M 3")
        lines.append("R 2")
        lines.append("R 3")
    return "\n".join(lines)


def _inject_entangle_measure(rounds: int) -> str:
    lines: list[str] = []
    for _ in range(rounds):
        lines.extend(_pair_block(3, 0))
        lines.extend(_pair_block(3, 1))
        lines.extend(_pair_block(3, 2))
        lines.append("M 3")
        lines.append("R 3")
    return "\n".join(lines)


def _inject_bell_cultivate(rounds: int) -> str:
    lines: list[str] = []
    for _ in range(rounds):
        lines.extend(_pair_block(2, 3))
        lines.extend(_pair_block(2, 0))
        lines.extend(_pair_block(3, 1))
        lines.append("R_ZZ(0.25) 0 1")
        lines.append("M 2")
        lines.append("M 3")
        lines.append("R 2")
        lines.append("R 3")
    return "\n".join(lines)


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
FERMIONIC_SWAP_NET_6_L3 = _fermionic_swap_network(6, 3)
FERMIONIC_SWAP_NET_8_L3 = _fermionic_swap_network(8, 3)
FERMIONIC_SWAP_NET_ONSITE_6_L3 = _fermionic_swap_network_onsite(6, 3)
FERMIONIC_SWAP_NET_HUBBARD_6_L3 = _fermionic_swap_network_hubbard(6, 3)
STAR_HUB_4_L1 = _star_hub_entangler(4, 1)
BELL_PUMPING_R2 = _bell_pumping(2)
INJECT_ENTANGLE_MEASURE_R2 = _inject_entangle_measure(2)
INJECT_BELL_CULTIVATE_R2 = _inject_bell_cultivate(2)
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
            "negative_sign_block": (NEGATIVE_SIGN_BLOCK, 5, 3),
            "late_window_block": (LATE_WINDOW_BLOCK, 23, 21),
            "two_disjoint_pair_blocks": (DISJOINT_PAIRS_4, 10, 6),
            "three_disjoint_pair_blocks": (THREE_DISJOINT_PAIR_BLOCKS, 15, 9),
            "two_disjoint_pairs_x2": (TWO_DISJOINT_PAIRS_X2, 16, 12),
            "fermionic_swap_net_6_l3": (FERMIONIC_SWAP_NET_6_L3, 40, 24),
            "fermionic_swap_net_8_l3": (FERMIONIC_SWAP_NET_8_L3, 55, 33),
            "fermionic_swap_net_onsite_6_l3": (FERMIONIC_SWAP_NET_ONSITE_6_L3, 54, 38),
            "fermionic_swap_net_hubbard_6_l3": (FERMIONIC_SWAP_NET_HUBBARD_6_L3, 62, 18),
            "star_hub_4_l1": (STAR_HUB_4_L1, 15, 9),
            "bell_pumping_r2": (BELL_PUMPING_R2, 24, 16),
            "inject_entangle_measure_r2": (INJECT_ENTANGLE_MEASURE_R2, 30, 18),
            "inject_bell_cultivate_r2": (INJECT_BELL_CULTIVATE_R2, 32, 20),
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
            FERMIONIC_SWAP_NET_6_L3,
            FERMIONIC_SWAP_NET_ONSITE_6_L3,
            FERMIONIC_SWAP_NET_HUBBARD_6_L3,
            STAR_HUB_4_L1,
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

        for measured_circuit, outcomes in (
            (BELL_PUMPING_R2, [f"{bits:04b}" for bits in range(16)]),
            (INJECT_ENTANGLE_MEASURE_R2, [f"{bits:02b}" for bits in range(4)]),
            (INJECT_BELL_CULTIVATE_R2, [f"{bits:04b}" for bits in range(16)]),
        ):
            freq_base = _measurement_frequencies(measured_circuit, _peephole_only_pm(), outcomes)
            freq_opt = _measurement_frequencies(measured_circuit, _mcr_pipeline_pm(), outcomes)
            tvd = 0.5 * np.abs(freq_opt - freq_base).sum()
            assert tvd < 0.02, f"TVD={tvd:.4f} for circuit:\n{measured_circuit}"

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
