"""Emit a small markdown table for the experimental MCR T-count pass.

Runs the baseline peephole pass and the experimental MCR pass on a small
set of representative circuits, then prints the before/after T counts and
basic pass statistics.

Usage:
    python mcr_t_count_evaluation.py
"""

from __future__ import annotations

import clifft


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


EXAMPLES = {
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
    "kicked_xy_block": "\n".join(
        [
            "R_XX(0.25) 0 1",
            "R_Z(0.25) 0",
            "R_Z(0.25) 1",
            "R_XX(0.25) 0 1",
            "R_YY(0.25) 0 1",
        ]
    ),
    "negative_sign_block": "\n".join(
        [
            "X 0",
            "R_XX(0.25) 0 1",
            "R_Z(0.25) 0",
            "R_Z(0.25) 1",
            "R_XX(0.25) 0 1",
            "R_YY(0.25) 0 1",
        ]
    ),
    "late_window_block": "\n".join(
        [
            *(f"R_Z(0.25) {q}" for q in range(10, 28)),
            "R_XX(0.25) 0 1",
            "R_Z(0.25) 0",
            "R_Z(0.25) 1",
            "R_XX(0.25) 0 1",
            "R_YY(0.25) 0 1",
        ]
    ),
    "two_disjoint_pair_blocks": "\n".join(_pair_block(0, 1) + _pair_block(2, 3)),
    "three_disjoint_pair_blocks": "\n".join(
        _pair_block(0, 1) + _pair_block(2, 3) + _pair_block(4, 5)
    ),
    "two_disjoint_pairs_x2": "\n".join(
        _pair_block(0, 1) + _pair_block(2, 3) + _pair_block(0, 1) + _pair_block(2, 3)
    ),
    "fermionic_swap_net_6_l3": _fermionic_swap_network(6, 3),
    "fermionic_swap_net_8_l3": _fermionic_swap_network(8, 3),
    "fermionic_swap_net_onsite_6_l3": _fermionic_swap_network_onsite(6, 3),
    "fermionic_swap_net_hubbard_6_l3": _fermionic_swap_network_hubbard(6, 3),
    "star_hub_4_l1": _star_hub_entangler(4, 1),
    "bell_pumping_r2": _bell_pumping(2),
    "inject_entangle_measure_r2": _inject_entangle_measure(2),
    "inject_bell_cultivate_r2": _inject_bell_cultivate(2),
}


def _source_t_like_ops(text: str) -> int:
    total = 0
    for line in text.splitlines():
        if "(0.25)" in line:
            total += 1
        if line.startswith("T ") or line.startswith("T_DAG "):
            total += 1
    return total


def _after_peephole(text: str) -> int:
    hir = clifft.trace(clifft.parse(text))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates)


def _after_mcr(text: str) -> tuple[int, clifft.ExperimentalMcrTCountPass]:
    hir = clifft.trace(clifft.parse(text))
    mcr = clifft.ExperimentalMcrTCountPass()
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(mcr)
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates), mcr


def main() -> None:
    header = (
        "| example | source T-like ops | after peephole | "
        "after ExperimentalMcrTCountPass + peephole | "
        "window scans | over-cap scans | quadruples | swaps | merges | "
        "T removed |"
    )
    print(header)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, text in EXAMPLES.items():
        after_mcr, stats = _after_mcr(text)
        print(
            f"| {name} | {_source_t_like_ops(text)} | {_after_peephole(text)} | {after_mcr} | "
            f"{stats.window_scans} | {stats.window_scans_over_lookahead_cap} | "
            f"{stats.quadruples_found} | {stats.swaps_applied} | {stats.merges} | "
            f"{stats.t_removed} |"
        )


if __name__ == "__main__":
    main()
