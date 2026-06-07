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
