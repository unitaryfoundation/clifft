"""Issue #40 evaluation: peephole vs ExperimentalGlobalTcountPass (MCR + TODD).

Reproduces the benchmark table from PR #123's MCR circuits and prints
before/after T counts. Not wired into CI; run manually:

    uv run python docs/guide/scripts/global_tcount_evaluation.py
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


EXAMPLES: dict[str, str] = {
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
    "negative_sign_block": "\n".join(["X 0"] + _pair_block(0, 1)),
    "two_disjoint_pair_blocks": "\n".join(_pair_block(0, 1) + _pair_block(2, 3)),
    "three_disjoint_pair_blocks": "\n".join(
        _pair_block(0, 1) + _pair_block(2, 3) + _pair_block(4, 5)
    ),
}

# Reference T counts from PR #123 (ExperimentalMcrTCountPass).
MCR_REFERENCE = {
    "toggle_sandwich": 2,
    "kicked_xy_block": 3,
    "negative_sign_block": 5,
    "two_disjoint_pair_blocks": 6,
    "three_disjoint_pair_blocks": 9,
}


def _t_after_peephole(circuit: str) -> int:
    hir = clifft.trace(clifft.parse(circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return hir.num_t_gates


def _t_after_global(circuit: str) -> tuple[int, clifft.ExperimentalGlobalTcountPass]:
    hir = clifft.trace(clifft.parse(circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    global_pass = clifft.ExperimentalGlobalTcountPass()
    pm.add(global_pass)
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return hir.num_t_gates, global_pass


def main() -> None:
    header = (
        f"{'example':<30} {'peephole_T':>11} {'global_T':>9} "
        f"{'mcr_ref':>8} {'mcr_swaps':>10} {'todd_blocks':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, circuit in EXAMPLES.items():
        t_peep = _t_after_peephole(circuit)
        t_global, gp = _t_after_global(circuit)
        stats = gp.mcr_stats()
        print(
            f"{name:<30} {t_peep:>11} {t_global:>9} "
            f"{MCR_REFERENCE[name]:>8} {stats['swaps_applied']:>10} "
            f"{gp.todd_blocks_optimized:>12}"
        )


if __name__ == "__main__":
    main()
