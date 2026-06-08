"""T-count evaluation: peephole vs PhasePolynomialPass.

Prints peephole and pass T counts on representative circuits. Not wired
into CI.

Usage:
    uv run python docs/guide/scripts/phase_poly_evaluation.py
"""

from __future__ import annotations

import clifft

def _pair_block(q0: int, q1: int) -> str:
    return "\n".join(
        [
            f"R_XX(0.25) {q0} {q1}",
            f"R_Z(0.25) {q0}",
            f"R_Z(0.25) {q1}",
            f"R_XX(0.25) {q0} {q1}",
            f"R_YY(0.25) {q0} {q1}",
        ]
    )


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
    "kicked_xy_block": _pair_block(0, 1),
    "two_disjoint_pair_blocks": _pair_block(0, 1) + "\n" + _pair_block(2, 3),
    "ccx_toffoli": (
        "H 2\n"
        "CNOT 1 2\n"
        "T_DAG 2\n"
        "CNOT 0 2\n"
        "T 2\n"
        "CNOT 1 2\n"
        "T_DAG 2\n"
        "CNOT 0 2\n"
        "T_DAG 1\n"
        "T 2\n"
        "H 2\n"
        "CNOT 0 1\n"
        "T_DAG 1\n"
        "CNOT 0 1\n"
        "T 0\n"
        "T 1"
    ),
    "fredkin_fragment": (
        "H 2\n"
        "CNOT 2 1\n"
        "CNOT 0 2\n"
        "H 2\n"
        "T 0\n"
        "T_DAG 1\n"
        "T 2\n"
        "CNOT 0 1\n"
        "CNOT 2 0\n"
        "CNOT 1 2\n"
        "T_DAG 0\n"
        "T 1\n"
        "T_DAG 2\n"
        "CNOT 0 1\n"
        "CNOT 2 0\n"
        "H 2\n"
        "CNOT 2 1\n"
        "H 2"
    ),
    "controlled_s": "T 0\nT 1\nCNOT 0 1\nT_DAG 1\nCNOT 0 1",
    "small_clifford_t": (
        "H 0\nT 0\nH 0\nS 0\nH 0\nT 1\nH 1\nCX 0 1\nT 0\nH 0\nT 1\nH 1"
    ),
}


def _t_after_peephole(circuit: str) -> int:
    hir = clifft.trace(clifft.parse(circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates)


def _t_after_phase_poly(circuit: str) -> tuple[int, clifft.PhasePolynomialPass]:
    hir = clifft.trace(clifft.parse(circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    poly = clifft.PhasePolynomialPass()
    pm.add(poly)
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates), poly


CONCLUSION = """
Summary:
- HIR Pauli masks supply the phase-polynomial parity table; Clifford frame
  absorbs synthesis residuals without parser or VM changes.
- MCR reordering reduces T count where peephole alone stalls (toggle_sandwich).
- TOHPE on commuting blocks complements peephole for cross-axis pair destroy.
- Follow-up: lift the 32-qubit TOHPE cap, tune MCR window bounds, op-T-mize sweep.
"""


def main() -> None:
    header = (
        f"{'example':<26} {'peephole_T':>11} {'pass_T':>8} "
        f"{'mcr_swaps':>10} {'tohpe_red':>10} {'blocks':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, circuit in EXAMPLES.items():
        t_peep = _t_after_peephole(circuit)
        t_pass, poly = _t_after_phase_poly(circuit)
        mcr = poly.mcr_stats()
        print(
            f"{name:<26} {t_peep:>11} {t_pass:>8} "
            f"{mcr['swaps_applied']:>10} {poly.t_reductions:>10} {poly.blocks_optimized:>7}"
        )
    print(CONCLUSION)


if __name__ == "__main__":
    main()
