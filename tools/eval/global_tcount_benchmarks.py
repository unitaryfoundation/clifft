"""Benchmark circuit corpus for issue #40 global T-count evaluation.

Circuits are expressed in Clifft's Stim-compatible format. The MCR-family
examples follow the commuting-block structure used in op-T-mize style
benchmarks; larger circuits approximate synthesis workloads (QFT layers,
Toffoli chains) that Clifft is likely to see in near-Clifford simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCircuit:
    name: str
    category: str
    circuit: str
    max_qubits_for_statevector: int = 10


def _pair_block(q0: int, q1: int) -> list[str]:
    return [
        f"R_XX(0.25) {q0} {q1}",
        f"R_Z(0.25) {q0}",
        f"R_Z(0.25) {q1}",
        f"R_XX(0.25) {q0} {q1}",
        f"R_YY(0.25) {q0} {q1}",
    ]


def qft_layer(n: int) -> str:
    lines: list[str] = []
    for target in range(n):
        for control in range(target + 1, n):
            lines.append(f"R_Z(0.25) {control} {target}")
        if target < n - 1:
            lines.append(f"H {target}")
    if n > 0:
        lines.append(f"H {n - 1}")
    return "\n".join(lines)


def toffoli_chain(n_toffoli: int, n_qubits: int) -> str:
    lines: list[str] = []
    for i in range(n_toffoli):
        a = i % (n_qubits - 2)
        b = (i + 1) % (n_qubits - 2)
        t = (i + 2) % n_qubits
        lines.append(f"R_Z(0.25) {a} {b} {t}")
        lines.append(f"R_XX(0.25) {a} {b}")
        lines.append(f"R_Z(0.25) {b} {t}")
    return "\n".join(lines)


BENCHMARKS: list[BenchmarkCircuit] = [
    BenchmarkCircuit(
        name="toggle_sandwich",
        category="mcr",
        circuit="\n".join(
            [
                "R_XX(0.25) 0 1",
                "R_PAULI(0.25) X0*Y1",
                "R_PAULI(0.25) Y0*X1",
                "R_XX(0.25) 0 1",
                "R_YY(0.25) 0 1",
                "R_PAULI(0.25) Y0*X1",
            ]
        ),
    ),
    BenchmarkCircuit(
        name="kicked_xy_block",
        category="mcr",
        circuit="\n".join(_pair_block(0, 1)),
    ),
    BenchmarkCircuit(
        name="negative_sign_block",
        category="mcr",
        circuit="\n".join(["X 0"] + _pair_block(0, 1)),
    ),
    BenchmarkCircuit(
        name="two_disjoint_pair_blocks",
        category="mcr",
        circuit="\n".join(_pair_block(0, 1) + _pair_block(2, 3)),
        max_qubits_for_statevector=6,
    ),
    BenchmarkCircuit(
        name="three_disjoint_pair_blocks",
        category="mcr",
        circuit="\n".join(_pair_block(0, 1) + _pair_block(2, 3) + _pair_block(4, 5)),
        max_qubits_for_statevector=8,
    ),
    BenchmarkCircuit(
        name="qft_4q",
        category="synthesis",
        circuit=qft_layer(4),
        max_qubits_for_statevector=6,
    ),
    BenchmarkCircuit(
        name="qft_6q",
        category="synthesis",
        circuit=qft_layer(6),
        max_qubits_for_statevector=8,
    ),
    BenchmarkCircuit(
        name="toffoli_chain_8q",
        category="synthesis",
        circuit=toffoli_chain(12, 8),
        max_qubits_for_statevector=8,
    ),
]
