"""Evaluate ExperimentalMcrTCountPass on a broader circuit corpus.

Mixes Qiskit library circuits with hand-authored Pauli-rotation workloads that
are representative of chemistry / Trotterized near-Clifford structure.

Usage:
    uv run python docs/guide/scripts/mcr_t_count_corpus_evaluation.py
"""

from __future__ import annotations

from dataclasses import dataclass

import clifft

try:
    from qiskit import transpile
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import (
        CDKMRippleCarryAdder,
        IntegerComparator,
        MCXGate,
        WeightedAdder,
    )
except ImportError as exc:  # pragma: no cover - docs/dev helper
    raise SystemExit(
        "This script requires qiskit. Install dev dependencies with `uv sync --group dev`."
    ) from exc


_ALLOWED_QISKIT_GATES = frozenset(
    {"h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx", "cz", "swap", "measure", "barrier"}
)


@dataclass(frozen=True)
class CorpusCase:
    family: str
    name: str
    stim_text: str


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


def _hubbard_chain(num_qubits: int, layers: int) -> str:
    lines: list[str] = []
    for _ in range(layers):
        for q in range(num_qubits - 1):
            lines.extend(
                [
                    f"R_XX(0.25) {q} {q + 1}",
                    f"R_Z(0.25) {q}",
                    f"R_Z(0.25) {q + 1}",
                    f"R_YY(0.25) {q} {q + 1}",
                    f"R_ZZ(0.25) {q} {q + 1}",
                ]
            )
        for q in range(num_qubits):
            lines.append(f"R_X(0.25) {q}")
    return "\n".join(lines)


def _qaoa_ring(num_qubits: int, p_layers: int) -> str:
    lines = [f"H {q}" for q in range(num_qubits)]
    for _ in range(p_layers):
        for q in range(num_qubits):
            lines.append(f"R_X(0.25) {q}")
        for q in range(num_qubits):
            lines.append(f"R_ZZ(0.25) {q} {(q + 1) % num_qubits}")
    return "\n".join(lines)


def _double_excitation_4q(reps: int) -> str:
    block = [
        "R_PAULI(0.25) X0*Y1*Y2*X3",
        "R_PAULI(0.25) Y0*X1*Y2*X3",
        "R_PAULI(0.25) Y0*Y1*X2*X3",
        "R_PAULI(0.25) X0*X1*X2*X3",
        "R_PAULI(0.25) X0*Y1*X2*Y3",
        "R_PAULI(0.25) Y0*X1*X2*Y3",
        "R_PAULI(0.25) Y0*Y1*Y2*Y3",
        "R_PAULI(0.25) X0*X1*Y2*Y3",
    ]
    return "\n".join(block * reps)


def _qiskit_to_stim(qc: QuantumCircuit) -> str:
    lines: list[str] = []
    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name == "h":
            lines.extend(f"H {q}" for q in qubits)
        elif name == "s":
            lines.extend(f"S {q}" for q in qubits)
        elif name == "sdg":
            lines.extend(f"S_DAG {q}" for q in qubits)
        elif name == "t":
            lines.extend(f"T {q}" for q in qubits)
        elif name == "tdg":
            lines.extend(f"T_DAG {q}" for q in qubits)
        elif name == "x":
            lines.extend(f"X {q}" for q in qubits)
        elif name == "y":
            lines.extend(f"Y {q}" for q in qubits)
        elif name == "z":
            lines.extend(f"Z {q}" for q in qubits)
        elif name == "cx":
            lines.append(f"CX {qubits[0]} {qubits[1]}")
        elif name == "cz":
            lines.append(f"CZ {qubits[0]} {qubits[1]}")
        elif name == "swap":
            lines.append(f"SWAP {qubits[0]} {qubits[1]}")
        elif name == "measure":
            lines.append(f"M {qubits[0]}")
        elif name == "barrier":
            pass
        else:  # pragma: no cover - defensive guard for helper script
            raise ValueError(f"Unsupported transpiled gate: {name}")
    return "\n".join(lines)


def _qiskit_case(family: str, name: str, qc: QuantumCircuit) -> CorpusCase:
    tqc = transpile(
        qc,
        basis_gates=["h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx", "cz", "swap"],
        optimization_level=0,
    )
    unsupported = sorted(
        {
            inst.operation.name
            for inst in tqc.data
            if inst.operation.name not in _ALLOWED_QISKIT_GATES
        }
    )
    if unsupported:
        raise ValueError(f"{family}/{name} transpiled to unsupported gates: {unsupported}")
    return CorpusCase(family=family, name=name, stim_text=_qiskit_to_stim(tqc))


def _count_t_after(text: str, use_mcr: bool) -> int:
    hir = clifft.trace(clifft.parse(text))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    if use_mcr:
        pm.add(clifft.ExperimentalMcrTCountPass())
        pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates)


def _source_t_like_ops(text: str) -> int:
    total = 0
    for line in text.splitlines():
        if "(0.25)" in line:
            total += 1
        if line.startswith("T ") or line.startswith("T_DAG "):
            total += 1
    return total


def _corpus() -> list[CorpusCase]:
    cases = [
        _qiskit_case("ripple_adder", "full_3", CDKMRippleCarryAdder(3, kind="full")),
        _qiskit_case("ripple_adder", "full_4", CDKMRippleCarryAdder(4, kind="full")),
        _qiskit_case("comparator", "n4_v7", IntegerComparator(4, value=7, geq=True)),
        _qiskit_case("comparator", "n5_v13", IntegerComparator(5, value=13, geq=True)),
        _qiskit_case("weighted_adder", "n3_weights_1_2_3", WeightedAdder(3, [1, 2, 3])),
    ]

    for num_controls in (4, 5):
        qc = QuantumCircuit(num_controls + 2, num_controls + 2)
        qc.append(MCXGate(num_controls), list(range(num_controls + 1)))
        qc.measure(range(num_controls + 2), range(num_controls + 2))
        cases.append(_qiskit_case("mcx", f"controls_{num_controls}", qc))

    cases.extend(
        [
            CorpusCase("fermionic_swap_net", "n6_l3", _fermionic_swap_network(6, 3)),
            CorpusCase("fermionic_swap_net", "n8_l3", _fermionic_swap_network(8, 3)),
            CorpusCase("fermionic_swap_net_onsite", "n6_l3", _fermionic_swap_network_onsite(6, 3)),
            CorpusCase("fermionic_swap_net_onsite", "n8_l3", _fermionic_swap_network_onsite(8, 3)),
            CorpusCase(
                "fermionic_swap_net_hubbard", "n6_l3", _fermionic_swap_network_hubbard(6, 3)
            ),
            CorpusCase(
                "fermionic_swap_net_hubbard", "n8_l3", _fermionic_swap_network_hubbard(8, 3)
            ),
            CorpusCase("star_hub_entangler", "n4_l1", _star_hub_entangler(4, 1)),
            CorpusCase("bell_pumping", "rounds_2", _bell_pumping(2)),
            CorpusCase("inject_entangle_measure", "rounds_2", _inject_entangle_measure(2)),
            CorpusCase("inject_bell_cultivate", "rounds_2", _inject_bell_cultivate(2)),
            CorpusCase("hubbard_chain", "n4_l2", _hubbard_chain(4, 2)),
            CorpusCase("hubbard_chain", "n6_l2", _hubbard_chain(6, 2)),
            CorpusCase("qaoa_ring", "n6_p2", _qaoa_ring(6, 2)),
            CorpusCase("qaoa_ring", "n8_p2", _qaoa_ring(8, 2)),
            CorpusCase("double_excitation", "reps_1", _double_excitation_4q(1)),
            CorpusCase("double_excitation", "reps_2", _double_excitation_4q(2)),
        ]
    )
    return cases


def main() -> None:
    print(
        "| family | circuit | source T-like ops | after peephole | "
        "after ExperimentalMcrTCountPass + peephole | delta T | reduction |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")

    wins = 0
    total = 0
    for case in _corpus():
        total += 1
        peephole = _count_t_after(case.stim_text, use_mcr=False)
        mcr = _count_t_after(case.stim_text, use_mcr=True)
        delta = peephole - mcr
        if delta > 0:
            wins += 1
        reduction = 0.0 if peephole == 0 else 100.0 * delta / peephole
        print(
            f"| {case.family} | {case.name} | {_source_t_like_ops(case.stim_text)} | "
            f"{peephole} | {mcr} | {delta} | {reduction:.1f}% |"
        )

    print()
    print(f"Improved {wins} / {total} corpus entries.")


if __name__ == "__main__":
    main()
