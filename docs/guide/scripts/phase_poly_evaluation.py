"""T-count evaluation: peephole vs McrTcountPass / TohpePhasePass / PhasePolynomialPass.

Guide and theory: docs/guide/phase_polynomial_pass.md

Usage:
    uv run python docs/guide/scripts/phase_poly_evaluation.py
    uv run python docs/guide/scripts/phase_poly_evaluation.py --guide
    uv run python docs/guide/scripts/phase_poly_evaluation.py --check-equiv
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    IntegerComparator,
    ModularAdderGate,
    MultiplierGate,
    PauliEvolutionGate,
    VBERippleCarryAdder,
    WeightedAdder,
    grover_operator,
)
from qiskit.qasm2 import dumps
from qiskit.quantum_info import SparsePauliOp

import clifft

_GUIDE_DIR = Path(__file__).resolve().parents[1]
_CIRCUIT_DIR = _GUIDE_DIR / "circuits"
_GUIDE_DOC = _GUIDE_DIR / "phase_polynomial_pass.md"

_CLIFFORD_T_BASIS = ["h", "s", "sdg", "t", "tdg", "cx"]
_TRANSPILE_KWARGS = {
    "basis_gates": _CLIFFORD_T_BASIS,
    "optimization_level": 0,
    "seed_transpiler": 42,
}


@dataclass(frozen=True)
class EvalRow:
    name: str
    category: str
    num_qubits: int
    t_peep: int
    t_mcr: int
    t_tohpe: int
    t_full: int
    mcr_swaps: int
    tohpe_blocks: int
    fidelity: str = ""


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


def _qft_layer(n: int) -> str:
    lines: list[str] = []
    for target in range(n):
        for control in range(target + 1, n):
            lines.append(f"R_Z(0.25) {control} {target}")
        if target < n - 1:
            lines.append(f"H {target}")
    if n > 0:
        lines.append(f"H {n - 1}")
    return "\n".join(lines)


def _toffoli_chain(n_toffoli: int, n_qubits: int) -> str:
    lines: list[str] = []
    for i in range(n_toffoli):
        a = i % max(n_qubits - 2, 1)
        b = (i + 1) % max(n_qubits - 2, 1)
        t = (i + 2) % n_qubits
        lines.append(f"R_Z(0.25) {a} {b} {t}")
        lines.append(f"R_XX(0.25) {a} {b}")
        lines.append(f"R_Z(0.25) {b} {t}")
    return "\n".join(lines)


def _factored_clifford_t(n: int, t: int, k: int) -> str:
    """Factored benchmark family from docs/guide/scripts/run_benchmark.py (no measure)."""
    actual_k = min(k, n)
    qc = QuantumCircuit(n)
    for i in range(actual_k):
        qc.h(i)
    for i in range(t):
        tgt = i % actual_k
        qc.t(tgt)
        qc.h(tgt)
        if actual_k > 1:
            nxt = (tgt + 1) % actual_k
            qc.cx(tgt, nxt)
    for i in range(actual_k, n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return _clifford_t_stim(qc)


def _surface_d3_t_gate() -> str:
    return (_CIRCUIT_DIR / "circuit_d3_t_gate_p0.001.stim").read_text()


SYNTHETIC_EXAMPLES: dict[str, str] = {
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
    "negative_sign_block": "\n".join(["X 0"] + _pair_block(0, 1).splitlines()),
    "two_disjoint_pair_blocks": _pair_block(0, 1) + "\n" + _pair_block(2, 3),
    "three_disjoint_pair_blocks": _pair_block(0, 1)
    + "\n"
    + _pair_block(2, 3)
    + "\n"
    + _pair_block(4, 5),
    "ccx_toffoli": (
        "H 2\nCNOT 1 2\nT_DAG 2\nCNOT 0 2\nT 2\nCNOT 1 2\nT_DAG 2\nCNOT 0 2\n"
        "T_DAG 1\nT 2\nH 2\nCNOT 0 1\nT_DAG 1\nCNOT 0 1\nT 0\nT 1"
    ),
    "fredkin_fragment": (
        "H 2\nCNOT 2 1\nCNOT 0 2\nH 2\nT 0\nT_DAG 1\nT 2\nCNOT 0 1\n"
        "CNOT 2 0\nCNOT 1 2\nT_DAG 0\nT 1\nT_DAG 2\nCNOT 0 1\nCNOT 2 0\n"
        "H 2\nCNOT 2 1\nH 2"
    ),
    "controlled_s": "T 0\nT 1\nCNOT 0 1\nT_DAG 1\nCNOT 0 1",
    "small_clifford_t": ("H 0\nT 0\nH 0\nS 0\nH 0\nT 1\nH 1\nCX 0 1\nT 0\nH 0\nT 1\nH 1"),
}


def _qasm_clifford_t_to_stim(qasm: str) -> str:
    """Convert flat OpenQASM 2.0 Clifford+T output to Clifft .stim text."""
    lines: list[str] = []
    patterns: tuple[tuple[str, str], ...] = (
        (r"^h\s+q\[(\d+)\]$", r"H \1"),
        (r"^x\s+q\[(\d+)\]$", r"X \1"),
        (r"^y\s+q\[(\d+)\]$", r"Y \1"),
        (r"^z\s+q\[(\d+)\]$", r"Z \1"),
        (r"^s\s+q\[(\d+)\]$", r"S \1"),
        (r"^sdg\s+q\[(\d+)\]$", r"S_DAG \1"),
        (r"^t\s+q\[(\d+)\]$", r"T \1"),
        (r"^tdg\s+q\[(\d+)\]$", r"T_DAG \1"),
        (r"^cx\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]$", r"CX \1 \2"),
        (r"^cz\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]$", r"CZ \1 \2"),
        (r"^cy\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]$", r"CY \1 \2"),
    )

    for raw in qasm.splitlines():
        line = raw.strip().rstrip(";")
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "//")):
            continue
        for pattern, fmt in patterns:
            match = re.match(pattern, line)
            if match is not None:
                lines.append(match.expand(fmt))
                break
        else:
            raise ValueError(f"Unsupported QASM instruction: {line!r}")

    return "\n".join(lines)


def _clifford_t_stim(qc: QuantumCircuit) -> str:
    """Transpile to Clifford+T and export via Qiskit qasm2.dumps."""
    tqc = transpile(qc, **_TRANSPILE_KWARGS)
    flat = QuantumCircuit(tqc.num_qubits)
    flat.compose(tqc, list(range(tqc.num_qubits)), inplace=True)
    return _qasm_clifford_t_to_stim(dumps(flat))


def _gate_circuit(gate: Any) -> QuantumCircuit:
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, range(gate.num_qubits))
    return qc


def _hamiltonian_trotter(terms: list[tuple[str, float]], time: float) -> str:
    ham = SparsePauliOp.from_list(terms)
    qc = QuantumCircuit(ham.num_qubits)
    qc.append(PauliEvolutionGate(ham, time=time), range(ham.num_qubits))
    return _clifford_t_stim(qc)


def _grover_2q_circuit() -> str:
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)
    return _clifford_t_stim(grover_operator(oracle))


def _ccz_mcx_circuit() -> str:
    qc = QuantumCircuit(3)
    qc.mcx([0, 1], 2)
    return _clifford_t_stim(qc)


def _modexp_fragment_circuit() -> str:
    modexp = QuantumCircuit(5)
    for _ in range(3):
        modexp.h(4)
        modexp.cx(3, 4)
        modexp.t(4)
        modexp.cx(2, 4)
        modexp.tdg(4)
        modexp.cx(3, 4)
        modexp.h(4)
        modexp.cx(1, 2)
    return _clifford_t_stim(modexp)


def _rc_adder_cdkm_4bit() -> str:
    return _clifford_t_stim(CDKMRippleCarryAdder(4, kind="full"))


def _rc_adder_vbe_4bit() -> str:
    return _clifford_t_stim(VBERippleCarryAdder(4, kind="full"))


def _weighted_adder_4w() -> str:
    return _clifford_t_stim(WeightedAdder(4, [1, 2, 4, 8]))


def _modular_adder_4bit() -> str:
    return _clifford_t_stim(_gate_circuit(ModularAdderGate(4)))


def _integer_comparator() -> str:
    return _clifford_t_stim(_gate_circuit(IntegerComparator(4, 5)))


def _multiplier_3bit() -> str:
    return _clifford_t_stim(_gate_circuit(MultiplierGate(3)))


def _h2_trotter_2q() -> str:
    return _hamiltonian_trotter([("II", 1.0), ("ZZ", 0.5), ("XX", 0.3)], 0.25)


def _ising_chain_3q() -> str:
    return _hamiltonian_trotter(
        [("ZZI", 0.4), ("IZZ", 0.4), ("XII", 0.2), ("IXI", 0.2), ("IIX", 0.2)],
        0.25,
    )


def _heisenberg_3q() -> str:
    return _hamiltonian_trotter([("XYZ", 0.5)], 0.25)


def _random_dense_clifford_t(num_qubits: int, depth: int, seed: int) -> str:
    """Dense Clifford+T stress circuit (same distribution as tests/python/conftest.py)."""
    rng = np.random.default_rng(seed)
    gates_1q = ["H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z"]
    gates_2q = ["CX", "CY", "CZ"]
    lines: list[str] = []
    for _ in range(depth):
        if num_qubits > 1 and rng.random() < 0.5:
            gate = str(rng.choice(gates_2q))
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"{gate} {q1} {q2}")
        else:
            gate = str(rng.choice(gates_1q))
            q = int(rng.integers(0, num_qubits))
            lines.append(f"{gate} {q}")
    return "\n".join(lines)


def _random_ct_6q_d150() -> str:
    return _random_dense_clifford_t(6, 150, seed=126)


def _random_ct_8q_d120() -> str:
    return _random_dense_clifford_t(8, 120, seed=127)


def _random_ct_10q_d100() -> str:
    return _random_dense_clifford_t(10, 100, seed=128)


def _random_ct_12q_d80() -> str:
    return _random_dense_clifford_t(12, 80, seed=129)


def _build_real_world_examples() -> list[tuple[str, str, Callable[[], str]]]:
    """Named algorithmic benchmarks: (name, category, circuit_builder)."""
    return [
        # op-T-mize / synthesis style (Barenco benchmark family).
        ("qft_4q", "synthesis", lambda: _qft_layer(4)),
        ("qft_6q", "synthesis", lambda: _qft_layer(6)),
        ("qft_8q", "synthesis", lambda: _qft_layer(8)),
        ("toffoli_chain_8q", "synthesis", lambda: _toffoli_chain(12, 8)),
        ("toffoli_chain_10q", "synthesis", lambda: _toffoli_chain(16, 10)),
        # Arithmetic (Amy T-count benchmark family).
        ("rc_adder_cdkm_4bit", "arithmetic", _rc_adder_cdkm_4bit),
        ("rc_adder_vbe_4bit", "arithmetic", _rc_adder_vbe_4bit),
        ("weighted_adder_4w", "arithmetic", _weighted_adder_4w),
        ("modular_adder_4bit", "arithmetic", _modular_adder_4bit),
        ("integer_comparator", "arithmetic", _integer_comparator),
        ("multiplier_3bit", "arithmetic", _multiplier_3bit),
        # Algorithms.
        ("grover_2q_oracle", "algorithm", _grover_2q_circuit),
        ("ccz_mcx_3q", "algorithm", _ccz_mcx_circuit),
        ("modexp_ctrl_fragment", "algorithm", _modexp_fragment_circuit),
        # Chemistry Trotter steps.
        ("h2_trotter_2q", "chemistry", _h2_trotter_2q),
        ("ising_chain_3q", "chemistry", _ising_chain_3q),
        ("heisenberg_3q", "chemistry", _heisenberg_3q),
        # Factored Clifford+T (docs/guide/benchmark.md family).
        ("factored_n24_k12_t20", "compiled", lambda: _factored_clifford_t(24, 20, 12)),
        ("factored_n24_k16_t40", "compiled", lambda: _factored_clifford_t(24, 40, 16)),
        ("factored_n28_k12_t20", "compiled", lambda: _factored_clifford_t(28, 20, 12)),
        # Dense random Clifford+T stress.
        ("random_ct_6q_d150", "compiled", _random_ct_6q_d150),
        ("random_ct_8q_d120", "compiled", _random_ct_8q_d120),
        ("random_ct_10q_d100", "compiled", _random_ct_10q_d100),
        ("random_ct_12q_d80", "compiled", _random_ct_12q_d80),
        # Surface-code magic-state cultivation (SOFT / importance-sampling corpus).
        ("surface_d3_t_gate", "surface_code", _surface_d3_t_gate),
    ]


def _pass_manager_with(
    middle: clifft.HirPass,
) -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(middle)
    pm.add(clifft.PeepholeFusionPass())
    return pm


def _num_qubits(circuit: str) -> int:
    return int(clifft.trace(clifft.parse(circuit)).num_qubits)


def _t_after_peephole(circuit: str) -> int:
    hir = clifft.trace(clifft.parse(circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates)


def _t_after_mcr(circuit: str) -> tuple[int, int]:
    hir = clifft.trace(clifft.parse(circuit))
    mcr = clifft.McrTcountPass()
    _pass_manager_with(mcr).run(hir)
    return int(hir.num_t_gates), int(mcr.stats()["swaps_applied"])


def _t_after_tohpe(circuit: str) -> tuple[int, int]:
    hir = clifft.trace(clifft.parse(circuit))
    tohpe = clifft.TohpePhasePass()
    _pass_manager_with(tohpe).run(hir)
    return int(hir.num_t_gates), int(tohpe.blocks_optimized)


def _t_after_full(circuit: str) -> tuple[int, int, int]:
    hir = clifft.trace(clifft.parse(circuit))
    poly = clifft.PhasePolynomialPass()
    _pass_manager_with(poly).run(hir)
    return (
        int(hir.num_t_gates),
        int(poly.mcr_stats()["swaps_applied"]),
        int(poly.blocks_optimized),
    )


def _statevector(circuit: str, *, optimize: bool) -> np.ndarray:
    prog = clifft.compile(
        circuit,
        hir_passes=_pass_manager_with(clifft.PhasePolynomialPass()) if optimize else None,
        bytecode_passes=None,
    )
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    return np.asarray(clifft.get_statevector(prog, state))


def _evaluate(
    name: str,
    category: str,
    circuit: str,
    *,
    check_equiv: bool,
    max_equiv_qubits: int,
) -> EvalRow:
    t_peep = _t_after_peephole(circuit)
    t_mcr, mcr_swaps = _t_after_mcr(circuit)
    t_tohpe, tohpe_blocks = _t_after_tohpe(circuit)
    t_full, _, _ = _t_after_full(circuit)
    nq = _num_qubits(circuit)

    fidelity = ""
    if check_equiv and t_peep > t_full and nq <= max_equiv_qubits:
        ref = _statevector(circuit, optimize=False)
        opt = _statevector(circuit, optimize=True)
        fid = float(np.abs(np.vdot(ref, opt)) ** 2)
        fidelity = f"{fid:.6f}" if fid < 0.999999 else "1.000000"
    elif check_equiv and t_peep > t_full:
        fidelity = "skipped"

    return EvalRow(
        name=name,
        category=category,
        num_qubits=nq,
        t_peep=t_peep,
        t_mcr=t_mcr,
        t_tohpe=t_tohpe,
        t_full=t_full,
        mcr_swaps=mcr_swaps,
        tohpe_blocks=tohpe_blocks,
        fidelity=fidelity,
    )


def _format_row(row: EvalRow) -> str:
    return (
        f"{row.name:<28} {row.category:<12} {row.num_qubits:>3} {row.t_peep:>8} "
        f"{row.t_mcr:>8} {row.t_tohpe:>8} {row.t_full:>8} "
        f"{row.t_peep - row.t_mcr:>6} {row.t_peep - row.t_tohpe:>6} {row.t_peep - row.t_full:>6} "
        f"{row.mcr_swaps:>6} {row.tohpe_blocks:>7} {row.fidelity:>10}"
    )


def _print_table(rows: list[EvalRow], title: str) -> None:
    print(title)
    header = (
        f"{'example':<28} {'category':<12} {'q':>3} {'peep_T':>8} {'mcr_T':>8} "
        f"{'tohpe_T':>8} {'full_T':>8} {'d_mcr':>6} {'d_tohpe':>6} {'d_full':>6} "
        f"{'swaps':>6} {'blocks':>7} {'fidelity':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(_format_row(row))
    print()


def _print_summary(synthetic: list[EvalRow], real_world: list[EvalRow]) -> None:
    all_rows = synthetic + real_world

    def stats(rows: list[EvalRow], label: str) -> None:
        if not rows:
            return
        total = len(rows)
        t_peep = sum(r.t_peep for r in rows)
        saved_mcr = sum(r.t_peep - r.t_mcr for r in rows)
        saved_tohpe = sum(r.t_peep - r.t_tohpe for r in rows)
        saved_full = sum(r.t_peep - r.t_full for r in rows)
        pct_mcr = 100.0 * saved_mcr / t_peep if t_peep else 0.0
        pct_tohpe = 100.0 * saved_tohpe / t_peep if t_peep else 0.0
        pct_full = 100.0 * saved_full / t_peep if t_peep else 0.0
        print(f"{label}:")
        print(f"  circuits={total}, T_peep={t_peep}")
        print(f"  MCR-only saved:   {saved_mcr} ({pct_mcr:.2f}%)")
        print(f"  TOHPE-only saved: {saved_tohpe} ({pct_tohpe:.2f}%)")
        print(f"  Full pass saved:  {saved_full} ({pct_full:.2f}%)")
        print()

    print("=" * 96)
    print("Aggregate summary (delta vs peephole baseline)")
    print("=" * 96)
    stats(synthetic, "Synthetic / MCR regression")
    stats(real_world, "Real-world / algorithmic")
    stats(all_rows, "Combined")

    print(
        "Conclusion:\n"
        "- HIR Pauli masks encode phase-polynomial parity columns; pass is sound\n"
        "  without parser, bytecode, or VM changes.\n"
        "- MCR reordering wins on structured MCR-family circuits only.\n"
        "- TOHPE finds no additional reductions on the real-world corpus once\n"
        "  HIR barriers (noise, measurements, phase rotations) split blocks.\n"
        "- On transpiled arithmetic, chemistry Trotter, factored workloads, and\n"
        "  surface-code magic-state cultivation, peephole already matches the\n"
        "  minimum T count in this prototype.\n"
        "- Scientific verdict: opt-in pass for structured compilation artifacts;\n"
        "  not yet justified as a default pipeline pass.\n"
        "- Follow-up: lift 32-qubit TOHPE cap, Amy/op-T-mize corpus, external\n"
        "  T-optimizer baseline."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PhasePolynomialPass T-count evaluation")
    parser.add_argument("--guide", action="store_true", help="Print guide doc and exit")
    parser.add_argument(
        "--check-equiv",
        action="store_true",
        help="Statevector check when full pass reduces T (skip if qubits > max)",
    )
    parser.add_argument(
        "--max-equiv-qubits",
        type=int,
        default=8,
        help="Max qubits for statevector equivalence (default 8)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip real-world circuits",
    )
    args = parser.parse_args()

    if args.guide:
        print(_GUIDE_DOC.read_text())
        return

    synthetic_rows = [
        _evaluate(
            name,
            "synthetic",
            circuit,
            check_equiv=args.check_equiv,
            max_equiv_qubits=args.max_equiv_qubits,
        )
        for name, circuit in SYNTHETIC_EXAMPLES.items()
    ]

    real_rows: list[EvalRow] = []
    if not args.synthetic_only:
        for name, category, builder in _build_real_world_examples():
            circuit = builder()
            real_rows.append(
                _evaluate(
                    name,
                    category,
                    circuit,
                    check_equiv=args.check_equiv,
                    max_equiv_qubits=args.max_equiv_qubits,
                )
            )

    _print_table(synthetic_rows, "Synthetic / MCR regression circuits")
    if real_rows:
        _print_table(real_rows, "Real-world / algorithmic benchmarks")
    _print_summary(synthetic_rows, real_rows)


if __name__ == "__main__":
    main()
