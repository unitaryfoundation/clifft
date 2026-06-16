#!/usr/bin/env python3
"""T-count evaluation harness for ExactPhasePolynomialTCountPass.

This script is intentionally an evaluator, not an optimizer. It keeps the pass
boundary fixed by running:

    trace -> PeepholeFusionPass -> ExactPhasePolynomialTCountPass

or, with --collect-t-blocks:

    trace -> PeepholeFusionPass -> TGateBlockCollectionPass
        -> ExactPhasePolynomialTCountPass

and reporting T-count deltas plus pass statistics.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CircuitCase:
    name: str
    text: str
    source: str


@dataclass(frozen=True)
class Result:
    name: str
    source: str
    traced_t: int
    after_peephole_t: int
    after_collect_t: int | None
    after_exact_t: int
    blocks_collected: int
    t_gates_moved: int
    adjacent_swaps: int
    blocks_considered: int
    blocks_optimized: int
    t_removed: int


def ccz_complete(n: int) -> str:
    lines = [f"CCZ {a} {b} {c}" for a, b, c in combinations(range(n), 3)]
    return "\n".join(lines) + "\n"


def ccx_ladder(n: int) -> str:
    lines = [f"CCX {i} {i + 1} {i + 2}" for i in range(n - 2)]
    return "\n".join(lines) + "\n"


def builtin_cases(repo_root: Path) -> list[CircuitCase]:
    cases: list[CircuitCase] = []

    for n in range(3, 8):
        cases.append(CircuitCase(f"ccz_complete_{n}", ccz_complete(n), "generated"))
    for n in range(3, 8):
        cases.append(CircuitCase(f"ccx_ladder_{n}", ccx_ladder(n), "generated"))

    for rel in [
        "tests/fixtures/cultivation_d5.stim",
        "tests/fixtures/qv10.stim",
        "tests/fixtures/target_qec.stim",
        "docs/guide/circuits/circuit_d3_s_gate_p0.001.stim",
        "docs/guide/circuits/circuit_d3_t_gate_p0.001.stim",
    ]:
        path = repo_root / rel
        if path.exists():
            cases.append(CircuitCase(rel, path.read_text(), rel))
    return cases


def split_qc_tokens(line: str) -> list[str]:
    return line.replace(",", " ").split()


def parse_qc_text(text: str, name: str) -> str:
    """Import a conservative subset of the common .qc benchmark format.

    Supported gates are Clifford+T primitives and one-, two-, and three-argument
    `tof`/`toffoli` instructions. Wider multi-control Toffoli instructions are
    rejected rather than decomposed with implicit ancillas.
    """

    wire_order: list[str] = []
    wire_to_qubit: dict[str, int] = {}
    stim_lines: list[str] = []
    in_body = False

    def intern_wire(wire: str) -> int:
        if wire not in wire_to_qubit:
            wire_to_qubit[wire] = len(wire_order)
            wire_order.append(wire)
        return wire_to_qubit[wire]

    def add_declared_wires(tokens: Iterable[str]) -> None:
        for wire in tokens:
            if wire:
                intern_wire(wire)

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if line.upper() == "BEGIN":
            in_body = True
            continue
        if line.upper() == "END":
            in_body = False
            continue

        tokens = split_qc_tokens(line)
        if not tokens:
            continue

        head = tokens[0]
        if head.startswith("."):
            if head in {".v", ".i", ".o"}:
                add_declared_wires(tokens[1:])
            continue
        if not in_body:
            continue

        gate = head.lower()
        args = tokens[1:]

        def q(index: int) -> int:
            return intern_wire(args[index])

        try:
            if gate in {"h", "hadamard"} and len(args) == 1:
                stim_lines.append(f"H {q(0)}")
            elif gate in {"x", "not"} and len(args) == 1:
                stim_lines.append(f"X {q(0)}")
            elif gate == "y" and len(args) == 1:
                stim_lines.append(f"Y {q(0)}")
            elif gate == "z" and len(args) == 1:
                stim_lines.append(f"Z {q(0)}")
            elif gate in {"s", "p"} and len(args) == 1:
                stim_lines.append(f"S {q(0)}")
            elif gate in {"s*", "sdg", "s_dag", "p*", "pdg", "p_dag"} and len(args) == 1:
                stim_lines.append(f"S_DAG {q(0)}")
            elif gate == "t" and len(args) == 1:
                stim_lines.append(f"T {q(0)}")
            elif gate in {"t*", "tdg", "t_dag"} and len(args) == 1:
                stim_lines.append(f"T_DAG {q(0)}")
            elif gate in {"cx", "cnot"} and len(args) == 2:
                stim_lines.append(f"CX {q(0)} {q(1)}")
            elif gate == "cz" and len(args) == 2:
                stim_lines.append(f"CZ {q(0)} {q(1)}")
            elif gate in {"tof", "toffoli"}:
                if len(args) == 1:
                    stim_lines.append(f"X {q(0)}")
                elif len(args) == 2:
                    stim_lines.append(f"CX {q(0)} {q(1)}")
                elif len(args) == 3:
                    stim_lines.append(f"CCX {q(0)} {q(1)} {q(2)}")
                else:
                    raise ValueError(
                        f"{name}:{lineno}: unsupported {len(args) - 1}-control Toffoli"
                    )
            elif gate == "ccz" and len(args) == 3:
                stim_lines.append(f"CCZ {q(0)} {q(1)} {q(2)}")
            else:
                raise ValueError(f"{name}:{lineno}: unsupported .qc gate line: {raw_line}")
        except IndexError as exc:
            raise ValueError(f"{name}:{lineno}: malformed .qc gate line: {raw_line}") from exc

    return "\n".join(stim_lines) + ("\n" if stim_lines else "")


def qc_cases(paths: Iterable[Path], skip_unsupported: bool = False) -> list[CircuitCase]:
    cases: list[CircuitCase] = []
    for path in paths:
        try:
            stim_text = parse_qc_text(path.read_text(), str(path))
        except ValueError as exc:
            if not skip_unsupported:
                raise
            print(f"Skipping unsupported .qc file {path}: {exc}", file=sys.stderr)
            continue
        cases.append(CircuitCase(path.name, stim_text, str(path)))
    return cases


QREG_RE = re.compile(r"qreg\s+([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$")
QASM_QUBIT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$")


def parse_qasm_text(text: str, name: str) -> str:
    """Import a conservative Clifford+T subset of OpenQASM 2.0.

    The importer is for T-count benchmarking only. It ignores barriers and
    measurements, supports common Clifford+T gate names, and rejects custom or
    parameterized gates instead of attempting synthesis.
    """

    register_offsets: dict[str, int] = {}
    register_sizes: dict[str, int] = {}
    next_qubit = 0
    stim_lines: list[str] = []

    def strip_comment(raw_line: str) -> str:
        return raw_line.split("//", 1)[0].split("#", 1)[0].strip()

    def q(token: str, lineno: int) -> int:
        match = QASM_QUBIT_RE.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"{name}:{lineno}: unsupported QASM operand: {token}")
        reg = match.group(1)
        index = int(match.group(2))
        if reg not in register_offsets:
            raise ValueError(f"{name}:{lineno}: unknown qreg: {reg}")
        if index >= register_sizes[reg]:
            raise ValueError(f"{name}:{lineno}: qreg index out of range: {token}")
        return register_offsets[reg] + index

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = strip_comment(raw_line)
        if not line:
            continue
        if line.endswith(";"):
            line = line[:-1].strip()

        lowered = line.lower()
        if lowered.startswith("openqasm ") or lowered.startswith("include "):
            continue
        if lowered.startswith("creg "):
            continue
        if lowered.startswith("gate ") or lowered.startswith("opaque "):
            raise ValueError(f"{name}:{lineno}: custom QASM gates are not imported")
        if lowered.startswith("barrier "):
            continue
        if lowered.startswith("measure "):
            raise ValueError(f"{name}:{lineno}: measurements are not imported")
        if lowered.startswith("reset ") or lowered.startswith("if("):
            raise ValueError(f"{name}:{lineno}: unsupported QASM statement: {raw_line}")

        qreg_match = QREG_RE.fullmatch(line)
        if qreg_match is not None:
            reg = qreg_match.group(1)
            size = int(qreg_match.group(2))
            if reg in register_offsets:
                raise ValueError(f"{name}:{lineno}: duplicate qreg: {reg}")
            register_offsets[reg] = next_qubit
            register_sizes[reg] = size
            next_qubit += size
            continue

        if " " in line:
            gate, arg_text = line.split(None, 1)
        else:
            gate, arg_text = line, ""
        gate = gate.lower()
        args = [part.strip() for part in arg_text.split(",") if part.strip()]

        if gate in {"id", "iden"} and len(args) == 1:
            continue
        if gate in {"h", "x", "y", "z", "s", "sdg", "t", "tdg"} and len(args) == 1:
            mapped = {
                "h": "H",
                "x": "X",
                "y": "Y",
                "z": "Z",
                "s": "S",
                "sdg": "S_DAG",
                "t": "T",
                "tdg": "T_DAG",
            }[gate]
            stim_lines.append(f"{mapped} {q(args[0], lineno)}")
        elif gate in {"cx", "cnot"} and len(args) == 2:
            stim_lines.append(f"CX {q(args[0], lineno)} {q(args[1], lineno)}")
        elif gate == "cz" and len(args) == 2:
            stim_lines.append(f"CZ {q(args[0], lineno)} {q(args[1], lineno)}")
        elif gate == "ccx" and len(args) == 3:
            stim_lines.append(f"CCX {q(args[0], lineno)} {q(args[1], lineno)} {q(args[2], lineno)}")
        elif gate == "ccz" and len(args) == 3:
            stim_lines.append(f"CCZ {q(args[0], lineno)} {q(args[1], lineno)} {q(args[2], lineno)}")
        elif gate == "swap" and len(args) == 2:
            a = q(args[0], lineno)
            b = q(args[1], lineno)
            stim_lines.extend([f"CX {a} {b}", f"CX {b} {a}", f"CX {a} {b}"])
        else:
            raise ValueError(f"{name}:{lineno}: unsupported QASM gate line: {raw_line}")

    return "\n".join(stim_lines) + ("\n" if stim_lines else "")


def qasm_cases(paths: Iterable[Path], skip_unsupported: bool = False) -> list[CircuitCase]:
    cases: list[CircuitCase] = []
    for path in paths:
        try:
            stim_text = parse_qasm_text(path.read_text(), str(path))
        except ValueError as exc:
            if not skip_unsupported:
                raise
            print(f"Skipping unsupported QASM file {path}: {exc}", file=sys.stderr)
            continue
        cases.append(CircuitCase(path.name, stim_text, str(path)))
    return cases


def evaluate(
    case: CircuitCase, max_rank: int, collect_t_blocks: bool, collect_window: int
) -> Result:
    import clifft

    hir = clifft.trace(clifft.parse(case.text))
    traced_t = int(hir.num_t_gates)

    peephole = clifft.PeepholeFusionPass()
    pm = clifft.HirPassManager()
    pm.add(peephole)
    pm.run(hir)
    after_peephole = int(hir.num_t_gates)

    after_collect: int | None = None
    blocks_collected = 0
    t_gates_moved = 0
    adjacent_swaps = 0
    if collect_t_blocks:
        collect = clifft.TGateBlockCollectionPass(collect_window)
        pm = clifft.HirPassManager()
        pm.add(collect)
        pm.run(hir)
        after_collect = int(hir.num_t_gates)
        blocks_collected = int(collect.blocks_collected)
        t_gates_moved = int(collect.t_gates_moved)
        adjacent_swaps = int(collect.adjacent_swaps)

    exact = clifft.ExactPhasePolynomialTCountPass(max_rank)
    pm = clifft.HirPassManager()
    pm.add(exact)
    pm.run(hir)

    return Result(
        name=case.name,
        source=case.source,
        traced_t=traced_t,
        after_peephole_t=after_peephole,
        after_collect_t=after_collect,
        after_exact_t=int(hir.num_t_gates),
        blocks_collected=blocks_collected,
        t_gates_moved=t_gates_moved,
        adjacent_swaps=adjacent_swaps,
        blocks_considered=int(exact.blocks_considered),
        blocks_optimized=int(exact.blocks_optimized),
        t_removed=int(exact.t_removed),
    )


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def print_markdown(results: list[Result], include_collection: bool) -> None:
    if include_collection:
        print(
            "| Circuit | Source | Traced T | After peephole | After collect | After exact | "
            "Delta vs peephole | Collected blocks | T gates moved | Adjacent swaps | "
            "Exact blocks considered | Exact blocks optimized |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for result in results:
            delta = result.after_peephole_t - result.after_exact_t
            after_collect = result.after_collect_t
            assert after_collect is not None
            print(
                f"| {markdown_escape(result.name)} | {markdown_escape(result.source)} | "
                f"{result.traced_t} | {result.after_peephole_t} | {after_collect} | "
                f"{result.after_exact_t} | {delta} | {result.blocks_collected} | "
                f"{result.t_gates_moved} | {result.adjacent_swaps} | "
                f"{result.blocks_considered} | {result.blocks_optimized} |"
            )
        return

    print(
        "| Circuit | Source | Traced T | After peephole | After exact | "
        "Delta vs peephole | Exact blocks considered | Exact blocks optimized |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        delta = result.after_peephole_t - result.after_exact_t
        print(
            f"| {markdown_escape(result.name)} | {markdown_escape(result.source)} | "
            f"{result.traced_t} | {result.after_peephole_t} | {result.after_exact_t} | "
            f"{delta} | {result.blocks_considered} | {result.blocks_optimized} |"
        )


def gate_count(case: CircuitCase) -> int:
    return len([line for line in case.text.splitlines() if line.strip()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument(
        "--collect-t-blocks",
        action="store_true",
        help="Run the opt-in T-gate block collection pass before the exact decoder",
    )
    parser.add_argument(
        "--collect-window",
        type=int,
        default=64,
        help="Maximum consecutive non-T ops scanned while finding each next T candidate",
    )
    parser.add_argument("--no-builtins", action="store_true", help="Skip generated and repo cases")
    parser.add_argument("--qc", type=Path, action="append", default=[], help="Import a .qc file")
    parser.add_argument(
        "--qc-dir", type=Path, action="append", default=[], help="Import all .qc files"
    )
    parser.add_argument("--qasm", type=Path, action="append", default=[], help="Import a QASM file")
    parser.add_argument(
        "--qasm-dir", type=Path, action="append", default=[], help="Import all QASM files"
    )
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Skip unsupported imported .qc/QASM files instead of failing",
    )
    parser.add_argument(
        "--max-imported-gates",
        type=int,
        default=0,
        help="Skip imported cases with more source gate lines than this; 0 disables the cap",
    )
    args = parser.parse_args()

    cases: list[CircuitCase] = []
    if not args.no_builtins:
        cases.extend(builtin_cases(args.repo_root))

    qc_paths: list[Path] = list(args.qc)
    for directory in args.qc_dir:
        qc_paths.extend(sorted(directory.glob("*.qc")))
    cases.extend(qc_cases(qc_paths, args.skip_unsupported))

    qasm_paths: list[Path] = list(args.qasm)
    for directory in args.qasm_dir:
        qasm_paths.extend(sorted(directory.glob("*.qasm")))
    cases.extend(qasm_cases(qasm_paths, args.skip_unsupported))

    if args.max_imported_gates > 0:
        filtered_cases: list[CircuitCase] = []
        for case in cases:
            gates = gate_count(case)
            if gates > args.max_imported_gates:
                print(
                    f"Skipping {case.name}: {gates} imported gates exceeds "
                    f"--max-imported-gates={args.max_imported_gates}",
                    file=sys.stderr,
                )
                continue
            filtered_cases.append(case)
        cases = filtered_cases

    results = [
        evaluate(case, args.max_rank, args.collect_t_blocks, args.collect_window) for case in cases
    ]
    print_markdown(results, args.collect_t_blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
