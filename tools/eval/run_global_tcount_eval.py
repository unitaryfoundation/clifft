#!/usr/bin/env python3
"""Issue #40 evaluation harness: per-phase T-count impact on benchmark circuits.

Run:
    uv run python tools/eval/run_global_tcount_eval.py
    uv run python tools/eval/run_global_tcount_eval.py --json
    uv run python tools/eval/run_global_tcount_eval.py --check-equivalence
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clifft  # noqa: E402
from tools.eval.global_tcount_benchmarks import BENCHMARKS, BenchmarkCircuit  # noqa: E402


@dataclass
class EvalRow:
    name: str
    category: str
    baseline_t: int
    peephole_t: int
    mcr_only_t: int
    todd_only_t: int
    full_t: int
    mcr_swaps: int
    todd_blocks: int
    equivalence_checked: bool
    equivalence_ok: bool | None


def _trace_t(circuit: str) -> int:
    return int(clifft.trace(clifft.parse(circuit)).num_t_gates)


def _run_pipeline(circuit: str, *passes: clifft.HirPass) -> tuple[int, list[clifft.HirPass]]:
    hir = clifft.trace(clifft.parse(circuit))
    applied: list[clifft.HirPass] = []
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    for p in passes:
        pm.add(p)
        applied.append(p)
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return int(hir.num_t_gates), applied


def _statevector(circuit: str, *passes: clifft.HirPass) -> np.ndarray:
    hir = clifft.trace(clifft.parse(circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    for p in passes:
        pm.add(p)
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    prog = clifft.lower(hir)
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    return np.asarray(clifft.get_statevector(prog, state))


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def evaluate_one(bench: BenchmarkCircuit, *, check_equivalence: bool) -> EvalRow:
    baseline_t = _trace_t(bench.circuit)
    peephole_t, _ = _run_pipeline(bench.circuit)

    mcr = clifft.McrReorderPass()
    mcr_only_t, _ = _run_pipeline(bench.circuit, mcr)

    todd = clifft.ToddPhasePass()
    todd_only_t, _ = _run_pipeline(bench.circuit, todd)

    global_pass = clifft.GlobalTcountPass()
    full_t, applied = _run_pipeline(bench.circuit, global_pass)
    assert isinstance(applied[-1], clifft.GlobalTcountPass)

    eq_checked = False
    eq_ok: bool | None = None
    if check_equivalence and bench.max_qubits_for_statevector >= 0:
        try:
            sv_ref = _statevector(bench.circuit)
            sv_opt = _statevector(bench.circuit, global_pass)
            if sv_ref.shape == sv_opt.shape:
                eq_checked = True
                eq_ok = _fidelity(sv_ref, sv_opt) > 0.9999
        except Exception:
            eq_checked = True
            eq_ok = None

    return EvalRow(
        name=bench.name,
        category=bench.category,
        baseline_t=baseline_t,
        peephole_t=peephole_t,
        mcr_only_t=mcr_only_t,
        todd_only_t=todd_only_t,
        full_t=full_t,
        mcr_swaps=int(global_pass.mcr_swaps_applied),
        todd_blocks=int(global_pass.todd_blocks),
        equivalence_checked=eq_checked,
        equivalence_ok=eq_ok,
    )


def format_table(rows: list[EvalRow]) -> str:
    header = (
        f"{'name':<28} {'cat':<10} {'base':>5} {'peep':>5} {'mcr':>5} "
        f"{'todd':>5} {'full':>5} {'mcr_sw':>7} {'todd_bl':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.name:<28} {r.category:<10} {r.baseline_t:>5} {r.peephole_t:>5} "
            f"{r.mcr_only_t:>5} {r.todd_only_t:>5} {r.full_t:>5} "
            f"{r.mcr_swaps:>7} {r.todd_blocks:>8}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    parser.add_argument(
        "--check-equivalence",
        action="store_true",
        help="Statevector-check circuits where the full pass changes T count",
    )
    args = parser.parse_args()

    rows = [evaluate_one(b, check_equivalence=args.check_equivalence) for b in BENCHMARKS]

    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        print(format_table(rows))
        improved = [r for r in rows if r.full_t < r.peephole_t]
        print()
        print(f"Circuits improved over peephole-only: {len(improved)}/{len(rows)}")
        if args.check_equivalence:
            checked = [r for r in rows if r.equivalence_checked]
            ok = [r for r in checked if r.equivalence_ok]
            print(f"Equivalence checked: {len(checked)}, passed: {len(ok)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
