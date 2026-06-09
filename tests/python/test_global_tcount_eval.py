"""Regression tests for the issue #40 evaluation harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_eval_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "eval" / "run_global_tcount_eval.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "kicked_xy_block" in result.stdout


@pytest.mark.parametrize("name", ["kicked_xy_block", "three_disjoint_pair_blocks"])
def test_full_pass_beats_peephole_on_mcr_circuits(name: str) -> None:
    from tools.eval.global_tcount_benchmarks import BENCHMARKS
    from tools.eval.run_global_tcount_eval import evaluate_one

    bench = next(b for b in BENCHMARKS if b.name == name)
    row = evaluate_one(bench, check_equivalence=False)
    assert row.full_t <= row.peephole_t
