"""Semantic equivalence tests for the experimental global T-count passes."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import assert_statevectors_equal, random_clifford_t_circuit

import clifft
from tools.eval.global_tcount_benchmarks import BENCHMARKS


def _compile_with_passes(circuit: str, *extra: clifft.HirPass) -> clifft.Program:
    hir = clifft.trace(clifft.parse(circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    for p in extra:
        pm.add(p)
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    return clifft.lower(hir)


def _statevector(prog: clifft.Program) -> np.ndarray:
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    return np.asarray(clifft.get_statevector(prog, state))


@pytest.mark.parametrize("bench", BENCHMARKS[:5], ids=lambda b: b.name)
def test_global_pass_preserves_statevector(bench) -> None:
    ref = _statevector(_compile_with_passes(bench.circuit))
    opt = _statevector(_compile_with_passes(bench.circuit, clifft.GlobalTcountPass()))
    assert_statevectors_equal(opt, ref, msg=bench.name)


@pytest.mark.parametrize("bench", BENCHMARKS[:5], ids=lambda b: b.name)
def test_mcr_only_preserves_statevector(bench) -> None:
    ref = _statevector(_compile_with_passes(bench.circuit))
    opt = _statevector(_compile_with_passes(bench.circuit, clifft.McrReorderPass()))
    assert_statevectors_equal(opt, ref, msg=f"mcr:{bench.name}")


@pytest.mark.parametrize("seed", range(5))
def test_random_clifford_t_global_pass(seed: int) -> None:
    circuit = random_clifford_t_circuit(4, depth=30, seed=seed)
    ref = _statevector(_compile_with_passes(circuit))
    opt = _statevector(_compile_with_passes(circuit, clifft.GlobalTcountPass()))
    assert_statevectors_equal(opt, ref, msg=f"random seed={seed}")


def test_global_pass_reduces_t_on_kicked_xy_block() -> None:
    bench = next(b for b in BENCHMARKS if b.name == "kicked_xy_block")
    hir = clifft.trace(clifft.parse(bench.circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    before = int(hir.num_t_gates)

    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(clifft.GlobalTcountPass())
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir)
    after = int(hir.num_t_gates)

    assert after < before


def test_global_pass_not_in_default_pipeline() -> None:
    bench = next(b for b in BENCHMARKS if b.name == "kicked_xy_block")
    hir_default = clifft.trace(clifft.parse(bench.circuit))
    clifft.default_hir_pass_manager().run(hir_default)
    t_default = int(hir_default.num_t_gates)

    hir_explicit = clifft.trace(clifft.parse(bench.circuit))
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(clifft.GlobalTcountPass())
    pm.add(clifft.PeepholeFusionPass())
    pm.run(hir_explicit)
    t_explicit = int(hir_explicit.num_t_gates)

    assert t_explicit < t_default
