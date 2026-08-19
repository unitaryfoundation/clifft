"""Tests for compile() with configurable HIR optimization passes."""

import pytest
from conftest import assert_statevectors_equiv, random_dense_clifford_t_circuit

import clifft


def test_compile_default_matches_explicit_pipeline() -> None:
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"
    compiled = clifft.compile(text)

    hir = clifft.trace(clifft.parse(text))
    clifft.default_hir_pass_manager().run(hir)
    manual = clifft.lower(hir)

    assert compiled.num_actions == manual.num_actions
    assert compiled.peak_active_width == manual.peak_active_width


def test_compile_explicit_none_skips_optimization() -> None:
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"
    compiled = clifft.compile(text, hir_passes=None)
    manual = clifft.lower(clifft.trace(clifft.parse(text)))

    assert compiled.num_actions == manual.num_actions
    assert compiled.peak_active_width == manual.peak_active_width


def test_hir_passes_reduce_t_cancellation() -> None:
    text = "H 0\nT 0\nT_DAG 0\nM 0"
    unoptimized = clifft.compile(text, hir_passes=None)
    optimized = clifft.compile(text)
    assert optimized.peak_active_width <= unoptimized.peak_active_width


def test_compile_postselection_with_passes() -> None:
    program = clifft.compile(
        "H 0\nM 0\nDETECTOR rec[-1]",
        postselection_mask=[1],
        hir_passes=clifft.default_hir_pass_manager(),
    )
    assert program.num_detectors == 1
    assert program.has_postselection


@pytest.mark.parametrize(
    "num_qubits,depth,seed",
    [(3, 50, 5000), (4, 80, 5001), (5, 100, 5002), (6, 100, 5004)],
)
def test_statevector_equiv_with_passes(num_qubits: int, depth: int, seed: int) -> None:
    text = random_dense_clifford_t_circuit(num_qubits, depth, seed)
    baseline = clifft.get_statevector(clifft.compile(text, hir_passes=None))
    optimized = clifft.get_statevector(clifft.compile(text))
    assert_statevectors_equiv(optimized, baseline)


def test_custom_pass_manager_via_compile() -> None:
    manager = clifft.HirPassManager()
    manager.add(clifft.PeepholeFusionPass())
    program = clifft.compile("H 0\nT 0\nT_DAG 0\nM 0", hir_passes=manager)
    assert program.peak_active_width == 0
