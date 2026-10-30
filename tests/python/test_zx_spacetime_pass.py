"""End-to-end checks of ZxSpacetimePass against the standard pipeline."""

from pathlib import Path

import numpy as np
import pytest
from conftest import cross_binomial_tolerance

import clifft

_FIXTURES = Path(__file__).parent.parent / "fixtures"


def _zx_pipeline() -> tuple[clifft.HirPassManager, clifft.ZxSpacetimePass]:
    zx = clifft.ZxSpacetimePass()
    pm = clifft.HirPassManager()
    pm.add(zx)
    pm.add(clifft.PeepholeFusionPass())
    pm.add(clifft.PhasePolynomialPass())
    pm.add(clifft.RotationSimplificationPass())
    pm.add(clifft.StatevectorSqueezePass())
    return pm, zx


def _assert_marginals_match(a: np.ndarray, b: np.ndarray, label: str) -> None:
    shots = a.shape[0]
    pa = a.mean(axis=0, dtype=float)
    pb = b.mean(axis=0, dtype=float)
    for i in range(len(pa)):
        tol = cross_binomial_tolerance((pa[i] + pb[i]) / 2, shots, sigma=5.0)
        assert abs(pa[i] - pb[i]) < tol, f"{label} {i}: {pa[i]:.5f} vs {pb[i]:.5f}"


@pytest.fixture(scope="module")
def cultivation_d5() -> str:
    return (_FIXTURES / "cultivation_d5.stim").read_text()


def test_cultivation_d5_narrows_to_tsim_width(cultivation_d5: str) -> None:
    pm, zx = _zx_pipeline()
    program = clifft.compile(cultivation_d5, hir_passes=pm)
    assert zx.applied, zx.reason
    assert (zx.input_peak, zx.result_peak) == (10, 6)
    assert program.peak_active_width == 6
    assert clifft.compile(cultivation_d5).peak_active_width == 10


@pytest.mark.parametrize(
    "shots", [10_000, pytest.param(100_000, marks=pytest.mark.expensive, id="100k")]
)
def test_cultivation_d5_samples_match_standard_pipeline(cultivation_d5: str, shots: int) -> None:
    pm, zx = _zx_pipeline()
    lowered = clifft.compile(cultivation_d5, hir_passes=pm, normalize_syndromes=True)
    assert zx.applied, zx.reason
    standard = clifft.compile(cultivation_d5, normalize_syndromes=True)
    a = clifft.sample(standard, shots, seed=11)
    b = clifft.sample(lowered, shots, seed=12)
    _assert_marginals_match(a.measurements, b.measurements, "record")
    # Both are normalized by their own noiseless reference, so this also
    # checks that the references agree.
    _assert_marginals_match(a.detectors, b.detectors, "detector")
    _assert_marginals_match(a.observables, b.observables, "observable")


def test_unsupported_circuit_is_left_unchanged() -> None:
    pm, zx = _zx_pipeline()
    program = clifft.compile("H 0\nT 0\nEXP_VAL X0\nM 0\n", hir_passes=pm)
    assert not zx.applied
    assert zx.reason == "expectation-value probes name physical qubits"
    assert program.num_measurements == 1


def test_limits_leave_circuits_unchanged(cultivation_d5: str) -> None:
    pm = clifft.HirPassManager()
    zx = clifft.ZxSpacetimePass(max_reduce_work=1000)
    pm.add(zx)
    clifft.compile(cultivation_d5, hir_passes=pm)
    assert not zx.applied
    assert zx.reason == "the diagram is too dense to reduce within the work limit"

    pm = clifft.HirPassManager()
    zx = clifft.ZxSpacetimePass(max_qubits=8)
    pm.add(zx)
    clifft.compile(cultivation_d5, hir_passes=pm)
    assert not zx.applied
    assert zx.reason == "fresh qubits for resets would exceed the qubit limit"


@pytest.mark.parametrize("option", ["max_qubits", "max_reduce_work"])
def test_limits_must_be_positive(option: str) -> None:
    with pytest.raises(ValueError, match=f"{option} must be positive"):
        clifft.ZxSpacetimePass(**{option: 0})
