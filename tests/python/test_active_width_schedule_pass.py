"""Tests for active_width_trace and ActiveWidthSchedulePass: the structural
active-width analysis and the state-aware beam-search scheduling pass built
on top of it."""

from pathlib import Path
from typing import Any

import numpy as np
from conftest import cross_binomial_tolerance

import clifft

_FIXTURES = Path(__file__).parents[1] / "fixtures"

# Same set test_active_width_analysis.cc's fixture differential test uses;
# the Peephole+Squeeze pipeline is what plans successfully on every one of
# them.
_ALL_FIXTURES = [
    "coherent_d3_r3.stim",
    "coherent_d5_r5.stim",
    "cultivation_d5.stim",
    "surface_d7_r7_p001.stim",
    "qv10.stim",
    "surface_d11_r11_p001.stim",
    "surface_d5_r5_p05.stim",
    "target_qec.stim",
]


def _production_hir(name: str) -> Any:
    circuit = (_FIXTURES / name).read_text()
    hir = clifft.trace(clifft.parse(circuit))
    passes = clifft.HirPassManager()
    passes.add(clifft.PeepholeFusionPass())
    passes.add(clifft.StatevectorSqueezePass())
    passes.run(hir)
    return hir


def _schedule_pass_manager(
    schedule_pass: clifft.ActiveWidthSchedulePass | None = None,
) -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(clifft.StatevectorSqueezePass())
    pm.add(schedule_pass if schedule_pass is not None else clifft.ActiveWidthSchedulePass())
    return pm


def test_active_width_trace_peak_matches_lowered_program() -> None:
    for fixture in _ALL_FIXTURES:
        hir = _production_hir(fixture)
        trace = clifft.active_width_trace(hir)

        assert trace["peak"] == clifft.lower(hir).peak_active_width, fixture
        assert len(trace["widths"]) == hir.num_ops, fixture
        assert len(trace["effects"]) == hir.num_ops, fixture
        assert trace["initial"] <= trace["peak"], fixture
        assert trace["final"] <= trace["peak"], fixture


def test_coherent_d3_reaches_peak_four() -> None:
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()
    program = clifft.compile(circuit, hir_passes=_schedule_pass_manager())
    assert program.peak_active_width == 4


def test_coherent_d5_reaches_peak_at_most_thirteen() -> None:
    circuit = (_FIXTURES / "coherent_d5_r5.stim").read_text()
    # coherent_d5_r5 is a few thousand HIR ops with a wide branching factor;
    # a Debug extension takes on the order of 20 seconds at the default
    # beam_width of 8 here, so this test narrows the beam the same way
    # test_active_width_schedule_pass.cc's matching C++ fixture test does.
    # See that test for the peak and dense-work figures this mirrors.
    fast_pass = clifft.ActiveWidthSchedulePass(beam_width=1)
    program = clifft.compile(circuit, hir_passes=_schedule_pass_manager(fast_pass))
    assert program.peak_active_width <= 13


def _assert_columns_match(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    assert reference.shape == candidate.shape
    shots = reference.shape[0]
    reference_probabilities = reference.mean(axis=0, dtype=float)
    candidate_probabilities = candidate.mean(axis=0, dtype=float)

    # A six-sigma bound over this test's small number of comparisons keeps
    # the Gaussian union-bound false-failure probability negligible, the
    # same tolerance test_squeeze_benchmark_integration.py's three-way
    # oracle uses to compare pipeline variants against each other.
    for column, (reference_probability, candidate_probability) in enumerate(
        zip(reference_probabilities, candidate_probabilities, strict=True)
    ):
        pooled = float((reference_probability + candidate_probability) / 2.0)
        tolerance = cross_binomial_tolerance(pooled, shots, sigma=6.0)
        difference = abs(float(reference_probability - candidate_probability))
        assert difference < tolerance, (
            f"{label} column {column}: {reference_probability:.5f} vs "
            f"{candidate_probability:.5f}, difference {difference:.6f} "
            f">= tolerance {tolerance:.6f}"
        )


def test_coherent_d3_sampling_matches_unoptimized() -> None:
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()
    unoptimized = clifft.compile(circuit, hir_passes=None)
    scheduled = clifft.compile(circuit, hir_passes=_schedule_pass_manager())

    shots = 10_000
    reference = clifft.sample(unoptimized, shots, seed=61_001)
    candidate = clifft.sample(scheduled, shots, seed=61_002)

    for field in ("measurements", "detectors", "observables"):
        _assert_columns_match(getattr(reference, field), getattr(candidate, field), label=field)


def test_pass_statistics_are_populated() -> None:
    # Individual HirPass instances have no run() of their own in Python;
    # HirPassManager.add() wraps a non-owning delegate around the
    # Python-owned pass and HirPassManager.run() drives it, so the original
    # pass_ object's statistics are populated in place afterward.
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()
    hir = clifft.trace(clifft.parse(circuit))

    pass_ = clifft.ActiveWidthSchedulePass()
    pm = _schedule_pass_manager(pass_)
    pm.run(hir)

    assert pass_.incumbent_peak >= pass_.result_peak
    assert pass_.incumbent_dense_work >= pass_.result_dense_work
    assert isinstance(pass_.applied, bool)
    assert "ActiveWidthSchedulePass" in repr(pass_)


def test_keyword_only_construction() -> None:
    pass_ = clifft.ActiveWidthSchedulePass(
        noise_transparent=False,
        beam_width=2,
        sink_neutral_rotations=False,
    )
    assert isinstance(pass_, clifft.HirPass)
    assert pass_.applied is False

    default_pass = clifft.ActiveWidthSchedulePass()
    assert isinstance(default_pass, clifft.HirPass)
