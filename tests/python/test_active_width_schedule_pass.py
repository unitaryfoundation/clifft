"""Tests for active_width_trace and ActiveWidthSchedulePass: the structural
active-width analysis and the state-aware beam-search scheduling pass built
on top of it."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
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


def test_zero_beam_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        clifft.ActiveWidthSchedulePass(beam_width=0)


def test_negative_search_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="search_budget"):
        clifft.ActiveWidthSchedulePass(search_budget=-1.0)


def test_zero_search_budget_reports_positive_swept_ops() -> None:
    # A T isolated on one qubit, or entangled through a CX with nothing else
    # to hide its phase in, is something peephole/squeeze can already reduce
    # to a stabilizer state before the schedule pass ever sees it; that
    # reaches peak active width 0 and takes the pass's early exit without
    # running the beam search at all. Basis-rotating a T on each half of a
    # Bell pair (H, T, H around the CX on both qubits) keeps a genuine
    # non-Clifford phase alive through the entangling gate, so the search
    # runs and, even narrowed to width 1 immediately, sweeps the ops its own
    # initial closure and first candidate touch.
    text = "H 0\nT 0\nH 0\nCX 0 1\nT 1\nH 1\nM 0 1"
    pass_ = clifft.ActiveWidthSchedulePass(search_budget=0.0)
    clifft.compile(text, hir_passes=_schedule_pass_manager(pass_))

    assert pass_.swept_ops > 0


def test_none_search_budget_is_unbounded() -> None:
    # coherent_d3_r3 is the fixture the C++ suite's own "default search
    # budget narrows the search ... without losing its peak" test uses to
    # show the default (8) narrows the beam partway through the search. An
    # explicit None must reproduce the old always-full-beam_width search
    # instead, which never sweeps fewer ops than the narrowed default one.
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()

    default_pass = clifft.ActiveWidthSchedulePass()
    clifft.compile(circuit, hir_passes=_schedule_pass_manager(default_pass))

    unbounded_pass = clifft.ActiveWidthSchedulePass(search_budget=None)
    clifft.compile(circuit, hir_passes=_schedule_pass_manager(unbounded_pass))

    assert unbounded_pass.swept_ops >= default_pass.swept_ops


def test_non_finite_search_budget_is_rejected() -> None:
    # Infinity cannot mean "unbounded" here (None does instead) because
    # Release builds use -ffast-math, under which the compiler may treat
    # any comparison involving a non-finite double as unreachable; infinity
    # and NaN are therefore both rejected rather than given any meaning.
    for budget in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="search_budget"):
            clifft.ActiveWidthSchedulePass(search_budget=budget)
