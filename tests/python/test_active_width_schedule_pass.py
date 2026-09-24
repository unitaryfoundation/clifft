"""Tests for active_width_trace and ActiveWidthSchedulePass: the structural
active-width analysis and the state-aware beam-search scheduling pass built
on top of it."""

from pathlib import Path
from typing import Any

import pytest
from utils_conformance import active_width_passes as _schedule_pass_manager

import clifft

_FIXTURES = Path(__file__).parents[1] / "fixtures"

# These fixtures all plan after fusion and squeezing.
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


def test_active_width_trace_peak_matches_lowered_program() -> None:
    for fixture in _ALL_FIXTURES:
        hir = _production_hir(fixture)
        trace = clifft.active_width_trace(hir)

        assert trace["peak"] == clifft.lower(hir).peak_active_width, fixture
        assert len(trace["widths"]) == hir.num_ops, fixture
        assert len(trace["effects"]) == hir.num_ops, fixture
        assert trace["initial"] <= trace["peak"], fixture
        assert trace["final"] <= trace["peak"], fixture


def test_scheduling_is_opt_in_and_coherent_d3_reaches_peak_four() -> None:
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()
    baseline = clifft.compile(circuit)
    program = clifft.compile(circuit, hir_passes=_schedule_pass_manager())
    assert program.peak_active_width == 4
    assert baseline.peak_active_width > program.peak_active_width


def test_coherent_d5_reaches_peak_at_most_thirteen() -> None:
    circuit = (_FIXTURES / "coherent_d5_r5.stim").read_text()
    # Use a narrow beam to keep the Debug fixture test inexpensive.
    # C++ regression tests separately exercise default-budget scheduling.
    fast_pass = clifft.ActiveWidthSchedulePass(beam_width=1)
    program = clifft.compile(circuit, hir_passes=_schedule_pass_manager(fast_pass))
    assert program.peak_active_width <= 13


def test_pass_statistics_are_populated() -> None:
    # The manager runs the same pass instance, preserving its statistics.
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()
    hir = clifft.trace(clifft.parse(circuit))

    pass_ = clifft.ActiveWidthSchedulePass()
    pm = _schedule_pass_manager(pass_)
    pm.run(hir)

    assert pass_.incumbent_peak >= pass_.result_peak
    assert pass_.incumbent_dense_work >= pass_.result_dense_work
    assert isinstance(pass_.applied, bool)
    assert pass_.classification_probes > 0
    assert pass_.swept_ops > 0
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


def test_zero_search_budget_runs_greedy_scheduling() -> None:
    text = "H 0\nT 0\nH 0\nCX 0 1\nT 1\nH 1\nM 0 1"
    pass_ = clifft.ActiveWidthSchedulePass(search_budget=0.0)
    clifft.compile(text, hir_passes=_schedule_pass_manager(pass_))

    assert pass_.swept_ops > 0
    assert pass_.classification_probes > 0
    assert pass_.result_peak <= pass_.incumbent_peak


def test_none_search_budget_is_unbounded() -> None:
    # This fixture triggers narrowing with the default budget.
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()

    default_pass = clifft.ActiveWidthSchedulePass()
    clifft.compile(circuit, hir_passes=_schedule_pass_manager(default_pass))

    unbounded_pass = clifft.ActiveWidthSchedulePass(search_budget=None)
    clifft.compile(circuit, hir_passes=_schedule_pass_manager(unbounded_pass))

    assert unbounded_pass.swept_ops >= default_pass.swept_ops


def test_non_finite_search_budget_is_rejected() -> None:
    # None disables narrowing; nonfinite budget values are invalid.
    for budget in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="search_budget"):
            clifft.ActiveWidthSchedulePass(search_budget=budget)
