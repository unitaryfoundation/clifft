"""Tests for the active_width_trace and search_width_schedule bindings."""

from pathlib import Path
from typing import Any

from conftest import cross_binomial_tolerance

import clifft

_FIXTURES = Path(__file__).parents[1] / "fixtures"

# Same set test_active_width_analysis.cc's fixture differential test uses;
# the production pipeline is what plans successfully on every one of them.
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
    clifft.default_hir_pass_manager().run(hir)
    return hir


def _peephole_squeeze_hir(name: str) -> Any:
    # Explicitly peephole+squeeze rather than the full default pipeline: the
    # search-certificate tests below measure how much further
    # search_width_schedule can improve on top of that incumbent, which is a
    # different question from what the default pipeline (which now also
    # schedules) settles on.
    circuit = (_FIXTURES / name).read_text()
    hir = clifft.trace(clifft.parse(circuit))
    passes = clifft.HirPassManager()
    passes.add(clifft.PeepholeFusionPass())
    passes.add(clifft.StatevectorSqueezePass())
    passes.run(hir)
    return hir


def _assert_column_probabilities_match(reference: Any, candidate: Any, *, label: str) -> None:
    # Same six-sigma cross-binomial column comparison
    # test_squeeze_benchmark_integration.py's three-way oracle uses.
    assert reference.shape == candidate.shape
    shots = reference.shape[0]
    reference_probabilities = reference.mean(axis=0, dtype=float)
    candidate_probabilities = candidate.mean(axis=0, dtype=float)
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


def test_active_width_trace_peak_matches_lowered_program() -> None:
    for fixture in _ALL_FIXTURES:
        hir = _production_hir(fixture)
        trace = clifft.active_width_trace(hir)

        assert trace["peak"] == clifft.lower(hir).peak_active_width, fixture
        assert len(trace["widths"]) == hir.num_ops, fixture
        assert len(trace["effects"]) == hir.num_ops, fixture
        assert trace["initial"] <= trace["peak"], fixture
        assert trace["final"] <= trace["peak"], fixture


def test_search_width_schedule_certifies_coherent_d3_r3() -> None:
    hir = _peephole_squeeze_hir("coherent_d3_r3.stim")
    result = clifft.search_width_schedule(hir, noise_transparent=True, apply=False)

    assert result["incumbent_peak"] == 5
    assert result["upper_bound"] == 4
    assert result["lower_bound"] == 4
    assert result["optimal"] is True
    assert result["noise_transparent"] is True


def test_search_width_schedule_apply_lowers_peak_and_preserves_sampling() -> None:
    unoptimized_hir = _peephole_squeeze_hir("coherent_d3_r3.stim")
    unoptimized_program = clifft.lower(unoptimized_hir)
    assert unoptimized_program.peak_active_width == 5

    optimized_hir = _peephole_squeeze_hir("coherent_d3_r3.stim")
    result = clifft.search_width_schedule(optimized_hir, noise_transparent=True, apply=True)
    assert result["upper_bound"] == 4

    optimized_program = clifft.lower(optimized_hir)
    assert optimized_program.peak_active_width == 4

    shots = 20_000
    unoptimized_result = clifft.sample(unoptimized_program, shots, seed=90_001)
    optimized_result = clifft.sample(optimized_program, shots, seed=90_002)

    for field in ("measurements", "detectors", "observables"):
        _assert_column_probabilities_match(
            getattr(unoptimized_result, field),
            getattr(optimized_result, field),
            label=field,
        )


def test_search_width_schedule_apply_false_leaves_hir_unchanged() -> None:
    hir = _peephole_squeeze_hir("coherent_d3_r3.stim")
    before_peak = clifft.lower(hir).peak_active_width

    result = clifft.search_width_schedule(hir, noise_transparent=True, apply=False)
    assert result["upper_bound"] < result["incumbent_peak"]

    after_peak = clifft.lower(hir).peak_active_width
    assert after_peak == before_peak
