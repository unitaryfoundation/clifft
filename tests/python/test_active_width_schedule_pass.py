"""Tests for ActiveWidthSchedulePass: state-aware beam-search scheduling."""

from pathlib import Path

import numpy as np
from conftest import cross_binomial_tolerance

import clifft

_FIXTURES = Path(__file__).parents[1] / "fixtures"


def _schedule_pass_manager(
    schedule_pass: clifft.ActiveWidthSchedulePass | None = None,
) -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    pm.add(clifft.StatevectorSqueezePass())
    pm.add(schedule_pass if schedule_pass is not None else clifft.ActiveWidthSchedulePass())
    return pm


def test_coherent_d3_reaches_peak_four() -> None:
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()
    program = clifft.compile(circuit, hir_passes=_schedule_pass_manager())
    assert program.peak_active_width == 4


def test_coherent_d5_reaches_peak_at_most_thirteen() -> None:
    circuit = (_FIXTURES / "coherent_d5_r5.stim").read_text()
    # coherent_d5_r5 is a few thousand HIR ops. exact_node_budget already
    # defaults to 0 (see the C++ header comment: measured on the corpus,
    # exact repair never lowered a peak the beam had not already reached).
    # The beam search's own two-phase scoring keeps a full beam_width of 8
    # well under a second in Release, but Debug's lack of inlining leaves
    # per-candidate SearchFrontier::execute/undo (a std::set insert/erase
    # pair) as the dominant cost on this fixture's wide branching factor
    # regardless, so this test still narrows the beam to stay fast; see
    # test_active_width_schedule_pass.cc's matching C++ fixture test for
    # the beam_width 1 vs 8 dense-work comparison this mirrors.
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
    #
    # exact_node_budget defaults to 0, so this test passes a nonzero value
    # explicitly (cheap on this small fixture regardless) to keep the
    # exact-repair code path exercised from Python.
    circuit = (_FIXTURES / "coherent_d3_r3.stim").read_text()
    hir = clifft.trace(clifft.parse(circuit))

    pass_ = clifft.ActiveWidthSchedulePass(exact_node_budget=2000)
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
        exact_node_budget=500,
        sink_neutral_rotations=False,
    )
    assert isinstance(pass_, clifft.HirPass)
    assert pass_.applied is False

    default_pass = clifft.ActiveWidthSchedulePass()
    assert isinstance(default_pass, clifft.HirPass)
