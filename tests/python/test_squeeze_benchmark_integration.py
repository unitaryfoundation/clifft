"""Integration coverage for squeeze-pass behavior on coherent QEC fixtures."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim
from conftest import cross_binomial_tolerance

import clifft

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_D3_SHOTS = 10_000
_D3_PARITY_GROUPS = (
    (0, 1),
    (7, 8),
    (15, 16),
    (24, 25),
    (31, 32),
    (0, 8, 16),
    (8, 16, 24, 32),
    (33, 34),
    (40, 41, 42),
    (33, 40, 48, 56),
    (0, 16, 32, 57),
)


@dataclass(frozen=True)
class _PipelinePrograms:
    circuit: str
    unoptimized: Any
    peephole_only: Any
    production: Any


def _compile_pipeline_variants(name: str) -> _PipelinePrograms:
    circuit = (_FIXTURES / name).read_text()
    peephole = clifft.HirPassManager()
    peephole.add(clifft.PeepholeFusionPass())
    return _PipelinePrograms(
        circuit=circuit,
        unoptimized=clifft.compile(circuit, hir_passes=None),
        peephole_only=clifft.compile(circuit, hir_passes=peephole),
        production=clifft.compile(circuit),
    )


@pytest.fixture(scope="module")
def coherent_d3_programs() -> _PipelinePrograms:
    return _compile_pipeline_variants("coherent_d3_r3.stim")


@pytest.fixture(scope="module")
def coherent_d5_programs() -> _PipelinePrograms:
    return _compile_pipeline_variants("coherent_d5_r5.stim")


def _record_converter(circuit: str) -> Any:
    # Stim uses R_Z for reset, while Clifft accepts R_Z(angle) as a coherent
    # rotation. Replacing rotations by identity preserves only the record and
    # annotation structure needed by the measurement-to-detector converter.
    converter_circuit = re.sub(r"(?m)^(\s*)R_Z\([^)]*\)", r"\1I", circuit)
    return stim.Circuit(converter_circuit).compile_m2d_converter()


def _assert_annotations_match_records(converter: Any, result: Any) -> None:
    detectors, observables = converter.convert(
        measurements=result.measurements.astype(bool, copy=False),
        separate_observables=True,
    )
    np.testing.assert_array_equal(result.detectors, detectors)
    np.testing.assert_array_equal(result.observables, observables)


def _assert_column_probabilities_match(
    reference: np.ndarray, candidate: np.ndarray, *, label: str
) -> None:
    assert reference.shape == candidate.shape
    shots = reference.shape[0]
    reference_probabilities = reference.mean(axis=0, dtype=float)
    candidate_probabilities = candidate.mean(axis=0, dtype=float)

    # This test makes fewer than 300 comparisons. A six-sigma bound keeps the
    # Gaussian union-bound false-failure probability below one in a million.
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


def _selected_d3_parities(result: Any) -> np.ndarray:
    joint = np.concatenate((result.measurements, result.detectors, result.observables), axis=1)
    assert joint.shape[1] == 58
    return np.column_stack(
        [np.bitwise_xor.reduce(joint[:, group], axis=1) for group in _D3_PARITY_GROUPS]
    )


def _assert_d3_semantics_match(reference: Any, candidate: Any, *, label: str) -> None:
    for field in ("measurements", "detectors", "observables"):
        _assert_column_probabilities_match(
            getattr(reference, field),
            getattr(candidate, field),
            label=f"{label} {field}",
        )
    _assert_column_probabilities_match(
        _selected_d3_parities(reference),
        _selected_d3_parities(candidate),
        label=f"{label} selected parities",
    )


def test_coherent_d5_production_exercises_convoy_bypass(
    coherent_d5_programs: _PipelinePrograms,
) -> None:
    programs = coherent_d5_programs
    assert programs.unoptimized.peak_active_width == 24
    assert programs.peephole_only.peak_active_width == 24
    assert programs.production.peak_active_width == 13
    assert programs.production.peak_active_width < programs.peephole_only.peak_active_width


def test_coherent_d5_sampling_modes_preserve_annotations(
    coherent_d5_programs: _PipelinePrograms,
) -> None:
    programs = coherent_d5_programs
    converter = _record_converter(programs.circuit)

    # The width-24 squeeze-off plan is intentionally limited to one automatic
    # shot. Production is cheap enough to cross scalar, automatic, and an
    # explicit packed capacity without building the expensive full matrix.
    results = (
        clifft.sample(programs.peephole_only, 1, seed=51_001, batch_size="auto"),
        clifft.sample(programs.production, 5, seed=51_002, batch_size=1),
        clifft.sample(programs.production, 65, seed=51_003, batch_size="auto"),
        clifft.sample(programs.production, 65, seed=51_004, batch_size=65),
    )
    for result in results:
        assert result.measurements.shape[1] == 145
        assert result.detectors.shape[1] == 120
        assert result.observables.shape[1] == 1
        _assert_annotations_match_records(converter, result)


def test_coherent_d3_three_way_semantic_oracle(
    coherent_d3_programs: _PipelinePrograms,
) -> None:
    programs = coherent_d3_programs
    assert programs.unoptimized.peak_active_width == 8
    assert programs.peephole_only.peak_active_width == 8
    assert programs.production.peak_active_width == 4
    assert programs.production.peak_active_width < programs.peephole_only.peak_active_width

    samples = {
        "unoptimized packed": clifft.sample(
            programs.unoptimized, _D3_SHOTS, seed=53_001, batch_size=257
        ),
        "peephole-only packed": clifft.sample(
            programs.peephole_only, _D3_SHOTS, seed=53_002, batch_size=257
        ),
        "production packed": clifft.sample(
            programs.production, _D3_SHOTS, seed=53_003, batch_size=257
        ),
        "production scalar": clifft.sample(
            programs.production, _D3_SHOTS, seed=53_004, batch_size=1
        ),
        "production automatic": clifft.sample(
            programs.production, _D3_SHOTS, seed=53_005, batch_size="auto"
        ),
    }
    converter = _record_converter(programs.circuit)
    for result in samples.values():
        assert result.measurements.shape == (_D3_SHOTS, 33)
        assert result.detectors.shape == (_D3_SHOTS, 24)
        assert result.observables.shape == (_D3_SHOTS, 1)
        _assert_annotations_match_records(converter, result)

    reference = samples["unoptimized packed"]
    for label, result in samples.items():
        if result is not reference:
            _assert_d3_semantics_match(reference, result, label=label)
