"""Checks for corpus conversion and conservative screening diagnostics."""

import os
import re
from pathlib import Path

import numpy as np
import pytest
import stim
from cultivation_study import CORPUS, corpus, profile, variant_text

import clifft


def test_pinned_sources_and_conversion_preserve_physical_instructions():
    for entry in corpus()["files"]:
        if not entry["file"].endswith(".stim"):
            continue
        source = (CORPUS / entry["file"]).read_text()
        converted = variant_text(source, 0.001)
        restored = re.sub(r"(?m)^([ \t]*)T(_DAG)?(?=[ \t\r\n]|$)", r"\1S\2", converted)
        assert restored == source
        assert variant_text(source, 0.001, "S") == source


@pytest.mark.parametrize("path", sorted(CORPUS.glob("*.stim")), ids=lambda p: p.stem)
def test_converted_noiseless_circuits_have_trivial_outputs(path):
    program = clifft.compile(variant_text(path.read_text(), 0))
    sample = clifft.sample(program, 128, seed=72, threads=1)
    assert not sample.detectors.any()
    assert not sample.observables.any()
    converter = stim.Circuit(path.read_text()).compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=sample.measurements.astype(bool), separate_observables=True
    )
    np.testing.assert_array_equal(detectors, sample.detectors)
    np.testing.assert_array_equal(observables, sample.observables)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]", 1),
        ("H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]", 0),
        ("H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]", 1),
        ("M(0.1) 0\nDETECTOR rec[-1]", 1),
        ("H 0\nT 0\nH 0\nM(0.1) 0\nDETECTOR rec[-1]", 0),
    ],
)
def test_screening_distinguishes_fault_parities_from_quantum_outcomes(text, expected):
    binary = Path(os.environ.get("CLIFFT_CULTIVATION_PROFILER", "build-study/profile_cultivation"))
    if not binary.is_file():
        pytest.skip("build profile_cultivation to test native diagnostics")
    result = profile(text, binary.resolve())
    assert result["detector_count"] == 1
    assert result["state_independent_detector_count"] == expected
    assert all(
        1 <= line <= len(text.splitlines()) for a in result["actions"] for line in a["lines"]
    )
