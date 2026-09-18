"""Public opt-in routing and output contracts for compiled logical blocks."""

from pathlib import Path

import numpy as np
import pytest
import stim

import clifft

ROOT = Path(__file__).resolve().parents[2]


def control(distance):
    return (ROOT / f"tools/bench/fixtures/css_five_check_d{distance}.stim").read_text()


@pytest.mark.parametrize("distance", [3, 5])
def test_small_controls_keep_ordinary_sampling(distance):
    text = control(distance)
    a = clifft.compile(text)
    b = clifft.compile(text, logical_blocks=True)
    assert b.num_css_blocks == 0
    assert a.peak_active_width == b.peak_active_width
    np.testing.assert_array_equal(
        clifft.sample(a, 64, seed=781).measurements,
        clifft.sample(b, 64, seed=781).measurements,
    )


@pytest.mark.parametrize("distance", [3, 5])
def test_original_cultivation_fixtures_keep_ordinary_sampling(distance):
    text = (
        ROOT / f"tools/profile/fixtures/msc/msc_d{distance}_inject_cultivate_p1e-3.stim"
    ).read_text()
    a = clifft.compile(text)
    b = clifft.compile(text, logical_blocks=True)
    assert b.num_css_blocks == 0
    np.testing.assert_array_equal(
        clifft.sample(a, 32, seed=12).measurements,
        clifft.sample(b, 32, seed=12).measurements,
    )


@pytest.mark.parametrize("distance", [7, 9])
def test_large_controls_contract_and_preserve_stim_record_parities(distance):
    text = control(distance)
    plan = clifft.compile(text, logical_blocks=True)
    assert plan.num_css_blocks == 5
    assert plan.peak_active_width == 1
    samples = clifft.sample(plan, 32, seed=674, threads=2)
    # Stim evaluates the original output declarations on reported records,
    # including any readout flips, without sharing Clifft's parity machinery.
    parity = stim.Circuit()
    for line in text.splitlines():
        if line.startswith("MPP"):
            parity.append("M", [0])
        elif line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
            parity += stim.Circuit(line)
    detectors, observables = parity.compile_m2d_converter(skip_reference_sample=True).convert(
        measurements=samples.measurements.astype(bool), separate_observables=True
    )
    np.testing.assert_array_equal(samples.detectors, detectors)
    np.testing.assert_array_equal(samples.observables, observables)
    with pytest.raises(ValueError, match="packed"):
        clifft.sample(plan, 1, batch_size=8)


def test_custom_passes_and_unsupported_suffix_preserve_ordinary_semantics():
    text = control(7)
    passes = clifft.HirPassManager()
    passes.add(clifft.RemoveNoisePass())
    a = clifft.compile(text, hir_passes=passes)
    b = clifft.compile(text, hir_passes=passes, logical_blocks=True)
    assert b.num_css_blocks == 0
    np.testing.assert_array_equal(
        clifft.sample(a, 4, seed=816).measurements,
        clifft.sample(b, 4, seed=816).measurements,
    )
    assert clifft.compile(text + "H 0\n", logical_blocks=True).num_css_blocks == 0
    assert clifft.compile(text, hir_passes=None, logical_blocks=True).num_css_blocks == 5


def test_postselection_and_zero_fault_conditioning():
    text = control(7)
    ordinary = clifft.compile(text, logical_blocks=True)
    post = clifft.compile(
        text, postselection_mask=[1] * ordinary.num_detectors, logical_blocks=True
    )
    assert post.num_css_blocks == 5
    unselected = clifft.sample_k(ordinary, 32, k=0, seed=45)
    accepted = ~unselected.detectors.any(axis=1)
    result = clifft.sample_k_survivors(post, 32, k=0, seed=45, keep_records=True)
    assert result.passed_shots == accepted.sum()
    assert not result.detectors.any()
    np.testing.assert_array_equal(result.measurements, unselected.measurements[accepted])
