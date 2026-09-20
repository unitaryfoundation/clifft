"""Independent sums and logical-projector checks for folded contractions."""

# ruff: noqa: E402

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cirq")

from folded_check_contraction import CodeBoundary, FactorPlan, FoldedChecks
from folded_msc import Builder
from folded_msc_family import F5Builder
from validate_folded_msc import bind_faults


def test_two_parity_factors_match_brute_force():
    masks = [(3, 12), (17, 6), (20,), (0, 9), (8,), (16,)]
    plan = FactorPlan(masks, 5)
    rng = np.random.default_rng(741)
    for _ in range(12):
        local = rng.normal(size=(len(masks), 4)) + 1j * rng.normal(size=(len(masks), 4))
        expected = 0j
        for bits in range(32):
            value = 1 + 0j
            for factor, values in zip(masks, local, strict=True):
                label = sum(((bits & mask).bit_count() % 2) << i for i, mask in enumerate(factor))
                value *= values[label]
            expected += value / 32
        assert abs(plan.evaluate(local) - expected) < 1e-12


def test_large_leaf_is_rejected_before_expansion():
    with pytest.raises(ValueError, match="storage limit"):
        FactorPlan([((1 << 30) - 1,)], 30)


@pytest.mark.parametrize("distance", [3, 5])
def test_two_folded_checks_match_logical_projectors(distance):
    kernel = FoldedChecks(distance)
    builder = Builder(0) if distance == 3 else F5Builder(0)
    builder.logical_check("first")
    builder.logical_check("second")
    operations = bind_faults(builder.circuit, {}).operations
    logical = np.array([np.cos(0.37), np.exp(0.81j) * np.sin(0.37)])
    boundary = CodeBoundary(0, 0, logical, 0)
    observable = np.array([[0, np.exp(-1j * np.pi / 4)], [np.exp(1j * np.pi / 4), 0]])
    total = 0
    for a, b in itertools.product((0, 1), repeat=2):
        bound = kernel.bind(operations, boundary, (a, b))
        p_a = (np.eye(2) + (-1) ** a * observable) / 2
        p_b = (np.eye(2) + (-1) ** b * observable) / 2
        expected = p_b @ p_a @ logical
        actual = kernel.amplitudes(bound, 0)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        probability = kernel.marginal(bound, 0, 0)
        assert abs(probability - np.vdot(expected, expected).real) < 1e-12
        total += probability
    assert abs(total - 1) < 1e-12


def test_native_folded_samples_match_logical_born_probabilities(tmp_path):
    from audit_folded_contraction import export_bundle

    binary = Path("build-study/sample_folded_checks").resolve()
    if not binary.is_file():
        pytest.skip("build the folded contraction sampler")
    kernel = FoldedChecks(3)
    builder = Builder(0)
    builder.logical_check("first")
    builder.logical_check("second")
    operations = bind_faults(builder.circuit, {}).operations
    logical = np.array([np.cos(0.37), np.exp(0.81j) * np.sin(0.37)])
    boundary = CodeBoundary(0, 0, logical, 0)
    choices = [
        kernel.bind(operations, boundary, bits) for bits in itertools.product((0, 1), repeat=2)
    ]
    bundle = tmp_path / "bundle"
    export_bundle(bundle, kernel, [choices])
    output = subprocess.check_output([str(binary), str(bundle), "1", "2048", "742"], text=True)
    samples = [json.loads(line) for line in output.splitlines()][:-1]
    assert all(s["root_outcomes"] in (0, 3) and s["syndrome"] == 0 for s in samples)
    observable = np.array([[0, np.exp(-1j * np.pi / 4)], [np.exp(1j * np.pi / 4), 0]])
    probability = float((1 + np.vdot(logical, observable @ logical).real) / 2)
    frequency = sum(s["root_outcomes"] == 0 for s in samples) / len(samples)
    assert abs(frequency - probability) < 6 * np.sqrt(
        probability * (1 - probability) / len(samples)
    )
    for sample in samples:
        a, b = (complex(*value) for value in sample["logical"])
        sign = 1 if sample["root_outcomes"] == 0 else -1
        assert abs(2 * a.conjugate() * b - sign * np.exp(1j * np.pi / 4)) < 1e-12
