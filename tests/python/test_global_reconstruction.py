"""Output-specific sampling plans retain the original program's other contracts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import stim
from conftest import assert_statevectors_equiv
from utils_conformance import CPU_SAMPLING_MODES
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

import clifft
from clifft import _clifft_core

UNITARY = "H 0 1\nT 0 1\nH 0 1\n"
OUTPUTS = "M 0 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
SOURCE = UNITARY + OUTPUTS
NOISY = UNITARY + "DEPOLARIZE2(0.2) 0 1\n" + OUTPUTS


def compile_selected(source=SOURCE, **kwargs):
    # Keep the independent rotations concurrent in the original plan so this
    # small circuit exercises actual selection as well as the full MSC fixtures.
    program = clifft.compile(source, hir_passes=None, reconstruction=True, **kwargs)
    assert program.reconstruction_info()["selected"]
    assert program.reconstruction_info(False)["selected"]
    return program


def assert_same_result(a, b):
    for name in ("measurements", "detectors", "observables", "exp_vals", "observable_ones"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ("total_shots", "passed_shots", "logical_errors"):
        assert getattr(a, name) == getattr(b, name)


def test_program_metadata_and_exact_queries_keep_original_contract():
    original = clifft.compile(SOURCE, hir_passes=None)
    selected = compile_selected()
    assert isinstance(selected, clifft.Program)
    assert selected.inspect() == original.inspect()
    assert selected.inspect_action(0) == original.inspect_action(0)
    assert repr(selected) == repr(original)
    assert selected.peak_active_width == 2
    assert selected.reconstruction_info()["peak_active_width"] == 1
    records = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    probabilities = np.abs(qiskit_statevector(stim_to_qiskit_noiseless(UNITARY))) ** 2
    measured = clifft.compile(UNITARY + "M 0 1", hir_passes=None, reconstruction=True)
    assert measured.reconstruction_info()["selected"]
    np.testing.assert_allclose(clifft.record_probabilities(measured, records), probabilities)
    for program in (original, selected):
        with pytest.raises(ValueError, match="detectors"):
            clifft.record_probabilities(program, records)
    for record in records:
        assert _clifft_core._replay_record(
            selected, record.tolist()
        ) == _clifft_core._replay_record(original, record.tolist())
    for program in (original, selected):
        with pytest.raises(ValueError):
            clifft.get_statevector(program)

    unitary = clifft.compile(UNITARY, reconstruction=True)
    assert not unitary.reconstruction_info()["selected"]
    reference = qiskit_statevector(stim_to_qiskit_noiseless(UNITARY))
    qasm = clifft.compile(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q; t q; h q;',
        input_format="qasm2",
        reconstruction=True,
    )
    assert not qasm.reconstruction_info()["selected"]
    assert_statevectors_equiv(clifft.get_statevector(qasm), reference)
    assert_statevectors_equiv(clifft.get_statevector(unitary), reference)
    np.testing.assert_allclose(clifft.basis_probabilities(unitary, records), np.abs(reference) ** 2)
    lowered = clifft.lower(clifft.trace(clifft.parse(UNITARY + "M 0 1")))
    assert isinstance(lowered, clifft.Program)
    np.testing.assert_allclose(clifft.record_probabilities(lowered, records), probabilities)


@pytest.mark.parametrize("sampling_mode", CPU_SAMPLING_MODES, indirect=True, ids=lambda m: m.name)
def test_selected_sampling_replays_and_switches_output_contracts(sampling_mode):
    program = compile_selected(NOISY)
    ordinary = sampling_mode.sample(program, 131, seed=104)
    assert_same_result(ordinary, sampling_mode.sample(program, 131, seed=104))
    for keep_records in (False, True, False, True):
        result = sampling_mode.sample_survivors(program, 131, seed=27, keep_records=keep_records)
        assert result.total_shots == result.passed_shots == 131
        assert result.measurements.shape == (131 if keep_records else 0, 2)
        assert result.detectors.shape == (131 if keep_records else 0, 1)
        assert result.observables.shape == (131 if keep_records else 0, 1)
        if keep_records:
            np.testing.assert_array_equal(
                result.detectors[:, 0], result.measurements.sum(axis=1) % 2
            )
            np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, 0])
            assert result.observable_ones[0] == result.observables.sum()
        assert_same_result(
            result,
            sampling_mode.sample_survivors(program, 131, seed=27, keep_records=keep_records),
        )


def test_one_program_supports_concurrent_output_contracts():
    program = compile_selected(NOISY)

    def run(keep_records):
        return clifft.sample_survivors(
            program, 131, seed=8, keep_records=keep_records, batch_size=65
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [False, True] * 4))
    for index, result in enumerate(results):
        assert_same_result(result, results[index % 2])


@pytest.mark.parametrize("keep_records", [False, True])
def test_reference_parities_and_postselection_preserve_joint_law(keep_records):
    source = (
        UNITARY + "M 0 1\nDETECTOR rec[-2]\nREADOUT_NOISE(0.2) rec[-2]\n"
        "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
    )
    program = compile_selected(
        source, postselection_mask=[1, 0], expected_detectors=[1, 1], expected_observables=[1]
    )
    result = clifft.sample_survivors(
        program, 20000, seed=24, keep_records=keep_records, batch_size=65
    )
    # The first detector snapshots m0 before the classical flip. Conditioning
    # on m0=1 leaves m1 independent and gives final m0=0 with probability 0.2.
    p_one = np.sin(np.pi / 8) ** 2
    assert abs(result.passed_shots / result.total_shots - p_one) < 0.015
    assert abs(result.logical_errors / result.passed_shots - 0.2) < 0.04
    if keep_records:
        assert not result.detectors[:, 0].any()
        np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, 0] ^ 1)
        np.testing.assert_array_equal(
            result.detectors[:, 1], np.bitwise_xor.reduce(result.measurements, axis=1) ^ 1
        )
        assert abs(result.measurements[:, 1].mean() - p_one) < 0.04


@pytest.mark.parametrize("keep_records", [False, True])
def test_normalization_uses_the_original_reference(keep_records):
    selected = compile_selected(normalize_syndromes=True, postselection_mask=[1])
    hir, det_ref, obs_ref = _clifft_core._prepare_hir_for_lowering(
        SOURCE, normalize_syndromes=True, postselection_mask=[1]
    )
    explicit = compile_selected(
        postselection_mask=[1], expected_detectors=det_ref, expected_observables=obs_ref
    )
    assert_same_result(
        clifft.sample_survivors(selected, 513, seed=9, keep_records=keep_records),
        clifft.sample_survivors(explicit, 513, seed=9, keep_records=keep_records),
    )


def test_selected_clifford_distribution_matches_stim():
    # Canceling rotations keep this reference within Stim's supported gates,
    # while an unoptimized original still has a nonzero active width.
    clifford = "H 0\nCX 0 1\nDEPOLARIZE2(0.2) 0 1\nM 0 1\n"
    program = compile_selected("H 0\nT 0\nT_DAG 0\nH 0\n" + clifford + "DETECTOR rec[-1]\n")
    actual = clifft.sample(program, 20000, seed=12, batch_size=65).measurements
    expected = stim.Circuit(clifford).compile_sampler(seed=94).sample(20000)
    weights = np.array([1, 2])
    actual_law = np.bincount(actual @ weights, minlength=4) / len(actual)
    expected_law = np.bincount(expected @ weights, minlength=4) / len(expected)
    np.testing.assert_allclose(actual_law, expected_law, atol=0.025, rtol=0)


def test_selected_nonclifford_joint_law_matches_aer():
    program = compile_selected()
    reference = np.abs(qiskit_statevector(stim_to_qiskit_noiseless(UNITARY))) ** 2
    samples = clifft.sample(program, 20000, seed=18, batch_size=65)
    law = np.bincount(samples.measurements @ np.array([1, 2]), minlength=4) / 20000
    np.testing.assert_allclose(law, reference, atol=0.015, rtol=0)


@pytest.mark.parametrize("postselection", [False, True])
def test_forced_faults_keep_original_sites_and_seeded_results(
    importance_sampling_mode, postselection
):
    mask = [1] if postselection else None
    selected = compile_selected(NOISY, postselection_mask=mask)
    original = clifft.compile(NOISY, hir_passes=None, postselection_mask=mask)
    np.testing.assert_array_equal(
        selected.noise_site_probabilities, original.noise_site_probabilities
    )
    mode = importance_sampling_mode
    for k in (0, 1):
        for keep_records in (False, True):
            assert_same_result(
                mode.sample_k_survivors(selected, 131, k, seed=94, keep_records=keep_records),
                mode.sample_k_survivors(original, 131, k, seed=94, keep_records=keep_records),
            )
        if not postselection:
            assert_same_result(
                mode.sample_k(selected, 131, k, seed=94), mode.sample_k(original, 131, k, seed=94)
            )


@pytest.mark.parametrize(
    "source, reason",
    [
        (UNITARY + "EXP_VAL Z0\n" + OUTPUTS, "EXP_VAL"),
        ("R_X(0.13) 0\nM 0\n", "unsupported"),
        (UNITARY + OUTPUTS + "READOUT_NOISE(0.1,0.2) rec[-1]\n", "readout"),
        ("H 0\nM 0\n", "already zero"),
        (SOURCE, "does not reduce"),
    ],
)
def test_unsupported_or_unprofitable_sampling_falls_back(source, reason):
    original = clifft.compile(source)
    fallback = clifft.compile(source, reconstruction=True)
    assert not fallback.reconstruction_info()["selected"]
    assert reason.lower() in fallback.reconstruction_info()["fallback_reason"].lower()
    assert_same_result(clifft.sample(original, 65, seed=9), clifft.sample(fallback, 65, seed=9))


@pytest.mark.parametrize("pass_type", [clifft.RemoveNoisePass, clifft.DropNonUnitaryPass])
def test_intentional_hir_transformations_are_not_undone(pass_type):
    passes = clifft.HirPassManager()
    passes.add(pass_type())
    original = clifft.compile(NOISY, hir_passes=passes)
    fallback = clifft.compile(NOISY, hir_passes=passes, reconstruction=True)
    assert not fallback.reconstruction_info()["selected"]
    assert "HIR passes" in fallback.reconstruction_info()["fallback_reason"]
    assert_same_result(clifft.sample(original, 65, seed=9), clifft.sample(fallback, 65, seed=9))


def test_reconstruction_does_not_suppress_invalid_source_errors():
    for source in ("M 0\nDETECTOR rec[-2]", "X_ERROR(1.1) 0", "UNKNOWN 0"):
        with pytest.raises(clifft.ParseError):
            clifft.compile(source, reconstruction=True)


@pytest.mark.expensive
@pytest.mark.parametrize("distance, width", [(5, 6), (7, 8)])
def test_full_noisy_msc_uses_reconstructed_plans_with_default_passes(distance, width):
    source = (
        Path(__file__).parents[1] / "fixtures" / f"reconstruction_msc_d{distance}.stim"
    ).read_text()
    program = clifft.compile(source, reconstruction=True)
    for keep_records in (False, True):
        info = program.reconstruction_info(keep_records)
        assert info["selected"] and info["peak_active_width"] == width
        result = clifft.sample_survivors(
            program, 65, seed=7, keep_records=keep_records, batch_size=65
        )
        assert result.total_shots == result.passed_shots == 65
        assert result.measurements.shape == (65 if keep_records else 0, program.num_measurements)


def joint_frequencies(result, attempted, projections):
    rows = np.concatenate((result.measurements, result.detectors, result.observables), axis=1)
    parity = (rows @ projections) % 2
    return np.concatenate(([len(rows)], rows.sum(axis=0), parity.sum(axis=0))) / attempted


@pytest.mark.expensive
@pytest.mark.parametrize("distance", [5, 7])
def test_full_noisy_msc_joint_statistics_and_acceptance(distance):
    source = (
        Path(__file__).parents[1] / "fixtures" / f"reconstruction_msc_d{distance}.stim"
    ).read_text()
    num_detectors = sum(
        node.gate == clifft.GateType.DETECTOR for node in clifft.parse(source).nodes
    )
    rng = np.random.default_rng(917)
    for mask in (None, [int(i < 3) for i in range(num_detectors)], [1] * num_detectors):
        original = clifft.compile(source, postselection_mask=mask, normalize_syndromes=True)
        selected = clifft.compile(
            source, reconstruction=True, postselection_mask=mask, normalize_syndromes=True
        )
        assert selected.reconstruction_info()["selected"]
        assert selected.reconstruction_info(False)["selected"]
        shots = 8192 if mask == [1] * num_detectors else 4096
        reference = clifft.sample_survivors(
            original, shots, seed=34, keep_records=True, batch_size=1
        )
        actual = clifft.sample_survivors(selected, shots, seed=81, keep_records=True, batch_size=65)
        columns = original.num_measurements + original.num_detectors + original.num_observables
        projections = rng.integers(0, 2, size=(columns, 32), dtype=np.uint8)
        p = joint_frequencies(reference, shots, projections)
        q = joint_frequencies(actual, shots, projections)
        pooled = (p + q) / 2
        # Compare marginals and joint parities per attempted shot, so rejected
        # mass and accepted-output correlations both contribute to the check.
        tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 5 / shots
        assert np.all(np.abs(p - q) <= tolerance)
        counts = clifft.sample_survivors(
            selected, shots, seed=103, keep_records=False, batch_size=65
        )
        p = (
            np.r_[reference.passed_shots, reference.logical_errors, reference.observable_ones]
            / shots
        )
        q = np.r_[counts.passed_shots, counts.logical_errors, counts.observable_ones] / shots
        pooled = (p + q) / 2
        tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 5 / shots
        assert np.all(np.abs(p - q) <= tolerance)
