"""Tests for global reconstruction sampling plans and fallback behavior."""

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

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
    assert program.sampling_plan(keep_records=True).reconstruction_selected
    assert program.sampling_plan(keep_records=False).reconstruction_selected
    return program


def assert_same_result(a, b):
    for name in ("measurements", "detectors", "observables", "exp_vals", "observable_ones"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ("total_shots", "passed_shots", "logical_errors"):
        assert getattr(a, name) == getattr(b, name)


@pytest.mark.parametrize("keep_records", [False, True])
def test_plan_views_preserve_metadata_without_reconstruction(keep_records):
    programs = (
        clifft.compile(SOURCE, reconstruction=False),
        clifft.lower(clifft.trace(clifft.parse(SOURCE))),
    )
    for program in programs:
        plan = program.sampling_plan(keep_records=keep_records)
        original = program.original_plan
        assert isinstance(plan, clifft.PlanView)
        assert not plan.reconstruction_selected
        assert plan.fallback_reason
        assert not original.reconstruction_selected
        assert not original.fallback_reason
        assert plan.peak_active_width == original.peak_active_width == program.peak_active_width
        assert plan.num_actions == original.num_actions == program.num_actions
        assert plan.inspect() == original.inspect() == program.inspect()
        for i in range(program.num_actions):
            assert plan.inspect_action(i) == original.inspect_action(i) == program.inspect_action(i)
        assert repr(program) == (
            f"Program({program.num_actions} actions, "
            f"peak_active_width={program.peak_active_width}, "
            f"{program.num_measurements} measurements)"
        )


@pytest.mark.parametrize("policy", ["auto", True])
def test_reconstruction_budgets_apply_to_each_output(policy):
    source = "H 0 1 2\nT 0 1 2\nH 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-3]"
    bounded = clifft.compile(
        source, hir_passes=None, reconstruction=policy, reconstruction_max_schedule_operations=7
    )
    assert bounded.sampling_plan(keep_records=False).reconstruction_selected
    assert not bounded.sampling_plan(keep_records=True).reconstruction_selected
    default = clifft.compile(source, hir_passes=None, reconstruction=policy)
    assert default.sampling_plan(keep_records=True).reconstruction_selected

    smaller = clifft.compile(
        source, hir_passes=None, reconstruction=policy, reconstruction_max_dilation_qubits=1
    )
    for keep in (False, True):
        assert not smaller.sampling_plan(keep_records=keep).reconstruction_selected
        assert smaller.sampling_plan(keep_records=keep).fallback_reason


@pytest.mark.parametrize("value", [-1, 2**32])
def test_reconstruction_budgets_reject_out_of_range_values(value):
    with pytest.raises(ValueError):
        clifft.compile(NOISY, reconstruction_max_schedule_operations=value)
    with pytest.raises(ValueError):
        clifft.compile(NOISY, reconstruction_max_dilation_qubits=value)


def test_zero_reconstruction_budget_keeps_the_original():
    for limited in (
        clifft.compile(NOISY, reconstruction=True, reconstruction_max_schedule_operations=0),
        clifft.compile(NOISY, reconstruction=True, reconstruction_max_dilation_qubits=0),
    ):
        assert not limited.sampling_plan(keep_records=True).reconstruction_selected


def test_explicit_schedule_budget_enlarges_auto_search_on_a_narrow_circuit():
    # The repeated single-qubit rotations fill the automatic schedule budget.
    source = "H 0 1\nT 1\n" + "T 0\n" * 2049 + "H 0 1\nM 0 1\nDETECTOR rec[-1]"
    default = clifft.compile(source, hir_passes=None)
    relaxed = clifft.compile(source, hir_passes=None, reconstruction_max_schedule_operations=4096)
    for keep in (False, True):
        assert default.sampling_plan(keep_records=keep).peak_active_width == 2
        plan = relaxed.sampling_plan(keep_records=keep)
        assert plan.reconstruction_selected and plan.peak_active_width == 1


def test_sampling_plan_requires_explicit_output_mode():
    program = clifft.compile(SOURCE)
    with pytest.raises(TypeError):
        program.sampling_plan()
    with pytest.raises(TypeError):
        program.sampling_plan(False)


def test_plan_views_select_output_specific_executable():
    # With no detectors or observables, only retained-record sampling reconstructs.
    program = clifft.compile(UNITARY + "M 0 1", hir_passes=None, reconstruction=True)
    records = program.sampling_plan(keep_records=True)
    counts = program.sampling_plan(keep_records=False)
    original = program.original_plan
    assert records.reconstruction_selected
    assert not records.fallback_reason
    assert records.peak_active_width == program.peak_active_width == 1
    assert original.peak_active_width == 2
    assert records.num_actions == program.num_actions != original.num_actions
    assert records.inspect() == program.inspect() != original.inspect()
    for i in range(records.num_actions):
        assert records.inspect_action(i) == program.inspect_action(i)
    with pytest.raises(IndexError):
        program.inspect_action(program.num_actions)
    assert not counts.reconstruction_selected
    assert counts.fallback_reason
    assert counts.peak_active_width == original.peak_active_width
    assert counts.num_actions == original.num_actions
    assert counts.inspect() == original.inspect()
    with pytest.warns(DeprecationWarning):
        assert program.peak_rank == program.peak_active_width
    assert repr(program) == (
        f"Program({records.num_actions} actions, peak_active_width=1, 2 measurements, "
        "original_peak_active_width=2, survivor_peak_active_width=2)"
    )
    for keep_records in (True, False):
        result = clifft.sample_survivors(program, 65, seed=9, keep_records=keep_records)
        assert result.total_shots == result.passed_shots == 65
        assert result.measurements.shape == (65 if keep_records else 0, 2)


@pytest.mark.parametrize("mode", ["original", "records", "counts"])
def test_plan_view_keeps_owner_alive_and_is_read_only(mode):
    program = compile_selected()
    plan = (
        program.original_plan
        if mode == "original"
        else program.sampling_plan(keep_records=mode == "records")
    )
    expected = plan.inspect()
    expected_action = plan.inspect_action(plan.num_actions - 1)
    expected_width = plan.peak_active_width
    del program
    gc.collect()
    assert plan.inspect() == expected
    assert plan.inspect_action(plan.num_actions - 1) == expected_action
    assert plan.peak_active_width == expected_width
    with pytest.raises(IndexError):
        plan.inspect_action(plan.num_actions)
    for name in ("peak_active_width", "num_actions", "reconstruction_selected", "fallback_reason"):
        with pytest.raises(AttributeError):
            setattr(plan, name, getattr(plan, name))


def test_exact_queries_use_original_plan():
    original = clifft.compile(SOURCE, hir_passes=None, reconstruction=False)
    selected = compile_selected()
    assert isinstance(selected, clifft.Program)
    assert selected.original_plan.inspect() == original.inspect()
    assert selected.original_plan.num_actions == original.num_actions
    assert selected.original_plan.peak_active_width == original.peak_active_width == 2
    assert selected.peak_active_width == 1
    records = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    probabilities = np.abs(qiskit_statevector(stim_to_qiskit_noiseless(UNITARY))) ** 2
    measured = clifft.compile(UNITARY + "M 0 1", hir_passes=None, reconstruction=True)
    assert measured.sampling_plan(keep_records=True).reconstruction_selected
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
    assert not unitary.sampling_plan(keep_records=True).reconstruction_selected
    reference = qiskit_statevector(stim_to_qiskit_noiseless(UNITARY))
    qasm = clifft.compile(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q; t q; h q;',
        input_format="qasm2",
        reconstruction=True,
    )
    assert not qasm.sampling_plan(keep_records=True).reconstruction_selected
    assert_statevectors_equiv(clifft.get_statevector(qasm), reference)
    assert_statevectors_equiv(clifft.get_statevector(unitary), reference)
    np.testing.assert_allclose(clifft.basis_probabilities(unitary, records), np.abs(reference) ** 2)
    lowered = clifft.lower(clifft.trace(clifft.parse(UNITARY + "M 0 1")))
    assert isinstance(lowered, clifft.Program)
    np.testing.assert_allclose(clifft.record_probabilities(lowered, records), probabilities)


@pytest.mark.parametrize("sampling_mode", CPU_SAMPLING_MODES, indirect=True, ids=lambda m: m.name)
def test_selected_sampling_replays_and_switches_keep_records(sampling_mode):
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


def test_one_program_supports_concurrent_keep_records_modes():
    program = compile_selected(NOISY)

    def run(keep_records):
        return clifft.sample_survivors(
            program, 131, seed=8, keep_records=keep_records, batch_size=65
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [False, True] * 4))
    for index, result in enumerate(results):
        assert_same_result(result, results[index % 2])


@pytest.mark.parametrize("sampling_mode", CPU_SAMPLING_MODES, indirect=True, ids=lambda m: m.name)
def test_inverted_padding_preserves_feedback_and_postselection(sampling_mode):
    source = (
        UNITARY + "M 0 1\nMPAD !0 !1\nCX rec[-2] 2\nCX rec[-1] 3\nM 2 3\n"
        "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1]\n"
    )
    shots = 131
    for expected_detector in (0, 1):
        program = compile_selected(
            source, postselection_mask=[1], expected_detectors=[expected_detector]
        )
        for keep_records in (False, True):
            result = sampling_mode.sample_survivors(
                program, shots, seed=42, keep_records=keep_records
            )
            assert result.total_shots == shots
            assert result.passed_shots == expected_detector * shots
            assert result.logical_errors == result.passed_shots
            np.testing.assert_array_equal(result.observable_ones, [result.passed_shots])
            if keep_records:
                expected = np.tile([1, 0, 1, 0], (result.passed_shots, 1))
                np.testing.assert_array_equal(result.measurements[:, -4:], expected)
                assert not result.detectors.any()
                assert result.observables.all()


@pytest.mark.parametrize("batch_size", [1, 65])
def test_syndrome_rows_preserve_analytic_joint_distribution(batch_size):
    program = compile_selected()
    assert program.sampling_plan(keep_records="syndromes").reconstruction_selected
    shots = 20000
    result = clifft.sample(program, shots, seed=73, batch_size=batch_size, keep_records="syndromes")
    # Independent H T H rotations give P(m_i=1)=sin(pi/8)^2.
    # The joint output is D=m0 XOR m1 and O=m0, indexed by D + 2*O.
    p = np.sin(np.pi / 8) ** 2
    expected = np.array([(1 - p) ** 2, p * (1 - p), p**2, p * (1 - p)])
    joint = result.detectors[:, 0] + 2 * result.observables[:, 0]
    actual = np.bincount(joint, minlength=4) / shots
    tolerance = 6 * np.sqrt(expected * (1 - expected) / shots) + 1 / shots
    assert np.all(np.abs(actual - expected) < tolerance), (actual, expected)


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
    # Concurrent canceling rotations preserve the Clifford output law and leave
    # room for a scheduled width reduction before HIR optimization cancels them.
    clifford = "H 0\nCX 0 1\nDEPOLARIZE2(0.2) 0 1\nM 0 1\n"
    program = compile_selected("H 0 2\nT 0 2\nT_DAG 0 2\nH 0 2\n" + clifford + "DETECTOR rec[-1]\n")
    actual = clifft.sample(program, 20000, seed=12, batch_size=65).measurements
    expected = stim.Circuit(clifford).compile_sampler(seed=94).sample(20000)
    weights = np.array([1, 2])
    actual_law = np.bincount(actual @ weights, minlength=4) / len(actual)
    expected_law = np.bincount(expected @ weights, minlength=4) / len(expected)
    np.testing.assert_allclose(actual_law, expected_law, atol=0.025, rtol=0)


@pytest.mark.parametrize("postselection", [False, True])
def test_forced_faults_keep_original_sites_and_seeded_results(
    importance_sampling_mode, postselection
):
    mask = [1] if postselection else None
    selected = compile_selected(NOISY, postselection_mask=mask)
    original = clifft.compile(NOISY, hir_passes=None, postselection_mask=mask, reconstruction=False)
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
        ("R_X(0.13) 0\nM 0\n", "does not reduce"),
        (UNITARY + OUTPUTS + "READOUT_NOISE(0.1,0.2) rec[-1]\n", "readout"),
        ("H 0\nM 0\n", "already zero"),
        (SOURCE, "does not reduce"),
    ],
)
def test_unsupported_or_unprofitable_sampling_falls_back(source, reason):
    original = clifft.compile(source, reconstruction=False)
    fallback = clifft.compile(source, reconstruction=True)
    assert not fallback.sampling_plan(keep_records=True).reconstruction_selected
    assert reason.lower() in fallback.sampling_plan(keep_records=True).fallback_reason.lower()
    for keep_records in (False, True):
        plan = fallback.sampling_plan(keep_records=keep_records)
        assert not plan.reconstruction_selected
        assert plan.peak_active_width == original.peak_active_width
        assert plan.num_actions == original.num_actions
    assert_same_result(clifft.sample(original, 65, seed=9), clifft.sample(fallback, 65, seed=9))


def test_general_angle_and_three_qubit_noise_select_with_default_passes():
    # Entangled general-angle rotations with resets, feedback, 3-qubit noise, and
    # readout flips stay at width 2 under default passes + scheduling, while
    # global reconstruction drops peak active width to 1 on both output variants.
    unitary_prefix = (
        "H 0 1\n"
        "CX 0 2 1 3\n"
        "R_X(0.19) 0\n"
        "U3(0.31, -0.5, 0.13) 1\n"
        "R_PAULI(-0.27) !X2*Z0\n"
        "SPP !Y0*Z2\n"
        "TPP X3*!Z1\n"
        "R_ZZ(0.23) 1 3\n"
    )
    noisy_circuit = (
        "H 0 1\n"
        "CX 0 2 1 3\n"
        "DEPOLARIZE3(0.189) 0 1 2\n"
        "R_X(0.19) 0\n"
        "U3(0.31, -0.5, 0.13) 1\n"
        "R_PAULI(-0.27) !X2*Z0\n"
        "SPP !Y0*Z2\n"
        "TPP X3*!Z1\n"
        "R_ZZ(0.23) 1 3\n"
        "MR(0.08) 2\n"
        "CZ rec[-1] 0\n"
        "M 0 1 3\n"
        "DETECTOR rec[-4]\n"
        "DETECTOR rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-3]\n"
        "OBSERVABLE_INCLUDE(1) rec[-1]\n"
    )
    passes = clifft.default_hir_pass_manager()
    passes.add(clifft.ActiveWidthSchedulePass())
    original = clifft.compile(
        noisy_circuit,
        hir_passes=passes,
        postselection_mask=[1, 0],
        expected_detectors=[1, 0],
        reconstruction=False,
    )
    selected = clifft.compile(
        noisy_circuit,
        hir_passes=passes,
        reconstruction=True,
        postselection_mask=[1, 0],
        expected_detectors=[1, 0],
    )
    assert original.peak_active_width == 2
    for keep_records in (False, True):
        plan = selected.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected and not plan.fallback_reason
        assert plan.peak_active_width == 1

    shots = 8192
    ref = clifft.sample_survivors(original, shots, seed=51, keep_records=True, batch_size=1)
    got = clifft.sample_survivors(selected, shots, seed=79, keep_records=True, batch_size=65)
    columns = original.num_measurements + original.num_detectors + original.num_observables
    rng = np.random.default_rng(404)
    projections = rng.integers(0, 2, size=(columns, 16), dtype=np.uint8)
    p = joint_frequencies(ref, shots, projections)
    q = joint_frequencies(got, shots, projections)
    pooled = (p + q) / 2
    tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 5 / shots
    assert np.all(np.abs(p - q) <= tolerance)

    counts = clifft.sample_survivors(selected, shots, seed=93, keep_records=False, batch_size=65)
    p_cnt = np.r_[ref.passed_shots, ref.logical_errors, ref.observable_ones] / shots
    q_cnt = np.r_[counts.passed_shots, counts.logical_errors, counts.observable_ones] / shots
    pooled_cnt = (p_cnt + q_cnt) / 2
    tol_cnt = 7 * np.sqrt(2 * pooled_cnt * (1 - pooled_cnt) / shots) + 5 / shots
    assert np.all(np.abs(p_cnt - q_cnt) <= tol_cnt)

    # Exact queries and fixed-fault sampling still route to original_plan.
    np.testing.assert_array_equal(
        selected.noise_site_probabilities, original.noise_site_probabilities
    )
    assert_same_result(
        clifft.sample_k_survivors(selected, 65, 1, seed=17, keep_records=True),
        clifft.sample_k_survivors(original, 65, 1, seed=17, keep_records=True),
    )
    measured_unitary = clifft.compile(
        unitary_prefix + "M 0 1 2 3", hir_passes=None, reconstruction=True
    )
    assert measured_unitary.sampling_plan(keep_records=True).reconstruction_selected
    pure_unitary = clifft.compile(unitary_prefix, reconstruction=True)
    expected_probs = np.abs(clifft.get_statevector(pure_unitary)) ** 2
    records = [format(i, "04b")[::-1] for i in range(16)]
    np.testing.assert_allclose(
        clifft.record_probabilities(measured_unitary, records), expected_probs, atol=1e-12
    )
    np.testing.assert_allclose(
        clifft.basis_probabilities(pure_unitary, records), expected_probs, atol=1e-12
    )
    for record in ([0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]):
        assert _clifft_core._replay_record(measured_unitary, record) == _clifft_core._replay_record(
            clifft.compile(unitary_prefix + "M 0 1 2 3", hir_passes=None), record
        )


@pytest.mark.parametrize("pass_type", [clifft.RemoveNoisePass, clifft.DropNonUnitaryPass])
def test_intentional_hir_transformations_are_not_undone(pass_type):
    passes = clifft.HirPassManager()
    passes.add(pass_type())
    original = clifft.compile(NOISY, hir_passes=passes)
    fallback = clifft.compile(NOISY, hir_passes=passes, reconstruction=True)
    assert not fallback.sampling_plan(keep_records=True).reconstruction_selected
    assert "HIR passes" in fallback.sampling_plan(keep_records=True).fallback_reason
    assert_same_result(clifft.sample(original, 65, seed=9), clifft.sample(fallback, 65, seed=9))


def test_reconstruction_does_not_suppress_invalid_source_errors():
    for source in ("M 0\nDETECTOR rec[-2]", "X_ERROR(1.1) 0", "UNKNOWN 0"):
        with pytest.raises(clifft.ParseError):
            clifft.compile(source, reconstruction=True)


def test_python_owned_pass_preserves_reconstruction_selection():
    passes = clifft.HirPassManager()
    passes.add(clifft.PeepholeFusionPass())
    program = clifft.compile(NOISY, hir_passes=passes, reconstruction=True)
    expected = compile_selected(NOISY)
    for keep_records in (False, True):
        assert program.sampling_plan(keep_records=keep_records).reconstruction_selected
        assert_same_result(
            clifft.sample_survivors(program, 65, seed=9, keep_records=keep_records),
            clifft.sample_survivors(expected, 65, seed=9, keep_records=keep_records),
        )


@pytest.mark.expensive
@pytest.mark.parametrize(
    "fixture, width",
    [
        ("reconstruction_msc_d5.stim", 6),
        ("reconstruction_msc_d7.stim", 8),
        ("coherent_d5_r5.stim", 11),
    ],
)
@pytest.mark.parametrize("schedule", [False, True])
def test_full_noisy_fixtures_use_reconstructed_plans_with_default_passes(fixture, width, schedule):
    source = (Path(__file__).parents[1] / "fixtures" / fixture).read_text()
    passes = clifft.default_hir_pass_manager()
    if schedule:
        passes.add(clifft.ActiveWidthSchedulePass())
    program = clifft.compile(source, hir_passes=passes, reconstruction=True)
    assert program.peak_active_width == width
    assert program.original_plan.peak_active_width > width
    assert (
        program.sampling_plan(keep_records=False).num_actions
        != program.sampling_plan(keep_records=True).num_actions
    )
    for keep_records in (False, True):
        plan = program.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected and plan.peak_active_width == width
        result = clifft.sample_survivors(
            program, 65, seed=7, keep_records=keep_records, batch_size=65
        )
        assert result.total_shots == result.passed_shots == 65
        assert result.measurements.shape == (65 if keep_records else 0, program.num_measurements)


def test_postselected_small_width_reduction_requires_less_rotation_work():
    source = (Path(__file__).parents[1] / "fixtures" / "coherent_d5_r5.stim").read_text()
    mask = [1] * clifft.trace(clifft.parse(source)).num_detectors
    automatic = clifft.compile(source, postselection_mask=mask)
    explicit = clifft.compile(source, postselection_mask=mask, reconstruction=True)
    for keep in (True, "syndromes"):
        original = automatic.sampling_plan(keep_records=keep)
        narrower = explicit.sampling_plan(keep_records=keep)
        assert not original.reconstruction_selected
        assert narrower.reconstruction_selected
        assert narrower.peak_active_width == original.peak_active_width - 2


@pytest.mark.parametrize(
    "fault_qubits, fired_detectors, logical_flip",
    [((), [], 0), ((0,), [89], 1), ((0, 3, 9, 15, 23), [], 1)],
    ids=["fault-free", "detected-Z", "logical-Z"],
)
def test_d5_source_faults_before_logical_y(fault_qubits, fired_detectors, logical_flip):
    fixture = Path(__file__).parents[1] / "fixtures" / "reconstruction_msc_d5.stim"
    ideal = fixture.read_text().replace("(0.001)", "(0)")
    reference = clifft.compute_reference_syndrome(clifft.trace(clifft.parse(ideal)))
    lines = ideal.splitlines()
    if fault_qubits:
        # Force Z outcomes of the last depolarizing layer before the transversal T gates.
        site = max(i for i, line in enumerate(lines) if line.startswith("DEPOLARIZE1("))
        assert set(fault_qubits) <= set(map(int, lines[site].split()[1:]))
        lines[site] = "Z_ERROR(1) " + " ".join(map(str, fault_qubits))
    source = "\n".join(lines)

    shots = 32
    num_detectors = len(reference["detectors"])
    expected_detectors = np.zeros((shots, num_detectors), dtype=np.uint8)
    expected_detectors[:, fired_detectors] = 1
    for reconstruction in (False, True):
        options = dict(
            reconstruction=reconstruction,
            expected_detectors=reference["detectors"],
            expected_observables=reference["observables"],
        )
        unfiltered = clifft.compile(source, **options)
        postselected = clifft.compile(source, postselection_mask=[1] * num_detectors, **options)
        for mode in (True, "syndromes"):
            for program in (unfiltered, postselected):
                plan = program.sampling_plan(keep_records=mode)
                assert plan.reconstruction_selected == reconstruction
            result = clifft.sample(unfiltered, shots, seed=7, keep_records=mode)
            np.testing.assert_array_equal(result.detectors, expected_detectors)
            np.testing.assert_array_equal(result.observables, np.full((shots, 1), logical_flip))

            survivors = clifft.sample_survivors(postselected, shots, seed=7, keep_records=mode)
            assert survivors.total_shots == shots
            assert survivors.passed_shots == (0 if fired_detectors else shots)
            assert survivors.logical_errors == survivors.passed_shots * logical_flip


def joint_frequencies(result, attempted, projections):
    tables = (
        (result.measurements, result.detectors, result.observables)
        if result.measurements.shape[0] == result.detectors.shape[0]
        else (result.detectors, result.observables)
    )
    rows = np.concatenate(tables, axis=1)
    parity = (rows @ projections) % 2
    return np.concatenate(([len(rows)], rows.sum(axis=0), parity.sum(axis=0))) / attempted


@pytest.mark.expensive
@pytest.mark.parametrize("distance", [5, 7])
@pytest.mark.parametrize("postselection", ["none", "partial", "all"])
def test_full_noisy_msc_joint_statistics_and_acceptance(distance, postselection):
    source = (
        Path(__file__).parents[1] / "fixtures" / f"reconstruction_msc_d{distance}.stim"
    ).read_text()
    num_detectors = sum(
        node.gate == clifft.GateType.DETECTOR for node in clifft.parse(source).nodes
    )
    if postselection == "none":
        mask = None
    elif postselection == "partial":
        mask = [int(i < 3) for i in range(num_detectors)]
    else:
        mask = [1] * num_detectors
    original = clifft.compile(
        source, reconstruction=False, postselection_mask=mask, normalize_syndromes=True
    )
    for keep in (True, "syndromes"):
        assert not original.sampling_plan(keep_records=keep).reconstruction_selected
    selected = clifft.compile(
        source, reconstruction=True, postselection_mask=mask, normalize_syndromes=True
    )
    assert selected.sampling_plan(keep_records=True).reconstruction_selected
    assert selected.sampling_plan(keep_records=False).reconstruction_selected
    shots = 8192 if postselection == "all" else 4096
    reference = clifft.sample_survivors(original, shots, seed=34, keep_records=True, batch_size=1)
    actual = clifft.sample_survivors(
        selected,
        shots,
        seed=81,
        keep_records=True,
        batch_size="tune",
        tuning_budget_seconds=0,
        sharing=False,
    )
    assert actual.batch_tuning.baseline_batch_size == (512 if distance == 5 else 128)
    columns = original.num_measurements + original.num_detectors + original.num_observables
    rng = np.random.default_rng(917)
    projections = rng.integers(0, 2, size=(columns, 32), dtype=np.uint8)
    p = joint_frequencies(reference, shots, projections)
    q = joint_frequencies(actual, shots, projections)
    pooled = (p + q) / 2
    # Compare marginals and joint parities per attempted shot, so rejected
    # mass and accepted-output correlations both contribute to the check.
    tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 5 / shots
    assert np.all(np.abs(p - q) <= tolerance)
    counts = clifft.sample_survivors(selected, shots, seed=103, keep_records=False)
    p = np.r_[reference.passed_shots, reference.logical_errors, reference.observable_ones] / shots
    q = np.r_[counts.passed_shots, counts.logical_errors, counts.observable_ones] / shots
    pooled = (p + q) / 2
    tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 5 / shots
    assert np.all(np.abs(p - q) <= tolerance)


def test_pure_clifford_memory_suffix():
    # 1200 syndrome rounds (2400 resets + measurements) after non-Clifford preparation
    # must stay within default prefix dilation and schedule budgets.
    rounds = 1200
    lines = ["H 0 1", "T 0 1", "CX 0 1", "H 0 1"]
    for r in range(rounds):
        lines.append("R 2\nCX 0 2\nCX 1 2\nX_ERROR(0.002) 2\nM 2")
        if r > 0:
            lines.append("DETECTOR rec[-2] rec[-1]")
    lines.append("M 0 1")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]")
    source = "\n".join(lines) + "\n"

    original = clifft.compile(source, hir_passes=None, reconstruction=False)
    selected = clifft.compile(source, hir_passes=None, reconstruction=True)
    assert original.peak_active_width == 2
    for keep_records in (False, True):
        plan = selected.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected and not plan.fallback_reason
        assert plan.peak_active_width == 1

    shots = 2048
    rng = np.random.default_rng(712)
    modes: tuple[tuple[bool | Literal["syndromes"], int, int], ...] = (
        (True, 19, 43),
        ("syndromes", 59, 83),
    )
    for record_mode, seed_a, seed_b in modes:
        ref = clifft.sample(original, shots, seed=seed_a, keep_records=record_mode)
        got = clifft.sample(selected, shots, seed=seed_b, keep_records=record_mode)
        columns = (
            (original.num_measurements if record_mode is True else 0)
            + original.num_detectors
            + original.num_observables
        )
        projections = rng.integers(0, 2, size=(columns, 32), dtype=np.uint8)
        p = joint_frequencies(ref, shots, projections)
        q = joint_frequencies(got, shots, projections)
        pooled = (p + q) / 2
        tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 5 / shots
        assert np.all(np.abs(p - q) <= tolerance)


def test_wide_output_bailout():
    # Wide circuits exceeding max_outputs bail out cleanly to original_plan.
    wide = "H 0 1\nT 0 1\nH 0 1\n" + ("R 2\nM 2\n" * 4100) + "M 0 1\nDETECTOR rec[-1]\n"
    fallback = clifft.compile(wide, hir_passes=None, reconstruction=True)
    for keep_records in (False, True):
        plan = fallback.sampling_plan(keep_records=keep_records)
        assert not plan.reconstruction_selected
        assert "output limit exceeded" in plan.fallback_reason
        assert plan.peak_active_width == fallback.original_plan.peak_active_width


def _wide_reconstructible_source(include_syndromes: bool = True, parity_qubit: int = 64) -> str:
    ancillas = " ".join(str(q) for q in range(2, parity_qubit))
    parity_cx = " ".join(f"{q} {parity_qubit}" for q in range(2, parity_qubit))
    lines = [
        f"H 0 1 {ancillas}",
        f"T 0 1 {ancillas}",
        "DEPOLARIZE2(0.2) 0 1",
        "H 0 1",
        f"CX {parity_cx}",
        f"M 0 1 {parity_qubit}",
    ]
    if include_syndromes:
        lines.append("DETECTOR rec[-3] rec[-2]")
        lines.append("OBSERVABLE_INCLUDE(0) rec[-3]")
    return "\n".join(lines) + "\n"


def _wide_detector_only_source() -> str:
    qs = list(range(2, 162))
    qstr = " ".join(map(str, qs))
    czs = " ".join(
        f"{qs[i]} {qs[(i + s) % len(qs)]}" for s in (1, 17, 43, 71) for i in range(len(qs))
    )
    return (
        f"H 0 1 {qstr}\n"
        f"T 0 1 {qstr}\n"
        "DEPOLARIZE2(0.2) 0 1\n"
        "H 0 1\n"
        f"CZ {czs}\n"
        f"MX {qstr}\n"
        "M 0 1\n"
        "DETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\n"
    )


def test_wide_original_compiles_with_reconstruction_and_guards_unavailable_original():
    source = _wide_reconstructible_source(include_syndromes=True)
    with pytest.raises(OverflowError, match="dense-state limit"):
        clifft.compile(source, reconstruction=False)

    program = clifft.compile(source, reconstruction=True)
    assert program.num_qubits == 65
    assert program.num_measurements == 3
    assert program.num_detectors == 1
    assert program.num_observables == 1
    assert program.peak_active_width == 1
    for keep_records in (False, True, "all", "syndromes", "measurements"):
        plan = program.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected
        assert not plan.fallback_reason
        assert plan.peak_active_width == 1
    assert "original_plan=unavailable" in repr(program)

    # Sampling succeeds across output modes and matches the 2-qubit core plus a fair Z64 coin.
    reference = clifft.compile(
        UNITARY
        + "DEPOLARIZE2(0.2) 0 1\nH 2\nM 0 1 2\n"
        + "DETECTOR rec[-3] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
    )
    shots = 4096
    ref = clifft.sample(reference, shots, seed=31)
    got = clifft.sample(program, shots, seed=53)
    projections = np.random.default_rng(42).integers(0, 2, size=(5, 16), dtype=np.uint8)
    p = joint_frequencies(ref, shots, projections)
    q = joint_frequencies(got, shots, projections)
    pooled = (p + q) / 2
    tolerance = 6 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 5 / shots
    assert np.all(np.abs(p - q) <= tolerance)

    det, obs = clifft.sample_detectors(program, 257, seed=11, separate_observables=True)
    assert det.shape == (257, 1) and obs.shape == (257, 1)
    surv = clifft.sample_survivors(program, 257, seed=11, keep_records=False)
    assert surv.total_shots == surv.passed_shots == 257

    # Exact-query, replay, and fixed-fault APIs require original_plan and raise OverflowError.
    expected_orig_err = "original plan is unavailable"
    with pytest.raises(OverflowError, match=expected_orig_err):
        _ = program.original_plan
    with pytest.raises(OverflowError, match=expected_orig_err):
        _ = program.noise_site_probabilities
    with pytest.raises(OverflowError, match=expected_orig_err):
        clifft.get_statevector(program)
    with pytest.raises(OverflowError, match=expected_orig_err):
        clifft.basis_probabilities(program, ["0" * 65])
    with pytest.raises(OverflowError, match=expected_orig_err):
        clifft.record_probabilities(program, ["000"])
    with pytest.raises(OverflowError, match=expected_orig_err):
        _clifft_core._replay_record(program, [0, 0, 0])
    with pytest.raises(OverflowError, match=expected_orig_err):
        clifft.sample_k(program, 16, 1, seed=7)
    with pytest.raises(OverflowError, match=expected_orig_err):
        clifft.sample_k_survivors(program, 16, 1, seed=7)


def test_wide_original_asymmetric_variants_and_total_failure():
    # Unreferenced expander graph-state measurements keep retained-record width >= 60
    # while detector-only reconstruction elides the unmeasured T rotations to width 1.
    detector_only_source = _wide_detector_only_source()
    with pytest.raises(OverflowError, match="dense-state limit"):
        clifft.compile(detector_only_source, reconstruction=False)
    detector_only = clifft.compile(detector_only_source, reconstruction=True)
    assert detector_only.sampling_plan(keep_records=False).reconstruction_selected
    assert detector_only.sampling_plan(keep_records="syndromes").reconstruction_selected
    assert "records_plan=unavailable, original_plan=unavailable" in repr(detector_only)

    det, obs = clifft.sample_detectors(detector_only, 129, seed=19, separate_observables=True)
    assert det.shape == (129, 1) and obs.shape == (129, 1)
    syn = clifft.sample(detector_only, 129, seed=19, keep_records="syndromes")
    assert syn.measurements.shape == (0, detector_only.num_measurements)
    assert syn.detectors.shape == (129, 1)
    surv_counts = clifft.sample_survivors(detector_only, 129, seed=19, keep_records=False)
    assert surv_counts.passed_shots == 129

    expected_err = "dense active-width limit"
    for bad_records in (True, "all", "measurements"):
        with pytest.raises(OverflowError, match=expected_err):
            detector_only.sampling_plan(keep_records=bad_records)
        with pytest.raises(OverflowError, match=expected_err):
            clifft.sample(detector_only, 16, seed=3, keep_records=bad_records)
    with pytest.raises(OverflowError, match=expected_err):
        clifft.sample_survivors(detector_only, 16, seed=3, keep_records=True)
    for attr in ("peak_active_width", "num_actions"):
        with pytest.raises(OverflowError, match=expected_err):
            getattr(detector_only, attr)
    with pytest.raises(OverflowError, match=expected_err):
        detector_only.inspect()
    with pytest.raises(OverflowError, match=expected_err):
        detector_only.inspect_action(0)

    # Zero detectors and observables make only retained-record sampling available.
    record_only_source = _wide_reconstructible_source(include_syndromes=False)
    record_only = clifft.compile(record_only_source, reconstruction=True)
    assert record_only.sampling_plan(keep_records=True).reconstruction_selected
    assert "original_plan=unavailable, survivor_plan=unavailable" in repr(record_only)
    rec_sample = clifft.sample(record_only, 65, seed=23)
    assert rec_sample.measurements.shape == (65, 3)
    with pytest.raises(OverflowError, match="no requested sampling outputs"):
        record_only.sampling_plan(keep_records=False)
    with pytest.raises(OverflowError, match="no requested sampling outputs"):
        clifft.sample_survivors(record_only, 16, seed=23, keep_records=False)

    # When neither variant reconstructs below width 60, compile raises OverflowError.
    unsupported = _wide_reconstructible_source(include_syndromes=True) + "EXP_VAL Z0\n"
    with pytest.raises(OverflowError, match="EXP_VAL"):
        clifft.compile(unsupported, reconstruction=True)


def test_wide_original_post_lowering_reduction_and_syndrome_normalization():
    # Scheduled reconstruction width exceeds 60 (116) while post-lowering optimization
    # fuses T q and TPP Zq*Z256 on |0> into Clifford S gates, collapsing width to 0.
    data_qs = list(range(256))
    qstr = " ".join(map(str, data_qs))
    tpps = " ".join(f"Z{q}*Z256" for q in data_qs)
    czs = " ".join(f"{q} {(q + s) % 256}" for s in (1, 17, 43, 71) for q in data_qs)
    post_lowering_source = f"H {qstr}\nT {qstr}\nTPP {tpps}\nCZ {czs}\nMX {qstr}\n"
    with pytest.raises(OverflowError, match="dense-state limit"):
        clifft.compile(post_lowering_source, reconstruction=False)
    post_lowered = clifft.compile(post_lowering_source, reconstruction=True)
    assert post_lowered.sampling_plan(keep_records=True).reconstruction_selected
    assert post_lowered.peak_active_width == 0
    assert clifft.sample(post_lowered, 16, seed=5).measurements.shape == (16, 256)

    # normalize_syndromes=True falls back to a reconstructed noiseless detector plan
    # when sequential reference planning exceeds width 60, with or without postselection.
    norm_both = clifft.compile(
        _wide_reconstructible_source(include_syndromes=True),
        normalize_syndromes=True,
        reconstruction=True,
    )
    assert norm_both.peak_active_width == 1

    norm_record_only = clifft.compile(
        _wide_reconstructible_source(include_syndromes=False),
        normalize_syndromes=True,
        reconstruction=True,
    )
    assert norm_record_only.peak_active_width == 1

    det_ones_source = (
        _wide_reconstructible_source(include_syndromes=False)
        + "X 65\nM 65\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    with pytest.raises(OverflowError, match="dense-state limit"):
        clifft.compile(
            det_ones_source, postselection_mask=[1], normalize_syndromes=True, reconstruction=False
        )
    norm_ps = clifft.compile(
        det_ones_source, postselection_mask=[1], normalize_syndromes=True, reconstruction=True
    )
    assert norm_ps.sampling_plan(keep_records=False).peak_active_width == 0
    surv = clifft.sample_survivors(norm_ps, 32, seed=13, keep_records=True)
    assert surv.total_shots == surv.passed_shots == 32
    assert surv.logical_errors == 0
    assert not surv.detectors.any()
    assert not surv.observables.any()


def test_normalization_reconstructs_wide_reference_below_planner_limit():
    source = _wide_reconstructible_source(include_syndromes=False, parity_qubit=60) + (
        "X 61\nM 61\nDETECTOR rec[-1]\nREADOUT_NOISE(1) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    # The noiseless sequential plan fits the planner but its dense state exceeds an EiB.
    clean = source.replace("DEPOLARIZE2(0.2) 0 1\n", "").replace("READOUT_NOISE(1) rec[-1]\n", "")
    assert 56 <= clifft.compile(clean, reconstruction=False).peak_active_width < 60

    program = clifft.compile(
        source, postselection_mask=[1], normalize_syndromes=True, reconstruction=True
    )
    assert program.sampling_plan(keep_records=False).peak_active_width == 0
    result = clifft.sample_survivors(program, 32, seed=13, keep_records="syndromes")
    # Both noiseless parities are one. Only the observable sees the actual readout flip.
    assert result.passed_shots == 32
    assert not result.detectors.any()
    assert result.observables.all()
