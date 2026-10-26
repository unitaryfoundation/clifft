"""Public selection, ownership, and output contracts for noncomputational programs."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import utils_noncomp_enumerator as dense

import clifft
from clifft import _clifft_core, noncomp

CLASSIFIER = [[1, 0, 0.3, 0.3, 0.6], [0, 1, 0.7, 0.7, 0.4]]
DEPOLARIZE = noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25))


def model(**kwargs):
    return noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER), **kwargs)


@pytest.fixture(params=["convenience", "prepared"])
def public_sample(request):
    def sample(circuit, noise, **kwargs):
        if request.param == "convenience":
            return noncomp.sample(circuit, noise, **kwargs)
        options = {
            key: kwargs.pop(key)
            for key in ("execution", "final_status", "max_active_width")
            if key in kwargs
        }
        return noncomp.compile(circuit, noise, **options).sample(**kwargs)

    return sample


def same_rows(left, right):
    for name in ("measurements", "detectors", "observables", "heralds", "final_status"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


@pytest.mark.parametrize("execution", ["auto", "fixed", "trajectory"])
def test_public_convenience_preserves_positional_arguments(execution):
    text = "H 0\nT 0\nH 0\nM 0"
    noise = model()
    positional = noncomp.sample(
        text,
        noise,
        129,
        31,
        1,
        2,
        execution=execution,
        final_status=True,
        batch_size=1,
        collect_stats=True,
    )
    keyword = noncomp.sample(
        text,
        noise,
        shots=129,
        seed=31,
        max_active_width=1,
        threads=2,
        execution=execution,
        final_status=True,
        batch_size=1,
        collect_stats=True,
    )
    same_rows(positional, keyword)
    assert positional.shots == 129
    assert positional.stats is not None
    assert keyword.stats is not None
    assert positional.stats.workers == keyword.stats.workers == 2
    assert positional.stats.peak_active_width == keyword.stats.peak_active_width == 1
    with pytest.raises(ValueError, match="active_width"):
        noncomp.sample(text, noise, 129, 31, 0, 2, execution=execution)


def test_shared_api_forwards_positional_width_to_preparation(noncomp_sampling_api):
    text = "H 0\nT 0\nH 0\nM 0"
    noise = model()
    positional = noncomp_sampling_api(text, noise, 65, 31, 1)
    keyword = noncomp_sampling_api(text, noise, shots=65, seed=31, max_active_width=1)
    same_rows(positional, keyword)
    with pytest.raises(ValueError, match="active_width"):
        noncomp_sampling_api(text, noise, 65, 31, 0)


@pytest.mark.parametrize("batch_size", [1, 65, "auto"])
def test_public_default_selects_fixed_and_preserves_result_shape(public_sample, batch_size):
    text = "LEAKAGE(0.4) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    result = public_sample(
        text, model(), shots=131, seed=24, batch_size=batch_size, collect_stats=True
    )
    assert result.final_status is None
    assert result.shots == 131
    assert (
        result.num_qubits
        == result.num_measurements
        == result.num_detectors
        == result.num_observables
        == 1
    )
    measurements, detectors, observables = result
    np.testing.assert_array_equal(measurements, detectors)
    np.testing.assert_array_equal(measurements, observables)
    np.testing.assert_array_equal(result.symbols(), measurements)
    stats = result.stats
    assert stats.strategy == "fixed"
    assert stats.fallback_reason is None
    assert stats.preparation_compilations == 1
    assert stats.initial_compilations == stats.continuation_compilations == 0
    assert stats.fixed_shots == 131
    assert stats.trajectory_shots == 0
    assert stats.scalar_shots == (131 if batch_size == 1 else 0)
    assert stats.packed_shots == (0 if batch_size == 1 else 131)
    assert stats.packed_batches == (0 if batch_size == 1 else 3 if batch_size == 65 else 1)
    assert stats.batch_size == (131 if batch_size == "auto" else batch_size)
    assert stats.sampling_seconds >= 0
    with pytest.raises(FrozenInstanceError):
        stats.fixed_shots = 0


@pytest.mark.parametrize("batch_size", [1, 65])
@pytest.mark.parametrize("execution", ["auto", "fixed"])
@pytest.mark.parametrize(
    "text",
    [
        "H 0\nCX 0 1\nLEAKAGE(0.37) 0\nM 0 0 1",
        "H 1\nT 1\nCX 1 2\nLEAKAGE(0.31) 0\nCX 0 1\nH 1 2\nM 0 1 2",
        "H 0\nCX 0 1\nLOSS(0.41) 1\nCX 0 1\nT 0\nH 0\nM 0 1",
    ],
)
def test_public_fixed_joint_distribution_matches_dense_model(
    public_sample, batch_size, execution, text
):
    noise = model(gate_partner_effects={"leaked": DEPOLARIZE, "lost": DEPOLARIZE})
    expected = dense.enumerate_exact(
        text,
        initial=[1, 0, 0, 0, 0],
        transitions={},
        classifier=CLASSIFIER,
        depolarize_partner=True,
    )
    assert expected.dropped_mass < 1e-10
    result = public_sample(
        text,
        noise,
        shots=16000,
        seed=43,
        batch_size=batch_size,
        execution=execution,
        collect_stats=True,
    )
    assert result.stats.fixed_shots == 16000
    assert result.stats.packed_shots == (16000 if batch_size == 65 else 0)
    assert (
        dense.tvd(expected.record_probs, dense.empirical_record_probs(result.measurements)) < 0.025
    )


def test_requested_leaked_status_preserves_bell_pair_correlation(public_sample):
    text = "H 0\nCX 0 1\nLEAKAGE(1) 0\nM 1"
    noise = model()
    omitted = public_sample(text, noise, shots=129, seed=7, collect_stats=True)
    assert omitted.stats.strategy == "fixed"
    assert omitted.final_status is None
    detailed = public_sample(text, noise, shots=129, seed=7, final_status=True, collect_stats=True)
    assert detailed.stats.strategy == "trajectory"
    assert detailed.stats.fallback_reason == "detailed_leaked_status"
    assert detailed.stats.preparation_compilations == 0
    assert detailed.stats.initial_compilations == 1
    assert detailed.stats.continuation_compilations == 129
    assert detailed.stats.trajectory_shots == detailed.stats.scalar_shots == 129
    np.testing.assert_array_equal(detailed.final_status[:, 0], detailed.measurements[:, 0] + 1)
    assert set(detailed.final_status[:, 0]) == {
        noncomp.QubitStatus.LEAK_G,
        noncomp.QubitStatus.LEAK_E,
    }
    with pytest.raises(ValueError, match="detailed_leaked_status"):
        noncomp.compile(text, noise, final_status=True, execution="fixed")
    restored = public_sample(
        text + "\nR 0", noise, shots=129, final_status=True, collect_stats=True
    )
    assert restored.stats.strategy == "fixed"
    assert not restored.final_status.any()


@pytest.mark.parametrize("execution", ["auto", "fixed", "trajectory"])
@pytest.mark.parametrize("final_status", [False, True])
def test_public_zero_shots_keep_contract_without_execution(public_sample, execution, final_status):
    result = public_sample(
        "M 0\nDETECTOR rec[-1]",
        model(),
        shots=0,
        execution=execution,
        final_status=final_status,
        collect_stats=True,
    )
    assert result.measurements.shape == result.heralds.shape == result.detectors.shape == (0, 1)
    assert result.observables.shape == (0, 0)
    if final_status:
        assert result.final_status.shape == (0, 1)
    else:
        assert result.final_status is None
    assert result.stats.fixed_shots == result.stats.trajectory_shots == 0
    assert result.stats.workers == result.stats.packed_batches == 0
    assert result.stats.initial_compilations == result.stats.continuation_compilations == 0


@pytest.mark.parametrize("shots", [0, 1, 65])
@pytest.mark.parametrize("batch_size", [0, -1, 2**32, "tune", "bad", 1.5])
def test_public_rejects_invalid_batch_options(public_sample, shots, batch_size):
    with pytest.raises((TypeError, ValueError), match="batch_size|incompatible"):
        public_sample("M 0", model(), shots=shots, batch_size=batch_size)


@pytest.mark.parametrize("execution", ["trajectory", "auto"])
@pytest.mark.parametrize("shots", [0, 1, 65])
def test_public_rejects_explicit_packing_for_trajectory_selection(public_sample, execution, shots):
    with pytest.raises(ValueError, match="selected strategy is trajectory"):
        public_sample(
            "LOSS(1) 0\nCX 0 1\nM 1", model(), shots=shots, execution=execution, batch_size=65
        )


@pytest.mark.parametrize("execution", ["auto", "fixed", "trajectory"])
def test_validation_errors_are_not_optimization_fallbacks(public_sample, execution):
    with pytest.raises(ValueError, match="classifier is required"):
        public_sample("LEAKAGE(0.2) 0\nM 0", noncomp.Model(), shots=0, execution=execution)
    with pytest.raises(ValueError, match="EXP_VAL"):
        noncomp.compile("EXP_VAL Z0", model(), execution=execution)


@pytest.mark.parametrize("batch_size", [1, 65])
def test_public_fixed_fusion_preserves_status_conditioned_certain_noise(public_sample, batch_size):
    text = (
        "H 1\nT 1\nT 1\nS_DAG 1\nH 1\nLOSS(0.4) 0\n"
        "PAULI_CHANNEL_2(0,0,0,0,1,0,0,0,0,0,0,0,0,0,0) 0 1\n"
        "MR 1\nHERALD_LOSS_EVENT 0\nCX rec[-2] 2\nM 1 2\n"
        "DETECTOR rec[-1] rec[-4]\nOBSERVABLE_INCLUDE(0) rec[-1]\nM 0"
    )
    noise = noncomp.Model(
        classifier=noncomp.Classifier([[1, 0, 0.3, 0.3, 0], [0, 1, 0.7, 0.7, 0], [0, 0, 0, 0, 1]])
    )
    result = public_sample(
        text,
        noise,
        shots=257,
        seed=73,
        max_active_width=0,
        final_status=True,
        batch_size=batch_size,
        collect_stats=True,
    )
    assert result.stats is not None
    assert result.stats.strategy == "fixed"
    assert result.stats.peak_active_width == 0
    assert result.stats.fixed_shots == 257
    assert result.stats.packed_shots == (257 if batch_size > 1 else 0)
    lost = result.measurements[:, 1]
    assert lost.any() and not lost.all()
    np.testing.assert_array_equal(result.measurements[:, 0], 1 - lost)
    np.testing.assert_array_equal(result.measurements[:, 3], 1 - lost)
    assert not result.measurements[:, 2].any()
    assert not result.detectors.any()
    np.testing.assert_array_equal(result.observables[:, 0], 1 - lost)
    np.testing.assert_array_equal(result.heralds[:, 4], lost)
    np.testing.assert_array_equal(result.final_status[:, 0] == noncomp.QubitStatus.LOST, lost)


def test_public_width_fallback_counts_discarded_preparation_and_keeps_lazy_trajectory():
    text = "H 0\nT 0\nH 0\nM 0"
    program = noncomp.compile(text, model(), max_active_width=0)
    explanation = program.explain()
    assert program.strategy == "trajectory"
    assert "active_width" in explanation
    assert "candidate width 1" in explanation
    assert "Active-width cap: 0" in explanation
    empty = program.sample(0, collect_stats=True)
    assert empty.stats is not None
    assert empty.stats.preparation_compilations == 1
    assert empty.stats.initial_compilations == 0
    assert empty.stats.peak_active_width is None
    with pytest.raises(ValueError, match="exceeds max_active_width"):
        program.sample(65, seed=9, collect_stats=True)
    assert program.explain() == explanation
    with pytest.raises(ValueError, match="active_width"):
        noncomp.compile(text, model(), execution="fixed", max_active_width=0)
    lazy = noncomp.compile(
        "H 0\nT 0\nH 0\nM 0", model(), execution="trajectory", max_active_width=0
    )
    assert lazy.sample(0).measurements.shape == (0, 1)
    with pytest.raises(ValueError, match="exceeds max_active_width"):
        lazy.sample(1)


@pytest.mark.parametrize("execution", ["auto", "trajectory"])
def test_public_prepared_program_is_owned_reusable_and_concurrent(execution):
    matrix = [row[:] for row in CLASSIFIER]
    noise = noncomp.Model(classifier=noncomp.Classifier(matrix))
    circuit = clifft.parse("LOSS(0.4) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    program = noncomp.compile(circuit, noise, execution=execution, final_status=True)
    expected = program.sample(1025, seed=81, threads=1, collect_stats=True)
    assert expected.stats is not None
    direct = noncomp.sample(
        circuit, noise, shots=1025, seed=81, execution=execution, final_status=True
    )
    same_rows(expected, direct)
    matrix[0][4], matrix[1][4] = 1, 0
    noncomp.Model.__init__(
        noise, initial_state=[0, 0, 0, 0, 1], classifier=noncomp.Classifier(matrix)
    )
    del circuit, noise, matrix
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(program.sample, 1025, 81, workers) for workers in (2, 3)]
        for future in futures:
            actual = future.result()
            same_rows(actual, expected)
            assert actual.stats is None
    again = program.sample(1025, 81, collect_stats=True)
    assert again.stats is not None
    same_rows(again, expected)
    assert again.stats.preparation_compilations == expected.stats.preparation_compilations
    assert again.stats.initial_compilations == expected.stats.initial_compilations
    assert again.stats.continuation_compilations == expected.stats.continuation_compilations
    assert again.stats.preparation_seconds == expected.stats.preparation_seconds


def test_public_trajectory_override_replays_legacy_records_and_status():
    text = "H 0\nCX 0 1\nLEAKAGE(0.4) 0\nM 0 1\nDETECTOR rec[-1] rec[-2]"
    noise = model()
    old = _clifft_core._sample_noncomputational(clifft.parse(text), noise._handle, 257, 15, None, 2)
    new = noncomp.sample(
        text,
        noise,
        shots=257,
        seed=15,
        threads=2,
        execution="trajectory",
        final_status=True,
        collect_stats=True,
    )
    for index, name in enumerate(
        ("measurements", "detectors", "observables", "final_status", "heralds")
    ):
        np.testing.assert_array_equal(old[index], getattr(new, name))
    assert new.stats is not None
    assert new.stats.strategy == "trajectory"
    assert new.stats.fallback_reason is None


@pytest.mark.parametrize(
    "reason",
    [
        "initial_state",
        "named_transition",
        "spreading",
        "level_sensitive_readout",
        "partner_channel",
        "unsupported_operation",
    ],
)
def test_public_explanation_reports_existing_rejection_rules(reason):
    noise = model()
    text = "LEAKAGE(1) 0\nM 0"
    if reason == "initial_state":
        noise = model(initial_state=[0.5, 0.5, 0, 0, 0])
    elif reason == "named_transition":
        transition = np.zeros((5, 5))
        transition[0, 0] = 0.1
        noise = model(transitions={"project": transition})
        text = "H 0\nLEVEL_TRANSITION[project] 0"
    elif reason == "spreading":
        text = "LEAKAGE(1) 0\nLEAKAGE_INTERACTION(0.25,0.25,0.25,0.1) 0 1"
    elif reason == "level_sensitive_readout":
        noise = noncomp.Model(classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]))
    elif reason == "partner_channel":
        text = "LOSS(1) 0\nCX 0 1\nM 1"
    elif reason == "unsupported_operation":
        text = "MPP Z0*Z1"
    program = noncomp.compile(text, noise)
    assert program.strategy == "trajectory"
    assert reason in program.explain()
    if reason in ("named_transition", "spreading", "level_sensitive_readout", "partner_channel"):
        assert "circuit line 2" in program.explain()
    result = program.sample(0, collect_stats=True)
    assert result.stats is not None
    assert result.stats.fallback_reason == reason
    with pytest.raises(ValueError, match=reason):
        noncomp.compile(text, noise, execution="fixed")


@pytest.mark.parametrize("execution", ["bad", "infer", "packed"])
def test_public_rejects_unknown_execution_options(execution):
    with pytest.raises(ValueError, match="execution must be"):
        noncomp.compile("M 0", model(), execution=execution)


@pytest.mark.parametrize("shots", [0, 65])
@pytest.mark.parametrize("final_status", [False, True])
def test_public_empty_circuit_preserves_zero_width_status(public_sample, shots, final_status):
    result = public_sample("", model(), shots=shots, final_status=final_status, collect_stats=True)
    assert result.measurements.shape == result.heralds.shape == (shots, 0)
    assert result.detectors.shape == result.observables.shape == (shots, 0)
    if final_status:
        assert result.final_status.shape == (shots, 0)
    else:
        assert result.final_status is None
    assert result.stats.fixed_shots == shots
