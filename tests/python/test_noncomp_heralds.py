"""Nondestructive leakage and loss probes in the ordinary measurement record."""

import numpy as np
import pytest
import stim
from conftest import binomial_tolerance, cross_binomial_tolerance, noncomp_transition_matrix

import clifft
from clifft import noncomp

HERALDS = ["HERALD_LEAKAGE_EVENT", "HERALD_LOSS_EVENT"]
CLASSIFIER = noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]])


@pytest.mark.parametrize("level", list(noncomp.Level))
def test_status_heralds_distinguish_levels_without_classifier(level, noncomp_sampling_api):
    initial = [0.0] * 5
    initial[level] = 1.0
    result = noncomp_sampling_api(
        """
        REPEAT 2 {
            HERALD_LEAKAGE_EVENT 0 1
            HERALD_LOSS_EVENT 1 0
        }
        """,
        noncomp.Model(initial_state=initial),
        shots=257,
        seed=71,
    )
    leak = level in (noncomp.Level.LEAK_G, noncomp.Level.LEAK_E)
    lost = level == noncomp.Level.LOST
    expected = [leak, leak, lost, lost] * 2
    assert (result.measurements == expected).all()
    expected_status = {
        noncomp.Level.G: noncomp.QubitStatus.COMPUTATIONAL,
        noncomp.Level.E: noncomp.QubitStatus.COMPUTATIONAL,
        noncomp.Level.LEAK_G: noncomp.QubitStatus.LEAK_G,
        noncomp.Level.LEAK_E: noncomp.QubitStatus.LEAK_E,
        noncomp.Level.LOST: noncomp.QubitStatus.LOST,
    }[level]
    assert (result.final_status == expected_status).all()
    assert not result.heralds.any()
    np.testing.assert_array_equal(result.symbols(), result.measurements)


@pytest.mark.parametrize("basis", ["M", "MX"])
def test_negative_status_probes_preserve_computational_bell_correlations(
    basis, noncomp_sampling_api
):
    # X-basis correlations catch accidental collapse by a status probe.
    prefix = "H 0\nCX 0 1\n"
    probes = "HERALD_LEAKAGE_EVENT 0 1\nHERALD_LOSS_EVENT 0 1\n"
    result = noncomp_sampling_api(
        prefix + probes + f"{basis} 0 1", noncomp.Model(), shots=2048, seed=72
    )
    reference = stim.Circuit(prefix + f"{basis} 0 1").compile_sampler(seed=73).sample(2048)
    assert not result.measurements[:, :4].any()
    measurements = result.measurements[:, 4:]
    assert (measurements[:, 0] == measurements[:, 1]).all()
    assert abs(measurements[:, 0].mean() - reference[:, 0].mean()) < cross_binomial_tolerance(
        0.5, 2048
    )


@pytest.mark.parametrize("gate,transition", zip(HERALDS, ["LEAKAGE", "LOSS"]))
@pytest.mark.parametrize(
    "noise_args,false_positive,miss",
    [
        ("0,0", 0, 0),
        ("0,0.3", 0, 0.3),
        ("0,1", 0, 1),
        ("0.2,0.3", 0.2, 0.3),
        ("1,0", 1, 0),
        ("0.3", 0.3, 0.3),
    ],
)
def test_readout_noise_on_status_records_preserves_true_status(
    gate, transition, noise_args, false_positive, miss, noncomp_sampling_api
):
    # Certain transitions exercise a continuation. Readout errors must be
    # independent across records and leave the underlying site statuses intact.
    result = noncomp_sampling_api(
        f"{transition}(1) 0\n{gate} 0 1 0\n"
        f"READOUT_NOISE({noise_args}) rec[-3] rec[-2] rec[-1]\n{gate} 0",
        noncomp.Model(),
        shots=2048,
        seed=74,
    )
    reference = (
        stim.Circuit(f"MPAD({miss}) 1\nMPAD({false_positive}) 0\nMPAD({miss}) 1\nMPAD 1")
        .compile_sampler(seed=75)
        .sample(2048)
    )
    actual = result.measurements
    assert actual[:, 3].all()
    for column, probability in enumerate((1 - miss, false_positive, 1 - miss)):
        assert abs(
            actual[:, column].mean() - reference[:, column].mean()
        ) < cross_binomial_tolerance(probability, 2048)
    joint = (actual[:, 0] & actual[:, 2]).mean()
    assert abs(joint - (1 - miss) ** 2) < binomial_tolerance((1 - miss) ** 2, 2048)
    positive_status = (
        noncomp.QubitStatus.LEAK_G if transition == "LEAKAGE" else noncomp.QubitStatus.LOST
    )
    assert (result.final_status == [positive_status, noncomp.QubitStatus.COMPUTATIONAL]).all()
    assert not result.heralds.any()


def test_status_history_tracks_jumps_recovery_and_resets(noncomp_sampling_api):
    recover = noncomp_transition_matrix(
        {
            (noncomp.Level.E, noncomp.Level.LEAK_G): 1.0,
            (noncomp.Level.E, noncomp.Level.LEAK_E): 1.0,
        }
    )
    result = noncomp_sampling_api(
        """
        LEAKAGE(0.4) 0
        HERALD_LEAKAGE_EVENT 0
        LOSS(0.5) 0
        HERALD_LEAKAGE_EVENT 0
        HERALD_LOSS_EVENT 0
        LEVEL_TRANSITION[recover] 0
        HERALD_LEAKAGE_EVENT 0
        HERALD_LOSS_EVENT 0
        R 0
        HERALD_LEAKAGE_EVENT 0
        HERALD_LOSS_EVENT 0
        """,
        noncomp.Model(transitions={"recover": recover}),
        shots=1024,
        seed=76,
    )
    m = result.measurements
    assert (m[:, 1] == (m[:, 0] & (1 - m[:, 2]))).all()
    assert not m[:, [3, 5]].any()
    np.testing.assert_array_equal(m[:, 2], m[:, 4])
    np.testing.assert_array_equal(m[:, 2], m[:, 6])
    assert (m[:, 2] == (result.final_status[:, 0] == noncomp.QubitStatus.LOST)).all()
    assert abs(m[:, 0].mean() - 0.4) < binomial_tolerance(0.4, 1024)
    assert abs(m[:, 2].mean() - 0.5) < binomial_tolerance(0.5, 1024)


@pytest.mark.parametrize("gate,transition", zip(HERALDS, ["LEAKAGE", "LOSS"]))
def test_noisy_herald_records_drive_feedback_and_qec_outputs(
    gate, transition, noncomp_sampling_api
):
    result = noncomp_sampling_api(
        f"""
        REPEAT 2 {{
            {transition}(1) 0
            {gate} 0
            READOUT_NOISE(0, 0.3) rec[-1]
            CX rec[-1] 1
            H 2
            CZ rec[-1] 2
            M 1
            MX 2
            DETECTOR rec[-1] rec[-3]
            DETECTOR rec[-2] rec[-3]
            OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
            R 0 1 2
        }}
        """,
        noncomp.Model(classifier=CLASSIFIER, reset_restores_lost=True),
        shots=1024,
        seed=77,
    )
    for offset in (0, 3):
        flag = result.measurements[:, offset]
        assert flag.any() and not flag.all()
        np.testing.assert_array_equal(flag, result.measurements[:, offset + 1])
        np.testing.assert_array_equal(flag, result.measurements[:, offset + 2])
    assert not result.detectors.any()
    assert not result.observables.any()
    assert not result.heralds.any()
    assert (result.final_status == noncomp.QubitStatus.COMPUTATIONAL).all()


def test_status_records_are_separate_from_classifier_heralds(noncomp_sampling_api):
    classifier = noncomp.Classifier([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1]])
    result = noncomp_sampling_api(
        "LEAKAGE(1) 0\nHERALD_LEAKAGE_EVENT 0\nM 0\nHERALD_LEAKAGE_EVENT 0",
        noncomp.Model(classifier=classifier),
        shots=257,
        seed=78,
    )
    assert result.measurements[:, [0, 2]].all()
    assert (result.heralds == [0, 1, 0]).all()
    assert (result.symbols() == [1, 2, 1]).all()


@pytest.mark.parametrize("gate", HERALDS)
def test_status_heralds_reject_ordinary_compilation_and_transition_hooks(gate):
    with pytest.raises(ValueError, match=r"clifft\.noncomp\.sample"):
        clifft.compile(f"{gate} 0")
    with pytest.raises(ValueError, match="hook"):
        noncomp.Model(transitions={gate: noncomp_transition_matrix({})})


def test_herald_prefix_is_reproducible_across_continuations_and_workers():
    circuit = """
        LEAKAGE(1) 0
        HERALD_LEAKAGE_EVENT 0
        READOUT_NOISE(0, 0.3) rec[-1]
        LOSS(0.5) 1
        HERALD_LOSS_EVENT 1
        READOUT_NOISE(0, 0.2) rec[-1]
        LEAKAGE(0.5) 2
        HERALD_LEAKAGE_EVENT 2
        READOUT_NOISE(0, 0.4) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """
    one = noncomp.sample(circuit, noncomp.Model(), shots=257, seed=79, threads=1)
    two = noncomp.sample(circuit, noncomp.Model(), shots=257, seed=79, threads=2)
    for field in ("measurements", "detectors", "observables", "heralds", "final_status"):
        np.testing.assert_array_equal(getattr(one, field), getattr(two, field))
    np.testing.assert_array_equal(one.observables[:, 0], one.measurements[:, 0])
