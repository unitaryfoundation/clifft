"""Public joint-law checks for the packed classical input program."""

from collections import Counter

import numpy as np
import pytest
import stim

from clifft import noncomp


@pytest.mark.parametrize("batch_size", [65, 2048])
@pytest.mark.parametrize("restore_lost", [False, True])
def test_packed_classical_transition_reset_and_classifier_joint_law(batch_size, restore_lost):
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[1, 0, 0.1, 0.1, 0.2], [0, 1, 0.6, 0.6, 0.3], [0, 0, 0.3, 0.3, 0.5]]
        ),
        reset_restores_lost=restore_lost,
    )
    text = """
        LEAKAGE(0.25) 0
        HERALD_LEAKAGE_EVENT 0
        LOSS(0.3) 0
        HERALD_LOSS_EVENT 0
        HERALD_LEAKAGE_EVENT 0
        R 0
        HERALD_LEAKAGE_EVENT 0
        HERALD_LOSS_EVENT 0
        M 0
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-6]
    """
    program = noncomp.compile(text, model, final_status=True)
    result = program.sample(8195, seed=81, threads=3, batch_size=batch_size, collect_stats=True)
    assert result.stats is not None
    assert result.stats.packed_classical_shots == result.shots
    assert result.stats.continuation_compilations == 0
    assert result.stats.batch_size == batch_size
    serial = program.sample(result.shots, seed=81, threads=1, batch_size=batch_size)
    for name in ("measurements", "heralds", "detectors", "observables", "final_status"):
        np.testing.assert_array_equal(getattr(result, name), getattr(serial, name))
    assert not result.heralds[:, :5].any()
    np.testing.assert_array_equal(
        result.detectors[:, 0], result.measurements[:, 5] ^ result.measurements[:, 4]
    )
    np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, 0])
    assert result.final_status is not None
    joint = np.column_stack((result.measurements, result.heralds[:, 5], result.final_status[:, 0]))
    actual = Counter(map(tuple, joint))
    expected: dict[tuple[int, ...], float] = {}
    for leaked, p_leaked in [(0, 0.75), (1, 0.25)]:
        for lost, p_lost in [(0, 0.7), (1, 0.3)]:
            final_lost = lost and not restore_lost
            readout = (
                [(0, 0, 1.0)]
                if not final_lost
                else [(0, 0, 0.2), (1, 0, 0.3), (0, 1, 0.25), (1, 1, 0.25)]
            )
            for bit, herald, probability in readout:
                row = (
                    leaked,
                    lost,
                    leaked and not lost,
                    0,
                    final_lost,
                    bit,
                    herald,
                    3 if final_lost else 0,
                )
                expected[row] = expected.get(row, 0.0) + p_leaked * p_lost * probability
    assert set(actual) <= set(expected)
    assert (
        sum(abs(actual[row] / result.shots - probability) for row, probability in expected.items())
        / 2
        < 0.03
    )


@pytest.mark.parametrize("batch_size", [65, 2048])
@pytest.mark.parametrize("probability", [0.0, 0.001, 0.3, 1.0])
def test_packed_classical_two_qubit_channel_has_exclusive_pauli_outcomes(batch_size, probability):
    # Two Bell pairs encode both Pauli components as four observable bits.
    quantum = f"H 0 1\nCX 0 2 1 3\nDEPOLARIZE2({probability}) 0 1\nCX 0 2 1 3\nH 0 1\nM 0 2 1 3"
    model = noncomp.Model(
        classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]])
    )
    program = noncomp.compile("LEAKAGE(1) 4\n" + quantum + "\nR 4", model, final_status=True)
    result = program.sample(16387, seed=94, batch_size=batch_size, collect_stats=True)
    assert result.stats is not None
    assert result.stats.packed_classical_shots == result.shots
    codes = result.measurements @ np.array([1, 2, 4, 8])
    actual = np.bincount(codes, minlength=16) / result.shots
    expected = np.full(16, probability / 15)
    expected[0] = 1 - probability
    if probability == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert np.count_nonzero(codes) > 0
        np.testing.assert_allclose(actual, expected, atol=0.012)
    if probability == 1:
        assert actual[0] == 0
    reference = stim.Circuit(quantum).compile_sampler(seed=95).sample(result.shots)
    reference_joint = np.bincount(reference @ np.array([1, 2, 4, 8]), minlength=16) / result.shots
    assert np.abs(actual - reference_joint).sum() / 2 < 0.035
    assert result.final_status is not None
    assert not result.final_status.any()
