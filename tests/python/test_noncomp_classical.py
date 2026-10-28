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


def test_packed_classical_worker_storage_counts_whole_pages():
    model = noncomp.Model(
        classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]])
    )
    sizes = []
    for qubit in (0, 1):
        program = noncomp.compile(f"LOSS(0.1) {qubit}\nM {qubit}", model)
        result = program.sample(65, seed=7, batch_size=65, collect_stats=True)
        assert result.stats is not None
        assert result.stats.packed_classical_shots == 65
        sizes.append(result.stats.packed_worker_bytes)
    # Reindexing the sole operated qubit adds ledger columns but stays within
    # one page. The quantum plan and other retained buffer shapes are identical.
    assert sizes[0] == sizes[1]


@pytest.mark.parametrize("batch_size", [65, 2048])
@pytest.mark.parametrize(
    ("text", "restore_lost"),
    [
        pytest.param(
            "REPEAT 2 {\nR 0\nLEAKAGE(0.25) 0\nM !0\nHERALD_LEAKAGE_EVENT 0\n}\n"
            "R 0\nX_ERROR(0.4) 0\nM 0",
            False,
            id="repeated-readout-and-reset",
        ),
        pytest.param(
            "H 1\nT 1\nCX 1 2\nLOSS(0.3) 0\nCX 0 1\nE(0.2) X0\n"
            "ELSE_CORRELATED_ERROR(0.4) X2\nDEPOLARIZE2(0.2) 0 2\nM !0\n"
            "CX rec[-1] 2\nH 1 2\nM 1 2\nHERALD_LOSS_EVENT 0",
            False,
            id="non-clifford-partner-and-conditioned-noise",
        ),
        pytest.param(
            "H 0\nR_Z(0.173) 0\nH 0\nLOSS(0.35) 0\nM !0\nMR 0\n"
            "READOUT_NOISE(0.2,0.7) rec[-1]\nCX rec[-1] 1\nM 0 1",
            True,
            id="restoring-readout-and-feedback",
        ),
    ],
)
def test_packed_classical_joint_law_matches_scalar_reference(text, restore_lost, batch_size):
    effect = noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25))
    model = noncomp.Model(
        classifier=noncomp.Classifier(
            [[0.9, 0.2, 0.15, 0.15, 0.2], [0.1, 0.8, 0.55, 0.55, 0.3], [0, 0, 0.3, 0.3, 0.5]]
        ),
        reset_restores_lost=restore_lost,
        gate_partner_effects={"leaked": effect, "lost": effect},
    )
    text += "\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    program = noncomp.compile(text, model, final_status=True, execution="fixed")
    shots = 32771
    scalar = program.sample(shots, seed=71, batch_size=1, collect_stats=True)
    packed = program.sample(shots, seed=72, threads=3, batch_size=batch_size, collect_stats=True)
    assert scalar.stats is not None and packed.stats is not None
    assert scalar.stats.scalar_shots == shots
    assert packed.stats.packed_classical_shots == shots
    assert packed.stats.workers == 3

    distributions = []
    for result in (scalar, packed):
        np.testing.assert_array_equal(
            result.detectors[:, 0], result.measurements[:, -1] ^ result.measurements[:, -2]
        )
        np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, -1])
        assert result.final_status is not None
        assert np.isin(
            result.final_status, [noncomp.QubitStatus.COMPUTATIONAL, noncomp.QubitStatus.LOST]
        ).all()
        joint = np.column_stack(
            (result.measurements, result.heralds, result.final_status == noncomp.QubitStatus.LOST)
        )
        # These short circuits have few reachable joint cells. Check cell-wise
        # sampling uncertainty instead of a loose whole-distribution threshold.
        codes = joint @ (1 << np.arange(joint.shape[1]))
        distributions.append(np.bincount(codes, minlength=1 << joint.shape[1]) / shots)
    pooled = (distributions[0] + distributions[1]) / 2
    tolerance = 6 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 2 / shots
    assert np.all(np.abs(distributions[0] - distributions[1]) < tolerance)

    serial = program.sample(shots, seed=72, threads=1, batch_size=batch_size)
    for name in ("measurements", "heralds", "detectors", "observables", "final_status"):
        np.testing.assert_array_equal(getattr(packed, name), getattr(serial, name))
