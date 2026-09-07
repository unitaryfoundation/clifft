"""Noise and continuation histories must not change subsequent shots."""

import numpy as np
import pytest
import stim
from conftest import cross_binomial_tolerance

from clifft import noncomp


@pytest.mark.parametrize("annotation", ["LOSS", "LEAKAGE"])
@pytest.mark.parametrize("damping", ["exact", "neglect"])
@pytest.mark.parametrize(
    "noise",
    [
        "X_ERROR(0.02) 2\nX_ERROR(0.02) 2\nX_ERROR(0.04) 3",
        "DEPOLARIZE1(0.12) 2 3",
        "DEPOLARIZE2(0.15) 2 3",
        "PAULI_CHANNEL_1(0.02,0.03,0.04) 2 3",
    ],
    ids=["bit-flips", "depolarize-one", "depolarize-two", "categorical"],
)
def test_noisy_continuations_preserve_spectators_and_seeded_rows(annotation, damping, noise):
    # Spectators never interact with the trapping Bell pair. Stim can therefore
    # independently check their joint distribution even though it has no loss model.
    circuit = "\n".join(
        [
            "H 0\nCX 0 1",
            noise,
            f"{annotation}(0.5) 1",
            noise,
            f"{annotation}(0.5) 0",
            noise,
            "M(0.03) 0 1 2 3",
            "DETECTOR rec[-2]\nDETECTOR rec[-1]",
            "OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]",
        ]
    )
    classifier = noncomp.Classifier([[1, 0, 1, 0, 0.3], [0, 1, 0, 1, 0.4], [0, 0, 0, 0, 0.3]])
    model = noncomp.Model(classifier=classifier, damping=damping)
    shots = 8192
    serial = noncomp.sample(circuit, model, shots=shots, seed=23, threads=1)
    parallel = noncomp.sample(circuit, model, shots=shots, seed=23, threads=3)
    for field in ("measurements", "detectors", "observables", "final_status", "heralds"):
        np.testing.assert_array_equal(
            getattr(serial, field), getattr(parallel, field), err_msg=field
        )

    computational = serial.final_status == noncomp.QubitStatus.COMPUTATIONAL
    for qubit in (0, 1):
        assert computational[:, qubit].any()
        assert (~computational[:, qubit]).any()
    assert computational[:, :2].all(axis=1).any()
    assert (~computational[:, :2]).all(axis=1).any()
    assert computational[:, 2:].all()
    np.testing.assert_array_equal(serial.detectors, serial.measurements[:, 2:])
    np.testing.assert_array_equal(
        serial.observables[:, 0], serial.measurements[:, 2] ^ serial.measurements[:, 3]
    )

    reference_circuit = stim.Circuit("\n".join([noise, noise, noise, "M(0.03) 2 3"]))
    reference = reference_circuit.compile_sampler(seed=71).sample(shots)
    expected = np.bincount(reference[:, 0] + 2 * reference[:, 1], minlength=4) / shots
    actual = (
        np.bincount(serial.measurements[:, 2] + 2 * serial.measurements[:, 3], minlength=4) / shots
    )
    for outcome in range(4):
        pooled = float((actual[outcome] + expected[outcome]) / 2)
        assert abs(actual[outcome] - expected[outcome]) < cross_binomial_tolerance(pooled, shots)
