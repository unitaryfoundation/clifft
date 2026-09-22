"""Independent reference checks for noise around postselection."""

import numpy as np
import stim
from utils_conformance import CpuSamplingMode


def test_noise_preserves_conditioned_joint_distribution(sampling_mode: CpuSamplingMode) -> None:
    text = """
        CORRELATED_ERROR(0.25) X0 X1
        ELSE_CORRELATED_ERROR(0.25) X1
        M 0
        DETECTOR rec[-1]
        H 2
        CX 2 3
        DEPOLARIZE2(0.12) 2 3
        M 2
        DETECTOR rec[-1]
        CX rec[-1] 1
        M 1 3
        R 0
        H 0
        PAULI_CHANNEL_1(0.1, 0.2, 0.05) 0
        MX 0
    """
    shots = 100000
    reference = stim.Circuit(text).compile_sampler(seed=23).sample(shots)
    reference = reference[~np.any(reference[:, :2], axis=1)]
    program = sampling_mode.compile(text, postselection_mask=[1, 1])
    result = sampling_mode.sample_survivors(program, shots=shots, seed=25, keep_records=True)
    repeated = sampling_mode.sample_survivors(program, shots=shots, seed=25, keep_records=True)
    assert np.array_equal(result.measurements, repeated.measurements)
    assert not np.any(result.detectors)
    assert abs(result.passed_shots - len(reference)) < 6 * np.sqrt(shots * 0.25)
    weights = 1 << np.arange(5)
    actual_counts = np.bincount(result.measurements @ weights, minlength=32)
    reference_counts = np.bincount(reference @ weights, minlength=32)
    actual = actual_counts / result.passed_shots
    expected = reference_counts / len(reference)
    variance = expected * (1 - expected) * (1 / len(reference) + 1 / result.passed_shots)
    assert np.all(np.abs(actual - expected) <= 6 * np.sqrt(variance) + 0.001)


def test_noise_preserves_noisy_nonclifford_interference(sampling_mode: CpuSamplingMode) -> None:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    text = """
        X_ERROR(0.2) 0
        M 0
        DETECTOR rec[-1]
        H 1
        Z_ERROR(0.3) 1
        T 1
        T 1
        T 1
        H 1
        M 1
    """
    program = sampling_mode.compile(text, postselection_mask=[1])
    result = sampling_mode.sample_survivors(program, shots=100000, seed=26, keep_records=True)
    probabilities = []
    for error in [False, True]:
        circuit = QuantumCircuit(1)
        circuit.h(0)
        if error:
            circuit.z(0)
        circuit.t(0)
        circuit.t(0)
        circuit.t(0)
        circuit.h(0)
        circuit.save_probabilities()
        probabilities.append(
            AerSimulator(method="statevector").run(circuit).result().data(0)["probabilities"][1]
        )
    expected = 0.7 * probabilities[0] + 0.3 * probabilities[1]
    assert abs(np.mean(result.measurements[:, 1]) - expected) < 0.01
