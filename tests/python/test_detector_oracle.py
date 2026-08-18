"""Compare Clifft and Stim detector outputs under deterministic errors.

The circuits use probability-one Pauli errors so both simulators should flip
the same detectors and observables on every shot.
"""

import numpy as np
import stim

import clifft


def sample_clifft(circuit_text: str, shots: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample from Clifft, returning (detectors, observables) as bool arrays."""
    prog = clifft.compile(circuit_text)
    result = clifft.sample(prog, shots, seed=seed)
    return result.detectors.astype(bool), result.observables.astype(bool)


def sample_stim(circuit_text: str, shots: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample from Stim, returning (detectors, observables) as bool arrays."""
    circuit = stim.Circuit(circuit_text)
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots=shots, separate_observables=True)
    return det, obs


class TestBellStateDetector:
    """Test detector behavior on Bell states with injected errors."""

    CLEAN_BELL = """
        H 0
        CX 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """

    def test_clean_bell_detector_always_zero(self) -> None:
        """Clean Bell state: measurements always correlated, detector = 0."""
        shots = 100

        clifft_det, _ = sample_clifft(self.CLEAN_BELL, shots)
        stim_det, _ = sample_stim(self.CLEAN_BELL, shots)

        # Both should always be False (0)
        assert np.all(~clifft_det), "Clifft: Clean Bell detector should always be 0"
        assert np.all(~stim_det), "Stim: Clean Bell detector should always be 0"

    def test_x_error_after_cx_flips_detector(self) -> None:
        """X error after CX breaks correlation, detector = 1."""
        circuit = """
            H 0
            CX 0 1
            X_ERROR(1) 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 100

        clifft_det, _ = sample_clifft(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Both should always be True (1) - error flips detector
        assert np.all(clifft_det), "Clifft: X_ERROR should flip detector"
        assert np.all(stim_det), "Stim: X_ERROR should flip detector"
        # Exact match
        np.testing.assert_array_equal(clifft_det, stim_det)

    def test_z_error_before_measurement_detected(self) -> None:
        """Z error before H+M changes measurement basis outcome."""
        # In X basis: H puts qubit in |+>, Z flips to |->, M gives 1
        circuit = """
            H 0
            Z_ERROR(1) 0
            H 0
            M 0
            DETECTOR rec[-1]
        """
        shots = 100

        clifft_det, _ = sample_clifft(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Error flips the expected measurement outcome
        assert np.all(clifft_det), "Clifft: Z_ERROR should be detected"
        assert np.all(stim_det), "Stim: Z_ERROR should be detected"
        np.testing.assert_array_equal(clifft_det, stim_det)


class TestMultiQubitErrorPropagation:
    """Test that errors propagate correctly through multi-qubit gates."""

    def test_x_error_propagates_through_cx(self) -> None:
        """X error on control propagates to target via CX.

        Circuit: X_ERROR on q0, then CX 0 1, then measure both.
        The X error on control propagates to target, so both flip.
        Detector on rec[-1] ^ rec[-2] should still be 0 (both flipped).
        """
        circuit = """
            X_ERROR(1) 0
            CX 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 100

        clifft_det, _ = sample_clifft(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # X propagates through CX: X_0 -> X_0 X_1, both qubits flip
        # Detector XORs them: 1 ^ 1 = 0
        assert np.all(~clifft_det), "Clifft: Propagated X errors should cancel in detector"
        assert np.all(~stim_det), "Stim: Propagated X errors should cancel in detector"
        np.testing.assert_array_equal(clifft_det, stim_det)

    def test_z_error_on_target_invisible_before_entanglement(self) -> None:
        """Z error on |0> is invisible (eigenstate of Z).

        The Z_ERROR occurs before the CX gate, on qubit 1 in state |0>.
        Since |0> is an eigenstate of Z, the error has no observable effect.
        """
        circuit = """
            H 0
            Z_ERROR(1) 1
            CX 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 100

        clifft_det, _ = sample_clifft(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Z error on |0> is invisible (eigenstate), detector stays 0
        assert np.all(~clifft_det), "Clifft: Z error on |0> should be invisible"
        assert np.all(~stim_det), "Stim: Z error on |0> should be invisible"
        np.testing.assert_array_equal(clifft_det, stim_det)


class TestObservableTracking:
    """Test observable value computation matches Stim."""

    def test_observable_tracks_logical_value(self) -> None:
        """Observable accumulates measurement parities correctly."""
        circuit = """
            H 0
            CX 0 1
            M 0 1
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-2]
        """
        shots = 100

        _, clifft_obs = sample_clifft(circuit, shots)
        _, stim_obs = sample_stim(circuit, shots)

        # Bell state: both measurements same, XOR = 0
        assert np.all(~clifft_obs), "Clifft: Bell state observable should be 0"
        assert np.all(~stim_obs), "Stim: Bell state observable should be 0"
        np.testing.assert_array_equal(clifft_obs, stim_obs)

    def test_error_flips_observable(self) -> None:
        """X error on one qubit flips the observable."""
        circuit = """
            H 0
            CX 0 1
            X_ERROR(1) 1
            M 0 1
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-2]
        """
        shots = 100

        _, clifft_obs = sample_clifft(circuit, shots)
        _, stim_obs = sample_stim(circuit, shots)

        # X error breaks correlation: 0,1 or 1,0 -> XOR = 1
        assert np.all(clifft_obs), "Clifft: X error should flip observable"
        assert np.all(stim_obs), "Stim: X error should flip observable"
        np.testing.assert_array_equal(clifft_obs, stim_obs)


class TestMultipleDetectors:
    """Test circuits with multiple detectors."""

    def test_independent_detectors(self) -> None:
        """Multiple independent detectors are computed correctly."""
        circuit = """
            H 0
            H 2
            CX 0 1
            CX 2 3
            M 0 1 2 3
            DETECTOR rec[-4] rec[-3]
            DETECTOR rec[-2] rec[-1]
        """
        shots = 100

        clifft_det, _ = sample_clifft(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Two independent Bell pairs, both detectors = 0
        assert clifft_det.shape == (shots, 2)
        assert np.all(~clifft_det), "Clifft: Both detectors should be 0"
        assert np.all(~stim_det), "Stim: Both detectors should be 0"
        np.testing.assert_array_equal(clifft_det, stim_det)

    def test_error_affects_specific_detector(self) -> None:
        """Error on one Bell pair only affects its detector."""
        circuit = """
            H 0
            H 2
            CX 0 1
            CX 2 3
            X_ERROR(1) 1
            M 0 1 2 3
            DETECTOR rec[-4] rec[-3]
            DETECTOR rec[-2] rec[-1]
        """
        shots = 100

        clifft_det, _ = sample_clifft(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # First detector (q0,q1) should fire, second (q2,q3) should not
        assert np.all(clifft_det[:, 0]), "Clifft: First detector should fire"
        assert np.all(~clifft_det[:, 1]), "Clifft: Second detector should not fire"
        np.testing.assert_array_equal(clifft_det, stim_det)


class TestYError:
    """Test Y error handling (Y = iXZ)."""

    def test_y_error_flips_detector(self) -> None:
        """Y error on Bell state flips detector (acts like X)."""
        circuit = """
            H 0
            CX 0 1
            Y_ERROR(1) 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 100

        clifft_det, _ = sample_clifft(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Y = iXZ, and X part flips the measurement
        assert np.all(clifft_det), "Clifft: Y_ERROR should flip detector"
        assert np.all(stim_det), "Stim: Y_ERROR should flip detector"
        np.testing.assert_array_equal(clifft_det, stim_det)


class TestActiveInterfereErrorTracking:
    """Test error frame tracking through active interference and array compaction."""

    def test_x_error_with_active_interfere(self) -> None:
        """Error frame tracking must survive OP_MEAS_ACTIVE_INTERFERE.

        H 0 -> |+>
        T 0 -> creates an active coordinate (interference required)
        X_ERROR(1) 0 -> error_parity = 1
        M 0 -> OP_MEAS_ACTIVE_INTERFERE + array compaction
        M 0 -> deterministic re-measurement (checks error frame integrity)
        DETECTOR -> two consecutive measurements must always match
        """
        circuit = """
            H 0
            T 0
            X_ERROR(1) 0
            M 0
            M 0
            DETECTOR rec[-1] rec[-2]
        """
        shots = 100

        # We cannot use Stim as an oracle here because Stim cannot simulate T gates.
        # But quantum mechanics guarantees the two consecutive measurements must match.
        clifft_det, _ = sample_clifft(circuit, shots)

        assert np.all(~clifft_det), "Clifft: Measurements should match (error frame corrupted!)"
