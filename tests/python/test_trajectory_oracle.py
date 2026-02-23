"""Exact Trajectory Validation - Physics Oracle.

This module validates that UCC's Heisenberg rewinding of Pauli masks
matches Stim's geometry by comparing detector and observable outputs
for circuits with deterministically injected errors.

The key insight: when we inject a 100% probability error (e.g., X_ERROR(1)),
both engines should deterministically flip the same detectors.
"""

import numpy as np
import stim

import ucc


def sample_ucc(circuit_text: str, shots: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample from UCC, returning (detectors, observables) as bool arrays."""
    prog = ucc.compile(circuit_text)
    _, det, obs = ucc.sample(prog, shots, seed=seed)
    return det.astype(bool), obs.astype(bool)


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

        ucc_det, _ = sample_ucc(self.CLEAN_BELL, shots)
        stim_det, _ = sample_stim(self.CLEAN_BELL, shots)

        # Both should always be False (0)
        assert np.all(~ucc_det), "UCC: Clean Bell detector should always be 0"
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

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Both should always be True (1) - error flips detector
        assert np.all(ucc_det), "UCC: X_ERROR should flip detector"
        assert np.all(stim_det), "Stim: X_ERROR should flip detector"
        # Exact match
        np.testing.assert_array_equal(ucc_det, stim_det)

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

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Error flips the expected measurement outcome
        assert np.all(ucc_det), "UCC: Z_ERROR should be detected"
        assert np.all(stim_det), "Stim: Z_ERROR should be detected"
        np.testing.assert_array_equal(ucc_det, stim_det)


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

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # X propagates through CX: X_0 -> X_0 X_1, both qubits flip
        # Detector XORs them: 1 ^ 1 = 0
        assert np.all(~ucc_det), "UCC: Propagated X errors should cancel in detector"
        assert np.all(~stim_det), "Stim: Propagated X errors should cancel in detector"
        np.testing.assert_array_equal(ucc_det, stim_det)

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

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Z error on |0> is invisible (eigenstate), detector stays 0
        assert np.all(~ucc_det), "UCC: Z error on |0> should be invisible"
        assert np.all(~stim_det), "Stim: Z error on |0> should be invisible"
        np.testing.assert_array_equal(ucc_det, stim_det)


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

        _, ucc_obs = sample_ucc(circuit, shots)
        _, stim_obs = sample_stim(circuit, shots)

        # Bell state: both measurements same, XOR = 0
        assert np.all(~ucc_obs), "UCC: Bell state observable should be 0"
        assert np.all(~stim_obs), "Stim: Bell state observable should be 0"
        np.testing.assert_array_equal(ucc_obs, stim_obs)

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

        _, ucc_obs = sample_ucc(circuit, shots)
        _, stim_obs = sample_stim(circuit, shots)

        # X error breaks correlation: 0,1 or 1,0 -> XOR = 1
        assert np.all(ucc_obs), "UCC: X error should flip observable"
        assert np.all(stim_obs), "Stim: X error should flip observable"
        np.testing.assert_array_equal(ucc_obs, stim_obs)


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

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Two independent Bell pairs, both detectors = 0
        assert ucc_det.shape == (shots, 2)
        assert np.all(~ucc_det), "UCC: Both detectors should be 0"
        assert np.all(~stim_det), "Stim: Both detectors should be 0"
        np.testing.assert_array_equal(ucc_det, stim_det)

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

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # First detector (q0,q1) should fire, second (q2,q3) should not
        assert np.all(ucc_det[:, 0]), "UCC: First detector should fire"
        assert np.all(~ucc_det[:, 1]), "UCC: Second detector should not fire"
        np.testing.assert_array_equal(ucc_det, stim_det)


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

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Y = iXZ, and X part flips the measurement
        assert np.all(ucc_det), "UCC: Y_ERROR should flip detector"
        assert np.all(stim_det), "Stim: Y_ERROR should flip detector"
        np.testing.assert_array_equal(ucc_det, stim_det)


class TestDepolarizingChannelDeterministic:
    """Test DEPOLARIZE1/2 at 100% probability (deterministic for testing)."""

    def test_depolarize1_full_strength(self) -> None:
        """DEPOLARIZE1(1) applies random Pauli with expected 2/3 firing rate.

        At p=1, DEPOLARIZE1 always applies X, Y, or Z (each with p=1/3).
        On a Z-basis measurement after |0>, X and Y flip the result.
        So 2/3 of the time the detector fires.

        For validation, we check both engines produce similar statistics.
        """
        circuit = """
            DEPOLARIZE1(1) 0
            M 0
            DETECTOR rec[-1]
        """
        shots = 100
        seed = 42

        ucc_det, _ = sample_ucc(circuit, shots, seed=seed)
        stim_det, _ = sample_stim(circuit, shots, seed=seed)

        # With different RNG streams, we can't expect exact match per-shot,
        # but the statistics should be similar (roughly 2/3 fire)
        ucc_rate = float(np.mean(ucc_det))
        stim_rate = float(np.mean(stim_det))

        # Both should be roughly 2/3 (X and Y flip, Z doesn't)
        assert 0.4 < ucc_rate < 0.9, f"UCC rate {ucc_rate} outside expected range"
        assert 0.4 < stim_rate < 0.9, f"Stim rate {stim_rate} outside expected range"


class TestMeasureMergeErrorTracking:
    """Test error frame tracking when AG_PIVOT follows MEASURE_MERGE."""

    def test_x_error_with_measure_merge(self) -> None:
        """Error frame tracking must be correct when AG_PIVOT follows MEASURE_MERGE.

        H 0 -> |+>
        T 0 -> creates GF2 dimension (MEASURE_MERGE required)
        X_ERROR(1) 0 -> error_parity = 1
        M 0 -> MEASURE_MERGE + AG_PIVOT(use_prev_outcome=True)
        M 0 -> MEASURE_DETERMINISTIC (checks error frame integrity)
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
        ucc_det, _ = sample_ucc(circuit, shots)

        assert np.all(~ucc_det), "UCC: Measurements should match (error frame corrupted!)"


class TestComplexCircuit:
    """Test the example circuit from the plan."""

    def test_plan_example_circuit_clean(self) -> None:
        """Example from plan: H 0, CX 0 1, M 0 1, DETECTOR rec[-1] rec[-2]."""
        circuit = """
            H 0
            CX 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 100

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # Clean circuit: detector always 0
        np.testing.assert_array_equal(ucc_det, stim_det)
        assert np.all(~ucc_det)

    def test_bell_state_with_x_error_fires_detector(self) -> None:
        """X error on control qubit breaks Bell correlation, firing detector.

        We use X_ERROR(1) for deterministic error injection that both
        UCC and Stim's detector sampler recognize.
        """
        circuit = """
            H 0
            CX 0 1
            X_ERROR(1) 0
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 100

        ucc_det, _ = sample_ucc(circuit, shots)
        stim_det, _ = sample_stim(circuit, shots)

        # X error on q0 after CX: only q0 flips, detector fires
        np.testing.assert_array_equal(ucc_det, stim_det)
        assert np.all(ucc_det), "Detector should fire with X error"
