"""Integration tests for importance sampling (forced k-fault) API."""

import numpy as np
import numpy.typing as npt
import pytest
from conftest import binomial_tolerance
from utils_conformance import SMALL_CIRCUIT_SHOTS, CpuSamplingMode

import clifft


def poisson_binomial_pmf(probs: npt.NDArray[np.float64], max_k: int) -> npt.NDArray[np.float64]:
    """Compute exact Poisson-Binomial PMF via DP."""
    dp = np.zeros(max_k + 1)
    dp[0] = 1.0
    for p in probs:
        for k in range(max_k, 0, -1):
            dp[k] = dp[k] * (1.0 - p) + dp[k - 1] * p
        dp[0] *= 1.0 - p
    return dp


class TestNoiseSiteProbabilities:
    def test_basic_extraction(self) -> None:
        prog = clifft.compile(
            """
            R 0 1
            DEPOLARIZE1(0.03) 0
            X_ERROR(0.01) 1
            M(0.005) 0
            M 1
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        probs = prog.noise_site_probabilities
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float64
        assert len(probs) == 3
        np.testing.assert_allclose(probs[0], 0.03, atol=1e-12)
        np.testing.assert_allclose(probs[1], 0.01, atol=1e-12)
        np.testing.assert_allclose(probs[2], 0.005, atol=1e-12)

    def test_no_noise(self) -> None:
        prog = clifft.compile("R 0\nH 0\nM 0", normalize_syndromes=True)
        probs = prog.noise_site_probabilities
        assert len(probs) == 0


class TestSampleK:
    def test_k0_no_errors(self, sampling_mode: CpuSamplingMode) -> None:
        """With k=0 forced faults, no errors should appear."""
        prog = sampling_mode.compile(
            """
            R 0 1 2
            X_ERROR(0.1) 0 1 2
            M 0 1 2
            DETECTOR rec[-1] rec[-2]
            DETECTOR rec[-2] rec[-3]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        result = sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=0, seed=42)
        assert result.measurements.shape == (SMALL_CIRCUIT_SHOTS, 3)
        assert np.all(result.observables == 0)
        assert np.all(result.detectors == 0)

    def test_k_equals_n_forces_all(self, sampling_mode: CpuSamplingMode) -> None:
        """k=N should force every noise site to fire."""
        prog = sampling_mode.compile(
            """
            R 0
            X_ERROR(0.5) 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        n_sites = len(prog.noise_site_probabilities)
        assert n_sites == 1
        result = sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=42)
        assert np.all(result.observables == 1)

    def test_k_exceeds_n_raises(self, sampling_mode: CpuSamplingMode) -> None:
        prog = sampling_mode.compile(
            """
            R 0
            X_ERROR(0.1) 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        n_sites = len(prog.noise_site_probabilities)
        with pytest.raises(ValueError):
            sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=n_sites + 1, seed=42)

    def test_zero_mass_stratum_raises(self, sampling_mode: CpuSamplingMode) -> None:
        """k > 0 on a noiseless circuit should raise (zero-mass stratum)."""
        prog = sampling_mode.compile(
            "R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", normalize_syndromes=True
        )
        assert len(prog.noise_site_probabilities) == 0
        # k=0 is fine
        sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=0, seed=42)
        # k=1 is impossible
        with pytest.raises(ValueError):
            sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=42)
        with pytest.raises(ValueError):
            sampling_mode.sample_k_survivors(prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=42)

    def test_exactly_k_faults_per_shot(self, sampling_mode: CpuSamplingMode) -> None:
        """Verify exactly k measurements flip per shot with X_ERROR."""
        prog = sampling_mode.compile(
            """
            R 0 1 2
            X_ERROR(0.01) 0
            X_ERROR(0.05) 1
            X_ERROR(0.1) 2
            M 0 1 2
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        for k in range(4):
            result = sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=k, seed=42 + k)
            flips_per_shot = np.sum(result.measurements, axis=1)
            np.testing.assert_array_equal(flips_per_shot, k)

    def test_deterministic_with_seed(self, sampling_mode: CpuSamplingMode) -> None:
        prog = sampling_mode.compile(
            """
            R 0 1
            DEPOLARIZE1(0.05) 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        r1 = sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=99)
        r2 = sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=99)
        np.testing.assert_array_equal(r1.measurements, r2.measurements)
        np.testing.assert_array_equal(r1.detectors, r2.detectors)
        np.testing.assert_array_equal(r1.observables, r2.observables)

    @pytest.mark.parametrize("batch_size", [1, 65])
    def test_threads_preserve_seeded_rows(self, batch_size: int) -> None:
        prog = clifft.compile(
            "X_ERROR(0.1) 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\nEXP_VAL Z2",
            normalize_syndromes=True,
        )
        serial = clifft.sample_k(prog, shots=257, k=1, seed=99, threads=1, batch_size=batch_size)
        threaded = clifft.sample_k(prog, shots=257, k=1, seed=99, threads=2, batch_size=batch_size)
        np.testing.assert_array_equal(threaded.measurements, serial.measurements)
        np.testing.assert_array_equal(threaded.detectors, serial.detectors)
        np.testing.assert_array_equal(threaded.observables, serial.observables)
        np.testing.assert_array_equal(threaded.exp_vals, serial.exp_vals)

    def test_readout_noise_forcing(self, sampling_mode: CpuSamplingMode) -> None:
        """k=1 with only readout noise should flip every shot."""
        prog = sampling_mode.compile(
            """
            R 0
            M(0.1) 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        assert len(prog.noise_site_probabilities) == 1
        result = sampling_mode.sample_k(prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=42)
        assert np.all(result.observables == 1)


class TestSampleKSurvivors:
    def test_k0_no_errors(self, sampling_mode: CpuSamplingMode) -> None:
        prog = sampling_mode.compile(
            """
            R 0 1 2
            X_ERROR(0.1) 0 1 2
            M 0 1 2
            DETECTOR rec[-1] rec[-2]
            DETECTOR rec[-2] rec[-3]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        result = sampling_mode.sample_k_survivors(prog, shots=SMALL_CIRCUIT_SHOTS, k=0, seed=42)
        assert isinstance(result, clifft.SampleResult)
        assert result.total_shots == SMALL_CIRCUIT_SHOTS
        assert result.passed_shots == SMALL_CIRCUIT_SHOTS
        assert result.logical_errors == 0
        assert result.measurements.shape == (0, prog.num_measurements)

    def test_keep_records(self, sampling_mode: CpuSamplingMode) -> None:
        prog = sampling_mode.compile(
            """
            R 0 1
            X_ERROR(0.1) 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        result = sampling_mode.sample_k_survivors(
            prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=42, keep_records=True
        )
        passed = result.passed_shots
        assert passed > 0
        assert result.measurements.shape == (passed, prog.num_measurements)
        assert result.detectors.shape == (passed, prog.num_detectors)
        assert result.observables.shape == (passed, prog.num_observables)

    def test_threads_preserve_seeded_survivors(self) -> None:
        prog = clifft.compile(
            "X_ERROR(0.1) 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]",
            postselection_mask=[1],
        )
        serial = clifft.sample_k_survivors(
            prog, shots=257, k=1, seed=100, keep_records=True, threads=1
        )
        threaded = clifft.sample_k_survivors(
            prog, shots=257, k=1, seed=100, keep_records=True, threads=2
        )
        assert threaded.passed_shots == serial.passed_shots
        assert threaded.logical_errors == serial.logical_errors
        np.testing.assert_array_equal(threaded.observable_ones, serial.observable_ones)
        np.testing.assert_array_equal(threaded.measurements, serial.measurements)
        np.testing.assert_array_equal(threaded.detectors, serial.detectors)
        np.testing.assert_array_equal(threaded.observables, serial.observables)

    def test_survivors_replay_seeded_rows(self, sampling_mode: CpuSamplingMode) -> None:
        prog = sampling_mode.compile(
            "X_ERROR(0.1) 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]",
            postselection_mask=[1],
        )
        first = sampling_mode.sample_k_survivors(prog, shots=257, k=1, seed=101, keep_records=True)
        replay = sampling_mode.sample_k_survivors(prog, shots=257, k=1, seed=101, keep_records=True)
        assert first.passed_shots == replay.passed_shots
        assert first.logical_errors == replay.logical_errors
        np.testing.assert_array_equal(first.measurements, replay.measurements)
        np.testing.assert_array_equal(first.observables, replay.observables)


class TestImportanceSamplingEndToEnd:
    """Integration test: verify the stratified importance sampling workflow."""

    def test_single_qubit_k0_vs_k1(self, sampling_mode: CpuSamplingMode) -> None:
        """Single qubit: k=0 has no error, k=1 always has error."""
        prog = sampling_mode.compile(
            """
            R 0
            X_ERROR(0.1) 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """,
            normalize_syndromes=True,
        )
        probs = prog.noise_site_probabilities
        assert len(probs) == 1

        r0 = sampling_mode.sample_k_survivors(prog, shots=SMALL_CIRCUIT_SHOTS, k=0, seed=42)
        assert r0.logical_errors == 0

        r1 = sampling_mode.sample_k_survivors(prog, shots=SMALL_CIRCUIT_SHOTS, k=1, seed=42)
        assert r1.logical_errors == r1.passed_shots

    def test_weighted_error_rate_single_qubit(self, sampling_mode: CpuSamplingMode) -> None:
        """Stratified estimate matches exact for single-qubit X_ERROR."""
        p_phys = 0.05
        circuit_text = f"""
            R 0
            X_ERROR({p_phys}) 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = sampling_mode.compile(circuit_text, normalize_syndromes=True)
        probs = prog.noise_site_probabilities
        max_k = len(probs)
        pmf = poisson_binomial_pmf(probs, max_k)

        # Stratified estimate using the general postselection-safe formula:
        # p_fail = sum(P(K=k) * errors_k / shots_k) / sum(P(K=k) * passed_k / shots_k)
        weighted_errors = 0.0
        weighted_survival = 0.0
        for k in range(max_k + 1):
            if pmf[k] < 1e-15:
                continue
            result = sampling_mode.sample_k_survivors(
                prog, shots=SMALL_CIRCUIT_SHOTS, k=k, seed=42 + k
            )
            total = result.total_shots
            if total == 0:
                continue
            weighted_errors += pmf[k] * result.logical_errors / total
            weighted_survival += pmf[k] * result.passed_shots / total

        p_fail_stratified = weighted_errors / weighted_survival if weighted_survival > 0 else 0.0

        # For single qubit with X_ERROR(p), p_L = p exactly.
        p_fail_exact = p_phys

        assert p_fail_stratified == pytest.approx(p_fail_exact, abs=1e-12)

    def test_weighted_error_rate_two_qubits(self, sampling_mode: CpuSamplingMode) -> None:
        """Stratified estimate matches the exact marginal error probability."""
        circuit_text = """
            R 0 1
            X_ERROR(0.1) 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = sampling_mode.compile(circuit_text, normalize_syndromes=True)
        probs = prog.noise_site_probabilities
        max_k = len(probs)
        pmf = poisson_binomial_pmf(probs, max_k)

        # Stratified estimate using the general postselection-safe formula
        weighted_errors = 0.0
        weighted_survival = 0.0
        for k in range(max_k + 1):
            if pmf[k] < 1e-15:
                continue
            result = sampling_mode.sample_k_survivors(prog, shots=1025, k=k, seed=42 + k)
            total = result.total_shots
            if total == 0:
                continue
            weighted_errors += pmf[k] * result.logical_errors / total
            weighted_survival += pmf[k] * result.passed_shots / total

        p_fail_stratified = weighted_errors / weighted_survival if weighted_survival > 0 else 0.0

        # Observable is rec[-1] = qubit 1. Error whenever qubit 1 flips.
        # p_fail = p = 0.1
        assert abs(p_fail_stratified - 0.1) < pmf[1] * binomial_tolerance(0.5, 1025)
