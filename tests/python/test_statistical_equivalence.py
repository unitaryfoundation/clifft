"""Statistical Equivalence Tests - Bulk Distribution Validation.

This module validates that UCC's AOT noise scheduling, multi-Pauli
depolarizing decomposition, readout noise, and classical logic pipelines
produce statistically equivalent distributions to Stim.

Tests run large numbers of shots and verify that marginal firing
probabilities of all detectors and observables fall within strict
statistical bounds.
"""

from pathlib import Path

import numpy as np
import pytest
import stim
from conftest import cross_binomial_tolerance

import ucc

# Path to the shared target QEC circuit file (also used by benchmarks)
_TARGET_QEC_PATH = Path(__file__).parent.parent.parent / "tools" / "bench" / "target_qec.stim"


def _load_target_qec_circuit() -> str:
    """Load the CSS d=3 cultivation circuit from the shared file."""
    return _TARGET_QEC_PATH.read_text()


class TestTargetQECCircuit:
    """Statistical equivalence tests on the CSS d=3 cultivation circuit.

    Circuit source:
        Gidney, C., Jones, C., & Shutty, N. (2024).
        Data for "Magic state cultivation: growing T states as cheap as CNOT gates".
        Zenodo. https://doi.org/10.5281/zenodo.13777072
        License: Apache 2.0

    The circuit is a CSS stabilizer code with:
    - 14 qubits
    - Multiple stabilizer measurement rounds
    - DEPOLARIZE1/2 noise at p=0.001
    - Readout noise at p=0.001
    - 22 detectors and 1 logical observable
    """

    @pytest.fixture(scope="class")
    def circuit_text(self) -> str:
        """Load the target circuit from shared file."""
        return _load_target_qec_circuit()

    @pytest.fixture(scope="class")
    def ucc_program(self, circuit_text: str) -> ucc.Program:
        """Compile UCC program."""
        return ucc.compile(circuit_text)

    @pytest.fixture(scope="class")
    def stim_sampler(self, circuit_text: str) -> stim.CompiledDetectorSampler:
        """Compile Stim detector sampler with fixed seed."""
        circuit = stim.Circuit(circuit_text)
        return circuit.compile_detector_sampler(seed=42)

    def test_circuit_metadata_matches(self, circuit_text: str, ucc_program: ucc.Program) -> None:
        """Verify UCC and Stim agree on circuit structure."""
        stim_circuit = stim.Circuit(circuit_text)

        assert ucc_program.num_detectors == stim_circuit.num_detectors
        assert ucc_program.num_observables == stim_circuit.num_observables

    def test_marginal_probabilities_within_bounds(
        self, ucc_program: ucc.Program, stim_sampler: stim.CompiledDetectorSampler
    ) -> None:
        """All detector and observable marginals match within 5-sigma bounds.

        This is the core statistical equivalence test. We run 100k shots in
        both engines and verify that every detector and observable has a
        marginal firing probability that matches within strict statistical
        tolerance.
        """
        shots = 100_000
        seed = 12345

        # Sample from both engines
        _, ucc_det, ucc_obs = ucc.sample(ucc_program, shots, seed=seed)
        stim_det, stim_obs = stim_sampler.sample(shots, separate_observables=True)

        # Compute marginal probabilities
        ucc_det_probs = ucc_det.astype(float).mean(axis=0)
        stim_det_probs = stim_det.astype(float).mean(axis=0)
        ucc_obs_probs = ucc_obs.astype(float).mean(axis=0)
        stim_obs_probs = stim_obs.astype(float).mean(axis=0)

        # Check all detectors
        for i in range(len(ucc_det_probs)):
            p_est = (ucc_det_probs[i] + stim_det_probs[i]) / 2
            tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_det_probs[i] - stim_det_probs[i])
            assert diff < tol, (
                f"Detector {i} marginal mismatch: "
                f"UCC={ucc_det_probs[i]:.5f} Stim={stim_det_probs[i]:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )

        # Check all observables
        for i in range(len(ucc_obs_probs)):
            p_est = (ucc_obs_probs[i] + stim_obs_probs[i]) / 2
            tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_obs_probs[i] - stim_obs_probs[i])
            assert diff < tol, (
                f"Observable {i} marginal mismatch: "
                f"UCC={ucc_obs_probs[i]:.5f} Stim={stim_obs_probs[i]:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )


class TestSimpleCircuitEquivalence:
    """Quick statistical checks on simpler circuits."""

    def test_bell_state_with_noise(self) -> None:
        """Bell state with depolarizing noise matches Stim."""
        circuit = """
            H 0
            DEPOLARIZE1(0.01) 0
            CX 0 1
            DEPOLARIZE2(0.01) 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 10_000

        prog = ucc.compile(circuit)
        stim_circuit = stim.Circuit(circuit)
        stim_sampler = stim_circuit.compile_detector_sampler(seed=99)

        _, ucc_det, _ = ucc.sample(prog, shots, seed=99)
        stim_det, _ = stim_sampler.sample(shots, separate_observables=True)

        ucc_rate = float(ucc_det.astype(float).mean())
        stim_rate = float(stim_det.astype(float).mean())

        tol = cross_binomial_tolerance((ucc_rate + stim_rate) / 2, shots)
        assert abs(ucc_rate - stim_rate) < tol

    def test_repeated_measurements_with_readout_noise(self) -> None:
        """Repeated measurements with readout noise match Stim."""
        circuit = """
            M(0.05) 0
            M(0.05) 0
            DETECTOR rec[-1] rec[-2]
        """
        shots = 10_000

        prog = ucc.compile(circuit)
        stim_circuit = stim.Circuit(circuit)
        stim_sampler = stim_circuit.compile_detector_sampler(seed=123)

        _, ucc_det, _ = ucc.sample(prog, shots, seed=123)
        stim_det, _ = stim_sampler.sample(shots, separate_observables=True)

        # Detector fires when readout noise causes disagreement
        # Expected rate: 2 * 0.05 * 0.95 ~ 0.095 (one or the other flips)
        ucc_rate = float(ucc_det.astype(float).mean())
        stim_rate = float(stim_det.astype(float).mean())

        tol = cross_binomial_tolerance((ucc_rate + stim_rate) / 2, shots)
        assert abs(ucc_rate - stim_rate) < tol

    def test_stabilizer_round_with_reset(self) -> None:
        """Circuit with resets has correct detector behavior."""
        circuit = """
            R 0
            H 0
            CX 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        """
        shots = 1_000

        prog = ucc.compile(circuit)
        stim_circuit = stim.Circuit(circuit)
        stim_sampler = stim_circuit.compile_detector_sampler(seed=42)

        _, ucc_det, _ = ucc.sample(prog, shots, seed=42)
        stim_det, _ = stim_sampler.sample(shots, separate_observables=True)

        # Clean Bell state: detector should always be 0
        assert np.all(ucc_det == 0), "UCC: Clean Bell detector should be 0"
        assert np.all(stim_det == 0), "Stim: Clean Bell detector should be 0"


class TestTopologicalQECCodes:
    """Statistical equivalence tests using Stim's generated QEC circuits.

    Tests different lattice connectivities using stim.Circuit.generated().
    Color code is skipped because it uses unsupported gates (C_XYZ).
    """

    @pytest.mark.parametrize(
        "code_task",
        [
            "repetition_code:memory",
            "surface_code:rotated_memory_x",
            # "color_code:memory_xyz" - uses unsupported C_XYZ gate
        ],
    )
    def test_qec_code_statistical_equivalence(self, code_task: str) -> None:
        """Generated QEC circuit matches Stim within statistical bounds.

        Uses distance=3, rounds=2, and after_clifford_depolarization=0.01.
        Runs 10,000 shots and verifies detector and observable marginals.
        """
        shots = 10_000
        seed = 42

        # Generate circuit from Stim
        stim_circuit = stim.Circuit.generated(
            code_task,
            distance=3,
            rounds=2,
            after_clifford_depolarization=0.01,
        )
        circuit_str = str(stim_circuit)

        # Compile both
        ucc_prog = ucc.compile(circuit_str)
        stim_sampler = stim_circuit.compile_detector_sampler(seed=seed)

        # Verify metadata matches
        assert ucc_prog.num_detectors == stim_circuit.num_detectors
        assert ucc_prog.num_observables == stim_circuit.num_observables

        # Sample from both
        _, ucc_det, ucc_obs = ucc.sample(ucc_prog, shots, seed=seed)
        stim_det, stim_obs = stim_sampler.sample(shots, separate_observables=True)

        # Check detector marginals
        ucc_det_probs = ucc_det.astype(float).mean(axis=0)
        stim_det_probs = stim_det.astype(float).mean(axis=0)

        for i in range(len(ucc_det_probs)):
            p_est = (ucc_det_probs[i] + stim_det_probs[i]) / 2
            tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_det_probs[i] - stim_det_probs[i])
            assert diff < tol, (
                f"{code_task} detector {i} mismatch: "
                f"UCC={ucc_det_probs[i]:.5f} Stim={stim_det_probs[i]:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )

        # Check observable marginals
        ucc_obs_probs = ucc_obs.astype(float).mean(axis=0)
        stim_obs_probs = stim_obs.astype(float).mean(axis=0)

        for i in range(len(ucc_obs_probs)):
            p_est = (ucc_obs_probs[i] + stim_obs_probs[i]) / 2
            tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_obs_probs[i] - stim_obs_probs[i])
            assert diff < tol, (
                f"{code_task} observable {i} mismatch: "
                f"UCC={ucc_obs_probs[i]:.5f} Stim={stim_obs_probs[i]:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )


def _generate_random_noisy_circuit(num_qubits: int, depth_per_qubit: int, seed: int) -> str:
    """Generate a random noisy circuit without detectors.

    Emits random H, S, CX, R, RX, M gates with X_ERROR, DEPOLARIZE1, DEPOLARIZE2
    noise at p=0.05. Does not emit DETECTORs so Stim won't reject it.

    Args:
        num_qubits: Number of qubits in the circuit
        depth_per_qubit: Approximate number of gates per qubit
        seed: Random seed for reproducibility

    Returns:
        Circuit string compatible with both UCC and Stim
    """
    rng = np.random.default_rng(seed)
    lines: list[str] = []
    total_gates = num_qubits * depth_per_qubit
    noise_prob = 0.05

    # Gate types: single-qubit, two-qubit, reset, measurement
    single_gates = ["H", "S"]
    two_qubit_gates = ["CX"]
    reset_gates = ["R", "RX"]

    for _ in range(total_gates):
        gate_type = rng.choice(["single", "two", "reset", "measure"], p=[0.35, 0.25, 0.15, 0.25])

        if gate_type == "single":
            gate = rng.choice(single_gates)
            q = rng.integers(0, num_qubits)
            lines.append(f"{gate} {q}")
            # Add single-qubit noise
            if rng.random() < 0.5:
                noise = rng.choice(["X_ERROR", "DEPOLARIZE1"])
                lines.append(f"{noise}({noise_prob}) {q}")

        elif gate_type == "two" and num_qubits >= 2:
            gate = rng.choice(two_qubit_gates)
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"{gate} {q1} {q2}")
            # Add two-qubit noise
            if rng.random() < 0.5:
                lines.append(f"DEPOLARIZE2({noise_prob}) {q1} {q2}")

        elif gate_type == "reset":
            gate = rng.choice(reset_gates)
            q = rng.integers(0, num_qubits)
            lines.append(f"{gate} {q}")

        elif gate_type == "measure":
            q = rng.integers(0, num_qubits)
            lines.append(f"M {q}")

    return "\n".join(lines)


class TestUnstructuredNoiseFuzzing:
    """Statistical equivalence tests on random noisy circuits.

    Tests chaotic topologies with random gates and noise to validate
    that UCC correctly tracks both local basis states and non-local
    entanglement correlations.
    """

    @pytest.mark.parametrize("num_qubits", [2, 4, 6])
    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1337])
    def test_random_noisy_circuit(self, num_qubits: int, seed: int) -> None:
        """Random noisy circuit marginals match Stim.

        Compares both 1-body marginals (per-measurement probabilities)
        and adjacent 2-body parities (XOR of consecutive measurements)
        to validate joint correlations.
        """
        shots = 50_000
        depth_per_qubit = 12

        circuit_str = _generate_random_noisy_circuit(num_qubits, depth_per_qubit, seed)

        # Compile both
        try:
            ucc_prog = ucc.compile(circuit_str)
        except Exception as e:
            pytest.fail(f"UCC compilation failed: {e}\nCircuit:\n{circuit_str}")

        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=seed)

        # Sample from both
        ucc_meas, _, _ = ucc.sample(ucc_prog, shots, seed=seed)
        stim_meas = stim_sampler.sample(shots)

        # Skip if no measurements were generated
        if ucc_meas.shape[1] == 0:
            pytest.skip("No measurements in generated circuit")

        # 1-body marginals: P(measurement_i = 1)
        ucc_p1 = ucc_meas.astype(float).mean(axis=0)
        stim_p1 = stim_meas.astype(float).mean(axis=0)

        for i in range(len(ucc_p1)):
            p_est = (ucc_p1[i] + stim_p1[i]) / 2
            tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_p1[i] - stim_p1[i])
            assert diff < tol, (
                f"1-body marginal {i} mismatch (qubits={num_qubits}, seed={seed}): "
                f"UCC={ucc_p1[i]:.5f} Stim={stim_p1[i]:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )

        # Adjacent 2-body parities: P(meas[i] XOR meas[i+1] = 1)
        if ucc_meas.shape[1] >= 2:
            ucc_parity = ucc_meas[:, :-1] ^ ucc_meas[:, 1:]
            stim_parity = stim_meas[:, :-1] ^ stim_meas[:, 1:]

            ucc_p2 = ucc_parity.astype(float).mean(axis=0)
            stim_p2 = stim_parity.astype(float).mean(axis=0)

            for i in range(len(ucc_p2)):
                p_est = (ucc_p2[i] + stim_p2[i]) / 2
                tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
                diff = abs(ucc_p2[i] - stim_p2[i])
                assert diff < tol, (
                    f"2-body parity {i},{i + 1} mismatch (qubits={num_qubits}, seed={seed}): "
                    f"UCC={ucc_p2[i]:.5f} Stim={stim_p2[i]:.5f} "
                    f"diff={diff:.6f} > tol={tol:.6f}"
                )


def _generate_midcircuit_clifford_circuit(num_qubits: int, depth: int, seed: int) -> str:
    """Generate a Clifford circuit with mid-circuit measurements and resets.

    Produces circuits that interleave entangling Clifford gates with
    mid-circuit measurements and resets, forcing the compiler to handle
    collapse events mid-stream. Uses only Clifford gates so the output
    is verifiable against Stim.

    Args:
        num_qubits: Number of qubits.
        depth: Approximate number of gate layers.
        seed: Random seed.

    Returns:
        Circuit string in .stim format (no detectors/observables).
    """
    rng = np.random.default_rng(seed)
    lines: list[str] = []
    gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z"]
    gates_2q = ["CX", "CY", "CZ"]
    noise_prob = 0.02

    for _ in range(depth):
        r = rng.random()
        if r < 0.35:
            # 1-qubit gate
            gate = rng.choice(gates_1q)
            q = rng.integers(0, num_qubits)
            lines.append(f"{gate} {q}")
        elif r < 0.6 and num_qubits > 1:
            # 2-qubit gate with optional noise
            gate = rng.choice(gates_2q)
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"{gate} {q1} {q2}")
            if rng.random() < 0.3:
                lines.append(f"DEPOLARIZE2({noise_prob}) {q1} {q2}")
        elif r < 0.8:
            # Mid-circuit measurement
            q = rng.integers(0, num_qubits)
            lines.append(f"M {q}")
        else:
            # Reset + re-prepare
            q = rng.integers(0, num_qubits)
            lines.append(f"R {q}")
            if rng.random() < 0.5:
                lines.append(f"H {q}")

    # Final measurement of all qubits
    all_q = " ".join(str(i) for i in range(num_qubits))
    lines.append(f"M {all_q}")
    return "\n".join(lines)


class TestMidCircuitMeasurementEvolution:
    """Validate mid-circuit measurement + reset + Clifford evolution.

    These circuits interleave collapse events with entangling Clifford
    gates, testing that the factored state correctly re-evolves after
    measurement and reset operations.
    """

    @pytest.mark.parametrize("num_qubits", [2, 3, 4])
    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_midcircuit_marginals(self, num_qubits: int, seed: int) -> None:
        """Mid-circuit measurement circuits match Stim on 1-body marginals."""
        shots = 50_000
        circuit_str = _generate_midcircuit_clifford_circuit(num_qubits, depth=30, seed=seed)

        ucc_prog = ucc.compile(circuit_str)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=seed)

        ucc_meas, _, _ = ucc.sample(ucc_prog, shots, seed=seed)
        stim_meas = stim_sampler.sample(shots)

        if ucc_meas.shape[1] == 0:
            pytest.skip("No measurements in generated circuit")

        ucc_p = ucc_meas.astype(float).mean(axis=0)
        stim_p = stim_meas.astype(float).mean(axis=0)

        for i in range(len(ucc_p)):
            p_est = (ucc_p[i] + stim_p[i]) / 2
            tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_p[i] - stim_p[i])
            assert diff < tol, (
                f"Marginal {i} mismatch (q={num_qubits}, seed={seed}): "
                f"UCC={ucc_p[i]:.5f} Stim={stim_p[i]:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )

    @pytest.mark.parametrize("num_qubits", [3, 4])
    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_midcircuit_3body_parity(self, num_qubits: int, seed: int) -> None:
        """3-body parity checks on mid-circuit measurement circuits.

        Validates triple-measurement XOR correlations to catch phase
        or entanglement tracking errors invisible to marginal checks.
        """
        shots = 50_000
        circuit_str = _generate_midcircuit_clifford_circuit(num_qubits, depth=30, seed=seed)

        ucc_prog = ucc.compile(circuit_str)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=seed)

        ucc_meas, _, _ = ucc.sample(ucc_prog, shots, seed=seed)
        stim_meas = stim_sampler.sample(shots)

        n_meas = ucc_meas.shape[1]
        if n_meas < 3:
            pytest.skip("Fewer than 3 measurements")

        # Check consecutive 3-body parities: meas[i] XOR meas[i+1] XOR meas[i+2]
        for i in range(n_meas - 2):
            ucc_par = (ucc_meas[:, i] ^ ucc_meas[:, i + 1] ^ ucc_meas[:, i + 2]).astype(float)
            stim_par = (stim_meas[:, i] ^ stim_meas[:, i + 1] ^ stim_meas[:, i + 2]).astype(float)
            ucc_rate = float(ucc_par.mean())
            stim_rate = float(stim_par.mean())
            p_est = (ucc_rate + stim_rate) / 2
            tol = cross_binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_rate - stim_rate)
            assert diff < tol, (
                f"3-body parity {i},{i + 1},{i + 2} mismatch "
                f"(q={num_qubits}, seed={seed}): "
                f"UCC={ucc_rate:.5f} Stim={stim_rate:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance between two probability distributions."""
    return float(0.5 * np.sum(np.abs(p - q)))


def _counts_to_distribution(samples: np.ndarray) -> np.ndarray:
    """Convert binary measurement matrix to a probability distribution.

    Maps each row to an integer bitstring index, then computes the
    empirical probability of each outcome.

    Args:
        samples: Boolean array of shape (shots, n_meas).

    Returns:
        Probability array of length 2^n_meas.
    """
    n_meas = samples.shape[1]
    n_outcomes = 1 << n_meas
    # Convert each row to integer index
    powers = 1 << np.arange(n_meas)
    indices = samples.astype(int) @ powers
    counts = np.bincount(indices, minlength=n_outcomes)
    dist: np.ndarray = counts / counts.sum()
    return dist


class TestTVDSmallMeasurementSpace:
    """Full distribution TVD checks for circuits with few measurements.

    When the measurement count is small enough (m <= 8), we can enumerate
    all 2^m outcomes and compare the full empirical distributions via TVD.
    The threshold is TVD < C * sqrt(k_eff / N) where C ~= 4 and
    k_eff = 2^m - 1.
    """

    TVD_C = 4.0  # conservative threshold multiplier

    @staticmethod
    def _generate_small_measurement_circuit(num_qubits: int, n_measurements: int, seed: int) -> str:
        """Generate a circuit with exactly n_measurements measurement ops.

        Interleaves entangling gates, T gates, and noise with a controlled
        number of mid-circuit + final measurements.
        """
        rng = np.random.default_rng(seed)
        lines: list[str] = []
        gates_1q = ["H", "S", "S_DAG"]
        gates_2q = ["CX", "CY", "CZ"]
        meas_emitted = 0

        # Emit some gates then a measurement, repeat
        while meas_emitted < n_measurements:
            # 3-8 gates before each measurement
            n_gates = rng.integers(3, 9)
            for _ in range(n_gates):
                if num_qubits > 1 and rng.random() < 0.4:
                    gate = rng.choice(gates_2q)
                    q1, q2 = rng.choice(num_qubits, size=2, replace=False)
                    lines.append(f"{gate} {q1} {q2}")
                    if rng.random() < 0.3:
                        lines.append(f"DEPOLARIZE2(0.02) {q1} {q2}")
                else:
                    gate = rng.choice(gates_1q)
                    q = rng.integers(0, num_qubits)
                    lines.append(f"{gate} {q}")
            # Measure a qubit
            q = rng.integers(0, num_qubits)
            lines.append(f"M {q}")
            meas_emitted += 1
            # Sometimes reset after measurement
            if rng.random() < 0.3:
                lines.append(f"R {q}")

        return "\n".join(lines)

    @pytest.mark.parametrize("n_measurements", [4, 6, 8])
    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_tvd_small_circuits(self, n_measurements: int, seed: int) -> None:
        """TVD of full distribution is below statistical threshold."""
        num_qubits = 3
        shots = 100_000

        circuit_str = self._generate_small_measurement_circuit(num_qubits, n_measurements, seed)

        ucc_prog = ucc.compile(circuit_str)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=seed)

        ucc_meas, _, _ = ucc.sample(ucc_prog, shots, seed=seed)
        stim_meas = stim_sampler.sample(shots)

        assert (
            ucc_meas.shape[1] == n_measurements
        ), f"Expected {n_measurements} measurements, got {ucc_meas.shape[1]}"

        ucc_dist = _counts_to_distribution(ucc_meas)
        stim_dist = _counts_to_distribution(stim_meas)

        tvd = _tvd(ucc_dist, stim_dist)
        k_eff = (1 << n_measurements) - 1
        threshold = self.TVD_C * np.sqrt(k_eff / shots)

        assert tvd < threshold, (
            f"TVD={tvd:.6f} >= threshold={threshold:.6f} "
            f"(m={n_measurements}, k_eff={k_eff}, N={shots}, seed={seed})"
        )
