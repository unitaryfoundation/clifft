"""Statistical Equivalence Tests - Bulk Distribution Validation.

This module validates that UCC's AOT noise scheduling, multi-Pauli
depolarizing decomposition, readout noise, and classical logic pipelines
produce statistically equivalent distributions to Stim.

Tests run large numbers of shots and verify that marginal firing
probabilities of all detectors and observables fall within strict
statistical bounds.
"""

import numpy as np
import pytest
import stim

import ucc

# CSS d=3 cultivation circuit for statistical equivalence testing.
#
# Source:
#     Gidney, C., Jones, C., & Shutty, N. (2024).
#     Data for "Magic state cultivation: growing T states as cheap as CNOT gates".
#     Zenodo. https://doi.org/10.5281/zenodo.13777072
#
# License: Apache 2.0
#
# Circuit parameters:
#     - c=inject[bell]+cultivate: Bell state injection with cultivation
#     - p=0.001: Physical error rate
#     - noise=uniform: Uniform depolarizing noise model
#     - g=css: CSS stabilizer code
#     - q=14: 14 qubits
#     - b=Y: Y-basis logical measurement
#     - r=5: 5 stabilizer rounds
#     - d1=3: Distance 3 code
#
# fmt: off
CSS_D3_CIRCUIT = """\
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(0, 1) 1
QUBIT_COORDS(1, 0) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(1, 2) 4
QUBIT_COORDS(2, 0) 5
QUBIT_COORDS(2, 1) 6
QUBIT_COORDS(2, 2) 7
QUBIT_COORDS(2, 3) 8
QUBIT_COORDS(3, 0) 9
QUBIT_COORDS(3, 1) 10
QUBIT_COORDS(3, 2) 11
QUBIT_COORDS(4, 0) 12
QUBIT_COORDS(4, 1) 13
RX 0 4 10 9 2
R 7 3 8 6 11 12 5
X_ERROR(0.001) 7 3 8 6 11 12 5
Z_ERROR(0.001) 0 4 10 9 2
DEPOLARIZE1(0.001) 1 13
TICK
CX 2 5 4 7 9 12 10 6
DEPOLARIZE2(0.001) 2 5 4 7 9 12 10 6
DEPOLARIZE1(0.001) 0 1 3 8 11 13
TICK
S_DAG 0
DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13
TICK
CX 0 2 4 3 7 8 9 5 10 11
DEPOLARIZE2(0.001) 0 2 4 3 7 8 9 5 10 11
DEPOLARIZE1(0.001) 1 6 12 13
TICK
CX 3 2 6 5 8 7 11 10
DEPOLARIZE2(0.001) 3 2 6 5 8 7 11 10
DEPOLARIZE1(0.001) 0 1 4 9 12 13
TICK
CX 2 0 3 4 5 9
DEPOLARIZE2(0.001) 2 0 3 4 5 9
DEPOLARIZE1(0.001) 1 6 7 8 10 11 12 13
TICK
CX 2 3 5 6
DEPOLARIZE2(0.001) 2 3 5 6
DEPOLARIZE1(0.001) 0 1 4 7 8 9 10 11 12 13
TICK
CX 2 5
DEPOLARIZE2(0.001) 2 5
DEPOLARIZE1(0.001) 0 1 3 4 6 7 8 9 10 11 12 13
TICK
M(0.001) 4 7 10 5
MX(0.001) 2
CX rec[-2] 0 rec[-2] 8
DETECTOR(1, 2, 0) rec[-5]
DETECTOR(2, 2, 0) rec[-4]
DETECTOR(3, 1, 0) rec[-3]
SHIFT_COORDS(0, 0, 1)
DEPOLARIZE1(0.001) 4 7 10 5 2 0 1 3 6 8 9 11 12 13
TICK
RX 2 4 10
R 5 7 13
X_ERROR(0.001) 5 7 13
Z_ERROR(0.001) 2 4 10
DEPOLARIZE1(0.001) 0 1 3 6 8 9 11 12
TICK
CX 2 5 4 7 10 13
DEPOLARIZE2(0.001) 2 5 4 7 10 13
DEPOLARIZE1(0.001) 0 1 3 6 8 9 11 12
TICK
CX 2 0 5 9 7 11 10 6
DEPOLARIZE2(0.001) 2 0 5 9 7 11 10 6
DEPOLARIZE1(0.001) 1 3 4 8 12 13
TICK
CX 2 3 5 6 7 8 10 11
DEPOLARIZE2(0.001) 2 3 5 6 7 8 10 11
DEPOLARIZE1(0.001) 0 1 4 9 12 13
TICK
CX 4 3 7 6 10 9 13 12
DEPOLARIZE2(0.001) 4 3 7 6 10 9 13 12
DEPOLARIZE1(0.001) 0 1 2 5 8 11
TICK
CX 0 2 6 10 9 5 11 7
DEPOLARIZE2(0.001) 0 2 6 10 9 5 11 7
DEPOLARIZE1(0.001) 1 3 4 8 12 13
TICK
CX 3 2 6 5 8 7 11 10
DEPOLARIZE2(0.001) 3 2 6 5 8 7 11 10
DEPOLARIZE1(0.001) 0 1 4 9 12 13
TICK
CX 3 4 6 7 9 10 12 13
DEPOLARIZE2(0.001) 3 4 6 7 9 10 12 13
DEPOLARIZE1(0.001) 0 1 2 5 8 11
TICK
CX 2 5 4 7 10 13
DEPOLARIZE2(0.001) 2 5 4 7 10 13
DEPOLARIZE1(0.001) 0 1 3 6 8 9 11 12
TICK
MX(0.001) 2 4 10
M(0.001) 5 7 13
DETECTOR(1.25, 0.25, 0, -1, -9) rec[-7] rec[-6]
DETECTOR(1.5, 1.875, 0, -1, -9) rec[-5]
DETECTOR(1.75, 0.25, 0, -1, -9) rec[-3]
DETECTOR(2, 1.875, 0, -1, -9) rec[-2]
DETECTOR(3, 0.875, 0, -1, -9) rec[-4]
DETECTOR(3.5, 0.875, 0, -1, -9) rec[-1]
SHIFT_COORDS(0, 0, 1)
DEPOLARIZE1(0.001) 2 4 10 5 7 13 0 1 3 6 8 9 11 12
TICK
RX 13 10 5 2 7 1
Z_ERROR(0.001) 13 10 5 2 7 1
DEPOLARIZE1(0.001) 0 3 4 6 8 9 11 12
TICK
S_DAG 0 3 6 8 9 11 12
DEPOLARIZE1(0.001) 0 3 6 8 9 11 12 1 2 4 5 7 10 13
TICK
CX 1 0 2 3 5 6 7 8 10 9 13 12
DEPOLARIZE2(0.001) 1 0 2 3 5 6 7 8 10 9 13 12
DEPOLARIZE1(0.001) 4 11
TICK
CX 3 1 6 7 10 13
DEPOLARIZE2(0.001) 3 1 6 7 10 13
DEPOLARIZE1(0.001) 0 2 4 5 8 9 11 12
TICK
CX 6 3 10 11
DEPOLARIZE2(0.001) 6 3 10 11
DEPOLARIZE1(0.001) 0 1 2 4 5 7 8 9 12 13
TICK
CX 6 10
DEPOLARIZE2(0.001) 6 10
DEPOLARIZE1(0.001) 0 1 2 3 4 5 7 8 9 11 12 13
TICK
MX(0.001) 6
DEPOLARIZE1(0.001) 6 0 1 2 3 4 5 7 8 9 10 11 12 13
TICK
RX 6
Z_ERROR(0.001) 6
DEPOLARIZE1(0.001) 0 1 2 3 4 5 7 8 9 10 11 12 13
TICK
CX 6 10
DEPOLARIZE2(0.001) 6 10
DEPOLARIZE1(0.001) 0 1 2 3 4 5 7 8 9 11 12 13
TICK
CX 6 3 10 11
DEPOLARIZE2(0.001) 6 3 10 11
DEPOLARIZE1(0.001) 0 1 2 4 5 7 8 9 12 13
TICK
CX 3 1 6 7 10 13
DEPOLARIZE2(0.001) 3 1 6 7 10 13
DEPOLARIZE1(0.001) 0 2 4 5 8 9 11 12
TICK
CX 1 0 2 3 5 6 7 8 10 9 13 12
DEPOLARIZE2(0.001) 1 0 2 3 5 6 7 8 10 9 13 12
DEPOLARIZE1(0.001) 4 11
TICK
S 0 3 6 8 9 11 12
DEPOLARIZE1(0.001) 0 3 6 8 9 11 12 1 2 4 5 7 10 13
TICK
MX(0.001) 13 10 5 2 7 1
OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-2] rec[-1]
DETECTOR(1.60714, 0.75, 0, -1, -9) rec[-12] rec[-11] rec[-7]
DETECTOR(4, 1, 1) rec[-6]
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-7]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
SHIFT_COORDS(0, 0, 3)
DEPOLARIZE1(0.001) 13 10 5 2 7 1 0 3 4 6 8 9 11 12
TICK
MPP Y0*Y3*Y6*Y8*Y9*Y11*Y12 X0*X3*X6*X9 Z0*Z3*Z6*Z9 X3*X6*X8*X11 Z3*Z6*Z8*Z11 X6*X9*X11*X12 Z6*Z9*Z11*Z12
OBSERVABLE_INCLUDE(0) rec[-7]
DETECTOR(0.625, 0.125, 0, -1, -9) rec[-20] rec[-19] rec[-14] rec[-6]
DETECTOR(0.875, 0.125, 0, -1, -9) rec[-17] rec[-5]
DETECTOR(1.25, 1.4375, 0, -1, -9) rec[-20] rec[-14] rec[-4]
DETECTOR(1.5, 1.4375, 0, -1, -9) rec[-16] rec[-3]
DETECTOR(2.5, 0.9375, 0, -1, -9) rec[-14] rec[-2]
DETECTOR(2.75, 0.9375, 0, -1, -9) rec[-15] rec[-1]
"""
# fmt: on


def binomial_tolerance(p: float, n: int, *, sigma: float = 5.0) -> float:
    """Compute tolerance for binomial proportion estimate.

    Returns sigma standard deviations of the binomial standard error.
    Default 5σ gives <1 in 3.5 million false positive rate per assertion.

    Args:
        p: Expected probability (0 < p < 1)
        n: Number of samples (shots)
        sigma: Number of standard deviations for the bound

    Returns:
        Tolerance value such that |observed - p| < tolerance with high probability
    """
    # Clamp p to avoid zero variance for p=0 or p=1
    p_clamped = max(min(p, 0.99), 0.01)
    std_err = float(np.sqrt((p_clamped * (1 - p_clamped)) / n))
    return sigma * std_err


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
        """Return the target circuit."""
        return CSS_D3_CIRCUIT

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
            tol = binomial_tolerance(p_est, shots, sigma=5.0)
            diff = abs(ucc_det_probs[i] - stim_det_probs[i])
            assert diff < tol, (
                f"Detector {i} marginal mismatch: "
                f"UCC={ucc_det_probs[i]:.5f} Stim={stim_det_probs[i]:.5f} "
                f"diff={diff:.6f} > tol={tol:.6f}"
            )

        # Check all observables
        for i in range(len(ucc_obs_probs)):
            p_est = (ucc_obs_probs[i] + stim_obs_probs[i]) / 2
            tol = binomial_tolerance(p_est, shots, sigma=5.0)
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

        tol = binomial_tolerance((ucc_rate + stim_rate) / 2, shots)
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
        # Expected rate: 2 * 0.05 * 0.95 ≈ 0.095 (one or the other flips)
        ucc_rate = float(ucc_det.astype(float).mean())
        stim_rate = float(stim_det.astype(float).mean())

        tol = binomial_tolerance((ucc_rate + stim_rate) / 2, shots)
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
            tol = binomial_tolerance(p_est, shots, sigma=5.0)
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
            tol = binomial_tolerance(p_est, shots, sigma=5.0)
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
        shots = 10_000
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
            tol = binomial_tolerance(p_est, shots, sigma=5.0)
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
                tol = binomial_tolerance(p_est, shots, sigma=5.0)
                diff = abs(ucc_p2[i] - stim_p2[i])
                assert diff < tol, (
                    f"2-body parity {i},{i + 1} mismatch (qubits={num_qubits}, seed={seed}): "
                    f"UCC={ucc_p2[i]:.5f} Stim={stim_p2[i]:.5f} "
                    f"diff={diff:.6f} > tol={tol:.6f}"
                )
