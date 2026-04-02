"""Python integration tests for ucc.compile and ucc.sample."""

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equal,
    binomial_tolerance,
    random_clifford_circuit,
)

import ucc


class TestCompile:
    """Tests for ucc.compile()."""

    def test_compile_simple(self) -> None:
        """Compile a simple circuit."""
        prog = ucc.compile("H 0\nT 0\nM 0")
        assert prog.peak_rank == 1
        assert prog.num_measurements == 1
        assert prog.num_instructions >= 1

    def test_compile_pure_clifford(self) -> None:
        """Pure Clifford circuit has peak_rank 0."""
        prog = ucc.compile("H 0\nCX 0 1\nM 0\nM 1")
        assert prog.peak_rank == 0
        assert prog.num_measurements == 2

    def test_compile_multiple_t_gates(self) -> None:
        """Multiple independent T gates increase peak_rank."""
        prog = ucc.compile("""
            H 0
            H 1
            T 0
            T 1
        """)
        assert prog.peak_rank == 2


class TestSample:
    """Tests for ucc.sample()."""

    def test_sample_deterministic_zero(self) -> None:
        """Measurement of |0> always gives 0."""
        prog = ucc.compile("M 0")
        result = ucc.sample(prog, 100, seed=42)
        assert np.all(result.measurements[:, 0] == 0)

    def test_sample_deterministic_one(self) -> None:
        """Measurement of |1> always gives 1."""
        prog = ucc.compile("X 0\nM 0")
        result = ucc.sample(prog, 100, seed=42)
        assert np.all(result.measurements[:, 0] == 1)

    def test_sample_superposition(self) -> None:
        """|+> state gives roughly 50/50 distribution."""
        prog = ucc.compile("H 0\nM 0")
        shots = 1000
        result = ucc.sample(prog, shots, seed=42)
        p0 = float(np.mean(result.measurements[:, 0] == 0))
        p1 = float(np.mean(result.measurements[:, 0] == 1))
        tolerance = binomial_tolerance(0.5, shots)
        assert abs(p0 - 0.5) < tolerance, f"p(0)={p0} outside {tolerance:.3f} tolerance"
        assert abs(p1 - 0.5) < tolerance, f"p(1)={p1} outside {tolerance:.3f} tolerance"

    def test_sample_bell_state_correlated(self) -> None:
        """Bell state measurements are always correlated."""
        prog = ucc.compile("""
            H 0
            CX 0 1
            M 0
            M 1
        """)
        result = ucc.sample(prog, 500, seed=99)
        assert np.all(
            result.measurements[:, 0] == result.measurements[:, 1]
        ), "Bell state not correlated"

    def test_sample_reproducible(self) -> None:
        """Same seed produces same results."""
        prog = ucc.compile("H 0\nM 0")
        result1 = ucc.sample(prog, 100, seed=12345)
        result2 = ucc.sample(prog, 100, seed=12345)
        assert np.array_equal(result1.measurements, result2.measurements)

    def test_sample_different_seeds(self) -> None:
        """Different seeds produce different results."""
        prog = ucc.compile("H 0\nM 0")
        result1 = ucc.sample(prog, 100, seed=1)
        result2 = ucc.sample(prog, 100, seed=2)
        # With 100 random bits, probability of match is 2^-100
        assert not np.array_equal(result1.measurements, result2.measurements)

    def test_sample_shape(self) -> None:
        """Results have correct shape and type."""
        prog = ucc.compile("H 0\nM 0\nH 1\nM 1")
        result = ucc.sample(prog, 50, seed=0)
        assert isinstance(result.measurements, np.ndarray)
        assert result.measurements.dtype == np.uint8
        assert result.measurements.shape == (50, 2)
        # No detectors/observables in this circuit
        assert result.detectors.shape == (50, 0)
        assert result.observables.shape == (50, 0)

    def test_sample_reset_works(self) -> None:
        """Reset correctly resets to |0>."""
        prog = ucc.compile("""
            X 0
            R 0
            M 0
        """)
        result = ucc.sample(prog, 100, seed=42)
        # Only one visible measurement (from M 0, after reset)
        # R's internal measurement is hidden, matching Stim behavior
        assert result.measurements.shape == (
            100,
            1,
        ), f"Expected 1 visible measurement, got {result.measurements.shape}"
        # Measurement after reset should always be 0
        assert np.all(result.measurements[:, 0] == 0), "Reset failed"

    def test_sample_mr_visible(self) -> None:
        """MR (measure-and-reset) produces visible measurement unlike R."""
        # R produces 0 visible measurements, MR produces 1
        prog_r = ucc.compile("R 0")
        prog_mr = ucc.compile("MR 0")

        assert prog_r.num_measurements == 0, "R should have 0 visible measurements"
        assert prog_mr.num_measurements == 1, "MR should have 1 visible measurement"

        # MR on |0> should always measure 0
        result = ucc.sample(prog_mr, 100, seed=42)
        assert result.measurements.shape == (100, 1)
        assert np.all(result.measurements == 0), "MR on |0> should always measure 0"

        # MR after X should measure 1
        prog = ucc.compile("X 0\nMR 0")
        result = ucc.sample(prog, 100, seed=42)
        assert np.all(result.measurements == 1), "MR after X should measure 1"

    def test_gap_sampling_sparse_errors(self) -> None:
        """Verify geometric gap sampling correctly models independent errors."""
        # 50 qubits (within 64-qubit MVP limit), each has a 2% chance
        # of flipping. With linear sampling this is 50 RNG rolls.
        # With gap sampling, it's ~1 roll per shot.
        n_qubits = 50
        p = 0.02
        shots = 10000

        # Build circuit
        lines: list[str] = []
        for i in range(n_qubits):
            lines.append(f"X_ERROR({p}) {i}")
        for i in range(n_qubits):
            lines.append(f"M {i}")

        prog = ucc.compile("\n".join(lines))
        result = ucc.sample(prog, shots, seed=42)

        # 1. Overall error rate should be exactly p
        overall_rate = float(np.mean(result.measurements))
        tolerance = binomial_tolerance(p, n_qubits * shots, sigma=5.0)
        assert abs(overall_rate - p) < tolerance, f"Overall rate {overall_rate} off"

        # 2. Per-qubit error rate should be uniformly p across the array.
        # This catches bugs where the jump math favors early or late indices.
        per_qubit_rates = np.mean(result.measurements, axis=0)
        q_tol = binomial_tolerance(p, shots, sigma=5.0)
        for i, rate in enumerate(per_qubit_rates):
            assert abs(rate - p) < q_tol, f"Qubit {i} rate {rate} outside tolerance"

        # 3. Check for lack of artificial correlation (adjacent suppression).
        # The probability of (i and i+1) both being 1 should equal p^2.
        adjacent_both_1 = float(
            np.mean((result.measurements[:, :-1] == 1) & (result.measurements[:, 1:] == 1))
        )
        expected_adj = p * p
        adj_tol = binomial_tolerance(expected_adj, (n_qubits - 1) * shots, sigma=5.0)
        assert (
            abs(adjacent_both_1 - expected_adj) < adj_tol
        ), f"Adjacency correlation off: {adjacent_both_1}"


class TestStatevector:
    """Tests for ucc.get_statevector()."""

    def test_statevector_pure_clifford(self) -> None:
        """Pure Clifford circuit matches expected statevector."""
        # H|0> = |+> = [1/sqrt(2), 1/sqrt(2)]
        prog = ucc.compile("H 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_bell_state(self) -> None:
        """Bell state matches expected statevector."""
        prog = ucc.compile("H 0\nCX 0 1")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # |Phi+> = (|00> + |11>)/sqrt(2)
        expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_single_t_gate(self) -> None:
        """H-T circuit: [1/sqrt(2), e^{ipi/4}/sqrt(2)]."""
        prog = ucc.compile("H 0\nT 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        expected = np.array([1 / np.sqrt(2), np.exp(1j * np.pi / 4) / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_t_dagger(self) -> None:
        """H-T_dag circuit: [1/sqrt(2), e^{-ipi/4}/sqrt(2)]."""
        prog = ucc.compile("H 0\nT_DAG 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        expected = np.array([1 / np.sqrt(2), np.exp(-1j * np.pi / 4) / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_two_t_equals_s(self) -> None:
        """T-T = S: H-T-T should equal H-S."""
        prog = ucc.compile("H 0\nT 0\nT 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # H-S: [1/sqrt(2), i/sqrt(2)]
        expected = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_four_t_equals_z(self) -> None:
        """T^4 = Z: H-T-T-T-T should equal H-Z."""
        prog = ucc.compile("H 0\nT 0\nT 0\nT 0\nT 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # H-Z: [1/sqrt(2), -1/sqrt(2)]
        expected = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_t_on_zero(self) -> None:
        """T|0> = |0> (global phase only)."""
        prog = ucc.compile("T 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # T|0> = |0> up to global phase
        expected = np.array([1, 0], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_two_qubit_t(self) -> None:
        """Two-qubit circuit with T on qubit 0."""
        prog = ucc.compile("H 0\nH 1\nT 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # T on q0 affects indices where bit 0 is set (indices 1, 3)
        phase = np.exp(1j * np.pi / 4)
        expected = np.array([0.5, 0.5 * phase, 0.5, 0.5 * phase], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_bell_plus_t(self) -> None:
        """Bell state with T on control qubit."""
        prog = ucc.compile("H 0\nCX 0 1\nT 0")
        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # Bell state: (|00> + |11>)/sqrt(2)
        # T on q0: |00>->|00>, |11>->e^{ipi/4}|11>
        phase = np.exp(1j * np.pi / 4)
        expected = np.array([1 / np.sqrt(2), 0, 0, phase / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_normalized(self) -> None:
        """Statevector is always normalized."""
        circuits = [
            "H 0\nT 0",
            "H 0\nH 1\nT 0\nT 1",
            "H 0\nCX 0 1\nT 0",
            "H 0\nT 0\nT 0\nT 0",
        ]
        for circuit in circuits:
            prog = ucc.compile(circuit)
            state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
            ucc.execute(prog, state)
            sv = ucc.get_statevector(prog, state)
            norm = float(np.sqrt(np.sum(np.abs(sv) ** 2)))
            assert abs(norm - 1.0) < 1e-10, f"Not normalized: {circuit}"

    def test_statevector_after_measurement(self) -> None:
        """Statevector correctly handles active rank after measurement."""
        circuit = "H 0\nT 0\nM 0"
        prog = ucc.compile(circuit)

        state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements, seed=42)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # Statevector must be perfectly normalized (catches garbage memory bug)
        norm = float(np.sqrt(np.sum(np.abs(sv) ** 2)))
        assert abs(norm - 1.0) < 1e-10, f"Not normalized: {norm}"

        # The state must be cleanly collapsed AND match the measurement record
        outcome = state.meas_record[0]
        if outcome == 0:
            assert abs(abs(sv[0]) - 1.0) < 1e-5, f"Expected |0> but got {sv}"
            assert abs(sv[1]) < 1e-5, f"Expected no |1> component but got {sv}"
        else:
            assert abs(abs(sv[1]) - 1.0) < 1e-5, f"Expected |1> but got {sv}"
            assert abs(sv[0]) < 1e-5, f"Expected no |0> component but got {sv}"


class TestCliffordValidation:
    """Validate pure-Clifford statevectors against Stim."""

    def test_random_clifford_single_qubit(self) -> None:
        """Random 1-qubit Clifford circuits match Stim."""
        import stim

        for seed in range(10):
            circuit_str = random_clifford_circuit(1, 5, seed)

            # UCC statevector
            prog = ucc.compile(circuit_str)
            state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
            ucc.execute(prog, state)
            ucc_sv = ucc.get_statevector(prog, state)

            # Stim statevector
            stim_circuit = stim.Circuit(circuit_str)
            sim = stim.TableauSimulator()
            sim.do_circuit(stim_circuit)
            stim_sv = sim.state_vector(endian="little")

            assert_statevectors_equal(ucc_sv, stim_sv, msg=f"circuit:\n{circuit_str}")

    def test_random_clifford_multi_qubit(self) -> None:
        """Random 2-4 qubit Clifford circuits match Stim."""
        import stim

        for num_qubits in [2, 3, 4]:
            for seed in range(5):
                circuit_str = random_clifford_circuit(num_qubits, 10, seed)

                # UCC statevector
                prog = ucc.compile(circuit_str)
                state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
                ucc.execute(prog, state)
                ucc_sv = ucc.get_statevector(prog, state)

                # Stim statevector
                stim_circuit = stim.Circuit(circuit_str)
                sim = stim.TableauSimulator()
                sim.do_circuit(stim_circuit)
                stim_sv = sim.state_vector(endian="little")

                assert_statevectors_equal(
                    ucc_sv, stim_sv, msg=f"{num_qubits}q circuit:\n{circuit_str}"
                )


class TestSamplingValidation:
    """Validate sampling distributions against Stim."""

    def test_deterministic_clifford_sampling(self) -> None:
        """Deterministic measurements match Stim exactly."""
        import stim

        # Circuit where measurements have deterministic outcomes
        circuits = [
            "M 0",  # |0> always gives 0
            "X 0\nM 0",  # |1> always gives 1
            "H 0\nCX 0 1\nM 0\nM 1",  # Bell state: correlated
        ]

        for circuit_str in circuits:
            # UCC sampling
            prog = ucc.compile(circuit_str)
            result = ucc.sample(prog, 100, seed=42)

            # Stim sampling (seed is in compile_sampler, not sample)
            stim_circuit = stim.Circuit(circuit_str)
            stim_sampler = stim_circuit.compile_sampler(seed=42)
            stim_results = stim_sampler.sample(100)

            # For deterministic circuits, all shots should match
            # (Note: seeds may differ, but deterministic outcomes should be consistent)
            if prog.num_measurements == 1:
                # Single measurement: check value consistency
                ucc_vals = set(tuple(r) for r in result.measurements)
                stim_vals = set(tuple(r) for r in stim_results)
                assert ucc_vals == stim_vals, f"Mismatch for: {circuit_str}"
            else:
                # Multi-measurement: check correlation structure
                for ucc_shot in result.measurements:
                    if "CX" in circuit_str:  # Bell state
                        assert ucc_shot[0] == ucc_shot[1], "Bell correlation broken"

    def test_statistical_distribution_h(self) -> None:
        """H gate sampling matches Stim statistically."""
        import stim

        circuit_str = "H 0\nM 0"
        shots = 10000

        # UCC sampling
        prog = ucc.compile(circuit_str)
        result = ucc.sample(prog, shots, seed=12345)
        ucc_p0 = np.mean(result.measurements[:, 0] == 0)

        # Stim sampling (seed is in compile_sampler, not sample)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=54321)
        stim_results = stim_sampler.sample(shots)
        stim_p0 = np.mean(stim_results[:, 0] == 0)

        # Both should be close to 0.5, and close to each other
        tolerance = binomial_tolerance(0.5, shots)
        assert abs(ucc_p0 - 0.5) < tolerance, f"UCC p0={ucc_p0} outside {tolerance:.4f} tol"
        assert abs(stim_p0 - 0.5) < tolerance, f"Stim p0={stim_p0} outside {tolerance:.4f} tol"
        # For comparing two independent estimates, variance adds: 2*std_err
        cross_tolerance = 2 * tolerance
        assert abs(ucc_p0 - stim_p0) < cross_tolerance, f"UCC vs Stim: {ucc_p0} vs {stim_p0}"

    def test_statistical_distribution_bell(self) -> None:
        """Bell state sampling matches Stim statistically."""
        import stim

        circuit_str = "H 0\nCX 0 1\nM 0\nM 1"
        shots = 10000

        # UCC sampling
        prog = ucc.compile(circuit_str)
        result = ucc.sample(prog, shots, seed=999)
        ucc_00 = np.mean((result.measurements[:, 0] == 0) & (result.measurements[:, 1] == 0))
        ucc_11 = np.mean((result.measurements[:, 0] == 1) & (result.measurements[:, 1] == 1))

        # Stim sampling (seed is in compile_sampler, not sample)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=888)
        stim_results = stim_sampler.sample(shots)
        stim_00 = np.mean((stim_results[:, 0] == 0) & (stim_results[:, 1] == 0))
        stim_11 = np.mean((stim_results[:, 0] == 1) & (stim_results[:, 1] == 1))

        # Bell state: 50% |00>, 50% |11>
        tolerance = binomial_tolerance(0.5, shots)
        assert abs(ucc_00 - 0.5) < tolerance, f"UCC |00>={ucc_00} outside {tolerance:.4f} tol"
        assert abs(ucc_11 - 0.5) < tolerance, f"UCC |11>={ucc_11} outside {tolerance:.4f} tol"
        assert abs(stim_00 - 0.5) < tolerance, f"Stim |00>={stim_00} outside {tolerance:.4f} tol"
        assert abs(stim_11 - 0.5) < tolerance, f"Stim |11>={stim_11} outside {tolerance:.4f} tol"

    def test_meas_active_interfere_y_observable(self) -> None:
        """OP_MEAS_ACTIVE_INTERFERE correctly computes interference with Y-phases."""
        # H 0; T 0 rotates the state to (|0> + e^{ipi/4}|1>)/sqrt(2)
        # S 0 adds phase: (|0> + e^{i*3pi/4}|1>)/sqrt(2)
        # MX 0 forces an OP_MEAS_ACTIVE_INTERFERE where the rewound observable is Y.
        circuit = "H 0\nT 0\nS 0\nMX 0"
        prog = ucc.compile(circuit)

        # P(+) = |<+|psi>|^2 = |1 + e^{i3pi/4}|^2 / 4
        #      = (2 - sqrt(2)) / 4 ~ 0.1464
        shots = 10000
        expected_p0 = (2 - np.sqrt(2)) / 4  # ~ 0.1464
        result = ucc.sample(prog, shots, seed=42)
        p0 = float(np.mean(result.measurements[:, 0] == 0))
        tolerance = binomial_tolerance(expected_p0, shots)

        assert abs(p0 - expected_p0) < tolerance, (
            f"Y-measurement interference failed: p0={p0}, "
            f"expected {expected_p0:.4f} +/- {tolerance:.4f}"
        )


class TestNoiseAndQEC:
    """Tests for noise simulation and QEC features."""

    def test_sample_returns_sample_result(self) -> None:
        """sample() returns a SampleResult with attribute access and unpacking."""
        prog = ucc.compile("H 0\nM 0")
        result = ucc.sample(prog, 10, seed=0)
        assert isinstance(result, ucc.SampleResult)
        # Attribute access
        assert result.measurements.shape == (10, 1)
        assert result.detectors.shape == (10, 0)
        assert result.observables.shape == (10, 0)
        # Tuple unpacking still works
        meas, det, obs = result
        assert meas.shape == (10, 1)
        assert det.shape == (10, 0)
        assert obs.shape == (10, 0)

    def test_program_detector_observable_counts(self) -> None:
        """Program reports correct detector and observable counts."""
        prog = ucc.compile("""
            H 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """)
        assert prog.num_measurements == 1
        assert prog.num_detectors == 1
        assert prog.num_observables == 1

    def test_detector_computes_parity(self) -> None:
        """DETECTOR computes XOR of referenced measurements."""
        # Bell state: M 0 and M 1 always match, so XOR = 0
        prog = ucc.compile("""
            H 0
            CX 0 1
            M 0
            M 1
            DETECTOR rec[-1] rec[-2]
        """)
        result = ucc.sample(prog, 100, seed=42)
        # All detectors should be 0 (perfect correlation)
        assert np.all(result.detectors == 0)

    def test_observable_accumulates_xor(self) -> None:
        """Multiple OBSERVABLE_INCLUDE to same index XOR together."""
        # Bell state: two identical measurements XOR to 0
        prog = ucc.compile("""
            H 0
            CX 0 1
            M 0
            M 1
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-2]
        """)
        result = ucc.sample(prog, 100, seed=42)
        # Observable is XOR of two identical bits = 0
        assert np.all(result.observables == 0)

    def test_observable_tracks_logical_value(self) -> None:
        """Observable correctly tracks logical qubit value."""
        prog = ucc.compile("""
            H 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        """)
        result = ucc.sample(prog, 100, seed=42)
        # Observable should equal measurement (single reference)
        assert np.array_equal(result.measurements[:, 0], result.observables[:, 0])

    def test_readout_noise_flips_bits(self) -> None:
        """M(p) readout noise flips measurement results."""
        # 100% readout noise flips |0> -> measured as 1
        prog = ucc.compile("M(1.0) 0")
        result = ucc.sample(prog, 100, seed=42)
        assert np.all(result.measurements == 1)

    def test_readout_noise_probabilistic(self) -> None:
        """M(0.5) readout noise gives ~50% flip rate."""
        prog = ucc.compile("M(0.5) 0")
        result = ucc.sample(prog, 1000, seed=42)
        flip_rate = float(np.mean(result.measurements))
        # Should be ~50% (measuring |0> with 50% flip = 50% ones)
        tolerance = binomial_tolerance(0.5, 1000)
        assert abs(flip_rate - 0.5) < tolerance

    def test_pauli_noise_x_error(self) -> None:
        """X_ERROR(1.0) always flips qubit."""
        prog = ucc.compile("""
            X_ERROR(1.0) 0
            M 0
        """)
        result = ucc.sample(prog, 100, seed=42)
        # X flips |0> to |1>
        assert np.all(result.measurements == 1)

    def test_pauli_noise_z_error(self) -> None:
        """Z_ERROR doesn't affect computational basis measurement."""
        prog = ucc.compile("""
            Z_ERROR(1.0) 0
            M 0
        """)
        result = ucc.sample(prog, 100, seed=42)
        # Z|0> = |0>, so still measure 0
        assert np.all(result.measurements == 0)

    def test_depolarize1_probabilistic(self) -> None:
        """DEPOLARIZE1 applies X, Y, or Z with equal probability."""
        prog = ucc.compile("""
            DEPOLARIZE1(1.0) 0
            M 0
        """)
        result = ucc.sample(prog, 3000, seed=42)
        # X and Y flip |0>->|1>, Z doesn't. Expected: 2/3 ones.
        ones_rate = float(np.mean(result.measurements))
        expected = 2.0 / 3.0
        tolerance = binomial_tolerance(expected, 3000)
        assert abs(ones_rate - expected) < tolerance

    def test_noise_detector_interaction(self) -> None:
        """Noise causes detector to fire."""
        # Two measurements with X_ERROR in between
        # First M gives 0, X_ERROR flips, second M gives 1
        # Detector XORs them: 0 XOR 1 = 1
        prog = ucc.compile("""
            M 0
            X_ERROR(1.0) 0
            M 0
            DETECTOR rec[-1] rec[-2]
        """)
        result = ucc.sample(prog, 10, seed=0)
        # First meas = 0, second meas = 1, detector = 1
        assert np.all(result.measurements[:, 0] == 0)
        assert np.all(result.measurements[:, 1] == 1)
        assert np.all(result.detectors[:, 0] == 1)

    def test_sample_shape_with_qec(self) -> None:
        """sample() returns correct shapes with detectors/observables."""
        prog = ucc.compile("""
            H 0
            M 0
            M 1
            DETECTOR rec[-1]
            DETECTOR rec[-2]
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(1) rec[-2]
        """)
        shots = 50
        result = ucc.sample(prog, shots, seed=0)
        assert result.measurements.shape == (shots, 2)
        assert result.detectors.shape == (shots, 3)
        assert result.observables.shape == (shots, 2)


class TestPostselection:
    """Tests for compile() with postselection_mask."""

    def test_compile_with_postselection_mask(self) -> None:
        """Compile with postselection_mask kwarg works via nanobind."""
        prog = ucc.compile(
            "M 0\nDETECTOR rec[-1]\n",
            postselection_mask=[1],
        )
        assert prog.num_detectors == 1
        assert prog.num_measurements == 1

    def test_has_postselection_flag(self) -> None:
        """has_postselection is True when mask is non-trivial, False otherwise."""
        circuit = "M 0\nDETECTOR rec[-1]\n"
        prog_no_mask = ucc.compile(circuit)
        assert prog_no_mask.has_postselection is False

        prog_zero_mask = ucc.compile(circuit, postselection_mask=[0])
        assert prog_zero_mask.has_postselection is False

        prog_with_mask = ucc.compile(circuit, postselection_mask=[1])
        assert prog_with_mask.has_postselection is True

    def test_sample_raises_on_postselected_program(self) -> None:
        """sample() raises ValueError when program has postselection."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
        """
        prog = ucc.compile(circuit, postselection_mask=[1])
        with pytest.raises(ValueError, match="sample_survivors"):
            ucc.sample(prog, 10)

    def test_sample_k_raises_on_postselected_program(self) -> None:
        """sample_k() raises ValueError when program has postselection."""
        circuit = """
            R 0
            DEPOLARIZE1(0.1) 0
            M 0
            DETECTOR rec[-1]
        """
        prog = ucc.compile(circuit, postselection_mask=[1])
        with pytest.raises(ValueError, match="sample_k_survivors"):
            ucc.sample_k(prog, 10, k=1)

    def test_sample_ok_without_postselection(self) -> None:
        """sample() works fine when program has no postselection."""
        circuit = "M 0\nDETECTOR rec[-1]\n"
        prog = ucc.compile(circuit)
        result = ucc.sample(prog, 10, seed=42)
        assert result.detectors.shape == (10, 1)

    def test_empty_mask_is_default(self) -> None:
        """Empty postselection_mask produces same result as no mask."""
        circuit = "M 0\nDETECTOR rec[-1]\n"
        prog_default = ucc.compile(circuit)
        prog_empty = ucc.compile(circuit, postselection_mask=[])
        assert prog_default.num_instructions == prog_empty.num_instructions


class TestSampleSurvivors:
    """Tests for sample_survivors() API."""

    def test_counting_only_fast_path(self) -> None:
        """keep_records=False returns survivor metadata with empty arrays."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = ucc.compile(circuit, postselection_mask=[1])
        result = ucc.sample_survivors(prog, 1000, seed=42)

        assert result.total_shots == 1000
        assert 0 < result.passed_shots < 1000
        assert result.discards == 1000 - result.passed_shots
        # Survivors have meas[0]==0, so observable parity is always 0
        assert result.observable_ones[0] == 0
        assert result.logical_errors == 0
        assert isinstance(result, ucc.SampleResult)
        assert result.measurements.shape == (0, prog.num_measurements)
        assert result.detectors.shape == (0, prog.num_detectors)
        assert result.observables.shape == (0, prog.num_observables)

    def test_keep_records_returns_arrays(self) -> None:
        """keep_records=True populates survivor measurement and syndrome arrays."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = ucc.compile(circuit, postselection_mask=[1])
        result = ucc.sample_survivors(prog, 200, seed=42, keep_records=True)

        passed = result.passed_shots
        assert passed > 0
        assert result.measurements.shape == (passed, prog.num_measurements)
        assert result.detectors.shape == (passed, 1)
        assert result.observables.shape == (passed, 1)
        assert np.all(result.measurements[:, 0] == 0)
        assert np.all(result.detectors[:, 0] == 0)
        # All surviving observables should be 0
        assert np.all(result.observables == 0)

    def test_no_postselection_all_pass(self) -> None:
        """Without postselection, all shots pass."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
        """
        prog = ucc.compile(circuit)  # no mask
        result = ucc.sample_survivors(prog, 100, seed=42)

        assert result.total_shots == 100
        assert result.passed_shots == 100
        assert result.discards == 0

    def test_observable_ones_counts_errors(self) -> None:
        """observable_ones correctly counts logical errors in survivors."""
        # No postselection, random observable. ~50% should be 1.
        circuit = """
            H 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = ucc.compile(circuit)
        result = ucc.sample_survivors(prog, 10000, seed=42)

        assert result.passed_shots == 10000
        ones = int(result.observable_ones[0])
        shots = 10000
        tol = binomial_tolerance(0.5, shots) * shots
        assert (
            abs(ones - shots * 0.5) < tol
        ), f"Expected ~{shots * 0.5:.0f} ones, got {ones} (tol={tol:.0f})"

    def test_target_qec_circuit(self) -> None:
        """Smoke test with the real d=3 MSC cultivation circuit."""
        import pathlib

        import stim

        circuit_path = pathlib.Path("tools/bench/target_qec.stim")
        stim_circuit = stim.Circuit.from_file(str(circuit_path))
        text = circuit_path.read_text()

        # Build postselection mask from coord[4] == -9
        coords = stim_circuit.get_detector_coordinates()
        mask = [0] * stim_circuit.num_detectors
        for k, v in coords.items():
            if len(v) >= 5 and v[4] == -9:
                mask[k] = 1

        prog = ucc.compile(text, postselection_mask=mask)
        result = ucc.sample_survivors(prog, 5000, seed=42)

        assert result.total_shots == 5000
        assert result.passed_shots > 0
        assert result.discards > 0
        assert len(result.observable_ones) == 1

    def test_keep_records_100_percent_discard(self) -> None:
        """keep_records=True with all shots discarded returns empty arrays."""
        # Circuit: deterministic meas=1, postselect -> always discards
        circuit = """
            X 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = ucc.compile(circuit, postselection_mask=[1])
        result = ucc.sample_survivors(prog, 100, seed=42, keep_records=True)

        assert result.total_shots == 100
        assert result.passed_shots == 0
        assert result.discards == 100
        assert result.measurements.shape == (0, 1)
        assert result.detectors.shape == (0, 1)
        assert result.observables.shape == (0, 1)

    def test_logical_errors_multi_observable(self) -> None:
        """logical_errors counts shots, not total observable flips."""
        # Two observables both keyed to same random measurement.
        circuit = """
            H 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(1) rec[-1]
        """
        prog = ucc.compile(circuit)
        result = ucc.sample_survivors(prog, 10000, seed=42)

        assert result.passed_shots == 10000
        # Both observables fire on same shots
        ones_0 = int(result.observable_ones[0])
        ones_1 = int(result.observable_ones[1])
        assert ones_0 == ones_1
        # logical_errors == per-shot count, not sum of per-observable
        assert result.logical_errors == ones_0


class TestSyndromeNormalization:
    def test_normalize_syndromes_multiple_observables_xord(self) -> None:
        """Test normalize_syndromes=True on a circuit where multiple includes XOR together."""
        import numpy as np

        # Circuit design:
        # X 0, X 1, X 2 -> All measurements evaluate to 1
        # DET 0: M0 (evaluates to 1)
        # DET 1: M0 ^ M1 (evaluates to 1 ^ 1 = 0)
        # OBS 0: M0 ^ M1 ^ M2 (1 ^ 1 ^ 1 = 1) -> 3 includes!
        # OBS 1: M1 (1) -> 1 include
        circuit = """
            X 0 1 2
            M 0 1 2
            DETECTOR rec[-3]
            DETECTOR rec[-3] rec[-2]

            OBSERVABLE_INCLUDE(0) rec[-3]
            OBSERVABLE_INCLUDE(0) rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]

            OBSERVABLE_INCLUDE(1) rec[-2]
        """

        # 1. Baseline: Without normalization, physical parities match the math above
        prog_raw = ucc.compile(circuit, normalize_syndromes=False)
        result_raw = ucc.sample(prog_raw, shots=10, seed=0)

        assert np.all(result_raw.detectors[:, 0] == 1)
        assert np.all(result_raw.detectors[:, 1] == 0)
        assert np.all(result_raw.observables[:, 0] == 1)
        assert np.all(result_raw.observables[:, 1] == 1)

        # 2. Normalized: All output parities must be strictly 0.
        # We also apply a postselection mask on DET 0 (which natively evaluates to 1).
        # Since it is normalized, it becomes 0, meaning shots should SURVIVE.
        prog_norm = ucc.compile(
            circuit,
            normalize_syndromes=True,
            postselection_mask=[1, 0],
        )

        res = ucc.sample_survivors(prog_norm, shots=10, seed=0, keep_records=True)

        assert res.passed_shots == 10  # Normalized 1^1=0, so shots survive!
        assert np.all(res.detectors == 0)
        assert np.all(res.observables == 0)
        assert res.logical_errors == 0

    def test_normalize_syndromes_conflict_raises(self) -> None:
        """Cannot provide explicit expected parities with normalize_syndromes=True."""
        import pytest

        circuit = "M 0\nDETECTOR rec[-1]\n"
        with pytest.raises(ValueError):
            ucc.compile(
                circuit,
                normalize_syndromes=True,
                expected_detectors=[0],
            )

    def test_normalize_syndromes_no_noise_passthrough(self) -> None:
        """Normalization on a circuit without noise produces all-zero syndromes."""
        import numpy as np

        circuit = """
            X 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = ucc.compile(circuit, normalize_syndromes=True)
        result = ucc.sample(prog, shots=5, seed=0)

        assert np.all(result.detectors == 0)
        assert np.all(result.observables == 0)

    def test_normalize_syndromes_with_noise_detects_errors(self) -> None:
        """With noise and normalization, errors show up as 1s in syndromes."""
        import numpy as np

        # A circuit where noise can flip the detector
        circuit = """
            X_ERROR(1.0) 0
            M 0
            DETECTOR rec[-1]
        """
        prog = ucc.compile(circuit, normalize_syndromes=True)
        result = ucc.sample(prog, shots=10, seed=0)

        # With 100% X error, measurement flips from 0 to 1.
        # Reference (noiseless) detector parity = 0.
        # Noisy parity = 1. Normalized: 1 ^ 0 = 1 (error detected).
        assert np.all(result.detectors[:, 0] == 1)

    def test_explicit_expected_detectors(self) -> None:
        """Explicit expected_detectors without normalize_syndromes works."""
        import numpy as np

        circuit = """
            X 0
            M 0
            DETECTOR rec[-1]
        """
        # Raw detector parity = 1. With expected_detectors=[1], normalized = 0.
        prog = ucc.compile(circuit, expected_detectors=[1])
        result = ucc.sample(prog, shots=5, seed=0)

        assert np.all(result.detectors[:, 0] == 0)

    def test_explicit_expected_observables(self) -> None:
        """Explicit expected_observables without normalize_syndromes works."""
        import numpy as np

        circuit = """
            X 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        # Raw obs = 1. With expected_observables=[1], normalized = 0.
        prog = ucc.compile(circuit, expected_observables=[1])
        result = ucc.sample(prog, shots=5, seed=0)

        assert np.all(result.observables[:, 0] == 0)

    def test_compute_reference_syndrome_api(self) -> None:
        """compute_reference_syndrome is accessible from Python."""
        circuit = ucc.parse("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
        hir = ucc.trace(circuit)
        ref = ucc.compute_reference_syndrome(hir)

        assert ref["detectors"] == [1]
        assert ref["observables"] == [1]

    def test_remove_noise_pass_api(self) -> None:
        """RemoveNoisePass is accessible from Python."""
        circuit = ucc.parse("X_ERROR(0.1) 0\nM 0\n")
        hir = ucc.trace(circuit)

        original_count = hir.num_ops
        strip = ucc.RemoveNoisePass()

        pm = ucc.HirPassManager()
        pm.add(strip)
        pm.run(hir)

        assert hir.num_ops < original_count
        for op in hir:
            assert op.op_type != ucc.OpType.NOISE


class TestExpVal:
    """Tests for EXP_VAL expectation value probes via Python bindings."""

    def test_sample_returns_exp_vals(self) -> None:
        """sample() populates exp_vals for circuits with EXP_VAL."""
        prog = ucc.compile("EXP_VAL Z0")
        result = ucc.sample(prog, 10, seed=42)
        assert result.exp_vals.shape == (10, 1)
        assert result.exp_vals.dtype == np.float64
        np.testing.assert_allclose(result.exp_vals[:, 0], 1.0, atol=1e-12)

    def test_no_exp_val_gives_empty(self) -> None:
        """Circuits without EXP_VAL have shape (shots, 0) exp_vals."""
        prog = ucc.compile("H 0\nM 0")
        result = ucc.sample(prog, 5, seed=0)
        assert result.exp_vals.shape == (5, 0)
        assert prog.num_exp_vals == 0

    def test_program_num_exp_vals(self) -> None:
        """Program.num_exp_vals reports the correct count."""
        prog = ucc.compile("EXP_VAL X0 Z1")
        assert prog.num_exp_vals == 2

    def test_hir_num_exp_vals(self) -> None:
        """HirModule.num_exp_vals reports the correct count."""
        hir = ucc.trace(ucc.parse("EXP_VAL X0*Y1 Z2"))
        assert hir.num_exp_vals == 2

    def test_state_exp_vals(self) -> None:
        """State.exp_vals is accessible and correctly sized."""
        state = ucc.State(peak_rank=1, num_measurements=0, num_exp_vals=2)
        assert len(state.exp_vals) == 2

    def test_state_constructor_requires_keywords(self) -> None:
        """State constructor rejects positional arguments beyond self."""
        with pytest.raises(TypeError):
            ucc.State(1, 0)

    def test_state_constructor_seed_keyword(self) -> None:
        """Keyword seed does not affect exp_vals sizing."""
        state = ucc.State(peak_rank=1, num_measurements=0, seed=42)
        assert len(state.exp_vals) == 0

    def test_exp_val_multiple_probes(self) -> None:
        """Multiple EXP_VAL probes return consecutive columns."""
        prog = ucc.compile("H 0\nEXP_VAL X0\nEXP_VAL Z0")
        result = ucc.sample(prog, 5, seed=0)
        assert result.exp_vals.shape == (5, 2)
        np.testing.assert_allclose(result.exp_vals[:, 0], 1.0, atol=1e-12)  # <X> on |+>
        np.testing.assert_allclose(result.exp_vals[:, 1], 0.0, atol=1e-12)  # <Z> on |+>

    def test_exp_val_does_not_disturb_measurement(self) -> None:
        """EXP_VAL is non-destructive: measurements after it are unaffected."""
        prog = ucc.compile("EXP_VAL Z0\nM 0")
        result = ucc.sample(prog, 100, seed=0)
        # |0> state: all measurements should be 0
        assert np.all(result.measurements == 0)
        np.testing.assert_allclose(result.exp_vals[:, 0], 1.0, atol=1e-12)
