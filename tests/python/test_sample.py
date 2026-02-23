"""Python integration tests for ucc.compile and ucc.sample."""

import numpy as np

import ucc


def assert_statevectors_equal(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float = 1e-4, msg: str = ""
) -> None:
    """Assert two statevectors are equal up to global phase.

    Uses fidelity: |<ψ|φ>|² ≥ 1 - rtol
    """
    fidelity = float(np.abs(np.vdot(expected, actual)) ** 2)
    if fidelity < 1.0 - rtol:
        raise AssertionError(f"Fidelity {fidelity:.6f} < {1.0 - rtol}. {msg}")


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
        """Measurement of |0⟩ always gives 0."""
        prog = ucc.compile("M 0")
        results = ucc.sample(prog, 100, seed=42)
        for shot in results:
            assert shot[0] == 0

    def test_sample_deterministic_one(self) -> None:
        """Measurement of |1⟩ always gives 1."""
        prog = ucc.compile("X 0\nM 0")
        results = ucc.sample(prog, 100, seed=42)
        for shot in results:
            assert shot[0] == 1

    def test_sample_superposition(self) -> None:
        """|+⟩ state gives roughly 50/50 distribution."""
        prog = ucc.compile("H 0\nM 0")
        results = ucc.sample(prog, 1000, seed=42)
        zeros = sum(1 for r in results if r[0] == 0)
        ones = sum(1 for r in results if r[0] == 1)
        # Allow 10% tolerance
        assert 400 < zeros < 600
        assert 400 < ones < 600

    def test_sample_bell_state_correlated(self) -> None:
        """Bell state measurements are always correlated."""
        prog = ucc.compile("""
            H 0
            CX 0 1
            M 0
            M 1
        """)
        results = ucc.sample(prog, 500, seed=99)
        for shot in results:
            assert shot[0] == shot[1], f"Bell state not correlated: {shot}"

    def test_sample_reproducible(self) -> None:
        """Same seed produces same results."""
        prog = ucc.compile("H 0\nM 0")
        results1 = ucc.sample(prog, 100, seed=12345)
        results2 = ucc.sample(prog, 100, seed=12345)
        assert np.array_equal(results1, results2)

    def test_sample_different_seeds(self) -> None:
        """Different seeds produce different results."""
        prog = ucc.compile("H 0\nM 0")
        results1 = ucc.sample(prog, 100, seed=1)
        results2 = ucc.sample(prog, 100, seed=2)
        # With 100 random bits, probability of match is 2^-100
        assert not np.array_equal(results1, results2)

    def test_sample_shape(self) -> None:
        """Results have correct shape and type."""
        prog = ucc.compile("H 0\nM 0\nH 1\nM 1")
        results = ucc.sample(prog, 50, seed=0)
        assert isinstance(results, np.ndarray)
        assert results.dtype == np.uint8
        assert results.shape == (50, 2)

    def test_sample_reset_works(self) -> None:
        """Reset correctly resets to |0⟩."""
        prog = ucc.compile("""
            X 0
            R 0
            M 0
        """)
        results = ucc.sample(prog, 100, seed=42)
        # Second measurement (after reset) should always be 0
        for shot in results:
            assert shot[1] == 0, f"Reset failed: {shot}"


class TestStatevector:
    """Tests for ucc.get_statevector()."""

    def test_statevector_pure_clifford(self) -> None:
        """Pure Clifford circuit matches expected statevector."""
        # H|0⟩ = |+⟩ = [1/√2, 1/√2]
        prog = ucc.compile("H 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_bell_state(self) -> None:
        """Bell state matches expected statevector."""
        prog = ucc.compile("H 0\nCX 0 1")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_single_t_gate(self) -> None:
        """H-T circuit: [1/√2, e^{iπ/4}/√2]."""
        prog = ucc.compile("H 0\nT 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        expected = np.array([1 / np.sqrt(2), np.exp(1j * np.pi / 4) / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_t_dagger(self) -> None:
        """H-T† circuit: [1/√2, e^{-iπ/4}/√2]."""
        prog = ucc.compile("H 0\nT_DAG 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        expected = np.array([1 / np.sqrt(2), np.exp(-1j * np.pi / 4) / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_two_t_equals_s(self) -> None:
        """T-T = S: H-T-T should equal H-S."""
        prog = ucc.compile("H 0\nT 0\nT 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # H-S: [1/√2, i/√2]
        expected = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_four_t_equals_z(self) -> None:
        """T^4 = Z: H-T-T-T-T should equal H-Z."""
        prog = ucc.compile("H 0\nT 0\nT 0\nT 0\nT 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # H-Z: [1/√2, -1/√2]
        expected = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_t_on_zero(self) -> None:
        """T|0⟩ = |0⟩ (global phase only)."""
        prog = ucc.compile("T 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # T|0⟩ = |0⟩ up to global phase
        expected = np.array([1, 0], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_two_qubit_t(self) -> None:
        """Two-qubit circuit with T on qubit 0."""
        prog = ucc.compile("H 0\nH 1\nT 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # T on q0 affects indices where bit 0 is set (indices 1, 3)
        phase = np.exp(1j * np.pi / 4)
        expected = np.array([0.5, 0.5 * phase, 0.5, 0.5 * phase], dtype=complex)
        assert_statevectors_equal(sv, expected)

    def test_statevector_bell_plus_t(self) -> None:
        """Bell state with T on control qubit."""
        prog = ucc.compile("H 0\nCX 0 1\nT 0")
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # Bell state: (|00⟩ + |11⟩)/√2
        # T on q0: |00⟩→|00⟩, |11⟩→e^{iπ/4}|11⟩
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
            state = ucc.State(prog.peak_rank, prog.num_measurements)
            ucc.execute(prog, state)
            sv = ucc.get_statevector(prog, state)
            norm = float(np.sqrt(np.sum(np.abs(sv) ** 2)))
            assert abs(norm - 1.0) < 1e-10, f"Not normalized: {circuit}"


class TestCliffordValidation:
    """Validate pure-Clifford statevectors against Stim."""

    def _random_clifford_circuit(self, num_qubits: int, depth: int, seed: int) -> str:
        """Generate a random pure-Clifford circuit."""
        rng = np.random.default_rng(seed)
        gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z"]
        gates_2q = ["CX", "CY", "CZ"]

        lines: list[str] = []
        for _ in range(depth):
            if num_qubits > 1 and rng.random() < 0.4:
                # 2-qubit gate
                gate = rng.choice(gates_2q)
                q1, q2 = rng.choice(num_qubits, size=2, replace=False)
                lines.append(f"{gate} {q1} {q2}")
            else:
                # 1-qubit gate
                gate = rng.choice(gates_1q)
                q = rng.integers(0, num_qubits)
                lines.append(f"{gate} {q}")

        return "\n".join(lines)

    def test_random_clifford_single_qubit(self) -> None:
        """Random 1-qubit Clifford circuits match Stim."""
        import stim

        for seed in range(10):
            circuit_str = self._random_clifford_circuit(1, 5, seed)

            # UCC statevector
            prog = ucc.compile(circuit_str)
            state = ucc.State(prog.peak_rank, prog.num_measurements)
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
                circuit_str = self._random_clifford_circuit(num_qubits, 10, seed)

                # UCC statevector
                prog = ucc.compile(circuit_str)
                state = ucc.State(prog.peak_rank, prog.num_measurements)
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
            "M 0",  # |0⟩ always gives 0
            "X 0\nM 0",  # |1⟩ always gives 1
            "H 0\nCX 0 1\nM 0\nM 1",  # Bell state: correlated
        ]

        for circuit_str in circuits:
            # UCC sampling
            prog = ucc.compile(circuit_str)
            ucc_results = ucc.sample(prog, 100, seed=42)

            # Stim sampling (seed is in compile_sampler, not sample)
            stim_circuit = stim.Circuit(circuit_str)
            stim_sampler = stim_circuit.compile_sampler(seed=42)
            stim_results = stim_sampler.sample(100)

            # For deterministic circuits, all shots should match
            # (Note: seeds may differ, but deterministic outcomes should be consistent)
            if prog.num_measurements == 1:
                # Single measurement: check value consistency
                ucc_vals = set(tuple(r) for r in ucc_results)
                stim_vals = set(tuple(r) for r in stim_results)
                assert ucc_vals == stim_vals, f"Mismatch for: {circuit_str}"
            else:
                # Multi-measurement: check correlation structure
                for ucc_shot in ucc_results:
                    if "CX" in circuit_str:  # Bell state
                        assert ucc_shot[0] == ucc_shot[1], "Bell correlation broken"

    def test_statistical_distribution_h(self) -> None:
        """H gate sampling matches Stim statistically."""
        import stim

        circuit_str = "H 0\nM 0"
        shots = 10000

        # UCC sampling
        prog = ucc.compile(circuit_str)
        ucc_results = ucc.sample(prog, shots, seed=12345)
        ucc_p0 = np.mean(ucc_results[:, 0] == 0)

        # Stim sampling (seed is in compile_sampler, not sample)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=54321)
        stim_results = stim_sampler.sample(shots)
        stim_p0 = np.mean(stim_results[:, 0] == 0)

        # Both should be close to 0.5, and close to each other
        assert abs(ucc_p0 - 0.5) < 0.02, f"UCC p0={ucc_p0} too far from 0.5"
        assert abs(stim_p0 - 0.5) < 0.02, f"Stim p0={stim_p0} too far from 0.5"
        assert abs(ucc_p0 - stim_p0) < 0.03, f"UCC vs Stim: {ucc_p0} vs {stim_p0}"

    def test_statistical_distribution_bell(self) -> None:
        """Bell state sampling matches Stim statistically."""
        import stim

        circuit_str = "H 0\nCX 0 1\nM 0\nM 1"
        shots = 10000

        # UCC sampling
        prog = ucc.compile(circuit_str)
        ucc_results = ucc.sample(prog, shots, seed=999)
        ucc_00 = np.mean((ucc_results[:, 0] == 0) & (ucc_results[:, 1] == 0))
        ucc_11 = np.mean((ucc_results[:, 0] == 1) & (ucc_results[:, 1] == 1))

        # Stim sampling (seed is in compile_sampler, not sample)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=888)
        stim_results = stim_sampler.sample(shots)
        stim_00 = np.mean((stim_results[:, 0] == 0) & (stim_results[:, 1] == 0))
        stim_11 = np.mean((stim_results[:, 0] == 1) & (stim_results[:, 1] == 1))

        # Bell state: 50% |00⟩, 50% |11⟩
        assert abs(ucc_00 - 0.5) < 0.02, f"UCC |00⟩={ucc_00}"
        assert abs(ucc_11 - 0.5) < 0.02, f"UCC |11⟩={ucc_11}"
        assert abs(stim_00 - 0.5) < 0.02, f"Stim |00⟩={stim_00}"
        assert abs(stim_11 - 0.5) < 0.02, f"Stim |11⟩={stim_11}"


class TestOracleValidation:
    """Validate Clifford+T statevectors against standalone numpy oracle."""

    def _oracle_statevector(self, circuit_str: str) -> np.ndarray:
        """Use the standalone numpy oracle to compute statevector."""
        from clifford_t_oracle import simulate_circuit

        return simulate_circuit(circuit_str)

    def _random_clifford_t_circuit(self, num_qubits: int, depth: int, seed: int) -> str:
        """Generate a random universal Clifford+T circuit."""
        rng = np.random.default_rng(seed)
        gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG"]

        lines: list[str] = []
        for _ in range(depth):
            if num_qubits > 1 and rng.random() < 0.3:
                q1, q2 = rng.choice(num_qubits, size=2, replace=False)
                lines.append(f"CX {q1} {q2}")
            else:
                gate = rng.choice(gates_1q)
                q = rng.integers(0, num_qubits)
                lines.append(f"{gate} {q}")
        return "\n".join(lines)

    def test_oracle_single_t(self) -> None:
        """Single T gate matches oracle."""
        circuit = "H 0\nT 0"

        # UCC statevector
        prog = ucc.compile(circuit)
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        ucc_sv = ucc.get_statevector(prog, state)

        # Oracle statevector
        oracle_sv = self._oracle_statevector(circuit)
        assert_statevectors_equal(ucc_sv, oracle_sv)

    def test_oracle_two_t_gates(self) -> None:
        """Two T gates match oracle."""
        circuit = "H 0\nT 0\nT 0"

        prog = ucc.compile(circuit)
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        ucc_sv = ucc.get_statevector(prog, state)

        oracle_sv = self._oracle_statevector(circuit)
        assert_statevectors_equal(ucc_sv, oracle_sv)

    def test_oracle_t_dag(self) -> None:
        """T† matches oracle."""
        circuit = "H 0\nT_DAG 0"

        prog = ucc.compile(circuit)
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        ucc_sv = ucc.get_statevector(prog, state)

        oracle_sv = self._oracle_statevector(circuit)
        assert_statevectors_equal(ucc_sv, oracle_sv)

    def test_oracle_bell_plus_t(self) -> None:
        """Bell + T matches oracle."""
        circuit = "H 0\nCX 0 1\nT 0"

        prog = ucc.compile(circuit)
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        ucc_sv = ucc.get_statevector(prog, state)

        oracle_sv = self._oracle_statevector(circuit)
        assert_statevectors_equal(ucc_sv, oracle_sv)

    def test_oracle_multi_qubit_clifford_t(self) -> None:
        """Multi-qubit Clifford+T matches oracle."""
        circuit = "H 0\nH 1\nCX 0 1\nT 0\nT 1"

        prog = ucc.compile(circuit)
        state = ucc.State(prog.peak_rank, prog.num_measurements)
        ucc.execute(prog, state)
        ucc_sv = ucc.get_statevector(prog, state)

        oracle_sv = self._oracle_statevector(circuit)
        assert_statevectors_equal(ucc_sv, oracle_sv)

    def test_oracle_random_clifford_t_fuzz(self) -> None:
        """Random Clifford+T circuits match oracle."""
        for num_qubits in [2, 3, 4]:
            for seed in range(5):
                circuit_str = self._random_clifford_t_circuit(num_qubits, 15, seed)

                prog = ucc.compile(circuit_str)
                state = ucc.State(prog.peak_rank, prog.num_measurements)
                ucc.execute(prog, state)
                ucc_sv = ucc.get_statevector(prog, state)

                oracle_sv = self._oracle_statevector(circuit_str)
                assert_statevectors_equal(ucc_sv, oracle_sv, msg=f"circuit:\n{circuit_str}")

    def test_statevector_after_measurement(self) -> None:
        """Statevector correctly handles active rank after measurement."""
        # H 0, T 0 creates a superposition. M 0 collapses it.
        # peak_rank is 1, but final_rank is 0.
        circuit = "H 0\nT 0\nM 0"
        prog = ucc.compile(circuit)

        state = ucc.State(prog.peak_rank, prog.num_measurements, seed=42)
        ucc.execute(prog, state)
        sv = ucc.get_statevector(prog, state)

        # Statevector must be perfectly normalized (catches garbage memory bug)
        norm = float(np.sqrt(np.sum(np.abs(sv) ** 2)))
        assert abs(norm - 1.0) < 1e-10, f"Not normalized: {norm}"

        # The state must be cleanly collapsed AND match the measurement record
        outcome = state.meas_record[0]
        if outcome == 0:
            # Measured |0⟩ eigenspace: |ψ⟩ ∝ |0⟩
            assert abs(abs(sv[0]) - 1.0) < 1e-5, f"Expected |0⟩ but got {sv}"
            assert abs(sv[1]) < 1e-5, f"Expected no |1⟩ component but got {sv}"
        else:
            # Measured |1⟩ eigenspace: |ψ⟩ ∝ |1⟩
            assert abs(abs(sv[1]) - 1.0) < 1e-5, f"Expected |1⟩ but got {sv}"
            assert abs(sv[0]) < 1e-5, f"Expected no |0⟩ component but got {sv}"

    def test_sample_measure_merge_y_observable(self) -> None:
        """Test OP_MEASURE_MERGE correctly computes interference with Y-phases."""
        # H 0; T 0 rotates the state to (|0⟩ + e^{iπ/4}|1⟩)/√2
        # S 0 shifts the X-axis to the Y-axis.
        # MX 0 forces an OP_MEASURE_MERGE where the rewound observable is Y.
        circuit = "H 0\nT 0\nS 0\nMX 0"
        prog = ucc.compile(circuit)

        # Analytically: after H T S, state is proportional to:
        #   |0⟩ + e^{iπ/4} * (i)|1⟩ = |0⟩ + e^{i 3π/4}|1⟩
        # MX measures ⟨+|ψ⟩ = 1/√2 (1 + e^{i 3π/4}) = (1 - 1/√2)/√2 + ...
        # P(0) = |1 + e^{i 3π/4}|^2 / 2 = (1 - √2/2) ≈ 0.146
        # This is strictly asymmetric, proving complex interference is working.
        results = ucc.sample(prog, 10000, seed=42)
        p0 = float(np.mean(results[:, 0] == 0))

        assert 0.12 < p0 < 0.17, f"Y-measurement interference failed, got p0={p0}"
