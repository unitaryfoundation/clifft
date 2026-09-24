"""Python integration tests for clifft.compile and clifft.sample."""

import warnings
from typing import Any

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equiv,
    binomial_tolerance,
    random_clifford_circuit,
)
from utils_conformance import SMALL_CIRCUIT_SHOTS, CpuSamplingMode, skip_unavailable_intra_shot

import clifft


class TestCompile:
    """Tests for clifft.compile()."""

    def test_compile_simple(self) -> None:
        """Compile a simple circuit."""
        prog = clifft.compile("H 0\nT 0\nM 0", hir_passes=None)
        assert prog.peak_active_width == 1
        assert prog.num_measurements == 1
        assert prog.num_actions >= 1

    def test_peak_rank_is_a_deprecated_alias(self) -> None:
        prog = clifft.compile("H 0\nT 0", hir_passes=None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expected = prog.peak_active_width
        with pytest.warns(DeprecationWarning, match="peak_active_width"):
            assert prog.peak_rank == expected

    def test_repr_names_peak_active_width(self) -> None:
        prog = clifft.compile("H 0\nT 0", hir_passes=None)
        assert ", peak_active_width=1, " in repr(prog)

    def test_compile_pure_clifford(self) -> None:
        """Pure Clifford circuits have peak active width zero."""
        prog = clifft.compile("H 0\nCX 0 1\nM 0\nM 1")
        assert prog.peak_active_width == 0
        assert prog.num_measurements == 2

    def test_compile_multiple_t_gates(self) -> None:
        """Multiple independent T gates increase peak active width."""
        prog = clifft.compile("""
            H 0
            H 1
            T 0
            T 1
        """)
        assert prog.peak_active_width == 2

    def test_lower_returns_the_public_program_type(self) -> None:
        """Explicit parse and trace use the same production lowering boundary."""
        hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
        program = clifft.lower(hir)

        assert isinstance(program, clifft.Program)
        assert program.num_actions > 0
        assert clifft.sample(program, 1, seed=1).measurements.shape == (1, 1)


class TestSample:
    """Tests for clifft.sample()."""

    def test_sample_deterministic_zero(self, sampling_mode: CpuSamplingMode) -> None:
        """Measurement of |0> always gives 0."""
        prog = sampling_mode.compile("M 0")
        result = sampling_mode.sample(prog, 100, seed=42)
        assert np.all(result.measurements[:, 0] == 0)

    def test_sample_deterministic_one(self, sampling_mode: CpuSamplingMode) -> None:
        """Measurement of |1> always gives 1."""
        prog = sampling_mode.compile("X 0\nM 0")
        result = sampling_mode.sample(prog, 100, seed=42)
        assert np.all(result.measurements[:, 0] == 1)

    def test_sample_superposition(self, sampling_mode: CpuSamplingMode) -> None:
        """|+> state gives roughly 50/50 distribution."""
        prog = sampling_mode.compile("H 0\nM 0")
        shots = 1000
        result = sampling_mode.sample(prog, shots, seed=42)
        p0 = float(np.mean(result.measurements[:, 0] == 0))
        p1 = float(np.mean(result.measurements[:, 0] == 1))
        tolerance = binomial_tolerance(0.5, shots)
        assert abs(p0 - 0.5) < tolerance, f"p(0)={p0} outside {tolerance:.3f} tolerance"
        assert abs(p1 - 0.5) < tolerance, f"p(1)={p1} outside {tolerance:.3f} tolerance"

    def test_sample_bell_state_correlated(self, sampling_mode: CpuSamplingMode) -> None:
        """Bell state measurements are always correlated."""
        prog = sampling_mode.compile("""
            H 0
            CX 0 1
            M 0
            M 1
        """)
        result = sampling_mode.sample(prog, 500, seed=99)
        assert np.all(
            result.measurements[:, 0] == result.measurements[:, 1]
        ), "Bell state not correlated"

    def test_biased_entangled_measurements_match_analytic_distribution(
        self, sampling_mode: CpuSamplingMode
    ) -> None:
        """A coherent T rotation biases both members of an entangled pair."""
        program = sampling_mode.compile("H 0\nT 0\nH 0\nCX 0 1\nM 0 1")
        shots = 50_000
        result = sampling_mode.sample(program, shots, seed=42)

        expected_p0 = (1.0 + np.cos(np.pi / 4.0)) / 2.0
        tolerance = binomial_tolerance(expected_p0, shots)
        for column in range(2):
            observed_p0 = float(np.mean(result.measurements[:, column] == 0))
            assert abs(observed_p0 - expected_p0) < tolerance

        np.testing.assert_array_equal(result.measurements[:, 0], result.measurements[:, 1])

    def test_repeated_active_state_expansion_and_compaction(
        self, sampling_mode: CpuSamplingMode
    ) -> None:
        """Repeated inject-entangle-measure rounds retain a bounded active state."""
        rounds = 256
        lines = ["H 0", "T 0"]
        for _ in range(rounds):
            lines.extend(["H 1", "T 1", "CX 1 0", "M 1", "R 1"])
        lines.append("M 0")

        program = sampling_mode.compile("\n".join(lines), hir_passes=None)
        assert program.peak_active_width == 2

        result = sampling_mode.sample(program, shots=SMALL_CIRCUIT_SHOTS, seed=7)
        assert result.measurements.shape == (SMALL_CIRCUIT_SHOTS, rounds + 1)
        assert np.all((result.measurements == 0) | (result.measurements == 1))

    def test_sample_reproducible(self, sampling_mode: CpuSamplingMode) -> None:
        """Same seed produces same results."""
        prog = sampling_mode.compile("H 0\nM 0")
        result1 = sampling_mode.sample(prog, 100, seed=12345)
        result2 = sampling_mode.sample(prog, 100, seed=12345)
        assert np.array_equal(result1.measurements, result2.measurements)

    @pytest.mark.parametrize("batch_size", ["auto", 1, 2, 65])
    def test_sample_batch_configuration_replays_seeded_rows(
        self, sampling_api: Any, batch_size: Any
    ) -> None:
        """A seed replays exactly when the batching configuration is unchanged."""
        prog = sampling_api.compile("X_ERROR(0.125) 0\nH 1\nM 0 1\nDETECTOR rec[-2] rec[-1]")
        first = sampling_api.sample(prog, 257, seed=12345, batch_size=batch_size)
        replay = sampling_api.sample(prog, 257, seed=12345, batch_size=batch_size)
        np.testing.assert_array_equal(first.measurements, replay.measurements)
        np.testing.assert_array_equal(first.detectors, replay.detectors)

    def test_scalar_and_packed_modes_are_statistically_equivalent(self, sampling_api: Any) -> None:
        """Changing execution mode may change rows but not the sampled distribution."""
        prog = sampling_api.compile("X_ERROR(0.125) 0\nH 1\nM 0 1")
        shots = 50_000
        scalar = sampling_api.sample(prog, shots, seed=12346, batch_size=1)
        packed = sampling_api.sample(prog, shots, seed=12346, batch_size=257)
        expected = (0.125, 0.5)
        for column, probability in enumerate(expected):
            tolerance = binomial_tolerance(probability, shots)
            assert abs(float(np.mean(scalar.measurements[:, column])) - probability) < tolerance
            assert abs(float(np.mean(packed.measurements[:, column])) - probability) < tolerance

    @pytest.mark.parametrize("batch_size", [0, -1, "all", 1.5])
    def test_sample_rejects_invalid_batch_size(self, sampling_api: Any, batch_size: Any) -> None:
        """Only positive integer capacities and the auto sentinel are accepted."""
        prog = sampling_api.compile("M 0")
        with pytest.raises((TypeError, ValueError), match="batch_size|incompatible"):
            sampling_api.sample(prog, 1, batch_size=batch_size)

    def test_packed_batch_rejects_intra_shot_workers(self, sampling_api: Any) -> None:
        prog = sampling_api.compile("H 0\nT 0\nM 0")
        with pytest.raises(ValueError, match="batch_size|intra-shot"):
            sampling_api.sample(
                prog,
                64,
                thread_layout=(1, 2),
                intra_shot_min_active_width=0,
                batch_size=64,
            )

    def test_explicit_batch_size_rejects_unsafe_state_footprint(self, sampling_api: Any) -> None:
        circuit = "\n".join(f"H {qubit}\nT {qubit}" for qubit in range(20))
        prog = sampling_api.compile(circuit)
        with pytest.raises(ValueError, match="64 MiB packed-state limit"):
            sampling_api.sample(prog, 4096, batch_size=2048)

    @pytest.mark.parametrize("threads", [2, 3])
    @pytest.mark.parametrize("batch_size", [1, 65])
    def test_sample_threads_preserve_seeded_rows(
        self, sampling_api: Any, threads: int, batch_size: int
    ) -> None:
        """Worker count and dynamic scheduling do not change seeded rows."""
        prog = sampling_api.compile(
            "H 0 1\nT 0\nEXP_VAL X0\nM 0 1\nEXP_VAL Z0\n"
            "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        )
        # Explicit capacities exercise multiple batches without changing RNG boundaries.
        serial = sampling_api.sample(prog, 257, seed=12345, threads=1, batch_size=batch_size)
        threaded = sampling_api.sample(
            prog, 257, seed=12345, threads=threads, batch_size=batch_size
        )
        np.testing.assert_array_equal(threaded.measurements, serial.measurements)
        np.testing.assert_array_equal(threaded.detectors, serial.detectors)
        np.testing.assert_array_equal(threaded.observables, serial.observables)
        np.testing.assert_array_equal(threaded.exp_vals, serial.exp_vals)

    def test_auto_batch_boundaries_ignore_worker_count(self, sampling_api: Any) -> None:
        """Changing worker count preserves automatic batch RNG boundaries."""
        prog = sampling_api.compile(
            "X_ERROR(0.125) 0\nH 1\nT 1\nEXP_VAL X1\nM 0 1\n"
            "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        )
        # Two full automatic batches and a tail leave work for both workers.
        shots = 2 * 2048 + 1

        serial = sampling_api.sample(prog, shots, seed=42, threads=1)
        threaded = sampling_api.sample(prog, shots, seed=42, threads=2)

        np.testing.assert_array_equal(threaded.measurements, serial.measurements)
        np.testing.assert_array_equal(threaded.detectors, serial.detectors)
        np.testing.assert_array_equal(threaded.observables, serial.observables)
        np.testing.assert_array_equal(threaded.exp_vals, serial.exp_vals)

    @pytest.mark.parametrize("threads", [0, -1, "all", 1.5])
    def test_sample_rejects_invalid_threads(self, sampling_api: Any, threads: Any) -> None:
        """Only positive integers and the auto sentinel are accepted."""
        prog = sampling_api.compile("M 0")
        with pytest.raises((TypeError, ValueError), match="threads|incompatible"):
            sampling_api.sample(prog, 1, threads=threads)

    @pytest.mark.parametrize("batch_size", [1, 65])
    def test_sample_thread_layout_preserves_seeded_rows(
        self, sampling_api: Any, batch_size: int
    ) -> None:
        """An explicit layout is a thin override of automatic worker selection."""
        prog = sampling_api.compile(
            "H 0 1\nT 0\nM 0 1\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        )
        serial = sampling_api.sample(prog, 257, seed=12346, threads=1, batch_size=batch_size)
        threaded = sampling_api.sample(
            prog, 257, seed=12346, threads="auto", thread_layout=(2, 1), batch_size=batch_size
        )
        np.testing.assert_array_equal(threaded.measurements, serial.measurements)
        np.testing.assert_array_equal(threaded.detectors, serial.detectors)
        np.testing.assert_array_equal(threaded.observables, serial.observables)

    def test_sample_runtime_intra_shot_threshold_preserves_seeded_rows(
        self, sampling_api: Any
    ) -> None:
        """Expert layouts can lower the kernel crossover without rebuilding."""
        # The probes retain four active qubits: two rotation chunks even with AVX-512.
        prog = sampling_api.compile(
            "H 0 1 2 3\nT 0 1 2 3\nEXP_VAL X0*X1*X2*X3\nR_X(0.125) 0\nEXP_VAL Z0\nM 0 1 2 3"
        )
        assert prog.peak_active_width == 4
        serial = sampling_api.sample(prog, 31, seed=12347, threads=1)
        try:
            threaded = sampling_api.sample(
                prog,
                31,
                seed=12347,
                thread_layout=(1, 2),
                intra_shot_min_active_width=3,
            )
        except ValueError as error:
            skip_unavailable_intra_shot(error)
        np.testing.assert_array_equal(threaded.measurements, serial.measurements)
        np.testing.assert_allclose(threaded.exp_vals, serial.exp_vals, atol=1e-12, rtol=0)

    @pytest.mark.parametrize(
        ("thread_layout", "min_active_width"),
        [(None, 0), ((1, 1), -1), ((1, 1), 2**40), ((1, 1), 1.5)],
    )
    def test_sample_rejects_invalid_runtime_intra_shot_threshold(
        self, sampling_api: Any, thread_layout: Any, min_active_width: Any
    ) -> None:
        """The expert threshold is bounded and requires an explicit layout."""
        prog = sampling_api.compile("M 0")
        with pytest.raises(
            (TypeError, ValueError), match="intra_shot_min_active_width|incompatible"
        ):
            sampling_api.sample(
                prog,
                1,
                thread_layout=thread_layout,
                intra_shot_min_active_width=min_active_width,
            )

    @pytest.mark.parametrize("thread_layout", [(0, 1), (1, 0), (1,), "auto", (1, 1.5)])
    def test_sample_rejects_invalid_thread_layout(
        self, sampling_api: Any, thread_layout: Any
    ) -> None:
        """Explicit layouts require two positive integral worker counts."""
        prog = sampling_api.compile("M 0")
        with pytest.raises((TypeError, ValueError), match="thread_layout|incompatible"):
            sampling_api.sample(prog, 1, thread_layout=thread_layout)

    def test_sample_different_seeds(self, sampling_mode: CpuSamplingMode) -> None:
        """Different seeds produce different results."""
        prog = sampling_mode.compile("H 0\nM 0")
        result1 = sampling_mode.sample(prog, 100, seed=1)
        result2 = sampling_mode.sample(prog, 100, seed=2)
        # With 100 random bits, probability of match is 2^-100
        assert not np.array_equal(result1.measurements, result2.measurements)

    def test_sample_shape(self, sampling_mode: CpuSamplingMode) -> None:
        """Results have correct shape and type."""
        prog = sampling_mode.compile("H 0\nM 0\nH 1\nM 1")
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)
        assert isinstance(result.measurements, np.ndarray)
        assert result.measurements.dtype == np.uint8
        assert result.measurements.shape == (SMALL_CIRCUIT_SHOTS, 2)
        # No detectors/observables in this circuit
        assert result.detectors.shape == (SMALL_CIRCUIT_SHOTS, 0)
        assert result.observables.shape == (SMALL_CIRCUIT_SHOTS, 0)

    def test_sample_reset_works(self, sampling_mode: CpuSamplingMode) -> None:
        """Reset correctly resets to |0>."""
        prog = sampling_mode.compile("""
            X 0
            R 0
            M 0
        """)
        result = sampling_mode.sample(prog, 100, seed=42)
        # Only one visible measurement (from M 0, after reset)
        # R's internal measurement is hidden, matching Stim behavior
        assert result.measurements.shape == (
            100,
            1,
        ), f"Expected 1 visible measurement, got {result.measurements.shape}"
        # Measurement after reset should always be 0
        assert np.all(result.measurements[:, 0] == 0), "Reset failed"

    def test_sample_mr_visible(self, sampling_mode: CpuSamplingMode) -> None:
        """MR (measure-and-reset) produces visible measurement unlike R."""
        # R produces 0 visible measurements, MR produces 1
        prog_r = sampling_mode.compile("R 0")
        prog_mr = sampling_mode.compile("MR 0")

        assert prog_r.num_measurements == 0, "R should have 0 visible measurements"
        assert prog_mr.num_measurements == 1, "MR should have 1 visible measurement"

        # MR on |0> should always measure 0
        result = sampling_mode.sample(prog_mr, 100, seed=42)
        assert result.measurements.shape == (100, 1)
        assert np.all(result.measurements == 0), "MR on |0> should always measure 0"

        # MR after X should measure 1
        prog = sampling_mode.compile("X 0\nMR 0")
        result = sampling_mode.sample(prog, 100, seed=42)
        assert np.all(result.measurements == 1), "MR after X should measure 1"

    def test_gap_sampling_sparse_errors(self, sampling_mode: CpuSamplingMode) -> None:
        """Verify geometric gap sampling correctly models independent errors."""
        # 50 qubits (within the default build limit), each has a 2% chance
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

        prog = sampling_mode.compile("\n".join(lines))
        result = sampling_mode.sample(prog, shots, seed=42)

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

    def test_active_measurement_feedback_and_reset(self, sampling_mode: CpuSamplingMode) -> None:
        """A measured bit changes a later rotation on a still-active qubit."""
        program = sampling_mode.compile(
            "H 0 1 2 3 4\nT 0 1 2 3 4\nEXP_VAL X0*X1*X2*X3*X4\n"
            "H 0\nM 0\nCX rec[-1] 1\nR_Z(0.125) 1\nEXP_VAL X1\nR 2\nM 2\n"
            "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
        )
        # Measurement removes one coordinate before the remaining active rotation.
        assert program.peak_active_width == 5
        shots = 257
        result = sampling_mode.sample(program, shots, seed=1912)
        first = result.measurements[:, 0]
        probability = (1 - np.sqrt(0.5)) / 2
        assert 0 < first.sum() < shots
        assert abs(first.mean() - probability) < binomial_tolerance(probability, shots)
        np.testing.assert_array_equal(result.measurements[:, 1], np.zeros(shots))
        np.testing.assert_array_equal(result.detectors, np.zeros((shots, 1)))
        np.testing.assert_array_equal(result.observables[:, 0], first)
        # Conditional X reverses the T|+> azimuth before Rz adds pi/8.
        azimuth = np.where(first == 0, np.pi / 4, -np.pi / 4)
        expected = np.column_stack((np.full(shots, 2**-2.5), np.cos(azimuth + np.pi / 8)))
        np.testing.assert_allclose(result.exp_vals, expected, atol=1e-12, rtol=0)


class TestStatevector:
    """Tests for clifft.get_statevector()."""

    def test_statevector_pure_clifford(self) -> None:
        """Pure Clifford circuit matches expected statevector."""
        # H|0> = |+> = [1/sqrt(2), 1/sqrt(2)]
        prog = clifft.compile("H 0")
        sv = clifft.get_statevector(prog)

        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_bell_state(self) -> None:
        """Bell state matches expected statevector."""
        prog = clifft.compile("H 0\nCX 0 1")
        sv = clifft.get_statevector(prog)

        # |Phi+> = (|00> + |11>)/sqrt(2)
        expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_single_t_gate(self) -> None:
        """H-T circuit: [1/sqrt(2), e^{ipi/4}/sqrt(2)]."""
        prog = clifft.compile("H 0\nT 0")
        sv = clifft.get_statevector(prog)

        expected = np.array([1 / np.sqrt(2), np.exp(1j * np.pi / 4) / np.sqrt(2)], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_t_dagger(self) -> None:
        """H-T_dag circuit: [1/sqrt(2), e^{-ipi/4}/sqrt(2)]."""
        prog = clifft.compile("H 0\nT_DAG 0")
        sv = clifft.get_statevector(prog)

        expected = np.array([1 / np.sqrt(2), np.exp(-1j * np.pi / 4) / np.sqrt(2)], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_two_t_equals_s(self) -> None:
        """T-T = S: H-T-T should equal H-S."""
        prog = clifft.compile("H 0\nT 0\nT 0")
        sv = clifft.get_statevector(prog)

        # H-S: [1/sqrt(2), i/sqrt(2)]
        expected = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_four_t_equals_z(self) -> None:
        """T^4 = Z: H-T-T-T-T should equal H-Z."""
        prog = clifft.compile("H 0\nT 0\nT 0\nT 0\nT 0")
        sv = clifft.get_statevector(prog)

        # H-Z: [1/sqrt(2), -1/sqrt(2)]
        expected = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_t_on_zero(self) -> None:
        """T|0> = |0> (global phase only)."""
        prog = clifft.compile("T 0")
        sv = clifft.get_statevector(prog)

        # T|0> = |0> up to global phase
        expected = np.array([1, 0], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_two_qubit_t(self) -> None:
        """Two-qubit circuit with T on qubit 0."""
        prog = clifft.compile("H 0\nH 1\nT 0")
        sv = clifft.get_statevector(prog)

        # T on q0 affects indices where bit 0 is set (indices 1, 3)
        phase = np.exp(1j * np.pi / 4)
        expected = np.array([0.5, 0.5 * phase, 0.5, 0.5 * phase], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_bell_plus_t(self) -> None:
        """Bell state with T on control qubit."""
        prog = clifft.compile("H 0\nCX 0 1\nT 0")
        sv = clifft.get_statevector(prog)

        # Bell state: (|00> + |11>)/sqrt(2)
        # T on q0: |00>->|00>, |11>->e^{ipi/4}|11>
        phase = np.exp(1j * np.pi / 4)
        expected = np.array([1 / np.sqrt(2), 0, 0, phase / np.sqrt(2)], dtype=complex)
        assert_statevectors_equiv(sv, expected)

    def test_statevector_normalized(self) -> None:
        """Statevector is always normalized."""
        circuits = [
            "H 0\nT 0",
            "H 0\nH 1\nT 0\nT 1",
            "H 0\nCX 0 1\nT 0",
            "H 0\nT 0\nT 0\nT 0",
        ]
        for circuit in circuits:
            prog = clifft.compile(circuit)
            sv = clifft.get_statevector(prog)
            norm = float(np.sqrt(np.sum(np.abs(sv) ** 2)))
            assert abs(norm - 1.0) < 1e-12, f"Not normalized: {circuit}"

    def test_statevector_clifford_amplitudes_keep_double_precision(self) -> None:
        """Tableau expansion does not retain Stim's float-rounded amplitudes."""
        sv = clifft.get_statevector(clifft.compile("H 0"))
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
        np.testing.assert_allclose(sv, expected, atol=1e-15, rtol=0)


class TestCliffordValidation:
    """Validate exact Clifford evolution against Stim."""

    def test_random_clifford_single_qubit(self, statevector_from_circuit: Any) -> None:
        """Random 1-qubit Clifford circuits match Stim."""
        import stim

        for seed in range(10):
            circuit_str = random_clifford_circuit(1, 5, seed)

            clifft_sv = statevector_from_circuit(circuit_str)

            # Stim statevector
            stim_circuit = stim.Circuit(circuit_str)
            sim = stim.TableauSimulator()
            sim.do_circuit(stim_circuit)
            stim_sv = sim.state_vector(endian="little")

            assert_statevectors_equiv(clifft_sv, stim_sv, atol=1e-7, msg=f"circuit:\n{circuit_str}")

    def test_random_clifford_multi_qubit(self, statevector_from_circuit: Any) -> None:
        """Random 2-4 qubit Clifford circuits match Stim."""
        import stim

        for num_qubits in [2, 3, 4]:
            for seed in range(5):
                circuit_str = random_clifford_circuit(num_qubits, 10, seed)

                clifft_sv = statevector_from_circuit(circuit_str)

                # Stim statevector
                stim_circuit = stim.Circuit(circuit_str)
                sim = stim.TableauSimulator()
                sim.do_circuit(stim_circuit)
                stim_sv = sim.state_vector(endian="little")

                assert_statevectors_equiv(
                    clifft_sv,
                    stim_sv,
                    atol=1e-7,
                    msg=f"{num_qubits}q circuit:\n{circuit_str}",
                )


class TestSamplingValidation:
    """Validate sampling distributions against Stim."""

    def test_deterministic_clifford_sampling(self, sampling_mode: CpuSamplingMode) -> None:
        """Deterministic measurements match Stim exactly."""
        import stim

        # Circuit where measurements have deterministic outcomes
        circuits = [
            "M 0",  # |0> always gives 0
            "X 0\nM 0",  # |1> always gives 1
            "H 0\nCX 0 1\nM 0\nM 1",  # Bell state: correlated
        ]

        for circuit_str in circuits:
            # Clifft sampling
            prog = sampling_mode.compile(circuit_str)
            result = sampling_mode.sample(prog, 100, seed=42)

            # Stim sampling (seed is in compile_sampler, not sample)
            stim_circuit = stim.Circuit(circuit_str)
            stim_sampler = stim_circuit.compile_sampler(seed=42)
            stim_results = stim_sampler.sample(100)

            # For deterministic circuits, all shots should match
            # (Note: seeds may differ, but deterministic outcomes should be consistent)
            if prog.num_measurements == 1:
                # Single measurement: check value consistency
                clifft_vals = set(tuple(r) for r in result.measurements)
                stim_vals = set(tuple(r) for r in stim_results)
                assert clifft_vals == stim_vals, f"Mismatch for: {circuit_str}"
            else:
                # Multi-measurement: check correlation structure
                for clifft_shot in result.measurements:
                    if "CX" in circuit_str:  # Bell state
                        assert clifft_shot[0] == clifft_shot[1], "Bell correlation broken"

    def test_statistical_distribution_h(self, sampling_mode: CpuSamplingMode) -> None:
        """H gate sampling matches Stim statistically."""
        import stim

        circuit_str = "H 0\nM 0"
        shots = 10000

        # Clifft sampling
        prog = sampling_mode.compile(circuit_str)
        result = sampling_mode.sample(prog, shots, seed=12345)
        clifft_p0 = np.mean(result.measurements[:, 0] == 0)

        # Stim sampling (seed is in compile_sampler, not sample)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=54321)
        stim_results = stim_sampler.sample(shots)
        stim_p0 = np.mean(stim_results[:, 0] == 0)

        # Both should be close to 0.5, and close to each other
        tolerance = binomial_tolerance(0.5, shots)
        assert (
            abs(clifft_p0 - 0.5) < tolerance
        ), f"Clifft p0={clifft_p0} outside {tolerance:.4f} tol"
        assert abs(stim_p0 - 0.5) < tolerance, f"Stim p0={stim_p0} outside {tolerance:.4f} tol"
        # For comparing two independent estimates, variance adds: 2*std_err
        cross_tolerance = 2 * tolerance
        assert (
            abs(clifft_p0 - stim_p0) < cross_tolerance
        ), f"Clifft vs Stim: {clifft_p0} vs {stim_p0}"

    def test_statistical_distribution_bell(self, sampling_mode: CpuSamplingMode) -> None:
        """Bell state sampling matches Stim statistically."""
        import stim

        circuit_str = "H 0\nCX 0 1\nM 0\nM 1"
        shots = 10000

        # Clifft sampling
        prog = sampling_mode.compile(circuit_str)
        result = sampling_mode.sample(prog, shots, seed=999)
        clifft_00 = np.mean((result.measurements[:, 0] == 0) & (result.measurements[:, 1] == 0))
        clifft_11 = np.mean((result.measurements[:, 0] == 1) & (result.measurements[:, 1] == 1))

        # Stim sampling (seed is in compile_sampler, not sample)
        stim_circuit = stim.Circuit(circuit_str)
        stim_sampler = stim_circuit.compile_sampler(seed=888)
        stim_results = stim_sampler.sample(shots)
        stim_00 = np.mean((stim_results[:, 0] == 0) & (stim_results[:, 1] == 0))
        stim_11 = np.mean((stim_results[:, 0] == 1) & (stim_results[:, 1] == 1))

        # Bell state: 50% |00>, 50% |11>
        tolerance = binomial_tolerance(0.5, shots)
        assert (
            abs(clifft_00 - 0.5) < tolerance
        ), f"Clifft |00>={clifft_00} outside {tolerance:.4f} tol"
        assert (
            abs(clifft_11 - 0.5) < tolerance
        ), f"Clifft |11>={clifft_11} outside {tolerance:.4f} tol"
        assert abs(stim_00 - 0.5) < tolerance, f"Stim |00>={stim_00} outside {tolerance:.4f} tol"
        assert abs(stim_11 - 0.5) < tolerance, f"Stim |11>={stim_11} outside {tolerance:.4f} tol"

    def test_meas_active_interfere_y_observable(self, sampling_mode: CpuSamplingMode) -> None:
        """An active Y measurement correctly computes phase-sensitive interference."""
        # H 0; T 0 rotates the state to (|0> + e^{ipi/4}|1>)/sqrt(2)
        # S 0 adds phase: (|0> + e^{i*3pi/4}|1>)/sqrt(2)
        # MX 0 plans MEASURE_ACTIVE with a rewound Y observable.
        circuit = "H 0\nT 0\nS 0\nMX 0"
        prog = sampling_mode.compile(circuit)

        # P(+) = |<+|psi>|^2 = |1 + e^{i3pi/4}|^2 / 4
        #      = (2 - sqrt(2)) / 4 ~ 0.1464
        shots = 10000
        expected_p0 = (2 - np.sqrt(2)) / 4  # ~ 0.1464
        result = sampling_mode.sample(prog, shots, seed=42)
        p0 = float(np.mean(result.measurements[:, 0] == 0))
        tolerance = binomial_tolerance(expected_p0, shots)

        assert abs(p0 - expected_p0) < tolerance, (
            f"Y-measurement interference failed: p0={p0}, "
            f"expected {expected_p0:.4f} +/- {tolerance:.4f}"
        )


class TestNoiseAndQEC:
    """Tests for noise simulation and QEC features."""

    def test_sample_returns_sample_result(self, sampling_mode: CpuSamplingMode) -> None:
        """sample() returns a SampleResult with attribute access and unpacking."""
        prog = sampling_mode.compile("H 0\nM 0")
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)
        assert isinstance(result, clifft.SampleResult)
        # Attribute access
        assert result.measurements.shape == (SMALL_CIRCUIT_SHOTS, 1)
        assert result.detectors.shape == (SMALL_CIRCUIT_SHOTS, 0)
        assert result.observables.shape == (SMALL_CIRCUIT_SHOTS, 0)
        # Tuple unpacking still works
        meas, det, obs = result
        assert meas.shape == (SMALL_CIRCUIT_SHOTS, 1)
        assert det.shape == (SMALL_CIRCUIT_SHOTS, 0)
        assert obs.shape == (SMALL_CIRCUIT_SHOTS, 0)

    def test_program_detector_observable_counts(self, sampling_api: Any) -> None:
        """Program reports correct detector and observable counts."""
        prog = sampling_api.compile("""
            H 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """)
        assert prog.num_measurements == 1
        assert prog.num_detectors == 1
        assert prog.num_observables == 1

    def test_detector_computes_parity(self, sampling_mode: CpuSamplingMode) -> None:
        """DETECTOR computes XOR of referenced measurements."""
        # Bell state: M 0 and M 1 always match, so XOR = 0
        prog = sampling_mode.compile("""
            H 0
            CX 0 1
            M 0
            M 1
            DETECTOR rec[-1] rec[-2]
        """)
        result = sampling_mode.sample(prog, 100, seed=42)
        # All detectors should be 0 (perfect correlation)
        assert np.all(result.detectors == 0)

    def test_observable_accumulates_xor(self, sampling_mode: CpuSamplingMode) -> None:
        """Multiple OBSERVABLE_INCLUDE to same index XOR together."""
        # Bell state: two identical measurements XOR to 0
        prog = sampling_mode.compile("""
            H 0
            CX 0 1
            M 0
            M 1
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-2]
        """)
        result = sampling_mode.sample(prog, 100, seed=42)
        # Observable is XOR of two identical bits = 0
        assert np.all(result.observables == 0)

    def test_observable_tracks_logical_value(self, sampling_mode: CpuSamplingMode) -> None:
        """Observable correctly tracks logical qubit value."""
        prog = sampling_mode.compile("""
            H 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        """)
        result = sampling_mode.sample(prog, 100, seed=42)
        # Observable should equal measurement (single reference)
        assert np.array_equal(result.measurements[:, 0], result.observables[:, 0])

    def test_readout_noise_flips_bits(self, sampling_mode: CpuSamplingMode) -> None:
        """M(p) readout noise flips measurement results."""
        # 100% readout noise flips |0> -> measured as 1
        prog = sampling_mode.compile("M(1.0) 0")
        result = sampling_mode.sample(prog, 100, seed=42)
        assert np.all(result.measurements == 1)

    def test_readout_noise_probabilistic(self, sampling_mode: CpuSamplingMode) -> None:
        """M(0.5) readout noise gives ~50% flip rate."""
        prog = sampling_mode.compile("M(0.5) 0")
        result = sampling_mode.sample(prog, 1000, seed=42)
        flip_rate = float(np.mean(result.measurements))
        # Should be ~50% (measuring |0> with 50% flip = 50% ones)
        tolerance = binomial_tolerance(0.5, 1000)
        assert abs(flip_rate - 0.5) < tolerance

    def test_pauli_noise_x_error(self, sampling_mode: CpuSamplingMode) -> None:
        """X_ERROR(1.0) always flips qubit."""
        prog = sampling_mode.compile("""
            X_ERROR(1.0) 0
            M 0
        """)
        result = sampling_mode.sample(prog, 100, seed=42)
        # X flips |0> to |1>
        assert np.all(result.measurements == 1)

    def test_correlated_error_else_branch(self, sampling_mode: CpuSamplingMode) -> None:
        """ELSE_CORRELATED_ERROR fires when the earlier link does not."""
        prog = sampling_mode.compile("""
            E(0.0) X0
            ELSE_CORRELATED_ERROR(1.0) X1
            M 0 1
        """)
        result = sampling_mode.sample(prog, 100, seed=42)
        assert np.all(result.measurements[:, 0] == 0)
        assert np.all(result.measurements[:, 1] == 1)

    def test_correlated_error_chain_probabilistic(self, sampling_mode: CpuSamplingMode) -> None:
        """Correlated-error chains convert conditional probabilities correctly."""
        prog = sampling_mode.compile("""
            E(0.5) X0
            ELSE_CORRELATED_ERROR(0.5) X1
            M 0 1
        """)
        shots = 5000
        result = sampling_mode.sample(prog, shots, seed=42)
        q0 = result.measurements[:, 0]
        q1 = result.measurements[:, 1]

        q0_rate = float(np.mean(q0))
        q1_rate = float(np.mean(q1))
        assert abs(q0_rate - 0.5) < binomial_tolerance(0.5, shots)
        assert abs(q1_rate - 0.25) < binomial_tolerance(0.25, shots)
        assert not np.any(q0 & q1)

    def test_pauli_noise_z_error(self, sampling_mode: CpuSamplingMode) -> None:
        """Z_ERROR doesn't affect computational basis measurement."""
        prog = sampling_mode.compile("""
            Z_ERROR(1.0) 0
            M 0
        """)
        result = sampling_mode.sample(prog, 100, seed=42)
        # Z|0> = |0>, so still measure 0
        assert np.all(result.measurements == 0)

    @pytest.mark.parametrize(
        "circuit,expected",
        [
            ("PAULI_CHANNEL_1(1, 0, 0) 0\nM 0", [1]),
            (
                "PAULI_CHANNEL_2(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1\nM 0 1",
                [1, 0],
            ),
            (
                f"H 0 1 2\nPAULI_CHANNEL_3({', '.join(['0'] * 62 + ['1'])}) 0 1 2\nMX 0 1 2",
                [1, 1, 1],
            ),
        ],
    )
    def test_pauli_channels(
        self, sampling_mode: CpuSamplingMode, circuit: str, expected: list[int]
    ) -> None:
        """Explicit one-, two-, and three-qubit Pauli channels execute."""
        result = sampling_mode.sample(
            sampling_mode.compile(circuit), shots=SMALL_CIRCUIT_SHOTS, seed=42
        )
        expected_rows = np.tile(expected, (SMALL_CIRCUIT_SHOTS, 1))
        np.testing.assert_array_equal(result.measurements, expected_rows)

    def test_depolarize1_probabilistic(self, sampling_mode: CpuSamplingMode) -> None:
        """DEPOLARIZE1 applies X, Y, or Z with equal probability."""
        prog = sampling_mode.compile("""
            DEPOLARIZE1(1.0) 0
            M 0
        """)
        result = sampling_mode.sample(prog, 3000, seed=42)
        # X and Y flip |0>->|1>, Z doesn't. Expected: 2/3 ones.
        ones_rate = float(np.mean(result.measurements))
        expected = 2.0 / 3.0
        tolerance = binomial_tolerance(expected, 3000)
        assert abs(ones_rate - expected) < tolerance

    def test_noise_detector_interaction(self, sampling_mode: CpuSamplingMode) -> None:
        """Noise causes detector to fire."""
        # Two measurements with X_ERROR in between
        # First M gives 0, X_ERROR flips, second M gives 1
        # Detector XORs them: 0 XOR 1 = 1
        prog = sampling_mode.compile("""
            M 0
            X_ERROR(1.0) 0
            M 0
            DETECTOR rec[-1] rec[-2]
        """)
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)
        # First meas = 0, second meas = 1, detector = 1
        assert np.all(result.measurements[:, 0] == 0)
        assert np.all(result.measurements[:, 1] == 1)
        assert np.all(result.detectors[:, 0] == 1)

    def test_sample_shape_with_qec(self, sampling_mode: CpuSamplingMode) -> None:
        """sample() returns correct shapes with detectors/observables."""
        prog = sampling_mode.compile("""
            H 0
            M 0
            M 1
            DETECTOR rec[-1]
            DETECTOR rec[-2]
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(1) rec[-2]
        """)
        shots = SMALL_CIRCUIT_SHOTS
        result = sampling_mode.sample(prog, shots=shots, seed=0)
        assert result.measurements.shape == (shots, 2)
        assert result.detectors.shape == (shots, 3)
        assert result.observables.shape == (shots, 2)


class TestPostselection:
    """Tests for compile() with postselection_mask."""

    def test_compile_with_postselection_mask(self, sampling_api: Any) -> None:
        """Compile with postselection_mask kwarg works via nanobind."""
        prog = sampling_api.compile(
            "M 0\nDETECTOR rec[-1]\n",
            postselection_mask=[1],
        )
        assert prog.num_detectors == 1
        assert prog.num_measurements == 1

    def test_has_postselection_flag(self, sampling_api: Any) -> None:
        """has_postselection is True when mask is non-trivial, False otherwise."""
        circuit = "M 0\nDETECTOR rec[-1]\n"
        prog_no_mask = sampling_api.compile(circuit)
        assert prog_no_mask.has_postselection is False

        prog_zero_mask = sampling_api.compile(circuit, postselection_mask=[0])
        assert prog_zero_mask.has_postselection is False

        prog_with_mask = sampling_api.compile(circuit, postselection_mask=[1])
        assert prog_with_mask.has_postselection is True

    def test_sample_raises_on_postselected_program(self, sampling_mode: CpuSamplingMode) -> None:
        """sample() raises ValueError when program has postselection."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
        """
        prog = sampling_mode.compile(circuit, postselection_mask=[1])
        with pytest.raises(ValueError, match="sample_survivors"):
            sampling_mode.sample(prog, 10)

    def test_sample_k_raises_on_postselected_program(self) -> None:
        """sample_k() raises ValueError when program has postselection."""
        circuit = """
            R 0
            DEPOLARIZE1(0.1) 0
            M 0
            DETECTOR rec[-1]
        """
        prog = clifft.compile(circuit, postselection_mask=[1])
        with pytest.raises(ValueError, match="sample_k_survivors"):
            clifft.sample_k(prog, 10, k=1)

    def test_sample_ok_without_postselection(self, sampling_mode: CpuSamplingMode) -> None:
        """sample() works fine when program has no postselection."""
        circuit = "M 0\nDETECTOR rec[-1]\n"
        prog = sampling_mode.compile(circuit)
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=42)
        assert result.detectors.shape == (SMALL_CIRCUIT_SHOTS, 1)

    def test_empty_mask_is_default(self, sampling_api: Any) -> None:
        """Empty postselection_mask produces same result as no mask."""
        circuit = "M 0\nDETECTOR rec[-1]\n"
        prog_default = sampling_api.compile(circuit)
        prog_empty = sampling_api.compile(circuit, postselection_mask=[])
        assert prog_default.num_actions == prog_empty.num_actions


class TestSampleSurvivors:
    """Tests for sample_survivors() API."""

    @pytest.mark.parametrize("keep_records", [False, True])
    def test_active_postselection_preserves_survivor_outputs(
        self, sampling_mode: CpuSamplingMode, keep_records: bool
    ) -> None:
        """Survivor accounting and retained probes agree with independent expectations."""
        program = sampling_mode.compile(
            "H 0 1 2 3 4\nT 0 1 2 3 4\nEXP_VAL X0*X1*X2*X3*X4\n"
            "R_X(0.125) 4\nEXP_VAL Z4\nM 0\nDETECTOR rec[-1]\n"
            "H 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nEXP_VAL Z1",
            postselection_mask=[1],
        )
        assert program.peak_active_width == 5
        shots = 257
        result = sampling_mode.sample_survivors(
            program, shots, seed=1913, keep_records=keep_records
        )
        passed = result.passed_shots
        assert result.total_shots == shots
        assert result.discards == shots - passed
        assert 0 < passed < shots
        assert abs(passed / shots - 0.5) < binomial_tolerance(0.5, shots)
        # Qubit 1 is independent of the postselected qubit and measures H T |+>.
        probability = (1 - np.sqrt(0.5)) / 2
        assert 0 < result.logical_errors < passed
        assert abs(result.logical_errors / passed - probability) < binomial_tolerance(
            probability, passed
        )
        np.testing.assert_array_equal(result.observable_ones, [result.logical_errors])
        rows = passed if keep_records else 0
        assert result.measurements.shape == (rows, 2)
        assert result.detectors.shape == (rows, 1)
        assert result.observables.shape == (rows, 1)
        assert result.exp_vals.shape == (rows, 3)
        if keep_records:
            np.testing.assert_array_equal(result.measurements[:, 0], np.zeros(passed))
            np.testing.assert_array_equal(result.detectors, np.zeros((passed, 1)))
            np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, 1])
            assert result.observables.sum() == result.logical_errors
            expected = [2**-2.5, np.sin(np.pi / 8) / np.sqrt(2)]
            np.testing.assert_allclose(
                result.exp_vals[:, :2], np.broadcast_to(expected, (passed, 2)), atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(
                result.exp_vals[:, 2],
                1 - 2 * result.measurements[:, 1].astype(int),
                atol=1e-12,
                rtol=0,
            )

    def test_zero_shots_returns_empty_result(self, sampling_mode: CpuSamplingMode) -> None:
        """Zero shots preserves the existing empty-result contract."""
        program = sampling_mode.compile("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
        result = sampling_mode.sample_survivors(program, 0, seed=42)

        assert result.total_shots == 0
        assert result.passed_shots == 0
        assert len(result.observable_ones) == 0
        assert result.measurements.shape == (0, 1)
        assert result.detectors.shape == (0, 0)
        assert result.observables.shape == (0, 1)

    def test_counting_only_fast_path(self, sampling_mode: CpuSamplingMode) -> None:
        """keep_records=False returns survivor metadata with empty arrays."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = sampling_mode.compile(circuit, postselection_mask=[1])
        result = sampling_mode.sample_survivors(prog, 1000, seed=42)

        assert result.total_shots == 1000
        assert 0 < result.passed_shots < 1000
        assert result.discards == 1000 - result.passed_shots
        # Survivors have meas[0]==0, so observable parity is always 0
        assert result.observable_ones[0] == 0
        assert result.logical_errors == 0
        assert isinstance(result, clifft.SampleResult)
        assert result.measurements.shape == (0, prog.num_measurements)
        assert result.detectors.shape == (0, prog.num_detectors)
        assert result.observables.shape == (0, prog.num_observables)

    def test_keep_records_returns_arrays(self, sampling_mode: CpuSamplingMode) -> None:
        """keep_records=True populates survivor measurement and syndrome arrays."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = sampling_mode.compile(circuit, postselection_mask=[1])
        result = sampling_mode.sample_survivors(prog, 200, seed=42, keep_records=True)

        passed = result.passed_shots
        assert passed > 0
        assert result.measurements.shape == (passed, prog.num_measurements)
        assert result.detectors.shape == (passed, 1)
        assert result.observables.shape == (passed, 1)
        assert np.all(result.measurements[:, 0] == 0)
        assert np.all(result.detectors[:, 0] == 0)
        # All surviving observables should be 0
        assert np.all(result.observables == 0)

    def test_aggregate_matches_retained_rows(self, sampling_mode: CpuSamplingMode) -> None:
        """Counts-only output exactly aggregates the retained rows."""
        circuit = """
            H 0 1 2
            M 0 1 2
            DETECTOR rec[-3]
            OBSERVABLE_INCLUDE(0) rec[-2]
            OBSERVABLE_INCLUDE(1) rec[-2] rec[-1]
        """
        program = sampling_mode.compile(circuit, postselection_mask=[1])
        rows = sampling_mode.sample_survivors(program, 513, seed=91842, keep_records=True)
        aggregate = sampling_mode.sample_survivors(program, 513, seed=91842, keep_records=False)

        assert aggregate.total_shots == rows.total_shots
        assert aggregate.passed_shots == rows.passed_shots
        assert aggregate.discards == rows.discards
        expected_ones = np.sum(rows.observables, axis=0, dtype=np.uint64)
        expected_logical_errors = int(np.count_nonzero(np.any(rows.observables, axis=1)))
        np.testing.assert_array_equal(aggregate.observable_ones, expected_ones)
        assert aggregate.logical_errors == expected_logical_errors
        assert aggregate.measurements.shape == (0, program.num_measurements)
        assert aggregate.detectors.shape == (0, program.num_detectors)
        assert aggregate.observables.shape == (0, program.num_observables)

    def test_survivors_replay_seeded_rows(self, sampling_mode: CpuSamplingMode) -> None:
        prog = sampling_mode.compile(
            "H 0\nM 0\nDETECTOR rec[-1]\nH 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
            postselection_mask=[1],
        )
        first = sampling_mode.sample_survivors(prog, 257, seed=43, keep_records=True)
        replay = sampling_mode.sample_survivors(prog, 257, seed=43, keep_records=True)
        assert first.passed_shots == replay.passed_shots
        assert first.logical_errors == replay.logical_errors
        np.testing.assert_array_equal(first.measurements, replay.measurements)
        np.testing.assert_array_equal(first.observables, replay.observables)

    @pytest.mark.parametrize("threads", [2, 3])
    @pytest.mark.parametrize("batch_size", [1, 65, "auto"])
    @pytest.mark.parametrize("keep_records", [False, True])
    def test_threads_preserve_survivor_results(
        self, sampling_api: Any, threads: int, batch_size: int | str, keep_records: bool
    ) -> None:
        """Survivor counts and retained row order are independent of worker schedules."""
        prog = sampling_api.compile(
            "H 0\nM 0\nDETECTOR rec[-1]\nH 1\nM 1\nEXP_VAL Z1\nOBSERVABLE_INCLUDE(0) rec[-1]",
            postselection_mask=[1],
        )
        # More packed batches give fast aggregate workers time to share the work.
        shots = 32 * 65 + 1 if batch_size == 65 else 257
        serial = sampling_api.sample_survivors(
            prog, shots, seed=54321, keep_records=keep_records, threads=1, batch_size=batch_size
        )
        threaded = sampling_api.sample_survivors(
            prog,
            shots,
            seed=54321,
            keep_records=keep_records,
            threads=threads,
            batch_size=batch_size,
        )
        assert 0 < serial.passed_shots < serial.total_shots
        assert threaded.total_shots == serial.total_shots
        assert threaded.passed_shots == serial.passed_shots
        assert threaded.discards == serial.discards
        assert threaded.logical_errors == serial.logical_errors
        np.testing.assert_array_equal(threaded.observable_ones, serial.observable_ones)
        np.testing.assert_array_equal(threaded.measurements, serial.measurements)
        np.testing.assert_array_equal(threaded.detectors, serial.detectors)
        np.testing.assert_array_equal(threaded.observables, serial.observables)
        np.testing.assert_array_equal(threaded.exp_vals, serial.exp_vals)

    def test_no_postselection_all_pass(self, sampling_mode: CpuSamplingMode) -> None:
        """Without postselection, all shots pass."""
        circuit = """
            H 0
            M 0
            DETECTOR rec[-1]
        """
        prog = sampling_mode.compile(circuit)  # no mask
        result = sampling_mode.sample_survivors(prog, 100, seed=42)

        assert result.total_shots == 100
        assert result.passed_shots == 100
        assert result.discards == 0

    def test_no_postselection_retains_all_records(self, sampling_mode: CpuSamplingMode) -> None:
        program = sampling_mode.compile(
            "X 1\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        )
        shots = 257
        result = sampling_mode.sample_survivors(program, shots, seed=42, keep_records=True)

        assert result.total_shots == shots
        assert result.passed_shots == shots
        assert result.discards == 0
        assert result.logical_errors == shots
        np.testing.assert_array_equal(result.observable_ones, [shots])
        np.testing.assert_array_equal(result.measurements, np.broadcast_to([0, 1], (shots, 2)))
        np.testing.assert_array_equal(result.detectors, np.zeros((shots, 1), dtype=np.uint8))
        np.testing.assert_array_equal(result.observables, np.ones((shots, 1), dtype=np.uint8))

    def test_observable_ones_counts_errors(self, sampling_mode: CpuSamplingMode) -> None:
        """observable_ones correctly counts logical errors in survivors."""
        # No postselection, random observable. ~50% should be 1.
        circuit = """
            H 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = sampling_mode.compile(circuit)
        result = sampling_mode.sample_survivors(prog, 10000, seed=42)

        assert result.passed_shots == 10000
        ones = int(result.observable_ones[0])
        shots = 10000
        tol = binomial_tolerance(0.5, shots) * shots
        assert (
            abs(ones - shots * 0.5) < tol
        ), f"Expected ~{shots * 0.5:.0f} ones, got {ones} (tol={tol:.0f})"

    def test_target_qec_circuit(self, sampling_mode: CpuSamplingMode) -> None:
        """Smoke test with the real d=3 MSC cultivation circuit."""
        import pathlib

        import stim

        circuit_path = pathlib.Path("tests/fixtures/target_qec.stim")
        stim_circuit = stim.Circuit.from_file(str(circuit_path))
        text = circuit_path.read_text()

        # Build postselection mask from coord[4] == -9
        coords = stim_circuit.get_detector_coordinates()
        mask = [0] * stim_circuit.num_detectors
        for k, v in coords.items():
            if len(v) >= 5 and v[4] == -9:
                mask[k] = 1

        prog = sampling_mode.compile(text, postselection_mask=mask)
        result = sampling_mode.sample_survivors(prog, 5000, seed=42)

        assert result.total_shots == 5000
        assert result.passed_shots > 0
        assert result.discards > 0
        assert len(result.observable_ones) == 1

    def test_keep_records_100_percent_discard(self, sampling_mode: CpuSamplingMode) -> None:
        """keep_records=True with all shots discarded returns empty arrays."""
        # Circuit: deterministic meas=1, postselect -> always discards
        circuit = """
            X 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = sampling_mode.compile(circuit, postselection_mask=[1])
        result = sampling_mode.sample_survivors(prog, 100, seed=42, keep_records=True)

        assert result.total_shots == 100
        assert result.passed_shots == 0
        assert result.discards == 100
        assert result.measurements.shape == (0, 1)
        assert result.detectors.shape == (0, 1)
        assert result.observables.shape == (0, 1)

    def test_logical_errors_multi_observable(self, sampling_mode: CpuSamplingMode) -> None:
        """logical_errors counts shots, not total observable flips."""
        # Two observables both keyed to same random measurement.
        circuit = """
            H 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(1) rec[-1]
        """
        prog = sampling_mode.compile(circuit)
        result = sampling_mode.sample_survivors(prog, 10000, seed=42)

        assert result.passed_shots == 10000
        # Both observables fire on same shots
        ones_0 = int(result.observable_ones[0])
        ones_1 = int(result.observable_ones[1])
        assert ones_0 == ones_1
        # logical_errors == per-shot count, not sum of per-observable
        assert result.logical_errors == ones_0


class TestSyndromeNormalization:
    def test_normalize_syndromes_multiple_observables_xord(
        self, sampling_mode: CpuSamplingMode
    ) -> None:
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
        prog_raw = sampling_mode.compile(circuit, normalize_syndromes=False)
        result_raw = sampling_mode.sample(prog_raw, shots=SMALL_CIRCUIT_SHOTS, seed=0)

        assert np.all(result_raw.detectors[:, 0] == 1)
        assert np.all(result_raw.detectors[:, 1] == 0)
        assert np.all(result_raw.observables[:, 0] == 1)
        assert np.all(result_raw.observables[:, 1] == 1)

        # 2. Normalized: All output parities must be strictly 0.
        # We also apply a postselection mask on DET 0 (which natively evaluates to 1).
        # Since it is normalized, it becomes 0, meaning shots should SURVIVE.
        prog_norm = sampling_mode.compile(
            circuit,
            normalize_syndromes=True,
            postselection_mask=[1, 0],
        )

        res = sampling_mode.sample_survivors(
            prog_norm, shots=SMALL_CIRCUIT_SHOTS, seed=0, keep_records=True
        )

        assert res.passed_shots == SMALL_CIRCUIT_SHOTS  # Normalized 1^1=0, so shots survive!
        assert np.all(res.detectors == 0)
        assert np.all(res.observables == 0)
        assert res.logical_errors == 0

    def test_normalize_syndromes_conflict_raises(self, sampling_api: Any) -> None:
        """Cannot provide explicit expected parities with normalize_syndromes=True."""
        import pytest

        circuit = "M 0\nDETECTOR rec[-1]\n"
        with pytest.raises(ValueError):
            sampling_api.compile(
                circuit,
                normalize_syndromes=True,
                expected_detectors=[0],
            )

    def test_normalize_syndromes_no_noise_passthrough(self, sampling_mode: CpuSamplingMode) -> None:
        """Normalization on a circuit without noise produces all-zero syndromes."""
        import numpy as np

        circuit = """
            X 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        prog = sampling_mode.compile(circuit, normalize_syndromes=True)
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)

        assert np.all(result.detectors == 0)
        assert np.all(result.observables == 0)

    def test_normalize_syndromes_with_noise_detects_errors(
        self, sampling_mode: CpuSamplingMode
    ) -> None:
        """With noise and normalization, errors show up as 1s in syndromes."""
        import numpy as np

        # A circuit where noise can flip the detector
        circuit = """
            X_ERROR(1.0) 0
            M 0
            DETECTOR rec[-1]
        """
        prog = sampling_mode.compile(circuit, normalize_syndromes=True)
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)

        # With 100% X error, measurement flips from 0 to 1.
        # Reference (noiseless) detector parity = 0.
        # Noisy parity = 1. Normalized: 1 ^ 0 = 1 (error detected).
        assert np.all(result.detectors[:, 0] == 1)

    def test_explicit_expected_detectors(self, sampling_mode: CpuSamplingMode) -> None:
        """Explicit expected_detectors without normalize_syndromes works."""
        import numpy as np

        circuit = """
            X 0
            M 0
            DETECTOR rec[-1]
        """
        # Raw detector parity = 1. With expected_detectors=[1], normalized = 0.
        prog = sampling_mode.compile(circuit, expected_detectors=[1])
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)

        assert np.all(result.detectors[:, 0] == 0)

    def test_explicit_expected_observables(self, sampling_mode: CpuSamplingMode) -> None:
        """Explicit expected_observables without normalize_syndromes works."""
        import numpy as np

        circuit = """
            X 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        # Raw obs = 1. With expected_observables=[1], normalized = 0.
        prog = sampling_mode.compile(circuit, expected_observables=[1])
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)

        assert np.all(result.observables[:, 0] == 0)

    def test_compute_reference_syndrome_api(self) -> None:
        """compute_reference_syndrome is accessible from Python."""
        circuit = clifft.parse("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
        hir = clifft.trace(circuit)
        ref = clifft.compute_reference_syndrome(hir)

        assert ref["detectors"] == [1]
        assert ref["observables"] == [1]

    def test_remove_noise_pass_api(self) -> None:
        """RemoveNoisePass is accessible from Python."""
        circuit = clifft.parse("X_ERROR(0.1) 0\nM 0\n")
        hir = clifft.trace(circuit)

        original_count = hir.num_ops
        strip = clifft.RemoveNoisePass()

        pm = clifft.HirPassManager()
        pm.add(strip)
        pm.run(hir)

        assert hir.num_ops < original_count
        for op in hir:
            assert op.op_type != clifft.OpType.NOISE


class TestExpVal:
    """Tests for EXP_VAL expectation value probes via Python bindings."""

    def test_sample_returns_exp_vals(self, sampling_mode: CpuSamplingMode) -> None:
        """sample() populates exp_vals for circuits with EXP_VAL."""
        prog = sampling_mode.compile("EXP_VAL Z0")
        shots = SMALL_CIRCUIT_SHOTS
        result = sampling_mode.sample(prog, shots=shots, seed=42)
        assert result.exp_vals.shape == (shots, 1)
        assert result.exp_vals.dtype == np.float64
        np.testing.assert_allclose(result.exp_vals[:, 0], 1.0, atol=1e-12)

    def test_no_exp_val_gives_empty(self, sampling_mode: CpuSamplingMode) -> None:
        """Circuits without EXP_VAL have shape (shots, 0) exp_vals."""
        prog = sampling_mode.compile("H 0\nM 0")
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)
        assert result.exp_vals.shape == (SMALL_CIRCUIT_SHOTS, 0)
        assert prog.num_exp_vals == 0

    def test_program_num_exp_vals(self) -> None:
        """Program.num_exp_vals reports the correct count."""
        prog = clifft.compile("EXP_VAL X0 Z1")
        assert prog.num_exp_vals == 2

    def test_hir_num_exp_vals(self) -> None:
        """HirModule.num_exp_vals reports the correct count."""
        hir = clifft.trace(clifft.parse("EXP_VAL X0*Y1 Z2"))
        assert hir.num_exp_vals == 2

    def test_exp_val_multiple_probes(self, sampling_mode: CpuSamplingMode) -> None:
        """Multiple EXP_VAL probes return consecutive columns."""
        prog = sampling_mode.compile("H 0\nEXP_VAL X0\nEXP_VAL Z0")
        result = sampling_mode.sample(prog, shots=SMALL_CIRCUIT_SHOTS, seed=0)
        assert result.exp_vals.shape == (SMALL_CIRCUIT_SHOTS, 2)
        np.testing.assert_allclose(result.exp_vals[:, 0], 1.0, atol=1e-12)  # <X> on |+>
        np.testing.assert_allclose(result.exp_vals[:, 1], 0.0, atol=1e-12)  # <Z> on |+>

    def test_exp_val_does_not_disturb_measurement(self, sampling_mode: CpuSamplingMode) -> None:
        """EXP_VAL is non-destructive: measurements after it are unaffected."""
        prog = sampling_mode.compile("EXP_VAL Z0\nM 0")
        result = sampling_mode.sample(prog, 100, seed=0)
        # |0> state: all measurements should be 0
        assert np.all(result.measurements == 0)
        np.testing.assert_allclose(result.exp_vals[:, 0], 1.0, atol=1e-12)
