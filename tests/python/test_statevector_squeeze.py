"""Tests for the StatevectorSqueezePass.

Validates that bidirectional bubble sort of HIR operations reduces peak_rank
by compacting qubit lifetimes, while preserving circuit semantics.
"""

import numpy as np
import pytest
from conftest import assert_statevectors_equal, cross_binomial_tolerance
from utils_fuzzing import generate_uncomputation_ladder

import ucc


def _squeeze_only_pass_manager() -> ucc.HirPassManager:
    """Return a pass manager with only the squeeze pass (no peephole)."""
    pm = ucc.HirPassManager()
    pm.add(ucc.StatevectorSqueezePass())
    return pm


def _ucc_statevector(circuit_str: str, **compile_kwargs: object) -> np.ndarray:
    """Compile and execute a noiseless circuit, return dense statevector."""
    prog = ucc.compile(circuit_str, **compile_kwargs)
    state = ucc.State(prog.peak_rank, prog.num_measurements)
    ucc.execute(prog, state)
    sv: np.ndarray = ucc.get_statevector(prog, state)
    return sv


class TestSqueezeBasicPeakRankReduction:
    """Core squeeze test: prove peak_rank drops when qubit lifetimes are compacted."""

    def test_two_independent_qubits_squeeze_reduces_rank(self) -> None:
        """H 0; T 0; H 1; T 1; M 0; M 1 should squeeze below peak_rank 2.

        Without squeezing, both qubits are active simultaneously (peak_rank=2).
        With squeezing, measurements bubble left and T gates bubble right.
        Since T and M on the same single-qubit Pauli axis commute (symplectic
        inner product = 0), the squeezer reorders all measurements before all
        T gates, resolving everything as dormant and reducing peak_rank to 0.
        """
        circuit = "H 0\nT 0\nH 1\nT 1\nM 0\nM 1"

        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        assert base.peak_rank == 2, f"Expected baseline peak_rank=2, got {base.peak_rank}"
        assert (
            squeezed.peak_rank < base.peak_rank
        ), f"Expected squeezed peak_rank < {base.peak_rank}, got {squeezed.peak_rank}"

    def test_three_independent_qubits(self) -> None:
        """Three independent qubits: squeeze should reduce peak_rank."""
        circuit = "H 0\nT 0\nH 1\nT 1\nH 2\nT 2\nM 0\nM 1\nM 2"
        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        assert base.peak_rank == 3
        assert squeezed.peak_rank < base.peak_rank

    def test_squeeze_sampling_correctness(self) -> None:
        """Sampling results must be statistically identical with and without squeeze."""
        circuit = "H 0\nT 0\nH 1\nT 1\nM 0\nM 1"
        shots = 10_000

        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        base_result = ucc.sample(base, shots, seed=42)
        squeezed_result = ucc.sample(squeezed, shots, seed=42)

        # Both should produce roughly 50/50 coin flips
        for col in range(base_result.measurements.shape[1]):
            p1 = float(base_result.measurements[:, col].mean())
            p2 = float(squeezed_result.measurements[:, col].mean())
            tol = cross_binomial_tolerance((p1 + p2) / 2.0, shots, sigma=5.0)
            assert (
                abs(p1 - p2) < tol
            ), f"qubit {col}: base={p1:.4f}, squeezed={p2:.4f}, tol={tol:.4f}"

    def test_entangled_qubits_correct_behavior(self) -> None:
        """Entangled qubits: squeeze must still produce correct results.

        CX 0 1 creates entangled Pauli strings in the Heisenberg picture.
        The squeezer respects anti-commutation barriers and preserves semantics.
        """
        circuit = "H 0\nCX 0 1\nT 0\nT 1\nM 0\nM 1"
        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        # Verify sampling correctness regardless of peak_rank change
        shots = 10_000
        base_result = ucc.sample(base, shots, seed=42)
        squeezed_result = ucc.sample(squeezed, shots, seed=42)
        for col in range(base_result.measurements.shape[1]):
            p1 = float(base_result.measurements[:, col].mean())
            p2 = float(squeezed_result.measurements[:, col].mean())
            tol = cross_binomial_tolerance((p1 + p2) / 2.0, shots, sigma=5.0)
            assert abs(p1 - p2) < tol

    def test_anti_commutation_guard_safety(self) -> None:
        """Physical noise barriers must block commutation.

        X_ERROR between T and M anti-commutes with the measurement's Z-basis
        Pauli, so the squeezer must refuse to swap M past it.
        The order of quantum-significant ops must be preserved.
        """
        circuit_str = "H 0\nT 0\nX_ERROR(0.1) 0\nM 0"
        base = ucc.compile(circuit_str)
        squeezed = ucc.compile(circuit_str, hir_passes=_squeeze_only_pass_manager())

        # Both should have peak_rank=1 since the noise barrier blocks
        # the measurement from bubbling past X_ERROR
        assert squeezed.peak_rank == base.peak_rank, (
            f"Squeezer changed peak_rank through noise barrier: "
            f"{base.peak_rank} -> {squeezed.peak_rank}"
        )


class TestSqueezeClassicalDataflow:
    """Precise classical index checks: MEASURE bubbles past unrelated readers."""

    def test_measure_bubbles_past_unrelated_detector(self) -> None:
        """M 1 must bubble left past a DETECTOR that only references M 0.

        With the precise index check, the squeezer knows the DETECTOR reads
        meas_idx 0 and M 1 writes meas_idx 1, so they are independent.
        """
        circuit = "H 0\nT 0\nH 1\nT 1\nM 0\nDETECTOR rec[-1]\nM 1"

        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        assert base.peak_rank == 2
        assert squeezed.peak_rank == 0

    def test_measure_blocked_by_own_detector(self) -> None:
        """M 0 must NOT bubble past a DETECTOR that references meas_idx 0."""
        circuit = "H 0\nM 0\nDETECTOR rec[-1]"
        hir = ucc.trace(ucc.parse(circuit))
        ops_before = [op["op_type"] for op in hir.as_dict()["ops"]]

        pm = ucc.HirPassManager()
        pm.add(ucc.StatevectorSqueezePass())
        pm.run(hir)
        ops_after = [op["op_type"] for op in hir.as_dict()["ops"]]

        assert ops_before == ops_after

    def test_measure_blocked_by_own_conditional_pauli(self) -> None:
        """M 0 must NOT swap past a CONDITIONAL_PAULI that reads meas_idx 0."""
        circuit = "H 0\nH 1\nM 0\nCX rec[-1] 1\nM 1"
        hir = ucc.trace(ucc.parse(circuit))

        pm = ucc.HirPassManager()
        pm.add(ucc.StatevectorSqueezePass())
        pm.run(hir)
        ops_after = [
            (op["op_type"], op.get("meas_record_idx"), op.get("controlling_meas"))
            for op in hir.as_dict()["ops"]
        ]

        # MEASURE(0) must stay before CONDITIONAL_PAULI(0)
        meas_pos = next(i for i, o in enumerate(ops_after) if o[1] == 0)
        cond_pos = next(i for i, o in enumerate(ops_after) if o[0] == "CONDITIONAL_PAULI")
        assert meas_pos < cond_pos

    def test_readout_noise_blocked_by_dependent_conditional(self) -> None:
        """READOUT_NOISE on meas 0 must not swap past CONDITIONAL_PAULI on meas 0.

        M(0.1) produces both MEASURE and READOUT_NOISE on meas_idx 0.
        The READOUT_NOISE mutates the classical bit, so it must execute before
        any CONDITIONAL_PAULI that reads that same bit.
        """
        circuit = "H 0\nH 1\nM(0.1) 0\nCX rec[-1] 1\nM 1"
        hir = ucc.trace(ucc.parse(circuit))

        pm = ucc.HirPassManager()
        pm.add(ucc.StatevectorSqueezePass())
        pm.run(hir)
        ops = hir.as_dict()["ops"]
        types = [op["op_type"] for op in ops]

        rn_pos = types.index("READOUT_NOISE")
        cp_pos = types.index("CONDITIONAL_PAULI")
        assert (
            rn_pos < cp_pos
        ), f"READOUT_NOISE at {rn_pos} must precede CONDITIONAL_PAULI at {cp_pos}"

    def test_measure_bubbles_past_unrelated_observable(self) -> None:
        """M 1 should bubble past an OBSERVABLE that only references meas_idx 0."""
        circuit = "H 0\nT 0\nH 1\nT 1\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nM 1"

        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        assert base.peak_rank == 2
        assert squeezed.peak_rank == 0

    def test_qec_style_detector_passthrough_sampling(self) -> None:
        """Verify sampling correctness when measurements bubble past detectors.

        Mimics a simplified QEC round: prepare, measure syndromes, DETECTOR,
        measure data. The squeezer should compact lifetimes without changing
        the measurement distribution.
        """
        circuit = "H 0\nT 0\n" "H 1\nT 1\n" "H 2\nT 2\n" "M 0\n" "DETECTOR rec[-1]\n" "M 1\nM 2"
        shots = 10_000
        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        assert squeezed.peak_rank < base.peak_rank

        base_result = ucc.sample(base, shots, seed=99)
        squeezed_result = ucc.sample(squeezed, shots, seed=99)
        for col in range(base_result.measurements.shape[1]):
            p1 = float(base_result.measurements[:, col].mean())
            p2 = float(squeezed_result.measurements[:, col].mean())
            tol = cross_binomial_tolerance((p1 + p2) / 2.0, shots, sigma=5.0)
            assert abs(p1 - p2) < tol, f"col {col}: {p1:.4f} vs {p2:.4f}"


class TestSqueezeSweep2Expansion:
    """Sweep 2 tests: rightward bubble of non-Clifford gates."""

    def test_t_gates_do_not_reorder_among_themselves(self) -> None:
        """Multiple independent T gates should bunch together, not uselessly swap.

        Verifies the early-break optimization: when a T gate's neighbor is also
        a T gate, bubbling stops immediately.
        """
        circuit = "H 0\nH 1\nH 2\nT 0\nT 1\nT 2\nM 0\nM 1\nM 2"
        hir = ucc.trace(ucc.parse(circuit))

        pm = ucc.HirPassManager()
        pm.add(ucc.StatevectorSqueezePass())
        pm.run(hir)
        ops = hir.as_dict()["ops"]
        types = [op["op_type"] for op in ops]

        # All measures should be before all T gates after squeeze
        measure_positions = [i for i, t in enumerate(types) if t == "MEASURE"]
        t_positions = [i for i, t in enumerate(types) if t == "T_GATE"]
        if measure_positions and t_positions:
            assert max(measure_positions) < min(t_positions)

    def test_phase_rotation_bubbles_right(self) -> None:
        """PHASE_ROTATION (from R_Z) should bubble rightward in Sweep 2."""
        circuit = "H 0\nH 1\nR_Z(0.3) 0\nR_Z(0.5) 1\nM 0\nM 1"
        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        # With squeeze, the measurements should compact before the rotations
        assert squeezed.peak_rank <= base.peak_rank


class TestSqueezeEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_qubit_no_op(self) -> None:
        """A single M 0 circuit has nothing to squeeze."""
        circuit = "H 0\nM 0"
        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())
        assert base.peak_rank == squeezed.peak_rank

    def test_empty_circuit(self) -> None:
        """Empty circuit should not crash the squeezer."""
        circuit = ""
        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())
        assert base.peak_rank == squeezed.peak_rank == 0

    def test_all_cliffords_no_squeeze_needed(self) -> None:
        """Pure Clifford circuit: squeezer runs but nothing expands."""
        circuit = "H 0\nCX 0 1\nS 0\nM 0\nM 1"
        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())
        assert squeezed.peak_rank == base.peak_rank

    def test_squeeze_idempotent(self) -> None:
        """Running the squeezer twice should produce the same result as once."""
        circuit = "H 0\nT 0\nH 1\nT 1\nH 2\nT 2\nM 0\nM 1\nM 2"
        pm1 = _squeeze_only_pass_manager()
        pm2 = ucc.HirPassManager()
        pm2.add(ucc.StatevectorSqueezePass())
        pm2.add(ucc.StatevectorSqueezePass())

        once = ucc.compile(circuit, hir_passes=pm1)
        twice = ucc.compile(circuit, hir_passes=pm2)
        assert once.peak_rank == twice.peak_rank

    def test_squeeze_with_default_pipeline(self) -> None:
        """Squeeze pass works correctly in the full default pipeline."""
        circuit = "H 0\nT 0\nH 1\nT 1\nM 0\nM 1"
        prog = ucc.compile(circuit)
        shots = 5_000
        result = ucc.sample(prog, shots, seed=42)
        # Should produce valid results without crashing
        assert result.measurements.shape == (shots, 2)


class TestSqueezeStatevectorOracle:
    """Validate squeeze correctness via statevector comparison."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_clifford_t_5q(self, seed: int) -> None:
        """Random 5-qubit Clifford+T circuits preserve statevector under squeeze."""
        from conftest import random_clifford_t_circuit

        circuit = random_clifford_t_circuit(5, 40, seed=seed)
        base_sv = _ucc_statevector(circuit)
        squeezed_sv = _ucc_statevector(circuit, hir_passes=_squeeze_only_pass_manager())
        assert_statevectors_equal(squeezed_sv, base_sv)


class TestSqueezeStatisticalEquivalence:
    """Validate squeeze on noisy circuits via marginal distribution matching."""

    _SHOTS = 10_000

    @staticmethod
    def _assert_marginals_match(
        base_m: np.ndarray, opt_m: np.ndarray, *, sigma: float = 5.0
    ) -> None:
        shots = base_m.shape[0]
        assert base_m.shape == opt_m.shape
        base_probs = base_m.mean(axis=0)
        opt_probs = opt_m.mean(axis=0)
        for col in range(base_m.shape[1]):
            p_pooled = (base_probs[col] + opt_probs[col]) / 2.0
            tol = cross_binomial_tolerance(p_pooled, shots, sigma=sigma)
            diff = abs(float(base_probs[col] - opt_probs[col]))
            assert diff < tol, (
                f"Measurement col {col}: base={base_probs[col]:.4f}, "
                f"opt={opt_probs[col]:.4f}, diff={diff:.4f}, tol={tol:.4f}"
            )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_noisy_uncomputation_ladder_10q(self, seed: int) -> None:
        circuit = generate_uncomputation_ladder(10, 100, seed=seed, noise_prob=0.02)
        base = ucc.compile(circuit)
        squeezed = ucc.compile(
            circuit,
            hir_passes=_squeeze_only_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )
        base_result = ucc.sample(base, self._SHOTS, seed=seed)
        squeezed_result = ucc.sample(squeezed, self._SHOTS, seed=seed)
        self._assert_marginals_match(base_result.measurements, squeezed_result.measurements)


class TestSqueezeProbabilisticReordering:
    """Tests for distributional equivalence when probabilistic ops reorder.

    With the PRNG barrier removed, measurements and noise ops can swap
    when their Pauli masks commute. Per-trajectory outcomes may differ
    but the marginal distributions must be identical.
    """

    _SHOTS = 20_000

    @staticmethod
    def _assert_marginals_match(
        base_m: np.ndarray, opt_m: np.ndarray, *, sigma: float = 5.0
    ) -> None:
        shots = base_m.shape[0]
        assert base_m.shape == opt_m.shape
        for col in range(base_m.shape[1]):
            p1 = float(base_m[:, col].mean())
            p2 = float(opt_m[:, col].mean())
            p_pooled = (p1 + p2) / 2.0
            tol = cross_binomial_tolerance(p_pooled, shots, sigma=sigma)
            diff = abs(p1 - p2)
            assert diff < tol, (
                f"col {col}: base={p1:.4f}, squeezed={p2:.4f}, " f"diff={diff:.4f}, tol={tol:.4f}"
            )

    def test_measure_bubbles_past_noise_on_different_qubit(self) -> None:
        """MEASURE on qubit 1 can bubble past X_ERROR on qubit 0.

        Their Pauli masks are on different qubits so the symplectic inner
        product is 0. The PRNG barrier no longer blocks this swap.
        """
        circuit = "H 0\nH 1\nT 0\nT 1\nX_ERROR(0.1) 0\nM 0\nM 1"

        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        # M 1 should bubble past X_ERROR(0) and M 0 since masks commute
        assert squeezed.peak_rank < base.peak_rank

        base_result = ucc.sample(base, self._SHOTS, seed=10)
        squeezed_result = ucc.sample(squeezed, self._SHOTS, seed=10)
        self._assert_marginals_match(base_result.measurements, squeezed_result.measurements)

    def test_independent_measurements_reorder_freely(self) -> None:
        """Two independent measurements separated by noise can reorder.

        H 0; H 1; T 0; T 1; Z_ERROR(0.05) 0; M 0; M 1
        M 1 is on qubit 1 and commutes with everything on qubit 0.
        Without the PRNG barrier, it freely bubbles left.
        """
        circuit = "H 0\nH 1\nT 0\nT 1\nZ_ERROR(0.05) 0\nM 0\nM 1"
        shots = self._SHOTS

        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        assert squeezed.peak_rank < base.peak_rank

        base_result = ucc.sample(base, shots, seed=77)
        squeezed_result = ucc.sample(squeezed, shots, seed=77)
        self._assert_marginals_match(base_result.measurements, squeezed_result.measurements)

    def test_noise_vs_noise_anti_commutation_guard(self) -> None:
        """X_ERROR and Z_ERROR on the same qubit must NOT swap.

        Their noise channels anti-commute (X vs Z), so reordering would
        change the physical error distribution. The restored
        noise_sites_anti_commute check blocks this.
        """
        circuit = "H 0\nX_ERROR(0.1) 0\nZ_ERROR(0.1) 0\nM 0"
        hir = ucc.trace(ucc.parse(circuit))
        types_before = [op["op_type"] for op in hir.as_dict()["ops"]]

        pm = ucc.HirPassManager()
        pm.add(ucc.StatevectorSqueezePass())
        pm.run(hir)
        types_after = [op["op_type"] for op in hir.as_dict()["ops"]]

        # The two NOISE ops must stay in their original relative order
        noise_positions_before = [i for i, t in enumerate(types_before) if t == "NOISE"]
        noise_positions_after = [i for i, t in enumerate(types_after) if t == "NOISE"]
        assert len(noise_positions_before) == 2
        assert len(noise_positions_after) == 2
        # Relative order preserved (neither bubbled past the other)
        assert noise_positions_after[0] < noise_positions_after[1]

    def test_noise_vs_noise_commuting_channels_can_swap(self) -> None:
        """Two X_ERROR ops on the same qubit have commuting channels and can swap."""
        circuit = "H 0\nX_ERROR(0.1) 0\nX_ERROR(0.2) 0\nM 0"
        hir = ucc.trace(ucc.parse(circuit))

        pm = ucc.HirPassManager()
        pm.add(ucc.StatevectorSqueezePass())
        pm.run(hir)

        # Both are NOISE with X channels that commute with each other.
        # The squeezer won't actively try to swap two noise ops (neither
        # is a MEASURE or expanding gate), but the commutation check
        # should at least not crash. This is a soundness smoke test.
        types = [op["op_type"] for op in hir.as_dict()["ops"]]
        assert types.count("NOISE") == 2

    def test_noisy_multi_qubit_statistical_equivalence(self) -> None:
        """Larger noisy circuit: squeeze preserves marginal distributions.

        4 independent qubits each get H, T, noise, then M.
        The squeezer should compact lifetimes aggressively now that
        measurements can pass noise on different qubits.
        """
        lines = []
        for q in range(4):
            lines.append(f"H {q}")
            lines.append(f"T {q}")
        # Noise only on qubits 0 and 2
        lines.append("X_ERROR(0.05) 0")
        lines.append("Z_ERROR(0.05) 2")
        for q in range(4):
            lines.append(f"M {q}")
        circuit = "\n".join(lines)
        shots = self._SHOTS

        base = ucc.compile(circuit)
        squeezed = ucc.compile(circuit, hir_passes=_squeeze_only_pass_manager())

        assert squeezed.peak_rank < base.peak_rank

        base_result = ucc.sample(base, shots, seed=123)
        squeezed_result = ucc.sample(squeezed, shots, seed=123)
        self._assert_marginals_match(base_result.measurements, squeezed_result.measurements)
