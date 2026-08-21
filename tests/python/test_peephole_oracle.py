"""Peephole optimizer correctness tests.

The tests compare state rays with the pass enabled and disabled, exercise
specific algebraic identities, and check complete T/T-dag cancellation in
U U-dag mirror circuits.
"""

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equiv,
    cross_binomial_tolerance,
    random_clifford_t_circuit,
    random_dense_clifford_t_circuit,
)

import clifft


def _peephole_pass_manager() -> clifft.HirPassManager:
    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())
    return pm


def _compile_optimized(circuit_str: str) -> clifft.Program:
    """Compile with only PeepholeFusionPass enabled."""
    circuit = clifft.parse(circuit_str)
    hir = clifft.trace(circuit)
    pm = _peephole_pass_manager()
    pm.run(hir)
    return clifft.lower(hir)


def _clifft_statevector(circuit_str: str, *, optimize: bool = False) -> np.ndarray:
    """Compile and execute circuit in Clifft, return dense statevector.

    The optimize=False baseline disables HIR optimization; the default
    clifft.compile() call would run the very pass under test.
    """
    if optimize:
        program = clifft.compile(circuit_str, hir_passes=_peephole_pass_manager())
    else:
        program = clifft.compile(circuit_str, hir_passes=None)
    return np.asarray(clifft.get_statevector(program))


# Specific algebraic identities.


class TestPeepholeAlgebraicIdentities:
    """Verify optimizer handles known algebraic cases correctly."""

    def test_t_tdag_cancel_preserves_statevector(self) -> None:
        """Adjacent T T_DAG on same qubit cancels to identity."""
        circuit = "H 0\nT 0\nT_DAG 0"
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline)

    def test_two_t_fuse_to_s(self) -> None:
        """T T = S preserves amplitudes."""
        circuit = "H 0\nT 0\nT 0"
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline)

    def test_four_t_equals_z(self) -> None:
        """T^4 = Z identity preserved through optimizer."""
        circuit = "H 0\nT 0\nT 0\nT 0\nT 0"
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline)

    def test_separated_t_gates_fuse(self) -> None:
        """T gates separated by commuting Cliffords still fuse."""
        circuit = "H 0\nH 1\nT 0\nS 1\nH 1\nT 0"
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline)

    def test_entangled_t_fusion(self) -> None:
        """T gates on entangled qubits preserve interference."""
        circuit = "H 0\nCX 0 1\nT 0\nT 1\nT_DAG 0"
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline)


class TestTerminalMeasurementPhaseElimination:
    """Phases diagonal in a later measurement basis are unobservable."""

    def test_measure_reset_corrections_preserve_distribution(self) -> None:
        """Disjoint MR corrections can be crossed without changing outputs."""
        exact_circuit = "H 0 1\nR_Z(0.3) 0 1\nMR 0 1"
        exact_baseline = clifft.compile(exact_circuit, hir_passes=None)
        exact_optimized = clifft.compile(
            exact_circuit,
            hir_passes=_peephole_pass_manager(),
        )
        assert exact_baseline.peak_active_width == 2
        assert exact_optimized.peak_active_width == 0

        records = ["00", "01", "10", "11"]
        np.testing.assert_allclose(
            clifft.record_probabilities(exact_optimized, records),
            clifft.record_probabilities(exact_baseline, records),
            atol=1e-12,
        )

        noisy_circuit = (
            "R 0 1\n"
            "H 0 1\n"
            "R_Z(0.3) 0 1\n"
            "X_ERROR(0.01) 0 1\n"
            "MR !0 !1\n"
            "DETECTOR rec[-2] rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]"
        )
        baseline = clifft.compile(noisy_circuit, hir_passes=None)
        optimized = clifft.compile(
            noisy_circuit,
            hir_passes=_peephole_pass_manager(),
        )
        assert baseline.peak_active_width == 2
        assert optimized.peak_active_width == 0

        shots = 30_000
        baseline_result = clifft.sample(baseline, shots, seed=242)
        optimized_result = clifft.sample(optimized, shots, seed=243)

        weights = 1 << np.arange(2, dtype=np.uint64)
        baseline_bins = np.asarray(baseline_result.measurements, dtype=np.uint64) @ weights
        optimized_bins = np.asarray(optimized_result.measurements, dtype=np.uint64) @ weights
        baseline_probs = np.bincount(baseline_bins.astype(np.int64), minlength=4) / shots
        optimized_probs = np.bincount(optimized_bins.astype(np.int64), minlength=4) / shots

        for baseline_p, optimized_p in zip(baseline_probs, optimized_probs, strict=True):
            pooled = float((baseline_p + optimized_p) / 2.0)
            tolerance = cross_binomial_tolerance(pooled, shots, sigma=6.0)
            assert abs(float(baseline_p - optimized_p)) <= tolerance

        for result in (baseline_result, optimized_result):
            np.testing.assert_array_equal(
                result.detectors[:, 0],
                np.logical_xor(result.measurements[:, 0], result.measurements[:, 1]),
            )
            np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, 1])

    @pytest.mark.parametrize("pauli_branch", ["", "X 0", "Y 0", "Z 0"])
    def test_exact_deterministic_pauli_branches(self, pauli_branch: str) -> None:
        """Every branch agrees exactly, including downstream feedback."""
        branch = f"{pauli_branch}\n" if pauli_branch else ""
        circuit = "H 0\n" "R_Z(0.37) 0\n" f"{branch}" "M 0\n" "CX rec[-1] 1\n" "M 1"
        records = ["00", "01", "10", "11"]
        baseline = clifft.compile(circuit, hir_passes=None)
        optimized = clifft.compile(
            circuit,
            hir_passes=_peephole_pass_manager(),
        )

        np.testing.assert_allclose(
            clifft.record_probabilities(optimized, records),
            clifft.record_probabilities(baseline, records),
            atol=1e-12,
        )
        assert baseline.peak_active_width == 1
        assert optimized.peak_active_width == 0

    def test_noisy_broadcast_distribution_and_classical_outputs(self) -> None:
        """The real noisy rewrite preserves the complete record distribution."""
        circuit = (
            "H 0 1\n"
            "R_Z(0.17) 0 1\n"
            "DEPOLARIZE1(0.2) 0 1\n"
            "M 0 1\n"
            "DETECTOR rec[-2]\n"
            "DETECTOR rec[-1]\n"
            "CX rec[-1] 2\n"
            "M 2\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]"
        )
        baseline = clifft.compile(circuit, hir_passes=None)
        optimized = clifft.compile(
            circuit,
            hir_passes=_peephole_pass_manager(),
        )
        assert baseline.peak_active_width == 2
        assert optimized.peak_active_width == 0

        shots = 30_000
        baseline_result = clifft.sample(baseline, shots, seed=238)
        optimized_result = clifft.sample(optimized, shots, seed=239)

        weights = 1 << np.arange(3, dtype=np.uint64)
        baseline_bins = np.asarray(baseline_result.measurements, dtype=np.uint64) @ weights
        optimized_bins = np.asarray(optimized_result.measurements, dtype=np.uint64) @ weights
        baseline_probs = np.bincount(baseline_bins.astype(np.int64), minlength=8) / shots
        optimized_probs = np.bincount(optimized_bins.astype(np.int64), minlength=8) / shots

        for baseline_p, optimized_p in zip(baseline_probs, optimized_probs, strict=True):
            pooled = float((baseline_p + optimized_p) / 2.0)
            tolerance = cross_binomial_tolerance(pooled, shots, sigma=6.0)
            assert abs(float(baseline_p - optimized_p)) <= tolerance

        for result in (baseline_result, optimized_result):
            np.testing.assert_array_equal(result.detectors[:, 0], result.measurements[:, 0])
            np.testing.assert_array_equal(result.detectors[:, 1], result.measurements[:, 1])
            np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, 2])


# Projective state preservation across S absorption.


class TestPeepholeProjectiveState:
    """S absorption must preserve relative amplitudes and phases.

    When the peephole pass fuses two T gates (or S-angle phase rotations)
    and absorbs the resulting S/S_dag into the Clifford frame, the tableau
    fixes the frame only up to global phase. These cases validate the physical
    state ray across signed, entangled, and composed coordinate frames.
    """

    # Each circuit triggers at least one S absorption: T+T fusion,
    # T_DAG+T_DAG fusion, rotation fusion to S/S_dag, and absorptions on
    # signed, Y-type, and multi-qubit axes.
    S_ABSORPTION_CIRCUITS = [
        "H 0\nT 0\nT 0\nH 0",
        "H 0\nT_DAG 0\nT_DAG 0\nH 0",
        "H 0\nR_Z(0.25) 0\nR_Z(0.25) 0\nH 0",
        "H 0\nR_Z(0.125) 0\nR_Z(0.375) 0\nH 0",
        "H 0\nR_Z(0.75) 0\nR_Z(0.75) 0\nH 0",
        "S_DAG 0\nH 0\nT 0\nT 0",
        "S_DAG 0\nH 0\nCX 2 3\nT 0\nCX 3 1\nT 0",
        "H 0\nCX 0 1\nT 1\nT 1\nCX 0 1\nH 0",
        "H 0\nCX 0 1\nT 1\nT 1",
        "H 0\nCX 0 1\nS 1\nT 1\nT 1\nH 1\nT 1\nT 1",
        "Y 0\nH 0\nT 0\nT 0\nT 0\nT 0",
        "H 1\nCX 1 0\nR_Z(0.75) 0\nR_Z(0.75) 0\nH 0",
        # Absorptions that leave rotations needing virtual-frame routing at
        # lowering; these exercise composed physical and planner frames.
        "H 0\nT 0\nT 0\nT 0\nH 0\nT 0",
        "CX 0 1\nY 1\nH 0\nR_Z(0.25) 0\nR_Z(0.25) 0\nX 0",
        "CX 0 1\nY 1\nH 0\nT_DAG 0\nT_DAG 0\nX 0",
        "S_DAG 0\nH 0\nCX 2 3\nT 0\nCX 3 1\nT 0\nH 1\nT 1",
    ]

    def test_h_t_t_h_relative_amplitudes(self) -> None:
        """H T T H preserves its expected state ray after peephole fusion."""
        prog = _compile_optimized("H 0\nT 0\nT 0\nH 0")
        sv = np.asarray(clifft.get_statevector(prog))
        expected = np.asarray([0.5 + 0.5j, 0.5 - 0.5j])
        assert_statevectors_equiv(sv, expected, atol=1e-10)

    @pytest.mark.parametrize("circuit", S_ABSORPTION_CIRCUITS)
    def test_statevector_match(self, circuit: str) -> None:
        """Optimization preserves the projective state."""
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline, atol=1e-10)

    @pytest.mark.parametrize("seed", range(100))
    def test_random_circuits(self, seed: int) -> None:
        """Random Clifford+T circuits agree projectively."""
        circuit = random_clifford_t_circuit(5, depth=30, seed=seed)
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline, atol=1e-8)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_deep_8q(self, seed: int) -> None:
        """Deeper 8-qubit circuits accumulate long virtual-frame gate logs,
        stressing chained coordinate composition across many links."""
        circuit = random_clifford_t_circuit(8, depth=60, seed=seed)
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline, atol=1e-8)

    @pytest.mark.parametrize("seed", range(5))
    def test_dense_random_circuits(self, seed: int) -> None:
        circuit = random_dense_clifford_t_circuit(5, depth=40, seed=seed)
        sv_baseline = _clifft_statevector(circuit)
        sv_optimized = _clifft_statevector(circuit, optimize=True)
        assert_statevectors_equiv(sv_optimized, sv_baseline, atol=1e-8)


# Mirror-circuit T-gate cancellation.

_DAGGER_MAP: dict[str, str] = {
    "H": "H",
    "S": "S_DAG",
    "S_DAG": "S",
    "T": "T_DAG",
    "T_DAG": "T",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "CX": "CX",
    "CY": "CY",
    "CZ": "CZ",
}


def _bounded_t_mirror_circuit(
    num_qubits: int, clifford_gate_count: int, t_count: int, seed: int
) -> str:
    """Generate a U U-dag mirror circuit with bounded T-gate count.

    Produces a Clifford circuit with exactly `t_count` T gates inserted
    at random positions, followed by its exact inverse. The combined
    circuit U U-dag = I.

    Args:
        num_qubits: Number of qubits.
        clifford_gate_count: Total number of random Clifford gates.
        t_count: Exact number of T gates to insert.
        seed: Random seed.

    Returns:
        Circuit string (without measurements).
    """
    rng = np.random.default_rng(seed)
    gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z"]
    gates_2q = ["CX", "CY", "CZ"]

    fwd: list[str] = []
    for _ in range(clifford_gate_count):
        if num_qubits > 1 and rng.random() < 0.4:
            g = str(rng.choice(gates_2q))
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            fwd.append(f"{g} {q1} {q2}")
        else:
            g = str(rng.choice(gates_1q))
            q = int(rng.integers(0, num_qubits))
            fwd.append(f"{g} {q}")

    positions = sorted(rng.choice(len(fwd) + 1, size=t_count, replace=False))
    for offset, pos in enumerate(positions):
        q = int(rng.integers(0, num_qubits))
        fwd.insert(int(pos) + offset, f"T {q}")

    inv: list[str] = []
    for line in reversed(fwd):
        parts = line.split()
        gate = _DAGGER_MAP[parts[0]]
        targets = " ".join(parts[1:])
        inv.append(f"{gate} {targets}")

    return "\n".join(fwd + inv)


class TestMirrorTGateAnnihilation:
    """Verify the peephole optimizer reaches zero peak active width on mirror circuits.

    Mirror circuits have structure U U-dag = I. Without the optimizer,
    each T gate expands the active Schrodinger array (peak active width up to
    t_count). With the optimizer, all T/T-dag pairs should cancel
    completely, leaving zero peak active width (pure Clifford).
    """

    NUM_QUBITS = 40
    CLIFFORD_DEPTH = 1000

    @pytest.mark.parametrize("t_count", [4, 8, 12])
    @pytest.mark.parametrize("seed", range(5))
    def test_mirror_has_zero_peak_active_width(self, t_count: int, seed: int) -> None:
        """Optimizer cancels all T gates in mirror circuits."""
        circuit = _bounded_t_mirror_circuit(self.NUM_QUBITS, self.CLIFFORD_DEPTH, t_count, seed)
        meas = "M " + " ".join(str(i) for i in range(self.NUM_QUBITS))
        circuit_with_meas = circuit + "\n" + meas

        prog_baseline = clifft.compile(circuit_with_meas, hir_passes=None)
        prog_optimized = _compile_optimized(circuit_with_meas)

        assert (
            prog_baseline.peak_active_width <= t_count
        ), f"Baseline peak_active_width={prog_baseline.peak_active_width} > t_count={t_count}"
        assert prog_optimized.peak_active_width == 0, (
            f"Optimized peak_active_width={prog_optimized.peak_active_width}, expected 0 "
            f"(t_count={t_count}, seed={seed})"
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_mirror_sampling_all_zeros(self, seed: int) -> None:
        """Optimized mirror circuit still produces all-zeros measurements."""
        circuit = _bounded_t_mirror_circuit(self.NUM_QUBITS, self.CLIFFORD_DEPTH, 12, seed)
        meas = "M " + " ".join(str(i) for i in range(self.NUM_QUBITS))
        circuit_with_meas = circuit + "\n" + meas

        prog = _compile_optimized(circuit_with_meas)
        assert prog.peak_active_width == 0

        result = clifft.sample(prog, 1000, seed=seed)
        nonzero = int(result.measurements.sum(axis=1).astype(bool).sum())
        assert nonzero == 0, f"{nonzero}/1000 shots non-zero (seed={seed})"

    def test_mirror_statevector_is_identity(self) -> None:
        """Small mirror circuit statevector equals |00...0>."""
        circuit = _bounded_t_mirror_circuit(4, 50, 6, seed=42)
        sv = _clifft_statevector(circuit, optimize=True)

        # |00...0> = [1, 0, 0, ..., 0] up to global phase
        fidelity = float(np.abs(sv[0]) ** 2)
        assert fidelity > 0.9999, f"Fidelity with |0> = {fidelity:.6f}"


# Peephole pass metadata.


class TestPeepholePassMetadata:
    def test_hir_t_gate_count(self) -> None:
        """HirModule reports correct T-gate count before and after optimization."""
        circuit = clifft.parse("H 0\nT 0\nT 0\nM 0")
        hir = clifft.trace(circuit)
        assert hir.num_t_gates == 2

        pm = _peephole_pass_manager()
        pm.run(hir)
        assert hir.num_t_gates == 0

    def test_peephole_pass_stats(self) -> None:
        """PeepholeFusionPass reports cancellation and fusion counts."""
        circuit = clifft.parse("H 0\nT 0\nT_DAG 0\nH 1\nT 1\nT 1\nM 0 1")
        hir = clifft.trace(circuit)

        peephole = clifft.PeepholeFusionPass()
        pm = clifft.HirPassManager()
        pm.add(peephole)
        pm.run(hir)

        assert peephole.cancellations == 1
        assert peephole.fusions == 1


# S-absorption with PeepholeFusionPass enabled and disabled.
#
# These tests compile each circuit twice -- once with no optimizations
# (forcing the executor to apply physical T/rotation actions) and once with
# peephole S-absorption active -- then assert the dense state rays match.
# This checks symplectic conjugation and tableau basis transformation against
# physical gate application.


def _assert_absorption_preserves_state(stim_text: str, rtol: float = 1e-6) -> clifft.Program:
    """Compile with and without optimization; assert statevector equivalence."""
    # Baseline: no HIR passes.
    prog_base = clifft.compile(stim_text, hir_passes=None)
    sv_base = np.array(clifft.get_statevector(prog_base))

    # Optimized: only PeepholeFusionPass.
    prog_opt = clifft.compile(stim_text, hir_passes=_peephole_pass_manager())
    sv_opt = np.array(clifft.get_statevector(prog_opt))

    assert_statevectors_equiv(
        sv_opt,
        sv_base,
        rtol=rtol,
        msg=f"Statevector mismatch for:\n{stim_text}",
    )
    return prog_opt


class TestNegativeSignTFusion:
    """Projective-state checks for T gates with negative Pauli signs.

    When the front-end encounters T after X (which conjugates Z -> -Z), the
    HIR T gate has sign=true. The identity T(-P) = exp(i*pi/4) * T_dag(+P)
    permits sign normalization up to global phase. The cases below exercise
    both fusion directions and verify their relative action.
    """

    def test_negative_sign_t_fusion(self) -> None:
        """Two T gates on -Z are projectively equivalent to S_dag."""
        _assert_absorption_preserves_state("X 0\nT 0\nT 0")

    def test_negative_sign_t_dag_fusion(self) -> None:
        """Two T_dag gates on -Z are projectively equivalent to S."""
        _assert_absorption_preserves_state("X 0\nT_DAG 0\nT_DAG 0")

    def test_negative_sign_t_cancellation(self) -> None:
        """T(-Z) + T_dag(-Z) = identity: cancellation should not corrupt phase."""
        _assert_absorption_preserves_state("X 0\nT 0\nT_DAG 0")

    def test_mixed_sign_t_cancellation_projective(self) -> None:
        """T gates on opposite axes cancel projectively."""
        _assert_absorption_preserves_state("T 0\nX 0\nT 0")

    def test_mixed_sign_t_fusion_projective(self) -> None:
        """T(+Z) and T_dag(-Z) fuse projectively to S."""
        _assert_absorption_preserves_state("T 0\nX 0\nT_DAG 0")

    def test_s_absorption_creates_negative_t_then_fuses(self) -> None:
        """S absorption conjugates downstream T to negative sign, which then fuses.

        T 0; T 0 -> S on Z(0). H changes frame. Third and fourth T are on X(0),
        which anti-commutes with Z(0). S conjugation produces Y(0) with sign=true.
        The two newly-negative T gates must fuse correctly.
        """
        _assert_absorption_preserves_state("T 0\nT 0\nH 0\nT 0\nT 0")

    def test_triple_t_on_negative_axis(self) -> None:
        """Three T gates on -Z: two fuse to S, one remains. Phase must be correct."""
        _assert_absorption_preserves_state("X 0\nT 0\nT 0\nT 0")

    def test_s_absorption_flips_phase_rotation_sign(self) -> None:
        """S absorbed on Z(0) conjugates downstream R_Z on X(0), flipping sign.

        The sign handling must preserve the relative rotation direction.
        """
        _assert_absorption_preserves_state("T 0\nT 0\nH 0\nR_Z(0.3) 0")

    def test_s_absorption_commuting_phase_rotation_unchanged(self) -> None:
        """S on Z(0) commutes with R_Z on Z(0): sign and alpha unchanged."""
        _assert_absorption_preserves_state("T 0\nT 0\nR_Z(0.3) 0")

    def test_chain_of_negative_sign_fusions(self) -> None:
        """Deep chain exercising repeated negative-sign normalization.

        Six T(-Z) gates produce three fusions with net projective action S_dag.
        """
        _assert_absorption_preserves_state("X 0\nT 0\nT 0\nT 0\nT 0\nT 0\nT 0")

    def test_multi_qubit_negative_sign_fusion(self) -> None:
        """Negative signs on entangled multi-qubit Pauli axes."""
        _assert_absorption_preserves_state("X 0\nX 1\nH 0\nCX 0 1\nT 1\nT 1")


class TestSAbsorptionDifferential:
    """Targeted circuits that stress every aspect of S-absorption."""

    def test_final_tableau_only(self) -> None:
        """S absorbed with no downstream active ops -- tests tableau projection.

        H 0; CX 0 1; T 1; T 1: the fused S on the entangled ZZ axis
        has no downstream ops to conjugate. Correctness depends entirely
        on the final_tableau physical-to-virtual mapping and the
        !is_dagger time-direction inversion.
        """
        _assert_absorption_preserves_state("H 0\nCX 0 1\nT 1\nT 1")

    def test_downstream_anti_commutation(self) -> None:
        """S on Z_0 conjugates a downstream T on X_0 (after H) to Y_0.

        T 0; T 0; H 0; T 0: the first two Ts fuse to S on Z(0).
        The H changes the frame. The third T acts on X(0), which
        anti-commutes with Z(0). S conjugation must transform it to Y(0).
        """
        _assert_absorption_preserves_state("T 0\nT 0\nH 0\nT 0")

    def test_multi_qubit_symplectic_sign(self) -> None:
        """Multi-qubit Pauli products stress the mask_plus/mask_minus popcount.

        R_XX(0.25) + R_XX(0.25) fuses to S on the XX axis. The downstream
        R_YY anti-commutes with XX, exercising the per-qubit cyclic phase
        tracking across multiple qubit pairs simultaneously.
        """
        _assert_absorption_preserves_state(
            "H 0\nH 1\nR_XX(0.25) 0 1\nR_XX(0.25) 0 1\nR_YY(0.25) 0 1"
        )

    def test_phase_rotation_fusion(self) -> None:
        """PHASE_ROTATION pairs fuse to S/S_dag and are absorbed.

        The two axes exercise both S and S_dag demotion while preserving the
        relative state amplitudes.
        """
        _assert_absorption_preserves_state(
            "R_Z(0.25) 0\nR_Z(0.25) 0\nH 1\nR_Z(0.75) 1\nR_Z(0.75) 1"
        )

    @pytest.mark.parametrize("angles", [(0.2, 0.3), (0.7, 0.8)])
    def test_fused_multi_qubit_phase_rotation_demotion(self, angles: tuple[float, float]) -> None:
        """Fused multi-qubit S/S_dag rotations reach peephole absorption."""
        first, second = angles
        _assert_absorption_preserves_state(f"R_XX({first}) 0 1\nR_XX({second}) 0 1")


class TestPauliRotationAbsorption:
    """Pauli-valued rotations are folded into the Clifford frame."""

    def test_fused_pauli_residue_preserves_state_ray(self) -> None:
        circuit = "H 0\nR_Z(0.3) 0\nR_Z(0.7) 0\nR_X(0.2) 0"
        optimized = _assert_absorption_preserves_state(circuit)
        assert optimized.peak_active_width == 0

    def test_fused_pauli_residue_preserves_record_probabilities(self) -> None:
        circuit = "H 0\nR_Z(0.3) 0\nR_Z(0.7) 0\nH 0\nM 0"
        baseline = clifft.compile(circuit, hir_passes=None)
        optimized = clifft.compile(circuit, hir_passes=_peephole_pass_manager())

        np.testing.assert_allclose(
            clifft.record_probabilities(optimized, ["0", "1"]),
            clifft.record_probabilities(baseline, ["0", "1"]),
            atol=1e-12,
        )
        assert baseline.peak_active_width == 1
        assert optimized.peak_active_width == 0
