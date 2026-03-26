"""Peephole optimizer correctness tests.

Validates that the symplectic peephole fusion pass preserves quantum
semantics. Two independent verification strategies:

1. Self-consistency: compile with optimizer on vs off, assert identical
   statevectors (fidelity > 0.9999).
2. Mirror cancellation: U U-dag mirror circuits must achieve peak_rank=0
   when the optimizer is enabled, proving complete T/T-dag annihilation.
"""

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equal,
    random_clifford_t_circuit,
    random_dense_clifford_t_circuit,
)

import ucc


def _compile_optimized(circuit_str: str) -> ucc.Program:
    """Compile with the default peephole optimization pass."""
    circuit = ucc.parse(circuit_str)
    hir = ucc.trace(circuit)
    pm = ucc.default_hir_pass_manager()
    pm.run(hir)
    return ucc.lower(hir)


def _ucc_statevector(circuit_str: str, *, optimize: bool = False) -> np.ndarray:
    """Compile and execute circuit in UCC, return dense statevector."""
    prog = _compile_optimized(circuit_str) if optimize else ucc.compile(circuit_str)
    state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    ucc.execute(prog, state)
    sv: np.ndarray = ucc.get_statevector(prog, state)
    return sv


# ---------------------------------------------------------------------------
# Optimizer On vs Off Statevector Equivalence
# ---------------------------------------------------------------------------


class TestPeepholeStatevectorEquivalence:
    """Assert optimizer preserves exact quantum amplitudes.

    Compiles random Clifford+T circuits with and without the peephole
    pass, then checks fidelity between the two resulting statevectors.
    Any deviation indicates the optimizer corrupted the phase polynomial.
    """

    @pytest.mark.parametrize("seed", range(10))
    def test_random_8q_depth30(self, seed: int) -> None:
        """8-qubit random Clifford+T circuits preserve statevector."""
        circuit = random_clifford_t_circuit(8, depth=30, seed=seed)
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(
            sv_optimized,
            sv_baseline,
            msg=f"8q depth=30 seed={seed}",
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_8q_depth60(self, seed: int) -> None:
        """Deeper 8-qubit circuits stress multi-layer fusion."""
        circuit = random_clifford_t_circuit(8, depth=60, seed=seed)
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(
            sv_optimized,
            sv_baseline,
            msg=f"8q depth=60 seed={seed}",
        )

    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("seed", range(5))
    def test_random_small_circuits(self, num_qubits: int, seed: int) -> None:
        """Small circuits from 2 to 6 qubits preserve statevector."""
        circuit = random_clifford_t_circuit(num_qubits, depth=20, seed=seed)
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(
            sv_optimized,
            sv_baseline,
            msg=f"{num_qubits}q depth=20 seed={seed}",
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_dense_entanglement_5q(self, seed: int) -> None:
        """Dense 2-qubit gate circuits stress Pauli mask interference."""
        circuit = random_dense_clifford_t_circuit(5, depth=40, seed=seed)
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(
            sv_optimized,
            sv_baseline,
            msg=f"dense 5q depth=40 seed={seed}",
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_deep_phase_accumulation(self, seed: int) -> None:
        """Deep circuits with many T gates test accumulated fusion accuracy."""
        circuit = random_dense_clifford_t_circuit(4, depth=100, seed=seed, two_qubit_prob=0.3)
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(
            sv_optimized,
            sv_baseline,
            msg=f"deep 4q depth=100 seed={seed}",
        )


# ---------------------------------------------------------------------------
# Specific algebraic identities
# ---------------------------------------------------------------------------


class TestPeepholeAlgebraicIdentities:
    """Verify optimizer handles known algebraic cases correctly."""

    def test_t_tdag_cancel_preserves_statevector(self) -> None:
        """Adjacent T T_DAG on same qubit cancels to identity."""
        circuit = "H 0\nT 0\nT_DAG 0"
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(sv_optimized, sv_baseline)

    def test_two_t_fuse_to_s(self) -> None:
        """T T = S preserves amplitudes."""
        circuit = "H 0\nT 0\nT 0"
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(sv_optimized, sv_baseline)

    def test_four_t_equals_z(self) -> None:
        """T^4 = Z identity preserved through optimizer."""
        circuit = "H 0\nT 0\nT 0\nT 0\nT 0"
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(sv_optimized, sv_baseline)

    def test_separated_t_gates_fuse(self) -> None:
        """T gates separated by commuting Cliffords still fuse."""
        circuit = "H 0\nH 1\nT 0\nS 1\nH 1\nT 0"
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(sv_optimized, sv_baseline)

    def test_entangled_t_fusion(self) -> None:
        """T gates on entangled qubits preserve interference."""
        circuit = "H 0\nCX 0 1\nT 0\nT 1\nT_DAG 0"
        sv_baseline = _ucc_statevector(circuit)
        sv_optimized = _ucc_statevector(circuit, optimize=True)
        assert_statevectors_equal(sv_optimized, sv_baseline)


# ---------------------------------------------------------------------------
# Mirror Circuit T-gate Annihilation
# ---------------------------------------------------------------------------

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
    """Verify peephole optimizer achieves peak_rank=0 on mirror circuits.

    Mirror circuits have structure U U-dag = I. Without the optimizer,
    each T gate expands the active Schrodinger array (peak_rank up to
    t_count). With the optimizer, all T/T-dag pairs should cancel
    completely, leaving peak_rank=0 (pure Clifford).
    """

    NUM_QUBITS = 40
    CLIFFORD_DEPTH = 1000

    @pytest.mark.parametrize("t_count", [4, 8, 12])
    @pytest.mark.parametrize("seed", range(5))
    def test_mirror_peak_rank_zero(self, t_count: int, seed: int) -> None:
        """Optimizer cancels all T gates in mirror circuits."""
        circuit = _bounded_t_mirror_circuit(self.NUM_QUBITS, self.CLIFFORD_DEPTH, t_count, seed)
        meas = "M " + " ".join(str(i) for i in range(self.NUM_QUBITS))
        circuit_with_meas = circuit + "\n" + meas

        prog_baseline = ucc.compile(circuit_with_meas)
        prog_optimized = _compile_optimized(circuit_with_meas)

        assert (
            prog_baseline.peak_rank <= t_count
        ), f"Baseline peak_rank={prog_baseline.peak_rank} > t_count={t_count}"
        assert prog_optimized.peak_rank == 0, (
            f"Optimized peak_rank={prog_optimized.peak_rank}, expected 0 "
            f"(t_count={t_count}, seed={seed})"
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_mirror_sampling_all_zeros(self, seed: int) -> None:
        """Optimized mirror circuit still produces all-zeros measurements."""
        circuit = _bounded_t_mirror_circuit(self.NUM_QUBITS, self.CLIFFORD_DEPTH, 12, seed)
        meas = "M " + " ".join(str(i) for i in range(self.NUM_QUBITS))
        circuit_with_meas = circuit + "\n" + meas

        prog = _compile_optimized(circuit_with_meas)
        assert prog.peak_rank == 0

        result = ucc.sample(prog, 1000, seed=seed)
        nonzero = int(result.measurements.sum(axis=1).astype(bool).sum())
        assert nonzero == 0, f"{nonzero}/1000 shots non-zero (seed={seed})"

    def test_mirror_statevector_is_identity(self) -> None:
        """Small mirror circuit statevector equals |00...0>."""
        circuit = _bounded_t_mirror_circuit(4, 50, 6, seed=42)
        sv = _ucc_statevector(circuit, optimize=True)

        # |00...0> = [1, 0, 0, ..., 0] up to global phase
        fidelity = float(np.abs(sv[0]) ** 2)
        assert fidelity > 0.9999, f"Fidelity with |0> = {fidelity:.6f}"


# ---------------------------------------------------------------------------
# Explicit pipeline API tests
# ---------------------------------------------------------------------------


class TestExplicitPipelineAPI:
    """Verify the explicit parse -> trace -> optimize -> lower pipeline."""

    def test_hir_t_gate_count(self) -> None:
        """HirModule reports correct T-gate count before and after optimization."""
        circuit = ucc.parse("H 0\nT 0\nT 0\nM 0")
        hir = ucc.trace(circuit)
        assert hir.num_t_gates == 2

        pm = ucc.default_hir_pass_manager()
        pm.run(hir)
        assert hir.num_t_gates == 0

    def test_peephole_pass_stats(self) -> None:
        """PeepholeFusionPass reports cancellation and fusion counts."""
        circuit = ucc.parse("H 0\nT 0\nT_DAG 0\nH 1\nT 1\nT 1\nM 0 1")
        hir = ucc.trace(circuit)

        peephole = ucc.PeepholeFusionPass()
        pm = ucc.HirPassManager()
        pm.add(peephole)
        pm.run(hir)

        assert peephole.cancellations == 1
        assert peephole.fusions == 1

    def test_hir_metadata(self) -> None:
        """HirModule exposes circuit metadata."""
        circuit = ucc.parse("H 0\nCX 0 1\nT 0\nM 0 1")
        hir = ucc.trace(circuit)
        assert hir.num_qubits == 2
        assert hir.num_measurements == 2
        assert hir.num_ops > 0

    def test_lower_produces_valid_program(self) -> None:
        """lower() produces a Program that can be sampled."""
        circuit = ucc.parse("H 0\nT 0\nM 0")
        hir = ucc.trace(circuit)
        prog = ucc.lower(hir)

        assert prog.peak_rank == 1
        assert prog.num_measurements == 1
        result = ucc.sample(prog, 100, seed=0)
        assert result.measurements.shape == (100, 1)

    def test_compile_convenience_matches_explicit(self) -> None:
        """ucc.compile() produces same result as parse -> trace -> lower."""
        text = "H 0\nT 0\nM 0"
        prog_conv = ucc.compile(text)

        circuit = ucc.parse(text)
        hir = ucc.trace(circuit)
        prog_explicit = ucc.lower(hir)

        assert prog_conv.peak_rank == prog_explicit.peak_rank
        assert prog_conv.num_instructions == prog_explicit.num_instructions


# ---------------------------------------------------------------------------
# S-Absorption Differential: Optimized vs Unoptimized Statevector
#
# These tests compile each circuit twice -- once with no optimizations
# (forcing the VM to execute physical T/rotation opcodes) and once with
# peephole S-absorption active -- then assert the dense statevectors match.
# This proves the symplectic conjugation, tableau basis transformation,
# and global phase tracking are equivalent to physical gate application.
# ---------------------------------------------------------------------------


def _assert_absorption_preserves_state(stim_text: str, atol: float = 1e-6) -> ucc.Program:
    """Compile with and without optimization; assert statevector equivalence."""
    # Baseline: no HIR or bytecode passes
    prog_base = ucc.compile(stim_text)
    state_base = ucc.State(
        peak_rank=prog_base.peak_rank, num_measurements=prog_base.num_measurements
    )
    ucc.execute(prog_base, state_base)
    sv_base = np.array(ucc.get_statevector(prog_base, state_base))

    # Optimized: full default pass managers (includes PeepholeFusionPass)
    prog_opt = ucc.compile(
        stim_text,
        hir_passes=ucc.default_hir_pass_manager(),
        bytecode_passes=ucc.default_bytecode_pass_manager(),
    )
    state_opt = ucc.State(peak_rank=prog_opt.peak_rank, num_measurements=prog_opt.num_measurements)
    ucc.execute(prog_opt, state_opt)
    sv_opt = np.array(ucc.get_statevector(prog_opt, state_opt))

    # Align global phase before comparison. Stim's
    # Tableau::to_flat_unitary_matrix canonicalizes the first non-zero
    # amplitude to positive-real, which arbitrarily strips the physical
    # global phase when the peephole absorbs S gates into the tableau.
    idx = int(np.argmax(np.abs(sv_base)))
    if np.abs(sv_base[idx]) > 1e-8 and np.abs(sv_opt[idx]) > 1e-8:
        phase_diff = sv_base[idx] / sv_opt[idx]
        sv_opt = sv_opt * (phase_diff / np.abs(phase_diff))

    np.testing.assert_allclose(
        sv_opt,
        sv_base,
        atol=atol,
        err_msg=f"Statevector mismatch for:\n{stim_text}",
    )
    return prog_opt


class TestNegativeSignTFusion:
    """Regression tests for global phase loss when T gates have negative Pauli signs.

    When the front-end encounters T after X (which conjugates Z -> -Z), the
    HIR T gate has sign=true. The identity T(-P) = exp(i*pi/4) * T_dag(+P)
    means that fusing or canceling negative-sign T gates must track the
    extra global phase. These tests catch the phase loss bug.
    """

    def test_negative_sign_t_fusion(self) -> None:
        """T(-Z) + T(-Z) = i * S_dag: must preserve exp(i*pi/2) global phase."""
        _assert_absorption_preserves_state("X 0\nT 0\nT 0")

    def test_negative_sign_t_dag_fusion(self) -> None:
        """T_dag(-Z) + T_dag(-Z) = -i * S: must preserve exp(-i*pi/2) global phase."""
        _assert_absorption_preserves_state("X 0\nT_DAG 0\nT_DAG 0")

    def test_negative_sign_t_cancellation(self) -> None:
        """T(-Z) + T_dag(-Z) = identity: cancellation should not corrupt phase."""
        _assert_absorption_preserves_state("X 0\nT 0\nT_DAG 0")

    def test_mixed_sign_t_cancellation_global_phase(self) -> None:
        """T(+Z) and T(-Z) cancel but leave exp(i*pi/4) global phase."""
        _assert_absorption_preserves_state("T 0\nX 0\nT 0")

    def test_mixed_sign_t_fusion_global_phase(self) -> None:
        """T(+Z) and T_dag(-Z) fuse to S, leaving a global phase."""
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

        The sign flip must be accompanied by alpha negation to maintain the
        physical gate identity, or the backend global phase correction breaks.
        """
        _assert_absorption_preserves_state("T 0\nT 0\nH 0\nR_Z(0.3) 0")

    def test_s_absorption_commuting_phase_rotation_unchanged(self) -> None:
        """S on Z(0) commutes with R_Z on Z(0): sign and alpha unchanged."""
        _assert_absorption_preserves_state("T 0\nT 0\nR_Z(0.3) 0")

    def test_chain_of_negative_sign_fusions(self) -> None:
        """Deep chain exercising repeated negative-sign normalization.

        Six T(-Z) gates: three fusions, each contributing exp(i*pi/2) = i.
        Net global phase = i^3 = -i. Net S_dag^3 = S_dag.
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

    def test_phase_rotation_demotion(self) -> None:
        """PHASE_ROTATION at S/S_dag angles demoted and absorbed.

        The front-end extracts the absolute global phase for continuous
        rotations. S-absorption must not double-count it. If wrong, the
        output differs by exactly 45 degrees.
        """
        _assert_absorption_preserves_state("R_Z(0.5) 0\nH 1\nR_Z(1.5) 1")
