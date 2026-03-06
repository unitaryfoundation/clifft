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
    pm = ucc.default_pass_manager()
    pm.run(hir)
    return ucc.lower(hir)


def _ucc_statevector(circuit_str: str, *, optimize: bool = False) -> np.ndarray:
    """Compile and execute circuit in UCC, return dense statevector."""
    prog = _compile_optimized(circuit_str) if optimize else ucc.compile(circuit_str)
    state = ucc.State(prog.peak_rank, prog.num_measurements)
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

        results, _, _ = ucc.sample(prog, 1000, seed=seed)
        nonzero = int(results.sum(axis=1).astype(bool).sum())
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

        pm = ucc.default_pass_manager()
        pm.run(hir)
        assert hir.num_t_gates == 0

    def test_peephole_pass_stats(self) -> None:
        """PeepholeFusionPass reports cancellation and fusion counts."""
        circuit = ucc.parse("H 0\nT 0\nT_DAG 0\nH 1\nT 1\nT 1\nM 0 1")
        hir = ucc.trace(circuit)

        peephole = ucc.PeepholeFusionPass()
        pm = ucc.PassManager()
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
        meas, _, _ = ucc.sample(prog, 100, seed=0)
        assert meas.shape == (100, 1)

    def test_compile_convenience_matches_explicit(self) -> None:
        """ucc.compile() produces same result as parse -> trace -> lower."""
        text = "H 0\nT 0\nM 0"
        prog_conv = ucc.compile(text)

        circuit = ucc.parse(text)
        hir = ucc.trace(circuit)
        prog_explicit = ucc.lower(hir)

        assert prog_conv.peak_rank == prog_explicit.peak_rank
        assert prog_conv.num_instructions == prog_explicit.num_instructions
