"""Structural Oracle Tests.

Validates exact destructive interference, memory lifecycle bounds,
and biased amplitude statistics without requiring an external oracle.
These tests exploit analytical properties of the circuits themselves.
"""

import numpy as np
import pytest
from conftest import binomial_tolerance

import ucc


def _compile_optimized(circuit_str: str) -> ucc.Program:
    """Compile with the default peephole optimization pass."""
    circuit = ucc.parse(circuit_str)
    hir = ucc.trace(circuit)
    pm = ucc.default_pass_manager()
    pm.run(hir)
    return ucc.lower(hir)


# ---------------------------------------------------------------------------
# Bounded-T Mirror Fuzzer
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
) -> tuple[str, int]:
    """Generate a U U-dag mirror circuit with bounded T-gate count.

    Produces a massive Clifford circuit with exactly `t_count` T gates
    inserted at random positions, followed by its exact inverse.
    The combined circuit U U-dag = I, so measuring all qubits must
    yield the all-zeros bitstring.

    Args:
        num_qubits: Number of qubits.
        clifford_gate_count: Total number of random Clifford gates.
        t_count: Exact number of T gates to insert.
        seed: Random seed.

    Returns:
        Tuple of (circuit_string, t_count) where circuit_string includes
        final measurements of all qubits.
    """
    rng = np.random.default_rng(seed)
    gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z"]
    gates_2q = ["CX", "CY", "CZ"]

    # Build forward Clifford circuit
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

    # Insert exactly t_count T gates at random positions
    positions = sorted(rng.choice(len(fwd) + 1, size=t_count, replace=False))
    for offset, pos in enumerate(positions):
        q = int(rng.integers(0, num_qubits))
        fwd.insert(int(pos) + offset, f"T {q}")

    # Compute exact inverse
    inv: list[str] = []
    for line in reversed(fwd):
        parts = line.split()
        gate = _DAGGER_MAP[parts[0]]
        targets = " ".join(parts[1:])
        inv.append(f"{gate} {targets}")

    # Combine: U, U-dag, then measure all
    meas = "M " + " ".join(str(i) for i in range(num_qubits))
    full = fwd + inv + [meas]
    return "\n".join(full), t_count


class TestBoundedTMirrorFuzzer:
    """40-qubit mirror circuits with bounded T gates.

    Validates that U U-dag = I at scale by checking that every shot
    yields the all-zeros bitstring. The peak rank must stay <= t_count
    since only T gates expand the active Schrodinger array.

    The baseline tests verify the compiler and VM produce correct results
    without optimization. The optimized tests verify the peephole pass
    cancels all T/T-dag pairs, collapsing peak_rank to 0.
    """

    NUM_QUBITS = 40
    CLIFFORD_DEPTH = 1000
    T_COUNT = 12
    SHOTS = 10_000

    @pytest.mark.parametrize("seed", range(5))
    def test_mirror_40q_12t_baseline(self, seed: int) -> None:
        """40-qubit mirror with 12 T gates yields all-zeros without optimizer."""
        circuit, t_count = _bounded_t_mirror_circuit(
            self.NUM_QUBITS, self.CLIFFORD_DEPTH, self.T_COUNT, seed=seed
        )
        prog = ucc.compile(circuit)

        assert prog.peak_rank <= t_count, f"peak_rank={prog.peak_rank} exceeds t_count={t_count}"

        meas, _, _ = ucc.sample(prog, self.SHOTS, seed=seed)
        shots_nonzero = int(meas.sum(axis=1).astype(bool).sum())
        assert shots_nonzero == 0, f"{shots_nonzero}/{self.SHOTS} shots were non-zero (seed={seed})"

    @pytest.mark.parametrize("seed", range(5))
    def test_mirror_40q_12t_optimized(self, seed: int) -> None:
        """Optimizer cancels all T gates: peak_rank=0, all-zeros output."""
        circuit, _ = _bounded_t_mirror_circuit(
            self.NUM_QUBITS, self.CLIFFORD_DEPTH, self.T_COUNT, seed=seed
        )
        prog = _compile_optimized(circuit)

        assert (
            prog.peak_rank == 0
        ), f"peak_rank={prog.peak_rank}, expected 0 after optimization (seed={seed})"

        meas, _, _ = ucc.sample(prog, self.SHOTS, seed=seed)
        shots_nonzero = int(meas.sum(axis=1).astype(bool).sum())
        assert shots_nonzero == 0, f"{shots_nonzero}/{self.SHOTS} shots were non-zero (seed={seed})"

    def test_mirror_peak_rank_scales_with_t_count(self) -> None:
        """Peak rank grows with T count, not qubit count."""
        for t_count in [4, 8, 12]:
            circuit, _ = _bounded_t_mirror_circuit(self.NUM_QUBITS, 500, t_count, seed=99)
            prog = ucc.compile(circuit)
            assert prog.peak_rank <= t_count, f"t_count={t_count}: peak_rank={prog.peak_rank}"


# ---------------------------------------------------------------------------
# Breathing Memory Lifecycle
# ---------------------------------------------------------------------------


def _breathing_circuit(n_rounds: int) -> str:
    """Generate a circuit that breathes k: 1 -> 2 -> 1 repeatedly.

    Qubit 0 starts in an active non-Clifford state (H;T -> k=1).
    Each round injects qubit 1 into the active array (H;T -> k=2),
    entangles it with qubit 0 (CX), then measures qubit 1 (k -> 1)
    and resets it for the next round.

    Args:
        n_rounds: Number of inject-entangle-measure rounds.

    Returns:
        Circuit string in .stim format.
    """
    lines = ["H 0", "T 0"]  # Qubit 0 enters active array (k=1)
    for _ in range(n_rounds):
        lines.append("H 1")
        lines.append("T 1")  # k: 1 -> 2
        lines.append("CX 1 0")  # Entangle
        lines.append("M 1")  # k: 2 -> 1
        lines.append("R 1")  # Reset for next round
    lines.append("M 0")
    return "\n".join(lines)


class TestBreathingMemoryLifecycle:
    """Stress the virtual register manager with repeated expand/compact.

    Each round injects a T-state qubit, entangles it, and measures it,
    forcing the array to repeatedly expand and contract. The gamma
    scalar accumulates hundreds of 1/sqrt(2) factors from measurement
    normalization; the amortized renormalization must prevent underflow.
    """

    @pytest.mark.parametrize("n_rounds", [10, 100, 500])
    def test_peak_rank_bounded(self, n_rounds: int) -> None:
        """Peak rank stays at exactly 2 regardless of round count."""
        circuit = _breathing_circuit(n_rounds)
        prog = ucc.compile(circuit)
        assert prog.peak_rank == 2, f"n_rounds={n_rounds}: peak_rank={prog.peak_rank}, expected 2"

    def test_breathing_500_rounds_completes(self) -> None:
        """500-round breathing circuit runs without underflow or crash."""
        circuit = _breathing_circuit(500)
        prog = ucc.compile(circuit)

        assert prog.peak_rank == 2
        # Memory: 2^2 * 16 bytes = 64 bytes (trivial)

        meas, _, _ = ucc.sample(prog, 1000, seed=42)
        # All 1000 shots must complete (no NaN, no crash)
        assert meas.shape == (1000, 501)  # 500 mid-circuit + 1 final
        # No NaN-induced garbage: every measurement must be 0 or 1
        assert np.all((meas == 0) | (meas == 1))

    def test_breathing_1000_rounds_completes(self) -> None:
        """1000-round breathing circuit -- extreme gamma stress test."""
        circuit = _breathing_circuit(1000)
        prog = ucc.compile(circuit)

        assert prog.peak_rank == 2

        meas, _, _ = ucc.sample(prog, 100, seed=7)
        assert meas.shape == (100, 1001)
        assert np.all((meas == 0) | (meas == 1))


# ---------------------------------------------------------------------------
# Biased Amplitude Statistics
# ---------------------------------------------------------------------------

# Analytical circuits with exact P(0) values.
# Each entry: (name, circuit_string, expected P(0))
_BIASED_CIRCUITS: list[tuple[str, str, float]] = [
    # H;T;H rotates |0> by pi/8 around Z then back to comp. basis.
    # P(0) = cos^2(pi/8) = (1 + cos(pi/4)) / 2
    (
        "H-T-H single rotation",
        "H 0\nT 0\nH 0\nM 0",
        (1.0 + np.cos(np.pi / 4)) / 2.0,
    ),
    # H;T_DAG;H is the conjugate rotation.
    # P(0) = cos^2(pi/8) (same magnitude, opposite phase)
    (
        "H-Tdag-H conjugate rotation",
        "H 0\nT_DAG 0\nH 0\nM 0",
        (1.0 + np.cos(np.pi / 4)) / 2.0,
    ),
    # H;T;T;H = H;S;H. S adds pi/2 phase, so P(0) = cos^2(pi/4) = 0.5
    (
        "H-S-H quarter turn",
        "H 0\nT 0\nT 0\nH 0\nM 0",
        0.5,
    ),
    # Three T gates: H;T;T;T;H. Phase = 3*pi/4.
    # P(0) = (1 + cos(3*pi/4)) / 2
    (
        "H-TTT-H three-eighth turn",
        "H 0\nT 0\nT 0\nT 0\nH 0\nM 0",
        (1.0 + np.cos(3.0 * np.pi / 4.0)) / 2.0,
    ),
]


class TestBiasedAmplitudeStatistics:
    """Validate RNG branch selection on asymmetric probability splits.

    Uses circuits with analytically known measurement biases to verify
    that the VM's Born-rule sampling produces correct distributions.
    """

    SHOTS = 100_000

    @pytest.mark.parametrize(
        "name,circuit,expected_p0",
        _BIASED_CIRCUITS,
        ids=[c[0] for c in _BIASED_CIRCUITS],
    )
    def test_biased_single_qubit(self, name: str, circuit: str, expected_p0: float) -> None:
        """Single-qubit biased circuit matches analytical P(0)."""
        prog = ucc.compile(circuit)
        meas, _, _ = ucc.sample(prog, self.SHOTS, seed=42)

        observed_p0 = float(1.0 - meas[:, 0].astype(float).mean())
        tol = binomial_tolerance(expected_p0, self.SHOTS, sigma=5.0)
        diff = abs(observed_p0 - expected_p0)
        assert diff < tol, (
            f"{name}: P(0)={observed_p0:.6f}, expected={expected_p0:.6f}, "
            f"diff={diff:.6f} > tol={tol:.6f}"
        )

    def test_biased_entangled_pair(self) -> None:
        """Entangled 2-qubit circuit with T-gate bias.

        H 0; T 0; H 0; CX 0 1; M 0 1
        The T rotation biases qubit 0 before CX copies it to qubit 1.
        Both qubits have cos^2(pi/8) marginal, and m0 XOR m1 = 0 always.
        """
        circuit = "H 0\nT 0\nH 0\nCX 0 1\nM 0 1"
        prog = ucc.compile(circuit)
        meas, _, _ = ucc.sample(prog, self.SHOTS, seed=42)

        m0 = meas[:, 0].astype(float)
        m1 = meas[:, 1].astype(float)

        # Both qubits have cos^2(pi/8) bias for |0>
        expected_p0 = (1.0 + np.cos(np.pi / 4)) / 2.0
        for qi, mi in [(0, m0), (1, m1)]:
            observed = float(1.0 - mi.mean())
            tol = binomial_tolerance(expected_p0, self.SHOTS, sigma=5.0)
            assert (
                abs(observed - expected_p0) < tol
            ), f"Qubit {qi} marginal: {observed:.6f} vs {expected_p0:.6f}"

        # CX copies q0 to q1: parity must be exactly 0
        parity_nonzero = int((meas[:, 0] ^ meas[:, 1]).sum())
        assert parity_nonzero == 0, f"{parity_nonzero}/{self.SHOTS} shots had m0 != m1"

    @pytest.mark.parametrize("seed", range(3))
    def test_biased_multi_t_rotation(self, seed: int) -> None:
        """Verify bias from N sequential T gates matches cos^2(N*pi/8)."""
        for n_t in [1, 2, 3, 5, 7]:
            lines = ["H 0"] + ["T 0"] * n_t + ["H 0", "M 0"]
            circuit = "\n".join(lines)
            expected_p0 = (1.0 + np.cos(n_t * np.pi / 4.0)) / 2.0

            prog = ucc.compile(circuit)
            meas, _, _ = ucc.sample(prog, self.SHOTS, seed=seed)
            observed_p0 = float(1.0 - meas[:, 0].astype(float).mean())

            tol = binomial_tolerance(expected_p0, self.SHOTS, sigma=5.0)
            diff = abs(observed_p0 - expected_p0)
            assert diff < tol, (
                f"n_t={n_t}, seed={seed}: P(0)={observed_p0:.6f}, "
                f"expected={expected_p0:.6f}, diff={diff:.6f} > tol={tol:.6f}"
            )
