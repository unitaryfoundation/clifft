"""SVM state-lifecycle and sampling checks with analytic expectations.

These tests cover repeated active-rank expansion and compaction, along with
small circuits whose output probabilities are known in closed form.
"""

from typing import Any

import numpy as np
import pytest
from conftest import binomial_tolerance

from clifft import _legacy


def _repeated_expand_measure_circuit(n_rounds: int) -> str:
    """Generate a circuit whose active rank repeatedly changes 1 -> 2 -> 1.

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


class TestRepeatedExpansionAndCompaction:
    """Exercise the virtual register manager through repeated rank changes.

    Each round injects a T-state qubit, entangles it, and measures it,
    forcing the array to repeatedly expand and contract. The gamma
    scalar accumulates hundreds of 1/sqrt(2) factors from measurement
    normalization; the amortized renormalization must prevent underflow.
    """

    @pytest.mark.parametrize("n_rounds", [10, 100, 500])
    def test_peak_rank_bounded(self, n_rounds: int) -> None:
        """Peak rank stays at exactly 2 regardless of round count."""
        circuit = _repeated_expand_measure_circuit(n_rounds)
        prog = _legacy.compile(circuit, hir_passes=None, bytecode_passes=None)
        assert prog.peak_rank == 2, f"n_rounds={n_rounds}: peak_rank={prog.peak_rank}, expected 2"

    def test_500_rounds_complete(self) -> None:
        """A 500-round circuit completes without underflow."""
        circuit = _repeated_expand_measure_circuit(500)
        prog = _legacy.compile(circuit, hir_passes=None, bytecode_passes=None)

        assert prog.peak_rank == 2
        # Memory: 2^2 * 16 bytes = 64 bytes (trivial)

        result = _legacy.sample(prog, 1000, seed=42)
        # All 1000 shots must complete (no NaN, no crash)
        assert result.measurements.shape == (1000, 501)  # 500 mid-circuit + 1 final
        # No NaN-induced garbage: every measurement must be 0 or 1
        assert np.all((result.measurements == 0) | (result.measurements == 1))

    def test_1000_rounds_complete(self) -> None:
        """Normalization remains finite through 1000 rounds."""
        circuit = _repeated_expand_measure_circuit(1000)
        prog = _legacy.compile(circuit, hir_passes=None, bytecode_passes=None)

        assert prog.peak_rank == 2

        result = _legacy.sample(prog, 100, seed=7)
        assert result.measurements.shape == (100, 1001)
        assert np.all((result.measurements == 0) | (result.measurements == 1))


# Biased-amplitude statistics.

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
    def test_biased_single_qubit(
        self, name: str, circuit: str, expected_p0: float, sampling_api: Any
    ) -> None:
        """Single-qubit biased circuit matches analytical P(0)."""
        prog = sampling_api.compile(circuit)
        result = sampling_api.sample(prog, self.SHOTS, seed=42)

        observed_p0 = float(1.0 - result.measurements[:, 0].astype(float).mean())
        tol = binomial_tolerance(expected_p0, self.SHOTS, sigma=5.0)
        diff = abs(observed_p0 - expected_p0)
        assert diff < tol, (
            f"{name}: P(0)={observed_p0:.6f}, expected={expected_p0:.6f}, "
            f"diff={diff:.6f} > tol={tol:.6f}"
        )

    def test_biased_entangled_pair(self, sampling_api: Any) -> None:
        """Entangled 2-qubit circuit with T-gate bias.

        H 0; T 0; H 0; CX 0 1; M 0 1
        The T rotation biases qubit 0 before CX copies it to qubit 1.
        Both qubits have cos^2(pi/8) marginal, and m0 XOR m1 = 0 always.
        """
        circuit = "H 0\nT 0\nH 0\nCX 0 1\nM 0 1"
        prog = sampling_api.compile(circuit)
        result = sampling_api.sample(prog, self.SHOTS, seed=42)

        m0 = result.measurements[:, 0].astype(float)
        m1 = result.measurements[:, 1].astype(float)

        # Both qubits have cos^2(pi/8) bias for |0>
        expected_p0 = (1.0 + np.cos(np.pi / 4)) / 2.0
        for qi, mi in [(0, m0), (1, m1)]:
            observed = float(1.0 - mi.mean())
            tol = binomial_tolerance(expected_p0, self.SHOTS, sigma=5.0)
            assert (
                abs(observed - expected_p0) < tol
            ), f"Qubit {qi} marginal: {observed:.6f} vs {expected_p0:.6f}"

        # CX copies q0 to q1: parity must be exactly 0
        parity_nonzero = int((result.measurements[:, 0] ^ result.measurements[:, 1]).sum())
        assert parity_nonzero == 0, f"{parity_nonzero}/{self.SHOTS} shots had m0 != m1"

    @pytest.mark.parametrize("seed", range(3))
    def test_biased_multi_t_rotation(self, seed: int, sampling_api: Any) -> None:
        """Verify bias from N sequential T gates matches cos^2(N*pi/8)."""
        for n_t in [1, 2, 3, 5, 7]:
            lines = ["H 0"] + ["T 0"] * n_t + ["H 0", "M 0"]
            circuit = "\n".join(lines)
            expected_p0 = (1.0 + np.cos(n_t * np.pi / 4.0)) / 2.0

            prog = sampling_api.compile(circuit)
            result = sampling_api.sample(prog, self.SHOTS, seed=seed)
            observed_p0 = float(1.0 - result.measurements[:, 0].astype(float).mean())

            tol = binomial_tolerance(expected_p0, self.SHOTS, sigma=5.0)
            diff = abs(observed_p0 - expected_p0)
            assert diff < tol, (
                f"n_t={n_t}, seed={seed}: P(0)={observed_p0:.6f}, "
                f"expected={expected_p0:.6f}, diff={diff:.6f} > tol={tol:.6f}"
            )
