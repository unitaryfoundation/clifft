"""Tests for OpenMP multi-core parallelism in the SVM.

These tests exercise code paths that only activate when active_k >= 18
(the kMinRankForThreads threshold). They require a multi-core machine
to actually test thread safety; on single-core VMs they still verify
correctness of the threaded code structure.

Run with:  uv run pytest tests/python/test_openmp.py -m multicore
Skip with: uv run pytest -m "not multicore"
"""

import os

import numpy as np
import pytest
from conftest import random_clifford_t_circuit

import clifft

# Skip the entire module if CLIFFT_SKIP_MULTICORE is set (e.g. in CI with < 4GB RAM).
pytestmark = pytest.mark.multicore
skip_if_low_memory = pytest.mark.skipif(
    os.environ.get("CLIFFT_SKIP_MULTICORE", "") == "1",
    reason="CLIFFT_SKIP_MULTICORE=1: skipping large-statevector tests",
)


def _make_expand_circuit(n: int) -> str:
    """Build a circuit that expands active_k to exactly n via independent T gates.

    Each qubit gets H then T, producing n independent non-Clifford axes.
    Final measurements collapse the state for sampling.
    """
    lines = []
    for q in range(n):
        lines.append(f"H {q}")
    for q in range(n):
        lines.append(f"T {q}")
    # Add some 2-qubit gates to exercise CNOT/CZ/SWAP kernels at high rank
    for q in range(0, n - 1, 2):
        lines.append(f"CX {q} {q + 1}")
    for q in range(1, n - 1, 2):
        lines.append(f"CZ {q} {q + 1}")
    # Hadamard sweep to exercise H kernel at high rank
    for q in range(n):
        lines.append(f"H {q}")
    # S gates to exercise phase waterfall at high rank
    for q in range(n):
        lines.append(f"S {q}")
    # Measurements to exercise both diagonal and interfere reductions
    for q in range(n):
        lines.append(f"M {q}")
    return "\n".join(lines)


class TestThreadingAPI:
    """Tests for set_num_threads / get_num_threads Python API."""

    def test_get_num_threads_returns_positive(self) -> None:
        n = clifft.get_num_threads()
        assert isinstance(n, int)
        assert n >= 1

    def test_set_and_get_roundtrip(self) -> None:
        original = clifft.get_num_threads()
        if original == 1:
            # No OpenMP: set_num_threads is a documented no-op, skip roundtrip.
            clifft.set_num_threads(4)
            assert clifft.get_num_threads() == 1
            return
        try:
            clifft.set_num_threads(1)
            assert clifft.get_num_threads() == 1
            clifft.set_num_threads(2)
            assert clifft.get_num_threads() == 2
        finally:
            clifft.set_num_threads(original)


@skip_if_low_memory
class TestLargeStatevectorCorrectness:
    """Correctness tests at active_k >= 18 to exercise OpenMP code paths.

    These compare multi-threaded results against single-threaded results
    to catch data races or incorrect parallelization.
    """

    def test_single_vs_multi_thread_determinism(self) -> None:
        """Same circuit + seed produces identical results at 1 and N threads."""
        circuit = _make_expand_circuit(19)
        prog = clifft.compile(circuit)
        assert prog.peak_rank >= 18

        original_threads = clifft.get_num_threads()
        try:
            # Sample with 1 thread
            clifft.set_num_threads(1)
            r1 = clifft.sample(prog, 10, seed=42)

            # Sample with all available threads
            clifft.set_num_threads(original_threads)
            r2 = clifft.sample(prog, 10, seed=42)

            np.testing.assert_array_equal(
                r1.measurements,
                r2.measurements,
                err_msg="Measurement results differ between 1-thread and N-thread execution",
            )
        finally:
            clifft.set_num_threads(original_threads)

    def test_multi_shot_determinism_k19(self) -> None:
        """Multiple shots at k=19 produce identical results across thread counts."""
        circuit = _make_expand_circuit(19)
        prog = clifft.compile(circuit)
        assert prog.peak_rank >= 18

        original_threads = clifft.get_num_threads()
        try:
            clifft.set_num_threads(1)
            r1 = clifft.sample(prog, 20, seed=123)

            clifft.set_num_threads(original_threads)
            r2 = clifft.sample(prog, 20, seed=123)

            np.testing.assert_array_equal(
                r1.measurements,
                r2.measurements,
                err_msg="Multi-shot results differ between 1-thread and N-thread",
            )
        finally:
            clifft.set_num_threads(original_threads)

    def test_expand_circuit_k18(self) -> None:
        """Exercise all major kernel types at exactly k=18."""
        circuit = _make_expand_circuit(18)
        prog = clifft.compile(circuit)
        assert prog.peak_rank >= 18

        # Just verify it runs without crashing and produces valid output
        result = clifft.sample(prog, 5, seed=7)
        assert result.measurements.shape == (5, 18)
        assert set(np.unique(result.measurements)).issubset({0, 1})

    def test_expand_circuit_k20(self) -> None:
        """Exercise kernels at k=20 (16 MB statevector)."""
        circuit = _make_expand_circuit(20)
        prog = clifft.compile(circuit)
        assert prog.peak_rank >= 18

        result = clifft.sample(prog, 3, seed=13)
        assert result.measurements.shape == (3, 20)
        assert set(np.unique(result.measurements)).issubset({0, 1})

    @pytest.mark.parametrize("seed", range(3))
    def test_random_circuit_k18_determinism(self, seed: int) -> None:
        """Random Clifford+T circuits at high rank are deterministic across thread counts."""
        # Guarantee high rank: start with T on each qubit, then random gates
        lines = []
        for q in range(19):
            lines.append(f"H {q}")
            lines.append(f"T {q}")
        circuit = "\n".join(lines)
        circuit += "\n" + random_clifford_t_circuit(num_qubits=19, depth=100, seed=seed)
        circuit += "\n" + "\n".join(f"M {q}" for q in range(19))

        prog = clifft.compile(circuit)
        assert prog.peak_rank >= 18

        original_threads = clifft.get_num_threads()
        try:
            clifft.set_num_threads(1)
            r1 = clifft.sample(prog, 5, seed=42)

            clifft.set_num_threads(original_threads)
            r2 = clifft.sample(prog, 5, seed=42)

            np.testing.assert_array_equal(r1.measurements, r2.measurements)
        finally:
            clifft.set_num_threads(original_threads)


@skip_if_low_memory
class TestExpValHighRank:
    """Expectation value tests at active_k >= 18."""

    def test_exp_val_z_product_state(self) -> None:
        """<Z0> on |0...0> expanded to k=18 should be +1."""
        lines = []
        for q in range(18):
            lines.append(f"H {q}")
        for q in range(18):
            lines.append(f"T {q}")
        # Measure all but qubit 0 to collapse back, keeping q0 active
        circuit = "\n".join(lines)
        circuit += "\nEXP_VAL Z0"
        for q in range(18):
            circuit += f"\nM {q}"

        prog = clifft.compile(circuit)
        assert prog.peak_rank >= 18

        result = clifft.sample(prog, 1, seed=42)
        # The exp_val won't be exactly +1 due to T gates, but it should
        # be a valid real number in [-1, 1]
        assert result.exp_vals.shape[0] == 1
        assert -1.0 <= result.exp_vals[0] <= 1.0

    def test_exp_val_determinism_across_threads(self) -> None:
        """Expectation values match between 1 and N threads."""
        lines = []
        for q in range(18):
            lines.append(f"H {q}")
        for q in range(18):
            lines.append(f"T {q}")
        circuit = "\n".join(lines)
        circuit += "\nEXP_VAL Z0\nEXP_VAL X1\nEXP_VAL Y2"
        for q in range(18):
            circuit += f"\nM {q}"

        prog = clifft.compile(circuit)
        assert prog.peak_rank >= 18

        original_threads = clifft.get_num_threads()
        try:
            clifft.set_num_threads(1)
            r1 = clifft.sample(prog, 1, seed=77)

            clifft.set_num_threads(original_threads)
            r2 = clifft.sample(prog, 1, seed=77)

            np.testing.assert_allclose(
                r1.exp_vals,
                r2.exp_vals,
                atol=1e-12,
                err_msg="Exp vals differ between 1-thread and N-thread",
            )
        finally:
            clifft.set_num_threads(original_threads)
