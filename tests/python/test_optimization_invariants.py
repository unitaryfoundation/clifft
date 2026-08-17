"""Differential checks for the default HIR optimization pipeline.

The pipeline is checked through statevector equivalence and statistical
distribution matching (marginal probabilities agree within binomial
tolerance on noisy circuits).
"""

from typing import Any

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equiv,
    cross_binomial_tolerance,
    random_clifford_t_circuit,
    random_dense_clifford_t_circuit,
)
from utils_fuzzing import (
    generate_random_commutation_circuit,
    generate_star_graph_stress_circuit,
    generate_uncomputation_ladder,
)

import clifft

_MAX_PEAK_RANK = 12
_SEEDS = [0, 1, 2, 3, 4]

# Default HIR pipeline against statevectors.


def _clifft_statevector(circuit_str: str, **compile_kwargs: Any) -> np.ndarray:
    """Compile and execute a noiseless circuit, return dense statevector."""
    return np.asarray(clifft.get_statevector(clifft.compile(circuit_str, **compile_kwargs)))


class TestDefaultHirStatevectorEquivalence:
    """Check that the default HIR pipeline preserves noiseless statevectors.

    For small noiseless Clifford+T circuits, expand both the unoptimized and
    HIR-optimized factored states to dense 2^n statevectors and assert
    fidelity ~= 1. This validates algebraic correctness up to global phase,
    regardless of active/dormant geometry changes.
    """

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_random_clifford_t_5q(self, seed: int) -> None:
        circuit = random_clifford_t_circuit(5, 40, seed=seed)
        base_sv = _clifft_statevector(circuit, hir_passes=None)
        opt_sv = _clifft_statevector(
            circuit,
            hir_passes=clifft.default_hir_pass_manager(),
        )
        assert_statevectors_equiv(opt_sv, base_sv)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_dense_clifford_t_4q(self, seed: int) -> None:
        circuit = random_dense_clifford_t_circuit(4, 50, seed=seed)
        base_sv = _clifft_statevector(circuit, hir_passes=None)
        opt_sv = _clifft_statevector(
            circuit,
            hir_passes=clifft.default_hir_pass_manager(),
        )
        assert_statevectors_equiv(opt_sv, base_sv)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_dense_clifford_t_8q(self, seed: int) -> None:
        circuit = random_dense_clifford_t_circuit(8, 60, seed=seed)
        base_sv = _clifft_statevector(circuit, hir_passes=None)
        opt_sv = _clifft_statevector(
            circuit,
            hir_passes=clifft.default_hir_pass_manager(),
        )
        assert_statevectors_equiv(opt_sv, base_sv)


# Full default pipeline against sampling distributions.

_STAT_SHOTS = 10_000


class TestDefaultOptimizerStatisticalEquivalence:
    """Check that the default optimizers preserve noisy measurement distributions.

    Since HIR optimization can change active/dormant geometry (and thus PRNG
    trajectory), exact trajectory matching is impossible for stochastic
    circuits. Instead we sample many shots and verify that every measurement
    marginal probability matches within 5-sigma binomial tolerance.
    """

    @staticmethod
    def _assert_marginals_match(
        base_m: np.ndarray, opt_m: np.ndarray, *, sigma: float = 5.0
    ) -> None:
        """Assert per-column marginal probabilities match within tolerance."""
        shots = base_m.shape[0]
        assert base_m.shape == opt_m.shape

        base_probs = base_m.mean(axis=0)
        opt_probs = opt_m.mean(axis=0)

        for col in range(base_m.shape[1]):
            p_pooled = (base_probs[col] + opt_probs[col]) / 2.0
            tol = cross_binomial_tolerance(p_pooled, shots, sigma=sigma)
            diff = abs(float(base_probs[col] - opt_probs[col]))
            assert diff < tol, (
                f"Measurement column {col}: base={base_probs[col]:.4f}, "
                f"opt={opt_probs[col]:.4f}, diff={diff:.4f}, tol={tol:.4f}"
            )

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("circuit_kind", ["star", "commutation", "uncomputation"])
    def test_symbolic_hir_pipeline(self, circuit_kind: str, seed: int) -> None:
        """Production HIR optimization preserves noisy output distributions."""
        if circuit_kind == "star":
            circuit = generate_star_graph_stress_circuit(10, 100, seed=seed)
        elif circuit_kind == "commutation":
            circuit = generate_random_commutation_circuit(20, 200, seed=seed)
        else:
            circuit = generate_uncomputation_ladder(10, 100, seed=seed, noise_prob=0.02)

        base = clifft.compile(circuit, hir_passes=None)
        optimized = clifft.compile(circuit)
        assert base.peak_rank <= _MAX_PEAK_RANK
        assert optimized.peak_rank <= _MAX_PEAK_RANK

        base_result = clifft.sample(base, _STAT_SHOTS, seed=seed)
        optimized_result = clifft.sample(optimized, _STAT_SHOTS, seed=seed)
        self._assert_marginals_match(
            base_result.measurements,
            optimized_result.measurements,
        )
