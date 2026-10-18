"""Differential checks for the default HIR optimization and scheduling pipelines.

The pipelines are checked through statevector equivalence and statistical
distribution matching (column marginals, pairwise XOR parities, and full joint
bitstring distributions on small-record circuits, including detectors and
observables).
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
from utils_conformance import default_scheduled_passes, unitary_reference
from utils_fuzzing import (
    generate_planted_hazard_circuit,
    generate_planted_unitary_circuit,
    generate_random_commutation_circuit,
    generate_star_graph_stress_circuit,
    generate_uncomputation_ladder,
)

import clifft

_MAX_PEAK_ACTIVE_WIDTH = 12
_SEEDS = [0, 1, 2, 3, 4]

# Default HIR pipeline against statevectors.


def _clifft_statevector(circuit_str: str, **compile_kwargs: Any) -> np.ndarray:
    """Compile and execute a noiseless circuit, return dense statevector."""
    return np.asarray(clifft.get_statevector(clifft.compile(circuit_str, **compile_kwargs)))


class TestDefaultHirStatevectorEquivalence:
    """Check that the default HIR pipeline preserves noiseless statevectors.

    For small noiseless Clifford+T circuits, expand both the unoptimized and
    HIR-optimized factored states to dense 2^n statevectors and compare their
    phase-aligned amplitudes. This validates algebraic correctness up to global
    phase regardless of active/dormant geometry changes.
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

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("scaffold_kind", ["phase_sandwich", "stabilizer_reduced"])
    def test_planted_unitary_scaffolds_against_aer(self, scaffold_kind: str, seed: int) -> None:
        circuit = generate_planted_unitary_circuit(scaffold_kind, seed=seed)
        aer_sv = unitary_reference(circuit)
        base_sv = _clifft_statevector(circuit, hir_passes=None)
        default_sv = _clifft_statevector(
            circuit,
            hir_passes=clifft.default_hir_pass_manager(),
        )
        scheduled_sv = _clifft_statevector(
            circuit,
            hir_passes=default_scheduled_passes(),
        )
        assert_statevectors_equiv(base_sv, aer_sv)
        assert_statevectors_equiv(default_sv, aer_sv)
        assert_statevectors_equiv(scheduled_sv, aer_sv)


# Full default and scheduled pipelines against sampling distributions.

_STAT_SHOTS = 10_000


class TestDefaultOptimizerStatisticalEquivalence:
    """Check that the default optimizers preserve noisy measurement distributions.

    Since HIR optimization can change active/dormant geometry (and thus PRNG
    trajectory), exact trajectory matching is impossible for stochastic
    circuits. Instead we sample many shots and verify column marginals,
    pairwise XOR correlations, and (for small record widths) full joint
    bitstring frequencies across measurements, detectors, and observables.
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
                f"Column {col}: base={base_probs[col]:.4f}, "
                f"opt={opt_probs[col]:.4f}, diff={diff:.4f}, tol={tol:.4f}"
            )

    @classmethod
    def _assert_joint_and_pairwise_match(
        cls,
        base_arr: np.ndarray,
        opt_arr: np.ndarray,
        *,
        sigma: float = 5.5,
    ) -> None:
        """Assert marginals, pairwise XOR correlations, and small-width joint histograms match."""
        assert base_arr.shape == opt_arr.shape
        shots, cols = base_arr.shape
        if cols == 0:
            return

        cls._assert_marginals_match(base_arr, opt_arr, sigma=sigma)

        # Pairwise XOR parities detect correlation drops that preserve 1-bit marginals.
        pair_cols = min(cols, 16)
        for i in range(pair_cols):
            for j in range(i + 1, pair_cols):
                base_xor = float(np.bitwise_xor(base_arr[:, i], base_arr[:, j]).mean())
                opt_xor = float(np.bitwise_xor(opt_arr[:, i], opt_arr[:, j]).mean())
                p_pooled = (base_xor + opt_xor) / 2.0
                tol = cross_binomial_tolerance(p_pooled, shots, sigma=sigma)
                diff = abs(base_xor - opt_xor)
                assert diff < tol, (
                    f"Pairwise XOR ({i}, {j}): base={base_xor:.4f}, "
                    f"opt={opt_xor:.4f}, diff={diff:.4f}, tol={tol:.4f}"
                )

        if cols <= 8:
            weights = 1 << np.arange(cols, dtype=np.int64)
            base_keys = base_arr.astype(np.int64) @ weights
            opt_keys = opt_arr.astype(np.int64) @ weights
            base_freq = np.bincount(base_keys, minlength=1 << cols) / shots
            opt_freq = np.bincount(opt_keys, minlength=1 << cols) / shots
            for bitstring in range(1 << cols):
                p_pooled = float(base_freq[bitstring] + opt_freq[bitstring]) / 2.0
                tol = cross_binomial_tolerance(p_pooled, shots, sigma=sigma)
                diff = abs(float(base_freq[bitstring] - opt_freq[bitstring]))
                assert diff < tol, (
                    f"Joint bitstring {bitstring:#0{cols + 2}b}: "
                    f"base={base_freq[bitstring]:.4f}, opt={opt_freq[bitstring]:.4f}, "
                    f"diff={diff:.4f}, tol={tol:.4f}"
                )

    @classmethod
    def _assert_sample_results_match(cls, base_result: Any, opt_result: Any) -> None:
        cls._assert_joint_and_pairwise_match(base_result.measurements, opt_result.measurements)
        cls._assert_joint_and_pairwise_match(base_result.detectors, opt_result.detectors)
        cls._assert_joint_and_pairwise_match(base_result.observables, opt_result.observables)
        # Place detectors and observables first so cross-correlations with
        # measurements are always covered by the pairwise window even on
        # wide measurement records, and small-width circuits check the full
        # joint (detectors, observables, measurements) histogram.
        base_combined = np.hstack(
            [base_result.detectors, base_result.observables, base_result.measurements]
        )
        opt_combined = np.hstack(
            [opt_result.detectors, opt_result.observables, opt_result.measurements]
        )
        cls._assert_joint_and_pairwise_match(base_combined, opt_combined)

    def test_joint_and_pairwise_checker_rejects_correlation_mismatch(self) -> None:
        """Negative control: two circuits with 0.5 column marginals but different correlations."""
        correlated = clifft.compile("H 0\nM 0\nCX rec[-1] 1\nM 1\n", hir_passes=None)
        independent = clifft.compile("H 0\nH 1\nM 0\nM 1\n", hir_passes=None)
        corr_res = clifft.sample(correlated, 4_000, seed=11)
        indep_res = clifft.sample(independent, 4_000, seed=12)

        # Single-column marginals are 0.5 in both circuits and pass marginal checks.
        self._assert_marginals_match(corr_res.measurements, indep_res.measurements)
        with pytest.raises(AssertionError, match="Pairwise XOR"):
            self._assert_joint_and_pairwise_match(corr_res.measurements, indep_res.measurements)

        # Cross-array negative control: each individual output array (measurements,
        # detectors, observables) has identical distribution, but detector vs
        # observable correlation differs when DETECTOR references rec[-2] vs rec[-1].
        det_same = clifft.compile(
            "H 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
            hir_passes=None,
        )
        det_wrong = clifft.compile(
            "H 0 1\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
            hir_passes=None,
        )
        same_res = clifft.sample(det_same, 4_000, seed=7)
        wrong_res = clifft.sample(det_wrong, 4_000, seed=7)
        self._assert_joint_and_pairwise_match(same_res.measurements, wrong_res.measurements)
        self._assert_joint_and_pairwise_match(same_res.detectors, wrong_res.detectors)
        self._assert_joint_and_pairwise_match(same_res.observables, wrong_res.observables)
        with pytest.raises(AssertionError, match="Pairwise XOR"):
            self._assert_sample_results_match(same_res, wrong_res)

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
        assert base.peak_active_width <= _MAX_PEAK_ACTIVE_WIDTH
        assert optimized.peak_active_width <= _MAX_PEAK_ACTIVE_WIDTH

        base_result = clifft.sample(base, _STAT_SHOTS, seed=seed)
        optimized_result = clifft.sample(optimized, _STAT_SHOTS, seed=seed + 100)
        self._assert_sample_results_match(base_result, optimized_result)

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize(
        "scaffold_kind",
        ["phase_sandwich", "stabilizer_reduced", "feedback_reset", "multi_round_schedule"],
    )
    def test_planted_hazard_pipeline(self, scaffold_kind: str, seed: int) -> None:
        """Planted algebraic scaffolds with boundary hazards preserve joint distributions."""
        circuit = generate_planted_hazard_circuit(scaffold_kind, seed=seed)

        base = clifft.compile(circuit, hir_passes=None)
        default_opt = clifft.compile(circuit)
        scheduled_opt = clifft.compile(circuit, hir_passes=default_scheduled_passes())

        assert default_opt.peak_active_width <= base.peak_active_width
        assert scheduled_opt.peak_active_width <= default_opt.peak_active_width
        if scaffold_kind == "multi_round_schedule":
            assert scheduled_opt.peak_active_width < base.peak_active_width

        base_result = clifft.sample(base, _STAT_SHOTS, seed=seed + 200)
        default_result = clifft.sample(default_opt, _STAT_SHOTS, seed=seed + 300)
        scheduled_result = clifft.sample(scheduled_opt, _STAT_SHOTS, seed=seed + 400)

        self._assert_sample_results_match(base_result, default_result)
        self._assert_sample_results_match(base_result, scheduled_result)
