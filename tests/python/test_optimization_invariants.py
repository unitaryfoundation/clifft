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
)
from utils_conformance import (
    default_scheduled_passes,
    unitary_reference,
)
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
    """Check that the default and scheduled HIR pipelines preserve planted unitary statevectors."""

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

    @staticmethod
    def _assert_bounded_moment_match(
        mean_a: float,
        mean_b: float,
        shots: int,
        label: str,
        *,
        sigma: float = 5.5,
    ) -> None:
        """Compare sample means of random variables bounded in [-1, 1].

        For any Y in [-1, 1], Z = (1 + Y) / 2 lies in [0, 1], so by the
        Bhatia-Davis inequality Var(Z) <= E[Z](1 - E[Z]) and
        StdErr(Y_bar_a - Y_bar_b) <= 2 * StdErr(Bernoulli(E[Z])).
        """
        p_pooled = float(np.clip((mean_a + mean_b + 2.0) / 4.0, 0.0, 1.0))
        tol = max(2.0 * cross_binomial_tolerance(p_pooled, shots, sigma=sigma), 1e-9)
        diff = abs(mean_a - mean_b)
        assert diff < tol, (
            f"EXP_VAL {label}: base={mean_a:.4f}, opt={mean_b:.4f}, "
            f"diff={diff:.4f}, tol={tol:.4f}"
        )

    @classmethod
    def _assert_exp_vals_match(
        cls,
        base_classical: np.ndarray,
        base_ev: np.ndarray,
        opt_classical: np.ndarray,
        opt_ev: np.ndarray,
        *,
        sigma: float = 5.5,
    ) -> None:
        """Compare probe values and their joint correlations with classical outputs."""
        assert base_ev.shape == opt_ev.shape
        shots, num_probes = base_ev.shape
        if num_probes == 0:
            return

        # Per-probe first and second moments distinguish constant 0 probes from
        # +/-1 probes with zero mean (e.g., H 0; EXP_VAL Z0; M 0 vs H 0; M 0; EXP_VAL Z0).
        for p in range(num_probes):
            cls._assert_bounded_moment_match(
                float(np.mean(base_ev[:, p])),
                float(np.mean(opt_ev[:, p])),
                shots,
                f"mean[{p}]",
                sigma=sigma,
            )
            cls._assert_bounded_moment_match(
                float(np.mean(base_ev[:, p] ** 2)),
                float(np.mean(opt_ev[:, p] ** 2)),
                shots,
                f"second_moment[{p}]",
                sigma=sigma,
            )

        pair_probes = min(num_probes, 16)
        for p1 in range(pair_probes):
            for p2 in range(p1 + 1, pair_probes):
                cls._assert_bounded_moment_match(
                    float(np.mean(base_ev[:, p1] * base_ev[:, p2])),
                    float(np.mean(opt_ev[:, p1] * opt_ev[:, p2])),
                    shots,
                    f"probe_corr[{p1},{p2}]",
                    sigma=sigma,
                )

        cols = base_classical.shape[1]
        pair_cols = min(cols, 16)
        for c in range(pair_cols):
            base_spin = 1.0 - 2.0 * base_classical[:, c].astype(np.float64)
            opt_spin = 1.0 - 2.0 * opt_classical[:, c].astype(np.float64)
            for p in range(num_probes):
                cls._assert_bounded_moment_match(
                    float(np.mean(base_spin * base_ev[:, p])),
                    float(np.mean(opt_spin * opt_ev[:, p])),
                    shots,
                    f"classical_probe_corr[col={c},probe={p}]",
                    sigma=sigma,
                )

        if 0 < cols <= 8:
            weights = 1 << np.arange(cols, dtype=np.int64)
            base_keys = base_classical.astype(np.int64) @ weights
            opt_keys = opt_classical.astype(np.int64) @ weights
            num_bins = 1 << cols
            for p in range(num_probes):
                base_slice_mean = (
                    np.bincount(base_keys, weights=base_ev[:, p], minlength=num_bins) / shots
                )
                opt_slice_mean = (
                    np.bincount(opt_keys, weights=opt_ev[:, p], minlength=num_bins) / shots
                )
                base_slice_sq = (
                    np.bincount(base_keys, weights=base_ev[:, p] ** 2, minlength=num_bins) / shots
                )
                opt_slice_sq = (
                    np.bincount(opt_keys, weights=opt_ev[:, p] ** 2, minlength=num_bins) / shots
                )
                for bitstring in range(num_bins):
                    cls._assert_bounded_moment_match(
                        float(base_slice_mean[bitstring]),
                        float(opt_slice_mean[bitstring]),
                        shots,
                        f"joint_slice_mean[bits={bitstring:#0{cols + 2}b},probe={p}]",
                        sigma=sigma,
                    )
                    cls._assert_bounded_moment_match(
                        float(base_slice_sq[bitstring]),
                        float(opt_slice_sq[bitstring]),
                        shots,
                        f"joint_slice_sq[bits={bitstring:#0{cols + 2}b},probe={p}]",
                        sigma=sigma,
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
        cls._assert_exp_vals_match(
            base_combined, base_result.exp_vals, opt_combined, opt_result.exp_vals
        )

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

        # EXP_VAL probe negative controls:
        # 1. Moving EXP_VAL Z0 across M 0 on |+> changes per-shot probe values from
        #    always 0.0 to +/-1.0 (while keeping measurement marginals at 0.5 and
        #    unconditional probe mean near 0.0).
        probe_before_meas = clifft.compile("H 0\nEXP_VAL Z0\nM 0\n", hir_passes=None)
        probe_after_meas = clifft.compile("H 0\nM 0\nEXP_VAL Z0\n", hir_passes=None)
        before_res = clifft.sample(probe_before_meas, 4_000, seed=19)
        after_res = clifft.sample(probe_after_meas, 4_000, seed=20)
        self._assert_joint_and_pairwise_match(before_res.measurements, after_res.measurements)
        with pytest.raises(AssertionError, match="EXP_VAL"):
            self._assert_sample_results_match(before_res, after_res)

        # 2. Flipping the state between M 0 and EXP_VAL Z0 preserves both the
        #    measurement distribution (50/50) and the marginal probe distribution
        #    (+/-1.0 with 50/50 probability), but negates their joint correlation.
        probe_flipped = clifft.compile("H 0\nM 0\nX 0\nEXP_VAL Z0\n", hir_passes=None)
        flipped_res = clifft.sample(probe_flipped, 4_000, seed=21)
        with pytest.raises(AssertionError, match="EXP_VAL classical_probe_corr"):
            self._assert_sample_results_match(after_res, flipped_res)

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

        assert base.num_exp_vals > 0
        assert default_opt.peak_active_width <= base.peak_active_width
        assert scheduled_opt.peak_active_width <= default_opt.peak_active_width
        if scaffold_kind == "multi_round_schedule":
            assert scheduled_opt.peak_active_width < base.peak_active_width

        base_result = clifft.sample(base, _STAT_SHOTS, seed=seed + 200)
        default_result = clifft.sample(default_opt, _STAT_SHOTS, seed=seed + 300)
        scheduled_result = clifft.sample(scheduled_opt, _STAT_SHOTS, seed=seed + 400)

        self._assert_sample_results_match(base_result, default_result)
        self._assert_sample_results_match(base_result, scheduled_result)
