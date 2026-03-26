"""Differential invariant tests for optimization passes.

Validates that both bytecode passes (NoiseBlockPass, MultiGatePass,
SwapMeasPass, ExpandTPass) and HIR passes (PeepholeFusionPass) are
mathematically sound.

Bytecode passes: must preserve PRNG trajectory exactly (bit-for-bit
identical classical arrays given the same seed).

HIR passes: validated via statevector oracle (exact unitary preservation)
and statistical distribution matching (marginal probabilities agree
within binomial tolerance on noisy circuits).
"""

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equal,
    cross_binomial_tolerance,
    random_clifford_t_circuit,
    random_dense_clifford_t_circuit,
)
from utils_fuzzing import (
    generate_commutation_gauntlet,
    generate_star_graph_honeypot,
    generate_uncomputation_ladder,
)

import ucc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_PEAK_RANK = 12  # OOM guard: 4096 amplitudes (~64 KB)


def run_differential_trajectory(circuit_str: str, shots: int, seed: int) -> None:
    """Compile with and without bytecode passes, assert identical output.

    The baseline uses no passes at all. The optimized version uses only
    bytecode passes (no HIR passes) so that the lowered instruction stream
    is identical up to bytecode-level rewrites.
    """
    base_prog = ucc.compile(circuit_str)
    opt_prog = ucc.compile(
        circuit_str,
        bytecode_passes=ucc.default_bytecode_pass_manager(),
    )

    # --- Structural invariants ---
    assert (
        base_prog.peak_rank <= _MAX_PEAK_RANK
    ), f"Generator bug: peak_rank={base_prog.peak_rank} exceeds OOM guard {_MAX_PEAK_RANK}"
    assert (
        opt_prog.peak_rank <= base_prog.peak_rank
    ), f"Optimizer inflated peak_rank: {opt_prog.peak_rank} > {base_prog.peak_rank}"
    assert opt_prog.num_instructions <= base_prog.num_instructions, (
        f"Optimizer inflated instruction count: "
        f"{opt_prog.num_instructions} > {base_prog.num_instructions}"
    )
    assert opt_prog.num_measurements == base_prog.num_measurements, (
        f"Optimizer changed measurement count: "
        f"{opt_prog.num_measurements} != {base_prog.num_measurements}"
    )

    # --- PRNG trajectory synchronization ---
    base_result = ucc.sample(base_prog, shots, seed=seed)
    opt_result = ucc.sample(opt_prog, shots, seed=seed)

    np.testing.assert_array_equal(
        base_result.measurements,
        opt_result.measurements,
        err_msg="Measurement records diverged after bytecode optimization",
    )
    np.testing.assert_array_equal(
        base_result.detectors,
        opt_result.detectors,
        err_msg="Detector records diverged after bytecode optimization",
    )
    np.testing.assert_array_equal(
        base_result.observables,
        opt_result.observables,
        err_msg="Observable records diverged after bytecode optimization",
    )


# ---------------------------------------------------------------------------
# Generator configurations: (num_qubits, depth) pairs
# ---------------------------------------------------------------------------

_SMALL_CONFIGS = [(10, 100), (20, 200)]
_LARGE_CONFIGS = [(50, 500)]
_SEEDS = [0, 1, 2, 3, 4]
_SHOTS = 100


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestCommutationGauntlet:
    """Bytecode invariants on commutation gauntlet circuits."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _SMALL_CONFIGS)
    def test_small(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_commutation_gauntlet(nq, depth, seed=seed)
        run_differential_trajectory(circuit, _SHOTS, seed=seed)

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _LARGE_CONFIGS)
    def test_large(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_commutation_gauntlet(nq, depth, seed=seed)
        run_differential_trajectory(circuit, _SHOTS, seed=seed)


class TestStarGraphHoneypot:
    """Bytecode invariants on star-graph honeypot circuits."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _SMALL_CONFIGS)
    def test_small(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_star_graph_honeypot(nq, depth, seed=seed)
        run_differential_trajectory(circuit, _SHOTS, seed=seed)

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _LARGE_CONFIGS)
    def test_large(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_star_graph_honeypot(nq, depth, seed=seed)
        run_differential_trajectory(circuit, _SHOTS, seed=seed)


class TestUncomputationLadder:
    """Bytecode invariants on uncomputation ladder circuits."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _SMALL_CONFIGS)
    def test_small(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_uncomputation_ladder(nq, depth, seed=seed)
        run_differential_trajectory(circuit, _SHOTS, seed=seed)

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _LARGE_CONFIGS)
    def test_large(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_uncomputation_ladder(nq, depth, seed=seed)
        run_differential_trajectory(circuit, _SHOTS, seed=seed)


class TestHirPeepholeUncomputationLadder:
    """Validate HIR PeepholeFusionPass via noiseless uncomputation ladders.

    Because U * U_dag = I analytically, all final measurements are
    deterministic (outcome 0). The VM epsilon patch ensures both the
    unoptimized path (which sees FP dust in active measurements) and the
    optimized path (which collapses peak_rank via T-gate cancellation)
    bypass the PRNG entirely, producing identical all-zero records.
    """

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _SMALL_CONFIGS)
    def test_small(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_uncomputation_ladder(nq, depth, seed=seed, noise_prob=0.0)
        base = ucc.compile(circuit)
        opt = ucc.compile(circuit, hir_passes=ucc.default_hir_pass_manager())

        assert base.peak_rank <= _MAX_PEAK_RANK
        assert opt.peak_rank <= base.peak_rank

        base_result = ucc.sample(base, _SHOTS, seed=seed)
        opt_result = ucc.sample(opt, _SHOTS, seed=seed)

        # All measurements must be deterministic zero
        np.testing.assert_array_equal(
            base_result.measurements,
            np.zeros_like(base_result.measurements),
            err_msg="Unoptimized ladder produced non-zero measurements",
        )
        np.testing.assert_array_equal(
            opt_result.measurements,
            np.zeros_like(opt_result.measurements),
            err_msg="Optimized ladder produced non-zero measurements",
        )

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("nq,depth", _LARGE_CONFIGS)
    def test_large(self, nq: int, depth: int, seed: int) -> None:
        circuit = generate_uncomputation_ladder(nq, depth, seed=seed, noise_prob=0.0)
        base = ucc.compile(circuit)
        opt = ucc.compile(circuit, hir_passes=ucc.default_hir_pass_manager())

        assert base.peak_rank <= _MAX_PEAK_RANK
        assert opt.peak_rank <= base.peak_rank

        base_result = ucc.sample(base, _SHOTS, seed=seed)
        opt_result = ucc.sample(opt, _SHOTS, seed=seed)

        np.testing.assert_array_equal(
            base_result.measurements, np.zeros_like(base_result.measurements)
        )
        np.testing.assert_array_equal(
            opt_result.measurements, np.zeros_like(opt_result.measurements)
        )

    def test_dust_clamps_telemetry(self) -> None:
        """Prove the unoptimized ladder generates FP dust that the VM clamps."""
        # nq=4, depth=50 reliably produces base peak_rank=3 (active measurements
        # that encounter analytically-zero branches) while the optimizer reduces
        # peak_rank to 1. Both paths clamp dust, but the optimizer reduces it.
        circuit = generate_uncomputation_ladder(4, 50, seed=42, noise_prob=0.0)

        base = ucc.compile(circuit)
        assert base.peak_rank > 1, "Need active measurements to generate dust"
        base_state = ucc.State(base.peak_rank, base.num_measurements, seed=42)
        ucc.execute(base, base_state)
        assert (
            base_state.dust_clamps > 0
        ), "Unoptimized ladder should clamp FP dust in active measurements"

        opt = ucc.compile(circuit, hir_passes=ucc.default_hir_pass_manager())
        opt_state = ucc.State(opt.peak_rank, opt.num_measurements, seed=42)
        ucc.execute(opt, opt_state)
        assert (
            opt_state.dust_clamps <= base_state.dust_clamps
        ), "Optimizer should not increase the number of dust clamps"


# ---------------------------------------------------------------------------
# HIR Pass Validation: Statevector Oracle
# ---------------------------------------------------------------------------


def _ucc_statevector(circuit_str: str, **compile_kwargs: object) -> np.ndarray:
    """Compile and execute a noiseless circuit, return dense statevector."""
    prog = ucc.compile(circuit_str, **compile_kwargs)
    state = ucc.State(prog.peak_rank, prog.num_measurements)
    ucc.execute(prog, state)
    sv: np.ndarray = ucc.get_statevector(prog, state)
    return sv


class TestHirPeepholeStatevectorOracle:
    """Prove PeepholeFusionPass preserves the unitary via statevector comparison.

    For small noiseless Clifford+T circuits, expand both the unoptimized and
    HIR-optimized factored states to dense 2^n statevectors and assert
    fidelity ~= 1. This validates algebraic correctness regardless of
    active/dormant geometry changes.
    """

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_random_clifford_t_5q(self, seed: int) -> None:
        circuit = random_clifford_t_circuit(5, 40, seed=seed)
        base_sv = _ucc_statevector(circuit)
        opt_sv = _ucc_statevector(circuit, hir_passes=ucc.default_hir_pass_manager())
        assert_statevectors_equal(opt_sv, base_sv)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_dense_clifford_t_4q(self, seed: int) -> None:
        circuit = random_dense_clifford_t_circuit(4, 50, seed=seed)
        base_sv = _ucc_statevector(circuit)
        opt_sv = _ucc_statevector(circuit, hir_passes=ucc.default_hir_pass_manager())
        assert_statevectors_equal(opt_sv, base_sv)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_dense_clifford_t_8q(self, seed: int) -> None:
        circuit = random_dense_clifford_t_circuit(8, 60, seed=seed)
        base_sv = _ucc_statevector(circuit)
        opt_sv = _ucc_statevector(circuit, hir_passes=ucc.default_hir_pass_manager())
        assert_statevectors_equal(opt_sv, base_sv)


# ---------------------------------------------------------------------------
# HIR Pass Validation: Statistical Distribution Matching
# ---------------------------------------------------------------------------

_STAT_SHOTS = 10_000


class TestHirPeepholeStatisticalEquivalence:
    """Prove PeepholeFusionPass preserves measurement distributions on noisy circuits.

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
    def test_star_graph_noisy_10q(self, seed: int) -> None:
        circuit = generate_star_graph_honeypot(10, 100, seed=seed)
        base = ucc.compile(circuit)
        opt = ucc.compile(
            circuit,
            hir_passes=ucc.default_hir_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )

        base_result = ucc.sample(base, _STAT_SHOTS, seed=seed)
        opt_result = ucc.sample(opt, _STAT_SHOTS, seed=seed)
        self._assert_marginals_match(base_result.measurements, opt_result.measurements)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_commutation_gauntlet_noisy_20q(self, seed: int) -> None:
        circuit = generate_commutation_gauntlet(20, 200, seed=seed)
        base = ucc.compile(circuit)
        opt = ucc.compile(
            circuit,
            hir_passes=ucc.default_hir_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )

        base_result = ucc.sample(base, _STAT_SHOTS, seed=seed)
        opt_result = ucc.sample(opt, _STAT_SHOTS, seed=seed)
        self._assert_marginals_match(base_result.measurements, opt_result.measurements)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_uncomputation_ladder_noisy_10q(self, seed: int) -> None:
        circuit = generate_uncomputation_ladder(10, 100, seed=seed, noise_prob=0.02)
        base = ucc.compile(circuit)
        opt = ucc.compile(
            circuit,
            hir_passes=ucc.default_hir_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )

        base_result = ucc.sample(base, _STAT_SHOTS, seed=seed)
        opt_result = ucc.sample(opt, _STAT_SHOTS, seed=seed)
        self._assert_marginals_match(base_result.measurements, opt_result.measurements)
