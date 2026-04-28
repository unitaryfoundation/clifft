"""Differential A/B tests for bytecode optimization passes.

Compiles circuits, then runs BytecodePassManager on a copy, and asserts
the outputs are strictly identical. This proves the fused opcodes
(OP_NOISE_BLOCK, OP_EXPAND_T, OP_SWAP_MEAS_INTERFERE) preserve both
statevector correctness and PRNG trajectory synchronization.
"""

import numpy as np
import pytest
from conftest import random_dense_clifford_t_circuit

import clifft


def compile_unoptimized(stim_text: str) -> clifft.Program:
    """Compile with no HIR or bytecode optimization (baseline for A/B tests)."""
    return clifft.compile(stim_text, hir_passes=None, bytecode_passes=None)


def compile_optimized(stim_text: str) -> clifft.Program:
    """Compile with default bytecode passes only.

    HIR passes are skipped so the bytecode A/B tests share an identical
    HIR (and therefore identical PRNG trajectories) on both sides.
    """
    return clifft.compile(
        stim_text, hir_passes=None, bytecode_passes=clifft.default_bytecode_pass_manager()
    )


# ---------------------------------------------------------------------------
# Statevector Oracle: optimized vs unoptimized produce identical statevectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_qubits,depth,seed",
    [
        (3, 50, 1000),
        (3, 100, 1001),
        (4, 80, 1002),
        (4, 120, 1003),
        (5, 100, 1004),
        (5, 150, 1005),
        (5, 200, 1006),
        (6, 100, 1007),
        (6, 150, 1008),
        (6, 200, 1009),
    ],
)
def test_statevector_oracle(num_qubits: int, depth: int, seed: int) -> None:
    """Statevectors must be perfectly identical for optimized vs baseline."""
    stim_text = random_dense_clifford_t_circuit(num_qubits, depth, seed)

    prog_base = compile_unoptimized(stim_text)
    prog_opt = compile_optimized(stim_text)

    state_base = clifft.State(
        peak_rank=prog_base.peak_rank, num_measurements=prog_base.num_measurements
    )
    state_opt = clifft.State(
        peak_rank=prog_opt.peak_rank, num_measurements=prog_opt.num_measurements
    )

    clifft.execute(prog_base, state_base)
    clifft.execute(prog_opt, state_opt)

    sv_base = clifft.get_statevector(prog_base, state_base)
    sv_opt = clifft.get_statevector(prog_opt, state_opt)

    fidelity = float(np.abs(np.vdot(sv_base, sv_opt)) ** 2)
    assert (
        fidelity > 0.999999
    ), f"Fidelity {fidelity:.9f} for {num_qubits}q depth={depth} seed={seed}"


# ---------------------------------------------------------------------------
# Exact Trajectory: optimized noise sampling matches baseline bit-for-bit
# ---------------------------------------------------------------------------


NOISY_CIRCUIT = """\
H 0
CX 0 1
CX 1 2
DEPOLARIZE1(0.001) 0 1 2
H 0
T 0
H 1
T 1
DEPOLARIZE1(0.001) 0 1 2
CX 0 1
CX 1 2
DEPOLARIZE1(0.001) 0 1 2
H 0
H 1
H 2
M 0 1 2
DETECTOR rec[-1] rec[-2]
DETECTOR rec[-2] rec[-3]
OBSERVABLE_INCLUDE(0) rec[-3]
"""


def test_exact_trajectory_noisy() -> None:
    """Optimized and baseline must produce bit-identical sample arrays.

    This proves OP_NOISE_BLOCK keeps the gap-sampler perfectly synced
    with the original per-site OP_NOISE dispatch.
    """
    prog_base = compile_unoptimized(NOISY_CIRCUIT)
    prog_opt = compile_optimized(NOISY_CIRCUIT)

    # Verify optimization actually changed something
    assert (
        prog_opt.num_instructions < prog_base.num_instructions
    ), "Optimized program should have fewer instructions"

    base_result = clifft.sample(prog_base, shots=5000, seed=42)
    opt_result = clifft.sample(prog_opt, shots=5000, seed=42)

    np.testing.assert_array_equal(
        base_result.measurements, opt_result.measurements, err_msg="Measurement records diverged"
    )
    np.testing.assert_array_equal(
        base_result.detectors, opt_result.detectors, err_msg="Detector records diverged"
    )
    np.testing.assert_array_equal(
        base_result.observables, opt_result.observables, err_msg="Observable records diverged"
    )


# ---------------------------------------------------------------------------
# Exact Trajectory: noiseless circuit with EXPAND_T and SWAP_MEAS fusion
# ---------------------------------------------------------------------------


NOISELESS_WITH_MEAS = """\
H 0
CX 0 1
CX 0 2
T 0
T 1
H 2
CX 2 1
CX 2 0
M 0 1 2
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-3]
"""


def test_exact_trajectory_noiseless() -> None:
    """Noiseless circuit with T gates and measurements must match exactly.

    Tests OP_EXPAND_T and OP_SWAP_MEAS_INTERFERE fusion correctness.
    """
    prog_base = compile_unoptimized(NOISELESS_WITH_MEAS)
    prog_opt = compile_optimized(NOISELESS_WITH_MEAS)

    base_result = clifft.sample(prog_base, shots=5000, seed=99)
    opt_result = clifft.sample(prog_opt, shots=5000, seed=99)

    np.testing.assert_array_equal(
        base_result.measurements, opt_result.measurements, err_msg="Measurement records diverged"
    )
    np.testing.assert_array_equal(
        base_result.detectors, opt_result.detectors, err_msg="Detector records diverged"
    )
    np.testing.assert_array_equal(
        base_result.observables, opt_result.observables, err_msg="Observable records diverged"
    )


# ---------------------------------------------------------------------------
# Custom BytecodePassManager: user can compose their own pipeline
# ---------------------------------------------------------------------------


def test_custom_bytecode_pass_manager() -> None:
    """Users can build a custom BytecodePassManager and run individual passes."""
    prog = compile_unoptimized(NOISY_CIRCUIT)
    n_before = prog.num_instructions

    bpm = clifft.BytecodePassManager()
    bpm.add(clifft.NoiseBlockPass())
    bpm.run(prog)

    # NoiseBlockPass should reduce instruction count (coalesces noise sites)
    assert prog.num_instructions < n_before


def test_multi_gate_pass_opt_in() -> None:
    """MultiGatePass can be added explicitly to the pipeline."""
    stim_text = random_dense_clifford_t_circuit(4, 100, 2000)

    prog_base = compile_unoptimized(stim_text)
    prog_opt = compile_unoptimized(stim_text)

    bpm = clifft.BytecodePassManager()
    bpm.add(clifft.NoiseBlockPass())
    bpm.add(clifft.ExpandTPass())
    bpm.add(clifft.SwapMeasPass())
    bpm.add(clifft.MultiGatePass())
    bpm.run(prog_opt)

    # MultiGatePass should reduce further vs no optimization
    assert prog_opt.num_instructions < prog_base.num_instructions

    # Verify correctness
    state_base = clifft.State(
        peak_rank=prog_base.peak_rank, num_measurements=prog_base.num_measurements
    )
    state_opt = clifft.State(
        peak_rank=prog_opt.peak_rank, num_measurements=prog_opt.num_measurements
    )
    clifft.execute(prog_base, state_base)
    clifft.execute(prog_opt, state_opt)

    sv_base = clifft.get_statevector(prog_base, state_base)
    sv_opt = clifft.get_statevector(prog_opt, state_opt)

    fidelity = float(np.abs(np.vdot(sv_base, sv_opt)) ** 2)
    assert fidelity > 0.999999
