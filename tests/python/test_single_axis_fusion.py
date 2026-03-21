"""Tests for the SingleAxisFusionPass and OP_ARRAY_U2.

Verifies that fusing consecutive single-axis operations into precomputed
2x2 unitary sweeps produces identical statevectors and sampling results.
"""

import numpy as np
import pytest
from conftest import assert_statevectors_equal, random_dense_clifford_t_circuit

import ucc


def _compile_no_fusion(text: str) -> ucc.Program:
    """Compile with default passes but WITHOUT SingleAxisFusionPass."""
    circuit = ucc.parse(text)
    hir = ucc.trace(circuit)
    pm = ucc.default_hir_pass_manager()
    pm.run(hir)
    prog = ucc.lower(hir)
    bpm = ucc.BytecodePassManager()
    bpm.add(ucc.NoiseBlockPass())
    bpm.add(ucc.MultiGatePass())
    bpm.add(ucc.ExpandTPass())
    bpm.add(ucc.ExpandRotPass())
    bpm.add(ucc.SwapMeasPass())
    bpm.run(prog)
    return prog


def _compile_with_fusion(text: str) -> ucc.Program:
    """Compile with all default passes INCLUDING SingleAxisFusionPass."""
    circuit = ucc.parse(text)
    hir = ucc.trace(circuit)
    pm = ucc.default_hir_pass_manager()
    pm.run(hir)
    prog = ucc.lower(hir)
    bpm = ucc.default_bytecode_pass_manager()
    bpm.run(prog)
    return prog


def _get_sv(prog: ucc.Program) -> np.ndarray:
    """Execute one shot and extract the statevector."""
    state = ucc.State(prog.peak_rank, prog.num_measurements)
    ucc.execute(prog, state)
    return np.array(ucc.get_statevector(prog, state))


# ---------------------------------------------------------------------------
# U3 Collapse Test
# ---------------------------------------------------------------------------


def test_u3_single_gate_fusion() -> None:
    """A single U3 gate decomposes into Rz-H-Rz-H-Rz, all fusible into one U2."""
    text = "U3(0.1, 0.2, 0.3) 0"

    prog_no_fuse = _compile_no_fusion(text)
    prog_fused = _compile_with_fusion(text)

    assert len(prog_fused) < len(prog_no_fuse)

    u2_count = sum(1 for inst in prog_fused if inst.opcode == ucc.Opcode.OP_ARRAY_U2)
    assert u2_count >= 1, f"Expected at least 1 OP_ARRAY_U2, got {u2_count}"

    sv_ref = _get_sv(prog_no_fuse)
    sv_opt = _get_sv(prog_fused)
    assert_statevectors_equal(sv_opt, sv_ref, rtol=1e-10)


def test_u3_two_qubits_with_entanglement() -> None:
    """Two U3 gates with a CX in between stress multi-axis fusion boundaries."""
    text = "U3(0.5, 0.3, -0.2) 0\nU3(0.7, -0.1, 0.4) 1\nCX 0 1\nU3(0.2, 0.6, -0.8) 0"

    prog_no_fuse = _compile_no_fusion(text)
    prog_fused = _compile_with_fusion(text)

    assert len(prog_fused) < len(prog_no_fuse)

    sv_ref = _get_sv(prog_no_fuse)
    sv_opt = _get_sv(prog_fused)
    # The fused and unfused paths accumulate floating-point error differently
    # due to S-absorption changing the virtual coordinate decomposition.
    assert_statevectors_equal(sv_opt, sv_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# Frame Flow Test: force specific Pauli frame states before fusion
# ---------------------------------------------------------------------------


def test_frame_flow_x_error_before_fused_block() -> None:
    """X_ERROR(1.0) deterministically sets p_x, forcing the X branch of the FSM."""
    text = "H 0\nX_ERROR(1.0) 0\nT 0\nH 0\nS 0\nM 0"

    prog_no_fuse = _compile_no_fusion(text)
    prog_fused = _compile_with_fusion(text)

    meas_ref, _, _ = ucc.sample(prog_no_fuse, 1000, seed=42)
    meas_opt, _, _ = ucc.sample(prog_fused, 1000, seed=42)

    np.testing.assert_array_equal(meas_ref, meas_opt)


def test_frame_flow_z_error_before_fused_block() -> None:
    """Z_ERROR(1.0) deterministically sets p_z, forcing the Z branch of the FSM."""
    text = "H 0\nZ_ERROR(1.0) 0\nT 0\nH 0\nS 0\nM 0"

    prog_no_fuse = _compile_no_fusion(text)
    prog_fused = _compile_with_fusion(text)

    meas_ref, _, _ = ucc.sample(prog_no_fuse, 1000, seed=42)
    meas_opt, _, _ = ucc.sample(prog_fused, 1000, seed=42)

    np.testing.assert_array_equal(meas_ref, meas_opt)


# ---------------------------------------------------------------------------
# Randomized Dense Clifford+T Fuzzer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_qubits", [4, 5, 6])
@pytest.mark.parametrize("seed", [42, 99, 137])
def test_random_dense_clifford_t_fusion_equivalence(num_qubits: int, seed: int) -> None:
    """Fused and unfused statevectors match for random Clifford+T circuits."""
    text = random_dense_clifford_t_circuit(num_qubits, depth=40, seed=seed)

    prog_no_fuse = _compile_no_fusion(text)
    prog_fused = _compile_with_fusion(text)

    assert len(prog_fused) <= len(prog_no_fuse)

    sv_ref = _get_sv(prog_no_fuse)
    sv_opt = _get_sv(prog_fused)
    assert_statevectors_equal(sv_opt, sv_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# Instruction Count Validation
# ---------------------------------------------------------------------------


def test_fusion_reduces_instruction_count_for_qv_style_circuit() -> None:
    """Verify substantial instruction reduction on a QV-like circuit."""
    text = ""
    for q in range(4):
        text += f"U3(0.{q+1}, 0.{q+2}, 0.{q+3}) {q}\n"
    text += "CX 0 1\nCX 2 3\n"
    for q in range(4):
        text += f"U3(0.{q+5}, 0.{q+6}, 0.{q+7}) {q}\n"

    prog_no_fuse = _compile_no_fusion(text)
    prog_fused = _compile_with_fusion(text)

    ratio = len(prog_fused) / len(prog_no_fuse)
    assert ratio < 0.8, f"Expected >20% reduction, got {1-ratio:.1%}"

    u2_count = sum(1 for inst in prog_fused if inst.opcode == ucc.Opcode.OP_ARRAY_U2)
    assert u2_count > 0


# ---------------------------------------------------------------------------
# Noise circuit: fusion must preserve stochastic sampling behavior
# ---------------------------------------------------------------------------


def test_fusion_preserves_noisy_sampling() -> None:
    """Fusion around noise channels preserves measurement distribution."""
    text = "H 0\nT 0\nH 0\n" "DEPOLARIZE1(0.01) 0\n" "S 0\nH 0\nT 0\n" "M 0\n"

    prog_no_fuse = _compile_no_fusion(text)
    prog_fused = _compile_with_fusion(text)

    shots = 10_000
    meas_ref, _, _ = ucc.sample(prog_no_fuse, shots, seed=42)
    meas_opt, _, _ = ucc.sample(prog_fused, shots, seed=42)

    np.testing.assert_array_equal(meas_ref, meas_opt)
