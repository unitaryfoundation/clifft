"""Tests for clifft.compile() with optional hir_passes and bytecode_passes.

Verifies that the pass manager arguments are wired correctly through
the compile convenience function, producing the same results as the
explicit step-by-step pipeline.
"""

import numpy as np
import pytest
from conftest import random_dense_clifford_t_circuit

import clifft

# ---------------------------------------------------------------------------
# Default behavior: omitting pass managers runs the default optimizers
# ---------------------------------------------------------------------------


def test_compile_default_runs_default_optimizers() -> None:
    """compile(text) with no kwargs equals the full default-optimized pipeline."""
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"

    prog_convenience = clifft.compile(text)

    circuit = clifft.parse(text)
    hir = clifft.trace(circuit)
    clifft.default_hir_pass_manager().run(hir)
    prog_manual = clifft.lower(hir)
    clifft.default_bytecode_pass_manager().run(prog_manual)

    assert prog_convenience.num_instructions == prog_manual.num_instructions
    assert prog_convenience.peak_rank == prog_manual.peak_rank


def test_compile_explicit_none_skips_optimization() -> None:
    """Passing None explicitly disables the corresponding optimization stage."""
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"

    prog_off = clifft.compile(text, hir_passes=None, bytecode_passes=None)

    circuit = clifft.parse(text)
    hir = clifft.trace(circuit)
    prog_manual = clifft.lower(hir)

    assert prog_off.num_instructions == prog_manual.num_instructions
    assert prog_off.peak_rank == prog_manual.peak_rank


# ---------------------------------------------------------------------------
# Full pipeline: compile() with both pass managers matches manual pipeline
# ---------------------------------------------------------------------------


def test_compile_both_passes_matches_manual() -> None:
    """compile() with both pass managers matches the full manual pipeline."""
    text = "H 0\nT 0\nCNOT 0 1\nT 1\nM 0 1"

    prog = clifft.compile(
        text,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )

    # Manual pipeline
    circuit = clifft.parse(text)
    hir = clifft.trace(circuit)
    pm = clifft.default_hir_pass_manager()
    pm.run(hir)
    prog_manual = clifft.lower(hir)
    bpm = clifft.default_bytecode_pass_manager()
    bpm.run(prog_manual)

    assert prog.num_instructions == prog_manual.num_instructions
    assert prog.peak_rank == prog_manual.peak_rank


# ---------------------------------------------------------------------------
# HIR-only optimization
# ---------------------------------------------------------------------------


def test_compile_hir_only_matches_manual() -> None:
    """compile() with only hir_passes matches manual trace + optimize + lower."""
    text = "H 0\nT 0\nT_DAG 0\nM 0"  # T/T_DAG cancel

    prog = clifft.compile(text, hir_passes=clifft.default_hir_pass_manager(), bytecode_passes=None)

    circuit = clifft.parse(text)
    hir = clifft.trace(circuit)
    pm = clifft.default_hir_pass_manager()
    pm.run(hir)
    prog_manual = clifft.lower(hir)

    assert prog.num_instructions == prog_manual.num_instructions
    assert prog.peak_rank == prog_manual.peak_rank


def test_hir_passes_reduce_t_cancellation() -> None:
    """Peephole fusion cancels T/T_DAG, reducing peak rank."""
    text = "H 0\nT 0\nT_DAG 0\nM 0"

    prog_no_opt = clifft.compile(text, hir_passes=None, bytecode_passes=None)
    prog_opt = clifft.compile(text, hir_passes=clifft.default_hir_pass_manager())

    # Without optimization, T and T_DAG both expand the array.
    # With peephole, they cancel and peak_rank should be lower.
    assert prog_opt.peak_rank <= prog_no_opt.peak_rank


# ---------------------------------------------------------------------------
# Bytecode-only optimization
# ---------------------------------------------------------------------------


def test_compile_bytecode_only_matches_manual() -> None:
    """compile() with only bytecode_passes matches manual lower + optimize."""
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"

    prog = clifft.compile(
        text, hir_passes=None, bytecode_passes=clifft.default_bytecode_pass_manager()
    )

    circuit = clifft.parse(text)
    hir = clifft.trace(circuit)
    prog_manual = clifft.lower(hir)
    bpm = clifft.default_bytecode_pass_manager()
    bpm.run(prog_manual)

    assert prog.num_instructions == prog_manual.num_instructions
    assert prog.peak_rank == prog_manual.peak_rank


def test_bytecode_passes_fuse_instructions() -> None:
    """Bytecode passes should fuse ops, reducing instruction count."""
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"

    prog_no_opt = clifft.compile(text, hir_passes=None, bytecode_passes=None)
    prog_opt = clifft.compile(
        text, hir_passes=None, bytecode_passes=clifft.default_bytecode_pass_manager()
    )

    # Fused instructions should not increase instruction count
    assert prog_opt.num_instructions <= prog_no_opt.num_instructions


# ---------------------------------------------------------------------------
# Postselection combined with pass managers
# ---------------------------------------------------------------------------


def test_compile_postselection_with_passes() -> None:
    """Postselection mask works together with pass managers."""
    text = "H 0\nM 0\nDETECTOR rec[-1]"

    prog = clifft.compile(
        text,
        postselection_mask=[1],
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )

    assert prog.num_detectors == 1
    # Should have an OP_POSTSELECT in the bytecode
    opcodes = [prog[i].opcode for i in range(prog.num_instructions)]
    assert clifft.Opcode.OP_POSTSELECT in opcodes


def test_postselection_without_passes_unchanged() -> None:
    """Adding passes does not break postselection semantics."""
    text = "H 0\nM 0\nDETECTOR rec[-1]"
    mask = [1]

    prog_base = clifft.compile(text, postselection_mask=mask, hir_passes=None, bytecode_passes=None)
    prog_opt = clifft.compile(text, postselection_mask=mask)

    assert prog_base.num_detectors == prog_opt.num_detectors
    assert prog_base.num_measurements == prog_opt.num_measurements


# ---------------------------------------------------------------------------
# Statevector correctness: optimized compile matches unoptimized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_qubits,depth,seed",
    [
        (3, 50, 5000),
        (4, 80, 5001),
        (5, 100, 5002),
        (5, 150, 5003),
        (6, 100, 5004),
    ],
)
def test_statevector_identical_with_passes(num_qubits: int, depth: int, seed: int) -> None:
    """Statevectors from optimized compile must match unoptimized exactly.

    Uses measurement-free circuits so the statevector is deterministic
    and we can compare fidelity directly.
    """
    stim_text = random_dense_clifford_t_circuit(num_qubits, depth, seed)

    prog_base = clifft.compile(stim_text, hir_passes=None, bytecode_passes=None)
    prog_opt = clifft.compile(stim_text)

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
# Custom pass manager: user can pick specific passes
# ---------------------------------------------------------------------------


def test_custom_pass_manager_via_compile() -> None:
    """A manually-built HirPassManager works when passed to compile()."""
    text = "H 0\nT 0\nT_DAG 0\nM 0"

    pm = clifft.HirPassManager()
    pm.add(clifft.PeepholeFusionPass())

    prog = clifft.compile(text, hir_passes=pm)
    assert prog.peak_rank == 0  # T/T_DAG cancelled, no active dims


def test_custom_bytecode_pass_manager_via_compile() -> None:
    """A manually-built BytecodePassManager works when passed to compile()."""
    text = "H 0\nT 0\nCNOT 0 1\nM 0 1"

    bpm = clifft.BytecodePassManager()
    bpm.add(clifft.ExpandTPass())

    prog = clifft.compile(text, hir_passes=None, bytecode_passes=bpm)
    # Should have fused expand+T into OP_EXPAND_T
    opcodes = [prog[i].opcode for i in range(prog.num_instructions)]
    assert clifft.Opcode.OP_EXPAND_T in opcodes or clifft.Opcode.OP_EXPAND_T_DAG in opcodes
