"""Private legacy SVM hooks retained as a temporary test and performance oracle."""

from __future__ import annotations

from typing import Any

from clifft import _DEFAULT_PASSES, _DefaultPasses
from clifft._clifft_core import (
    BytecodePass,
    BytecodePassManager,
    ExpandRotPass,
    ExpandTPass,
    HirModule,
    HirPassManager,
    Instruction,
    MultiGatePass,
    NoiseBlockPass,
    Opcode,
    SingleAxisFusionPass,
    SwapMeasPass,
    TileAxisFusionPass,
    _compile_legacy,
    _execute_legacy,
    _get_statevector_legacy,
    _LegacyProgram,
    _LegacyState,
    _lower_legacy,
    _sample_legacy,
    _sample_noncomputational_legacy,
    default_bytecode_pass_manager,
    default_hir_pass_manager,
    svm_backend,
)
from clifft._sample_result import SampleResult

__all__ = [
    "BytecodePass",
    "BytecodePassManager",
    "ExpandRotPass",
    "ExpandTPass",
    "Instruction",
    "MultiGatePass",
    "NoiseBlockPass",
    "Opcode",
    "Program",
    "SingleAxisFusionPass",
    "State",
    "SwapMeasPass",
    "TileAxisFusionPass",
    "compile",
    "default_bytecode_pass_manager",
    "execute",
    "get_statevector",
    "lower",
    "sample",
    "sample_noncomputational",
    "statevector",
    "svm_backend",
]

Program = _LegacyProgram
State = _LegacyState
execute = _execute_legacy
get_statevector = _get_statevector_legacy


def compile(
    stim_text: str,
    postselection_mask: list[int] | None = None,
    expected_detectors: list[int] | None = None,
    expected_observables: list[int] | None = None,
    normalize_syndromes: bool = False,
    hir_passes: HirPassManager | None | _DefaultPasses = _DEFAULT_PASSES,
    bytecode_passes: BytecodePassManager | None | _DefaultPasses = _DEFAULT_PASSES,
) -> Any:
    if isinstance(hir_passes, _DefaultPasses):
        hir_passes = default_hir_pass_manager()
    if isinstance(bytecode_passes, _DefaultPasses):
        bytecode_passes = default_bytecode_pass_manager()
    return _compile_legacy(
        stim_text,
        postselection_mask if postselection_mask is not None else [],
        expected_detectors if expected_detectors is not None else [],
        expected_observables if expected_observables is not None else [],
        normalize_syndromes,
        hir_passes,
        bytecode_passes,
    )


def lower(
    hir: HirModule,
    postselection_mask: list[int] | None = None,
    expected_detectors: list[int] | None = None,
    expected_observables: list[int] | None = None,
) -> Any:
    return _lower_legacy(
        hir,
        postselection_mask if postselection_mask is not None else [],
        expected_detectors if expected_detectors is not None else [],
        expected_observables if expected_observables is not None else [],
    )


def sample(program: Any, shots: int, seed: int | None = None) -> SampleResult:
    if program.has_postselection:
        raise ValueError(
            "sample() cannot be used with post-selected programs because it "
            "returns a fixed number of rows and cannot discard shots."
        )
    measurements, detectors, observables, exp_vals = _sample_legacy(program, shots, seed)
    return SampleResult(measurements, detectors, observables, exp_vals=exp_vals)


def statevector(stim_text: str, **compile_kwargs: Any) -> Any:
    """Compile and expand a pure-state circuit through the legacy oracle."""
    program = compile(stim_text, **compile_kwargs)
    state = State(peak_rank=program.peak_rank, num_measurements=program.num_measurements)
    execute(program, state)
    return get_statevector(program, state)


def sample_noncomputational(
    circuit: Any,
    model: Any,
    shots: int,
    seed: int | None = None,
    max_rank: int | None = None,
) -> Any:
    """Sample leakage and loss through the private legacy trajectory oracle."""
    from clifft.noncomp import _sample_with

    return _sample_with(
        circuit,
        model,
        shots,
        seed,
        max_rank,
        _sample_noncomputational_legacy,
    )
