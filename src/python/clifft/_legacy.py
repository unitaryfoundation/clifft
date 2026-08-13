"""Private legacy SVM hooks retained as a temporary test and performance oracle."""

from __future__ import annotations

from typing import Any

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
    "svm_backend",
]

Program = _LegacyProgram
State = _LegacyState
execute = _execute_legacy
get_statevector = _get_statevector_legacy


class _DefaultPasses:
    pass


_DEFAULT_PASSES = _DefaultPasses()


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
    measurements, detectors, observables, exp_vals = _sample_legacy(program, shots, seed)
    return SampleResult(measurements, detectors, observables, exp_vals=exp_vals)
