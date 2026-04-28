"""Clifft.

A multi-level AOT compiler and Schrodinger Virtual Machine for quantum circuits
(Clifford + T and beyond).
"""

# ruff: noqa: E402
from __future__ import annotations

from clifft._build_config import CPU_BASELINE, REQUIRES_X86_64_V3_BASELINE
from clifft._cpu_check import ensure_supported_cpu

ensure_supported_cpu(CPU_BASELINE, REQUIRES_X86_64_V3_BASELINE)


# Warn when imported inside a multiprocessing worker (e.g. sinter) with
# multiple OpenMP threads.  Each worker spawning its own thread pool causes
# severe oversubscription on multi-core machines.
def _check_multiprocessing_omp() -> None:
    import multiprocessing

    # "MainProcess" is hardcoded in CPython's multiprocessing.process._MainProcess
    # (all platforms). Worker processes get names like "Process-1", "SpawnProcess-1", etc.
    if multiprocessing.current_process().name == "MainProcess":
        return
    # Inside a worker — check if OMP threads > 1.  Import the extension
    # here to avoid circular imports; this runs after ensure_supported_cpu.
    try:
        from clifft._clifft_core import get_num_threads
    except ImportError:
        return
    if get_num_threads() > 1:
        import warnings

        warnings.warn(
            "clifft is running inside a multiprocessing worker with "
            f"{get_num_threads()} OpenMP threads. This causes "
            "oversubscription — each worker spawns its own thread pool. "
            "Set OMP_NUM_THREADS=1 or call clifft.set_num_threads(1) "
            "in your worker initializer.",
            stacklevel=2,
        )


_check_multiprocessing_omp()
del _check_multiprocessing_omp

from clifft._clifft_core import (
    AstNode,
    BytecodePass,
    BytecodePassManager,
    Circuit,
    ExpandRotPass,
    ExpandTPass,
    GateType,
    HeisenbergOp,
    HirModule,
    HirPass,
    HirPassManager,
    Instruction,
    MultiGatePass,
    NoiseBlockPass,
    Opcode,
    OpType,
    ParseError,
    PeepholeFusionPass,
    Program,
    RemoveNoisePass,
    SingleAxisFusionPass,
    State,
    StatevectorSqueezePass,
    SwapMeasPass,
    Target,
    compile as _compile_core,
    compute_reference_syndrome,
    default_bytecode_pass_manager,
    default_hir_pass_manager,
    execute,
    get_num_threads,
    get_statevector,
    lower,
    max_sim_qubits,
    parse,
    parse_file,
    sample,
    sample_k,
    sample_k_survivors,
    sample_survivors,
    set_num_threads,
    svm_backend,
    trace,
    version,
)
from clifft._sample_result import SampleResult


class _DefaultPasses:
    """Sentinel marker for compile()'s default optimization passes."""


_DEFAULT_PASSES = _DefaultPasses()


def compile(
    stim_text: str,
    postselection_mask: list[int] | None = None,
    expected_detectors: list[int] | None = None,
    expected_observables: list[int] | None = None,
    normalize_syndromes: bool = False,
    hir_passes: HirPassManager | None | _DefaultPasses = _DEFAULT_PASSES,
    bytecode_passes: BytecodePassManager | None | _DefaultPasses = _DEFAULT_PASSES,
) -> Program:
    """Compile a quantum circuit string to executable bytecode.

    Runs the full pipeline: parse -> trace -> [HIR optimize] ->
    lower -> [bytecode optimize].

    By default both optimization stages run with their default pass
    managers. To skip optimization, pass ``hir_passes=None`` and/or
    ``bytecode_passes=None``. To use a custom pipeline, pass an
    explicit ``HirPassManager`` / ``BytecodePassManager``.

    When ``normalize_syndromes=True``, a noiseless reference shot is
    executed internally to extract expected detector and observable
    parities. Detectors and observables are then XOR-normalized so
    that 0 means 'matches noiseless reference' and 1 means 'error'.

    Args:
        stim_text: Circuit in .stim text format.
        postselection_mask: Optional list of uint8 flags, one per detector.
            Detectors where mask[i] != 0 become post-selection checks
            that abort the shot early if their parity is non-zero.
        expected_detectors: Optional noiseless reference parities for detectors.
        expected_observables: Optional noiseless reference parities for observables.
        normalize_syndromes: If True, auto-compute reference parities from a
            noiseless reference shot (mutually exclusive with explicit parities).
        hir_passes: HirPassManager to run on the HIR before lowering.
            Defaults to ``default_hir_pass_manager()``. Pass ``None`` to skip.
        bytecode_passes: BytecodePassManager to run after lowering.
            Defaults to ``default_bytecode_pass_manager()``. Pass ``None`` to skip.
    """
    if isinstance(hir_passes, _DefaultPasses):
        hir_passes = default_hir_pass_manager()
    if isinstance(bytecode_passes, _DefaultPasses):
        bytecode_passes = default_bytecode_pass_manager()
    return _compile_core(
        stim_text,
        postselection_mask if postselection_mask is not None else [],
        expected_detectors if expected_detectors is not None else [],
        expected_observables if expected_observables is not None else [],
        normalize_syndromes,
        hir_passes,
        bytecode_passes,
    )

__all__ = [
    "AstNode",
    "BytecodePass",
    "BytecodePassManager",
    "Circuit",
    "ExpandRotPass",
    "ExpandTPass",
    "GateType",
    "HeisenbergOp",
    "HirModule",
    "HirPass",
    "HirPassManager",
    "Instruction",
    "MultiGatePass",
    "NoiseBlockPass",
    "Opcode",
    "OpType",
    "ParseError",
    "PeepholeFusionPass",
    "Program",
    "RemoveNoisePass",
    "SampleResult",
    "SingleAxisFusionPass",
    "State",
    "StatevectorSqueezePass",
    "SwapMeasPass",
    "Target",
    "compile",
    "compute_reference_syndrome",
    "default_bytecode_pass_manager",
    "default_hir_pass_manager",
    "execute",
    "get_num_threads",
    "get_statevector",
    "lower",
    "max_sim_qubits",
    "parse",
    "parse_file",
    "sample",
    "sample_k",
    "sample_k_survivors",
    "sample_survivors",
    "set_num_threads",
    "svm_backend",
    "trace",
    "version",
]

__version__ = version()
