"""Clifft.

A multi-level AOT compiler and Schrodinger Virtual Machine for quantum circuits
(Clifford + T and beyond).
"""

# ruff: noqa: E402

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
    compile,
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
