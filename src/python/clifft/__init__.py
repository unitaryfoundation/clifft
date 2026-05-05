"""Clifft.

A fast exact simulator for near-Clifford quantum circuits. Accepts
Stim-format circuits with non-Clifford extensions, compiles them through
a multi-level pipeline (HIR + bytecode), and executes the bytecode on a
Schrodinger Virtual Machine whose cost scales with the active dimension
rather than the full Hilbert space.
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
    MakeUnitaryPass,
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
    _probabilities_from_indices,
    compute_reference_syndrome,
    default_bytecode_pass_manager,
    default_hir_pass_manager,
    execute,
    get_num_threads,
    get_statevector,
    lower,
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
from clifft._clifft_core import (
    compile as _compile_core,
)
from clifft._sample_result import SampleResult


def _basis_indices_from_bitstrings(
    program: Program,
    bitstrings: list[str] | tuple[str, ...] | str | object,
    bit_order: str,
) -> list[int]:
    if bit_order not in ("big", "little"):
        raise ValueError("bit_order must be 'big' or 'little'")

    num_qubits = program.num_qubits
    if num_qubits >= 64:
        raise ValueError("probabilities() currently supports programs with fewer than 64 qubits")

    def index_from_string(bitstring: str, row: int) -> int:
        if len(bitstring) != num_qubits:
            raise ValueError(
                f"bitstring at index {row} has length {len(bitstring)}, " f"expected {num_qubits}"
            )
        index = 0
        for col, char in enumerate(bitstring):
            if char == "1":
                qubit = col if bit_order == "big" else num_qubits - 1 - col
                index |= 1 << qubit
            elif char != "0":
                raise ValueError(
                    f"bitstring at index {row} contains {char!r}; expected only '0' and '1'"
                )
        return index

    if isinstance(bitstrings, str):
        return [index_from_string(bitstrings, 0)]

    if isinstance(bitstrings, (list, tuple)):
        if all(isinstance(bitstring, str) for bitstring in bitstrings):
            return [index_from_string(bitstring, row) for row, bitstring in enumerate(bitstrings)]
        raise TypeError("bitstrings must be strings or a 2D bool/uint8 NumPy array")

    import numpy as np

    if not isinstance(bitstrings, np.ndarray):
        raise TypeError("bitstrings must be strings or a 2D bool/uint8 NumPy array")
    if bitstrings.ndim != 2:
        raise ValueError("bitstrings array must be 2D with shape (num_bitstrings, num_qubits)")
    if bitstrings.shape[1] != num_qubits:
        raise ValueError(
            f"bitstrings array has {bitstrings.shape[1]} columns, expected {num_qubits}"
        )
    if bitstrings.dtype not in (np.dtype("bool"), np.dtype("uint8")):
        raise TypeError("bitstrings array dtype must be bool or uint8")
    if bitstrings.dtype == np.dtype("uint8") and np.any((bitstrings != 0) & (bitstrings != 1)):
        raise ValueError("uint8 bitstrings array must contain only 0 and 1")

    indices: list[int] = []
    for row_bits in bitstrings:
        index = 0
        for col, bit in enumerate(row_bits):
            if bool(bit):
                qubit = col if bit_order == "big" else num_qubits - 1 - col
                index |= 1 << qubit
        indices.append(index)
    return indices


def probabilities(
    program: Program,
    bitstrings: list[str] | tuple[str, ...] | str | object,
    *,
    bit_order: str = "big",
) -> object:
    """Return exact probabilities for full computational-basis bitstrings.

    ``bit_order="big"`` maps the first character or array column to qubit 0.
    ``bit_order="little"`` maps the last character or array column to qubit 0.
    """
    return _probabilities_from_indices(
        program, _basis_indices_from_bitstrings(program, bitstrings, bit_order)
    )


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
    "MakeUnitaryPass",
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
    "parse",
    "parse_file",
    "probabilities",
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
