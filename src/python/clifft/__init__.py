"""Clifft.

A fast exact simulator for near-Clifford quantum circuits. Accepts
Stim-format circuits with non-Clifford extensions, compiles them through
a multi-level pipeline (HIR + bytecode), and executes the bytecode on a
Schrodinger Virtual Machine whose cost scales with the active dimension
rather than the full Hilbert space.
"""

# ruff: noqa: E402
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

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
    DropNonUnitaryPass,
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
    _basis_probabilities_from_bitmasks,
    _record_probabilities_from_records,
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

BasisBitstrings: TypeAlias = str | Sequence[str] | npt.NDArray[np.bool_] | npt.NDArray[np.uint8]
MeasurementRecords: TypeAlias = str | Sequence[str] | npt.NDArray[np.bool_] | npt.NDArray[np.uint8]


def _basis_masks_from_bitstrings(
    program: Program,
    bitstrings: BasisBitstrings,
    bit_order: str,
) -> npt.NDArray[np.uint64]:
    if bit_order not in ("big", "little"):
        raise ValueError("bit_order must be 'big' or 'little'")

    num_qubits = program.num_qubits
    word_count = (num_qubits + 63) // 64

    def set_bit(masks: npt.NDArray[np.uint64], row: int, qubit: int) -> None:
        masks[row, qubit // 64] |= np.uint64(1 << (qubit % 64))

    def fill_string_mask(masks: npt.NDArray[np.uint64], bitstring: str, row: int) -> None:
        if len(bitstring) != num_qubits:
            raise ValueError(
                f"bitstring at index {row} has length {len(bitstring)}, " f"expected {num_qubits}"
            )
        for col, char in enumerate(bitstring):
            if char == "1":
                qubit = col if bit_order == "big" else num_qubits - 1 - col
                set_bit(masks, row, qubit)
            elif char != "0":
                raise ValueError(
                    f"bitstring at index {row} contains {char!r}; expected only '0' and '1'"
                )

    if isinstance(bitstrings, str):
        masks = np.zeros((1, word_count), dtype=np.uint64)
        fill_string_mask(masks, bitstrings, 0)
        return masks

    if isinstance(bitstrings, np.ndarray):
        bit_array = bitstrings
    elif isinstance(bitstrings, Sequence):
        if all(isinstance(bitstring, str) for bitstring in bitstrings):
            masks = np.zeros((len(bitstrings), word_count), dtype=np.uint64)
            for row, bitstring in enumerate(bitstrings):
                fill_string_mask(masks, bitstring, row)
            return masks
        raise TypeError("bitstrings must be strings or a 2D bool/uint8 NumPy array")
    else:
        raise TypeError("bitstrings must be strings or a 2D bool/uint8 NumPy array")

    if bit_array.ndim != 2:
        raise ValueError("bitstrings array must be 2D with shape (num_bitstrings, num_qubits)")
    if bit_array.shape[1] != num_qubits:
        raise ValueError(
            f"bitstrings array has {bit_array.shape[1]} columns, expected {num_qubits}"
        )
    if bit_array.dtype not in (np.dtype("bool"), np.dtype("uint8")):
        raise TypeError("bitstrings array dtype must be bool or uint8")
    if bit_array.dtype == np.dtype("uint8") and np.any((bit_array != 0) & (bit_array != 1)):
        raise ValueError("uint8 bitstrings array must contain only 0 and 1")

    qubit_order_bits = bit_array if bit_order == "big" else bit_array[:, ::-1]
    packed_bytes = np.packbits(qubit_order_bits, axis=1, bitorder="little")
    padded_bytes = word_count * 8
    if packed_bytes.shape[1] != padded_bytes:
        word_aligned = np.zeros((bit_array.shape[0], padded_bytes), dtype=np.uint8)
        word_aligned[:, : packed_bytes.shape[1]] = packed_bytes
        packed_bytes = word_aligned
    # Native little-endian uint64 layout matches Clifft's supported CPU targets
    # and the C++ mask representation.
    return np.ascontiguousarray(
        packed_bytes.view(np.uint64).reshape(bit_array.shape[0], word_count)
    )


def basis_probabilities(
    program: Program,
    bitstrings: BasisBitstrings,
    *,
    bit_order: str = "big",
    return_log: bool = False,
) -> npt.NDArray[np.float64]:
    """Exact Born probabilities of computational-basis bitstrings.

    Requires a unitary program (no measurements, feedback, noise, detectors,
    observables, or post-selection). For circuits with measurements, use
    :func:`record_probabilities` instead.

    ``bit_order="big"`` maps the first character or array column to qubit 0.
    ``bit_order="little"`` maps the last character or array column to qubit 0.

    Pass ``return_log=True`` to get natural-log probabilities. Zero
    probabilities map to ``-inf`` in log output.
    """
    probs = cast(
        npt.NDArray[np.float64],
        _basis_probabilities_from_bitmasks(
            program, _basis_masks_from_bitstrings(program, bitstrings, bit_order)
        ),
    )
    if return_log:
        # TODO: move logspace accumulation into the C++ amplitude walk so
        # very-rare-bitstring queries don't lose precision through the
        # linear->log conversion. Current path is symmetric with
        # record_probabilities() at the API surface but does not give the
        # precision benefit; that benefit only kicks in once the underlying
        # |amplitude|^2 sum is itself tracked in logspace.
        with np.errstate(divide="ignore"):
            return np.log(probs)
    return probs


def _records_from_outcomes(
    program: Program,
    records: MeasurementRecords,
) -> npt.NDArray[np.uint8]:
    """Convert string or array measurement records into a 2D uint8 array.

    The output has shape ``(num_records, program.num_measurements)`` and
    contains only 0/1 entries. Each row is a single measurement record in
    record-position order (matching ``sample().measurements``).
    """
    num_meas = program.num_measurements

    def fill_string_record(out: npt.NDArray[np.uint8], record: str, row: int) -> None:
        if len(record) != num_meas:
            raise ValueError(f"record at index {row} has length {len(record)}, expected {num_meas}")
        for col, char in enumerate(record):
            if char == "0":
                out[row, col] = 0
            elif char == "1":
                out[row, col] = 1
            else:
                raise ValueError(
                    f"record at index {row} contains {char!r}; expected only '0' and '1'"
                )

    if isinstance(records, str):
        out = np.zeros((1, num_meas), dtype=np.uint8)
        fill_string_record(out, records, 0)
        return out

    if isinstance(records, np.ndarray):
        bit_array = records
    elif isinstance(records, Sequence):
        if all(isinstance(record, str) for record in records):
            out = np.zeros((len(records), num_meas), dtype=np.uint8)
            for row, record in enumerate(records):
                fill_string_record(out, record, row)
            return out
        raise TypeError("records must be strings or a 2D bool/uint8 NumPy array")
    else:
        raise TypeError("records must be strings or a 2D bool/uint8 NumPy array")

    if bit_array.ndim != 2:
        raise ValueError("records array must be 2D with shape (num_records, num_measurements)")
    if bit_array.shape[1] != num_meas:
        raise ValueError(f"records array has {bit_array.shape[1]} columns, expected {num_meas}")
    if bit_array.dtype not in (np.dtype("bool"), np.dtype("uint8")):
        raise TypeError("records array dtype must be bool or uint8")
    if bit_array.dtype == np.dtype("uint8") and np.any((bit_array != 0) & (bit_array != 1)):
        raise ValueError("uint8 records array must contain only 0 and 1")
    return np.ascontiguousarray(bit_array.astype(np.uint8, copy=False))


def record_probabilities(
    program: Program,
    records: MeasurementRecords,
    *,
    return_log: bool = False,
) -> npt.NDArray[np.float64]:
    """Exact joint probabilities of measurement records under ``sample()``.

    Requires at least one measurement. For a purely unitary program with no
    measurements, use :func:`basis_probabilities` instead.

    ``records`` is one of: a single record string (e.g. ``"010"``), a
    sequence of record strings, or a 2D ``bool`` / ``uint8`` array of
    shape ``(num_records, program.num_measurements)``. Each record is
    interpreted in measurement order -- position ``i`` is the i-th entry
    sample().measurements would emit.

    Records the program cannot emit are reported as ``0.0`` (or ``-inf``
    when ``return_log=True``). For deep circuits whose probabilities
    underflow float64, pass ``return_log=True`` so the log-domain values
    survive.
    """
    # Reject zero-measurement programs up front so a user who passes a real
    # record string against a unitary program gets the right hint rather
    # than the wrapper's record-length mismatch error.
    if program.num_measurements == 0:
        raise ValueError(
            "record_probabilities() requires a program with at least one "
            "measurement; use clifft.basis_probabilities() for unitary circuits."
        )
    record_array = _records_from_outcomes(program, records)
    log_probs = cast(
        npt.NDArray[np.float64],
        _record_probabilities_from_records(program, record_array),
    )
    # C++ marks unreachable records with the finite sentinel
    # numpy.finfo(float64).min (-DBL_MAX). Translate it back to the
    # Pythonic -inf for log output, and rely on the natural underflow
    # to 0.0 under np.exp for the linear case.
    if return_log:
        return np.where(log_probs == np.finfo(np.float64).min, -np.inf, log_probs)
    return np.exp(log_probs)


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
    "BasisBitstrings",
    "MeasurementRecords",
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
    "DropNonUnitaryPass",
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
    "basis_probabilities",
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
    "record_probabilities",
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
