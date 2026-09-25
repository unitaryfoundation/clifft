"""Optional Sinter integration for all-detector error detection.

Install ``clifft[sinter]`` before importing this module.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

import clifft

try:
    import sinter
except ModuleNotFoundError as error:
    if error.name not in {"sinter", "stim"}:
        raise
    raise ImportError("Install 'clifft[sinter]' to use clifft.sinter") from error


def _shot_count(value: int, *, minimum: int, name: str) -> int:
    count = operator.index(value)
    if isinstance(value, bool) or not minimum <= count < 2**32:
        raise ValueError(f"{name} must be an integer between {minimum} and 2**32 - 1")
    return count


@dataclass(frozen=True, kw_only=True)
class PerfectionistSampler(sinter.Sampler):
    """Count logical errors after rejecting every shot with a detection event.

    An absent postselection mask selects every detector. An explicit mask must
    be a little-endian packed ``uint8`` NumPy array selecting every detector,
    with one byte per eight detectors, rounded up. Padding bits are ignored;
    with no detectors, an absent or empty mask is accepted.

    Detector and observable bits are normalized against the noiseless
    reference. On surviving shots, any observable flip counts as one error,
    using zero observable prediction. Observable postselection and arbitrary
    decoders are unsupported. Leave Sinter's ``count_detection_events`` and
    ``count_observable_error_combos`` options false; Sinter 1.16 rejects these
    options for custom samplers.

    Tasks must contain a resolved ``stim.Circuit`` using instructions supported
    by Clifft. Heralded noise, sweep-bit controls, and Pauli targets in
    ``OBSERVABLE_INCLUDE`` are unsupported. Stim tags are ignored when compiling
    without modifying the task's circuit. A supplied detector error model
    participates in Sinter's task identity but is not used to predict corrections.

    Register an instance in ``sinter.collect(custom_decoders=...)``. Its
    configuration is pickle-safe; each worker compiles its own Clifft program
    and uses one native thread. Sampling returns aggregate counts without
    materializing survivor rows. Each call uses fresh native randomness;
    this adapter does not expose a seed or a retained random stream. Direct
    ``sample(0)`` calls return zero counts.

    Args:
        batch_size: Native lane capacity, defaulting to 1024. Each call uses at
            most the requested number of shots and 2048 lanes; larger capacities
            are capped, not rejected. This is separate from Sinter's
            ``max_batch_size``, which defaults to 1024 in Sinter 1.16. Increase
            that limit to use larger native batches. Use 1 for scalar execution
            or ``"auto"`` for Clifft's core policy, which currently selects
            scalar execution for postselection.
    """

    batch_size: int | Literal["auto"] = 1024

    def __post_init__(self) -> None:
        if self.batch_size != "auto":
            _shot_count(self.batch_size, minimum=1, name="batch_size")

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        """Validate an all-detector task and compile its normalized circuit."""
        if task.circuit is None:
            raise ValueError("PerfectionistSampler requires a resolved task.circuit")
        if task.postselected_observables_mask is not None:
            raise ValueError("PerfectionistSampler does not support observable postselection")

        num_detectors = task.circuit.num_detectors
        mask = task.postselection_mask
        if mask is not None:
            if (
                not isinstance(mask, np.ndarray)
                or mask.dtype != np.uint8
                or mask.shape != ((num_detectors + 7) // 8,)
            ):
                raise ValueError(
                    "postselection_mask must be a packed uint8 array for the detectors"
                )
            # Padding bits do not refer to detectors and must not affect acceptance.
            if not np.all(np.unpackbits(mask, bitorder="little", count=num_detectors)):
                raise ValueError("PerfectionistSampler requires all-detector postselection")

        program = clifft.compile(
            str(task.circuit.without_tags()),
            postselection_mask=[1] * num_detectors,
            normalize_syndromes=True,
        )
        return _CompiledPerfectionistSampler(program, self.batch_size)


class _CompiledPerfectionistSampler(sinter.CompiledSampler):
    def __init__(self, program: clifft.Program, batch_size: int | Literal["auto"]) -> None:
        self._program = program
        self._batch_size = batch_size

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        shots = _shot_count(suggested_shots, minimum=0, name="suggested_shots")
        if shots == 0:
            return sinter.AnonTaskStats()
        start = time.monotonic()
        result = clifft.sample_survivors(
            self._program,
            shots,
            keep_records=False,
            threads=1,
            batch_size=self._batch_size,
        )
        return sinter.AnonTaskStats(
            shots=result.total_shots,
            discards=result.discards,
            errors=result.logical_errors,
            seconds=time.monotonic() - start,
        )
