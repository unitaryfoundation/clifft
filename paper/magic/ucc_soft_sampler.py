"""Sinter-compatible sampler using UCC for Magic State Cultivation.

This module implements the sinter.Sampler interface, allowing UCC to be used
as a drop-in replacement for Stim's built-in sampler in Sinter's distributed
sampling pipeline. Post-selection is handled natively in the C++ VM via
OP_POSTSELECT instructions, which abort doomed shots at the earliest possible
instruction rather than sampling to completion and filtering afterward.

RNG Strategy:
    When seed=None (default), the C++ VM seeds its xoshiro256++ PRNG with
    256 bits of OS hardware entropy (std::random_device). The PRNG then
    streams forward naturally across all shots in the batch without
    reseeding. This guarantees no seed collisions across distributed
    workers at any scale -- each worker's 256-bit starting state is
    drawn independently from /dev/urandom.

Usage with sinter.collect()::

    from paper.magic.ucc_soft_sampler import UccSoftSampler
    results = sinter.collect(
        tasks=[task],
        custom_decoders={"ucc": UccSoftSampler()},
        ...
    )
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import sinter

import ucc

if TYPE_CHECKING:
    pass


class UccCompiledSampler(sinter.CompiledSampler):
    """A compiled sampler that executes a single task using UCC."""

    def __init__(self, task: sinter.Task) -> None:
        # Stim is a Clifford simulator and cannot represent non-Clifford
        # gates (T/T_DAG), so the raw circuit text is passed via
        # json_metadata["circuit_text"]. task.circuit is a sanitized copy
        # (T->I) used only for Sinter metadata like num_detectors.
        meta = task.json_metadata or {}
        stim_text: str = meta.get("circuit_text", str(task.circuit))

        num_det = task.circuit.num_detectors
        if task.postselection_mask is not None:
            mask = np.unpackbits(
                task.postselection_mask,
                count=num_det,
                bitorder="little",
            ).tolist()
        else:
            mask = [0] * num_det

        self._program = ucc.compile(
            stim_text,
            postselection_mask=mask,
            hir_passes=ucc.default_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        """Sample shots using hardware-entropy seeded PRNG."""
        t0 = time.monotonic()
        stats = ucc.sample_survivors(self._program, suggested_shots, keep_records=False)
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=stats["total_shots"],
            errors=stats["logical_errors"],
            discards=stats["discards"],
            seconds=t1 - t0,
        )


class UccSoftSampler(sinter.Sampler):
    """Sinter Sampler that uses UCC with native post-selection.

    Compiles the circuit once per task with OP_POSTSELECT instructions
    for flagged detectors, then samples using UCC's C++ VM.
    """

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        """Compile the task's circuit with UCC, wiring in postselection."""
        return UccCompiledSampler(task)
