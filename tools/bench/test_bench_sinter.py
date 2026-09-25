"""Track aggregate postselection on the Clifford S-gate cultivation fixture."""

from pathlib import Path
from typing import Any

import numpy as np
import sinter
import stim

from clifft.sinter import PerfectionistSampler


def test_sample_clifford_postselection(benchmark: Any) -> None:
    circuit = stim.Circuit.from_file(
        Path(__file__).resolve().parents[2] / "docs/guide/circuits/circuit_d3_s_gate_p0.001.stim"
    )
    task = sinter.Task(
        circuit=circuit,
        postselection_mask=np.packbits(
            np.ones(circuit.num_detectors, dtype=np.uint8), bitorder="little"
        ),
    )
    sampler = PerfectionistSampler(batch_size=2048).compiled_sampler_for_task(task)
    sampler.sample(1024)
    benchmark(sampler.sample, 16_384)
