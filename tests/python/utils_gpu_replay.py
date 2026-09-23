"""CPU replay references shared by HIP and CUDA conformance tests."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

import clifft
from clifft._clifft_core import _replay_record

if TYPE_CHECKING:
    from clifft.experimental import cuda, hip


@dataclass(frozen=True)
class ReplayCase:
    name: str
    circuit: str
    visible: int
    hidden: int
    probabilities: tuple[float, ...]
    min_active_width: int = 0


# These probabilities are analytic, independent of either executor. The reset
# cases distinguish a visible marginal from one fully forced branch.
REPLAY_CASES = (
    ReplayCase(
        "uniform-reset",
        "H 0\nR 0\nH 0\nT 0\nM 0\nEXP_VAL Z0\nOBSERVABLE_INCLUDE(0) rec[-1]",
        1,
        1,
        (0.25, 0.25, 0.25, 0.25),
    ),
    ReplayCase("deterministic-reset", "X 0\nR 0\nH 0\nM 0", 1, 1, (0, 0.5, 0, 0.5)),
    ReplayCase(
        "feedback-reset",
        "H 0\nM 0\nCX rec[-1] 1\nR 1\nM 1\nDETECTOR rec[-1]",
        2,
        1,
        (0.5, 0, 0, 0, 0, 0.5, 0, 0),
    ),
    ReplayCase(
        "active-reset",
        "H 0\nT 0\nH 0\nCX 0 1\nR 0\nH 1\nT 1\nH 1\nM 1",
        1,
        1,
        (math.cos(math.pi / 8) ** 4, math.sin(math.pi / 8) ** 4, 0.125, 0.125),
        min_active_width=1,
    ),
    ReplayCase(
        "active-measurement",
        "H 0\nT 0\nH 0\nM 0",
        1,
        0,
        (math.cos(math.pi / 8) ** 2, math.sin(math.pi / 8) ** 2),
    ),
    ReplayCase("correlated-records", "H 0\nCX 0 1\nM 0 1", 2, 0, (0.5, 0, 0, 0.5)),
)

# Keep the multi-Pauli CUDA regression as well as the small analytic cases.
PAULI_REPLAY_CIRCUIT = "H 0\nH 1\nT 0\nT 1\nCX 0 1\nMPP Y0*Z1\nR_PAULI(0.17) X0*Y1\nM 0"


class ReplayBranch(NamedTuple):
    record: list[int]
    reachable: bool
    log_probability: float


def cpu_replay_branches(program: clifft.Program) -> list[ReplayBranch]:
    """Enumerate full records in visible-then-hidden order, including impossible ones."""
    width = program.num_measurements + program.num_hidden_measurements
    branches = []
    for bits in itertools.product((0, 1), repeat=width):
        record = list(bits)
        result = _replay_record(program, record)
        branches.append(
            ReplayBranch(record, bool(result["reachable"]), float(result["log_probability"]))
        )
    return branches


def assert_forced_record_probabilities(
    cpu_program: clifft.Program,
    gpu_sampler: cuda.Sampler | hip.Sampler,
    *,
    absolute_tolerance: float,
) -> None:
    """Compare complete forced branches and the visible portion of their outputs."""
    visible = cpu_program.num_measurements
    assert gpu_sampler.program.num_measurements == visible
    assert gpu_sampler.program.num_records == visible + cpu_program.num_hidden_measurements
    for expected in cpu_replay_branches(cpu_program):
        actual = gpu_sampler.replay_shot(expected.record)
        assert actual.reachable == expected.reachable
        if expected.reachable:
            assert actual.survived
            assert actual.log_probability == pytest.approx(
                expected.log_probability, abs=absolute_tolerance
            )
            np.testing.assert_array_equal(
                actual.outputs.measurements,
                np.asarray([expected.record[:visible]], dtype=np.uint8),
            )
