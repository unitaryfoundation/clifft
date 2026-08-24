"""Developer-facing examples and boundary tests for the experimental HIP API."""

from __future__ import annotations

import pytest
from utils_hip import assert_forced_record_probabilities, assert_repeatable

import clifft
from clifft.experimental import hip


def test_hip_facade_explains_when_native_extension_is_absent() -> None:
    if hip.is_built():
        program = hip.compile("H 0\nT 0\nM 0")
        assert program.num_actions > 0
        assert "HIP executable" in program.inspect()
        return

    assert not hip.is_available()
    assert "CLIFFT_ENABLE_HIP=ON" in hip.backend_info()
    with pytest.raises(RuntimeError, match="CLIFFT_ENABLE_HIP"):
        hip.compile("M 0")


@pytest.mark.skipif(not hip.is_available(), reason="requires an AMD GPU visible to HIP")
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_hip_python_sampler_reuses_bounded_workspace(precision: hip.Precision) -> None:
    program = hip.compile("H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    sampler = hip.Sampler(program, precision=precision, max_batch_shots=7)

    assert sampler.precision == precision
    assert sampler.max_batch_shots == 7
    assert sampler.allocated_device_bytes > 0
    assert_repeatable(sampler, 257, 1234)


@pytest.mark.skipif(not hip.is_available(), reason="requires an AMD GPU visible to HIP")
@pytest.mark.parametrize(
    ("precision", "tolerance"),
    [("fp64", 1e-12), ("fp32", 2e-5)],
)
def test_hip_python_forced_replay_probes_each_branch(
    precision: hip.Precision,
    tolerance: float,
) -> None:
    circuit = "H 0\nT 0\nM 0\nEXP_VAL Z0\nOBSERVABLE_INCLUDE(0) rec[-1]"
    cpu_program = clifft.compile(circuit)
    sampler = hip.Sampler(hip.compile(circuit), precision=precision, max_batch_shots=1)

    assert_forced_record_probabilities(
        cpu_program,
        sampler,
        1,
        absolute_tolerance=tolerance,
    )
