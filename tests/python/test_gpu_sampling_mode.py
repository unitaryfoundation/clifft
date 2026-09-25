"""Routing and hardware witnesses for shared GPU sampling assertions."""

from unittest.mock import Mock

import numpy as np
import pytest
from conftest import pytest_sessionstart
from utils_conformance import GPU_SAMPLING_MODES, SMALL_CIRCUIT_SHOTS, GpuSamplingMode


@pytest.mark.parametrize("mode", GPU_SAMPLING_MODES, ids=lambda mode: mode.name)
def test_gpu_mode_routes_calls_through_one_retained_sampler(
    mode: GpuSamplingMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = Mock()
    import_api = Mock(return_value=api)
    monkeypatch.setattr("utils_conformance.import_module", import_api)
    program = mode.compile("M 0", postselection_mask=[0], hir_passes=None)
    import_api.assert_called_with(f"clifft.experimental.{mode.backend}")
    api.compile.assert_called_once_with("M 0", postselection_mask=[0], hir_passes=None)
    api.Sampler.assert_called_once_with(
        api.compile.return_value, precision="fp64", tier="auto", max_batch_shots=65
    )
    sampler = api.Sampler.return_value
    sampler.program = api.compile.return_value
    assert program.num_measurements is sampler.program.num_measurements

    assert mode.sample(program, 131, seed=17) is sampler.sample.return_value
    sampler.sample.assert_called_once_with(131, seed=17)
    assert (
        mode.sample_survivors(program, 67, seed=19, keep_records=True)
        is sampler.sample_survivors.return_value
    )
    sampler.sample_survivors.assert_called_once_with(67, seed=19, keep_records=True)
    api.Sampler.assert_called_once()

    # Backend failures must fail the test rather than skip or fall back to CPU.
    sampler.sample.side_effect = RuntimeError("device execution failed")
    with pytest.raises(RuntimeError, match="device execution failed"):
        mode.sample(program, 1)


@pytest.mark.parametrize("mode", GPU_SAMPLING_MODES, ids=lambda mode: mode.name)
def test_required_gpu_rejects_an_unavailable_device(
    mode: GpuSamplingMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = Mock()
    api.is_available.return_value = False
    api.backend_info.return_value = "device unavailable"
    monkeypatch.setattr("utils_conformance.import_module", lambda name: api)
    session = Mock()
    session.config.getoption.return_value = mode.backend
    with pytest.raises(
        pytest.UsageError, match=f"--require-gpu={mode.backend}: device unavailable"
    ):
        pytest_sessionstart(session)


@pytest.mark.parametrize("mode", GPU_SAMPLING_MODES, ids=lambda mode: mode.name)
@pytest.mark.parametrize("width", [1, 5])
def test_gpu_mode_executes_with_reported_precision_and_tier(
    mode: GpuSamplingMode, width: int
) -> None:
    mode.require_available()
    qubits = " ".join(str(q) for q in range(width))
    probe = "*".join(f"X{q}" for q in range(width))
    program = mode.compile(f"H {qubits}\nT {qubits}\nEXP_VAL {probe}\nM {qubits}")
    sampler = program.sampler
    assert program.peak_active_width == width
    assert isinstance(sampler, mode.api.Sampler)
    assert sampler.precision == "fp64"
    assert sampler.tier == ("thread_per_shot" if width <= 4 else "block_shared")
    assert sampler.max_batch_shots == 65
    assert sampler.allocated_device_bytes > 0
    print(
        f"{mode.api.backend_info()}; precision={sampler.precision}; "
        f"tier={sampler.tier}; active_width={width}; max_batch_shots={sampler.max_batch_shots}"
    )

    result = mode.sample(program, SMALL_CIRCUIT_SHOTS, seed=1911)
    survivors = mode.sample_survivors(program, SMALL_CIRCUIT_SHOTS, seed=1911, keep_records=True)
    assert result.measurements.shape == (SMALL_CIRCUIT_SHOTS, width)
    assert survivors.total_shots == survivors.passed_shots == SMALL_CIRCUIT_SHOTS
    np.testing.assert_array_equal(survivors.measurements, result.measurements)
    np.testing.assert_allclose(result.exp_vals, 2 ** (-width / 2), atol=1e-12, rtol=0)
    np.testing.assert_allclose(survivors.exp_vals, result.exp_vals, atol=1e-12, rtol=0)
