"""Compile and sample a fixed-input Draper adder without runtime generation."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import clifft

_SOURCE = (Path(__file__).parent / "fixtures/draper_adder_16_basis.stim").read_text()
_EXPECTED = np.array(
    [(37449 >> bit) & 1 for bit in range(16)] + [(56173 >> bit) & 1 for bit in range(16)],
    dtype=np.uint8,
)


def _passes(configuration: str) -> clifft.HirPassManager:
    if configuration == "default":
        return clifft.default_hir_pass_manager()
    passes = clifft.HirPassManager()
    passes.add(clifft.PeepholeFusionPass())
    passes.add(clifft.PhasePolynomialPass())
    passes.add(clifft.StatevectorSqueezePass())
    return passes


@pytest.fixture(params=["default", "without-rotation-simplification"])
def configuration(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def program(configuration: str) -> clifft.Program:
    result = clifft.compile(_SOURCE, hir_passes=_passes(configuration))
    sample = clifft.sample(result, 1, seed=23, threads=1)
    np.testing.assert_array_equal(sample.measurements[0], _EXPECTED)
    np.testing.assert_array_equal(sample.observables, 0)
    return result


def test_compile_draper_adder(benchmark: Any, configuration: str) -> None:
    benchmark(lambda: clifft.compile(_SOURCE, hir_passes=_passes(configuration)))


@pytest.mark.parametrize("batch_size", [1, "auto"])
def test_sample_draper_adder(
    benchmark: Any, program: clifft.Program, batch_size: int | str
) -> None:
    benchmark(lambda: clifft.sample(program, 1024, seed=24, threads=1, batch_size=batch_size))
