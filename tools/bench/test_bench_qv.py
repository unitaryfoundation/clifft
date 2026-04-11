"""Performance benchmarks for large-statevector circuits.

Uses a pre-generated 20-qubit Quantum Volume circuit (peak rank 20,
16 MB statevector) to measure per-shot throughput when the SVM array
operations dominate execution time.

Run with: just bench
"""

from pathlib import Path
from typing import Any

import pytest

import clifft

_CIRCUIT_PATH = Path(__file__).parent / "fixtures" / "qv20_seed42.stim"
_CIRCUIT_TEXT = _CIRCUIT_PATH.read_text()


@pytest.fixture(scope="module")
def clifft_program() -> clifft.Program:
    """Pre-compiled Clifft program for QV-20."""
    return clifft.compile(_CIRCUIT_TEXT)


def test_compile_qv20(benchmark: Any) -> None:
    """Measure Clifft compilation time for QV-20."""

    def compile_clifft() -> clifft.Program:
        return clifft.compile(_CIRCUIT_TEXT)

    benchmark(compile_clifft)


def test_sample_qv20_1shot(benchmark: Any, clifft_program: clifft.Program) -> None:
    """Measure Clifft single-shot execution time for QV-20.

    Single-shot timing isolates per-shot SVM array cost without
    amortizing compilation or state allocation across many shots.
    """

    def sample_one() -> object:
        return clifft.sample(clifft_program, 1, seed=0)

    benchmark(sample_one)


def test_sample_qv20_10shots(benchmark: Any, clifft_program: clifft.Program) -> None:
    """Measure Clifft 10-shot execution time for QV-20."""

    def sample_ten() -> object:
        return clifft.sample(clifft_program, 10, seed=0)

    benchmark(sample_ten)
