"""Smoke tests for Clifft alongside another OpenMP-using extension."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_COEXISTENCE_SCRIPT = r"""
import sys


def run_aer():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(12)
    circuit.h(range(12))
    circuit.save_statevector()
    simulator = AerSimulator(
        method="statevector",
        max_parallel_threads=2,
        statevector_parallel_threshold=1,
    )
    assert simulator.run(circuit, shots=1).result().success


def run_clifft(threaded):
    import clifft

    program = clifft.compile("H 0\nT 0\nM 0")
    options = {"threads": 1}
    if threaded:
        options = {
            "thread_layout": (1, 2),
            "intra_shot_min_active_width": 0,
        }
    try:
        result = clifft.sample(
            program,
            1,
            seed=1,
            **options,
        )
    except ValueError as exc:
        if "OpenMP-enabled build" in str(exc):
            raise SystemExit(77) from exc
        raise
    assert result.measurements.shape == (1, 1)


threaded = sys.argv[2] == "threaded"
if sys.argv[1] == "aer-first":
    run_aer()
    run_clifft(threaded)
else:
    run_clifft(threaded)
    run_aer()
"""


@pytest.mark.parametrize(
    ("order", "threaded"),
    [
        ("aer-first", False),
        pytest.param(
            "aer-first",
            True,
            marks=pytest.mark.xfail(
                condition=sys.platform == "darwin",
                reason="Aer-first intra-shot OpenMP can load conflicting runtimes on macOS",
                strict=False,
            ),
        ),
        ("clifft-first", True),
    ],
)
def test_qiskit_aer_and_clifft_openmp_coexistence(order: str, threaded: bool) -> None:
    """Serial use and the supported threaded order coexist with Qiskit Aer."""
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "2"
    mode = "threaded" if threaded else "serial"
    completed = subprocess.run(
        [sys.executable, "-c", _COEXISTENCE_SCRIPT, order, mode],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=30,
    )
    if completed.returncode == 77:
        pytest.skip("Clifft was built without OpenMP")
    assert completed.returncode == 0, completed.stdout + completed.stderr
