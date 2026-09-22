"""Smoke tests for Clifft alongside another OpenMP-using extension."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_COEXISTENCE_SCRIPT = r"""
import sys

import numpy as np


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
    result = simulator.run(circuit, shots=1).result()
    assert result.success
    np.testing.assert_allclose(result.get_statevector().data, np.full(4096, 1 / 64))


def run_clifft(threaded):
    import clifft

    program = clifft.compile("H 0 1\nT 0\nM 0 1")
    serial = clifft.sample(program, 31, seed=12347, threads=1)
    options = {"threads": 1}
    if threaded:
        options = {
            "thread_layout": (1, 2),
            "intra_shot_min_active_width": 0,
        }
    try:
        result = clifft.sample(
            program,
            31,
            seed=12347,
            **options,
        )
    except ValueError as exc:
        if "OpenMP-enabled build" in str(exc):
            raise SystemExit(77) from exc
        raise
    np.testing.assert_array_equal(result.measurements, serial.measurements)


threaded = sys.argv[2] == "threaded"
order = sys.argv[1]
if order.startswith("aer"):
    import qiskit_aer

    if order == "aer-run-first":
        run_aer()
    import clifft
else:
    import clifft

    if order == "clifft-run-first":
        run_clifft(threaded)
    import qiskit_aer

for _ in range(3):
    run_clifft(threaded)
    run_aer()
"""


@pytest.mark.parametrize(
    "order", ["aer-import-first", "aer-run-first", "clifft-import-first", "clifft-run-first"]
)
@pytest.mark.parametrize("threaded", [False, True])
def test_qiskit_aer_and_clifft_openmp_coexistence(order: str, threaded: bool) -> None:
    """Import order and prior execution must not corrupt either runtime's results."""
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
