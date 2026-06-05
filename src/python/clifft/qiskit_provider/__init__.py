"""Clifft Qiskit provider — run Qiskit circuits through the Clifft simulator.

Quick start::

    from clifft.qiskit_provider import ClifftProvider
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    backend = ClifftProvider().get_backend("clifft")
    job = backend.run(qc, shots=1000)
    print(job.result().get_counts())  # {'00': ~500, '11': ~500}
"""

from .backend import ClifftBackend
from .converter import (
    UnsupportedGateError,
    build_meas_map,
    circuit_to_stim,
    counts_from_measurements,
)
from .job import ClifftJob
from .provider import ClifftProvider

__all__ = [
    "ClifftBackend",
    "ClifftJob",
    "ClifftProvider",
    "UnsupportedGateError",
    "build_meas_map",
    "circuit_to_stim",
    "counts_from_measurements",
]
