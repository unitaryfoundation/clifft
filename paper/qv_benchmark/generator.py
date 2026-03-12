"""Quantum Volume circuit generator.

Generates QV circuits via Qiskit, transpiles to cx+u3 basis, and exports
QASM 2.0 strings for downstream simulator benchmarking.
"""

from __future__ import annotations

import qiskit.qasm2
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import quantum_volume
from qiskit.compiler import transpile

_BASIS_GATES: list[str] = ["cx", "u3"]


def _build_transpiled_qv(num_qubits: int, seed: int = 42) -> QuantumCircuit:
    """Build a transpiled Quantum Volume circuit.

    Parameters
    ----------
    num_qubits:
        Number of qubits (also sets the QV depth).
    seed:
        RNG seed for deterministic circuit generation.

    Returns
    -------
    QuantumCircuit
        Transpiled circuit in the cx+u3 basis.
    """
    qc: QuantumCircuit = quantum_volume(num_qubits, seed=seed)
    qc_transpiled: QuantumCircuit = transpile(
        qc,
        basis_gates=_BASIS_GATES,
        optimization_level=0,
    )
    return qc_transpiled


def generate_qv_qasm(num_qubits: int, seed: int = 42) -> str:
    """Generate a measured Quantum Volume circuit as a QASM 2.0 string.

    The circuit includes ``measure_all()`` so that every qubit is measured
    into a classical register.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    str
        QASM 2.0 source with measurement instructions.
    """
    qc: QuantumCircuit = _build_transpiled_qv(num_qubits, seed=seed)
    qc.measure_all()
    return str(qiskit.qasm2.dumps(qc))


def generate_qv_qasm_unmeasured(num_qubits: int, seed: int = 42) -> str:
    """Generate an unmeasured Quantum Volume circuit as a QASM 2.0 string.

    The returned circuit has no measurement gates, making it suitable for
    statevector simulation and ideal-probability extraction.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    str
        QASM 2.0 source without measurement instructions.
    """
    qc: QuantumCircuit = _build_transpiled_qv(num_qubits, seed=seed)
    return str(qiskit.qasm2.dumps(qc))
