"""Translate between Qiskit circuits/results and Clifft's Stim-text API."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError

# Qiskit gate name -> Stim gate name (Clifford+T basis).
_GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "S_DAG",
    "t": "T",
    "tdg": "T_DAG",
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
}

# Gates we transpile into (measure is handled separately, not a basis gate).
CLIFFT_BASIS = ["h", "s", "sdg", "x", "y", "z", "cx", "cy", "cz", "t", "tdg"]


def qiskit_to_stim(circuit: QuantumCircuit) -> tuple[str, list[int]]:
    """Convert a (basis-decomposed) QuantumCircuit to Stim text.

    Returns the Stim program and ``measured_clbits``: the classical-bit index
    targeted by each ``M`` line, in program order. clifft's
    ``measurements[:, j]`` is the j-th ``M``, so this list maps sample columns
    back to Qiskit clbits.

    Raises QiskitError on any operation outside the supported basis.
    """
    lines: list[str] = []
    measured_clbits: list[int] = []

    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]

        if name == "measure":
            clbit = circuit.find_bit(instruction.clbits[0]).index
            lines.append(f"M {qubits[0]}")
            measured_clbits.append(clbit)
        elif name in ("barrier", "id"):
            # no-ops for sampling
            continue
        elif name in _GATE_MAP:
            lines.append(f"{_GATE_MAP[name]} " + " ".join(str(q) for q in qubits))
        else:
            raise QiskitError(
                f"clifft backend does not support operation '{name}'. "
                "Supported basis: " + ", ".join(_GATE_MAP) + ", measure."
            )

    return "\n".join(lines), measured_clbits


def counts_from_measurements(
    measurements, measured_clbits: list[int], num_clbits: int
) -> dict[str, int]:
    """Convert clifft's (shots, num_meas) array to a Qiskit counts dict.

    Each shot's measured bits are packed into an integer where clbit ``i``
    contributes bit ``i`` (Qiskit's little-endian-by-clbit convention). The
    hex-string keys are rendered to binary by Qiskit's ``get_counts`` using
    ``memory_slots = num_clbits``, so ordering matches AerSimulator.
    """
    counts: dict[str, int] = {}
    for row in measurements:
        value = 0
        for col, clbit in enumerate(measured_clbits):
            if int(row[col]):
                value |= 1 << clbit
        key = hex(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
