"""Qiskit QuantumCircuit → Stim text converter.

Translates a Qiskit QuantumCircuit into the Stim circuit text format that
Clifft accepts.  Supports the Clifford+T basis plus continuous rotations.

Unsupported gates raise ``UnsupportedGateError`` with a clear message
suggesting the caller transpile first.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class UnsupportedGateError(Exception):
    """Raised when a gate cannot be expressed in Clifft's Stim format."""


# Map Qiskit gate names → Stim gate names (1-qubit, no args)
_CLIFFORD_1Q: dict[str, str] = {
    "h": "H",
    "s": "S",
    "sdg": "S_DAG",
    "t": "T",
    "tdg": "T_DAG",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "id": "I",
}

# 2-qubit Clifford gates: Qiskit name → Stim name
_CLIFFORD_2Q: dict[str, str] = {
    "cx": "CX",
    "cnot": "CX",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
}


def circuit_to_stim(qc: QuantumCircuit, *, clbit_order: str = "little") -> str:
    """Convert a Qiskit QuantumCircuit to a Stim circuit string.

    Args:
        qc: Qiskit QuantumCircuit.  Barriers are silently skipped.
            Unsupported gates raise ``UnsupportedGateError`` — transpile
            the circuit to the Clifford+T basis before calling this.
        clbit_order: ``"little"`` (default) maps classical bit *i* to
            Stim measurement record position *i*, matching Qiskit's
            little-endian ordering.  ``"big"`` reverses the bit order in
            the returned count strings.

    Returns:
        Stim-format circuit string, ready for ``clifft.compile()``.

    Raises:
        UnsupportedGateError: If an unrecognised gate is encountered.
    """
    if clbit_order not in ("little", "big"):
        raise ValueError("clbit_order must be 'little' or 'big'")

    lines: list[str] = []
    qubit_indices = {bit: i for i, bit in enumerate(qc.qubits)}
    clbit_indices = {bit: i for i, bit in enumerate(qc.clbits)}

    for instruction in qc.data:
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        name = op.name
        q = [qubit_indices[b] for b in qargs]

        # --- Barrier / global phase: skip silently ---
        if name in ("barrier", "delay", "global_phase"):
            continue

        # --- 1-qubit Clifford gates ---
        if name in _CLIFFORD_1Q:
            stim_name = _CLIFFORD_1Q[name]
            lines.append(f"{stim_name} {' '.join(map(str, q))}")
            continue

        # --- 2-qubit Clifford gates ---
        if name in _CLIFFORD_2Q:
            stim_name = _CLIFFORD_2Q[name]
            lines.append(f"{stim_name} {q[0]} {q[1]}")
            continue

        # --- Rotation gates (angles in radians → half-turns for Stim) ---
        if name == "rx":
            alpha = _rad_to_half_turns(float(op.params[0]))
            lines.append(f"R_X({alpha:.10g}) {q[0]}")
            continue
        if name == "ry":
            alpha = _rad_to_half_turns(float(op.params[0]))
            lines.append(f"R_Y({alpha:.10g}) {q[0]}")
            continue
        if name in ("rz", "p", "u1"):
            alpha = _rad_to_half_turns(float(op.params[0]))
            lines.append(f"R_Z({alpha:.10g}) {q[0]}")
            continue
        if name in ("u", "u3"):
            theta = _rad_to_half_turns(float(op.params[0]))
            phi = _rad_to_half_turns(float(op.params[1]))
            lam = _rad_to_half_turns(float(op.params[2]))
            lines.append(f"U3({theta:.10g}, {phi:.10g}, {lam:.10g}) {q[0]}")
            continue
        if name == "u2":
            # u2(phi, lam) = u3(π/2, phi, lam)
            theta = 0.5
            phi = _rad_to_half_turns(float(op.params[0]))
            lam = _rad_to_half_turns(float(op.params[1]))
            lines.append(f"U3({theta:.10g}, {phi:.10g}, {lam:.10g}) {q[0]}")
            continue
        if name == "rxx":
            alpha = _rad_to_half_turns(float(op.params[0]))
            lines.append(f"R_XX({alpha:.10g}) {q[0]} {q[1]}")
            continue
        if name == "ryy":
            alpha = _rad_to_half_turns(float(op.params[0]))
            lines.append(f"R_YY({alpha:.10g}) {q[0]} {q[1]}")
            continue
        if name == "rzz":
            alpha = _rad_to_half_turns(float(op.params[0]))
            lines.append(f"R_ZZ({alpha:.10g}) {q[0]} {q[1]}")
            continue

        # --- Measure & reset ---
        if name == "measure":
            lines.append(f"M {q[0]}")
            continue
        if name == "reset":
            lines.append(f"R {q[0]}")
            continue

        # --- Unitary (arbitrary matrix): not expressible in Stim directly ---
        if name == "unitary":
            raise UnsupportedGateError(
                f"Gate 'unitary' cannot be converted to Stim automatically. "
                f"Transpile to the Clifford+T basis first using "
                f"qiskit.compiler.transpile(circuit, basis_gates=['h','s','t','cx','measure','reset'])."
            )

        raise UnsupportedGateError(
            f"Gate '{name}' is not supported by the Clifft Qiskit provider. "
            f"Transpile to the Clifford+T basis first:\n"
            f"  from qiskit import transpile\n"
            f"  transpile(circuit, basis_gates=['h','s','sdg','t','tdg','cx','x','y','z','measure','reset'])"
        )

    return "\n".join(lines) if lines else "# empty circuit"


def counts_from_measurements(
    measurements: "import numpy; numpy.ndarray",  # type: ignore[type-arg]
    num_clbits: int,
    *,
    clbit_order: str = "little",
) -> dict[str, int]:
    """Convert a Clifft measurement array to a Qiskit-style counts dict.

    Args:
        measurements: uint8 array of shape ``(shots, num_measurements)``.
        num_clbits: Number of classical bits in the original circuit.
        clbit_order: ``"little"`` produces Qiskit's default little-endian
            bitstrings (rightmost bit = clbit 0).

    Returns:
        Dict mapping bitstring → count, e.g. ``{"00": 512, "11": 512}``.
    """
    counts: dict[str, int] = {}
    for row in measurements:
        bits = row[:num_clbits]
        if clbit_order == "little":
            bitstr = "".join(str(b) for b in reversed(bits))
        else:
            bitstr = "".join(str(b) for b in bits)
        counts[bitstr] = counts.get(bitstr, 0) + 1
    return counts


def _rad_to_half_turns(radians: float) -> float:
    """Convert radians to Stim's half-turn unit (alpha * π = radians)."""
    return radians / math.pi
