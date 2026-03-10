"""Stim-text to Qiskit QuantumCircuit translator for validation.

Translates noiseless Clifford+T circuits from .stim text format into
Qiskit QuantumCircuit objects. Used as an independent oracle for
statevector validation against UCC.

Supported gates: H, S, S_DAG, T, T_DAG, X, Y, Z, CX, CY, CZ, M, MX, MY, R, RX, MR, MRX.
Noise instructions and annotations (TICK, DETECTOR, etc.) are skipped.
"""

from __future__ import annotations

import re

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def stim_to_qiskit(stim_text: str) -> QuantumCircuit:
    """Convert a .stim circuit string to a Qiskit QuantumCircuit.

    Parses the stim text line-by-line, extracts gate operations, and builds
    the equivalent Qiskit circuit. Noise instructions, annotations, and
    coordinate metadata are silently skipped.

    Args:
        stim_text: Circuit in .stim text format.

    Returns:
        Equivalent Qiskit QuantumCircuit.

    Raises:
        ValueError: If an unsupported gate is encountered.
    """
    num_qubits = _find_num_qubits(stim_text)
    num_clbits = _count_measurements(stim_text)
    qc = QuantumCircuit(num_qubits, num_clbits)
    clbit_idx = 0

    for line in stim_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        gate, arg, targets = _parse_line(line)
        if gate is None:
            continue

        clbit_idx = _apply_gate(qc, gate, targets, clbit_idx)

    return qc


def qiskit_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Get the statevector from a Qiskit circuit using AerSimulator.

    Args:
        qc: A Qiskit QuantumCircuit (should be noiseless, no measurements).

    Returns:
        Dense statevector as a numpy array of complex128, in little-endian
        qubit ordering (qubit 0 = LSB) to match UCC's convention.
    """
    qc_copy = qc.copy()
    qc_copy.save_statevector()
    sim = AerSimulator(method="statevector")
    result = sim.run(qc_copy, shots=1).result()
    sv = result.data()["statevector"]
    # Qiskit's Statevector index convention: index bit i corresponds to qubit i.
    # This already matches Stim/UCC little-endian convention (qubit 0 = LSB).
    return np.asarray(sv.data, dtype=np.complex128)


def stim_to_qiskit_noiseless(stim_text: str) -> QuantumCircuit:
    """Convert a .stim circuit to Qiskit, stripping all measurements.

    Useful for statevector comparison where measurements would collapse state.

    Args:
        stim_text: Circuit in .stim text format.

    Returns:
        Qiskit QuantumCircuit with only unitary gates (no measurements).
    """
    num_qubits = _find_num_qubits(stim_text)
    qc = QuantumCircuit(num_qubits)

    for line in stim_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        gate, arg, targets = _parse_line(line)
        if gate is None:
            continue

        if gate in ("M", "MX", "MY", "MR", "MRX", "MRY", "R", "RX", "RY"):
            raise ValueError(
                f"Gate '{gate}' not supported in noiseless statevector oracle. "
                "Measurements and resets collapse state."
            )

        _apply_gate(qc, gate, targets, 0)

    return qc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Gates that are annotations/noise and should be skipped
_SKIP_GATES = frozenset(
    {
        "TICK",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "MPP",  # Pauli product measurements need special handling
    }
)

# Regex to strip parenthesized arguments like (0.001)
_ARG_RE = re.compile(r"\(([^)]+)\)")


def _parse_line(line: str) -> tuple[str | None, float | None, list[int]]:
    """Parse a single stim line into (gate, arg, qubit_targets).

    Returns (None, None, []) for lines that should be skipped.
    """
    parts = line.split()
    if not parts:
        return None, None, []

    raw_gate = parts[0]

    # Extract parenthesized argument if present
    arg_match = _ARG_RE.search(raw_gate)
    arg = float(arg_match.group(1)) if arg_match else None
    gate = _ARG_RE.sub("", raw_gate).upper()

    if gate in _SKIP_GATES:
        return None, None, []

    # Parse targets: skip rec[] references, extract bare integer qubit indices
    targets: list[int] = []
    for tok in parts[1:]:
        if tok.startswith("rec[") or tok.startswith("!"):
            continue
        try:
            targets.append(int(tok))
        except ValueError:
            continue

    return gate, arg, targets


def _find_num_qubits(stim_text: str) -> int:
    """Find the number of qubits (max qubit index + 1).

    Returns at least 1 even for empty circuits to avoid creating a
    zero-qubit QuantumCircuit.
    """
    max_q = 0
    for line in stim_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        gate, _, targets = _parse_line(line)
        if gate is None:
            continue
        if targets:
            max_q = max(max_q, max(targets))
    return max(max_q + 1, 1)


def _count_measurements(stim_text: str) -> int:
    """Count total measurement operations for classical register sizing."""
    count = 0
    for line in stim_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        gate, _, targets = _parse_line(line)
        if gate in ("M", "MR", "MX", "MY", "MRX"):
            count += len(targets)
    return count


def _apply_gate(qc: QuantumCircuit, gate: str, targets: list[int], clbit_idx: int) -> int:
    """Apply a gate to the Qiskit circuit. Returns updated clbit_idx."""
    if gate == "H":
        for q in targets:
            qc.h(q)
    elif gate == "S":
        for q in targets:
            qc.s(q)
    elif gate == "S_DAG":
        for q in targets:
            qc.sdg(q)
    elif gate == "T":
        for q in targets:
            qc.t(q)
    elif gate == "T_DAG":
        for q in targets:
            qc.tdg(q)
    elif gate == "X":
        for q in targets:
            qc.x(q)
    elif gate == "Y":
        for q in targets:
            qc.y(q)
    elif gate == "Z":
        for q in targets:
            qc.z(q)
    elif gate == "CX":
        if len(targets) % 2 != 0:
            raise ValueError(f"CX requires even number of targets, got {len(targets)}")
        for i in range(0, len(targets), 2):
            qc.cx(targets[i], targets[i + 1])
    elif gate == "CY":
        if len(targets) % 2 != 0:
            raise ValueError(f"CY requires even number of targets, got {len(targets)}")
        for i in range(0, len(targets), 2):
            qc.cy(targets[i], targets[i + 1])
    elif gate == "CZ":
        if len(targets) % 2 != 0:
            raise ValueError(f"CZ requires even number of targets, got {len(targets)}")
        for i in range(0, len(targets), 2):
            qc.cz(targets[i], targets[i + 1])
    elif gate == "M":
        for q in targets:
            qc.measure(q, clbit_idx)
            clbit_idx += 1
    elif gate == "MX":
        # Stim MX semantics: H-measure-H preserves X-basis eigenstate.
        # The post-measurement H is correct for matching Stim's state update,
        # though stim_to_qiskit_noiseless() skips measurements entirely.
        for q in targets:
            qc.h(q)
            qc.measure(q, clbit_idx)
            qc.h(q)
            clbit_idx += 1
    elif gate == "MR":
        for q in targets:
            qc.measure(q, clbit_idx)
            qc.reset(q)
            clbit_idx += 1
    elif gate in ("R", "RX"):
        for q in targets:
            qc.reset(q)
    else:
        raise ValueError(f"Unsupported gate in Qiskit translator: {gate}")

    return clbit_idx
