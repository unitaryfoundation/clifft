"""Stim-text to Qiskit QuantumCircuit translator for validation.

Translates noiseless Clifford+T circuits from .stim text format into
Qiskit QuantumCircuit objects. Used as an independent oracle for
statevector validation against Clifft.

Supported gates: H, S, S_DAG, T, T_DAG, X, Y, Z, CX, CY, CZ, CH, CCZ, CCX,
R_X, R_Y, R_Z, U3, U, R_XX, RXX, R_YY, RYY, R_ZZ, RZZ, R_PAULI.
All rotation angles use half-turn units (alpha * pi = radians).
The oracle rejects nonunitary operations and unsupported syntax. TICK markers
and comments are allowed because they do not change the unitary.
"""

from __future__ import annotations

import re
from math import isfinite

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def qiskit_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Get the statevector from a Qiskit circuit using AerSimulator.

    Args:
        qc: A Qiskit QuantumCircuit (should be noiseless, no measurements).

    Returns:
        Dense statevector as a numpy array of complex128, in little-endian
        qubit ordering (qubit 0 = LSB) to match Clifft's convention.
    """
    qc_copy = qc.copy()
    qc_copy.save_statevector()
    sim = AerSimulator(method="statevector")
    result = sim.run(qc_copy, shots=1).result()
    sv = result.data()["statevector"]
    # Qiskit's Statevector index convention: index bit i corresponds to qubit i.
    # This already matches Stim/Clifft little-endian convention (qubit 0 = LSB).
    return np.asarray(sv.data, dtype=np.complex128)


def stim_to_qiskit_noiseless(stim_text: str) -> QuantumCircuit:
    """Convert the supported unitary subset without dropping physical operations.

    Noise, measurements, resets, feedback, and unsupported syntax are errors:
    silently removing them would make the oracle validate a different circuit.

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

        gate, args, targets, pauli_targets = _parse_line(line)
        if gate is None:
            continue

        _apply_gate(qc, gate, args, targets, pauli_targets)

    return qc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Target group size and parameter count for the exact unitary translator.
# R_PAULI accepts one product instead of groups of integer qubit targets.
_UNITARY_ARITIES = {
    **dict.fromkeys(("H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z"), (1, 0)),
    **dict.fromkeys(("CX", "CY", "CZ", "CH"), (2, 0)),
    **dict.fromkeys(("CCX", "CCZ"), (3, 0)),
    **dict.fromkeys(("R_X", "R_Y", "R_Z"), (1, 1)),
    **dict.fromkeys(("U3", "U"), (1, 3)),
    **dict.fromkeys(("R_XX", "RXX", "R_YY", "RYY", "R_ZZ", "RZZ"), (2, 1)),
    "R_PAULI": (0, 1),
}

# Regex to parse Pauli product targets like X0*Y1*Z2
_PAULI_RE = re.compile(r"([XYZ])(\d+)")


# Regex to extract gate name and optional parenthesized args from the raw line
_GATE_LINE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)"  # gate name
    r"(?:\(([^)]+)\))?"  # optional (args)
    r"(.*)$"  # rest of line (targets)
)


def _parse_line(
    line: str,
) -> tuple[str | None, list[float], list[int], list[tuple[str, int]]]:
    """Parse a single stim line into (gate, args, qubit_targets, pauli_targets).

    Returns (None, [], [], []) for lines that should be skipped.
    pauli_targets is a list of (pauli_char, qubit) tuples for R_PAULI.
    """
    # Strip inline comments
    comment_pos = line.find("#")
    if comment_pos >= 0:
        line = line[:comment_pos]
    line = line.strip()
    if not line:
        return None, [], [], []

    m = _GATE_LINE_RE.match(line)
    if not m:
        raise ValueError(f"Unsupported syntax in noiseless statevector oracle: {line}")

    gate = m.group(1).upper()
    args_str = m.group(2)  # may be None
    rest = m.group(3).strip()

    args: list[float] = []
    if args_str:
        args = [float(x.strip()) for x in args_str.split(",")]

    if gate == "TICK" and not args and not rest:
        return None, [], [], []
    if gate not in _UNITARY_ARITIES:
        raise ValueError(f"Gate '{gate}' not supported in noiseless statevector oracle")
    group_size, num_args = _UNITARY_ARITIES[gate]
    if len(args) != num_args or not all(isfinite(arg) for arg in args):
        raise ValueError(f"Invalid arguments in noiseless statevector oracle: {line}")
    tokens = rest.split()
    if group_size == 0:
        if len(tokens) != 1 or not re.fullmatch(r"[XYZ]\d+(?:\*[XYZ]\d+)*", tokens[0]):
            raise ValueError(f"Expected one Pauli product in noiseless oracle: {line}")
        qubits = [int(q) for q in re.findall(r"[XYZ](\d+)", tokens[0])]
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Repeated qubit in noiseless Pauli oracle: {line}")
    elif (
        not tokens
        or len(tokens) % group_size
        or any(not re.fullmatch(r"\d+", token) for token in tokens)
    ):
        raise ValueError(f"Invalid targets in noiseless statevector oracle: {line}")

    # Parse targets from the rest of the line
    targets: list[int] = []
    pauli_targets: list[tuple[str, int]] = []
    for tok in rest.split():
        if tok.startswith("rec[") or tok.startswith("!"):
            continue
        # Check for Pauli product syntax (X0*Y1*Z2)
        if any(c in tok for c in "XYZ") and not tok.startswith("rec"):
            for pm in _PAULI_RE.finditer(tok):
                pauli_targets.append((pm.group(1), int(pm.group(2))))
            continue
        try:
            targets.append(int(tok))
        except ValueError:
            continue

    return gate, args, targets, pauli_targets


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
        gate, _, targets, pauli_targets = _parse_line(line)
        if gate is None:
            continue
        if targets:
            max_q = max(max_q, max(targets))
        for _, q in pauli_targets:
            max_q = max(max_q, q)
    return max(max_q + 1, 1)


def _apply_gate(
    qc: QuantumCircuit,
    gate: str,
    args: list[float],
    targets: list[int],
    pauli_targets: list[tuple[str, int]],
) -> None:
    """Apply a supported unitary gate to the Qiskit circuit."""
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
    elif gate == "CH":
        if len(targets) % 2 != 0:
            raise ValueError(f"CH requires even number of targets, got {len(targets)}")
        for i in range(0, len(targets), 2):
            control = targets[i]
            target = targets[i + 1]
            # Aer does not assemble opaque CH instructions in all supported
            # versions, so use Qiskit's CHGate definition in basis gates.
            qc.s(target)
            qc.h(target)
            qc.t(target)
            qc.cx(control, target)
            qc.tdg(target)
            qc.h(target)
            qc.sdg(target)
    elif gate == "CCZ":
        if len(targets) % 3 != 0:
            raise ValueError(f"CCZ requires multiple of 3 targets, got {len(targets)}")
        for i in range(0, len(targets), 3):
            qc.h(targets[i + 2])
            qc.ccx(targets[i], targets[i + 1], targets[i + 2])
            qc.h(targets[i + 2])
    elif gate == "CCX":
        if len(targets) % 3 != 0:
            raise ValueError(f"CCX requires multiple of 3 targets, got {len(targets)}")
        for i in range(0, len(targets), 3):
            qc.ccx(targets[i], targets[i + 1], targets[i + 2])
    elif gate in ("R_X",):
        alpha = args[0] if args else 0.0
        for q in targets:
            qc.rx(alpha * np.pi, q)
    elif gate in ("R_Y",):
        alpha = args[0] if args else 0.0
        for q in targets:
            qc.ry(alpha * np.pi, q)
    elif gate in ("R_Z",):
        alpha = args[0] if args else 0.0
        for q in targets:
            qc.rz(alpha * np.pi, q)
    elif gate in ("U3", "U"):
        theta = args[0] * np.pi if len(args) > 0 else 0.0
        phi = args[1] * np.pi if len(args) > 1 else 0.0
        lam = args[2] * np.pi if len(args) > 2 else 0.0
        for q in targets:
            qc.u(theta, phi, lam, q)
    elif gate in ("R_XX", "RXX"):
        alpha = args[0] if args else 0.0
        for i in range(0, len(targets), 2):
            qc.rxx(alpha * np.pi, targets[i], targets[i + 1])
    elif gate in ("R_YY", "RYY"):
        alpha = args[0] if args else 0.0
        for i in range(0, len(targets), 2):
            qc.ryy(alpha * np.pi, targets[i], targets[i + 1])
    elif gate in ("R_ZZ", "RZZ"):
        alpha = args[0] if args else 0.0
        for i in range(0, len(targets), 2):
            qc.rzz(alpha * np.pi, targets[i], targets[i + 1])
    elif gate == "R_PAULI":
        from qiskit.quantum_info import Operator, SparsePauliOp
        from scipy.linalg import expm  # type: ignore[import-not-found]

        alpha = args[0] if args else 0.0
        n = qc.num_qubits
        label = ["I"] * n
        for pauli_char, q in pauli_targets:
            label[q] = pauli_char
        pauli_str = "".join(reversed(label))
        pauli_matrix = SparsePauliOp(pauli_str).to_matrix()
        unitary = expm(-1j * alpha * np.pi / 2.0 * pauli_matrix)
        qc.unitary(Operator(unitary), list(range(n)))
    else:
        raise ValueError(f"Unsupported gate in Qiskit translator: {gate}")
