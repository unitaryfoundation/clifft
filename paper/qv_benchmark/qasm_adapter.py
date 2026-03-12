"""Unified QASM 2.0 parser and simulator adapter.

Parses transpiled QASM 2.0 (cx + u3 basis) into an intermediate gate list,
then converts to formats consumed by UCC/Stim, Qulacs, Cirq, and Qiskit.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import cirq
    import qiskit.circuit
    import qulacs


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------


class GateOp(NamedTuple):
    """A single gate operation extracted from QASM."""

    name: str
    params: list[float]
    qubits: list[int]


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator for QASM parameter expressions
# ---------------------------------------------------------------------------

# Matches tokens we allow: numbers (int/float/scientific), ``pi``, operators,
# and parentheses.  Anything else causes a ``ValueError``.
_SAFE_TOKEN_RE: re.Pattern[str] = re.compile(
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?" r"|\.\d+(?:[eE][+-]?\d+)?" r"|pi" r"|[+\-*/()]" r"|\s+)"
)


def _safe_eval_expr(expr: str) -> float:
    """Evaluate a QASM parameter expression safely.

    Supports ``pi``, decimal literals, and the four basic arithmetic
    operators with parentheses.  No other names or builtins are allowed.

    Parameters
    ----------
    expr:
        A string such as ``"pi/2"``, ``"-pi"``, ``"3*pi/4"``.

    Returns
    -------
    float
        The evaluated numeric value.

    Raises
    ------
    ValueError
        If the expression contains disallowed tokens.
    """
    stripped: str = expr.strip()
    if not stripped:
        raise ValueError("Empty parameter expression")

    # Verify every character is covered by safe tokens.
    reconstructed: str = "".join(_SAFE_TOKEN_RE.findall(stripped))
    if reconstructed.replace(" ", "") != stripped.replace(" ", ""):
        raise ValueError(f"Unsafe token in expression: {expr!r}")

    # Replace the symbolic ``pi`` with its numeric value and evaluate.
    sanitized: str = stripped.replace("pi", repr(math.pi))
    return float(eval(sanitized, {"__builtins__": {}}, {}))  # noqa: S307


# ---------------------------------------------------------------------------
# QASM 2.0 parser
# ---------------------------------------------------------------------------

# Regexes for the three gate forms we care about.
_RE_U3: re.Pattern[str] = re.compile(r"u3\(([^)]+)\)\s+(\w+)\[(\d+)\]\s*;")
_RE_CX: re.Pattern[str] = re.compile(r"cx\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]\s*;")
_RE_MEASURE: re.Pattern[str] = re.compile(r"measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]\s*;")
_RE_BARRIER: re.Pattern[str] = re.compile(r"barrier\b")
_RE_QREG: re.Pattern[str] = re.compile(r"qreg\s+(\w+)\[(\d+)\]\s*;")


def parse_qasm(qasm: str) -> tuple[list[GateOp], int]:
    """Parse a QASM 2.0 string into a list of gate operations.

    Only handles the subset produced by Qiskit transpilation to the
    ``cx`` + ``u3`` basis: ``u3``, ``cx``, ``measure``, and ``barrier``.

    Parameters
    ----------
    qasm:
        QASM 2.0 source text.

    Returns
    -------
    tuple[list[GateOp], int]
        A (gates, num_qubits) pair.
    """
    ops: list[GateOp] = []
    num_qubits: int = 0

    for line in qasm.splitlines():
        line = line.strip()  # noqa: PLW2901

        # Skip header lines.
        if (
            not line
            or line.startswith("OPENQASM")
            or line.startswith("include")
            or line.startswith("creg")
        ):
            continue

        # qreg declaration -- extract qubit count.
        m_qreg: re.Match[str] | None = _RE_QREG.match(line)
        if m_qreg is not None:
            num_qubits = max(num_qubits, int(m_qreg.group(2)))
            continue

        # barrier -- record but typically skipped by adapters.
        if _RE_BARRIER.match(line):
            ops.append(GateOp(name="barrier", params=[], qubits=[]))
            continue

        # u3 gate.
        m_u3: re.Match[str] | None = _RE_U3.match(line)
        if m_u3 is not None:
            raw_params: str = m_u3.group(1)
            params: list[float] = [_safe_eval_expr(p) for p in raw_params.split(",")]
            qubit: int = int(m_u3.group(3))
            ops.append(GateOp(name="u3", params=params, qubits=[qubit]))
            continue

        # cx gate.
        m_cx: re.Match[str] | None = _RE_CX.match(line)
        if m_cx is not None:
            ctrl: int = int(m_cx.group(2))
            tgt: int = int(m_cx.group(4))
            ops.append(GateOp(name="cx", params=[], qubits=[ctrl, tgt]))
            continue

        # measure.
        m_meas: re.Match[str] | None = _RE_MEASURE.match(line)
        if m_meas is not None:
            q_idx: int = int(m_meas.group(2))
            c_idx: int = int(m_meas.group(4))
            ops.append(GateOp(name="measure", params=[], qubits=[q_idx, c_idx]))
            continue

    return ops, num_qubits


# ---------------------------------------------------------------------------
# Adapter 1: UCC Stim-superset format
# ---------------------------------------------------------------------------


def to_ucc_stim(qasm: str) -> str:
    """Convert QASM 2.0 to UCC's Stim-superset text format.

    Mapping
    -------
    * ``u3(t,p,l) q[i]`` -> ``U3(t/pi, p/pi, l/pi) i``
    * ``cx q[i],q[j]``   -> ``CX i j``
    * ``measure ...``     -> ``M i``
    * barriers are skipped

    UCC's rotation gates use half-turn units (multiples of pi), while
    QASM 2.0 uses radians. Parameters are divided by pi during conversion.

    Parameters
    ----------
    qasm:
        QASM 2.0 source.

    Returns
    -------
    str
        UCC/Stim-superset program text.
    """
    ops, _nq = parse_qasm(qasm)
    lines: list[str] = []
    for op in ops:
        if op.name == "u3":
            t, p, lam = (x / math.pi for x in op.params)
            lines.append(f"U3({t},{p},{lam}) {op.qubits[0]}")
        elif op.name == "cx":
            lines.append(f"CX {op.qubits[0]} {op.qubits[1]}")
        elif op.name == "measure":
            lines.append(f"M {op.qubits[0]}")
        # barriers are silently skipped
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Adapter 2: Qulacs
# ---------------------------------------------------------------------------


def to_qulacs_circuit(
    qasm: str,
) -> tuple[qulacs.QuantumCircuit, int]:
    """Build a Qulacs ``QuantumCircuit`` from a QASM 2.0 string.

    Parameters
    ----------
    qasm:
        QASM 2.0 source.

    Returns
    -------
    tuple[qulacs.QuantumCircuit, int]
        The constructed circuit and the number of qubits.
    """
    import qulacs as _qulacs  # noqa: PLC0415
    import qulacs.gate as _gate  # noqa: PLC0415

    ops, num_qubits = parse_qasm(qasm)
    circuit: _qulacs.QuantumCircuit = _qulacs.QuantumCircuit(num_qubits)
    register_idx: int = 0

    for op in ops:
        if op.name == "u3":
            theta, phi, lam = op.params
            circuit.add_gate(
                _gate.U3(op.qubits[0], theta, phi, lam),
            )
        elif op.name == "cx":
            circuit.add_gate(_gate.CNOT(op.qubits[0], op.qubits[1]))
        elif op.name == "measure":
            circuit.add_gate(
                _gate.Measurement(op.qubits[0], register_idx),
            )
            register_idx += 1
        # barriers are silently skipped

    return circuit, num_qubits


# ---------------------------------------------------------------------------
# Adapter 3: Cirq
# ---------------------------------------------------------------------------


def _strip_barriers(qasm: str) -> str:
    """Remove ``barrier`` lines from a QASM string.

    Some parsers (e.g. Cirq) do not recognise the ``barrier`` directive.
    """
    return "\n".join(line for line in qasm.splitlines() if not line.strip().startswith("barrier"))


def to_cirq_circuit(qasm: str) -> cirq.Circuit:
    """Load a QASM 2.0 string into a Cirq ``Circuit``.

    Cirq handles QASM natively via its contrib QASM importer.  Barrier
    directives are stripped beforehand because Cirq does not support them.

    Parameters
    ----------
    qasm:
        QASM 2.0 source.

    Returns
    -------
    cirq.Circuit
        The parsed Cirq circuit.
    """
    from cirq.contrib.qasm_import import circuit_from_qasm  # noqa: PLC0415

    return circuit_from_qasm(_strip_barriers(qasm))


# ---------------------------------------------------------------------------
# Adapter 4: Qiskit
# ---------------------------------------------------------------------------


def to_qiskit_circuit(qasm: str) -> qiskit.circuit.QuantumCircuit:
    """Load a QASM 2.0 string back into a Qiskit ``QuantumCircuit``.

    Parameters
    ----------
    qasm:
        QASM 2.0 source.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        The parsed Qiskit circuit.
    """
    from qiskit.circuit import QuantumCircuit  # noqa: PLC0415

    return QuantumCircuit.from_qasm_str(qasm)
