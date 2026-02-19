"""Parse .stim circuit files into the op-list format used by clifft's prototype.

The op-list format is ``list[tuple[str, list]]`` where each entry is
``(gate_name, args)``.  This module provides a **text-based** parser
rather than relying on the ``stim`` Python library so that non-standard
gates like ``T`` and ``T_DAG`` (which stim's API rejects) are handled
natively.

Conversion rules
~~~~~~~~~~~~~~~~
* **Single-qubit gates** (H, S, S_DAG, X, Y, Z, T, T_DAG, RX, R):
  One instruction per target qubit.
* **Two-qubit gates** (CX, CZ): Targets are consumed in pairs.
* **Measurements** (M, MR, MX): One per target qubit.  Inversion
  flags (``!``) are noted but otherwise ignored for proxy purposes.
* **MPP**: Each Pauli-product (delimited by spaces) becomes one
  ``("MPP", [[(pauli_char, qubit_idx), ...]])`` entry.
* **Noise** (DEPOLARIZE1, DEPOLARIZE2, X_ERROR, Z_ERROR): Included
  unless ``skip_noise=True``.
* **Annotations** (TICK, QUBIT_COORDS, SHIFT_COORDS, DETECTOR,
  OBSERVABLE_INCLUDE, MPAD): Skipped when ``skip_annotations=True``.
* **REPEAT blocks**: Recursively parsed and unrolled.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Gate classification
# ---------------------------------------------------------------------------

SINGLE_QUBIT_GATES = frozenset({
    "H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG", "RX", "R",
    "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
    "SQRT_Z", "SQRT_Z_DAG",
})

TWO_QUBIT_GATES = frozenset({"CX", "CZ", "CY", "CNOT", "SWAP", "ISWAP", "ISWAP_DAG"})

MEASUREMENT_GATES = frozenset({"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"})

NOISE_GATES = frozenset({
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR",
    "Y_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
    "E", "ELSE_CORRELATED_ERROR",
})

# Noise gates whose targets are consumed in pairs (like two-qubit gates)
NOISE_TWO_QUBIT = frozenset({"DEPOLARIZE2", "PAULI_CHANNEL_2"})

ANNOTATION_GATES = frozenset({
    "TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR",
    "OBSERVABLE_INCLUDE", "MPAD",
})

# Regex for parsing an instruction line:
#   GATE_NAME ( args )  targets
# where (args) is optional.
_INST_RE = re.compile(
    r'^\s*'
    r'(?P<name>[A-Z_]+[0-9]?)'
    r'(?:\((?P<args>[^)]*)\))?'
    r'(?:\s+(?P<targets>.+))?'
    r'\s*$'
)

# Regex for a Pauli target inside an MPP product, e.g. "X3", "Y10", "Z0"
_PAULI_TARGET_RE = re.compile(r'^(!?)([XYZ])(\d+)$')

# Regex for a plain qubit target, possibly with inversion flag
_QUBIT_TARGET_RE = re.compile(r'^(!?)(\d+)$')

# Regex for a measurement record reference, e.g. rec[-3]
_REC_RE = re.compile(r'^rec\[(-?\d+)\]$')

# ---------------------------------------------------------------------------
# Tokeniser: turn raw text into a stream of (line, line_no) respecting
# REPEAT … { … } block structure.
# ---------------------------------------------------------------------------


def _tokenize_lines(text: str) -> list[str]:
    """Split circuit text into logical lines, stripping comments."""
    lines: list[str] = []
    for raw in text.splitlines():
        # Strip trailing comment
        line = raw.split("#")[0].strip()
        if line:
            lines.append(line)
    return lines


def _find_matching_brace(lines: list[str], start: int) -> int:
    """Return the index of the ``}`` that closes the ``{`` at *start*.

    Parameters
    ----------
    lines : list[str]
        The logical lines of the circuit.
    start : int
        Index of the line containing the opening ``{``.
    """
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{")
        depth -= lines[i].count("}")
        if depth == 0:
            return i
    raise ValueError(f"Unmatched '{{' starting at line {start + 1}")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_gate_args(args_str: str | None) -> list[float]:
    """Parse the parenthesised gate arguments, e.g. ``'0.001'`` → ``[0.001]``."""
    if not args_str:
        return []
    return [float(x.strip()) for x in args_str.split(",") if x.strip()]


def _parse_mpp_targets(targets_str: str) -> list[list[tuple[str, int]]]:
    """Parse MPP target string into a list of Pauli products.

    Each product is a list of ``(pauli_char, qubit_idx)`` tuples.
    Products are separated by whitespace; within a product, individual
    Pauli targets are separated by ``*``.

    Example
    -------
    >>> _parse_mpp_targets("X0*X3*X7 Z0*Z3*Z7")
    [[('X', 0), ('X', 3), ('X', 7)], [('Z', 0), ('Z', 3), ('Z', 7)]]
    """
    products: list[list[tuple[str, int]]] = []
    for product_str in targets_str.split():
        factors: list[tuple[str, int]] = []
        for part in product_str.split("*"):
            m = _PAULI_TARGET_RE.match(part)
            if not m:
                raise ValueError(f"Cannot parse MPP target: {part!r}")
            pauli = m.group(2)  # X, Y, or Z
            qubit = int(m.group(3))
            factors.append((pauli, qubit))
        products.append(factors)
    return products


def _parse_qubit_targets(targets_str: str) -> list[int]:
    """Parse a space-separated list of qubit indices, stripping ``!`` flags."""
    qubits: list[int] = []
    for tok in targets_str.split():
        m = _QUBIT_TARGET_RE.match(tok)
        if m:
            qubits.append(int(m.group(2)))
        else:
            # Could be a rec[-k] in a DETECTOR line etc. — skip
            pass
    return qubits


# ---------------------------------------------------------------------------
# Core recursive parser
# ---------------------------------------------------------------------------


def _parse_block(
    lines: list[str],
    start: int,
    end: int,
    *,
    skip_noise: bool,
    skip_annotations: bool,
) -> list[tuple[str, list]]:
    """Parse lines[start:end] into an op-list.

    Handles REPEAT blocks via recursion.
    """
    ops: list[tuple[str, list]] = []
    i = start
    while i < end:
        line = lines[i]

        # --- REPEAT block ------------------------------------------------
        if line.startswith("REPEAT"):
            # Format: "REPEAT count {"
            parts = line.split()
            count = int(parts[1])
            # The '{' is on this line
            brace_close = _find_matching_brace(lines, i)
            body_ops = _parse_block(
                lines, i + 1, brace_close,
                skip_noise=skip_noise,
                skip_annotations=skip_annotations,
            )
            for _ in range(count):
                ops.extend(body_ops)
            i = brace_close + 1
            continue

        # --- Closing brace (shouldn't happen at top-level) ---------------
        if line == "}":
            i += 1
            continue

        # --- Regular instruction -----------------------------------------
        m = _INST_RE.match(line)
        if not m:
            # Unrecognised line — skip silently
            i += 1
            continue

        name = m.group("name")
        gate_args = _parse_gate_args(m.group("args"))
        targets_str = m.group("targets") or ""

        # Canonicalise some aliases
        if name == "CNOT":
            name = "CX"

        # --- Skip annotations -------------------------------------------
        if skip_annotations and name in ANNOTATION_GATES:
            i += 1
            continue

        # --- Skip noise if requested ------------------------------------
        if skip_noise and name in NOISE_GATES:
            i += 1
            continue

        # --- MPP ---------------------------------------------------------
        if name == "MPP":
            products = _parse_mpp_targets(targets_str)
            for product in products:
                ops.append(("MPP", [product]))
            i += 1
            continue

        # --- Noise instructions -----------------------------------------
        if name in NOISE_GATES:
            prob = gate_args[0] if gate_args else 0.0
            qubits = _parse_qubit_targets(targets_str)
            if name in NOISE_TWO_QUBIT:
                # Targets are pairs
                for j in range(0, len(qubits), 2):
                    ops.append((name, [qubits[j], qubits[j + 1], prob]))
            else:
                for q in qubits:
                    ops.append((name, [q, prob]))
            i += 1
            continue

        # --- Measurement gates ------------------------------------------
        if name in MEASUREMENT_GATES:
            qubits = _parse_qubit_targets(targets_str)
            for q in qubits:
                ops.append((name, [q]))
            i += 1
            continue

        # --- Two-qubit gates --------------------------------------------
        if name in TWO_QUBIT_GATES:
            qubits = _parse_qubit_targets(targets_str)
            if len(qubits) % 2 != 0:
                raise ValueError(
                    f"Two-qubit gate {name} has odd number of targets: {qubits}"
                )
            for j in range(0, len(qubits), 2):
                ops.append((name, [qubits[j], qubits[j + 1]]))
            i += 1
            continue

        # --- Single-qubit gates -----------------------------------------
        if name in SINGLE_QUBIT_GATES:
            qubits = _parse_qubit_targets(targets_str)
            for q in qubits:
                ops.append((name, [q]))
            i += 1
            continue

        # --- Annotations (not skipped) ----------------------------------
        if name in ANNOTATION_GATES:
            # Include as-is: (name, [raw_targets_string])
            ops.append((name, [targets_str]))
            i += 1
            continue

        # --- Unknown gate — include with raw target list ----------------
        qubits = _parse_qubit_targets(targets_str)
        ops.append((name, qubits))
        i += 1

    return ops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stim_circuit_to_ops(
    circuit_text: str,
    *,
    skip_noise: bool = False,
    skip_annotations: bool = True,
) -> list[tuple[str, list]]:
    """Convert stim circuit *text* into the op-list format.

    Parameters
    ----------
    circuit_text : str
        The raw ``.stim`` file contents.
    skip_noise : bool
        If ``True``, noise instructions (DEPOLARIZE1, X_ERROR, …) are omitted.
    skip_annotations : bool
        If ``True``, TICK / QUBIT_COORDS / DETECTOR / etc. are omitted.

    Returns
    -------
    list[tuple[str, list]]
        The op-list.  Each entry is ``(gate_name, args)``.
    """
    lines = _tokenize_lines(circuit_text)
    return _parse_block(
        lines, 0, len(lines),
        skip_noise=skip_noise,
        skip_annotations=skip_annotations,
    )


def stim_file_to_ops(
    path: str | Path,
    **kwargs,
) -> tuple[int, list[tuple[str, list]]]:
    """Load a ``.stim`` file and return ``(num_qubits, ops_list)``.

    ``num_qubits`` is inferred as ``max(qubit index) + 1`` across all ops.
    Additional keyword arguments are forwarded to :func:`stim_circuit_to_ops`.
    """
    text = Path(path).read_text()
    ops = stim_circuit_to_ops(text, **kwargs)
    max_q = _max_qubit(ops)
    n = max_q + 1 if max_q >= 0 else 0
    return n, ops


def _max_qubit(ops: list[tuple[str, list]]) -> int:
    """Return the maximum qubit index referenced in *ops*, or -1 if empty."""
    best = -1
    for name, args in ops:
        if name == "MPP":
            # args is [[(pauli, qubit), ...]]
            for product in args:
                for _pauli, qubit in product:
                    best = max(best, qubit)
        elif name in NOISE_GATES:
            # args may be [qubit, prob] or [q1, q2, prob]
            for a in args:
                if isinstance(a, int):
                    best = max(best, a)
        else:
            for a in args:
                if isinstance(a, int):
                    best = max(best, a)
    return best


# ---------------------------------------------------------------------------
# Convenience: pretty-print ops
# ---------------------------------------------------------------------------


def ops_summary(ops: list[tuple[str, list]]) -> dict[str, int]:
    """Return a counter of gate names."""
    from collections import Counter
    return dict(Counter(name for name, _ in ops))


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        circuit_path = sys.argv[1]
    else:
        circuit_path = str(
            Path(__file__).resolve().parent.parent / "circuits" / "target.stim"
        )

    print(f"Parsing: {circuit_path}")
    n, ops = stim_file_to_ops(circuit_path, skip_noise=False, skip_annotations=True)
    print(f"Qubits: {n}")
    print(f"Total ops: {len(ops)}")
    print(f"Gate counts: {ops_summary(ops)}")
    print()

    # Show first 40 and last 20 ops
    limit_head = 40
    limit_tail = 20
    for i, (gname, gargs) in enumerate(ops[:limit_head]):
        print(f"  [{i:4d}] {gname:16s} {gargs}")
    if len(ops) > limit_head + limit_tail:
        print(f"  ... ({len(ops) - limit_head - limit_tail} more ops) ...")
    for i in range(max(limit_head, len(ops) - limit_tail), len(ops)):
        gname, gargs = ops[i]
        print(f"  [{i:4d}] {gname:16s} {gargs}")

    # Also test with skip_noise=True
    print("\n--- With skip_noise=True ---")
    n2, ops2 = stim_file_to_ops(circuit_path, skip_noise=True, skip_annotations=True)
    print(f"Qubits: {n2}")
    print(f"Total ops (no noise): {len(ops2)}")
    print(f"Gate counts: {ops_summary(ops2)}")
