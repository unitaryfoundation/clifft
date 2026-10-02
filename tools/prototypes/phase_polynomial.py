"""Exact, state-specific phase-polynomial source rewrite for exploratory use.

The supported fragment starts in |0>, prepares independent |+> variables,
then uses CX/SWAP and diagonal Clifford+T gates. Proven X measurements may
appear inside that block; all other measurements must be terminal. Unsupported
inputs are returned unchanged. This is not an arbitrary-input unitary rewrite.
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

Polynomial = dict[int, int]


@dataclass(frozen=True)
class Reduction:
    circuit: str
    prepared_state: str
    applied: bool
    reason: str
    initial_width: int = 0
    logical_width: int = 0
    deterministic_records: int = 0
    input_t_count: int = 0
    output_t_count: int = 0


class _Unsupported(ValueError):
    pass


def _bits(mask: int) -> list[int]:
    return [1 << j for j in range(mask.bit_length()) if mask & (1 << j)]


def _add(poly: Polynomial, mask: int, coefficient: int) -> None:
    value = (poly.get(mask, 0) + coefficient) % 8
    if value:
        poly[mask] = value
    else:
        poly.pop(mask, None)


def _parity_phase(poly: Polynomial, mask: int, coefficient: int) -> None:
    bits = _bits(mask)
    # Higher-degree terms in the integer expansion of XOR vanish modulo eight.
    for degree in range(1, 4):
        factor = coefficient * (-2) ** (degree - 1)
        if factor % 8:
            for terms in itertools.combinations(bits, degree):
                _add(poly, sum(terms), factor)


def _cz_phase(poly: Polynomial, left: int, right: int) -> None:
    for a in _bits(left):
        for b in _bits(right):
            _add(poly, a | b, 4)


def _derivative(poly: Polynomial, flip: int) -> Polynomial:
    out: Polynomial = {}
    for mask, coefficient in poly.items():
        changed = mask & flip
        if not changed:
            continue
        base = mask ^ changed
        sub = changed
        while True:
            _add(out, base | sub, coefficient * (-1 if sub.bit_count() % 2 else 1))
            if not sub:
                break
            sub = (sub - 1) & changed
        _add(out, mask, -coefficient)
    return out


def _basis_add(basis: dict[int, tuple[int, int]], vector: int, label: int) -> bool:
    while vector:
        pivot = vector.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = (vector, label)
            return True
        row, row_label = basis[pivot]
        vector ^= row
        label ^= row_label
    return False


def _coordinates(rows: list[int], width: int, physical: int) -> int | None:
    basis: dict[int, tuple[int, int]] = {}
    for j in range(width):
        column = sum(1 << q for q, row in enumerate(rows) if row & (1 << j))
        _basis_add(basis, column, 1 << j)
    label = 0
    while physical:
        pivot = physical.bit_length() - 1
        if pivot not in basis:
            return None
        row, row_label = basis[pivot]
        physical ^= row
        label ^= row_label
    return label


def _rotation(qubit: int, coefficient: int) -> list[str]:
    names = {
        0: (),
        1: ("T",),
        2: ("S",),
        3: ("S", "T"),
        4: ("Z",),
        5: ("Z", "T"),
        6: ("S_DAG",),
        7: ("T_DAG",),
    }
    return [f"{gate} {qubit}" for gate in names[coefficient % 8]]


def _synthesize_phase(poly: Polynomial, coordinates: list[int]) -> list[str]:
    mapping = {1 << old: new for new, old in enumerate(coordinates)}
    output: list[str] = []
    for mask, coefficient in sorted(poly.items()):
        qubits = [mapping[b] for b in _bits(mask)]
        if len(qubits) == 1:
            output.extend(_rotation(qubits[0], coefficient))
        elif len(qubits) == 2:
            a, b = qubits
            if coefficient == 4:
                output.append(f"CZ {a} {b}")
            else:
                assert coefficient % 2 == 0
                # 2ab = a + b - (a XOR b), over the integers.
                output.extend(_rotation(a, coefficient // 2))
                output.extend(_rotation(b, coefficient // 2))
                output.append(f"CX {a} {b}")
                output.extend(_rotation(b, -coefficient // 2))
                output.append(f"CX {a} {b}")
        else:
            assert len(qubits) == 3 and coefficient == 4
            output.append("CCZ " + " ".join(map(str, qubits)))
    return output


def _encode(rows: list[int], columns: list[int]) -> list[str]:
    matrix = [
        sum(((row & col).bit_count() % 2) << j for j, col in enumerate(columns)) for row in rows
    ]
    undo: list[str] = []
    for j in range(len(columns)):
        pivot = next((q for q in range(j, len(rows)) if matrix[q] & (1 << j)), None)
        if pivot is None:
            raise _Unsupported("the prepared linear map is not injective")
        if pivot != j:
            matrix[j], matrix[pivot] = matrix[pivot], matrix[j]
            undo.append(f"SWAP {j} {pivot}")
        for q in range(len(rows)):
            if q != j and matrix[q] & (1 << j):
                matrix[q] ^= matrix[j]
                undo.append(f"CX {j} {q}")
    assert matrix == [1 << j for j in range(len(columns))] + [0] * (len(rows) - len(columns))
    return undo[::-1]


def _qubits(args: list[str]) -> list[int]:
    if not args or any(not re.fullmatch(r"\d+", q) for q in args):
        raise _Unsupported("unsupported qubit targets")
    return list(map(int, args))


def reduce_phase_polynomial(
    source: str, *, max_variables: int = 32, allow_t_expansion: bool = False
) -> Reduction:
    """Return a prepared-state equivalent circuit, preserving every record slot.

    Noise, feedback, resets, repeat blocks, arbitrary-angle rotations and
    Hadamards outside the initial preparation are conservative rejection cases.
    The variable cap bounds the cubic phase-polynomial representation.
    By default, candidates that increase T count are also left unchanged.
    Set allow_t_expansion only to explore the unguarded resynthesis.
    """
    if max_variables < 0 or max_variables > 64:
        raise ValueError("max_variables must be between zero and 64")
    try:
        return _reduce(source, max_variables, allow_t_expansion)
    except _Unsupported as error:
        return Reduction(source, "", False, str(error))


def _reduce(source: str, max_variables: int, allow_t_expansion: bool) -> Reduction:
    instructions = []
    qubits: set[int] = set()
    for line in source.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        fields = clean.split()
        gate, args = fields[0], fields[1:]
        if gate in ("H", "CX", "CNOT", "SWAP", "T", "T_DAG", "S", "S_DAG", "Z", "CZ", "I", "MX"):
            targets = _qubits(args)
            if gate in ("CX", "CNOT", "SWAP", "CZ"):
                if len(targets) % 2 or any(a == b for a, b in zip(targets[::2], targets[1::2])):
                    raise _Unsupported("invalid two-qubit targets")
            qubits.update(targets)
        elif gate == "MPP":
            for product_token in args:
                if not re.fullmatch(r"X\d+(?:\*X\d+)*", product_token):
                    raise _Unsupported("only X-product measurements are supported")
                targets = [int(q[1:]) for q in product_token.split("*")]
                if len(set(targets)) != len(targets):
                    raise _Unsupported("repeated measurement factors are unsupported")
                qubits.update(targets)
            if not args:
                raise _Unsupported("empty product measurement")
        elif not (
            re.fullmatch(r"DETECTOR(?:\([^)]*\))?", gate)
            or re.fullmatch(r"OBSERVABLE_INCLUDE\(\d+\)", gate)
        ):
            raise _Unsupported(f"unsupported instruction {gate}")
        instructions.append((clean, gate, args))
    n = max(qubits, default=-1) + 1
    if n > 65536:
        raise _Unsupported("physical qubit limit exceeded")
    rows = [0] * n
    width = 0
    initial = True
    terminal = False
    poly: Polynomial = {}
    tail: list[str] = []
    proven: list[int] = []
    records = 0
    input_t = 0
    for clean, gate, args in instructions:
        if gate == "H":
            if not initial:
                raise _Unsupported("Hadamard outside initial preparation")
            for q in _qubits(args):
                if rows[q]:
                    raise _Unsupported("repeated preparation Hadamard")
                rows[q] = 1 << width
                width += 1
                if width > max_variables:
                    raise _Unsupported("phase variable limit exceeded")
            continue
        if gate == "I":
            continue
        initial = False
        if gate.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
            # Absolute record order is unchanged, including padded checks.
            if any(not re.fullmatch(r"rec\[-[1-9]\d*\]", arg) for arg in args):
                raise _Unsupported("unsupported annotation targets")
            if any(int(arg[4:-1]) < -records for arg in args):
                raise _Unsupported("annotation refers to an unavailable record")
            tail.append(clean)
        elif gate in ("MX", "MPP"):
            products = (
                [[int(q)] for q in args]
                if gate == "MX"
                else [[int(q[1:]) for q in product.split("*")] for product in args]
            )
            for product in products:
                flip = None if terminal else _coordinates(rows, width, sum(1 << q for q in product))
                derivative = None if flip is None else _derivative(poly, flip)
                if derivative == {} or derivative == {0: 4}:
                    bit = int(bool(derivative))
                    tail.append(f"MPAD {bit}")
                    if not bit:
                        assert flip is not None
                        proven.append(flip)
                else:
                    terminal = True
                    tail.append("MPP " + "*".join(f"X{q}" for q in product))
                records += 1
        else:
            if terminal:
                raise _Unsupported("quantum gate after an unproved measurement")
            targets = _qubits(args)
            if gate in ("CX", "CNOT", "SWAP", "CZ"):
                for a, b in zip(targets[::2], targets[1::2]):
                    if gate in ("CX", "CNOT"):
                        rows[b] ^= rows[a]
                    elif gate == "SWAP":
                        rows[a], rows[b] = rows[b], rows[a]
                    else:
                        _cz_phase(poly, rows[a], rows[b])
            else:
                coefficient = {"T": 1, "T_DAG": 7, "S": 2, "S_DAG": 6, "Z": 4}[gate]
                for q in targets:
                    _parity_phase(poly, rows[q], coefficient)
                if gate in ("T", "T_DAG"):
                    input_t += len(targets)
    basis: dict[int, tuple[int, int]] = {}
    symmetry: list[int] = []
    for flip in proven:
        # A check at an earlier prefix need not remain a symmetry after later phases.
        if not _derivative(poly, flip) and _basis_add(basis, flip, 0):
            symmetry.append(flip)
    complement: list[int] = []
    for j in range(width):
        if _basis_add(basis, 1 << j, 0):
            complement.append(j)
    complement_mask = sum(1 << j for j in complement)
    quotient = {mask: c for mask, c in poly.items() if not mask & ~complement_mask}
    # Logical variables first makes the magic-bearing core explicit; the other
    # |+> variables carry only the proven translation symmetries.
    columns = [1 << j for j in complement] + symmetry
    phase = _synthesize_phase(quotient, complement)
    output_t = sum(line.split()[0] in ("T", "T_DAG") for line in phase)
    output_t += 7 * sum(line.startswith("CCZ ") for line in phase)
    # A sparse parity expression can become a dense monomial expression in
    # an unfortunate logical basis. Retain the compact input in that case.
    if output_t > input_t and not allow_t_expansion:
        return Reduction(
            source,
            "",
            False,
            f"resynthesis would increase T count from {input_t} to {output_t}",
            width,
            width,
            0,
            input_t,
            input_t,
        )
    prepared = [f"I {n - 1}"] if n else []
    if width:
        prepared.append("H " + " ".join(map(str, range(width))))
    prepared.extend(phase)
    prepared.extend(_encode(rows, columns))
    state = "\n".join(prepared) + "\n"
    return Reduction(
        state + "\n".join(tail) + "\n",
        state,
        True,
        "supported prepared-state fragment",
        width,
        len(complement),
        sum(line.startswith("MPAD ") for line in tail),
        input_t,
        output_t,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-variables", type=int, default=32)
    parser.add_argument("--allow-t-expansion", action="store_true")
    args = parser.parse_args()
    result = reduce_phase_polynomial(
        args.input.read_text(),
        max_variables=args.max_variables,
        allow_t_expansion=args.allow_t_expansion,
    )
    args.output.write_text(result.circuit)
    print(
        f"applied={result.applied} variables={result.initial_width}->{result.logical_width} "
        f"T={result.input_t_count}->{result.output_t_count} padded={result.deterministic_records}: "
        f"{result.reason}"
    )


if __name__ == "__main__":
    main()
