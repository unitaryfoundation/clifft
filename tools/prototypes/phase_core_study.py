"""Study exact Clifford compression and fixed Pauli fault trajectories.

This is offline research code, not a compiler pass or a sampling backend.
The fragment has initial H preparation, linear reversible gates, diagonal
Clifford+T gates, X-product checks, and terminal MX. Explicit X/Y/Z gates
represent selected Pauli faults; stochastic channels are not parsed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tools.prototypes.phase_polynomial import (
    Polynomial,
    _add,
    _basis_add,
    _bits,
    _coordinates,
    _cz_phase,
    _derivative,
    _encode,
    _parity_phase,
    _synthesize_phase,
)


def pauli_derivative(poly: Polynomial, flip: int) -> tuple[int, int] | None:
    """Return the constant and Z mask of a derivative that implements a Pauli."""
    derivative = _derivative(poly, flip)
    z = 0
    for mask, coefficient in derivative.items():
        if mask:
            if mask.bit_count() != 1 or coefficient != 4:
                return None
            z |= mask
    constant = derivative.get(0, 0)
    if constant % 2:
        return None
    return constant, z


def stabilizer_kernel(poly: Polynomial, width: int) -> list[int]:
    """Find every X support admitting a Pauli stabilizer of the phase state.

    A translated phase difference must be affine with linear coefficients
    divisible by four. Its non-Pauli part is linear in the translation modulo
    Pauli phases, so its kernel is found by GF(2) elimination. Unlike literal
    X symmetries, this also detects stabilizers carrying Y and Z factors.
    """
    constraints: dict[int, int] = {}
    for j in range(width):
        for mask, coefficient in _derivative(poly, 1 << j).items():
            degree = mask.bit_count()
            if degree == 1:
                assert coefficient % 2 == 0
                non_pauli = (coefficient // 2) % 2
            elif degree == 2:
                assert coefficient % 4 == 0
                non_pauli = (coefficient // 4) % 2
            elif degree == 0:
                non_pauli = coefficient % 2
            else:
                raise ValueError("phase is outside diagonal Clifford+T")
            if non_pauli:
                constraints[mask] = constraints.get(mask, 0) | (1 << j)
    pivots: dict[int, int] = {}
    for row in constraints.values():
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    kernel = []
    for free in range(width):
        if free in pivots:
            continue
        vector = 1 << free
        for pivot, row in sorted(pivots.items()):
            if (row & vector).bit_count() % 2:
                vector |= 1 << pivot
        assert pauli_derivative(poly, vector) is not None
        kernel.append(vector)
    return kernel


def is_diagonal_clifford(poly: Polynomial) -> bool:
    return all(
        (mask.bit_count() == 0)
        or (mask.bit_count() == 1 and coefficient % 2 == 0)
        or (mask.bit_count() == 2 and coefficient == 4)
        for mask, coefficient in poly.items()
    )


def single_t_parity(poly: Polynomial) -> int | None:
    """Identify a single parity T modulo diagonal Clifford corrections."""
    parity = sum(mask for mask, c in poly.items() if mask.bit_count() == 1 and c % 2)
    if not parity:
        return None
    remainder = dict(poly)
    _parity_phase(remainder, parity, -1)
    return parity if is_diagonal_clifford(remainder) else None


def _substitute_cx(poly: Polynomial, control: int, target: int) -> Polynomial:
    out: Polynomial = {}
    for mask, coefficient in poly.items():
        _add(out, mask, coefficient)
        if mask & target:
            rest = mask ^ target
            _add(out, rest | control, coefficient)
            _add(out, rest | control | target, -2 * coefficient)
    return out


def compress_phase(poly: Polynomial, width: int) -> tuple[Polynomial, Polynomial, list[str]]:
    """Split a phase state into a nullity-sized core and a Clifford decoder.

    A kernel basis fixes the decoder independently of Pauli fault signs.
    In that basis, all phase terms involving the other variables are Clifford.
    """
    kernel = stabilizer_kernel(poly, width)
    span: dict[int, tuple[int, int]] = {}
    for vector in kernel:
        _basis_add(span, vector, 0)
    complement = []
    for j in range(width):
        if _basis_add(span, 1 << j, 0):
            complement.append(1 << j)
    columns = complement + kernel
    encoding = _encode([1 << j for j in range(width)], columns)
    transformed = dict(poly)
    for instruction in reversed(encoding):
        gate, a, b = instruction.split()
        left, right = 1 << int(a), 1 << int(b)
        if gate == "CX":
            transformed = _substitute_cx(transformed, left, right)
        else:
            transformed = {
                (mask ^ ((left | right) if bool(mask & left) != bool(mask & right) else 0)): c
                for mask, c in transformed.items()
            }
    magic: Polynomial = {}
    clifford = dict(transformed)
    for mask, coefficient in transformed.items():
        degree = mask.bit_count()
        core_coefficient = (
            coefficient % 2
            if degree == 1
            else coefficient % 4
            if degree == 2
            else coefficient
            if degree
            else 0
        )
        if core_coefficient:
            assert mask.bit_length() <= len(complement)
            _add(magic, mask, core_coefficient)
            _add(clifford, mask, -core_coefficient)
    assert is_diagonal_clifford(clifford)
    return magic, clifford, encoding


@dataclass
class Check:
    source: str
    flip: int
    prefix: Polynomial


@dataclass
class Trace:
    width: int
    rows: list[int]
    offsets: list[int]
    phase: Polynomial
    records: list[str | Check]


def trace(source: str) -> Trace:
    instructions = []
    n = 0
    for line in source.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        gate, *args = clean.split()
        if gate.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
            instructions.append((clean, gate, args))
            continue
        if gate == "MPP":
            targets = [int(q[1:]) for product in args for q in product.split("*")]
            if any(not q.startswith("X") for product in args for q in product.split("*")):
                raise ValueError("only X-product input checks are supported")
        else:
            targets = list(map(int, args))
        n = max(n, max(targets, default=-1) + 1)
        instructions.append((clean, gate, args))
    rows = [0] * n
    offsets = [0] * n
    width = 0
    initial = True
    terminal = False
    poly: Polynomial = {}
    records: list[str | Check] = []
    for clean, gate, args in instructions:
        if gate.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
            records.append(clean)
            continue
        if gate == "MPP":
            for product in args:
                physical = sum(1 << int(q[1:]) for q in product.split("*"))
                flip = _coordinates(rows, width, physical)
                if flip is None:
                    raise ValueError("check leaves the prepared affine subspace")
                records.append(Check("MPP " + product, flip, dict(poly)))
            continue
        if gate == "MX":
            terminal = True
            records.append(clean)
            continue
        targets = list(map(int, args))
        if gate == "I":
            continue
        if terminal:
            raise ValueError("quantum operation after terminal MX")
        if gate == "H":
            if not initial:
                raise ValueError("Hadamard outside initial preparation")
            for q in targets:
                if rows[q]:
                    raise ValueError("repeated preparation Hadamard")
                rows[q] = 1 << width
                width += 1
            continue
        initial = False
        if gate in ("CX", "CNOT", "SWAP", "CZ"):
            if len(targets) % 2:
                raise ValueError("unpaired targets")
            for a, b in zip(targets[::2], targets[1::2]):
                if gate in ("CX", "CNOT"):
                    rows[b] ^= rows[a]
                    offsets[b] ^= offsets[a]
                elif gate == "SWAP":
                    rows[a], rows[b] = rows[b], rows[a]
                    offsets[a], offsets[b] = offsets[b], offsets[a]
                else:
                    _cz_phase(poly, rows[a], rows[b])
                    if offsets[a]:
                        _parity_phase(poly, rows[b], 4)
                    if offsets[b]:
                        _parity_phase(poly, rows[a], 4)
        elif gate in ("X", "Y"):
            for q in targets:
                if gate == "Y":
                    _parity_phase(poly, rows[q], 4)
                offsets[q] ^= 1
        elif gate in ("T", "T_DAG", "S", "S_DAG", "Z"):
            coefficient = {"T": 1, "T_DAG": 7, "S": 2, "S_DAG": 6, "Z": 4}[gate]
            for q in targets:
                _parity_phase(poly, rows[q], -coefficient if offsets[q] else coefficient)
        else:
            raise ValueError(f"unsupported research instruction {gate}")
    return Trace(width, rows, offsets, poly, records)


def deferred_pauli(final: Polynomial, check: Check) -> tuple[int, int, int] | None:
    suffix = dict(final)
    for mask, coefficient in check.prefix.items():
        _add(suffix, mask, -coefficient)
    derivative = pauli_derivative(suffix, check.flip)
    if derivative is None:
        return None
    constant, z = derivative
    phase = (constant - 2 * (check.flip & z).bit_count()) % 8
    assert phase in (0, 4)
    return check.flip, z, phase // 4


def defer_fixed_faults(source: str) -> str:
    """Reference rewrite for one selected fault path, preserving record order.

    Measurements are conjugated through their complete suffix. This permits
    genuinely random syndrome outcomes; it never pads noisy checks with zero.
    The generated circuit can be large and is not a performance optimization.
    """
    result = trace(source)
    output = [f"I {len(result.rows) - 1}"] if result.rows else []
    if result.width:
        output.append("H " + " ".join(map(str, range(result.width))))
    magic, clifford, coordinate_encoding = compress_phase(result.phase, result.width)
    output.extend(_synthesize_phase(magic, list(range(result.width))))
    output.extend(_synthesize_phase(clifford, list(range(result.width))))
    output.extend(coordinate_encoding)
    encoded = False
    for record in result.records:
        if isinstance(record, Check):
            if encoded:
                output.append(record.source)
                continue
            pauli = deferred_pauli(result.phase, record)
            if pauli is None:
                raise ValueError("the measurement suffix is not a Pauli conjugation")
            x, z, negative = pauli
            factors = []
            for bit in _bits(x | z):
                axis = "Y" if x & bit and z & bit else "X" if x & bit else "Z"
                factors.append(f"{axis}{bit.bit_length() - 1}")
            output.append("MPP " + ("!" if negative else "") + "*".join(factors))
        else:
            if record.startswith("MX") and not encoded:
                output.extend(_encode(result.rows, [1 << j for j in range(result.width)]))
                for q, offset in enumerate(result.offsets):
                    if offset:
                        output.append(f"X {q}")
                encoded = True
            output.append(record)
    if not encoded:
        output.extend(_encode(result.rows, [1 << j for j in range(result.width)]))
        output.extend(f"X {q}" for q, offset in enumerate(result.offsets) if offset)
    return "\n".join(output) + "\n"


def study_corpus(corpus: Path, *, fault_paths: int = 0, seed: int = 171309) -> dict:
    if not 0 <= fault_paths <= 64:
        raise ValueError("fault_paths must be between zero and 64")
    items = []
    rng = random.Random(seed)
    paths = 0
    faulty_checks = 0
    with (corpus / "circuits.csv").open(newline="") as handle:
        for entry in csv.DictReader(handle):
            source = (corpus / entry["circuit"]).read_text()
            result = trace(source)
            kernel = stabilizer_kernel(result.phase, result.width)
            checks = [record for record in result.records if isinstance(record, Check)]
            if fault_paths:
                magic, _, encoding = compress_phase(result.phase, result.width)
                lines = source.splitlines()
                candidates = [
                    j
                    for j, line in enumerate(lines)
                    if line.startswith(
                        ("H ", "CX ", "SWAP ", "T ", "T_DAG ", "S ", "S_DAG ", "CZ ", "MPP ")
                    )
                ]
                for _ in range(fault_paths):
                    selected = rng.sample(candidates, min(24, len(candidates)))
                    faults = {
                        j: f"{rng.choice('XYZ')} {rng.randrange(len(result.rows))}"
                        for j in selected
                    }
                    faulty_source = "\n".join(
                        line + ("\n" + faults[j] if j in faults else "")
                        for j, line in enumerate(lines)
                    )
                    faulty = trace(faulty_source)
                    correction = dict(faulty.phase)
                    for mask, coefficient in result.phase.items():
                        _add(correction, mask, -coefficient)
                    assert is_diagonal_clifford(correction)
                    assert stabilizer_kernel(faulty.phase, faulty.width) == kernel
                    faulty_magic, _, faulty_encoding = compress_phase(faulty.phase, faulty.width)
                    assert faulty_magic == magic and faulty_encoding == encoding
                    for record in faulty.records:
                        if isinstance(record, Check):
                            assert deferred_pauli(faulty.phase, record) is not None
                            faulty_checks += 1
                    paths += 1
            items.append(
                {
                    "circuit": entry["circuit"],
                    "initial_variables": result.width,
                    "stabilizer_nullity": result.width - len(kernel),
                    "single_t_modulo_clifford": single_t_parity(result.phase) is not None,
                    "checks": len(checks),
                    "deterministic_ideal_checks": sum(
                        _derivative(check.prefix, check.flip) in ({}, {0: 4}) for check in checks
                    ),
                    "pauli_deferrable_checks": sum(
                        deferred_pauli(result.phase, check) is not None for check in checks
                    ),
                }
            )
    return {
        "circuits": len(items),
        "fixed_fault_paths": {
            "paths": paths,
            "faults_per_path_up_to": 24,
            "same_core_and_coordinate_encoding": paths,
            "pauli_deferrable_checks": faulty_checks,
            "seed": seed,
        },
        "nullity_histogram": dict(sorted(Counter(i["stabilizer_nullity"] for i in items).items())),
        "single_t_modulo_clifford": sum(i["single_t_modulo_clifford"] for i in items),
        "all_checks_deferrable": sum(i["checks"] == i["pauli_deferrable_checks"] for i in items),
        "all_checks_deterministic_ideal": sum(
            i["checks"] == i["deterministic_ideal_checks"] for i in items
        ),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fault-paths", type=int, default=0)
    parser.add_argument("--seed", type=int, default=171309)
    args = parser.parse_args()
    report = study_corpus(args.corpus, fault_paths=args.fault_paths, seed=args.seed)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
