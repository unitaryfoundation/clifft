"""Extract and verify fixed-fault Clifford corrections for a diagonal phase region.

The polynomial analysis is structural. The experiment driver selects the pinned
BT27 fixture and its saved fault patterns; this is not an executor backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from profile_bt27_fault_specialization import FIXTURE, Faults, optimize, source

import clifft

Polynomial = dict[int, int]


def add(poly: Polynomial, mask: int, coefficient: int) -> None:
    value = (poly.get(mask, 0) + coefficient) % 8
    if value:
        poly[mask] = value
    else:
        poly.pop(mask, None)


def bits(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


def phase_polynomial(lines: list[str], width: int) -> tuple[Polynomial, list[int]]:
    coordinates = [1 << q for q in range(width)]
    poly: Polynomial = {}
    work = 0
    for line in lines:
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        gate, *targets = words
        qubits = [int(q) for q in targets]
        if any(q < 0 or q >= width for q in qubits):
            raise ValueError("Target outside phase coordinates")
        if gate == "CX" and len(qubits) == 2:
            control, target = qubits
            coordinates[target] ^= coordinates[control]
        elif gate in ("T", "T_DAG") and len(qubits) == 1:
            support = bits(coordinates[qubits[0]])
            if len(support) > 24:
                raise ValueError("Parity expansion exceeds the diagnostic work budget")
            for degree in range(1, min(3, len(support)) + 1):
                coefficient = (1 if gate == "T" else -1) * (-2) ** (degree - 1)
                for term in combinations(support, degree):
                    work += 1
                    if work > 1_000_000:
                        raise ValueError("Phase analysis exceeds the diagnostic work budget")
                    add(poly, sum(1 << q for q in term), coefficient)
        else:
            raise ValueError(f"Unsupported operation in a phase region: {line}")
    return poly, coordinates


def region(source: str) -> tuple[list[str], list[str], list[str], Polynomial]:
    lines = source.splitlines()
    indices = [i for i, line in enumerate(lines) if line.startswith(("T ", "T_DAG "))]
    if not indices:
        raise ValueError("No phase region")
    first, last = indices[0], indices[-1]
    body = lines[first : last + 1]
    width = 1 + max(int(q) for line in body for q in line.split()[1:])
    poly, coordinates = phase_polynomial(body, width)
    end = last + 1
    # Finish the current parity computation. Stop at its first identity map so
    # no measurement, feedback, or unrelated decoder is crossed.
    while coordinates != [1 << q for q in range(width)]:
        if end == len(lines) or not lines[end].startswith("CX "):
            raise ValueError("Phase region does not return to its input coordinates")
        words = lines[end].split()
        if len(words) != 3:
            raise ValueError("Expected a single CNOT pair")
        control, target = map(int, words[1:])
        if not (0 <= control < width and 0 <= target < width):
            raise ValueError("CNOT leaves phase coordinates")
        coordinates[target] ^= coordinates[control]
        end += 1
    return lines[:first], lines[first:end], lines[end:], poly


def translate(poly: Polynomial, shift: int) -> Polynomial:
    result: Polynomial = {}
    for mask, coefficient in poly.items():
        expanded = {0: coefficient}
        for q in bits(mask):
            updated: Polynomial = {}
            for term, value in expanded.items():
                if shift >> q & 1:
                    add(updated, term, value)
                    add(updated, term | (1 << q), -value)
                else:
                    add(updated, term | (1 << q), value)
            expanded = updated
        for term, value in expanded.items():
            add(result, term, value)
    return result


def fault_masks(faults: Faults) -> tuple[int, int]:
    x = z = 0
    for q, pauli in faults:
        if q < 0 or pauli not in ("X", "Y", "Z"):
            raise ValueError("Invalid Pauli fault")
        if pauli in ("X", "Y"):
            x ^= 1 << q
        if pauli in ("Y", "Z"):
            z ^= 1 << q
    return x, z


def correction(poly: Polynomial, faults: Faults) -> tuple[list[str], int, Polynomial]:
    x, z = fault_masks(faults)
    difference = translate(poly, x)
    for mask, coefficient in poly.items():
        add(difference, mask, -coefficient)
    for q in bits(z):
        add(difference, 1 << q, 4)
    gates = []
    for mask, coefficient in sorted(difference.items()):
        support = bits(mask)
        if not support:
            continue
        if len(support) == 1 and coefficient % 2 == 0:
            gate = {2: "S", 4: "Z", 6: "S_DAG"}[coefficient]
            gates.append(f"{gate} {support[0]}")
        elif len(support) == 2 and coefficient == 4:
            gates.append(f"CZ {support[0]} {support[1]}")
        else:
            raise ValueError("Translated phase is not a Clifford correction")
    gates += [f"X {q}" for q in bits(x)]
    return gates, difference.get(0, 0), difference


def moved_source(original: str, faults: Faults) -> str:
    prefix, body, suffix, poly = region(original)
    gates, _, _ = correction(poly, faults)
    return "\n".join(prefix + body + gates + suffix) + "\n"


def cubic_controls(poly: Polynomial) -> dict[str, Any]:
    if any(mask.bit_count() != 3 or value != 4 for mask, value in poly.items()):
        raise ValueError("Expected a homogeneous cubic phase for this control summary")
    cz: dict[tuple[int, int], list[int]] = {}
    z: dict[int, list[tuple[int, int]]] = {}
    for mask in sorted(poly):
        triple = bits(mask)
        for q in triple:
            a, b = (i for i in triple if i != q)
            pair = (a, b)
            cz.setdefault(pair, []).append(q)
            z.setdefault(q, []).append(pair)
    return {
        "cz": [
            {"targets": pair, "xor_x_faults": controls} for pair, controls in sorted(cz.items())
        ],
        "z": [
            {"target": q, "xor_x_fault_products": pairs, "xor_z_fault": q}
            for q, pairs in sorted(z.items())
        ],
    }


def verify_cubic_controls(controls: dict[str, Any], faults: Faults, expected: Polynomial) -> None:
    x, z = fault_masks(faults)
    actual: Polynomial = {}
    for row in controls["cz"]:
        parity = sum(x >> q & 1 for q in row["xor_x_faults"]) % 2
        if parity:
            add(actual, sum(1 << q for q in row["targets"]), 4)
    for row in controls["z"]:
        parity = (z >> row["target"] & 1) ^ (
            sum((x >> a & 1) * (x >> b & 1) for a, b in row["xor_x_fault_products"]) % 2
        )
        if parity:
            add(actual, 1 << row["target"], 4)
    if actual != {mask: value for mask, value in expected.items() if mask}:
        raise AssertionError("Symbolic controls disagree with direct phase translation")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def variant_structure(faults: Faults) -> dict[str, Any]:
    hir, info = optimize(source(faults))
    ops = [op.as_dict() for op in hir]
    rotations = [op for op in ops if op["op_type"] == "T_GATE"]
    axes = [{k: v for k, v in op.items() if k != "is_dagger"} for op in rotations]
    skeleton = [{k: v for k, v in op.items() if k not in ("pauli_string", "sign")} for op in ops]
    return {
        "hir_operations": len(ops),
        "t_count": len(rotations),
        "peak_width": info["peak_width"],
        "t_sequence_sha256": digest(rotations),
        "t_axes_sha256": digest(axes),
        "hir_skeleton_sha256": digest(skeleton),
        # inspect() omits dependencies. This fingerprint is descriptive only.
        "printed_executable_sha256": digest(clifft.lower(hir).inspect()),
        "measurements": {
            op["meas_record_idx"]: (op["pauli_string"], op["sign"])
            for op in ops
            if op["op_type"] == "MEASURE"
        },
        "dagger_positions": [i for i, op in enumerate(rotations) if op["is_dagger"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--structure", action="store_true", help="Also compile variant fingerprints"
    )
    parser.add_argument(
        "--cases-output", type=Path, help="Write input for the native frame diagnostic"
    )
    args = parser.parse_args()
    original = FIXTURE.read_text()
    prefix, body, _, poly = region(original)
    controls = cubic_controls(poly)
    cases = {}
    manifest = []
    structures = {}
    for key, row in json.loads(args.study.read_text())["results"].items():
        if "faults" not in row:
            continue
        faults = tuple((q, p) for q, p in row["faults"])
        manifest.append(" ".join([key] + [f"{p}{q}" for q, p in faults]))
        gates, global_phase, difference = correction(poly, faults)
        verify_cubic_controls(controls, faults, difference)
        cases[key] = {
            "gates": dict(Counter(gate.split()[0] for gate in gates)),
            "global_phase_mod8": global_phase,
            "coefficient_identity_checked": True,
        }
        if args.structure:
            structures[key] = variant_structure(faults)
        if len(cases) % 64 == 0:
            print(f"Analyzed {len(cases)} variants", flush=True)
    if structures:
        baseline = structures["identity"]["measurements"]
        for row in structures.values():
            measured = row.pop("measurements")
            if measured.keys() != baseline.keys():
                raise AssertionError("Measurement record identities changed")
            row["changed_measurement_masks_or_signs"] = sum(
                measured[k] != baseline[k] for k in baseline
            )
    result = {
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "study_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest(),
        "source_lines": [len(prefix) + 1, len(prefix) + len(body)],
        "phase_terms": [
            {"qubits": bits(mask), "coefficient": c} for mask, c in sorted(poly.items())
        ],
        "controls": controls,
        "cases": cases,
        "structure": structures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.cases_output:
        args.cases_output.write_text("\n".join(manifest) + "\n")
    print(
        f"Verified {len(cases)} corrections; {len(poly)} cubic terms; "
        f"{len(controls['cz'])} CZ controls"
    )


if __name__ == "__main__":
    main()
