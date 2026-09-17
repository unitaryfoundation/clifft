"""Offline certificates and controlled variants of syndrome-extraction schedules."""

from __future__ import annotations

import stim
from fold_cultivation import Operation, Reconstruction, pauli, stabilizers

SCHEDULES = ("original", "reverse_support", "reverse_checks")


def measurement_chunks(operations: list[Operation]) -> list[list[Operation]]:
    chunks: list[list[Operation]] = []
    current: list[Operation] = []
    for operation in operations:
        current.append(operation)
        if operation.name == "M":
            chunks.append(current)
            current = []
    if current:
        raise ValueError("syndrome has operations after its last measurement")
    return chunks


def reconstruction_with_schedule(distance: int, schedule: str) -> Reconstruction:
    if schedule not in SCHEDULES:
        raise ValueError("unknown syndrome schedule")
    reconstruction = Reconstruction(distance).build()
    if schedule == "original":
        return reconstruction
    for stage in reconstruction.stages:
        if not stage.name.startswith("syndrome"):
            continue
        chunks = measurement_chunks(stage.operations)
        if schedule == "reverse_checks":
            chunks.reverse()
        for chunk in chunks:
            indices = [j for j, op in enumerate(chunk) if op.name == "CX"]
            reversed_gates = [chunk[j] for j in reversed(indices)]
            for j, gate in zip(indices, reversed_gates, strict=True):
                chunk[j] = gate
        stage.operations = [op for chunk in chunks for op in chunk]
    return reconstruction


def certify_syndrome(
    operations: list[Operation], mapping: list[int], distance: int, width: int
) -> list[stim.PauliString]:
    """Verify each entire ancilla gadget is exactly a canonical Pauli measurement.

    Pulling back the readout alone is insufficient: a circuit could measure
    the right observable and also apply an unwanted unitary to the data.
    Equality of the full Clifford tableaux rules out that extra action.
    """
    canonical = [pauli(distance, axis, support) for axis, support in stabilizers(distance)]
    remaining = set(map(str, canonical))
    data = set(mapping)
    result = []
    for chunk in measurement_chunks(operations):
        if len(chunk) < 2 or chunk[0].name != "R" or len(chunk[0].targets) != 1:
            raise ValueError("syndrome gadget must begin with one fresh ancilla reset")
        ancilla = chunk[0].targets[0]
        if ancilla in data or chunk[-1] != Operation("M", (ancilla,)):
            raise ValueError("syndrome gadget must measure its own non-data ancilla")
        unitary = stim.Circuit(f"I {width - 1}")
        for op in chunk[1:-1]:
            if op.name not in {"H", "CX"} or any(q not in data | {ancilla} for q in op.targets):
                raise ValueError("unsupported syndrome unitary")
            unitary.append(op.name, op.targets)
        tableau = stim.Tableau.from_circuit(unitary)
        readout = stim.PauliString(width)
        readout[ancilla] = "Z"
        pulled = tableau.inverse()(readout)
        if pulled.sign != 1 or pulled[ancilla] != 3:
            raise ValueError("syndrome readout has unsupported sign or ancilla dependence")
        measured = stim.PauliString(len(mapping))
        for q, physical in enumerate(mapping):
            measured[q] = pulled[physical]
        if any(pulled[q] for q in range(width) if q not in data | {ancilla}):
            raise ValueError("syndrome readout touches an unrelated wire")
        if str(measured) not in remaining:
            raise ValueError("syndrome must measure each declared stabilizer exactly once")
        expected = stim.Circuit(f"I {width - 1}\nH {ancilla}")
        for q, physical in enumerate(mapping):
            if measured[q]:
                expected.append({1: "CX", 2: "CY", 3: "CZ"}[measured[q]], [ancilla, physical])
        expected.append("H", [ancilla])
        if tableau != stim.Tableau.from_circuit(expected):
            raise ValueError("syndrome gadget has an extra data or ancilla action")
        remaining.remove(str(measured))
        result.append(measured)
    if remaining:
        raise ValueError("syndrome does not measure a complete stabilizer basis")
    return result
