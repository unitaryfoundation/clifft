"""A local CSS-dual growth encoder for the full cultivation reconstruction."""

from __future__ import annotations

from fold_cultivation import Operation, Reconstruction
from fold_schedule import reconstruction_with_schedule

GROWTH_ENCODERS = ("original", "dual_rotate")


def reconstruction_with_growth(
    distance: int, encoder: str, schedule: str = "original"
) -> Reconstruction:
    if encoder not in GROWTH_ENCODERS:
        raise ValueError("unknown growth encoder")
    reconstruction = reconstruction_with_schedule(distance, schedule)
    if encoder == "original" or distance != 7:
        return reconstruction
    stage = next(s for s in reconstruction.stages if s.name == "grow_regular_d5_to_regular_d7")
    positions = {q: xy for xy, q in reconstruction.index.items()}

    def rotate(q: int) -> int:
        x, y = positions[q]
        return reconstruction.index[2 * distance - 2 - y, x]

    fresh = {op.targets[0] for op in stage.operations if op.name == "R"}
    plus = {op.targets[0] for op in stage.operations if op.name == "H"}
    # Global Hadamard conjugation exchanges fresh zero/plus preparations and
    # reverses CNOTs. Rotation restores this layout's X/Z stabilizer convention.
    stage.operations = (
        [Operation("R", (rotate(op.targets[0]),)) for op in stage.operations if op.name == "R"]
        + [Operation("H", (rotate(q),)) for q in sorted(fresh - plus)]
        + [
            Operation("CX", (rotate(op.targets[1]), rotate(op.targets[0])))
            for op in stage.operations
            if op.name == "CX"
        ]
    )
    return reconstruction


def growth_fault_histories(
    reconstruction: Reconstruction,
) -> list[tuple[str, list[list[Operation]]]]:
    """Sample fault locations across all four growth layers and fresh preparations."""
    index = next(
        j for j, s in enumerate(reconstruction.stages) if s.name == "grow_regular_d5_to_regular_d7"
    )
    operations = reconstruction.stages[index].operations
    cnots = [j for j, op in enumerate(operations) if op.name == "CX"]
    selected = []
    start = 0
    for size in (22, 22, 18, 18):
        selected += [cnots[start + j] for j in (0, size // 2, size - 1)]
        start += size
    if start != len(cnots):
        raise ValueError("growth fault strata require the four-layer encoder")
    for name in ("R", "H"):
        sites = [j for j, op in enumerate(operations) if op.name == name]
        selected += [sites[j] for j in (0, len(sites) // 2, len(sites) - 1)]
    cases = []
    for site in selected:
        operation = operations[site]
        for q in operation.targets:
            for axis in "X" if operation.name == "R" else "XYZ":
                stages = [list(stage.operations) for stage in reconstruction.stages]
                stages[index].insert(site + 1, Operation(axis, (q,)))
                cases.append((f"growth_gate_{site}_q{q}_{axis}", stages))
    return cases
