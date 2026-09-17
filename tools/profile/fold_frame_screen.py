"""Offline certified tensor-storage bounds for coherent large-fold states.

Each stabilizer term has exact Schmidt ranks from binary linear algebra.
Adding those ranks bounds a coherent sum without replacing it by a mixture.
No dense large state, approximate truncation, or runtime frame policy is used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clifford_branches import INV_SQRT2, BranchState, add, multiply, root
from fold_check import Check, Event
from fold_cultivation import Operation, Reconstruction


def basis(rows):
    basis: dict[int, int] = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot in basis:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
    return list(basis.values())


def rank(rows):
    return len(basis(rows))


def generators(simulator):
    result = []
    for pauli in simulator.canonical_stabilizers():
        x, z = pauli.to_numpy()
        result.append(
            sum(int(v) << q for q, v in enumerate(x))
            | sum(int(v) << (q + len(pauli)) for q, v in enumerate(z))
        )
    return result


def cut_ranks(branches, width, left):
    mask = sum((1 << q) | (1 << (width + q)) for q in left)
    return [1 << (rank([row & mask for row in branch]) - len(left)) for branch in branches]


def profile(branches, width, order):
    if sorted(order) != list(range(width)) or not branches:
        raise ValueError("need a complete order and at least one coherent term")
    if any(len(branch) != width or rank(branch) != width for branch in branches):
        raise ValueError("expected pure stabilizer generators")
    upper, lower, individual = [], [], []
    for cut in range(1, width):
        ranks = cut_ranks(branches, width, order[:cut])
        upper.append(min(sum(ranks), 1 << min(cut, width - cut)))
        # Matrix rank subadditivity and its reverse give coefficient-independent
        # bounds for nonzero terms, including when their coherent sum is zero.
        lower.append(max(0, 2 * max(ranks) - sum(ranks)))
        individual.append(max(ranks))
    bonds = [1, *upper, 1]
    return {
        "bond_upper": upper,
        "bond_lower": lower,
        "max_bond_upper": max(bonds),
        "max_bond_lower": max([0, *lower]),
        "max_single_term_bond": max([1, *individual]),
        "mps_coefficients_upper": sum(
            2 * a * b for a, b in zip(bonds[:-1], bonds[1:], strict=True)
        ),
    }


def copy_state(state):
    result = BranchState(state.width)
    result.terms = [term.copy() for term in state.terms]
    return result


def phase_gate(state, op):
    if op.name not in {"T", "T_DAG"}:
        state.operation(op)
        return
    terms = []
    for source in state.terms:
        for bit in (0, 1):
            term = source.copy()
            if term.project(op.targets[0], bit):
                term.coefficient = multiply(term.coefficient, root(bit if op.name == "T" else -bit))
                terms.append(term)
    state.terms = terms
    state.merge()


def ccz_decomposition(targets):
    a, b, c = targets
    return [
        Operation(name, qs)
        for name, qs in (
            ("T", (a,)),
            ("T", (b,)),
            ("T", (c,)),
            ("CX", (a, b)),
            ("T_DAG", (b,)),
            ("CX", (a, b)),
            ("CX", (a, c)),
            ("T_DAG", (c,)),
            ("CX", (b, c)),
            ("T", (c,)),
            ("CX", (a, c)),
            ("T_DAG", (c,)),
            ("CX", (b, c)),
        )
    ]


def partial_fold(before, reconstruction, distance, operations):
    mapping = reconstruction.check_mapping(distance)
    data = len(mapping)
    cats = {3: 3, 5: 5, 7: 8}[distance]
    mapping += [reconstruction.ancilla + q for q in range(cats)]
    inverse = {q: j for j, q in enumerate(mapping)}
    events = [
        Event("CZ" if op.name == "CCZ" else op.name, tuple(inverse[q] for q in op.targets))
        for op in operations
    ]
    odd: set[int] = set()
    for op in operations:
        if op.name in {"T", "T_DAG"}:
            odd.symmetric_difference_update(op.targets)
    if len(odd) > 1:
        raise ValueError("partial fold exceeds the bounded unpaired-phase screen")
    # Complete the branch to a Clifford, then undo the added phase exactly.
    # This permits checkpoints inside a T/CX/T_DAG triple without dropping it.
    events += [Event("T", (inverse[q],)) for q in sorted(odd)]
    state = copy_state(before)
    state.fold(Check(data, cats, tuple(events)), mapping)
    for q in sorted(odd):
        phase_gate(state, Operation("T_DAG", (q,)))
    return state


@dataclass
class Snapshot:
    stage: int
    point: str
    state: BranchState


def snapshots(
    reconstruction: Reconstruction, history: list[list[Operation]]
) -> tuple[list[Snapshot], BranchState]:
    width = max(q for stage in history for op in stage for q in op.targets) + 1
    state = BranchState(width)
    result = []
    target = reconstruction.distance
    for index, (stage, operations) in enumerate(zip(reconstruction.stages, history, strict=True)):
        if stage.name.startswith("fold_core_"):
            distance = int(stage.name.rsplit("d", 1)[1])
            if distance == target:
                total = len(stage.operations)
                diagonal = 3 * (2 * distance - 1)
                pairs = total - diagonal
                points = sorted(
                    {
                        0,
                        1,
                        2,
                        3,
                        diagonal // 2 // 3 * 3,
                        diagonal,
                        *[diagonal + pairs * k // 4 for k in (1, 2, 3, 4)],
                    }
                )
                # Associate inserted Pauli faults with the preceding ideal gate.
                boundaries = (
                    [0]
                    + [j for j, op in enumerate(operations) if op.name not in {"X", "Y", "Z"}][1:]
                    + [len(operations)]
                )
                if len(boundaries) != total + 1:
                    raise ValueError("history does not match the ideal core")
                for point in points:
                    result.append(
                        Snapshot(
                            index,
                            f"gate_{point}",
                            partial_fold(
                                state, reconstruction, distance, operations[: boundaries[point]]
                            ),
                        )
                    )
                middle = diagonal + pairs // 2
                begin = boundaries[middle]
                op = operations[begin]
                if op.name != "CCZ":
                    raise ValueError("expected middle controlled fold pair")
                expanded = partial_fold(state, reconstruction, distance, operations[:begin])
                for k, gate in enumerate(ccz_decomposition(op.targets), 1):
                    phase_gate(expanded, gate)
                    if k in (3, 8, 13):
                        result.append(
                            Snapshot(index, f"ccz_{middle}_part_{k}", copy_state(expanded))
                        )
            state = partial_fold(state, reconstruction, distance, operations)
        else:
            for op in operations:
                state.operation(op)
            if stage.equal_records:
                for term in state.terms:
                    term.coefficient = multiply(term.coefficient, add(INV_SQRT2, INV_SQRT2))
        state.merge()
        if not state.terms:
            break
    return result, state


def framed_generators(snapshot, frame):
    branches = []
    for term in snapshot.state.terms:
        simulator = term.tableau.copy()
        if frame is not None:
            simulator.do_tableau(frame, list(range(snapshot.state.width)))
        branches.append(generators(simulator))
    return branches


def support_profile(snapshot, frame):
    rows, anchors, sizes = [], [], []
    width = snapshot.state.width
    for term in snapshot.state.terms:
        simulator = term.tableau.copy()
        simulator.do_tableau(frame, list(range(width)))
        x_rows = [row & ((1 << width) - 1) for row in generators(simulator)]
        rows += x_rows
        sizes.append(1 << rank(x_rows))
        anchor = 0
        for q in range(width):
            bit = simulator.peek_z(q) == -1
            simulator.postselect_z(q, desired_value=bit)
            anchor |= int(bit) << q
        anchors.append(anchor)
    rows += [anchor ^ anchors[0] for anchor in anchors[1:]]
    rows = basis(rows)
    dimension = len(rows)
    return {
        "affine_span_dimension": dimension,
        "affine_dense_coefficients": 1 << dimension,
        "sparse_coefficients_upper": min(sum(sizes), 1 << dimension),
        "basis": rows,
    }


def conjugated_support(frame, operations, order):
    weights, spans = [], []
    positions = {q: k for k, q in enumerate(order)}
    for op in operations:
        if op.name not in {"T", "T_DAG", "CCZ"}:
            continue
        for q in op.targets:
            pauli = frame.z_output(q)
            support = [positions[k] for k in range(len(pauli)) if pauli[k]]
            weights.append(len(support))
            spans.append(max(support) - min(support) + 1 if support else 0)
    return {"z_generator_weights": weights, "z_generator_spans": spans}


def multiple_hook_histories(reconstruction):
    """Legal deterministic fault strata that reach the large fold before rejecting."""
    index = next(
        i
        for i, stage in enumerate(reconstruction.stages)
        if stage.name == f"fold_core_d{reconstruction.distance}"
    )
    operations = reconstruction.stages[index].operations
    cczs = [i for i, op in enumerate(operations) if op.name == "CCZ"]
    cases = []
    for count in sorted({min(k, len(cczs)) for k in (2, 4, 8, 16, 24)}):
        selected = random.Random(8731 + count).sample(cczs, count)
        flips: set[tuple[int, int]] = set()
        for last in selected:
            control = operations[last].targets[0]
            first = max(i for i in range(last) if control in operations[i].targets)
            flips.symmetric_difference_update({(first, control), (last, control)})
        changed = []
        for i, op in enumerate(operations):
            changed.append(op)
            changed.extend(Operation("X", (q,)) for site, q in sorted(flips) if site == i)
        history = [list(stage.operations) for stage in reconstruction.stages]
        history[index] = changed
        cases.append((f"multiple_hooks_{count}_faults_{len(flips)}", history))
    return cases
