"""Prepare a native complete-attempt experiment for reconstructed folded MSC."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import stim
from audit_folded_contraction import pauli, pauli_text
from compiled_gadget_contraction import write_native_plan
from folded_check_contraction import FoldedChecks
from folded_msc_family import make_circuit
from validate_folded_msc import noise_labels


def monomial_plan(operations, kernel, slots, noise_indices, stages, verification_rules=None):
    """Certify branch-independent records and precompute code translations."""
    n = kernel.n
    ancillas = {3: 3, 5: 6, 7: 14}[kernel.surface.distance]
    bits = [1 << (3 * stages + i) for i in range(ancillas)]
    flips = [0] * n
    next_symbol = 3 * stages + ancillas
    noise_symbols = {}
    pending = None
    expected = {
        op.stage: sum(item.name == "H" for _, item in operations if item.stage == op.stage)
        for _, op in operations
    }
    if any(count not in (2, 3) for count in expected.values()):
        raise ValueError("unsupported number of cat Hadamards")
    current = None
    stage = -1
    hadamards = 0
    events = []
    branch_mask = (1 << stages) - 1
    for index, op in operations:
        if op.stage != current:
            if current is not None and (hadamards != expected[current] or pending is not None):
                raise ValueError("incomplete cat Hadamard pair")
            current, stage, hadamards = op.stage, stage + 1, 0
        q = op.qubits[0]
        if op.probability is not None:
            events.append((9, noise_indices[index], 0, 0))
            for target_index, target in enumerate(op.qubits):
                noise_symbols[next_symbol] = (noise_indices[index], target_index)
                if target < n:
                    flips[target] ^= 1 << next_symbol
                else:
                    bits[target - n] ^= 1 << next_symbol
                next_symbol += 1
        elif op.name in {"R", "M"}:
            if op.name == "M" and pending is not None:
                placeholder, event_index, root_qubit = pending
                expression = bits[q - n]
                if not (expression & placeholder) or expression & branch_mask != 1 << stage:
                    raise ValueError("verification outcome does not eliminate one cat branch")
                uniform = 1 << (2 * stages + stage)
                # Conditioning the first verification readout removes its GHZ
                # branch; the surviving data-cat branches remain coherent.
                replacement = expression ^ placeholder ^ uniform
                noise = replacement ^ (1 << stage) ^ uniform
                symbols = [i for i in range(noise.bit_length()) if noise >> i & 1]
                if verification_rules is None or any(i not in noise_symbols for i in symbols):
                    raise ValueError("unsupported verification dependency")
                events[event_index] = (10, root_qubit, stages - 1 - stage, len(verification_rules))
                verification_rules.append(
                    dict(branch_bit=stages - 1 - stage, noise=[noise_symbols[i] for i in symbols])
                )
                bits = [
                    value ^ placeholder ^ replacement if value & placeholder else value
                    for value in bits
                ]
                flips = [
                    value ^ placeholder ^ replacement if value & placeholder else value
                    for value in flips
                ]
                pending = None
            if q < n or bits[q - n] & branch_mask:
                raise ValueError("cat branch leaks into a classical record")
            events.append((0 if op.name == "R" else 3, q, slots[index], 0))
            if op.name == "R":
                bits[q - n] = 0
        elif op.name == "H":
            if q < n or hadamards >= expected[current]:
                raise ValueError("unsupported cat Hadamard")
            if hadamards == 1 and expected[current] == 3:
                placeholder = 1 << next_symbol
                next_symbol += 1
                pending = (placeholder, len(events), q)
                events.append((10, q, stages - 1 - stage, 0))
                bits[q - n] = placeholder
            else:
                closing = hadamards == expected[current] - 1
                events.append((2 if closing else 1, q, stages - 1 - stage, 0))
                bits[q - n] = 1 << (stage + (stages if closing else 0))
            hadamards += 1
        elif op.name in {"T", "T_DAG"}:
            if q >= n:
                raise ValueError("expected data T factor")
            events.append((4 if op.name == "T" else 5, q, 0, 0))
        elif op.name == "CX":
            target = op.qubits[1]
            if q < n:
                raise ValueError("expected ancilla control")
            events.append((7 if target < n else 6, q, target, 0))
            if target < n:
                flips[target] ^= bits[q - n]
            else:
                bits[target - n] ^= bits[q - n]
        elif op.name == "CCZ":
            pair = tuple(op.qubits[1:])
            if q < n or pair not in kernel.pair_index:
                raise ValueError("unexpected folded CZ support")
            events.append((8, q, kernel.pair_index[pair], 0))
        else:
            raise ValueError(f"unsupported folded operation {op.name}")
    if stage + 1 != stages or hadamards != expected[current] or pending is not None:
        raise ValueError("incorrect folded stage count")
    if any(value & branch_mask for value in bits):
        raise ValueError("coherent cat branch survives in an unmeasured ancilla")
    shifts = []
    for branch in range(1 << stages):
        delta = 0
        for j in range(stages):
            if branch >> (stages - 1 - j) & 1:
                delta ^= sum(((f >> j) & 1) << q for q, f in enumerate(flips))
        shift = sum(((delta & dual).bit_count() % 2) << i for i, dual in enumerate(kernel.x_duals))
        restored = 0
        for i, row in enumerate(kernel.x_rows + [kernel.logical_x]):
            if shift >> i & 1:
                restored ^= row
        if restored != delta:
            raise ValueError("branch permutation leaves the computational code coset")
        shifts.append(shift)
    return events, shifts


def css_effect(operations, position, targets, label, kernel, slots, width):
    frame = stim.PauliString(width)
    for q, axis in zip(targets, label, strict=True):
        frame[q] = axis
    flipped = []
    for index, op in operations[position + 1 :]:
        if op.probability is not None:
            continue
        if op.name in {"H", "CX"}:
            frame = frame.after(stim.CircuitInstruction(op.name, op.qubits))
        elif op.name in {"M", "R"}:
            q = op.qubits[0]
            if frame[q] in (1, 2):
                flipped.append(slots[index])
            if op.name == "R":
                frame[q] = "I"
        else:
            raise ValueError("non-Clifford operation in CSS continuation")
    x = sum((frame[q] in (1, 2)) << q for q in range(kernel.n))
    z = sum((frame[q] in (2, 3)) << q for q in range(kernel.n))
    anc = sum((frame[q] in (1, 2)) << (q - kernel.n) for q in range(kernel.n, width))
    return x, z, anc, flipped


def compile_protocol(circuit, distance, directory, *, native_growth=False, body_only=False):
    if native_growth and (distance != 7 or body_only):
        raise ValueError("native growth requires the complete f7 protocol")
    if body_only and distance != 5:
        raise ValueError("only the f5 intermediate block is supported")
    kernel = FoldedChecks(distance)
    if kernel.n > 127:
        raise ValueError("native physical masks currently support at most 127 data qubits")
    body_stages = (
        (f"d{distance}_logical_check_1", f"d{distance}_logical_check_2")
        if distance > 3
        else ("logical_check_1", "logical_check_2")
    )
    post_stages = (
        (f"d{distance}_post_checks", "final_syndrome")
        if distance > 3
        else ("post_checks", "final_syndrome")
    )
    ops = list(enumerate(circuit.operations))
    start = next(i for i, op in ops if op.stage == body_stages[0])
    prefix_ops = circuit.operations[:start]
    body = [(i, op) for i, op in ops if op.stage in body_stages]
    post = [(i, op) for i, op in ops if op.stage in post_stages]
    final = [(i, op) for i, op in ops if op.stage == "final_logical"]
    if [i for i, _ in body + post + final] != list(range(start, len(ops))):
        raise ValueError("unsupported protocol stage sequence")
    visible = len(circuit.measurements)
    hidden = sum(op.name == "R" for _, op in ops)
    width = circuit.manifest()["num_qubits"]
    slots = {}
    m, r = 0, visible
    for i, op in ops:
        if op.name == "M":
            slots[i], m = m, m + 1
        elif op.name == "R":
            slots[i], r = r, r + 1
    prefix_visible = sum(op.name == "M" for op in prefix_ops)
    prefix_hidden = sum(op.name == "R" for op in prefix_ops)
    prefix = replace(
        circuit, operations=prefix_ops, measurements=circuit.measurements[:prefix_visible]
    )
    logical_x = pauli("X", kernel.logical_x, width)
    logical_z = pauli("Z", kernel.logical_z, width)
    probes = [
        pauli(axis, row, width)
        for axis, rows in (("X", kernel.x_rows), ("Z", kernel.z_rows))
        for row in rows
    ]
    probes += [logical_x, 1j * logical_x * logical_z, logical_z]
    probes += [pauli("Z", 1 << q, width) for q in range(kernel.n, width)]
    sites = []
    for i, op in ops[start:]:
        if op.probability is not None:
            sites.append(
                dict(
                    operation=i,
                    probability=op.probability,
                    targets=list(op.qubits),
                    labels=list(noise_labels(op)),
                )
            )
    noise_indices = {site["operation"]: j for j, site in enumerate(sites)}
    verification_rules: list[dict] = []
    body_events, body_shifts = monomial_plan(
        body, kernel, slots, noise_indices, 2, verification_rules
    )
    if body_only:
        if post or final:
            raise ValueError("intermediate folded block has a terminal continuation")
        final_events, final_shifts = [], [0, 0]
    else:
        final_events, final_shifts = monomial_plan(
            final, kernel, slots, noise_indices, 1, verification_rules
        )
    post_events = []
    check = 0
    for position, (i, op) in enumerate(post):
        if op.probability is not None:
            sites[noise_indices[i]]["effects"] = [
                css_effect(post, position, op.qubits, label, kernel, slots, width)
                for label in noise_labels(op)
            ]
        elif op.name in {"M", "R"}:
            if op.qubits[0] < kernel.n:
                raise ValueError("CSS extraction cannot measure data directly")
            if op.name == "R":
                kind, value = 2, 0
            else:
                j = check % (2 * kernel.rank)
                kind, value = (0, j) if j < kernel.rank else (1, kernel.z_rows[j - kernel.rank])
                check += 1
            post_events.append((kind, op.qubits[0] - kernel.n, slots[i], value))
    if check != (0 if body_only else 4 * kernel.rank):
        raise ValueError("expected noisy and final CSS extraction")
    # Verify the noiseless extraction supports and order against the code.
    for stage in post_stages:
        stage_ops = [op for _, op in post if op.stage == stage and op.probability is None]
        groups = []
        current = []
        for op in stage_ops:
            current.append(op)
            if op.name == "M":
                groups.append(current)
                current = []
        for j, group in enumerate(groups):
            support = kernel.surface.checks("X" if j < kernel.rank else "Z")[j % kernel.rank]
            anc = group[0].qubits[0]
            expected: list[tuple[str, tuple[int, ...]]] = [("R", (anc,))]
            if j < kernel.rank:
                expected += [("H", (anc,))]
            expected += [("CX", (anc, q) if j < kernel.rank else (q, anc)) for q in support]
            if j < kernel.rank:
                expected += [("H", (anc,))]
            expected += [("M", (anc,))]
            if [(op.name, op.qubits) for op in group] != expected:
                raise ValueError("unrecognized sequential CSS extraction")
    directory.mkdir(parents=True, exist_ok=True)
    if not native_growth:
        (directory / "growth.txt").unlink(missing_ok=True)
    (directory / "full.stim").write_text(circuit.text())
    (directory / "prefix.stim").write_text(
        prefix.text() + "".join(f"EXP_VAL {pauli_text(p)}\n" for p in probes)
    )
    (directory / "dimensions.txt").write_text(
        f"{kernel.rank} {len(kernel.masks)} {kernel.characters}\n"
    )
    write_native_plan(directory / "amplitude.txt", kernel.amplitude_plan, [], marginal=True)
    for i, plan in enumerate(kernel.marginals):
        write_native_plan(directory / f"marginal_{i}.txt", plan, [], marginal=True)
    prefix_sites = [
        dict(operation=i, labels=list(noise_labels(op)))
        for i, op in ops[:start]
        if op.probability is not None
    ]
    with (directory / "protocol.txt").open("w") as output:

        def put(*values):
            output.write(" ".join(map(str, values)) + "\n")

        put(
            kernel.n,
            width - kernel.n,
            visible,
            hidden,
            prefix_visible,
            prefix_hidden,
            sum(m["postselect"] for m in prefix.measurements),
            kernel.logical_x,
            kernel.logical_z,
        )
        for rows in (kernel.x_rows, kernel.z_rows, kernel.z_duals[:-1]):
            put(*rows)
        put(len(kernel.pairs))
        for pair in kernel.pairs:
            put(*pair)
        put(len(sites))
        for site in sites:
            put(site["probability"], len(site["targets"]), *site["targets"], len(site["labels"]))
            for label in site["labels"]:
                put(*("IXYZ".index(a) for a in label))
            effects = site.get("effects", [])
            put(len(effects))
            for x, z, anc, flipped in effects:
                put(x, z, anc, len(flipped), *flipped)
        for events, shifts in (
            (body_events, body_shifts),
            (final_events, final_shifts),
        ):
            put(len(shifts), *shifts, len(events))
            for event in events:
                put(*event)
        put(len(post_events))
        for event in post_events:
            put(*event)
        postselect = [m["index"] for m in circuit.measurements if m["postselect"]]
        logical_slots = [m["index"] for m in circuit.measurements if m["logical"]]
        if body_only and not logical_slots:
            logical_slots = [0]
        if len(logical_slots) != 1:
            raise ValueError("expected one final logical output")
        put(len(postselect), *postselect, logical_slots[0])
        put(len(verification_rules))
        for rule in verification_rules:
            put(rule["branch_bit"], len(rule["noise"]))
            for pair in rule["noise"]:
                put(*pair)
        partners = [m.get("parity_with", visible) for m in circuit.measurements if m["postselect"]]
        put(*partners)
    metadata = dict(
        distance=distance,
        circuit_sha256=hashlib.sha256(circuit.text().encode()).hexdigest(),
        prefix_visible=prefix_visible,
        prefix_hidden=prefix_hidden,
        visible=visible,
        hidden=hidden,
        prefix_sites=prefix_sites,
        sites=sites,
        postselect=postselect,
        detector_partners=partners,
        verification_rules=verification_rules,
        logical_slot=logical_slots[0],
        peak_scope=max(p.peak_scope for p in kernel.marginals),
    )
    if native_growth:
        from compile_folded_growth import add_native_growth

        add_native_growth(circuit, directory, metadata, kernel)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-growth", action="store_true")
    args = parser.parse_args()
    compile_protocol(
        make_circuit(args.probability, distance=args.distance),
        args.distance,
        args.output,
        native_growth=args.native_growth,
    )


if __name__ == "__main__":
    main()
