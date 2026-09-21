"""Certify and lower a structurally matched terminal folded region."""

import itertools
from typing import Any

from .contraction import FoldedChecks, kernel_plan


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


def noise_labels(op):
    if op.name == "X_ERROR":
        return ("X",)
    if op.name not in {"DEPOLARIZE1", "DEPOLARIZE2", "DEPOLARIZE3"}:
        raise ValueError("unsupported folded noise channel")
    return tuple("".join(p) for p in itertools.product("IXYZ", repeat=len(op.qubits)))[1:]


def css_effect(operations, position, targets, label, kernel, slots):
    x = sum((p in "XY") << q for p, q in zip(label, targets, strict=True))
    z = sum((p in "YZ") << q for p, q in zip(label, targets, strict=True))
    flipped = []
    for index, op in operations[position + 1 :]:
        if op.probability is not None:
            continue
        q = op.qubits[0]
        if op.name == "H":
            change = ((x ^ z) >> q & 1) << q
            x, z = x ^ change, z ^ change
        elif op.name == "CX":
            target = op.qubits[1]
            x ^= ((x >> q) & 1) << target
            z ^= ((z >> target) & 1) << q
        elif op.name in {"M", "R"}:
            if x >> q & 1:
                flipped.append(slots[index])
            if op.name == "R":
                x, z = x & ~(1 << q), z & ~(1 << q)
        else:
            raise ValueError("unsupported operation in folded CSS continuation")
    data_mask = (1 << kernel.n) - 1
    return x & data_mask, z & data_mask, x >> kernel.n, flipped


def compile_region(distance, operations, prefix_visible, prefix_hidden, visible, hidden, mapping):
    kernel = FoldedChecks(distance)
    slots = {}
    m, r = prefix_visible, visible + prefix_hidden
    for i, op in enumerate(operations):
        if op.name == "M":
            slots[i], m = m, m + 1
        elif op.name == "R":
            slots[i], r = r, r + 1
    if m != visible or r != visible + hidden:
        raise ValueError("folded region record dimensions disagree")
    sites = [
        dict(operation=i, probability=op.probability, targets=op.qubits, labels=noise_labels(op))
        for i, op in enumerate(operations)
        if op.probability is not None
    ]
    indices = {site["operation"]: i for i, site in enumerate(sites)}
    body = [(i, op) for i, op in enumerate(operations) if op.stage in {"check_1", "check_2"}]
    post = [(i, op) for i, op in enumerate(operations) if op.stage in {"post", "final_syndrome"}]
    final = [(i, op) for i, op in enumerate(operations) if op.stage == "final"]
    rules: list[dict[str, Any]] = []
    blocks = [
        monomial_plan(body, kernel, slots, indices, 2, rules),
        monomial_plan(final, kernel, slots, indices, 1, rules),
    ]
    post_events = []
    check = 0
    for position, (i, op) in enumerate(post):
        if op.probability is not None:
            sites[indices[i]]["effects"] = [
                css_effect(post, position, op.qubits, label, kernel, slots)
                for label in noise_labels(op)
            ]
        elif op.name in {"M", "R"}:
            if op.name == "R":
                kind, value = 2, 0
            else:
                j = check % (2 * kernel.rank)
                kind, value = (0, j) if j < kernel.rank else (1, kernel.z_rows[j - kernel.rank])
                check += 1
            post_events.append((kind, op.qubits[0] - kernel.n, slots[i], value))
    if check != 4 * kernel.rank:
        raise ValueError("incomplete folded CSS continuation")
    tokens: list[str] = []

    def put(*values):
        tokens.extend(map(str, values))

    put(1)
    put(
        kernel.n,
        {3: 3, 5: 6, 7: 14}[distance],
        visible,
        hidden,
        prefix_visible,
        prefix_hidden,
        kernel.logical_x,
        kernel.logical_z,
    )
    for rows in (kernel.x_rows, kernel.z_duals[:-1]):
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
    for events, shifts in blocks:
        put(len(shifts), *shifts, len(events))
        for event in events:
            put(*event)
    put(len(post_events))
    for event in post_events:
        put(*event)
    put(len(rules))
    for rule in rules:
        put(rule["branch_bit"], len(rule["noise"]))
        for pair in rule["noise"]:
            put(*pair)
    probes: list[str] = []
    for axis, rows in (("X", kernel.x_rows), ("Z", kernel.z_rows)):
        probes.extend(
            "*".join(f"{axis}{mapping[q]}" for q in range(kernel.n) if row >> q & 1) for row in rows
        )
    for x, z in (
        (kernel.logical_x, 0),
        (kernel.logical_x, kernel.logical_z),
        (0, kernel.logical_z),
    ):
        probes.append(
            "*".join(
                f"{'IXZY'[((x >> q) & 1) | (((z >> q) & 1) << 1)]}{mapping[q]}"
                for q in range(kernel.n)
                if (x | z) >> q & 1
            )
        )
    probes.extend(f"Z{mapping[q]}" for q in range(kernel.n, len(mapping)))
    return dict(
        block="\n".join(tokens) + "\n",
        contractions=[kernel_plan(kernel.amplitude_plan)]
        + [kernel_plan(p) for p in kernel.marginals],
        dimensions=[kernel.rank, len(kernel.masks), kernel.characters],
        probes="".join(f"EXP_VAL {p}\n" for p in probes),
    )
