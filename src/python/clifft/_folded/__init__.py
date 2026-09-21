"""Experimental recognition of certified terminal folded MSC regions."""

from dataclasses import replace

from .pattern import terminal_pattern

_NOISE = {"X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "DEPOLARIZE3"}
_OUTPUT = {"DETECTOR", "OBSERVABLE_INCLUDE"}
_IGNORE = {"QUBIT_COORDS", "SHIFT_COORDS", "TICK"}


def node_text(node):
    args = "" if not node.args else "(" + ",".join(f"{a:.17g}" for a in node.args) + ")"
    return node.gate.name + args + " " + " ".join(map(repr, node.targets))


def recognize(text, parse):
    circuit = parse(text)
    indexed = [
        (i, node) for i, node in enumerate(circuit.nodes) if node.gate.name not in _OUTPUT | _IGNORE
    ]
    if any(
        n.gate.name == "EXP_VAL"
        or n.tag
        or any(t.is_rec or t.is_inverted or t.has_pauli for t in n.targets)
        for _, n in indexed
    ):
        return None, "unsupported operation or target in folded candidate"
    for distance in (3, 5, 7):
        pattern = terminal_pattern(distance)
        expanded = parse("\n".join(op.text() for op in pattern)).nodes
        quantum = [n for _, n in indexed if n.gate.name not in _NOISE]
        expected = [n for n in expanded if n.gate.name not in _NOISE]
        if len(quantum) < len(expected):
            continue
        mapping: dict[int, int] = {}
        inverse: dict[int, int] = {}
        valid = True
        for a, b in zip(expected, quantum[-len(expected) :], strict=True):
            if a.gate != b.gate or a.args != b.args or len(a.targets) != len(b.targets):
                valid = False
                break
            for x, y in zip(a.targets, b.targets, strict=True):
                if (
                    mapping.setdefault(x.value, y.value) != y.value
                    or inverse.setdefault(y.value, x.value) != x.value
                ):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        if any(t.value not in inverse for _, node in indexed for t in node.targets):
            continue
        first_quantum = len(quantum) - len(expected)
        seen = 0
        begin = None
        for j, (_, node) in enumerate(indexed):
            if node.gate.name not in _NOISE:
                if seen == first_quantum:
                    begin = j
                    break
                seen += 1
        assert begin is not None
        actual = indexed[begin:]
        cursor = 0
        probabilities = {}
        for node in expanded:
            if node.gate.name in _NOISE:
                if cursor == len(actual) or actual[cursor][1].gate.name not in _NOISE:
                    continue
                other = actual[cursor][1]
                if node.gate != other.gate or [mapping[t.value] for t in node.targets] != [
                    t.value for t in other.targets
                ]:
                    valid = False
                    break
                probabilities[node.source_line - 1] = other.arg
            else:
                if cursor == len(actual):
                    valid = False
                    break
                other = actual[cursor][1]
                if (
                    node.gate != other.gate
                    or node.args != other.args
                    or [mapping[t.value] for t in node.targets] != [t.value for t in other.targets]
                ):
                    valid = False
                    break
            cursor += 1
        if not valid or cursor != len(actual):
            continue
        operations = [
            replace(op, probability=probabilities[i]) if op.probability is not None else op
            for i, op in enumerate(pattern)
            if op.probability is None or i in probabilities
        ]
        stop = actual[0][0]
        prefix_nodes = [node for i, node in indexed if i < stop]
        # Resets in the prefix must have the same hidden-record accounting as the compiler.
        if any(n.gate.name in {"RX", "RY", "MR", "MRX", "MRY"} for n in prefix_nodes):
            continue
        prefix_visible = sum(n.gate.name in {"M", "MX", "MY"} for n in prefix_nodes)
        prefix_hidden = sum(n.gate.name == "R" for n in prefix_nodes)
        hidden = sum(n.gate.name == "R" for _, n in indexed)
        from .compile import compile_region

        compiled = compile_region(
            distance,
            operations,
            prefix_visible,
            prefix_hidden,
            circuit.num_measurements,
            hidden,
            mapping,
        )
        compiled.update(
            prefix="\n".join(node_text(n) for n in prefix_nodes) + "\n" + compiled["probes"],
            distance=distance,
        )
        return compiled, f"specialized terminal folded MSC region at distance {distance}"
    return None, "no supported terminal folded MSC gate pattern"
