"""Export a certified fixed growth transfer between native folded kernels."""

from dataclasses import replace

from folded_check_contraction import FoldedChecks
from folded_growth import GrowthHandoff
from folded_msc_family import Surface


def add_native_growth(circuit, directory, metadata, large_kernel):
    from compile_folded_protocol import compile_block

    bridge = GrowthHandoff()
    start = next(
        i for i, op in enumerate(circuit.operations) if op.stage == "grow_regular5_to_regular7"
    )
    stop = next(i for i, op in enumerate(circuit.operations) if op.stage == "d7_logical_check_1")
    physical = circuit.operations[start:stop]
    if len(physical) != len(bridge.operations) or any(
        replace(a, probability=b.probability) != b
        for a, b in zip(bridge.operations, physical, strict=True)
    ):
        raise ValueError("physical growth differs from the certified bridge")
    inverse = {q: i for i, q in bridge.mapping.items()}
    inverse.update({85 + i: 41 + i for i in range(6)})
    small_operations = [
        replace(op, qubits=tuple(inverse[q] for q in op.qubits))
        for op in circuit.operations[:start]
    ]
    small_visible = sum(op.name == "M" for op in small_operations)
    small = replace(
        circuit,
        operations=small_operations,
        measurements=circuit.measurements[:small_visible],
        coordinates=Surface(5).points,
        protocol="f5_intermediate",
    )
    small_directory = directory / "f5"
    small_metadata = compile_block(small, FoldedChecks(5), small_directory, terminal=False)
    (directory / "prefix.stim").write_text((small_directory / "prefix.stim").read_text())
    visible, hidden = metadata["visible"], metadata["hidden"]
    slots = {}
    m, r = 0, visible
    for i, op in enumerate(circuit.operations):
        if op.name == "M":
            slots[i], m = m, m + 1
        elif op.name == "R":
            slots[i], r = r, r + 1
    records = [slots[start + i] for i, op in enumerate(physical) if op.name in {"M", "R"}]
    bridge_sites = []
    small_surface = Surface(5)
    source_z = [sum(1 << q for q in row) for row in small_surface.checks("Z")]
    with (directory / "growth.txt").open("w") as output:

        def put(*values):
            output.write(" ".join(map(str, values)) + "\n")

        def packed(value):
            return [
                (value >> (64 * i)) & ((1 << 64) - 1)
                for i in range((len(bridge.relations) + 63) // 64)
            ]

        put(2, visible, hidden)
        put(
            len(small_surface.points),
            large_kernel.n,
            len(source_z),
            large_kernel.rank,
            len(bridge.ancillas),
        )
        put(len(records), *records)
        put(*source_z)
        put(*large_kernel.z_duals[:-1])
        put(len(bridge.relations))
        for mask, sign in bridge.relations:
            put(mask, sign)
        put(len(bridge.sites))
        for position, channels in bridge.sites.items():
            op = physical[position]
            put(op.probability, len(channels))
            for effect in channels.values():
                put(*packed(effect))
            bridge_sites.append(dict(operation=start + position, labels=list(channels)))
    metadata.update(
        native_growth=True,
        prefix_stage="d5_logical_check_1",
        prefix_visible=small_metadata["prefix_visible"],
        prefix_hidden=small_metadata["prefix_hidden"],
        prefix_sites=small_metadata["prefix_sites"],
        earlier_sites=small_metadata["sites"],
        growth_sites=bridge_sites,
    )
    partition = [
        site
        for key in ("prefix_sites", "earlier_sites", "growth_sites", "sites")
        for site in metadata[key]
    ]
    expected = {i for i, op in enumerate(circuit.operations) if op.probability is not None}
    if len(partition) != len(expected) or {site["operation"] for site in partition} != expected:
        raise ValueError("native growth does not partition the physical fault sites")
