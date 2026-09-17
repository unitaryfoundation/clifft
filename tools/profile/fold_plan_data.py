"""Private data format for certified plans consumed by the reusable research worker."""

from __future__ import annotations

import struct

from fold_blocks import Fold, Protocol, project_mask
from fold_cultivation import coordinates, logical, pauli, stabilizers


class Writer:
    def __init__(self, protocol: Protocol):
        self.protocol = protocol
        self.data = bytearray(b"FLDBLK01")
        self.layouts = [protocol.prefix.layout]
        for _, plan, _ in protocol.groups:
            self.layouts += (
                [plan.preparation.layout, plan.core, plan.decode.layout]
                if isinstance(plan, Fold)
                else [plan.linear.layout]
            )
        self.offsets = {}
        offset = 0
        for layout in self.layouts:
            self.offsets[id(layout)] = offset
            offset += len(layout.slots)
        if offset > 8192:
            raise ValueError("fault capacity exceeded")
        self.fault_bits = offset

    def u(self, value):
        self.data.extend(struct.pack("<Q", int(value)))

    def mask(self, value):
        self.u(value & ((1 << 64) - 1))
        self.u(value >> 64)

    def seq(self, values, emit=None):
        self.u(len(values))
        for value in values:
            (emit or self.u)(value)

    def string(self, value):
        encoded = value.encode("ascii")
        self.u(len(encoded))
        self.data.extend(encoded)

    def mapping(self, rows, layout):
        self.u(len(rows))
        for row in rows:
            row <<= self.offsets[id(layout)]
            words = [(k, (row >> (64 * k)) & ((1 << 64) - 1)) for k in range(128)]
            words = [(k, v) for k, v in words if v]
            self.seq(words, lambda pair: [self.u(v) for v in pair])

    def linear(self, plan, selection):
        self.mapping([plan.x[q] for q in selection], plan.layout)
        self.mapping([plan.z[q] for q in selection], plan.layout)
        self.mapping(plan.records, plan.layout)

    def geometry(self, plan):
        for value in (
            plan.width,
            len(plan.x_masks),
            sum(len(leaf.multipliers[0]) for leaf in plan.leaves),
        ):
            self.u(value)
        self.mask(plan.logical_z)
        self.seq(plan.z_masks, self.mask)
        self.seq(plan.edges, lambda pair: [self.u(v) for v in pair])
        self.seq(plan.leaves, lambda leaf: [self.u(leaf.offset), self.u(len(leaf.multipliers[0]))])
        gathers: list[int] = []
        self.u(len(plan.steps))
        for step in plan.steps:
            for value in (step.offset, step.size, len(step.gathers), len(gathers)):
                self.u(value)
            gathers.extend(v for gather in step.gathers for v in gather)
        self.seq(gathers)
        self.seq([v for b in (0, 1) for leaf in plan.leaves for v in leaf.multipliers[b]])
        self.seq(plan.outputs)

    def fold(self, plan):
        self.geometry(plan.plan)
        self.u(len(plan.cats))
        self.mask(plan.equal_flag_mask)
        self.linear(plan.preparation, plan.cats)
        self.mapping(plan.decode.records, plan.decode.layout)
        actions = []
        offset = self.offsets[id(plan.core)]
        for boundary, faults in enumerate(plan.fault_actions):
            for cat, q, x, z in faults:
                # Zero denotes no fault bit; actual bit indices are stored plus one.
                actions.append(
                    [
                        int(cat),
                        q,
                        0,
                        0,
                        0,
                        offset + x.bit_length() if x else 0,
                        offset + z.bit_length() if z else 0,
                    ]
                )
            if boundary == len(plan.actions):
                break
            kind, targets = plan.actions[boundary]
            actions.append(
                [
                    {"T": 2, "T_DAG": 3, "CX": 4, "CZ": 5}[kind],
                    *targets,
                    *([0] * (4 - len(targets))),
                    0,
                    0,
                ]
            )
        self.seq(actions, lambda action: [self.u(v) for v in action])
        self.seq(list(plan.decode_table.flat), lambda v: self.u(int(v) + 1))

    def boundary(self, plan):
        self.geometry(plan.before)
        self.linear(plan.linear, plan.output_mapping)
        self.seq(plan.constraints, self.mask)
        self.seq(plan.syndrome_rows, self.mask)
        self.seq(plan.duals, lambda pair: [self.mask(v) for v in pair])
        self.seq(
            plan.transported,
            lambda pair: [self.mask(project_mask(v, plan.output_mapping)) for v in pair],
        )

    def write(self, path, cases=()):
        r = self.protocol.reconstruction
        self.u(r.distance)
        self.string(
            "\n".join(line for line in r.text(0.125).splitlines() if not line.startswith("EXP_VAL"))
            + "\n"
        )
        self.seq([str(pauli(r.distance, a, s)) for a, s in stabilizers(r.distance)], self.string)
        for axis in "XYZ":
            self.string(str(logical(r.distance, axis)))
        count, flags = 0, []
        for stage in r.stages:
            indices = []
            for op in stage.operations:
                if op.name == "M":
                    indices.append(count)
                    count += 1
            if stage.equal_records:
                if len(indices) != 6:
                    raise ValueError("unsupported flag group")
                flags.append(indices)
        self.seq(flags, lambda group: [self.u(v) for v in group])
        self.linear(self.protocol.prefix, self.protocol.input_mapping)
        self.u(len(self.protocol.groups))
        live = 1
        for _, plan, _ in self.protocol.groups:
            fold = isinstance(plan, Fold)
            live = live * 2 if fold else 1
            if live > 4:
                raise ValueError("branch capacity exceeded")
            self.u(int(fold))
            (self.fold if fold else self.boundary)(plan)
        sites = []
        for layout in self.layouts:
            offset = self.offsets[id(layout)]
            for j, (op, enabled) in enumerate(zip(layout.operations, layout.noisy, strict=True)):
                if not enabled:
                    continue
                boundary = j if op.name == "M" else j + 1
                x, z = [0] * 3, [0] * 3
                for k, q in enumerate(op.targets):
                    x[k] = offset + layout.slots[boundary, q, "X"] + 1
                    if (boundary, q, "Z") in layout.slots:
                        z[k] = offset + layout.slots[boundary, q, "Z"] + 1
                sites.append([len(op.targets), int(op.name in {"R", "M"}), *x, *z])
        self.seq(sites, lambda site: [self.u(v) for v in site])
        self.u(len(cases))
        for _, stages in cases:
            flat = [v for group in self.protocol.encode(stages) for v in group]
            bits = sum(
                v << self.offsets[id(layout)] for layout, v in zip(self.layouts, flat, strict=True)
            )
            for k in range(128):
                self.u((bits >> (64 * k)) & ((1 << 64) - 1))
            result = self.protocol.evaluate(self.protocol.encode(stages))
            expected = [result.acceptance] + [
                result.acceptance * v for v in (result.logical_xyz or [0, 0, 0])
            ]
            self.data.extend(struct.pack("<4d", *expected))
        path.write_bytes(self.data)
        return {
            "bytes": len(self.data),
            "fault_bits": self.fault_bits,
            "noise_locations": len(sites),
            "fixtures": len(cases),
            "data_width": len(coordinates(r.distance)),
        }
