"""Generate a standalone native program from a certified offline block plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fold_blocks import Boundary, Fold, Protocol, project_mask, syndrome_sector_histories
from fold_contraction import Plan


class Exporter:
    def __init__(self, protocol: Protocol):
        self.protocol = protocol
        header = Path(__file__).with_name("fold_blocks_native.h").resolve()
        self.lines = [f'#include "{header}"', "using namespace fold_blocks;"]
        self.counter = 0
        self.offsets: dict[int, int] = {}
        self.layouts = [protocol.prefix.layout]
        for _, plan, _ in protocol.groups:
            self.layouts += (
                [plan.preparation.layout, plan.core, plan.decode.layout]
                if isinstance(plan, Fold)
                else [plan.linear.layout]
            )
        offset = 0
        for layout in self.layouts:
            self.offsets[id(layout)] = offset
            offset += len(layout.slots)
        if offset > 128 * 64:
            raise ValueError("native diagnostic fault capacity exceeded")
        self.fault_bits = offset
        self.geometries = {d: self.geometry(Plan(d)) for d in (3, 5)}

    def array(self, kind, values):
        name = f"data_{self.counter}"
        self.counter += 1
        self.lines.append(f"static const std::array<{kind}, {len(values)}> {name}{{{{")
        self.lines.extend(f"    {value}," for value in values)
        self.lines.append("}};")
        return name

    @staticmethod
    def mask(value):
        if not 0 <= value < 1 << 64:
            raise ValueError("native mask exceeds one word")
        return f"{value}ULL"

    def masks(self, values):
        return self.array("Mask", [self.mask(v) for v in values])

    def obj(self, kind, fields):
        name = f"object_{self.counter}"
        self.counter += 1
        self.lines.append(f"static const {kind} {name}{{{','.join(map(str, fields))}}};")
        return name

    def mapping(self, rows, layout):
        offset = self.offsets[id(layout)]
        words: list[str] = []
        spans: list[str] = []
        if len(rows) > 64:
            raise ValueError("native map result exceeds one word")
        for row in rows:
            row <<= offset
            start = len(words)
            index = 0
            while row:
                value = row & ((1 << 64) - 1)
                if value:
                    words.append(f"{{{index},{self.mask(value)}}}")
                row >>= 64
                index += 1
            spans.append(f"{{{start},{len(words) - start}}}")
        return self.obj("Map", [self.array("Row", spans), self.array("Word", words)])

    def linear(self, plan, selection):
        return self.obj(
            "Linear",
            [
                self.mapping([plan.x[q] for q in selection], plan.layout),
                self.mapping([plan.z[q] for q in selection], plan.layout),
                self.mapping(plan.records, plan.layout),
            ],
        )

    def geometry(self, plan):
        if plan.width > 41 or plan.storage > 603:
            raise ValueError("native diagnostic state capacity exceeded")
        leaves = self.array(
            "Leaf", [f"{{{leaf.offset},{len(leaf.multipliers[0])}}}" for leaf in plan.leaves]
        )
        flat: list[int] = []
        steps = []
        for step in plan.steps:
            steps.append(f"{{{step.offset},{step.size},{len(step.gathers)},{len(flat)}}}")
            flat.extend(v for gather in step.gathers for v in gather)
        return self.obj(
            "Geometry",
            [
                plan.width,
                len(plan.x_masks),
                sum(len(leaf.multipliers[0]) for leaf in plan.leaves),
                self.mask(plan.logical_z),
                self.masks(plan.z_masks),
                self.array("Edge", [f"{{{a},{b}}}" for a, b in plan.edges]),
                leaves,
                self.array("Step", steps),
                self.array("unsigned", flat),
                self.array(
                    "unsigned",
                    [v for b in (0, 1) for leaf in plan.leaves for v in leaf.multipliers[b]],
                ),
                self.array("unsigned", plan.outputs),
            ],
        )

    def boundary(self, plan):
        return self.obj(
            "Boundary",
            [
                "&" + self.geometries[plan.before.distance],
                self.linear(plan.linear, plan.output_mapping),
                self.masks(plan.constraints),
                self.masks(plan.syndrome_rows),
                self.array("Pair", [f"{{{self.mask(a)},{self.mask(b)}}}" for a, b in plan.duals]),
                self.array(
                    "Pair",
                    [
                        f"{{{self.mask(project_mask(a, plan.output_mapping))},"
                        f"{self.mask(project_mask(b, plan.output_mapping))}}}"
                        for a, b in plan.transported
                    ],
                ),
            ],
        )

    def fold(self, plan):
        actions = []
        offset = self.offsets[id(plan.core)]
        for boundary, faults in enumerate(plan.fault_actions):
            for cat, q, x, z in faults:
                xb = offset + x.bit_length() - 1 if x else -1
                zb = offset + z.bit_length() - 1 if z else -1
                actions.append(f"{{{int(cat)},{q},0,0,0,{xb},{zb}}}")
            if boundary == len(plan.actions):
                break
            kind, targets = plan.actions[boundary]
            code = {"T": 2, "T_DAG": 3, "CX": 4, "CZ": 5}[kind]
            fields = list(targets) + [0] * (4 - len(targets))
            actions.append("{" + ",".join(map(str, [code] + fields + [-1, -1])) + "}")
        return self.obj(
            "Fold",
            [
                "&" + self.geometries[plan.plan.distance],
                len(plan.cats),
                self.linear(plan.preparation, plan.cats),
                self.mapping(plan.decode.records, plan.decode.layout),
                self.array("Action", actions),
                self.array("int", [int(v) for v in plan.decode_table.flat]),
            ],
        )

    def write(self, cases, path):
        stage_initializers = []
        live = 1
        for _, plan, _ in self.protocol.groups:
            if isinstance(plan, Fold):
                live *= 2
                if live > 4:
                    raise ValueError("native diagnostic branch capacity exceeded")
                stage_initializers.append(f"{{&{self.fold(plan)},nullptr}}")
            elif isinstance(plan, Boundary):
                live = 1
                stage_initializers.append(f"{{nullptr,&{self.boundary(plan)}}}")
        noise = []
        for layout in self.layouts:
            offset = self.offsets[id(layout)]
            for j, (op, enabled) in enumerate(zip(layout.operations, layout.noisy, strict=True)):
                if not enabled:
                    continue
                boundary = j if op.name == "M" else j + 1
                x, z = [-1] * 3, [-1] * 3
                if len(op.targets) > 3:
                    raise ValueError("unsupported native noise arity")
                for k, q in enumerate(op.targets):
                    x[k] = offset + layout.slots[boundary, q, "X"]
                    if (boundary, q, "Z") in layout.slots:
                        z[k] = offset + layout.slots[boundary, q, "Z"]
                noise.append(
                    "{"
                    + f"{len(op.targets)},{int(op.name in {'R', 'M'})},"
                    + "{"
                    + ",".join(map(str, x))
                    + "},{"
                    + ",".join(map(str, z))
                    + "}}"
                )
        protocol = self.obj(
            "Protocol",
            [
                self.linear(self.protocol.prefix, self.protocol.input_mapping),
                self.array("Stage", stage_initializers),
                self.array("NoiseSite", noise),
            ],
        )
        fixtures = []
        for _, stages in cases:
            history = self.protocol.encode(stages)
            flat = [v for group in history for v in group]
            bits = sum(
                value << self.offsets[id(layout)]
                for layout, value in zip(self.layouts, flat, strict=True)
            )
            words = [
                self.mask((bits >> j) & ((1 << 64) - 1)) for j in range(0, self.fault_bits, 64)
            ]
            result = self.protocol.evaluate(history)
            expected = [result.acceptance] + [
                v * result.acceptance for v in (result.logical_xyz or [0, 0, 0])
            ]
            fixtures.append(
                "{{" + ",".join(words) + "},{" + ",".join(f"{v:.17g}" for v in expected) + "}}"
            )
        data = self.array("Fixture", fixtures)
        self.lines.append(
            "int main(int argc, char** argv) {\n"
            f"    return benchmark({protocol}, {data}, argc, argv);\n"
            "}"
        )
        path.write_text("\n".join(self.lines) + "\n")
        return {
            "fault_bits": self.fault_bits,
            "noise_locations": len(noise),
            "fixtures": len(cases),
        }


def main():
    from clifford_branches import logical_tail_history, materialize, paired_hook_histories

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = Protocol(args.distance)
    r = protocol.reconstruction
    cases = [(str(seed), materialize(r, 0.001, seed)[0]) for seed in range(1000, 1064)]
    cases += paired_hook_histories(r) + [("logical_tail", logical_tail_history(r))]
    if args.distance == 5:
        cases += syndrome_sector_histories(protocol)
    cases += [(f"stress_{seed}", materialize(r, 0.01, seed)[0]) for seed in range(2000, 2032)]
    print(json.dumps(Exporter(protocol).write(cases, args.output), indent=2))


if __name__ == "__main__":
    main()
