"""Export a complete sampling comparison and independently check boundary faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark_fault_frames import export_plan
from clifford_branches import run_history
from export_fold_blocks import Exporter
from fold_blocks import Fold, Protocol
from fold_cultivation import Operation, coordinates, logical, pauli, stabilizers
from fold_frame_bridge import Bridge
from study_fault_frames import cases
from study_fold_adapter import held_out


def boundary_cases(protocol):
    r = protocol.reconstruction
    result = []
    for _, fold, groups in protocol.groups:
        if not isinstance(fold, Fold) or fold.plan.distance != 7:
            continue
        bridge = Bridge(fold)
        declined = []
        for boundary, sites in enumerate(fold.core.boundaries):
            for q, x, z in sites:
                for name, fault in (("X", x), ("Y", x | z), ("Z", z)):
                    correction = bridge.frame.evaluate(fault)
                    if not bridge.frame.data_only(correction, set(fold.mapping)):
                        declined.append((boundary, q, name))
        for k in range(12):
            boundary, q, name = declined[k * len(declined) // 12]
            history = [list(s.operations) for s in r.stages]
            stage = groups[1][0]
            history[stage].insert(boundary, Operation(name, (q,)))
            result.append((f"boundary_{stage}_{boundary}_{q}_{name}", history))
    cores = [i for i, s in enumerate(r.stages) if s.name == "fold_core_d7"]
    pairs = [i for i, op in enumerate(r.stages[cores[0]].operations) if op.name == "CCZ"]
    for k in (0, 9, 18, 27, 35):
        gate = pairs[k]
        for q in r.stages[cores[0]].operations[gate].targets[1:]:
            history = [list(s.operations) for s in r.stages]
            for stage in cores:
                history[stage].insert(gate + 1, Operation("X", (q,)))
            # The first X is data-only at the first measurement. Transporting
            # it through the second CCZ creates a genuine cat-data CZ, while
            # the second X cancels its data flip. These histories can survive.
            result.append((f"carried_data_hook_{k}_{q}", history))
    return result


def monomial(exporter, op, geometry):
    edges = sum(1 << geometry.edges.index(pair) for pair in op.edges)
    return (
        "{"
        + f"{exporter.mask(op.flips)},{exporter.mask(edges)},{op.global_phase},"
        + "{"
        + ",".join(map(str, op.linear))
        + "}}"
    )


class BridgeExporter(Exporter):
    def __init__(self, protocol):
        super().__init__(protocol)
        self.lines.insert(
            0, f'#include "{Path(__file__).with_name("benchmark_frame_bridge.h").resolve()}"'
        )
        self.bridges = {}

    def fold(self, plan):
        native = super().fold(plan)
        bridge = Bridge(plan)
        frame = export_plan(self, bridge.frame, self.offsets[id(plan.core)], set(plan.mapping))
        self.bridges[id(plan)] = self.obj(
            "fault_frame::Bridge",
            [
                "&" + frame,
                self.array("unsigned", plan.mapping),
                self.array("unsigned", plan.cats),
                self.array(
                    "fault_frame::Route", ["{" + ",".join(map(str, r)) + "}" for r in bridge.routes]
                ),
                "{{" + ",".join(monomial(self, op, plan.plan) for op in bridge.ideal) + "}}",
            ],
        )
        return native

    def entrypoint(self, kernel, fixtures):
        protocol = self.protocol
        r = protocol.reconstruction
        bridges, faults, regions = [], [], []
        for stage, (_, fold, _) in enumerate(protocol.groups):
            if not isinstance(fold, Fold):
                bridges.append("nullptr")
                continue
            bridges.append("&" + self.bridges[id(fold)])
            layouts = [fold.preparation.layout, fold.core, fold.decode.layout]
            offsets = [self.offsets[id(p)] for p in layouts]
            regions.append(
                "{"
                + ",".join(
                    map(
                        str,
                        [
                            stage,
                            offsets[0],
                            offsets[1],
                            offsets[1] + len(fold.core.slots),
                            offsets[2] + len(layouts[2].slots),
                        ],
                    )
                )
                + "}"
            )
            for layout, offset in zip(layouts, offsets, strict=True):
                for sites in layout.boundaries:
                    for _, x, z in sites:
                        xb = offset + x.bit_length() - 1 if x else -1
                        zb = offset + z.bit_length() - 1 if z else -1
                        for a, b in sorted({(xb, -1), (-1, zb), (xb, zb)} - {(-1, -1)}):
                            faults.append(f"{{{stage},{a},{b}}}")
        bridge_array = self.array("const fault_frame::Bridge*", bridges)
        fault_array = self.array("frame_bridge_probe::FaultCase", faults)
        region_array = self.array("frame_bridge_probe::Region", regions)
        checks = self.array(
            "std::string_view", [json.dumps(str(pauli(7, a, s))) for a, s in stabilizers(7)]
        )
        flags, count = [], 0
        for record_stage in r.stages:
            indices = []
            for op in record_stage.operations:
                if op.name == "M":
                    indices.append(count)
                    count += 1
            if record_stage.equal_records:
                flags.append("{" + ",".join(map(str, indices)) + "}")
        flag_array = self.array("std::array<unsigned,6>", flags)
        axes = ",".join(json.dumps(str(logical(7, axis))) for axis in "XYZ")
        body = "\n".join(
            line for line in r.text(0.001).splitlines() if not line.startswith("EXP_VAL")
        )
        # Include physical probes and a suffix Clifford in the survivor contract.
        circuit = r.text(0.001) + "EXP_VAL Z0\nH 0\nEXP_VAL X0\n"
        return f"""
static const fold_recognition::FamilySource source{{
    7,{len(coordinates(7))},R"body({body}\n)body",{checks},{{{axes}}},{flag_array},&{kernel}
}};
int main(int argc, char** argv) {{
    try {{
        unsigned shots = argc > 1 ? std::stoul(argv[1]) : 100000;
        unsigned trials = argc > 2 ? std::stoul(argv[2]) : 3;
        if (!shots || shots > 1000000 || !trials || trials > 20)
            throw std::invalid_argument("invalid experiment size");
        return frame_bridge_probe::run(source,{bridge_array},{fixtures},{fault_array},
            {region_array},R"circuit({circuit})circuit",shots,trials);
    }} catch (const std::exception& e) {{
        std::cerr << e.what() << std::endl;
        return 1;
    }}
}}
"""


def study(output, changed, independent):
    protocol = Protocol(7, reconstruction=held_out()) if changed else Protocol(7)
    r = protocol.reconstruction
    targeted = boundary_cases(protocol)
    histories = cases(protocol) + targeted
    exporter = BridgeExporter(protocol)
    metadata = exporter.write(histories, output, exporter.entrypoint)
    rows = []
    if independent:
        for name, history in targeted:
            expected = protocol.evaluate(protocol.encode(history))
            state, _ = run_history(r, history)
            values = [state.expectation()] + [state.expectation(logical(7, a)) for a in "XYZ"]
            reference = [expected.acceptance] + [
                expected.acceptance * v for v in (expected.logical_xyz or [0, 0, 0])
            ]
            error = float(np.max(np.abs(np.array(reference) - values)))
            if error > 2e-12:
                raise AssertionError((name, error))
            rows.append({"case": name, "acceptance": expected.acceptance, "max_error": error})
            print(json.dumps(rows[-1]), flush=True)
    return {
        "variant": "held_out" if changed else "original",
        **metadata,
        "boundary_cases": len(targeted),
        "independent_reference": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--independent", action="store_true")
    args = parser.parse_args()
    result = study(args.output, args.held_out, args.independent)
    args.metadata.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
