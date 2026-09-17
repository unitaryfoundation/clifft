"""Derive certified logical-block plans from Clifft-parsed circuit regions.

This bounded adapter retains the known injection and fold anchors and canonical
wire labels. Syndrome schedules and final growth gates come from the input AST.
The native recognizer checks noise, detector dependencies, and terminal outputs
against the newly certified body before any sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

from fold_blocks import Protocol
from fold_cultivation import Operation, Reconstruction
from fold_plan_data import Writer


class Adapter:
    def __init__(self, worker: Path):
        self.worker = worker
        self.anchors: dict[int, list[list[Operation]]] = {}

    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.stim"
            path.write_text(text)
            result = subprocess.run(
                [str(self.worker), "--parse", str(path)], capture_output=True, text=True
            )
        if result.returncode:
            raise ValueError(result.stderr.strip())
        return json.loads(result.stdout)

    def anchor_stages(self, r):
        if r.distance not in self.anchors:
            text = ""
            for stage in r.stages:
                text += (
                    "\n".join(
                        op.name + " " + " ".join(map(str, op.targets)) for op in stage.operations
                    )
                    + "\nTICK\n"
                )
            stages: list[list[Operation]] = []
            current: list[Operation] = []
            for node in self.parse(text)["nodes"]:
                if node["gate"] == "TICK":
                    stages.append(current)
                    current = []
                else:
                    current.append(Operation(node["gate"], tuple(node["targets"])))
            self.anchors[r.distance] = stages
        return self.anchors[r.distance]

    def derive(self, text):
        parsed = self.parse(text)
        # Canonical physical labels are an explicit restriction of this adapter.
        distance = {16: 3, 47: 5, 99: 7}.get(parsed["qubits"])
        if distance is None:
            raise ValueError("unsupported physical layout")
        nodes = parsed["nodes"]
        if any(node["tagged"] for node in nodes):
            raise ValueError("tagged operations are unsupported")
        r = Reconstruction(distance)
        r.build()
        anchors = self.anchor_stages(r)
        body = [
            node
            for node in nodes
            if not node["noise"] and node["gate"] not in {"TICK", "DETECTOR", "EXP_VAL"}
        ]
        pos = 0
        while pos < len(body) and body[pos]["clifford"]:
            pos += 1

        def operation(node):
            if node["args"] or any(q >= 1 << 28 for q in node["targets"]):
                raise ValueError("unsupported region operation")
            return Operation(node["gate"], tuple(node["targets"]))

        for stage, expected in zip(r.stages, anchors, strict=True):
            if stage.name.startswith("syndrome"):
                remaining = sum(op.name == "M" for op in stage.operations)
                actual = []
                while remaining and pos < len(body):
                    op = operation(body[pos])
                    if op.name not in {"R", "H", "CX", "M"}:
                        raise ValueError("unsupported syndrome gate")
                    actual.append(op)
                    remaining -= op.name == "M"
                    pos += 1
                if remaining:
                    raise ValueError("incomplete syndrome region")
                stage.operations = actual
            elif stage.name == "grow_regular_d5_to_regular_d7":
                actual = []
                while pos < len(body):
                    op = operation(body[pos])
                    if op == Operation("R", (r.ancilla,)):
                        break
                    if op.name not in {"R", "H", "CX"}:
                        raise ValueError("unsupported growth gate")
                    actual.append(op)
                    pos += 1
                stage.operations = actual
            else:
                actual = [operation(node) for node in body[pos : pos + len(expected)]]
                if actual != expected:
                    raise ValueError(f"unsupported fixed region {stage.name}")
                pos += len(expected)
        if any(not node["clifford"] for node in body[pos:]):
            raise ValueError("unsupported terminal operation")
        self.check_body(r, nodes)
        return Protocol(distance, reconstruction=r)

    def check_body(self, reconstruction, nodes):
        actual = [node for node in nodes if node["gate"] != "TICK"]
        text = "\n".join(
            line
            for line in reconstruction.text(0.125).splitlines()
            if not line.startswith("EXP_VAL")
        )
        expected = self.parse(text)["nodes"]
        pos = 0
        while pos < len(actual) and actual[pos]["clifford"]:
            pos += 1
        for node in expected:
            if node["noise"] and (
                pos == len(actual)
                or (node["gate"], node["targets"]) != (actual[pos]["gate"], actual[pos]["targets"])
            ):
                continue
            if pos == len(actual):
                raise ValueError("incomplete circuit body")
            candidate = actual[pos]
            if (node["gate"], node["targets"]) != (candidate["gate"], candidate["targets"]):
                raise ValueError("noise location or detector dependency differs")
            if node["noise"]:
                args = candidate["args"]
                if len(args) != 1 or not math.isfinite(args[0]) or not 0 <= args[0] <= 1:
                    raise ValueError("unsupported noise probability")
            elif node["args"] != candidate["args"]:
                raise ValueError("gate arguments differ")
            pos += 1
        if any(not node["clifford"] and node["gate"] != "EXP_VAL" for node in actual[pos:]):
            raise ValueError("unsupported output contract")

    def prepare(self, circuit, plan, cases=()):
        begin = time.perf_counter()
        protocol = self.derive(circuit.read_text())
        metadata = Writer(protocol).write(plan, cases)
        # Matching the actual AST to the derived body retains the existing noise
        # and output certificate; stripped annotations are never trusted implicitly.
        result = self.run(plan, circuit, shots=0)
        if not result["eligible"]:
            plan.unlink()
            raise ValueError(result["reason"])
        metadata["offline_seconds"] = time.perf_counter() - begin
        return protocol, metadata

    def run(self, plan, circuit, shots=100, seed=19331, keep=True, request="ordinary"):
        result = subprocess.run(
            [
                str(self.worker),
                str(plan),
                str(circuit),
                str(shots),
                str(seed),
                str(int(keep)),
                "1",
                request,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip())
        return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=100000)
    args = parser.parse_args()
    adapter = Adapter(args.worker)
    try:
        _, metadata = adapter.prepare(args.circuit, args.plan)
        print(
            json.dumps(
                {
                    "planning": metadata,
                    "sampling": adapter.run(args.plan, args.circuit, args.shots),
                },
                indent=2,
            )
        )
    except ValueError as error:
        print(json.dumps({"eligible": False, "reason": str(error)}))


if __name__ == "__main__":
    main()
