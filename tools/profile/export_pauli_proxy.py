"""Export structured physical operations and fixed-fault fixtures for external Stim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fold_blocks import Protocol as Blocks
from fold_cultivation import Operation
from pauli_proxy import Protocol, validation_cases


def mask(value: int) -> str:
    if not 0 <= value < 1 << 128:
        raise ValueError("proxy mask exceeds native capacity")
    return f"((Mask({value >> 64}ULL)<<64)|{value & ((1 << 64) - 1)}ULL)"


def encode_faults(protocol: Protocol, actual: list[list[Operation]]) -> list[tuple[int, int, int]]:
    faults: dict[int, tuple[int, int]] = {}
    offset = 0
    for stage, operations in zip(protocol.reconstruction.stages, actual, strict=True):
        boundary = 0
        for op in operations:
            if boundary < len(stage.operations) and op == stage.operations[boundary]:
                boundary += 1
            elif op.name in {"X", "Y", "Z"} and len(op.targets) == 1:
                key = offset + boundary
                x, z = faults.get(key, (0, 0))
                q = 1 << op.targets[0]
                faults[key] = (
                    x ^ (q if op.name in {"X", "Y"} else 0),
                    z ^ (q if op.name in {"Y", "Z"} else 0),
                )
            else:
                raise ValueError("fault history does not match the structured circuit")
        if boundary != len(stage.operations):
            raise ValueError("incomplete fixed history")
        offset += boundary
    return [(k, x, z) for k, (x, z) in sorted(faults.items()) if x or z]


def operation(op: Operation, noise: int = 0, previous: int = -1) -> str:
    targets = ",".join(map(str, (*op.targets, *([0] * (3 - len(op.targets))))))
    return f"{{Gate::{op.name},{{{targets}}},{len(op.targets)},{noise},{previous}}}"


def export(output: Path, distances: list[int]) -> list[dict]:
    header = Path(__file__).with_name("pauli_proxy_native.h").resolve()
    lines = [f'#include "{header}"', "using namespace pauli_proxy;"]
    plans = []
    metadata = []
    for distance in distances:
        proxy = Protocol(distance)
        reference = Blocks(distance)
        r = proxy.reconstruction
        cases = validation_cases(distance)
        fixtures = []
        for j, (_, stages) in enumerate(cases):
            reference_result = reference.evaluate(reference.encode(stages))
            faults = encode_faults(proxy, stages)
            entries = [f"{{{b},{mask(x)},{mask(z)}}}" for b, x, z in faults]
            name = f"faults_{distance}_{j}"
            lines.append(f"const Fault {name}[]={{" + ",".join(entries or ["{}"]) + "};")
            p = reference_result.acceptance
            xyz = reference_result.logical_xyz if p > 1e-12 else [0, 0, 0]
            assert xyz is not None
            expected = ",".join(repr(float(x)) for x in [p, *(p * v for v in xyz)])
            fixtures.append("{{" + name + f",{len(faults)}" + "},{" + expected + "}}")
        lines.append(f"const Fixture fixtures_{distance}[]={{" + ",".join(fixtures) + "};")
        operations = []
        noise_sites = 0
        for stage in r.stages:
            previous = -2 if stage.equal_records else -1
            for op in stage.operations:
                noise = (1 if op.name in {"R", "M"} else 2) if stage.noisy else 0
                noise_sites += bool(noise)
                operations.append(operation(op, noise, previous if op.name == "M" else -1))
                if op.name == "M" and stage.equal_records:
                    previous = op.targets[0]
        lines.append(f"const Operation ops_{distance}[]={{" + ",".join(operations) + "};")
        for name, probe in zip(("plus", "minus"), proxy.probes, strict=True):
            lines.append(
                f"const Operation {name}_{distance}[]={{"
                + ",".join(operation(op) for op in probe)
                + "};"
            )
        width = max(q for s in r.stages for op in s.operations for q in op.targets) + 1
        z = sum(1 << q for q in range(len(proxy.z)) if proxy.z[q])
        plans.append(
            f"{{{distance},{width},{r.ancilla},{mask(z)},ops_{distance},plus_{distance},"
            f"minus_{distance},fixtures_{distance}}}"
        )
        metadata.append({"distance": distance, "fixtures": len(cases), "noise_sites": noise_sites})
    lines += [
        "const Protocol protocols[]={" + ",".join(plans) + "};",
        "int main(int argc,char** argv){return benchmark(protocols,argc,argv);}",
    ]
    output.write_text("\n".join(lines) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, choices=(3, 5, 7), nargs="+", default=[3, 5, 7])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.distances), indent=2))


if __name__ == "__main__":
    main()
