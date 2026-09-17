"""Export certified families for the standalone native eligibility experiment."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from export_fold_blocks import Exporter
from fold_blocks import Protocol
from fold_cultivation import coordinates, logical, pauli, stabilizers
from fold_schedule import SCHEDULES, reconstruction_with_schedule


def export(output: Path, schedule: str = "original") -> list[dict[str, int]]:
    header = Path(__file__).with_name("fold_recognition_main.h").resolve()
    pieces = [f'#include "{header}"']
    metadata = []
    for distance in (3, 5, 7):
        protocol = Protocol(
            distance, reconstruction=reconstruction_with_schedule(distance, schedule)
        )
        reconstruction = protocol.reconstruction
        exporter = Exporter(protocol)
        exporter.lines[:2] = [f"namespace family_{distance} {{", "using namespace fold_blocks;"]

        def entrypoint(kernel, fixtures):
            body = "\n".join(
                line
                for line in reconstruction.text(0.125).splitlines()
                if not line.startswith("EXP_VAL")
            )
            checks = exporter.array(
                "std::string_view",
                [json.dumps(str(pauli(distance, a, s))) for a, s in stabilizers(distance)],
            )
            flags = []
            count = 0
            for stage in reconstruction.stages:
                indices = []
                for operation in stage.operations:
                    if operation.name == "M":
                        indices.append(count)
                        count += 1
                if stage.equal_records:
                    if len(indices) != 6:
                        raise ValueError("unsupported conditional flag record group")
                    flags.append("{" + ",".join(map(str, indices)) + "}")
            flag_array = exporter.array("std::array<unsigned,6>", flags)
            axes = ",".join(json.dumps(str(logical(distance, axis))) for axis in "XYZ")
            return (
                "static const fold_recognition::FamilySource source{"
                f'{distance},{len(coordinates(distance))},R"circuit({body}\n)circuit",'
                f"{checks},{{{axes}}},{flag_array},&{kernel}" + "};\n}"
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "family.cpp"
            metadata.append({"distance": distance, **exporter.write([], path, entrypoint)})
            pieces.append(path.read_text())
    pieces.append(
        "int main(int argc, char** argv) {\n"
        "    const std::array sources{family_3::source,family_5::source,family_7::source};\n"
        "    return fold_recognition::run(sources, argc, argv);\n}"
    )
    output.write_text("\n".join(pieces) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schedule", choices=SCHEDULES, default="original")
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.schedule), indent=2))
