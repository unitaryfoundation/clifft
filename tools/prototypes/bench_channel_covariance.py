"""Run the standalone native covariance compiler study with ordinary sampling."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from tools.prototypes.bench_native_phase import _noisy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--existing-bench", type=Path)
    parser.add_argument("--target-seconds", type=float, default=0.03)
    parser.add_argument("--repeats", type=int, default=3)
    scan_modes = parser.add_mutually_exclusive_group()
    scan_modes.add_argument("--scan", action="store_true")
    scan_modes.add_argument("--settings-scan", action="store_true")
    scan_modes.add_argument("--collection-scan", action="store_true")
    parser.add_argument("--circuits", type=int, nargs="+", default=[5, 262, 300, 689, 1037])
    args = parser.parse_args()
    mode = (
        "--collection-scan"
        if args.collection_scan
        else "--settings-scan"
        if args.settings_scan
        else "--scan"
        if args.scan
        else "--bench"
    )
    settings = [mode, str(args.target_seconds), str(args.repeats)]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader((args.corpus / "circuits.csv").open()))
    results: list[dict[str, Any]] = []
    for number in args.circuits:
        item = manifest[number - 1]
        path = args.output / f"gate_depolarizing_{number:04}.stim"
        path.write_text(_noisy((args.corpus / item["circuit"]).read_text(), "gate_depolarizing", 0))
        process = subprocess.run(
            [str(args.binary.resolve()), str(path), item["postselection_detectors"], *settings],
            check=True,
            capture_output=True,
            text=True,
        )
        result = dict(circuit=item["circuit"], **json.loads(process.stdout))
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.output / "corpus.json").write_text(json.dumps(results, indent=2) + "\n")
    existing = []
    if args.existing_bench:
        workloads = json.loads((args.existing_bench / "manifests/workloads.v1.json").read_text())[
            "workloads"
        ]
        for workload in workloads:
            source = args.existing_bench / "manifests" / workload["artifact"]["path"]
            count = (
                workload["expected_metadata"]["num_detectors"]
                if workload["semantics"]["postselect_all_detectors"]
                else 0
            )
            process = subprocess.run(
                [str(args.binary.resolve()), str(source), str(count), *settings],
                check=True,
                capture_output=True,
                text=True,
            )
            result = dict(workload=workload["id"], **json.loads(process.stdout))
            existing.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "existing.json").write_text(json.dumps(existing, indent=2) + "\n")


if __name__ == "__main__":
    main()
