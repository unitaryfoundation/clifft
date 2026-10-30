"""Collect reproducible summaries; preserve both QDK timing runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.experiment, args.output
    output.mkdir(parents=True, exist_ok=True)

    def read(path):
        return json.loads((root / path).read_text())

    final = {r["case"]: r for r in read("benchmark-final/summary.json")}
    all_hybrid = {r["case"]: r for r in read("benchmark-fused/summary.json")}
    all_hybrid.update(final)
    qdk_runs = [read("benchmark-qdk/summary.json"), read("benchmark-qdk-final/summary.json")]
    comparisons = {r["name"]: r for r in read("comparison/summary.json")}
    controls = {r["case"]: r for r in read("benchmark-auto/summary.json")}
    rows = []
    for name, value in all_hybrid.items():
        qdk = [r["us_per_shot"] for run in qdk_runs for r in run if r["case"] == name]
        baseline = comparisons.get(name, {}).get("baseline_us")
        rows.append(
            {
                "case": name,
                "hybrid_us": value["us_per_shot"],
                "hybrid_repeat_min_us": min(value["repeat_slopes_us"]),
                "hybrid_repeat_max_us": max(value["repeat_slopes_us"]),
                "qdk_run_min_us": min(qdk) if qdk else "",
                "qdk_run_max_us": max(qdk) if qdk else "",
                "trajectory_us": baseline if baseline is not None else "",
                "existing_auto_fixed_us": controls.get(name, {}).get("us_per_shot", ""),
                "peak_width": max(value["peak_widths"]),
                "isolated_headline_repeat": name in final,
            }
        )
    with (output / "timings.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sources = {
        "independent_validation.json": "validation-fused/summary.json",
        "representative_validation.json": "comparison/summary.json",
        "fusion_equivalence.json": "fusion-check/summary.json",
        "parity_readout_validation.json": "parity-adaptation.json",
        "qdk_validation.json": "qdk-validation/summary.json",
        "sanitizers.json": "sanitizer-check/summary.json",
        "workloads.json": "workloads/manifest.json",
        "hybrid_calls.jsonl": "benchmark-final/calls.jsonl",
        "qdk_calls_first.jsonl": "benchmark-qdk/calls.jsonl",
        "qdk_calls_repeat.jsonl": "benchmark-qdk-final/calls.jsonl",
        "fixed_control_calls.jsonl": "benchmark-auto/calls.jsonl",
        "exploratory_hybrid_calls.jsonl": "benchmark-fused/calls.jsonl",
    }
    for name, source in sources.items():
        shutil.copyfile(root / source, output / name)
    phases = {}
    for path in sorted((root / "profile-suite").glob("*/trajectory/result.json")):
        value = json.loads(path.read_text())
        phases[path.parent.parent.name] = value
    (output / "continuation_phases.json").write_text(json.dumps(phases, indent=2) + "\n")
    source_dir = Path(__file__).resolve().parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_dir.iterdir()
        if p.is_file()
    }
    metadata = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "date_utc": "2026-10-29",
        "clifft_reference": "0.12.1.dev28+g49b568528",
        "qdk": "1.33.1",
        "stim": "1.16.0",
        "timing_python": sys.version,
        "cpu": "AMD EPYC 9554P VM",
        "affinity": [2],
        "threads": 1,
        "compiler": "Ubuntu clang 18.1.3",
        "build": "Release -O3 -DNDEBUG -march=native ThinLTO",
        "binary_sha256": hashlib.sha256((root / "build/hybrid_loss").read_bytes()).hexdigest(),
        "source_sha256": hashes,
        "coefficient_budget": 20,
        "budget_storage_mib_including_scratch": 24,
        "timing_method": (
            "Three counts, three repetitions; slope of median endpoint times. "
            "Native output allocation included; binary-file writing excluded. "
            "QDK QIR per-call overhead removed by slope. "
            "Native prototype versus public Python APIs."
        ),
        "isolation": (
            "Six headline hybrid cases were repeated after all other work stopped. "
            "Both QDK runs retained. Exploratory full sweep overlapped validation "
            "on another core; do not treat its timings as isolated headline measurements."
        ),
        "scope": (
            "QDK comparison only the matched surface-loss subset. Cultivation uses "
            "explicit parity ancilla and p=0.001; its added loss and original "
            "Pauli-channel semantics are Clifft-specific. "
            "Neutral-atom comparisons retain both five-level models."
        ),
        "sanitizer_scope": (
            "ASan, UBSan, and assertions on four representative cases. "
            "LeakSanitizer disabled because of sandbox ptrace restrictions."
        ),
        "integration": (
            "Standalone experimental executor; no public API or automatic routing changes."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    archive = root / "reproduction.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in source_dir.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                bundle.write(path, "tools/hybrid_loss/" + str(path.relative_to(source_dir)))
        benchmark = source_dir.parents[1] / "benchmarks/qdk_skip_loss.py"
        bundle.write(benchmark, "benchmarks/qdk_skip_loss.py")
        for path in (root / "workloads").rglob("*"):
            if path.is_file():
                bundle.write(path, str(path.relative_to(root)))
    print(f"Wrote {len(rows)} timing rows, validation summaries, and {archive}")


if __name__ == "__main__":
    main()
