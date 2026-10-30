"""Compare the integrated public modes on the frozen representative corpus."""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Literal

from experiment import python_model

import clifft

Execution = Literal["online", "fixed", "trajectory"]
CASES: list[tuple[str, int, Execution]] = [
    ("surface_d5_loss0.001", 32, "trajectory"),
    ("surface_d7_loss0.001", 16, "trajectory"),
    ("cultivation_d3_loss0.001", 256, "trajectory"),
    ("cultivation_d5_loss0.001", 128, "trajectory"),
    ("neutral_two_row_exact", 256, "trajectory"),
    ("neutral_two_row_ldu_exact", 64, "trajectory"),
    ("cultivation_d3_depol_partner", 8192, "fixed"),
    ("cultivation_d5_depol_partner", 2048, "fixed"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--build-description", required=True)
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0)) if args.cpu is None else args.cpu
    os.sched_setaffinity(0, {cpu})
    report = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "python": platform.python_version(),
        "clifft": clifft.__version__,
        "package_path": clifft.__file__,
        "platform": platform.platform(),
        "cpu": cpu,
        "build": args.build_description,
        "method": "One worker; four warmup shots; median of three public prepared-call "
        "averages including result conversion; identical corpus/model in each mode. "
        "Seed differs by repeat. This bounded sweep is not the full crossover study.",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, baseline_shots, baseline in CASES:
        folder = args.workloads / name
        text = (folder / "circuit.stim").read_text()
        model_text = (folder / "model.json").read_text()
        model = python_model(json.loads(model_text))
        executions: tuple[Execution, Execution] = (baseline, "online")
        for execution in executions:
            shots = baseline_shots if execution != "online" else min(max(baseline_shots, 256), 2048)
            start = time.perf_counter()
            program = clifft.noncomp.compile(text, model, execution=execution)
            preparation = time.perf_counter() - start
            program.sample(4, seed=21)
            elapsed = []
            for repeat in range(3):
                start = time.perf_counter()
                result = program.sample(shots, seed=32 + repeat, collect_stats=True)
                elapsed.append((time.perf_counter() - start) * 1e6 / shots)
            assert result.stats is not None
            median = statistics.median(elapsed)
            row = {
                "case": name,
                "execution": execution,
                "shots": shots,
                "circuit_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "model_sha256": hashlib.sha256(model_text.encode()).hexdigest(),
                "us_per_shot": median,
                "repeats_us": elapsed,
                "preparation_seconds": preparation,
                "stats": dataclasses.asdict(result.stats),
            }
            report["results"].append(row)
            print(name, execution, round(median, 3), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
