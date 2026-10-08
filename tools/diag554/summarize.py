"""Summarize issue 554 diagnosis runs as a Markdown table."""

import json
import math
import pathlib
import re
import sys

LEGS = ["base-first", "head-first", "head-second", "base-second"]


def iterations(path):
    data = json.loads(path.read_text())
    return [b["cpu_time"] / 1e6 for b in data["benchmarks"] if b.get("run_type") == "iteration"]


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def main():
    out = pathlib.Path(sys.argv[1])
    cpu = "unknown"
    env = out / "env.txt"
    if env.exists():
        match = re.search(r"Model name:\s*(.*)", env.read_text())
        if match:
            cpu = match.group(1).strip()
    print(f"### Issue 554 diagnosis on {cpu}\n")
    print("| Experiment | base ms | head ms | change | legs (ms) |")
    print("|---|---:|---:|---:|---|")
    for directory in sorted(p for p in out.iterdir() if p.is_dir()):
        if not all((directory / f"{leg}.json").exists() for leg in LEGS):
            continue
        medians = {leg: median(iterations(directory / f"{leg}.json")) for leg in LEGS}
        base = math.sqrt(medians["base-first"] * medians["base-second"])
        head = math.sqrt(medians["head-first"] * medians["head-second"])
        legs = " / ".join(f"{medians[leg]:.2f}" for leg in LEGS)
        print(
            f"| {directory.name} | {base:.2f} | {head:.2f} | "
            f"{100 * (head / base - 1):+.1f}% | {legs} |"
        )
    single = out / "single"
    if single.exists():
        print("\n| Single process | median ms | repetitions (ms) |")
        print("|---|---:|---|")
        paths = sorted(single.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for path in paths:
            times = iterations(path)
            reps = ", ".join(f"{t:.2f}" for t in times)
            print(f"| {path.stem} | {median(times):.2f} | {reps} |")
    procs = out / "procs"
    if procs.exists():
        print("\n| Fresh processes | per-process ms |")
        print("|---|---|")
        for rev in ["base", "head"]:
            times = []
            for path in sorted(procs.glob(f"{rev}-*.json")):
                times.extend(iterations(path))
            print(f"| {rev} | {', '.join(f'{t:.2f}' for t in times)} |")


if __name__ == "__main__":
    main()
