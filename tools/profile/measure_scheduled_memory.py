"""Measure isolated-process memory for the default and scheduled baselines."""

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

from benchmark_scheduled_baseline import MODES, digest, scheduled_compile
from folded_msc_family import make_circuit as folded_circuit
from gadget_family import make_circuit as gadget_circuit

import clifft


def rss_kib():
    return int(
        next(
            line.split()[1]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
    )


def worker(family, distance, mode):
    source = (
        folded_circuit(0.001, distance=distance).text()
        if family == "folded"
        else gadget_circuit(distance, 0.001)
    )
    initial = rss_kib()
    plan, _, _ = scheduled_compile(source, mode)
    compiled = rss_kib()
    compile_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sampled = plan.peak_active_width <= 24
    if sampled:
        clifft.sample(plan, 1, seed=471, threads=1)
    return dict(
        family=family,
        distance=distance,
        mode=mode,
        peak_active_width=plan.peak_active_width,
        coefficient_bytes=16 * (1 << plan.peak_active_width),
        sampled=sampled,
        initial_rss_kib=initial,
        after_compile_rss_kib=compiled,
        after_sample_rss_kib=rss_kib() if sampled else None,
        compile_peak_rss_kib=compile_peak,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("FAMILY", "DISTANCE", "MODE"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.worker:
        family, distance, mode = args.worker
        print(json.dumps(worker(family, int(distance), mode)))
        return
    if args.output is None:
        parser.error("--output is required")
    report = dict(
        extension_sha256=digest(clifft._clifft_core.__file__),
        driver_sha256=digest(__file__),
        method="fresh process per case; Linux ru_maxrss; includes Python and compilation; "
        "one full shot at p=0.001; skip sampling above active width 24",
        cases=[],
    )
    for family, distances in (("folded", (3, 5)), ("synthetic", (3, 5, 7, 9))):
        for distance in distances:
            for mode in MODES:
                row = json.loads(
                    subprocess.check_output(
                        [sys.executable, __file__, "--worker", family, str(distance), mode],
                        text=True,
                    )
                )
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
