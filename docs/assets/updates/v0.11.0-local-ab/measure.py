"""Reproduce the local CPU A/B measurements accompanying the v0.11.0 post.

Run with --help for controller arguments. Each variant runs in a separate
persistent Python process, pinned to the same logical CPU. The JSON output
retains all paired timings, circuit hashes, build identities, and shot counts.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

WORKLOADS = {
    "coherent_d3": ("tests/fixtures/coherent_d3_r3.stim", False),
    "coherent_d5": ("tests/fixtures/coherent_d5_r5.stim", False),
    "cultivation_d5": ("tests/fixtures/cultivation_d5.stim", True),
    "qv10": ("tests/fixtures/qv10.stim", False),
    "surface_survivors": ("tests/fixtures/surface_d7_r7_p001.stim", True),
    "surface_ordinary": ("tests/fixtures/surface_d7_r7_p001.stim", False),
    "s_gate_cultivation": ("docs/guide/circuits/circuit_d3_s_gate_p0.001.stim", True),
    "early_rejection": (None, True),
}


def worker(args):
    import numpy

    import clifft

    path, survivors = WORKLOADS[args.workload]
    text = (
        (Path(args.root) / path).read_text()
        if path
        else """
        X_ERROR(0.99) 0
        M 0
        DETECTOR rec[-1]
        REPEAT 4096 {
            R 1
            DEPOLARIZE1(0.005) 1
            M 1
        }
    """
    )
    mask = [1] * clifft.compile(text).num_detectors if survivors else None
    compile_seconds = []
    for iteration in range(6):
        passes = clifft.default_hir_pass_manager()
        if args.schedule:
            passes.add(clifft.ActiveWidthSchedulePass())
        start = perf_counter()
        program = clifft.compile(text, postselection_mask=mask, hir_passes=passes)
        elapsed = perf_counter() - start
        if iteration:
            compile_seconds.append(elapsed)
    print(
        json.dumps(
            {
                "version": clifft.__version__,
                "runtime_isa": clifft.runtime_isa(),
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "peak_active_width": program.peak_active_width,
                "compile_seconds": compile_seconds,
                "fixture": path,
                "fixture_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "api": "sample_survivors" if survivors else "sample",
                "keep_records": args.workload == "early_rejection" if survivors else True,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = json.loads(line)
        start = perf_counter()
        if survivors:
            result = clifft.sample_survivors(
                program,
                request["shots"],
                seed=request["seed"],
                keep_records=args.workload == "early_rejection",
                threads=1,
                batch_size=1,
            )
        else:
            result = clifft.sample(
                program, request["shots"], seed=request["seed"], threads=1, batch_size=1
            )
        elapsed = perf_counter() - start
        passed = result.passed_shots if survivors else request["shots"]
        assert passed is not None
        assert 0 <= passed <= request["shots"]
        print(json.dumps({"seconds": elapsed, "passed_shots": passed}), flush=True)
        del result


def exchange(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"Benchmark worker exited: {process.poll()}")
    return json.loads(line)


def controller(args):
    os.sched_setaffinity(0, {args.cpu})
    names = (
        ["coherent_d3", "coherent_d5", "cultivation_d5", "qv10"]
        if args.mode == "scheduler"
        else [
            "cultivation_d5",
            "s_gate_cultivation",
            "surface_survivors",
            "surface_ordinary",
            "early_rejection",
        ]
    )
    records = []
    for name in names:
        processes = []
        metadata = []
        try:
            for variant, python in enumerate([args.python_a, args.python_b]):
                command = [
                    python,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--root",
                    args.root,
                    "--workload",
                    name,
                ]
                if args.mode == "scheduler" and variant:
                    command.append("--schedule")
                process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
                )
                processes.append(process)
                assert process.stdout is not None
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f"Worker failed to compile {name}")
                metadata.append(json.loads(line))
            assert metadata[0]["fixture_sha256"] == metadata[1]["fixture_sha256"]
            shots = 100
            calibration = [exchange(p, {"shots": shots, "seed": 42}) for p in processes]
            slowest = max(x["seconds"] for x in calibration)
            shots = max(1, min(1_000_000, math.ceil(shots * args.seconds / slowest)))
            for process in processes:
                exchange(process, {"shots": shots, "seed": 43})
            pairs = []
            for repeat in range(args.repeats):
                pair = {}
                for variant in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    pair[str(variant)] = exchange(
                        processes[variant], {"shots": shots, "seed": 100 + repeat}
                    )
                pair["speedup"] = pair["0"]["seconds"] / pair["1"]["seconds"]
                pairs.append(pair)
            row = {
                "workload": name,
                "shots_per_call": shots,
                "variants": metadata,
                "pairs": pairs,
                "median_speedup": statistics.median(p["speedup"] for p in pairs),
            }
            records.append(row)
            print(
                name,
                json.dumps(
                    {
                        "speedup": row["median_speedup"],
                        "shots": shots,
                        "compile_ms": [
                            1000 * statistics.median(m["compile_seconds"]) for m in metadata
                        ],
                    }
                ),
                flush=True,
            )
        finally:
            for process in processes:
                assert process.stdin is not None
                process.stdin.close()
                process.wait(timeout=10)
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    payload = {
        "mode": args.mode,
        "cpu": cpu,
        "logical_cpu": args.cpu,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": "exe.dev KVM virtual machine",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "variant_a_sha": args.sha_a,
        "variant_b_sha": args.sha_b,
        "compiler": args.compiler,
        "build": "Release; native CPU baseline; IPO off",
        "threads": 1,
        "batch_size": 1,
        "repeats": args.repeats,
        "seeds": list(range(100, 100 + args.repeats)),
        "target_seconds_per_slower_call": args.seconds,
        "timing": "perf_counter wall time; compilation excluded from sampling; alternating AB/BA",
        "command": sys.argv,
        "results": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", required=True, help="Checkout supplying circuit fixtures")
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--mode", choices=["scheduler", "noise"])
    parser.add_argument("--python-a")
    parser.add_argument("--python-b")
    parser.add_argument("--sha-a")
    parser.add_argument("--sha-b")
    parser.add_argument("--compiler")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        controller(args)


if __name__ == "__main__":
    main()
