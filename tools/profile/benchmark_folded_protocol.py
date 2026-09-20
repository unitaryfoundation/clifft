"""Time complete attempts against ordinary Clifft in the same PR-471 binary."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from compile_folded_protocol import compile_protocol
from folded_msc_family import make_circuit


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--f5-shots", type=int, default=256)
    parser.add_argument("--baseline-f5-shots", type=int, default=64)
    parser.add_argument("--schedules", nargs="+", default=["budgeted", "unbounded"])
    parser.add_argument("--distances", type=int, nargs="+", choices=[3, 5, 7], default=[3, 5])
    parser.add_argument("--native-growth", action="store_true")
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    files = [
        "compile_folded_protocol.py",
        "compile_folded_growth.py",
        "audit_folded_growth_handoff.py",
        "folded_check_contraction.py",
        "sample_folded_protocol.cpp",
        "folded_sampling_kernel.h",
        "gadget_contraction_kernel.h",
        "study_schedule.h",
        "folded_msc.py",
        "folded_msc_family.py",
        "folded_msc_f7.py",
        "benchmark_folded_protocol.py",
    ]
    report = dict(
        platform=platform.platform(),
        cpu_affinity=cpu,
        probability=args.probability,
        sampler=str(args.sampler.resolve()),
        sampler_sha256=sha256(args.sampler),
        source_sha256={name: sha256(Path(__file__).parent / name) for name in files},
        cases=[],
    )
    for distance in args.distances:
        bundle = args.bundles / f"f{distance}-p{args.probability}"
        start = time.perf_counter()
        metadata = compile_protocol(
            make_circuit(args.probability, distance=distance),
            distance,
            bundle,
            native_growth=args.native_growth and distance == 7,
        )
        compiler_seconds = time.perf_counter() - start
        native_shots = args.f5_shots if distance >= 5 else 4096
        baseline_shots = 0 if distance == 7 else args.baseline_f5_shots if distance == 5 else 65536
        for schedule in args.schedules:
            for early in (False, True):
                command = [
                    str(args.sampler.resolve()),
                    str(bundle.resolve()),
                    str(native_shots),
                    "0",
                    "18412",
                    str(int(early)),
                    schedule,
                    str(baseline_shots),
                ]
                result = json.loads(subprocess.check_output(command, text=True))
                native = statistics.median(result["seconds"]) / native_shots
                baseline = (
                    statistics.median(result["baseline_seconds"]) / baseline_shots
                    if baseline_shots
                    else None
                )
                result.update(
                    distance=distance,
                    native_growth=metadata.get("native_growth", False),
                    probability=args.probability,
                    schedule=schedule,
                    early_rejection=early,
                    circuit_sha256=metadata["circuit_sha256"],
                    compiler_seconds=compiler_seconds,
                    native_seconds_per_attempt=native,
                    baseline_seconds_per_attempt=baseline,
                    speedup=baseline / native if baseline is not None else None,
                    ordinary_coefficient_bytes=16 * (1 << result["ordinary_peak_width"]),
                    baseline_skip_reason="dense allocation limit" if not baseline_shots else None,
                    bundle_bytes=sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()),
                )
                report["cases"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                comparison = (
                    f"ordinary {baseline*1e3:.3f} ms, {baseline/native:.2f}x"
                    if baseline is not None
                    else "dense baseline skipped"
                )
                print(
                    f"f{distance} {schedule} early={early}: native {native*1e3:.3f} ms, "
                    + comparison,
                    flush=True,
                )


if __name__ == "__main__":
    main()
