"""Bounded public-API benchmarks with separately built comparison packages."""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def worker(args):
    sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str(args.package.resolve()))
    import clifft

    assert Path(clifft.__file__).resolve().is_relative_to(args.package.resolve())
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    text = (ROOT / f"tests/fixtures/folded/f{args.distance}.stim").read_text()
    kwargs: dict[str, Any] = {}
    if args.mode == "specialized":
        kwargs["specialize_folded"] = True
    elif args.mode == "scheduled":
        passes = clifft.default_hir_pass_manager()
        scheduler = getattr(clifft, "ActiveWidthSchedulePass")(search_budget=None)
        passes.add(scheduler)
        kwargs["hir_passes"] = passes
    elif args.mode == "disabled":
        kwargs["specialize_folded"] = False
    if args.early:
        kwargs["postselection_mask"] = [1] * sum(
            n.gate.name == "DETECTOR" for n in clifft.parse(text).nodes
        )
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    program = clifft.compile(text, **kwargs)
    compile_seconds = time.perf_counter() - start
    compile_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows = []
    # Never instantiate the ordinary distance-seven coefficient array.
    skipped = program.peak_active_width > 26
    if not skipped:
        sample = clifft.sample_survivors if args.early else clifft.sample
        sample_kwargs: dict[str, Any] = {"keep_records": True} if args.early else {}
        warm = sample(program, shots=1, seed=719, threads=1, batch_size=1, **sample_kwargs)
        del warm
        for repeat in range(3):
            gc.collect()
            start = time.perf_counter()
            result = sample(
                program,
                shots=args.shots,
                seed=9134 + repeat,
                threads=1,
                batch_size=1,
                **sample_kwargs,
            )
            elapsed = time.perf_counter() - start
            accepted = int((result.detectors == 0).all(axis=1).sum())
            rows.append(dict(seconds=elapsed, accepted=accepted))
            del result
    extension = Path(clifft.__file__).parent / "_clifft_core.abi3.so"
    print(
        json.dumps(
            dict(
                mode=args.mode,
                distance=args.distance,
                early=args.early,
                shots=args.shots,
                circuit_sha256=hashlib.sha256(text.encode()).hexdigest(),
                cpu=cpu,
                package=str(args.package.resolve()),
                extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                python=platform.python_version(),
                compile_seconds=compile_seconds,
                peak_active_width=program.peak_active_width,
                specialized=getattr(program, "has_folded_regions", False),
                import_peak_rss_kib=before_rss,
                compile_peak_rss_kib=compile_rss,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                batches=rows,
                median_seconds_per_attempt=statistics.median(r["seconds"] for r in rows)
                / args.shots
                if rows
                else None,
                skipped_dense_allocation=skipped,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--scheduled", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--dependencies", type=Path, default=Path(sysconfig.get_path("purelib")))
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--mode", default="ordinary")
    parser.add_argument("--early", action="store_true")
    parser.add_argument("--shots", type=int, default=8)
    args = parser.parse_args()
    if args.package:
        worker(args)
        return
    cases = []
    for distance in (3, 5, 7):
        for early in (False, True):
            cases.append(
                (args.scheduled, "scheduled", distance, early, 131072 if distance == 3 else 16)
            )
            cases.append(
                (args.current, "specialized", distance, early, {3: 2048, 5: 128, 7: 8}[distance])
            )
    # The same default pipeline before and after integration isolates disabled-path cost.
    for package, mode in ((args.parent, "ordinary"), (args.current, "disabled")):
        cases.append((package, mode, 3, False, 131072))
    if args.controls_only:
        cases = [
            (package, mode, 3, False, 131072)
            for package, mode in (
                (args.parent, "ordinary"),
                (args.current, "disabled"),
                (args.current, "disabled"),
                (args.parent, "ordinary"),
            )
        ]
    results = []
    for package, mode, distance, early, shots in cases:
        command = [
            sys.executable,
            "-S",
            str(Path(__file__).resolve()),
            "--dependencies",
            str(args.dependencies),
            "--package",
            str(package),
            "--mode",
            mode,
            "--distance",
            str(distance),
            "--shots",
            str(shots),
        ]
        if early:
            command.append("--early")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        completed = subprocess.run(command, capture_output=True, text=True, env=env)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        row = json.loads(completed.stdout)
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            mode,
            distance,
            early,
            row["median_seconds_per_attempt"],
            row["process_peak_rss_kib"],
            flush=True,
        )


if __name__ == "__main__":
    main()
