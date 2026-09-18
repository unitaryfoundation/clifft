"""Measure reusable public programs on noisy synthetic five-check CSS controls."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import clifft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0 or args.repeats <= 0:
        parser.error("shots and repeats must be positive")
    rows = []
    for distance in (3, 5, 7, 9):
        path = Path(__file__).parent / f"fixtures/css_five_check_d{distance}.stim"
        text = path.read_text()
        for enabled in (False, True):
            if distance == 9 and not enabled:
                # Planning is cheap; allocating the ordinary width-31 worker
                # alone would require 32 GiB before scratch and output storage.
                continue
            begin = time.perf_counter()
            plan = clifft.compile(text, logical_blocks=enabled)
            compile_seconds = time.perf_counter() - begin
            clifft.sample(plan, 8, seed=72, threads=1, batch_size=1)
            elapsed = []
            for repeat in range(args.repeats):
                begin = time.perf_counter()
                clifft.sample(plan, args.shots, seed=170 + repeat, threads=1, batch_size=1)
                elapsed.append(time.perf_counter() - begin)
            row = dict(
                circuit=path.name,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
                logical_blocks=enabled,
                blocks=plan.num_css_blocks,
                peak_active_width=plan.peak_active_width,
                compile_seconds=compile_seconds,
                sample_seconds=elapsed,
                median_seconds_per_shot=statistics.median(elapsed) / args.shots,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                version=clifft.__version__,
                platform=platform.platform(),
                shots=args.shots,
                repeats=args.repeats,
                threads=1,
                batch_size=1,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
