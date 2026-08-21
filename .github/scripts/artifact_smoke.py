#!/usr/bin/env python3
"""Smoke-test an installed clifft artifact (wheel or sdist build).

The QEMU/SDE wheel-smoke wrapper and the release workflow's abi3-verification
step both parse this script's stdout:
they grep for the ``isa=`` line and the trailing ``smoke ok`` line, so
keep those markers stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import clifft

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_python_version(value: str) -> tuple[int, int]:
    try:
        major, minor = value.split(".")
        return int(major), int(minor)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected X.Y, got {value!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-python",
        type=parse_python_version,
        default=None,
        metavar="X.Y",
        help="fail unless the running interpreter is exactly this major.minor version",
    )
    parser.add_argument(
        "--exercise-intra-shot",
        action="store_true",
        help="also run an explicit two-worker intra-shot OpenMP sample",
    )
    args = parser.parse_args()

    if args.expect_python is not None and sys.version_info[:2] != args.expect_python:
        expected = ".".join(str(part) for part in args.expect_python)
        got = ".".join(str(part) for part in sys.version_info[:2])
        print(f"error: expected python {expected}, got {got}", file=sys.stderr)
        raise SystemExit(1)

    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"version={clifft.__version__}  baseline={clifft.CPU_BASELINE}", flush=True)
    print(f"isa={clifft.runtime_isa()}", flush=True)

    symbolic = clifft.compile((_PROJECT_ROOT / "tests/fixtures/qv10.stim").read_text())
    result = clifft.sample(symbolic, shots=1, seed=280)
    assert result.measurements.shape[0] == 1, result.measurements.shape
    if args.exercise_intra_shot:
        threaded = clifft.sample(
            symbolic,
            shots=1,
            seed=280,
            thread_layout=(1, 2),
            intra_shot_min_active_width=0,
        )
        for serial_values, threaded_values in (
            (result.measurements, threaded.measurements),
            (result.detectors, threaded.detectors),
            (result.observables, threaded.observables),
        ):
            assert serial_values.shape == threaded_values.shape
            assert serial_values.tobytes() == threaded_values.tobytes()
        print("intra-shot=ok", flush=True)

    prog = clifft.compile("H 0\nCX 0 1\nM 0 1")
    ps = clifft.record_probabilities(prog, ["00", "11"])
    assert abs(float(ps[0]) - 0.5) < 1e-12 and abs(float(ps[1]) - 0.5) < 1e-12, ps

    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
