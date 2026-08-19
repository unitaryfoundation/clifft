#!/usr/bin/env python3
"""Smoke-test an installed clifft artifact (wheel or sdist build).

Must be run from the repository root -- the qv10.stim fixture path
below is relative. The QEMU/SDE wheel-smoke wrapper and the release
workflow's abi3-verification step both parse this script's stdout:
they grep for the ``isa=`` line and the trailing ``smoke ok`` line, so
keep those markers stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import clifft


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
    args = parser.parse_args()

    if args.expect_python is not None and sys.version_info[:2] != args.expect_python:
        expected = ".".join(str(part) for part in args.expect_python)
        got = ".".join(str(part) for part in sys.version_info[:2])
        print(f"error: expected python {expected}, got {got}", file=sys.stderr)
        raise SystemExit(1)

    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"version={clifft.__version__}  baseline={clifft.CPU_BASELINE}", flush=True)
    print(f"isa={clifft.runtime_isa()}", flush=True)

    symbolic = clifft.compile(Path("tests/fixtures/qv10.stim").read_text())
    result = clifft.sample(symbolic, shots=1, seed=280)
    assert result.measurements.shape[0] == 1, result.measurements.shape

    prog = clifft.compile("H 0\nCX 0 1\nM 0 1")
    ps = clifft.record_probabilities(prog, ["00", "11"])
    assert abs(float(ps[0]) - 0.5) < 1e-12 and abs(float(ps[1]) - 0.5) < 1e-12, ps

    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
