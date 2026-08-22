#!/usr/bin/env python3
"""Verify that a HIP host object contains real kernels for one GPU target."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "stdout", "")
        fail(f"command failed: {' '.join(command)}\n{output}")
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--objcopy", required=True)
    parser.add_argument("--bundler", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--expected-kernels", type=int, required=True)
    args = parser.parse_args()

    if not args.object.is_file():
        fail(f"HIP object does not exist: {args.object}")

    target = f"hipv4-amdgcn-amd-amdhsa--{args.architecture}"
    with tempfile.TemporaryDirectory(prefix="clifft-hip-audit-") as temporary:
        directory = Path(temporary)
        fatbin = directory / "device.fatbin"
        code_object = directory / f"{args.architecture}.o"
        run([args.objcopy, f"--dump-section=.hip_fatbin={fatbin}", str(args.object)])
        run(
            [
                args.bundler,
                "--type=o",
                f"--targets={target}",
                f"--input={fatbin}",
                f"--output={code_object}",
                "--unbundle",
            ]
        )
        notes = run([args.readelf, "--notes", str(code_object)])
        symbols = run([args.readelf, "--symbols", "--wide", str(code_object)])

    names = [
        match.group(1).strip("'\"")
        for match in re.finditer(r"^\s*\.name:\s+(\S+)", notes, re.MULTILINE)
    ]
    if len(names) != args.expected_kernels:
        fail(
            f"expected {args.expected_kernels} {args.architecture} kernels, "
            f"found {len(names)} in AMDGPU metadata\n{notes}"
        )
    missing_symbols = [name for name in names if name not in symbols]
    if missing_symbols:
        fail(f"kernel metadata names missing from symbol table: {missing_symbols}")

    print(f"HIP fatbin audit passed: {len(names)} {args.architecture} kernels")


if __name__ == "__main__":
    main()
