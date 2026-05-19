#!/usr/bin/env python3
"""Audit a CMake build for accidental host-specific compile flags.

Checks compile_commands.json for stray ``-march=native`` / ``-mcpu=native``
entries that would otherwise let one runner's CPU features leak into the
shipped binary via ccache reuse on a different runner (e.g. the libstim
``-march=native`` bug fixed in clifft#95).

Runtime-correctness of the shipped binary (e.g. AVX-512 leakage into
non-dispatched code paths) is out of scope here -- the QEMU wheel smoke
in ``.github/scripts/wheel_smoke.sh`` catches that class directly by
running the binary on an emulated sub-baseline CPU.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def audit_compile_commands(path: Path) -> None:
    entries = json.loads(path.read_text())
    native_commands: list[str] = []
    for entry in entries:
        command = entry.get("command")
        if command is None:
            arguments = entry.get("arguments", [])
        else:
            arguments = shlex.split(command)
        if "-march=native" in arguments or "-mcpu=native" in arguments:
            native_commands.append(entry.get("file", "<unknown>"))

    if native_commands:
        files = "\n  ".join(native_commands[:20])
        extra = (
            "" if len(native_commands) <= 20 else f"\n  ... and {len(native_commands) - 20} more"
        )
        fail(f"found native CPU tuning in compile_commands.json:\n  {files}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-commands", type=Path, required=True)
    args = parser.parse_args()

    if not args.compile_commands.exists():
        fail(f"compile commands file does not exist: {args.compile_commands}")

    audit_compile_commands(args.compile_commands)
    print("build-flag audit passed")


if __name__ == "__main__":
    main()
