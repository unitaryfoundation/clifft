#!/usr/bin/env python3
"""Audit dynamic exports from the installed Python extension."""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def extension_path() -> Path:
    module = importlib.import_module("clifft._clifft_core")
    module_file = module.__file__
    if module_file is None:
        raise RuntimeError("clifft._clifft_core does not have a __file__ attribute")
    return Path(module_file).resolve()


def nm_command(path: Path) -> list[str] | None:
    nm = shutil.which("nm")
    if nm is None:
        raise RuntimeError("nm was not found on PATH")

    system = platform.system()
    if system == "Linux":
        return [nm, "-D", "--defined-only", str(path)]
    if system == "Darwin":
        return [nm, "-gU", str(path)]
    return None


def expected_init_symbol() -> str:
    if platform.system() == "Darwin":
        return "_PyInit__clifft_core"
    return "PyInit__clifft_core"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension",
        type=Path,
        default=None,
        help="Path to _clifft_core extension. Defaults to importing clifft._clifft_core.",
    )
    args = parser.parse_args()

    path = args.extension.resolve() if args.extension is not None else extension_path()
    command = nm_command(path)
    if command is None:
        print(f"Skipping export audit on unsupported platform: {platform.system()}")
        return 0

    result = subprocess.run(command, check=True, text=True, capture_output=True)
    exported = result.stdout.splitlines()

    init_symbol = expected_init_symbol()
    if not any(init_symbol in line for line in exported):
        print(f"Expected exported Python init symbol not found: {init_symbol}", file=sys.stderr)
        print(f"Audited extension: {path}", file=sys.stderr)
        return 1

    stim_exports = [line for line in exported if "stim" in line]
    if stim_exports:
        print("Vendored Stim symbols are exported from the Python extension:", file=sys.stderr)
        for line in stim_exports[:20]:
            print(line, file=sys.stderr)
        if len(stim_exports) > 20:
            print(f"... {len(stim_exports) - 20} more", file=sys.stderr)
        print(f"Audited extension: {path}", file=sys.stderr)
        return 1

    print(f"Export audit passed for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
