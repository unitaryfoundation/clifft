#!/usr/bin/env python3
"""Verify that a macOS wheel keeps its statically linked OpenMP runtime private."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


def _is_openmp_symbol(symbol: str) -> bool:
    # Mach-O adds an underscore. LLVM also exposes Itanium-mangled template
    # functions and classes, including the weak definitions that dyld coalesces.
    return (
        re.match(
            r"^(?:(?:omp|kmpc?|GOMP)_|Z(?:N[KVr]*|T[ISV])?\d+_*kmp_)",
            symbol.lstrip("_"),
        )
        is not None
    )


def _wheel_members(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    extensions = [
        name
        for name in names
        if Path(name).name.startswith("_clifft_core") and name.endswith(".so")
    ]
    if len(extensions) != 1:
        raise RuntimeError(f"expected one _clifft_core extension, found {extensions}")
    bundled_libomp = [
        name for name in names if "libomp" in Path(name).name.lower() and name.endswith(".dylib")
    ]
    if bundled_libomp:
        raise RuntimeError(f"wheel contains a conflicting shared OpenMP runtime: {bundled_libomp}")
    return extensions[0]


def _dependencies(binary: Path) -> list[str]:
    completed = subprocess.run(
        ["otool", "-L", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip().split(" (compatibility version", maxsplit=1)[0]
        for line in completed.stdout.splitlines()[1:]
        if line.strip()
    ]


def audit(wheel: Path) -> None:
    extension_member = _wheel_members(wheel)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            extension = Path(archive.extract(extension_member, root))
        dependencies = _dependencies(extension)
        symbols = subprocess.run(
            ["nm", "-g", str(extension)], check=True, capture_output=True, text=True
        ).stdout.splitlines()

    # Neither undefined imports nor exported definitions may expose runtime
    # internals to another extension. Execution is checked by artifact_smoke.py.
    runtime_symbols = [
        line for line in symbols if line.split() and _is_openmp_symbol(line.split()[-1])
    ]
    if runtime_symbols:
        raise RuntimeError(f"extension exposes OpenMP runtime symbols: {runtime_symbols}")

    libomp_dependencies = [
        dependency for dependency in dependencies if "libomp" in Path(dependency).name.lower()
    ]
    if libomp_dependencies:
        raise RuntimeError(f"extension depends on a shared OpenMP runtime: {libomp_dependencies}")

    forbidden_prefixes = ("/opt/homebrew/", "/usr/local/")
    external = [
        dependency for dependency in dependencies if dependency.startswith(forbidden_prefixes)
    ]
    if external:
        raise RuntimeError(f"extension retains build-machine dependencies: {external}")

    print(f"wheel: {wheel}")
    print(f"extension: {extension_member}")
    print("macOS wheel OpenMP audit passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    audit(args.wheel.resolve())


if __name__ == "__main__":
    main()
