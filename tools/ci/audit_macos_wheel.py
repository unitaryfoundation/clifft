#!/usr/bin/env python3
"""Verify that a repaired macOS wheel carries and loads its OpenMP runtime."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path


def _wheel_members(wheel: Path) -> tuple[str, list[str]]:
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
    if not bundled_libomp:
        raise RuntimeError("wheel does not contain a bundled libomp dylib")
    return extensions[0], bundled_libomp


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
    extension_member, libomp_members = _wheel_members(wheel)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            extension = Path(archive.extract(extension_member, root))
        dependencies = _dependencies(extension)

    libomp_dependencies = [
        dependency for dependency in dependencies if "libomp" in Path(dependency).name.lower()
    ]
    if not libomp_dependencies:
        raise RuntimeError("_clifft_core does not declare a libomp dependency")
    if any(not dependency.startswith("@loader_path/") for dependency in libomp_dependencies):
        raise RuntimeError(f"libomp dependency is not wheel-relative: {libomp_dependencies}")

    bundled_names = {Path(member).name for member in libomp_members}
    referenced_names = {Path(dependency).name for dependency in libomp_dependencies}
    if not referenced_names <= bundled_names:
        raise RuntimeError(
            f"referenced libomp is not bundled: references={referenced_names}, "
            f"bundled={bundled_names}"
        )

    forbidden_prefixes = ("/opt/homebrew/", "/usr/local/")
    external = [
        dependency for dependency in dependencies if dependency.startswith(forbidden_prefixes)
    ]
    if external:
        raise RuntimeError(f"extension retains build-machine dependencies: {external}")

    print(f"wheel: {wheel}")
    print(f"extension: {extension_member}")
    print(f"bundled libomp: {', '.join(libomp_members)}")
    print(f"libomp load command: {', '.join(libomp_dependencies)}")
    print("macOS wheel OpenMP audit passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    audit(args.wheel.resolve())


if __name__ == "__main__":
    main()
