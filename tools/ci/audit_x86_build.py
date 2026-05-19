#!/usr/bin/env python3
"""Audit Linux x86_64 CI/release builds for accidental host-specific codegen."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
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


def symbol_from_objdump_line(line: str) -> str | None:
    match = re.match(r"^[0-9a-fA-F]+ <(.+)>:$", line.strip())
    if match:
        return match.group(1)
    return None


def audit_avx512(binary: Path, allowed_symbol: str) -> None:
    result = subprocess.run(
        ["objdump", "-d", "-C", str(binary)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        fail(f"objdump failed for {binary}:\n{result.stderr}")

    current_symbol = "<unknown>"
    offenders: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        symbol = symbol_from_objdump_line(line)
        if symbol is not None:
            current_symbol = symbol
            continue
        if "zmm" not in line:
            continue
        if allowed_symbol in current_symbol:
            continue
        offenders.append((current_symbol, line.strip()))

    if offenders:
        preview = "\n".join(f"  {symbol}: {line}" for symbol, line in offenders[:20])
        extra = "" if len(offenders) <= 20 else f"\n  ... and {len(offenders) - 20} more"
        fail(f"found AVX-512 instructions outside {allowed_symbol}:\n{preview}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-commands", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--allowed-avx512-symbol", default="clifft::avx512")
    args = parser.parse_args()

    if not args.compile_commands.exists():
        fail(f"compile commands file does not exist: {args.compile_commands}")
    if not args.binary.exists():
        fail(f"binary does not exist: {args.binary}")

    audit_compile_commands(args.compile_commands)
    audit_avx512(args.binary, args.allowed_avx512_symbol)
    print("x86 build audit passed")


if __name__ == "__main__":
    main()
