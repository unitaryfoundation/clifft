#!/usr/bin/env python3
"""Compare scalar, automatic, and explicit batching across public sampling APIs."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mode:
    api: str
    keep_records: int = 0
    postselection: str = "none"

    @property
    def name(self) -> str:
        if self.api in {"sample", "sample_k"}:
            return self.api
        return f"{self.api}/keep={self.keep_records}/ps={self.postselection}"


def parse_result(output: str) -> dict[str, str]:
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT "):
            return dict(field.split("=", 1) for field in line.split()[1:])
    raise RuntimeError("profile_sample did not emit a RESULT line")


def run_case(
    executable: Path,
    circuit: Path,
    mode: Mode,
    batch: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CLIFFT_CIRCUIT_FILE": str(circuit),
            "CLIFFT_PROFILE_API": mode.api,
            "CLIFFT_PROFILE_KEEP_RECORDS": str(mode.keep_records),
            "CLIFFT_PROFILE_POSTSELECTION": mode.postselection,
            "CLIFFT_PROFILE_FIXED_K": str(args.fixed_k),
            "CLIFFT_PROFILE_SHOTS": str(args.shots),
            "CLIFFT_PROFILE_THREADS": str(args.threads),
            "CLIFFT_PROFILE_WARMUPS": str(args.warmups),
            "CLIFFT_PROFILE_REPETITIONS": str(args.repetitions),
            "CLIFFT_PROFILE_BATCH_SIZE": batch,
        }
    )
    if mode.api not in {"sample_k", "sample_k_survivors"}:
        environment.pop("CLIFFT_PROFILE_FIXED_K")
    completed = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode.name} batch={batch} failed with exit {completed.returncode}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    result = parse_result(completed.stdout)
    result["mode"] = mode.name
    result["circuit"] = str(circuit)
    return result


def classify(scalar_ms: float, auto: dict[str, str], threshold: float) -> str:
    if int(auto["effective_batch"]) == 1:
        return "auto-ineligible"
    auto_ms = float(auto["median_ms"])
    if auto_ms > scalar_ms * (1.0 + threshold):
        return "regression"
    if auto_ms < scalar_ms * (1.0 - threshold):
        return "win"
    return "neutral"


def format_change(scalar_ms: float, candidate_ms: float) -> str:
    return f"{(candidate_ms / scalar_ms - 1.0) * 100.0:+.1f}%"


def build_modes(
    apis: list[str], keep_records_modes: list[int], postselection_modes: list[str]
) -> list[Mode]:
    modes = [Mode(api) for api in ("sample", "sample_k") if api in apis]
    for api in ("sample_survivors", "sample_k_survivors"):
        if api not in apis:
            continue
        for keep_records in keep_records_modes:
            for postselection in postselection_modes:
                modes.append(Mode(api, keep_records, postselection))
    return modes


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["circuit", "mode"]
    fields.extend(key for key in rows[0] if key not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-profile/profile_sample"))
    parser.add_argument(
        "--circuit", type=Path, default=Path("tests/fixtures/surface_d7_r7_p001.stim")
    )
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--fixed-k", type=int, default=1)
    parser.add_argument(
        "--apis",
        nargs="+",
        default=["sample", "sample_survivors", "sample_k", "sample_k_survivors"],
        choices=["sample", "sample_survivors", "sample_k", "sample_k_survivors"],
    )
    parser.add_argument("--keep-records", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--batches", nargs="+", default=["1", "auto", "64", "256", "1024", "2048"])
    parser.add_argument(
        "--postselection",
        nargs="+",
        default=["none", "all"],
        choices=["none", "all", "first-half", "last-half", "alternating"],
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.shots < 1 or args.threads < 0 or args.warmups < 0 or args.repetitions < 1:
        parser.error("shots and repetitions must be positive; threads and warmups non-negative")
    if "1" not in args.batches or "auto" not in args.batches:
        parser.error("batches must include both 1 and auto")
    if not 0.0 <= args.threshold < 1.0:
        parser.error("threshold must be in [0, 1)")

    executable = args.executable.resolve()
    circuit = args.circuit.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    if not circuit.is_file():
        parser.error(f"circuit does not exist: {circuit}")

    modes = build_modes(args.apis, args.keep_records, args.postselection)
    rows: list[dict[str, str]] = []
    total = len(modes) * len(args.batches)
    completed = 0
    for mode_index, mode in enumerate(modes):
        batches = args.batches if mode_index % 2 == 0 else list(reversed(args.batches))
        for batch in batches:
            completed += 1
            print(
                f"[{completed:02d}/{total:02d}] {mode.name} batch={batch}",
                file=sys.stderr,
                flush=True,
            )
            rows.append(run_case(executable, circuit, mode, batch, args))

    if args.output is not None:
        write_csv(args.output, rows)

    print(
        "| Mode | Auto lanes | Scalar ms | Auto ms | Auto delta | Class | "
        "Best explicit | Survival |"
    )
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for mode in modes:
        mode_rows = {row["requested_batch"]: row for row in rows if row["mode"] == mode.name}
        scalar = mode_rows["1"]
        auto = mode_rows["auto"]
        scalar_ms = float(scalar["median_ms"])
        auto_ms = float(auto["median_ms"])
        explicit = [row for batch, row in mode_rows.items() if batch not in {"1", "auto"}]
        if explicit:
            best = min(explicit, key=lambda row: float(row["median_ms"]))
            best_explicit = (
                f"{best['requested_batch']}: {float(best['median_ms']):.3f} ms "
                f"({format_change(scalar_ms, float(best['median_ms']))})"
            )
        else:
            best_explicit = "n/a"
        print(
            f"| {mode.name} | {auto['effective_batch']} | {scalar_ms:.3f} | "
            f"{auto_ms:.3f} | {format_change(scalar_ms, auto_ms)} | "
            f"{classify(scalar_ms, auto, args.threshold)} | {best_explicit} | "
            f"{float(auto['survival']):.3%} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
