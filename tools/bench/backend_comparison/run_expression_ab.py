"""Run balanced same-build direct/incremental expression comparisons."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    case_id: str
    circuit: str
    mode: str
    shots: int


CASES = (
    Case("surface_d7_r7_aggregate", "surface_d7_r7_aggregate.stim", "survivors", 200_000),
    Case("surface_d7_r7_k0", "surface_d7_r7_aggregate.stim", "importance-k0", 200_000),
    Case("cultivation_d3_aggregate", "cultivation_d3_aggregate.stim", "survivors", 500_000),
    Case("cultivation_d3_k0", "cultivation_d3_aggregate.stim", "importance-k0", 400_000),
    Case("distillation_aggregate", "distillation_aggregate.stim", "survivors", 200_000),
    Case("coherent_d3_r3_aggregate", "coherent_d3_r3_aggregate.stim", "survivors", 300_000),
    Case("regime_k12_l512", "regime_k12_l512.stim", "survivors", 20_000),
)


def run_sample(
    binary: Path,
    circuits_dir: Path,
    case: Case,
    evaluator: str,
    cpu: int,
    seed: int,
    collect_census: bool = False,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    command = [
        "taskset",
        "-c",
        str(cpu),
        str(binary),
        str(circuits_dir / case.circuit),
        evaluator,
        case.mode,
        str(case.shots),
        f"--seed-base={seed}",
    ]
    if collect_census:
        command.append("--census")
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{case.case_id}/{evaluator} failed:\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout)


def write_result(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=280_000)
    args = parser.parse_args()
    if args.blocks <= 0:
        parser.error("--blocks must be positive")

    document: dict[str, Any] = {
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "binary": str(args.binary.resolve()),
        "circuits_dir": str(args.circuits_dir.resolve()),
        "cpu": args.cpu,
        "blocks": args.blocks,
        "cases": [asdict(case) for case in CASES],
        "samples": [],
        "census": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_result(args.output, document)

    for case in CASES:
        for block in range(args.blocks):
            evaluators = (
                ("direct", "incremental", "incremental", "direct")
                if block % 2 == 0
                else ("incremental", "direct", "direct", "incremental")
            )
            seed_zero = args.seed_base + block * 2
            seeds = (seed_zero, seed_zero, seed_zero + 1, seed_zero + 1)
            for position, (evaluator, seed) in enumerate(zip(evaluators, seeds, strict=True)):
                print(
                    f"{case.case_id} block={block + 1}/{args.blocks} "
                    f"position={position + 1}/4 evaluator={evaluator} seed={seed}",
                    flush=True,
                )
                sample = run_sample(
                    args.binary, args.circuits_dir, case, evaluator, args.cpu, seed
                )
                sample.update(
                    {
                        "case_id": case.case_id,
                        "block": block,
                        "position": position,
                        "paired_seed": seed,
                    }
                )
                document["samples"].append(sample)
                write_result(args.output, document)

    for case_index, case in enumerate(CASES):
        seed = args.seed_base + 10_000 + case_index
        print(f"{case.case_id} direct runtime census seed={seed}", flush=True)
        census = run_sample(
            args.binary,
            args.circuits_dir,
            case,
            "direct",
            args.cpu,
            seed,
            collect_census=True,
        )
        census["case_id"] = case.case_id
        document["census"].append(census)
        write_result(args.output, document)


if __name__ == "__main__":
    main()
