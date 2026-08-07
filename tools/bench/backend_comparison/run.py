"""Orchestrate balanced, fresh-process backend-comparison measurements."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from cases import CASE_BY_ID, CASES, PAPER_COMMIT, Case, load_circuit, sha256_text

import clifft

RELEVANT_ENV = (
    "CLIFFT_CPU_BASELINE",
    "CLIFFT_FORCE_ISA",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def command_output(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, check=False, text=True, capture_output=True)
    return (result.stdout + result.stderr).strip()


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def build_environment(repo_root: Path, cpu: int, perf_path: Path) -> dict[str, Any]:
    import clifft._clifft_core as core

    cmake_caches = []
    for path in sorted((repo_root / "build").glob("**/CMakeCache.txt")):
        selected = []
        for line in path.read_text(errors="replace").splitlines():
            if any(
                key in line
                for key in (
                    "CMAKE_BUILD_TYPE",
                    "CMAKE_CXX_COMPILER:",
                    "CMAKE_CXX_FLAGS",
                    "CLIFFT_CPU_BASELINE",
                    "OpenMP_CXX_FLAGS",
                )
            ):
                selected.append(line)
        cmake_caches.append({"path": str(path), "selected_entries": selected})

    governor_paths = sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
    governors = {str(path): read_optional(path) for path in governor_paths}
    return {
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "uname": command_output(["uname", "-a"], repo_root),
        "lscpu_json": command_output(["lscpu", "--json"], repo_root),
        "requested_cpu": cpu,
        "parent_affinity": sorted(os.sched_getaffinity(0)),
        "governors": governors,
        "perf_event_paranoid": read_optional(Path("/proc/sys/kernel/perf_event_paranoid")),
        "kptr_restrict": read_optional(Path("/proc/sys/kernel/kptr_restrict")),
        "perf_path": str(perf_path),
        "perf_version": command_output([str(perf_path), "--version"], repo_root),
        "python": sys.version,
        "python_executable": sys.executable,
        "clifft_version": getattr(clifft, "__version__", "unknown"),
        "clifft_extension": str(Path(core.__file__).resolve()),
        "svm_isa": clifft.svm_backend(),
        "cmake_version": command_output(["cmake", "--version"], repo_root).splitlines()[0],
        "compiler_version": command_output(["c++", "--version"], repo_root).splitlines()[0],
        "relevant_environment": {name: os.environ.get(name) for name in RELEVANT_ENV},
        "cmake_caches": cmake_caches,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def selected_cases(case_ids: list[str], include_extended: bool) -> list[Case]:
    if case_ids:
        return [CASE_BY_ID[case_id] for case_id in case_ids]
    return [case for case in CASES if include_extended or not case.extended]


def run_worker(
    worker: Path,
    repo_root: Path,
    paper_dir: Path,
    case: Case,
    backend: str,
    seed: int,
    cpu: int,
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
        sys.executable,
        str(worker),
        "--case",
        case.case_id,
        "--backend",
        backend,
        "--seed",
        str(seed),
        "--paper-dir",
        str(paper_dir),
        "--repo-root",
        str(repo_root),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"worker failed for {case.case_id}/{backend}:\n{result.stdout}\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"worker emitted {len(lines)} nonempty lines for {case.case_id}/{backend}:\n"
            + result.stdout
        )
    parsed = json.loads(lines[0])
    if not isinstance(parsed, dict):
        raise RuntimeError(f"worker emitted a non-object for {case.case_id}/{backend}")
    return parsed


def materialize_and_extract_metadata(
    cases: list[Case],
    repo_root: Path,
    paper_dir: Path,
    circuits_dir: Path,
    metadata_tool: Path,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    circuits_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        text, source = load_circuit(case, repo_root, paper_dir)
        path = circuits_dir / f"{case.case_id}.stim"
        path.write_text(text)
        entry: dict[str, Any] = {
            "path": str(path),
            "source": source,
            "sha256": sha256_text(text),
        }
        if case.kind != "noncomp":
            command = [str(metadata_tool), str(path)]
            if case.postselection == "all_detectors":
                command.append("--postselect-all")
            result = subprocess.run(
                command, cwd=repo_root, check=False, text=True, capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"metadata extractor failed for {case.case_id}:\n"
                    f"{result.stdout}\n{result.stderr}"
                )
            entry["sampling_plan"] = json.loads(result.stdout)
            if case.case_id == "exp_val_k8_200_raw":
                widths = entry["sampling_plan"]["exp_val_widths"]
                if len(widths) != 200 or set(widths) != {8}:
                    raise RuntimeError(
                        "active EXP_VAL fixture does not execute all 200 probes at width 8"
                    )
        metadata[case.case_id] = entry
    return metadata


def append_sample(document: dict[str, Any], output: Path, sample: dict[str, Any]) -> None:
    document["samples"].append(sample)
    atomic_write_json(output, document)


def run_balanced_cell(
    document: dict[str, Any],
    output: Path,
    worker: Path,
    repo_root: Path,
    paper_dir: Path,
    case: Case,
    blocks: int,
    cpu: int,
    seed_base: int,
    comparison: str = "backend",
) -> None:
    for block in range(blocks):
        if comparison == "aa_control":
            arms = (
                [("a", "legacy"), ("b", "legacy"), ("b", "legacy"), ("a", "legacy")]
                if block % 2 == 0
                else [("b", "legacy"), ("a", "legacy"), ("a", "legacy"), ("b", "legacy")]
            )
        else:
            arms = (
                [
                    ("legacy", "legacy"),
                    ("symbolic", "symbolic"),
                    ("symbolic", "symbolic"),
                    ("legacy", "legacy"),
                ]
                if block % 2 == 0
                else [
                    ("symbolic", "symbolic"),
                    ("legacy", "legacy"),
                    ("legacy", "legacy"),
                    ("symbolic", "symbolic"),
                ]
            )
        seed_zero = seed_base + block * 2
        seeds = (seed_zero, seed_zero, seed_zero + 1, seed_zero + 1)
        for position, ((arm, backend), seed) in enumerate(zip(arms, seeds, strict=True)):
            print(
                f"{comparison} {case.case_id} block={block + 1}/{blocks} "
                f"position={position + 1}/4 arm={arm} seed={seed}",
                flush=True,
            )
            sample = run_worker(worker, repo_root, paper_dir, case, backend, seed, cpu)
            sample.update(
                {
                    "comparison": comparison,
                    "arm": arm,
                    "block": block,
                    "position": position,
                }
            )
            append_sample(document, output, sample)


def paired_aa_ratios(samples: list[dict[str, Any]], case_id: str) -> list[float]:
    paired: dict[tuple[int, int], dict[str, float]] = {}
    for sample in samples:
        if sample["comparison"] != "aa_control" or sample["case"]["case_id"] != case_id:
            continue
        key = (int(sample["block"]), int(sample["seed"]))
        paired.setdefault(key, {})[sample["arm"]] = float(sample["sample_seconds"])
    return [values["b"] / values["a"] for values in paired.values() if values.keys() >= {"a", "b"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", required=True, type=Path)
    parser.add_argument("--metadata-tool", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case", action="append", default=[], choices=sorted(CASE_BY_ID))
    parser.add_argument("--include-extended", action="store_true")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=280_000)
    parser.add_argument("--skip-aa", action="store_true")
    parser.add_argument(
        "--perf",
        type=Path,
        default=Path("/usr/lib/linux-tools-6.8.0-106/perf"),
    )
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")

    repo_root = Path(__file__).resolve().parents[3]
    paper_dir = args.paper_dir.resolve()
    metadata_tool = args.metadata_tool.resolve()
    if not metadata_tool.is_file():
        parser.error(f"metadata tool not found: {metadata_tool}")
    if not args.perf.is_file():
        parser.error(f"perf binary not found: {args.perf}")
    cases = selected_cases(args.case, args.include_extended)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "tools/bench/backend_comparison/results" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "raw.json"
    worker = Path(__file__).with_name("worker.py")

    git_commit = command_output(["git", "rev-parse", "HEAD"], repo_root)
    if git_commit != "3fdafa411e8e4eca812ff17df2a8f30c584fdc03":
        raise RuntimeError(f"unexpected Clifft commit: {git_commit}")
    document: dict[str, Any] = {
        "schema": "clifft_backend_comparison_run_v1",
        "status": "running",
        "clifft_commit": git_commit,
        "paper_commit": PAPER_COMMIT,
        "git_status": command_output(["git", "status", "--short"], repo_root),
        "blocks": args.blocks,
        "samples_per_arm": args.blocks * 2,
        "cpu": args.cpu,
        "seed_base": args.seed_base,
        "matrix": [case.as_dict() for case in cases],
        "environment": build_environment(repo_root, args.cpu, args.perf.resolve()),
        "static_case_metadata": {},
        "samples": [],
    }
    atomic_write_json(output, document)
    document["static_case_metadata"] = materialize_and_extract_metadata(
        cases,
        repo_root,
        paper_dir,
        output_dir / "circuits",
        metadata_tool,
    )
    atomic_write_json(output, document)

    if not args.skip_aa:
        control = CASE_BY_ID["surface_d7_r7_aggregate"]
        run_balanced_cell(
            document,
            output,
            worker,
            repo_root,
            paper_dir,
            control,
            args.blocks,
            args.cpu,
            args.seed_base,
            "aa_control",
        )
        aa_ratios = paired_aa_ratios(document["samples"], control.case_id)
        document["surface_aa_ratio_median"] = statistics.median(aa_ratios)
        document["surface_aa_ratio_range"] = [min(aa_ratios), max(aa_ratios)]
        if min(aa_ratios) < 0.95 or max(aa_ratios) > 1.05:
            dense_control = CASE_BY_ID["regime_k12_l512"]
            if dense_control.case_id not in document["static_case_metadata"]:
                document["static_case_metadata"].update(
                    materialize_and_extract_metadata(
                        [dense_control],
                        repo_root,
                        paper_dir,
                        output_dir / "circuits",
                        metadata_tool,
                    )
                )
                atomic_write_json(output, document)
            run_balanced_cell(
                document,
                output,
                worker,
                repo_root,
                paper_dir,
                dense_control,
                args.blocks,
                args.cpu,
                args.seed_base + 500,
                "aa_control",
            )

    for index, case in enumerate(cases):
        run_balanced_cell(
            document,
            output,
            worker,
            repo_root,
            paper_dir,
            case,
            args.blocks,
            args.cpu,
            args.seed_base + 1_000 + index * 100,
        )

    document["status"] = "complete"
    document["completed_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    atomic_write_json(output, document)
    print(f"complete: {output}", flush=True)


if __name__ == "__main__":
    main()
