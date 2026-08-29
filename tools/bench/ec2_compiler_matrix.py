#!/usr/bin/env python3
"""Collect and publish the issue 317 compiler matrix on an EC2 x86 host."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BRANCH = "codex/issue-317-ec2-compiler-matrix"
CORPUS_REPOSITORY = "https://github.com/unitaryfoundation/clifft-bench.git"
CORPUS_COMMIT = "ac97bbca5c5f2cc765eef0311611980d11e91d94"
RESULT_ROOT = Path("tools/bench/ec2-results/issue-317")
WORK_ROOT = Path(".ec2-compiler-matrix-work")
ASSISTED_BY = "Assisted-by: Codex (GPT-5) <noreply@openai.com>"
UV_VERSION = "0.12.5"


@dataclass(frozen=True)
class BuildConfiguration:
    name: str
    family: str
    compiler_candidates: tuple[str, ...]
    lto: bool


@dataclass(frozen=True)
class Workload:
    name: str
    relative_path: str
    source: str
    sha256: str
    peak_rank: int
    shots: int
    batch_size: str
    api: str
    postselection: str


CONFIGURATIONS = {
    "gcc": BuildConfiguration("gcc", "gcc", ("g++-13",), False),
    "gcc-lto": BuildConfiguration("gcc-lto", "gcc", ("g++-13",), True),
    "clang": BuildConfiguration("clang", "clang", ("clang++-18",), False),
    "clang-thinlto": BuildConfiguration("clang-thinlto", "clang", ("clang++-18",), True),
}

WORKLOADS = {
    "surface-d7-r7": Workload(
        "surface-d7-r7",
        "workloads/circuits/pure_surface_d7_r7_p1e-3.stim",
        "corpus",
        "30d1940101d70e05a63f0d2f877756ffaeaba7e8e17e94cd2ea40fce04b99583",
        0,
        300_000,
        "auto",
        "sample_survivors",
        "all",
    ),
    "cultivation-d3": Workload(
        "cultivation-d3",
        "workloads/circuits/msc_d3_inject_cultivate_p1e-3.stim",
        "corpus",
        "90a7d841e003e5ee38137cd9a3eb6529bb552e49c424bc6b0932a27d97cdb41f",
        4,
        150_000,
        "auto",
        "sample_survivors",
        "all",
    ),
    "distillation": Workload(
        "distillation",
        "workloads/circuits/distillation.stim",
        "corpus",
        "188bd53c48dbc21f840fb297df6f41c61f5bad6a856bba621f00ff42078921c1",
        5,
        100_000,
        "auto",
        "sample_survivors",
        "all",
    ),
    "coherent-d3-r3": Workload(
        "coherent-d3-r3",
        "workloads/circuits/coherent_d3_r3.stim",
        "corpus",
        "87d1308c83894e87c60aeb2dc31b74be89b3460a951929951f2c3ac92606827d",
        7,
        50_000,
        "auto",
        "sample_survivors",
        "all",
    ),
    "cultivation-d5": Workload(
        "cultivation-d5",
        "workloads/circuits/msc_d5_inject_cultivate_p1e-3.stim",
        "corpus",
        "c2b4566917bd9bf27a5705284dac02700ef0dcc7c03c91066670db376d633a6d",
        10,
        20_000,
        "auto",
        "sample_survivors",
        "all",
    ),
    "qv20": Workload(
        "qv20",
        "tools/bench/fixtures/qv20_seed42.stim",
        "repository",
        "6b3198b30e3cb9bf6233e9e3b3bb283d7f31d1d98f17afa05c80b4ece3c6edf8",
        20,
        1,
        "1",
        "sample",
        "none",
    ),
}


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {rendered}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def git(repo: Path, *arguments: str, check: bool = True) -> str:
    completed = run_command(["git", *arguments], cwd=repo, check=check)
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value) is None:
        raise argparse.ArgumentTypeError(
            "execution ID must use 1-80 letters, digits, dots, underscores, or dashes"
        )
    return value


def parse_selection(value: str, available: dict[str, Any], label: str) -> list[str]:
    selected = value.split(",")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError(f"duplicate {label} selection")
    return selected


def require_clean_tracked_tree(repo: Path) -> None:
    status = git(repo, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            f"tracked worktree changes must be committed before collection:\n{status}"
        )


def require_branch(repo: Path) -> None:
    branch = git(repo, "branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH!r}, found {branch!r}")


def install_dependencies(repo: Path) -> None:
    if not Path("/etc/os-release").exists():
        raise RuntimeError("dependency installation is supported only on Ubuntu")
    release = Path("/etc/os-release").read_text()
    if "ID=ubuntu" not in release or 'VERSION_ID="24.04"' not in release:
        raise RuntimeError("expected Ubuntu 24.04 for the reference EC2 run")
    packages = [
        "build-essential",
        "clang-18",
        "cmake",
        "g++-13",
        "gcc-13",
        "git",
        "libomp-18-dev",
        "lld-18",
        "llvm-18",
        "ninja-build",
        "pipx",
        "python3",
        "util-linux",
    ]
    run_command(["sudo", "apt-get", "update"], cwd=repo)
    run_command(["sudo", "apt-get", "install", "-y", *packages], cwd=repo)
    perf_packages = ["linux-tools-common", f"linux-tools-{platform.release()}"]
    completed = run_command(
        ["sudo", "apt-get", "install", "-y", *perf_packages], cwd=repo, check=False
    )
    if completed.returncode != 0:
        print("warning: matching perf tools were not installed; wall timings remain available")
    if find_uv() is None:
        run_command(["pipx", "install", f"uv=={UV_VERSION}"], cwd=repo)


def resolve_tool(candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    raise RuntimeError(f"required executable was not found: {' or '.join(candidates)}")


def find_uv() -> str | None:
    resolved = shutil.which("uv")
    if resolved is not None:
        return resolved
    local = Path.home() / ".local/bin/uv"
    return str(local) if local.is_file() else None


def prepare_corpus(repo: Path, work_root: Path, override: Path | None) -> Path:
    if override is not None:
        corpus = override.resolve()
    else:
        corpus = work_root / "clifft-bench-corpus"
        if not (corpus / ".git").exists():
            corpus.mkdir(parents=True, exist_ok=True)
            git(corpus, "init")
            git(corpus, "remote", "add", "origin", CORPUS_REPOSITORY)
        git(corpus, "fetch", "--depth", "1", "origin", CORPUS_COMMIT)
        git(corpus, "checkout", "--detach", "--force", "FETCH_HEAD")
        actual = git(corpus, "rev-parse", "HEAD")
        if actual != CORPUS_COMMIT:
            raise RuntimeError(f"expected corpus commit {CORPUS_COMMIT}, found {actual}")
    for workload in WORKLOADS.values():
        if workload.source != "corpus":
            continue
        path = corpus / workload.relative_path
        if not path.is_file() or sha256(path) != workload.sha256:
            raise RuntimeError(f"corpus digest mismatch: {path}")
    return corpus


def resolve_workload_path(repo: Path, corpus: Path, workload: Workload) -> Path:
    base = corpus if workload.source == "corpus" else repo
    path = base / workload.relative_path
    if not path.is_file() or sha256(path) != workload.sha256:
        raise RuntimeError(f"workload digest mismatch: {path}")
    return path


def compiler_metadata(compiler: str) -> dict[str, str]:
    completed = run_command([compiler, "--version"], cwd=Path.cwd())
    return {"path": compiler, "version": completed.stdout.strip()}


def build_configuration(
    repo: Path,
    work_root: Path,
    result_dir: Path,
    configuration: BuildConfiguration,
    jobs: int,
    openmp: str,
    fast_float_dir: Path | None,
) -> dict[str, Any]:
    compiler = resolve_tool(configuration.compiler_candidates)
    build_dir = work_root / "builds" / configuration.name
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    cmake_command = [
        "cmake",
        "-S",
        str(repo),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DCMAKE_INTERPROCEDURAL_OPTIMIZATION={'ON' if configuration.lto else 'OFF'}",
        "-DCLIFFT_CPU_BASELINE=x86-64-v2",
        f"-DCLIFFT_OPENMP={openmp}",
        "-DCLIFFT_BUILD_PROFILER=ON",
        "-DCLIFFT_BUILD_TESTS=OFF",
    ]
    clang_linker = None
    if configuration.family == "clang":
        clang_linker = shutil.which("ld.lld-18") or shutil.which("ld.lld")
        if clang_linker is not None:
            cmake_command.extend(
                [
                    f"-DCMAKE_EXE_LINKER_FLAGS=--ld-path={clang_linker}",
                    f"-DCMAKE_SHARED_LINKER_FLAGS=--ld-path={clang_linker}",
                ]
            )
    if fast_float_dir is not None:
        cmake_command.append(f"-DFETCHCONTENT_SOURCE_DIR_FAST_FLOAT={fast_float_dir}")
    configured = run_command(cmake_command, cwd=repo)
    built = run_command(
        ["cmake", "--build", str(build_dir), "--target", "profile_sample", f"-j{jobs}"],
        cwd=repo,
    )
    audit = run_command(
        [
            sys.executable,
            str(repo / "tools/ci/audit_build_flags.py"),
            "--compile-commands",
            str(build_dir / "compile_commands.json"),
        ],
        cwd=repo,
    )

    build_logs = result_dir / "build-logs"
    build_logs.mkdir(parents=True, exist_ok=True)
    (build_logs / f"{configuration.name}-configure.txt").write_text(
        configured.stdout + configured.stderr
    )
    (build_logs / f"{configuration.name}-build.txt").write_text(built.stdout + built.stderr)
    (build_logs / f"{configuration.name}-audit.txt").write_text(audit.stdout + audit.stderr)

    compile_commands = json.loads((build_dir / "compile_commands.json").read_text())
    commands = [entry["command"] for entry in compile_commands]
    has_lto = any("-flto" in command for command in commands)
    if has_lto != configuration.lto:
        raise RuntimeError(f"unexpected LTO flags for {configuration.name}")
    if configuration.name == "clang-thinlto" and not any(
        "-flto=thin" in command for command in commands
    ):
        raise RuntimeError("Clang IPO did not select ThinLTO")

    openmp_match = re.search(r"CLIFFT_OPENMP = [^\n]*\(enabled: (ON|OFF)\)", configured.stdout)
    if openmp_match is None:
        raise RuntimeError(f"could not identify OpenMP status for {configuration.name}")
    binary = build_dir / "profile_sample"
    if not binary.is_file():
        raise RuntimeError(f"profile binary was not built: {binary}")
    return {
        "name": configuration.name,
        "family": configuration.family,
        "lto": configuration.lto,
        "compiler": compiler_metadata(compiler),
        "clang_linker": clang_linker or "compiler default",
        "openmp_enabled": openmp_match.group(1) == "ON",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "compile_commands_sha256": sha256(build_dir / "compile_commands.json"),
        "cmake_command": cmake_command,
    }


def cpu_flags() -> set[str]:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def validate_isa(isa: str) -> None:
    required = {
        "scalar": set(),
        "avx2": {"avx2", "bmi2", "fma"},
        "avx512": {"avx2", "bmi2", "fma", "avx512f", "avx512dq"},
    }[isa]
    missing = sorted(required - cpu_flags())
    if missing:
        raise RuntimeError(f"CPU does not support {isa}; missing: {', '.join(missing)}")


def choose_cpu(requested: int | None) -> int:
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("the process has no available CPUs")
    if requested is not None:
        if requested not in available:
            raise RuntimeError(f"CPU {requested} is outside process affinity {available}")
        return requested
    return available[-1]


def perf_available(repo: Path, mode: str) -> tuple[bool, str | None]:
    if mode == "off":
        return False, "disabled"
    perf = shutil.which("perf")
    if perf is None:
        if mode == "on":
            raise RuntimeError("perf was requested but is not installed")
        return False, "not installed"
    completed = run_command(
        [perf, "stat", "-x,", "-e", "cycles:u", "--", "true"], cwd=repo, check=False
    )
    if completed.returncode != 0 or ",cycles:u," not in completed.stderr:
        reason = completed.stderr.strip() or f"exit code {completed.returncode}"
        if mode == "on":
            raise RuntimeError(f"perf was requested but is unavailable: {reason}")
        return False, reason
    return True, None


def profile_environment(
    workload: Workload,
    path: Path,
    *,
    isa: str,
    warmups: int,
    repetitions: int,
    quick: bool,
) -> dict[str, str]:
    shots = min(workload.shots, 1024) if quick and workload.shots > 1 else workload.shots
    return {
        "CLIFFT_CIRCUIT_FILE": str(path),
        "CLIFFT_FORCE_ISA": isa,
        "CLIFFT_PROFILE_API": workload.api,
        "CLIFFT_PROFILE_BATCH_SIZE": workload.batch_size,
        "CLIFFT_PROFILE_KEEP_RECORDS": "0",
        "CLIFFT_PROFILE_POSTSELECTION": workload.postselection,
        "CLIFFT_PROFILE_REPETITIONS": str(repetitions),
        "CLIFFT_PROFILE_SHOTS": str(shots),
        "CLIFFT_PROFILE_THREADS": "1",
        "CLIFFT_PROFILE_WARMUPS": str(warmups),
        "OMP_NUM_THREADS": "1",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "true",
    }


def parse_result(stdout: str) -> dict[str, Any]:
    result_line = next((line for line in stdout.splitlines() if line.startswith("RESULT ")), None)
    if result_line is None:
        raise RuntimeError(f"profile output has no RESULT line:\n{stdout}")
    result: dict[str, Any] = {}
    for item in result_line.removeprefix("RESULT ").split():
        key, value = item.split("=", 1)
        if key == "checksum":
            result[key] = value
        elif re.fullmatch(r"-?[0-9]+", value):
            result[key] = int(value)
        elif re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?", value, re.I):
            result[key] = float(value)
        else:
            result[key] = value
    plan_match = re.search(
        r"Plan: ([0-9]+) qubits, peak active width ([0-9]+), ([0-9]+) actions", stdout
    )
    if plan_match is None:
        raise RuntimeError("profile output has no plan metadata")
    result.update(
        {
            "plan_qubits": int(plan_match.group(1)),
            "peak_active_width": int(plan_match.group(2)),
            "plan_actions": int(plan_match.group(3)),
        }
    )
    return result


def parse_perf(stderr: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    for row in csv.reader(stderr.splitlines()):
        if len(row) < 3:
            continue
        event = row[2]
        if event not in {"cycles:u", "instructions:u"}:
            continue
        try:
            counters[event.removesuffix(":u")] = int(row[0])
        except ValueError:
            continue
    return counters


def run_profile(
    repo: Path,
    binary: Path,
    environment: dict[str, str],
    cpu: int,
    use_perf: bool,
) -> tuple[dict[str, Any], dict[str, int], str, str]:
    command = ["taskset", "-c", str(cpu)]
    if use_perf:
        command.extend(["perf", "stat", "-x,", "-e", "cycles:u,instructions:u", "--"])
    command.append(str(binary))
    child_env = os.environ.copy()
    child_env.update(environment)
    child_env["LC_ALL"] = "C"
    completed = run_command(command, cwd=repo, env=child_env, timeout=1800)
    return (
        parse_result(completed.stdout),
        parse_perf(completed.stderr),
        completed.stdout,
        completed.stderr,
    )


def validate_dispatch(
    repo: Path,
    binaries: dict[str, Path],
    qv_path: Path,
    cpu: int,
) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    checksums: set[str] = set()
    workload = WORKLOADS["qv20"]
    for configuration, binary in binaries.items():
        for isa in ("scalar", "avx2", "avx512"):
            validate_isa(isa)
            environment = profile_environment(
                workload, qv_path, isa=isa, warmups=0, repetitions=1, quick=False
            )
            result, _, _, _ = run_profile(repo, binary, environment, cpu, False)
            checksums.add(result["checksum"])
            validations.append(
                {
                    "configuration": configuration,
                    "isa": isa,
                    "checksum": result["checksum"],
                    "peak_active_width": result["peak_active_width"],
                }
            )
    if len(checksums) != 1:
        raise RuntimeError(f"dispatch smoke checksums differ: {sorted(checksums)}")
    return validations


def imds_metadata() -> dict[str, str] | None:
    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with urllib.request.urlopen(token_request, timeout=1) as response:
            token = response.read().decode()
        values = {}
        for key, endpoint in {
            "instance_id": "instance-id",
            "instance_type": "instance-type",
            "ami_id": "ami-id",
            "availability_zone": "placement/availability-zone",
        }.items():
            request = urllib.request.Request(
                f"http://169.254.169.254/latest/meta-data/{endpoint}",
                headers={"X-aws-ec2-metadata-token": token},
            )
            with urllib.request.urlopen(request, timeout=1) as response:
                values[key] = response.read().decode()
        values["region"] = values["availability_zone"][:-1]
        return values
    except (OSError, urllib.error.URLError):
        return None


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def machine_metadata(repo: Path, cpu: int, perf_status: tuple[bool, str | None]) -> dict[str, Any]:
    tools: dict[str, str | None] = {}
    for name in ("cmake", "ninja", "perf", "python3"):
        executable = shutil.which(name)
        if executable is None:
            tools[name] = None
            continue
        completed = run_command([executable, "--version"], cwd=repo, check=False)
        tools[name] = (completed.stdout or completed.stderr).strip()
    lscpu = run_command(["lscpu", "--json"], cwd=repo, check=False)
    sibling_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "boot_id": read_optional(Path("/proc/sys/kernel/random/boot_id")),
        "os_release": read_optional(Path("/etc/os-release")),
        "lscpu": json.loads(lscpu.stdout) if lscpu.returncode == 0 else lscpu.stderr,
        "cpu_flags": sorted(cpu_flags()),
        "benchmark_cpu": cpu,
        "benchmark_cpu_thread_siblings": read_optional(sibling_path),
        "load_average": list(os.getloadavg()),
        "ec2": imds_metadata(),
        "perf_available": perf_status[0],
        "perf_unavailable_reason": perf_status[1],
        "tools": tools,
    }


def make_summary(
    configurations: list[str], workloads: list[str], runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summary = []
    for workload_name in workloads:
        baseline_runs = [
            run
            for run in runs
            if run["workload"] == workload_name and run["configuration"] == "gcc"
        ]
        baseline = (
            statistics.median(run["result"]["median_ms"] for run in baseline_runs)
            if baseline_runs
            else None
        )
        for configuration in configurations:
            selected = [
                run
                for run in runs
                if run["workload"] == workload_name and run["configuration"] == configuration
            ]
            medians = [run["result"]["median_ms"] for run in selected]
            cycle_counts = [run["perf"]["cycles"] for run in selected if "cycles" in run["perf"]]
            instruction_counts = [
                run["perf"]["instructions"] for run in selected if "instructions" in run["perf"]
            ]
            aggregate = statistics.median(medians)
            summary.append(
                {
                    "workload": workload_name,
                    "peak_rank": WORKLOADS[workload_name].peak_rank,
                    "configuration": configuration,
                    "processes": len(selected),
                    "median_ms": aggregate,
                    "minimum_process_median_ms": min(medians),
                    "maximum_process_median_ms": max(medians),
                    "wall_delta_vs_gcc_pct": (
                        (aggregate / baseline - 1.0) * 100.0 if baseline is not None else None
                    ),
                    "median_process_cycles": (
                        statistics.median(cycle_counts) if cycle_counts else None
                    ),
                    "median_process_instructions": (
                        statistics.median(instruction_counts) if instruction_counts else None
                    ),
                    "effective_batches": sorted(
                        {run["result"]["effective_batch"] for run in selected}
                    ),
                    "checksums": sorted({run["result"]["checksum"] for run in selected}),
                }
            )
    return summary


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    fieldnames = list(summary[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


def write_result_readme(path: Path, result: dict[str, Any]) -> None:
    ec2 = result["machine"]["ec2"] or {}
    lscpu_data = result["machine"]["lscpu"]
    lscpu_rows = lscpu_data.get("lscpu", []) if isinstance(lscpu_data, dict) else []
    cpu_model = next(
        (row["data"] for row in lscpu_rows if row.get("field") == "Model name:"), "unknown"
    )
    lines = [
        f"# EC2 compiler matrix: {result['execution_id']}",
        "",
        f"- Source commit: `{result['source']['commit']}`",
        f"- Source branch: `{result['source']['branch']}`",
        f"- Instance: `{ec2.get('instance_type', 'not EC2')}`",
        f"- CPU: `{cpu_model}`",
        f"- Benchmark CPU: `{result['machine']['benchmark_cpu']}`",
        f"- Primary ISA: `{result['settings']['isa']}`",
        f"- OpenMP request: `{result['settings']['openmp']}`",
        "",
        "`result.json` is authoritative. `summary.csv` is derived from its per-process records.",
        "Each workload rotates configuration order across collection rounds.",
        "",
    ]
    path.write_text("\n".join(lines))


def collect(args: argparse.Namespace, repo: Path) -> Path:
    require_branch(repo)
    if not args.quick:
        require_clean_tracked_tree(repo)
    validate_isa(args.isa)
    configurations = parse_selection(args.configurations, CONFIGURATIONS, "configuration")
    workloads = parse_selection(args.workloads, WORKLOADS, "workload")
    if "gcc" not in configurations:
        raise RuntimeError("the gcc baseline must be included in every collected matrix")

    work_root = (repo / args.work_dir).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    if args.fast_float_dir is not None:
        args.fast_float_dir = args.fast_float_dir.resolve()
        if not (args.fast_float_dir / "include/fast_float/fast_float.h").is_file():
            raise RuntimeError(f"invalid fast_float source directory: {args.fast_float_dir}")
    result_root = work_root / "quick-results" if args.quick else repo / RESULT_ROOT
    final_dir: Path = result_root / args.execution_id
    incomplete_dir = final_dir.with_name(f".{args.execution_id}.incomplete")
    if final_dir.exists() or incomplete_dir.exists():
        raise RuntimeError(f"execution ID already exists: {args.execution_id}")
    incomplete_dir.mkdir(parents=True)

    corpus = prepare_corpus(repo, work_root, args.corpus_dir)
    cpu = choose_cpu(args.cpu)
    perf_status = perf_available(repo, args.perf)
    build_metadata: dict[str, dict[str, Any]] = {}
    binaries: dict[str, Path] = {}
    for name in configurations:
        print(f"building {name}", flush=True)
        metadata = build_configuration(
            repo,
            work_root,
            incomplete_dir,
            CONFIGURATIONS[name],
            args.jobs,
            args.openmp,
            args.fast_float_dir,
        )
        build_metadata[name] = metadata
        binaries[name] = Path(metadata["binary"])

    enabled_states = {metadata["openmp_enabled"] for metadata in build_metadata.values()}
    if len(enabled_states) != 1:
        raise RuntimeError("OpenMP must have the same enabled state in every configuration")
    if args.openmp == "ON" and enabled_states != {True}:
        raise RuntimeError("OpenMP was requested but is not enabled")

    qv_path = resolve_workload_path(repo, corpus, WORKLOADS["qv20"])
    validations = validate_dispatch(repo, binaries, qv_path, cpu)
    outer_repetitions = 1 if args.quick else args.outer_repetitions
    profile_repetitions = 1 if args.quick else args.profile_repetitions
    warmups = 0 if args.quick else args.warmups
    runs: list[dict[str, Any]] = []
    for workload_name in workloads:
        workload = WORKLOADS[workload_name]
        path = resolve_workload_path(repo, corpus, workload)
        checksums: set[str] = set()
        for round_index in range(outer_repetitions):
            offset = round_index % len(configurations)
            order = configurations[offset:] + configurations[:offset]
            for order_index, configuration in enumerate(order):
                print(
                    f"running {workload_name} round {round_index + 1}/{outer_repetitions} "
                    f"with {configuration}",
                    flush=True,
                )
                environment = profile_environment(
                    workload,
                    path,
                    isa=args.isa,
                    warmups=warmups,
                    repetitions=profile_repetitions,
                    quick=args.quick,
                )
                started = time.monotonic()
                result, counters, stdout, stderr = run_profile(
                    repo, binaries[configuration], environment, cpu, perf_status[0]
                )
                checksums.add(result["checksum"])
                if result["peak_active_width"] != workload.peak_rank:
                    raise RuntimeError(
                        f"{workload_name} expected peak rank {workload.peak_rank}, "
                        f"observed {result['peak_active_width']}"
                    )
                runs.append(
                    {
                        "workload": workload_name,
                        "configuration": configuration,
                        "round": round_index + 1,
                        "order_in_round": order_index + 1,
                        "elapsed_seconds": time.monotonic() - started,
                        "environment": environment,
                        "result": result,
                        "perf": counters,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                )
        if len(checksums) != 1:
            raise RuntimeError(f"{workload_name} checksums differ: {sorted(checksums)}")

    summary = make_summary(configurations, workloads, runs)
    result = {
        "schema_version": 1,
        "status": "complete",
        "execution_id": args.execution_id,
        "source": {
            "branch": git(repo, "branch", "--show-current"),
            "commit": git(repo, "rev-parse", "HEAD"),
            "describe": git(repo, "describe", "--always", "--dirty"),
        },
        "corpus": {
            "repository": CORPUS_REPOSITORY,
            "commit": CORPUS_COMMIT,
            "path": str(corpus),
        },
        "settings": {
            "isa": args.isa,
            "openmp": args.openmp,
            "outer_repetitions": outer_repetitions,
            "profile_repetitions": profile_repetitions,
            "warmups": warmups,
            "quick": args.quick,
            "configuration_order": configurations,
            "workload_order": workloads,
        },
        "machine": machine_metadata(repo, cpu, perf_status),
        "builds": build_metadata,
        "dispatch_validation": validations,
        "workloads": {name: WORKLOADS[name].__dict__ for name in workloads},
        "runs": runs,
        "summary": summary,
    }
    (incomplete_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_summary_csv(incomplete_dir / "summary.csv", summary)
    write_result_readme(incomplete_dir / "README.md", result)
    incomplete_dir.rename(final_dir)
    try:
        display_path = final_dir.relative_to(repo)
    except ValueError:
        display_path = final_dir
    print(f"completed result: {display_path}")
    return final_dir


def publish(args: argparse.Namespace, repo: Path) -> None:
    require_branch(repo)
    result_dir = repo / RESULT_ROOT / args.execution_id
    result_file = result_dir / "result.json"
    if not result_file.is_file():
        raise RuntimeError(f"completed result does not exist: {result_file}")
    result = json.loads(result_file.read_text())
    if result.get("status") != "complete" or result.get("execution_id") != args.execution_id:
        raise RuntimeError("result bundle is not complete or has the wrong execution ID")
    if result.get("settings", {}).get("quick"):
        raise RuntimeError("quick validation results are not publishable")
    write_summary_csv(result_dir / "summary.csv", result["summary"])

    status = git(repo, "status", "--porcelain", "--untracked-files=all")
    allowed_prefix = str(result_dir.relative_to(repo)) + "/"
    unexpected = []
    for line in status.splitlines():
        path = line[3:]
        if not path.startswith(allowed_prefix):
            unexpected.append(line)
    if unexpected:
        raise RuntimeError("refusing to publish with unrelated changes:\n" + "\n".join(unexpected))

    git(repo, "add", "--", str(result_dir.relative_to(repo)))
    uv = find_uv()
    if uv is None:
        raise RuntimeError("uv is required for the pre-commit check; run install-deps first")
    precommit_env = os.environ.copy()
    precommit_env["UV_CACHE_DIR"] = str((repo / WORK_ROOT / "uv-cache").resolve())
    run_command(
        [
            uv,
            "run",
            "--frozen",
            "--only-group",
            "dev",
            "pre-commit",
            "run",
            "--all-files",
            "--show-diff-on-failure",
        ],
        cwd=repo,
        env=precommit_env,
    )
    status_after_check = git(repo, "status", "--porcelain", "--untracked-files=all")
    unexpected_after_check = [
        line for line in status_after_check.splitlines() if not line[3:].startswith(allowed_prefix)
    ]
    if unexpected_after_check:
        raise RuntimeError(
            "pre-commit changed files outside the result bundle:\n"
            + "\n".join(unexpected_after_check)
        )
    git(repo, "add", "--", str(result_dir.relative_to(repo)))
    staged = git(repo, "diff", "--cached", "--name-only").splitlines()
    if not staged or any(not path.startswith(allowed_prefix) for path in staged):
        raise RuntimeError("staged paths are empty or extend beyond the result bundle")
    run_command(["git", "diff", "--cached", "--check"], cwd=repo)
    message = f"bench: add EC2 compiler matrix {args.execution_id}"
    run_command(["git", "commit", "--no-gpg-sign", "-m", message, "-m", ASSISTED_BY], cwd=repo)
    if args.push:
        run_command(["git", "push", "-u", "origin", f"HEAD:refs/heads/{BRANCH}"], cwd=repo)
        print(f"pushed result commit to origin/{BRANCH}")
    else:
        print("created the result commit; rerun publish with --push to send it to GitHub")


def default_execution_id() -> str:
    return "intel-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("install-deps", help="install pinned Ubuntu 24.04 build dependencies")

    collect_parser = subparsers.add_parser("run", help="build and collect the compiler matrix")
    collect_parser.add_argument(
        "--execution-id", type=checked_identifier, default=default_execution_id()
    )
    collect_parser.add_argument(
        "--configurations", default=",".join(CONFIGURATIONS), help="comma-separated build configs"
    )
    collect_parser.add_argument(
        "--workloads", default=",".join(WORKLOADS), help="comma-separated workload names"
    )
    collect_parser.add_argument("--isa", choices=("scalar", "avx2", "avx512"), default="avx512")
    collect_parser.add_argument("--openmp", choices=("OFF", "AUTO", "ON"), default="ON")
    collect_parser.add_argument("--perf", choices=("auto", "on", "off"), default="auto")
    collect_parser.add_argument("--outer-repetitions", type=int, default=4)
    collect_parser.add_argument("--profile-repetitions", type=int, default=7)
    collect_parser.add_argument("--warmups", type=int, default=2)
    collect_parser.add_argument("--cpu", type=int)
    collect_parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    collect_parser.add_argument("--work-dir", type=Path, default=WORK_ROOT)
    collect_parser.add_argument("--corpus-dir", type=Path)
    collect_parser.add_argument("--fast-float-dir", type=Path)
    collect_parser.add_argument("--quick", action="store_true", help="one short validation pass")

    publish_parser = subparsers.add_parser("publish", help="commit and optionally push one result")
    publish_parser.add_argument("--execution-id", type=checked_identifier, required=True)
    publish_parser.add_argument("--push", action="store_true", help="push the new commit to origin")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    try:
        if args.command == "install-deps":
            install_dependencies(repo)
        elif args.command == "run":
            for name, value in {
                "outer repetitions": args.outer_repetitions,
                "profile repetitions": args.profile_repetitions,
                "warmups": args.warmups,
                "jobs": args.jobs,
            }.items():
                if value < (0 if name == "warmups" else 1):
                    raise RuntimeError(
                        f"{name} must be positive"
                        if name != "warmups"
                        else "warmups must be non-negative"
                    )
            collect(args, repo)
        elif args.command == "publish":
            publish(args, repo)
        else:
            parser.error(f"unknown command: {args.command}")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
