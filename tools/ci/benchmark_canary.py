#!/usr/bin/env python3
"""Render an advisory PR comment from paired Google Benchmark runs."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import platform
import re
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

COMMENT_MARKER = "<!-- clifft-performance-canary -->"
MATERIAL_CHANGE_THRESHOLD = 0.05
MAX_RELATIVE_STDDEV = 0.05
POSSIBLE_CHANGE_THRESHOLD = 0.10
CANARY_MANIFEST_SCHEMA = 1
CANARY_REPETITIONS = 3
CANARY_MIN_TIME = 0.5
CANARY_WARMUP_TIME = 0.2
MAX_MANIFEST_BYTES = 16 * 1024
MAX_RESULT_BYTES = 1024 * 1024
MAX_BENCHMARK_ROWS = 100
BENCHMARK_SOURCE = Path(__file__).resolve().parents[2] / "benchmarks" / "clifft_benchmarks.cc"

DISPLAY_NAMES = {
    "squeeze_parallel_t_8192": "Squeeze 8192 T gates",
    "compile_plan_cultivation_d5": "Compile/plan cultivation d5",
    "sample_qv10_100_shots": "QV-10, 100 shots",
    "sample_cultivation_d5_1000_shots": "Cultivation d5, 1,000 shots",
    "sample_coherent_d5_r5_100_shots": "Coherent QEC d5/r5, width 13, 100 shots",
    "sample_surface_d7_r7_10000_shots": "Surface code d7/r7, 10,000 shots",
    "sample_surface_d5_r5_high_noise_10000_shots": ("Surface code d5/r5 high noise, 10,000 shots"),
    "sample_surface_d11_r11_1000_shots": "Surface code d11/r11, 1,000 shots",
    "sample_exp_val_20q_200_probes_100000_shots": ("EXP_VAL 20q/200 probes, 100,000 shots"),
}

TIME_UNIT_TO_NS = {
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
}


@dataclass(frozen=True)
class BenchmarkEstimate:
    median_ns: float
    relative_stddev: float
    sample_count: int


@dataclass(frozen=True)
class Comparison:
    name: str
    base_ns: float
    head_ns: float
    change: float
    first_change: float
    second_change: float
    max_relative_stddev: float

    @property
    def assessment(self) -> str:
        if abs(self.change) < MATERIAL_CHANGE_THRESHOLD:
            return "No material change"
        if self.max_relative_stddev >= MAX_RELATIVE_STDDEV:
            return "Inconclusive"
        if self.change > 0 and (
            self.first_change < MATERIAL_CHANGE_THRESHOLD
            or self.second_change < MATERIAL_CHANGE_THRESHOLD
        ):
            return "Inconclusive"
        if self.change < 0 and (
            self.first_change > -MATERIAL_CHANGE_THRESHOLD
            or self.second_change > -MATERIAL_CHANGE_THRESHOLD
        ):
            return "Inconclusive"
        if self.change >= POSSIBLE_CHANGE_THRESHOLD:
            return "Possible regression"
        if self.change >= MATERIAL_CHANGE_THRESHOLD:
            return "Notable slowdown"
        if self.change <= -POSSIBLE_CHANGE_THRESHOLD:
            return "Improvement"
        if self.change <= -MATERIAL_CHANGE_THRESHOLD:
            return "Notable improvement"
        raise AssertionError("material change was not classified")


def parse_google_benchmarks(path: Path) -> dict[str, BenchmarkEstimate]:
    results = {}
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"no Google Benchmark result files found: {path}")

    for result_path in paths:
        payload = json.loads(result_path.read_text())
        benchmarks = payload.get("benchmarks")
        if not isinstance(benchmarks, list):
            raise ValueError(f"malformed Google Benchmark output: {result_path}")
        if len(benchmarks) > MAX_BENCHMARK_ROWS:
            raise ValueError(f"too many Google Benchmark rows: {result_path}")

        repetitions: dict[str, list[float]] = defaultdict(list)
        for benchmark in benchmarks:
            if not isinstance(benchmark, dict):
                raise ValueError(f"malformed Google Benchmark result: {result_path}")
            if benchmark.get("error_occurred"):
                message = benchmark.get("error_message", "unknown benchmark error")
                raise ValueError(f"Google Benchmark run did not succeed: {message}")
            if benchmark.get("run_type", "iteration") != "iteration":
                continue
            name = benchmark.get("run_name", benchmark.get("name"))
            time_unit = benchmark.get("time_unit")
            if not isinstance(name, str) or time_unit not in TIME_UNIT_TO_NS:
                raise ValueError(f"malformed Google Benchmark result: {result_path}")
            cpu_time = float(benchmark.get("cpu_time", math.nan))
            nanoseconds = cpu_time * TIME_UNIT_TO_NS[time_unit]
            if not math.isfinite(nanoseconds) or nanoseconds <= 0:
                raise ValueError(f"invalid CPU time for {name!r}: {cpu_time}")
            repetitions[name].append(nanoseconds)

        for name, values in repetitions.items():
            if name in results:
                raise ValueError(f"duplicate benchmark {name!r}: {result_path}")
            median_ns = statistics.median(values)
            relative_stddev = (
                statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0
            )
            results[name] = BenchmarkEstimate(median_ns, relative_stddev, len(values))

    if not results:
        raise ValueError(f"no Google Benchmark results found: {path}")
    return results


def compare_runs(
    base_first_path: Path,
    head_first_path: Path,
    head_second_path: Path,
    base_second_path: Path,
) -> tuple[list[Comparison], list[str], list[str]]:
    base_first = parse_google_benchmarks(base_first_path)
    head_first = parse_google_benchmarks(head_first_path)
    head_second = parse_google_benchmarks(head_second_path)
    base_second = parse_google_benchmarks(base_second_path)

    if base_first.keys() != base_second.keys():
        raise ValueError("the two base runs contain different benchmarks")
    if head_first.keys() != head_second.keys():
        raise ValueError("the two PR runs contain different benchmarks")

    base_names = set(base_first)
    head_names = set(head_first)
    common_names = base_names & head_names
    if not common_names:
        raise ValueError("base and PR runs have no benchmarks in common")

    ordered_names = [name for name in DISPLAY_NAMES if name in common_names]
    ordered_names.extend(sorted(common_names - DISPLAY_NAMES.keys()))
    comparisons = []
    for name in ordered_names:
        estimates = (base_first[name], head_first[name], head_second[name], base_second[name])
        base_ns = math.sqrt(base_first[name].median_ns * base_second[name].median_ns)
        head_ns = math.sqrt(head_first[name].median_ns * head_second[name].median_ns)
        comparisons.append(
            Comparison(
                name=name,
                base_ns=base_ns,
                head_ns=head_ns,
                change=head_ns / base_ns - 1.0,
                first_change=head_first[name].median_ns / base_first[name].median_ns - 1.0,
                second_change=head_second[name].median_ns / base_second[name].median_ns - 1.0,
                max_relative_stddev=max(estimate.relative_stddev for estimate in estimates),
            )
        )

    return comparisons, sorted(head_names - base_names), sorted(base_names - head_names)


def format_duration(nanoseconds: float) -> str:
    if nanoseconds >= 1e9:
        return _format_value(nanoseconds / 1e9, "s")
    if nanoseconds >= 1e6:
        return _format_value(nanoseconds / 1e6, "ms")
    if nanoseconds >= 1e3:
        return _format_value(nanoseconds / 1e3, "us")
    return _format_value(nanoseconds, "ns")


def _format_value(value: float, unit: str) -> str:
    if value >= 100:
        rendered = f"{value:.0f}"
    elif value >= 10:
        rendered = f"{value:.1f}"
    else:
        rendered = f"{value:.2f}"
    return f"{rendered} {unit}"


def format_change(change: float) -> str:
    return f"{change:+.1%}"


def format_change_cell(change: float) -> str:
    rendered = format_change(change)
    if abs(change) >= MATERIAL_CHANGE_THRESHOLD:
        return f"**{rendered}**"
    return rendered


def _threshold_percent(threshold: float) -> str:
    return f"{threshold:.0%}"


def _summary(comparisons: list[Comparison]) -> str:
    assessments = [comparison.assessment for comparison in comparisons]
    regressions = assessments.count("Possible regression")
    notable_slowdowns = assessments.count("Notable slowdown")
    improvements = assessments.count("Improvement") + assessments.count("Notable improvement")
    inconclusive = assessments.count("Inconclusive")
    total = len(comparisons)

    if regressions:
        verb = "was" if regressions == 1 else "were"
        summary = (
            f"**Possible regression detected:** {regressions} of {total} benchmarks {verb} at "
            f"least {_threshold_percent(POSSIBLE_CHANGE_THRESHOLD)} slower."
        )
        return f"{_append_inconclusive(summary, inconclusive)} This does not block merging."
    if notable_slowdowns:
        verb = "was" if notable_slowdowns == 1 else "were"
        summary = (
            f"**Notable slowdown detected:** {notable_slowdowns} of {total} benchmarks {verb} "
            f"between {_threshold_percent(MATERIAL_CHANGE_THRESHOLD)} and "
            f"{_threshold_percent(POSSIBLE_CHANGE_THRESHOLD)} slower. No possible regression "
            "was detected."
        )
        return f"{_append_inconclusive(summary, inconclusive)} This does not block merging."
    if inconclusive:
        summary = "**No confirmed regressions detected.**"
        if improvements:
            noun = "benchmark" if improvements == 1 else "benchmarks"
            summary += (
                f" {improvements} {noun} showed an improvement of at least "
                f"{_threshold_percent(MATERIAL_CHANGE_THRESHOLD)}."
            )
        return f"{_append_inconclusive(summary, inconclusive)} This does not block merging."
    if improvements:
        noun = "benchmark" if improvements == 1 else "benchmarks"
        return (
            "**No possible regressions detected.** "
            f"{improvements} {noun} showed an improvement of at least "
            f"{_threshold_percent(MATERIAL_CHANGE_THRESHOLD)}."
        )
    return (
        f"**No material performance changes detected.** All {total} benchmarks remained within "
        f"{_threshold_percent(MATERIAL_CHANGE_THRESHOLD)} of the base."
    )


def _append_inconclusive(summary: str, count: int) -> str:
    if not count:
        return summary
    noun = "comparison was" if count == 1 else "comparisons were"
    return f"{summary} {count} benchmark {noun} inconclusive."


def _benchmark_source_links(head_sha: str) -> dict[str, str]:
    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not server or not repository or not BENCHMARK_SOURCE.is_file():
        return {}

    lines = BENCHMARK_SOURCE.read_text().splitlines()
    function_lines = {}
    function_pattern = re.compile(r"^void ([a-z0-9_]+)\(benchmark::State& state\) \{$")
    for index, line in enumerate(lines):
        match = function_pattern.match(line)
        if not match:
            continue
        source_line = index + 1
        comment_index = index - 1
        while comment_index >= 0 and lines[comment_index].startswith("//"):
            source_line = comment_index + 1
            comment_index -= 1
        function_lines[match.group(1)] = source_line

    registration_pattern = re.compile(r'^BENCHMARK\(([a-z0-9_]+)\)->Name\("([^"]+)"\);$')
    links = {}
    for line in lines:
        match = registration_pattern.match(line)
        if not match or match.group(1) not in function_lines:
            continue
        function, benchmark_name = match.groups()
        links[benchmark_name] = (
            f"{server.rstrip('/')}/{repository}/blob/{head_sha}/"
            f"benchmarks/clifft_benchmarks.cc#L{function_lines[function]}"
        )
    return links


def _environment_value(override: str | None, kind: str) -> str:
    if override:
        return override
    if kind == "cpu":
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return platform.processor() or platform.machine()
    try:
        first_line = subprocess.check_output(
            [os.environ.get("CXX", "c++"), "--version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"
    return first_line


def _run_url(override: str | None) -> str | None:
    if override:
        return override
    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def _comparison_note(base_label: str, base_sha: str, head_sha: str) -> str:
    return (
        "<sub>"
        f"Compared <code>{html.escape(base_label)}</code> (<code>{base_sha[:7]}</code>) with "
        f"this PR (<code>{head_sha[:7]}</code>) on the same runner using workload-local A/B/B/A "
        "ordering. Positive changes are slower. "
        f"Changes under {_threshold_percent(MATERIAL_CHANGE_THRESHOLD)} are reported as no "
        f"material change; changes of at least "
        f"{_threshold_percent(MATERIAL_CHANGE_THRESHOLD)} but under "
        f"{_threshold_percent(POSSIBLE_CHANGE_THRESHOLD)} are notable; changes of at least "
        f"{_threshold_percent(POSSIBLE_CHANGE_THRESHOLD)} are possible regressions or "
        "improvements. Material changes that do not repeat in both pairs with low noise are "
        "inconclusive."
        "</sub>"
    )


def render_report(
    comparisons: list[Comparison],
    *,
    base_label: str,
    base_sha: str,
    head_sha: str,
    repetitions: int,
    min_time: float,
    warmup_time: float,
    added: list[str],
    removed: list[str],
    cpu: str | None = None,
    compiler: str | None = None,
    cpu_core: str | None = None,
    run_url: str | None = None,
    benchmark_sha: str | None = None,
) -> str:
    lines = [
        COMMENT_MARKER,
        "## :baby_chick: Performance Canary",
        "",
        _summary(comparisons),
        "",
    ]
    collapse_results = (
        bool(comparisons)
        and not added
        and not removed
        and all(comparison.assessment == "No material change" for comparison in comparisons)
    )
    if collapse_results:
        noun = "result" if len(comparisons) == 1 else "results"
        lines.extend(
            [
                "<details>",
                f"<summary>View {len(comparisons)} benchmark {noun}</summary>",
                "",
            ]
        )
    lines.extend(
        [
            "| Benchmark | Base | PR | Runtime change | Assessment |",
            "|---|---:|---:|---:|---|",
        ]
    )
    source_links = _benchmark_source_links(benchmark_sha or head_sha)
    for comparison in comparisons:
        name = DISPLAY_NAMES.get(comparison.name, comparison.name).replace("|", "\\|")
        if comparison.name in source_links:
            name = f"[{name}]({source_links[comparison.name]})"
        lines.append(
            f"| {name} | {format_duration(comparison.base_ns)} | "
            f"{format_duration(comparison.head_ns)} | "
            f"{format_change_cell(comparison.change)} | {comparison.assessment} |"
        )

    lines.extend(["", _comparison_note(base_label, base_sha, head_sha)])
    if collapse_results:
        lines.extend(["", "</details>"])
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Environment and method</summary>",
            "",
            f"- Runner CPU: <code>{html.escape(_environment_value(cpu, 'cpu'))}</code>",
            f"- Pinned logical CPU: <code>{html.escape(cpu_core or 'unknown')}</code>",
            "- ISA: AVX2 runtime backend (forced)",
            f"- Compiler: <code>{html.escape(_environment_value(compiler, 'compiler'))}</code>",
            "- Build: Release, x86-64-v2, ThinLTO, lld, and OpenMP enabled.",
            f"- Each pass warms every workload for at least {warmup_time:g} seconds, then runs "
            f"{repetitions} Google Benchmark repetitions with at least {min_time:g} seconds of "
            "measured CPU time each.",
            "- One unreported full-suite process per revision runs before the measured pairs.",
            "- Displayed timings are the geometric mean of the two drift-balanced pass medians.",
            "- Material changes must repeat in both pairs with under "
            f"{_threshold_percent(MAX_RELATIVE_STDDEV)} within-pass relative standard deviation; "
            "otherwise they are inconclusive.",
            "- Results apply only to this runner and are intended to detect large regressions, "
            "not establish release performance.",
        ]
    )
    if benchmark_sha and benchmark_sha != head_sha:
        lines.append("- Workloads and fixtures come from the base revision for fork isolation.")
    if added:
        lines.append(f"- Added in the PR and not compared: {', '.join(added)}")
    if removed:
        lines.append(f"- Absent from the PR and not compared: {', '.join(removed)}")
    lines.extend(["", "</details>"])

    resolved_run_url = _run_url(run_url)
    if resolved_run_url:
        lines.extend(["", f"[View workflow run]({resolved_run_url})"])
    return "\n".join(lines) + "\n"


def render_failure(
    *, base_label: str, base_sha: str, head_sha: str, run_url: str | None = None
) -> str:
    lines = [
        COMMENT_MARKER,
        "## :baby_chick: Performance Canary",
        "",
        "**The canary could not produce a comparison.** This does not block merging.",
        "",
        f"Attempted to compare <code>{html.escape(base_label)}</code> "
        f"(<code>{html.escape(base_sha[:7])}</code>) with this PR "
        f"(<code>{html.escape(head_sha[:7])}</code>).",
    ]
    resolved_run_url = _run_url(run_url)
    if resolved_run_url:
        lines.extend(["", f"[View workflow run]({resolved_run_url})"])
    return "\n".join(lines) + "\n"


def write_manifest(
    path: Path,
    *,
    repository: str,
    pr_number: int,
    base_label: str,
    base_sha: str,
    head_repository: str,
    head_sha: str,
    run_id: int,
    repetitions: int,
    min_time: float,
    warmup_time: float,
    cpu_core: str,
    cpu: str | None = None,
    compiler: str | None = None,
) -> None:
    if (
        repetitions != CANARY_REPETITIONS
        or min_time != CANARY_MIN_TIME
        or warmup_time != CANARY_WARMUP_TIME
    ):
        raise ValueError("benchmark settings do not match the trusted reporter")
    payload = {
        "schema": CANARY_MANIFEST_SCHEMA,
        "repository": repository,
        "pr_number": pr_number,
        "base_label": base_label,
        "base_sha": base_sha,
        "head_repository": head_repository,
        "head_sha": head_sha,
        "run_id": run_id,
        "repetitions": repetitions,
        "min_time": min_time,
        "warmup_time": warmup_time,
        "cpu_core": cpu_core,
        "cpu": _environment_value(cpu, "cpu"),
        "compiler": _environment_value(compiler, "compiler"),
    }
    write_report(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validated_plain_text(value: object, field: str, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"invalid {field} in benchmark manifest")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise ValueError(f"invalid {field} in benchmark manifest")
    return value


def _load_fork_manifest(
    evidence: Path,
    *,
    expected_repository: str,
    expected_pr_number: int,
    expected_base_label: str,
    expected_base_sha: str,
    expected_head_repository: str,
    expected_head_sha: str,
    expected_run_id: int,
) -> dict[str, object]:
    manifest_path = evidence / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("benchmark manifest is missing or is not a regular file")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("benchmark manifest is too large")
    payload = json.loads(manifest_path.read_text())
    expected_fields = {
        "schema",
        "repository",
        "pr_number",
        "base_label",
        "base_sha",
        "head_repository",
        "head_sha",
        "run_id",
        "repetitions",
        "min_time",
        "warmup_time",
        "cpu_core",
        "cpu",
        "compiler",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("benchmark manifest has an unexpected schema")

    exact_values = {
        "schema": CANARY_MANIFEST_SCHEMA,
        "repository": expected_repository,
        "pr_number": expected_pr_number,
        "base_label": expected_base_label,
        "base_sha": expected_base_sha,
        "head_repository": expected_head_repository,
        "head_sha": expected_head_sha,
        "run_id": expected_run_id,
        "repetitions": CANARY_REPETITIONS,
        "min_time": CANARY_MIN_TIME,
        "warmup_time": CANARY_WARMUP_TIME,
    }
    for field, expected in exact_values.items():
        if type(payload[field]) is not type(expected) or payload[field] != expected:
            raise ValueError(f"benchmark manifest {field} does not match the workflow run")

    sha_pattern = re.compile(r"^[0-9a-f]{40}$")
    if not sha_pattern.fullmatch(expected_base_sha) or not sha_pattern.fullmatch(expected_head_sha):
        raise ValueError("workflow metadata contains an invalid commit SHA")
    _validated_plain_text(payload["repository"], "repository")
    _validated_plain_text(payload["base_label"], "base label")
    _validated_plain_text(payload["head_repository"], "head repository")
    cpu_core = _validated_plain_text(payload["cpu_core"], "CPU core", 16)
    if not cpu_core.isdigit():
        raise ValueError("invalid CPU core in benchmark manifest")
    _validated_plain_text(payload["cpu"], "CPU", 256)
    _validated_plain_text(payload["compiler"], "compiler", 512)
    return payload


def _validate_fork_results(path: Path, repetitions: int) -> None:
    expected_names = set(DISPLAY_NAMES)
    expected_files = {f"{name}.json" for name in expected_names}
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"benchmark evidence directory is missing: {path.name}")
    entries = list(path.iterdir())
    if {entry.name for entry in entries} != expected_files:
        raise ValueError(f"benchmark evidence has an unexpected file set: {path.name}")
    for result_path in entries:
        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError(f"benchmark evidence is not a regular file: {result_path.name}")
        if result_path.stat().st_size > MAX_RESULT_BYTES:
            raise ValueError(f"benchmark evidence is too large: {result_path.name}")

    estimates = parse_google_benchmarks(path)
    if set(estimates) != expected_names:
        raise ValueError(f"benchmark evidence has an unexpected workload set: {path.name}")
    for name, estimate in estimates.items():
        if estimate.sample_count != repetitions:
            raise ValueError(f"benchmark {name!r} has an unexpected repetition count")


def render_fork_report(
    evidence: Path,
    *,
    expected_repository: str,
    expected_pr_number: int,
    expected_base_label: str,
    expected_base_sha: str,
    expected_head_repository: str,
    expected_head_sha: str,
    expected_run_id: int,
    run_url: str,
) -> str:
    manifest = _load_fork_manifest(
        evidence,
        expected_repository=expected_repository,
        expected_pr_number=expected_pr_number,
        expected_base_label=expected_base_label,
        expected_base_sha=expected_base_sha,
        expected_head_repository=expected_head_repository,
        expected_head_sha=expected_head_sha,
        expected_run_id=expected_run_id,
    )
    result_directories = [
        evidence / "base-first",
        evidence / "head-first",
        evidence / "head-second",
        evidence / "base-second",
    ]
    for result_directory in result_directories:
        _validate_fork_results(result_directory, CANARY_REPETITIONS)
    comparisons, added, removed = compare_runs(*result_directories)
    if added or removed:
        raise ValueError("fork benchmark evidence changed the trusted workload set")
    return render_report(
        comparisons,
        base_label=expected_base_label,
        base_sha=expected_base_sha,
        head_sha=expected_head_sha,
        repetitions=CANARY_REPETITIONS,
        min_time=CANARY_MIN_TIME,
        warmup_time=CANARY_WARMUP_TIME,
        added=added,
        removed=removed,
        cpu=str(manifest["cpu"]),
        compiler=str(manifest["compiler"]),
        cpu_core=str(manifest["cpu_core"]),
        run_url=run_url,
        benchmark_sha=expected_base_sha,
    )


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-label", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--run-url")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    failure_parser = subparsers.add_parser("failure")
    _add_common_arguments(failure_parser)

    report_parser = subparsers.add_parser("report")
    _add_common_arguments(report_parser)
    report_parser.add_argument("--base-first", type=Path, required=True)
    report_parser.add_argument("--head-first", type=Path, required=True)
    report_parser.add_argument("--head-second", type=Path, required=True)
    report_parser.add_argument("--base-second", type=Path, required=True)
    report_parser.add_argument("--repetitions", type=int, required=True)
    report_parser.add_argument("--min-time", type=float, required=True)
    report_parser.add_argument("--warmup-time", type=float, required=True)
    report_parser.add_argument("--cpu")
    report_parser.add_argument("--compiler")
    report_parser.add_argument("--cpu-core")
    report_parser.add_argument("--benchmark-sha")

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--repository", required=True)
    manifest_parser.add_argument("--pr-number", type=int, required=True)
    manifest_parser.add_argument("--base-label", required=True)
    manifest_parser.add_argument("--base-sha", required=True)
    manifest_parser.add_argument("--head-repository", required=True)
    manifest_parser.add_argument("--head-sha", required=True)
    manifest_parser.add_argument("--run-id", type=int, required=True)
    manifest_parser.add_argument("--repetitions", type=int, required=True)
    manifest_parser.add_argument("--min-time", type=float, required=True)
    manifest_parser.add_argument("--warmup-time", type=float, required=True)
    manifest_parser.add_argument("--cpu-core", required=True)
    manifest_parser.add_argument("--cpu")
    manifest_parser.add_argument("--compiler")

    fork_report_parser = subparsers.add_parser("fork-report")
    fork_report_parser.add_argument("--output", type=Path, required=True)
    fork_report_parser.add_argument("--base-label", required=True)
    fork_report_parser.add_argument("--base-sha", required=True)
    fork_report_parser.add_argument("--head-sha", required=True)
    fork_report_parser.add_argument("--run-url", required=True)
    fork_report_parser.add_argument("--evidence", type=Path, required=True)
    fork_report_parser.add_argument("--repository", required=True)
    fork_report_parser.add_argument("--pr-number", type=int, required=True)
    fork_report_parser.add_argument("--head-repository", required=True)
    fork_report_parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()

    if args.command == "failure":
        content = render_failure(
            base_label=args.base_label,
            base_sha=args.base_sha,
            head_sha=args.head_sha,
            run_url=args.run_url,
        )
        write_report(args.output, content)
    elif args.command == "report":
        comparisons, added, removed = compare_runs(
            args.base_first, args.head_first, args.head_second, args.base_second
        )
        content = render_report(
            comparisons,
            base_label=args.base_label,
            base_sha=args.base_sha,
            head_sha=args.head_sha,
            repetitions=args.repetitions,
            min_time=args.min_time,
            warmup_time=args.warmup_time,
            added=added,
            removed=removed,
            cpu=args.cpu,
            compiler=args.compiler,
            cpu_core=args.cpu_core,
            run_url=args.run_url,
            benchmark_sha=args.benchmark_sha,
        )
        write_report(args.output, content)
    elif args.command == "manifest":
        write_manifest(
            args.output,
            repository=args.repository,
            pr_number=args.pr_number,
            base_label=args.base_label,
            base_sha=args.base_sha,
            head_repository=args.head_repository,
            head_sha=args.head_sha,
            run_id=args.run_id,
            repetitions=args.repetitions,
            min_time=args.min_time,
            warmup_time=args.warmup_time,
            cpu_core=args.cpu_core,
            cpu=args.cpu,
            compiler=args.compiler,
        )
    else:
        content = render_fork_report(
            args.evidence,
            expected_repository=args.repository,
            expected_pr_number=args.pr_number,
            expected_base_label=args.base_label,
            expected_base_sha=args.base_sha,
            expected_head_repository=args.head_repository,
            expected_head_sha=args.head_sha,
            expected_run_id=args.run_id,
            run_url=args.run_url,
        )
        write_report(args.output, content)


if __name__ == "__main__":
    main()
