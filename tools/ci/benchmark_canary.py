#!/usr/bin/env python3
"""Render an advisory PR comment from paired Google Benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

COMMENT_MARKER = "<!-- clifft-performance-canary -->"
NOTABLE_THRESHOLD = 0.05
POSSIBLE_REGRESSION_THRESHOLD = 0.10

DISPLAY_NAMES = {
    "squeeze_parallel_t_8192": "Squeeze 8192 T gates",
    "sample_qv10_100_shots": "QV-10, 100 shots",
    "sample_cultivation_d5_1000_shots": "Cultivation d5, 1,000 shots",
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
        if abs(self.change) < NOTABLE_THRESHOLD:
            return "No material change"
        if self.max_relative_stddev >= NOTABLE_THRESHOLD:
            return "Inconclusive"
        if self.change > 0 and (
            self.first_change < NOTABLE_THRESHOLD or self.second_change < NOTABLE_THRESHOLD
        ):
            return "Inconclusive"
        if self.change < 0 and (
            self.first_change > -NOTABLE_THRESHOLD or self.second_change > -NOTABLE_THRESHOLD
        ):
            return "Inconclusive"
        if self.change >= POSSIBLE_REGRESSION_THRESHOLD:
            return "Possible regression"
        if self.change >= NOTABLE_THRESHOLD:
            return "Notable slowdown"
        if self.change <= -POSSIBLE_REGRESSION_THRESHOLD:
            return "Improvement"
        if self.change <= -NOTABLE_THRESHOLD:
            return "Notable improvement"
        raise AssertionError("material change was not classified")


def parse_google_benchmarks(path: Path) -> dict[str, BenchmarkEstimate]:
    payload = json.loads(path.read_text())
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError(f"malformed Google Benchmark output: {path}")

    repetitions: dict[str, list[float]] = defaultdict(list)
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            raise ValueError(f"malformed Google Benchmark result: {path}")
        if benchmark.get("error_occurred"):
            message = benchmark.get("error_message", "unknown benchmark error")
            raise ValueError(f"Google Benchmark run did not succeed: {message}")
        if benchmark.get("run_type", "iteration") != "iteration":
            continue
        name = benchmark.get("run_name", benchmark.get("name"))
        time_unit = benchmark.get("time_unit")
        if not isinstance(name, str) or time_unit not in TIME_UNIT_TO_NS:
            raise ValueError(f"malformed Google Benchmark result: {path}")
        cpu_time = float(benchmark.get("cpu_time", math.nan))
        nanoseconds = cpu_time * TIME_UNIT_TO_NS[time_unit]
        if not math.isfinite(nanoseconds) or nanoseconds <= 0:
            raise ValueError(f"invalid CPU time for {name!r}: {cpu_time}")
        repetitions[name].append(nanoseconds)

    if not repetitions:
        raise ValueError(f"no Google Benchmark results found: {path}")

    results = {}
    for name, values in repetitions.items():
        median_ns = statistics.median(values)
        relative_stddev = (
            statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0
        )
        results[name] = BenchmarkEstimate(median_ns, relative_stddev)
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
    if abs(change) >= NOTABLE_THRESHOLD:
        return f"**{rendered}**"
    return rendered


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
            "least 10% slower. This does not block merging."
        )
        return _append_inconclusive(summary, inconclusive)
    if notable_slowdowns:
        verb = "was" if notable_slowdowns == 1 else "were"
        summary = (
            f"**Notable slowdown detected:** {notable_slowdowns} of {total} benchmarks {verb} "
            "between 5% and 10% slower. No possible regression was detected. This does not "
            "block merging."
        )
        return _append_inconclusive(summary, inconclusive)
    if inconclusive:
        noun = "comparison was" if inconclusive == 1 else "comparisons were"
        return (
            f"**No confirmed regressions detected.** {inconclusive} benchmark {noun} "
            "inconclusive. This does not block merging."
        )
    if improvements:
        noun = "benchmark" if improvements == 1 else "benchmarks"
        return (
            "**No possible regressions detected.** "
            f"{improvements} {noun} showed an improvement of at least 5%."
        )
    return (
        f"**No material performance changes detected.** All {total} benchmarks remained within "
        "5% of the base."
    )


def _append_inconclusive(summary: str, count: int) -> str:
    if not count:
        return summary
    noun = "comparison was" if count == 1 else "comparisons were"
    return f"{summary} {count} additional benchmark {noun} inconclusive."


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
) -> str:
    lines = [
        COMMENT_MARKER,
        "## Performance canary (advisory)",
        "",
        _summary(comparisons),
        "",
        "| Benchmark | Base | PR | Runtime change | Assessment |",
        "|---|---:|---:|---:|---|",
    ]
    for comparison in comparisons:
        name = DISPLAY_NAMES.get(comparison.name, comparison.name).replace("|", "\\|")
        lines.append(
            f"| {name} | {format_duration(comparison.base_ns)} | "
            f"{format_duration(comparison.head_ns)} | "
            f"{format_change_cell(comparison.change)} | {comparison.assessment} |"
        )

    lines.extend(
        [
            "",
            f"Compared `{base_label}` (`{base_sha[:7]}`) with this PR (`{head_sha[:7]}`) on "
            "the same runner using A/B/B/A ordering. Positive changes are slower. Changes under "
            "5% are reported as no material change; changes of at least 5% but under 10% are "
            "notable; changes of at least 10% are possible regressions or improvements.",
            "",
            "<details>",
            "<summary>Environment and method</summary>",
            "",
            f"- Runner CPU: {_environment_value(cpu, 'cpu')}",
            f"- Pinned logical CPU: {cpu_core or 'unknown'}",
            "- ISA: AVX2 runtime backend (forced)",
            f"- Compiler: {_environment_value(compiler, 'compiler')}",
            "- Build: Release, x86-64-v2, ThinLTO, lld, and OpenMP enabled.",
            f"- Each pass uses {repetitions} Google Benchmark repetitions with at least "
            f"{min_time:g} seconds of measured CPU time and {warmup_time:g} seconds of warmup "
            "per workload.",
            "- Displayed timings are the geometric mean of the two drift-balanced pass medians.",
            "- Material changes must repeat in both pairs with under 5% within-pass relative "
            "standard deviation; otherwise they are inconclusive.",
            "- Results apply only to this runner and are intended to detect large regressions, "
            "not establish release performance.",
        ]
    )
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
        "## Performance canary (advisory)",
        "",
        "**The canary could not produce a comparison.** This does not block merging.",
        "",
        f"Attempted to compare `{base_label}` (`{base_sha[:7]}`) with this PR (`{head_sha[:7]}`).",
    ]
    resolved_run_url = _run_url(run_url)
    if resolved_run_url:
        lines.extend(["", f"[View workflow run]({resolved_run_url})"])
    return "\n".join(lines) + "\n"


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
    args = parser.parse_args()

    if args.command == "failure":
        content = render_failure(
            base_label=args.base_label,
            base_sha=args.base_sha,
            head_sha=args.head_sha,
            run_url=args.run_url,
        )
    else:
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
        )
    write_report(args.output, content)


if __name__ == "__main__":
    main()
