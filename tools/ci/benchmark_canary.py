#!/usr/bin/env python3
"""Render an advisory PR comment from paired Catch2 benchmark runs."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

COMMENT_MARKER = "<!-- clifft-performance-canary -->"
NOTABLE_THRESHOLD = 0.05
POSSIBLE_REGRESSION_THRESHOLD = 0.10

DISPLAY_NAMES = {
    "squeeze 8192 parallel T gates": "Squeeze 8192 T gates",
    "QV-10 x100 shots": "QV-10, 100 shots",
    "cultivation-d5 x1000 shots": "Cultivation d5, 1,000 shots",
    "surface-d7-r7 p=1e-3 x10000 shots": "Surface code d7/r7, 10,000 shots",
    "surface-d5-r5 p=0.05 x10000 shots": "Surface code d5/r5 high noise, 10,000 shots",
    "surface-d11-r11 p=1e-3 x1000 shots": "Surface code d11/r11, 1,000 shots",
    "exp-val 20q 200 probes x100k": "EXP_VAL 20q/200 probes, 100,000 shots",
}


@dataclass(frozen=True)
class Comparison:
    name: str
    base_ns: float
    head_ns: float
    change: float

    @property
    def assessment(self) -> str:
        if self.change >= POSSIBLE_REGRESSION_THRESHOLD:
            return "Possible regression"
        if self.change >= NOTABLE_THRESHOLD:
            return "Notable slowdown"
        if self.change <= -POSSIBLE_REGRESSION_THRESHOLD:
            return "Improvement"
        if self.change <= -NOTABLE_THRESHOLD:
            return "Notable improvement"
        return "No material change"


def parse_catch2_benchmarks(path: Path) -> dict[str, float]:
    root = ElementTree.parse(path).getroot()
    overall = root.find("OverallResults")
    if overall is None or int(overall.attrib.get("failures", "1")) != 0:
        raise ValueError(f"Catch2 run did not succeed: {path}")

    results: dict[str, float] = {}
    for benchmark in root.iter("BenchmarkResults"):
        name = benchmark.attrib.get("name")
        mean = benchmark.find("mean")
        if not name or mean is None:
            raise ValueError(f"malformed Catch2 benchmark result: {path}")
        value = float(mean.attrib["value"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid mean for {name!r}: {value}")
        if name in results:
            raise ValueError(f"duplicate benchmark {name!r}: {path}")
        results[name] = value

    if not results:
        raise ValueError(f"no Catch2 benchmark results found: {path}")
    return results


def compare_runs(
    base_first_path: Path,
    head_first_path: Path,
    head_second_path: Path,
    base_second_path: Path,
) -> tuple[list[Comparison], list[str], list[str]]:
    base_first = parse_catch2_benchmarks(base_first_path)
    head_first = parse_catch2_benchmarks(head_first_path)
    head_second = parse_catch2_benchmarks(head_second_path)
    base_second = parse_catch2_benchmarks(base_second_path)

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
        base_ns = math.sqrt(base_first[name] * base_second[name])
        head_ns = math.sqrt(head_first[name] * head_second[name])
        comparisons.append(Comparison(name, base_ns, head_ns, head_ns / base_ns - 1.0))

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
    regressions = sum(c.change >= POSSIBLE_REGRESSION_THRESHOLD for c in comparisons)
    notable_slowdowns = sum(
        NOTABLE_THRESHOLD <= c.change < POSSIBLE_REGRESSION_THRESHOLD for c in comparisons
    )
    improvements = sum(c.change <= -NOTABLE_THRESHOLD for c in comparisons)
    total = len(comparisons)

    if regressions:
        verb = "was" if regressions == 1 else "were"
        return (
            f"**Possible regression detected:** {regressions} of {total} benchmarks {verb} at "
            "least 10% slower. This does not block merging."
        )
    if notable_slowdowns:
        verb = "was" if notable_slowdowns == 1 else "were"
        return (
            f"**Notable slowdown detected:** {notable_slowdowns} of {total} benchmarks {verb} "
            "between 5% and 10% slower. No possible regression was detected. This does not "
            "block merging."
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
            ["c++", "--version"], text=True, stderr=subprocess.STDOUT
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
    samples: int,
    added: list[str],
    removed: list[str],
    cpu: str | None = None,
    compiler: str | None = None,
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
            "- ISA: automatic runtime dispatch",
            f"- Compiler: {_environment_value(compiler, 'compiler')}",
            f"- Each pass uses {samples} Catch2 samples.",
            "- Displayed timings are the geometric mean of the two drift-balanced pass means.",
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
    report_parser.add_argument("--samples", type=int, required=True)
    report_parser.add_argument("--cpu")
    report_parser.add_argument("--compiler")
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
            samples=args.samples,
            added=added,
            removed=removed,
            cpu=args.cpu,
            compiler=args.compiler,
            run_url=args.run_url,
        )
    write_report(args.output, content)


if __name__ == "__main__":
    main()
