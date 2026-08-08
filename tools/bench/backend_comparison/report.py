"""Summarize raw backend-comparison JSON into concise Markdown and JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    samples = list(values)
    median = statistics.median(samples)
    deviations = [abs(value - median) for value in samples]
    return {
        "count": len(samples),
        "min": min(samples),
        "q1": quantile(samples, 0.25),
        "median": median,
        "q3": quantile(samples, 0.75),
        "max": max(samples),
        "mad": statistics.median(deviations),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def outcome_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = sum(int(sample["outcomes"]["attempted_shots"]) for sample in samples)
    accepted = sum(int(sample["outcomes"]["accepted_shots"]) for sample in samples)
    logical = sum(int(sample["outcomes"]["logical_errors"]) for sample in samples)
    return {
        "attempted": attempted,
        "accepted": accepted,
        "discard_fraction": (attempted - accepted) / attempted if attempted else 0.0,
        "acceptance_interval_95": wilson_interval(accepted, attempted),
        "logical_errors": logical,
        "conditional_logical_rate": logical / accepted if accepted else 0.0,
        "conditional_logical_interval_95": wilson_interval(logical, accepted),
    }


def paired_ratios(samples: list[dict[str, Any]], numerator: str, denominator: str) -> list[float]:
    paired: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for sample in samples:
        paired[(int(sample["block"]), int(sample["seed"]))][sample["arm"]] = float(
            sample["sample_seconds"]
        )
    ratios = []
    for values in paired.values():
        if numerator in values and denominator in values:
            ratios.append(values[numerator] / values[denominator])
    return ratios


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", dest="json_output", type=Path)
    args = parser.parse_args()

    document = json.loads(args.raw.read_text())
    samples = [sample for sample in document["samples"] if sample["status"] == "success"]
    aa_samples = [sample for sample in samples if sample["comparison"] == "aa_control"]
    backend_samples = [sample for sample in samples if sample["comparison"] == "backend"]

    aa_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in aa_samples:
        aa_by_case[sample["case"]["case_id"]].append(sample)
    aa_distributions = {
        case_id: distribution(paired_ratios(case_samples, "b", "a"))
        for case_id, case_samples in aa_by_case.items()
    }
    surface_aa = aa_distributions.get("surface_d7_r7_aggregate")
    noise_low = float(surface_aa["min"]) if surface_aa else 0.95
    noise_high = float(surface_aa["max"]) if surface_aa else 1.05

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in backend_samples:
        by_case[sample["case"]["case_id"]].append(sample)

    rows: list[dict[str, Any]] = []
    for case_id, case_samples in by_case.items():
        legacy = [sample for sample in case_samples if sample["arm"] == "legacy"]
        symbolic = [sample for sample in case_samples if sample["arm"] == "symbolic"]
        legacy_times = distribution(float(sample["sample_seconds"]) for sample in legacy)
        symbolic_times = distribution(float(sample["sample_seconds"]) for sample in symbolic)
        ratios = distribution(paired_ratios(case_samples, "symbolic", "legacy"))
        legacy_compile_values = [
            float(sample["compile_seconds"])
            for sample in legacy
            if sample["compile_seconds"] is not None
        ]
        symbolic_compile_values = [
            float(sample["compile_seconds"])
            for sample in symbolic
            if sample["compile_seconds"] is not None
        ]
        compile_ratio = None
        if legacy_compile_values and symbolic_compile_values:
            compile_ratio = statistics.median(symbolic_compile_values) / statistics.median(
                legacy_compile_values
            )
        row: dict[str, Any] = {
            "case_id": case_id,
            "regime": case_samples[0]["case"]["regime"],
            "output_mode": case_samples[0]["case"]["output_mode"],
            "shots": case_samples[0]["case"]["shots"],
            "legacy_seconds": legacy_times,
            "symbolic_seconds": symbolic_times,
            "paired_symbolic_over_legacy": ratios,
            "compile_symbolic_over_legacy": compile_ratio,
            "legacy_outcomes": outcome_summary(legacy),
            "symbolic_outcomes": outcome_summary(symbolic),
            "outside_aa_envelope": (
                float(ratios["median"]) < noise_low or float(ratios["median"]) > noise_high
            ),
            "median_absolute_gap_seconds": abs(
                float(symbolic_times["median"]) - float(legacy_times["median"])
            ),
        }
        rows.append(row)

    rows.sort(key=lambda row: row["case_id"])
    ranked = sorted(
        rows,
        key=lambda row: (
            row["outside_aa_envelope"],
            row["median_absolute_gap_seconds"],
        ),
        reverse=True,
    )
    summary = {
        "schema": "clifft_backend_comparison_summary_v1",
        "raw": str(args.raw.resolve()),
        "clifft_commit": document["clifft_commit"],
        "aa_ratio_b_over_a": aa_distributions,
        "rows": rows,
        "profile_ranking": [row["case_id"] for row in ranked],
    }

    markdown = []
    markdown.append("# Legacy versus symbolic-coordinate baseline")
    markdown.append("")
    if aa_distributions:
        for case_id, aa_distribution in aa_distributions.items():
            markdown.append(
                f"A/A `{case_id}` paired B/A ratio: median "
                f"{aa_distribution['median']:.4f}, range "
                f"{aa_distribution['min']:.4f}-{aa_distribution['max']:.4f}, "
                f"MAD {aa_distribution['mad']:.4f}."
            )
    else:
        markdown.append("A/A control skipped; the provisional noise envelope is +/-5%.")
    markdown.append("")
    markdown.append(
        "`symbolic/legacy` above 1 means the symbolic backend is slower. "
        "Outcome intervals are Wilson 95% intervals in the JSON summary."
    )
    markdown.append("")
    markdown.append(
        "| Case | Legacy ms | Symbolic ms | Symbolic/legacy | Legacy IQR | "
        "Symbolic IQR | Discard legacy/symbolic | Compile ratio |"
    )
    markdown.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        legacy_time = row["legacy_seconds"]
        symbolic_time = row["symbolic_seconds"]
        paired_ratio = row["paired_symbolic_over_legacy"]
        legacy_iqr = (float(legacy_time["q3"]) - float(legacy_time["q1"])) / float(
            legacy_time["median"]
        )
        symbolic_iqr = (float(symbolic_time["q3"]) - float(symbolic_time["q1"])) / float(
            symbolic_time["median"]
        )
        compile_ratio = row["compile_symbolic_over_legacy"]
        markdown.append(
            f"| `{row['case_id']}` | {format_ms(float(legacy_time['median']))} | "
            f"{format_ms(float(symbolic_time['median']))} | "
            f"{float(paired_ratio['median']):.3f}x | "
            f"{format_percent(legacy_iqr)} | {format_percent(symbolic_iqr)} | "
            f"{format_percent(row['legacy_outcomes']['discard_fraction'])} / "
            f"{format_percent(row['symbolic_outcomes']['discard_fraction'])} | "
            f"{compile_ratio:.2f}x |"
            if compile_ratio is not None
            else f"| `{row['case_id']}` | {format_ms(float(legacy_time['median']))} | "
            f"{format_ms(float(symbolic_time['median']))} | "
            f"{float(paired_ratio['median']):.3f}x | "
            f"{format_percent(legacy_iqr)} | {format_percent(symbolic_iqr)} | "
            f"{format_percent(row['legacy_outcomes']['discard_fraction'])} / "
            f"{format_percent(row['symbolic_outcomes']['discard_fraction'])} | n/a |"
        )
    markdown.append("")
    markdown.append("## Initial profile ranking")
    markdown.append("")
    for index, row in enumerate(ranked[:10], start=1):
        markdown.append(
            f"{index}. `{row['case_id']}`: ratio "
            f"{row['paired_symbolic_over_legacy']['median']:.3f}x, absolute median gap "
            f"{row['median_absolute_gap_seconds']:.3f}s."
        )
    markdown.append("")

    markdown_path = args.markdown or args.raw.with_name("summary.md")
    json_path = args.json_output or args.raw.with_name("summary.json")
    markdown_path.write_text("\n".join(markdown))
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    main()
