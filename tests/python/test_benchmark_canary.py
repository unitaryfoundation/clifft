"""Tests for the paired benchmark canary report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_benchmark_canary_module():
    path = Path(__file__).parents[2] / "tools" / "ci" / "benchmark_canary.py"
    spec = importlib.util.spec_from_file_location("clifft_benchmark_canary_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_canary = _load_benchmark_canary_module()


def _write_results(
    path: Path,
    results: dict[str, float],
    *,
    relative_stddev: float = 0.0,
    error: str | None = None,
) -> Path:
    benchmarks = []
    for name, value in results.items():
        for repetition, multiplier in enumerate(
            (1.0 - relative_stddev, 1.0, 1.0 + relative_stddev)
        ):
            benchmarks.append(
                {
                    "name": name,
                    "run_name": name,
                    "run_type": "iteration",
                    "repetition_index": repetition,
                    "cpu_time": value * multiplier,
                    "time_unit": "ns",
                    "error_occurred": error is not None,
                    "error_message": error or "",
                }
            )
        benchmarks.append(
            {
                "name": f"{name}_median",
                "run_name": name,
                "run_type": "aggregate",
                "aggregate_name": "median",
                "cpu_time": value,
                "time_unit": "ns",
            }
        )
    path.write_text(json.dumps({"context": {}, "benchmarks": benchmarks}))
    return path


def test_report_classifies_paired_changes(tmp_path: Path) -> None:
    base = {
        "stable": 100.0,
        "notable": 100.0,
        "regression": 100.0,
        "improvement": 100.0,
    }
    head = {
        "stable": 104.0,
        "notable": 106.0,
        "regression": 114.0,
        "improvement": 89.0,
    }
    comparisons, added, removed = benchmark_canary.compare_runs(
        _write_results(tmp_path / "base-first.json", base),
        _write_results(tmp_path / "head-first.json", head),
        _write_results(tmp_path / "head-second.json", head),
        _write_results(tmp_path / "base-second.json", base),
    )

    report = benchmark_canary.render_report(
        comparisons,
        base_label="main",
        base_sha="1234567890",
        head_sha="abcdef0123",
        repetitions=3,
        min_time=0.5,
        warmup_time=0.2,
        added=added,
        removed=removed,
        cpu="Example CPU",
        compiler="Example compiler",
        cpu_core="3",
        run_url="https://example.com/run",
    )

    assert "**Possible regression detected:** 1 of 4 benchmarks was at least 10% slower." in report
    assert "| stable | 100 ns | 104 ns | +4.0% | No material change |" in report
    assert "| notable | 100 ns | 106 ns | **+6.0%** | Notable slowdown |" in report
    assert "| regression | 100 ns | 114 ns | **+14.0%** | Possible regression |" in report
    assert "| improvement | 100 ns | 89.0 ns | **-11.0%** | Improvement |" in report
    assert "`main` (`1234567`)" in report
    assert "Example CPU" in report
    assert "Pinned logical CPU: 3" in report
    assert "Release, x86-64-v2, ThinLTO, lld" in report
    assert "[View workflow run](https://example.com/run)" in report


def test_report_notes_benchmark_set_changes(tmp_path: Path) -> None:
    comparisons, added, removed = benchmark_canary.compare_runs(
        _write_results(tmp_path / "base-first.json", {"common": 100.0, "removed": 20.0}),
        _write_results(tmp_path / "head-first.json", {"common": 100.0, "added": 30.0}),
        _write_results(tmp_path / "head-second.json", {"common": 100.0, "added": 30.0}),
        _write_results(tmp_path / "base-second.json", {"common": 100.0, "removed": 20.0}),
    )

    report = benchmark_canary.render_report(
        comparisons,
        base_label="main",
        base_sha="1234567",
        head_sha="abcdef0",
        repetitions=3,
        min_time=0.5,
        warmup_time=0.2,
        added=added,
        removed=removed,
        cpu="CPU",
        compiler="compiler",
    )

    assert "Added in the PR and not compared: added" in report
    assert "Absent from the PR and not compared: removed" in report


def test_compare_rejects_inconsistent_repeat(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two base runs"):
        benchmark_canary.compare_runs(
            _write_results(tmp_path / "base-first.json", {"one": 100.0}),
            _write_results(tmp_path / "head-first.json", {"one": 100.0}),
            _write_results(tmp_path / "head-second.json", {"one": 100.0}),
            _write_results(tmp_path / "base-second.json", {"other": 100.0}),
        )


def test_material_change_requires_consistent_low_noise_pairs(tmp_path: Path) -> None:
    inconsistent, _, _ = benchmark_canary.compare_runs(
        _write_results(tmp_path / "inconsistent-base-first.json", {"bench": 100.0}),
        _write_results(tmp_path / "inconsistent-head-first.json", {"bench": 120.0}),
        _write_results(tmp_path / "inconsistent-head-second.json", {"bench": 100.0}),
        _write_results(tmp_path / "inconsistent-base-second.json", {"bench": 100.0}),
    )
    noisy, _, _ = benchmark_canary.compare_runs(
        _write_results(tmp_path / "noisy-base-first.json", {"bench": 100.0}),
        _write_results(tmp_path / "noisy-head-first.json", {"bench": 120.0}, relative_stddev=0.06),
        _write_results(tmp_path / "noisy-head-second.json", {"bench": 120.0}),
        _write_results(tmp_path / "noisy-base-second.json", {"bench": 100.0}),
    )

    assert inconsistent[0].assessment == "Inconclusive"
    assert noisy[0].assessment == "Inconclusive"


def test_parser_rejects_failed_google_benchmark_run(tmp_path: Path) -> None:
    path = _write_results(tmp_path / "failed.json", {"one": 100.0}, error="benchmark failed")
    with pytest.raises(ValueError, match="did not succeed"):
        benchmark_canary.parse_google_benchmarks(path)


def test_parser_normalizes_time_units(tmp_path: Path) -> None:
    path = tmp_path / "units.json"
    path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": "one",
                        "run_name": "one",
                        "run_type": "iteration",
                        "cpu_time": 1.25,
                        "time_unit": "ms",
                    }
                ]
            }
        )
    )

    assert benchmark_canary.parse_google_benchmarks(path)["one"].median_ns == 1_250_000
