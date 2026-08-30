"""Tests for the paired benchmark canary report."""

from __future__ import annotations

import importlib.util
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


def _write_results(path: Path, results: dict[str, float], failures: int = 0) -> Path:
    benchmarks = "\n".join(
        f'<BenchmarkResults name="{name}"><mean value="{value}"/></BenchmarkResults>'
        for name, value in results.items()
    )
    path.write_text(
        f'<Catch2TestRun>{benchmarks}<OverallResults failures="{failures}"/></Catch2TestRun>'
    )
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
        _write_results(tmp_path / "base-first.xml", base),
        _write_results(tmp_path / "head-first.xml", head),
        _write_results(tmp_path / "head-second.xml", head),
        _write_results(tmp_path / "base-second.xml", base),
    )

    report = benchmark_canary.render_report(
        comparisons,
        base_label="main",
        base_sha="1234567890",
        head_sha="abcdef0123",
        samples=10,
        added=added,
        removed=removed,
        cpu="Example CPU",
        compiler="Example compiler",
        run_url="https://example.com/run",
    )

    assert "**Possible regression detected:** 1 of 4 benchmarks was at least 10% slower." in report
    assert "| stable | 100 ns | 104 ns | +4.0% | No material change |" in report
    assert "| notable | 100 ns | 106 ns | **+6.0%** | Notable slowdown |" in report
    assert "| regression | 100 ns | 114 ns | **+14.0%** | Possible regression |" in report
    assert "| improvement | 100 ns | 89.0 ns | **-11.0%** | Improvement |" in report
    assert "`main` (`1234567`)" in report
    assert "Example CPU" in report
    assert "[View workflow run](https://example.com/run)" in report


def test_report_notes_benchmark_set_changes(tmp_path: Path) -> None:
    comparisons, added, removed = benchmark_canary.compare_runs(
        _write_results(tmp_path / "base-first.xml", {"common": 100.0, "removed": 20.0}),
        _write_results(tmp_path / "head-first.xml", {"common": 100.0, "added": 30.0}),
        _write_results(tmp_path / "head-second.xml", {"common": 100.0, "added": 30.0}),
        _write_results(tmp_path / "base-second.xml", {"common": 100.0, "removed": 20.0}),
    )

    report = benchmark_canary.render_report(
        comparisons,
        base_label="main",
        base_sha="1234567",
        head_sha="abcdef0",
        samples=10,
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
            _write_results(tmp_path / "base-first.xml", {"one": 100.0}),
            _write_results(tmp_path / "head-first.xml", {"one": 100.0}),
            _write_results(tmp_path / "head-second.xml", {"one": 100.0}),
            _write_results(tmp_path / "base-second.xml", {"other": 100.0}),
        )


def test_parser_rejects_failed_catch_run(tmp_path: Path) -> None:
    path = _write_results(tmp_path / "failed.xml", {"one": 100.0}, failures=1)
    with pytest.raises(ValueError, match="did not succeed"):
        benchmark_canary.parse_catch2_benchmarks(path)
