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

REPOSITORY = "unitaryfoundation/clifft"
HEAD_REPOSITORY = "contributor/clifft"
BASE_SHA = "1" * 40
HEAD_SHA = "2" * 40


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


def _write_fork_evidence(path: Path) -> Path:
    path.mkdir()
    for directory_name, multiplier in (
        ("base-first", 1.0),
        ("head-first", 1.1),
        ("head-second", 1.1),
        ("base-second", 1.0),
    ):
        directory = path / directory_name
        directory.mkdir()
        for index, name in enumerate(benchmark_canary.DISPLAY_NAMES, start=1):
            _write_results(directory / f"{name}.json", {name: index * 100.0 * multiplier})
    benchmark_canary.write_manifest(
        path / "manifest.json",
        repository=REPOSITORY,
        pr_number=123,
        base_label="main",
        base_sha=BASE_SHA,
        head_repository=HEAD_REPOSITORY,
        head_sha=HEAD_SHA,
        run_id=456,
        repetitions=benchmark_canary.CANARY_REPETITIONS,
        min_time=benchmark_canary.CANARY_MIN_TIME,
        warmup_time=benchmark_canary.CANARY_WARMUP_TIME,
        cpu_core="3",
        cpu="Example CPU",
        compiler="Example compiler",
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
    assert "## :baby_chick: Performance Canary" in report
    assert "<summary>View 4 benchmark results</summary>" not in report
    assert "<sub>Compared <code>main</code> (<code>1234567</code>)" in report
    assert "Example CPU" in report
    assert "Pinned logical CPU: <code>3</code>" in report
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


def test_inconclusive_summary_retains_improvements() -> None:
    improvement = benchmark_canary.Comparison(
        name="improvement",
        base_ns=100.0,
        head_ns=88.0,
        change=-0.12,
        first_change=-0.12,
        second_change=-0.12,
        max_relative_stddev=0.01,
    )
    inconclusive = benchmark_canary.Comparison(
        name="inconclusive",
        base_ns=100.0,
        head_ns=112.0,
        change=0.12,
        first_change=0.12,
        second_change=0.12,
        max_relative_stddev=0.06,
    )

    summary = benchmark_canary._summary([improvement, inconclusive])

    assert "**No confirmed regressions detected.**" in summary
    assert "1 benchmark showed an improvement of at least 5%." in summary
    assert "1 benchmark comparison was inconclusive." in summary


def test_report_links_workload_name_to_source_rationale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.example")
    monkeypatch.setenv("GITHUB_REPOSITORY", "unitaryfoundation/clifft")
    source_lines = benchmark_canary.BENCHMARK_SOURCE.read_text().splitlines()
    rationale_line = next(
        index
        for index, line in enumerate(source_lines, start=1)
        if line.startswith("// Expansion-only suffixes")
    )
    source_links = benchmark_canary._benchmark_source_links("abcdef0")
    assert benchmark_canary.DISPLAY_NAMES.keys() <= source_links.keys()
    for source_link in source_links.values():
        linked_line = int(source_link.rsplit("#L", 1)[1])
        assert source_lines[linked_line - 1].startswith("//")
    comparison = benchmark_canary.Comparison(
        name="squeeze_parallel_t_8192",
        base_ns=100.0,
        head_ns=100.0,
        change=0.0,
        first_change=0.0,
        second_change=0.0,
        max_relative_stddev=0.0,
    )

    report = benchmark_canary.render_report(
        [comparison],
        base_label="main",
        base_sha="1234567",
        head_sha="abcdef0",
        repetitions=3,
        min_time=0.5,
        warmup_time=0.2,
        added=[],
        removed=[],
        cpu="CPU",
        compiler="compiler",
    )

    expected_link = (
        "https://github.example/unitaryfoundation/clifft/blob/abcdef0/"
        f"benchmarks/clifft_benchmarks.cc#L{rationale_line}"
    )
    assert "<summary>View 1 benchmark result</summary>" in report
    assert f"| [Squeeze 8192 T gates]({expected_link}) |" in report


def test_failure_report_uses_succinct_heading() -> None:
    report = benchmark_canary.render_failure(
        base_label="main",
        base_sha="1234567",
        head_sha="abcdef0",
    )

    assert "## :baby_chick: Performance Canary" in report
    assert "(advisory)" not in report


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


def test_parser_combines_one_benchmark_per_file(tmp_path: Path) -> None:
    directory = tmp_path / "results"
    directory.mkdir()
    _write_results(directory / "one.json", {"one": 100.0})
    _write_results(directory / "two.json", {"two": 200.0})

    assert set(benchmark_canary.parse_google_benchmarks(directory)) == {"one", "two"}


def test_fork_report_validates_evidence_and_uses_base_workloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _write_fork_evidence(tmp_path / "evidence")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.example")
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)

    report = benchmark_canary.render_fork_report(
        evidence,
        expected_repository=REPOSITORY,
        expected_pr_number=123,
        expected_base_label="main",
        expected_base_sha=BASE_SHA,
        expected_head_repository=HEAD_REPOSITORY,
        expected_head_sha=HEAD_SHA,
        expected_run_id=456,
        run_url="https://github.example/run/456",
    )

    workload_count = len(benchmark_canary.DISPLAY_NAMES)
    assert (
        f"**Possible regression detected:** {workload_count} of {workload_count} "
        "benchmarks were at least 10% slower." in report
    )
    assert f"/{REPOSITORY}/blob/{BASE_SHA}/benchmarks/clifft_benchmarks.cc#L" in report
    assert f"/{REPOSITORY}/blob/{HEAD_SHA}/benchmarks/clifft_benchmarks.cc#L" not in report
    assert "Runner CPU: ` Example CPU `" in report
    assert "Workloads and fixtures come from the base revision for fork isolation." in report
    assert "[View workflow run](https://github.example/run/456)" in report


def test_fork_report_rejects_manifest_not_bound_to_run(tmp_path: Path) -> None:
    evidence = _write_fork_evidence(tmp_path / "evidence")
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["head_sha"] = "3" * 40
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="head_sha does not match"):
        benchmark_canary.render_fork_report(
            evidence,
            expected_repository=REPOSITORY,
            expected_pr_number=123,
            expected_base_label="main",
            expected_base_sha=BASE_SHA,
            expected_head_repository=HEAD_REPOSITORY,
            expected_head_sha=HEAD_SHA,
            expected_run_id=456,
            run_url="https://github.example/run/456",
        )


def test_fork_report_rejects_base_revision_not_bound_to_artifact(tmp_path: Path) -> None:
    evidence = _write_fork_evidence(tmp_path / "evidence")
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_sha"] = "3" * 40
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="base_sha does not match"):
        benchmark_canary.render_fork_report(
            evidence,
            expected_repository=REPOSITORY,
            expected_pr_number=123,
            expected_base_label="main",
            expected_base_sha=BASE_SHA,
            expected_head_repository=HEAD_REPOSITORY,
            expected_head_sha=HEAD_SHA,
            expected_run_id=456,
            run_url="https://github.example/run/456",
        )


def test_fork_report_rejects_unexpected_workload_files(tmp_path: Path) -> None:
    evidence = _write_fork_evidence(tmp_path / "evidence")
    _write_results(evidence / "base-first" / "invented.json", {"invented": 100.0})

    with pytest.raises(ValueError, match="unexpected file set"):
        benchmark_canary.render_fork_report(
            evidence,
            expected_repository=REPOSITORY,
            expected_pr_number=123,
            expected_base_label="main",
            expected_base_sha=BASE_SHA,
            expected_head_repository=HEAD_REPOSITORY,
            expected_head_sha=HEAD_SHA,
            expected_run_id=456,
            run_url="https://github.example/run/456",
        )


def test_fork_report_rejects_multiline_environment_text(tmp_path: Path) -> None:
    evidence = _write_fork_evidence(tmp_path / "evidence")
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["compiler"] = "compiler\nforged report"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="invalid compiler"):
        benchmark_canary.render_fork_report(
            evidence,
            expected_repository=REPOSITORY,
            expected_pr_number=123,
            expected_base_label="main",
            expected_base_sha=BASE_SHA,
            expected_head_repository=HEAD_REPOSITORY,
            expected_head_sha=HEAD_SHA,
            expected_run_id=456,
            run_url="https://github.example/run/456",
        )


def test_fork_report_quotes_markdown_in_environment_text(tmp_path: Path) -> None:
    evidence = _write_fork_evidence(tmp_path / "evidence")
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cpu"] = "[click](https://example.invalid)"
    manifest["compiler"] = "compiler `with` backticks"
    manifest_path.write_text(json.dumps(manifest))

    report = benchmark_canary.render_fork_report(
        evidence,
        expected_repository=REPOSITORY,
        expected_pr_number=123,
        expected_base_label="main",
        expected_base_sha=BASE_SHA,
        expected_head_repository=HEAD_REPOSITORY,
        expected_head_sha=HEAD_SHA,
        expected_run_id=456,
        run_url="https://github.example/run/456",
    )

    assert "Runner CPU: ` [click](https://example.invalid) `" in report
    assert "Compiler: `` compiler `with` backticks ``" in report
    assert "<code>[click]" not in report


def test_manifest_rejects_settings_not_supported_by_reporter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="settings do not match"):
        benchmark_canary.write_manifest(
            tmp_path / "manifest.json",
            repository=REPOSITORY,
            pr_number=123,
            base_label="main",
            base_sha=BASE_SHA,
            head_repository=HEAD_REPOSITORY,
            head_sha=HEAD_SHA,
            run_id=456,
            repetitions=4,
            min_time=benchmark_canary.CANARY_MIN_TIME,
            warmup_time=benchmark_canary.CANARY_WARMUP_TIME,
            cpu_core="3",
            cpu="Example CPU",
            compiler="Example compiler",
        )
