"""Tests for import-time CPU baseline checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_cpu_check_module():
    path = Path(__file__).parents[2] / "src" / "python" / "clifft" / "_cpu_check.py"
    spec = importlib.util.spec_from_file_location("clifft_cpu_check_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu_check = _load_cpu_check_module()


def _linux_x86(monkeypatch: pytest.MonkeyPatch, flags: set[str]) -> None:
    monkeypatch.setattr(cpu_check.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cpu_check.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cpu_check, "_linux_cpu_flags", lambda: flags)


def test_generic_baseline_never_checks_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux_x86(monkeypatch, set())
    cpu_check.ensure_supported_cpu("generic")


def test_x86_64_v2_accepts_required_linux_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux_x86(
        monkeypatch,
        {"cx16", "lahf_lm", "popcnt", "pni", "ssse3", "sse4_1", "sse4_2"},
    )
    cpu_check.ensure_supported_cpu("x86-64-v2")


def test_x86_64_v2_reports_missing_linux_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux_x86(monkeypatch, {"cx16", "lahf_lm", "popcnt"})

    with pytest.raises(ImportError, match="ssse3"):
        cpu_check.ensure_supported_cpu("x86-64-v2")


def test_x86_64_v3_accepts_amd_abm_for_lzcnt(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux_x86(
        monkeypatch,
        {
            "cx16",
            "lahf_lm",
            "popcnt",
            "pni",
            "ssse3",
            "sse4_1",
            "sse4_2",
            "avx",
            "avx2",
            "bmi1",
            "bmi2",
            "f16c",
            "fma",
            "abm",
            "movbe",
            "xsave",
        },
    )
    cpu_check.ensure_supported_cpu("x86-64-v3")
