"""Keep the runtime audit sensitive to both C and weak C++ OpenMP symbols."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_audit_module():
    path = Path(__file__).parents[2] / "tools" / "ci" / "audit_macos_wheel.py"
    spec = importlib.util.spec_from_file_location("clifft_macos_wheel_audit_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


@pytest.mark.parametrize(
    "symbol",
    [
        "_omp_get_num_threads",
        "___kmpc_fork_call",
        "___kmp_threads",
        "_kmp_set_defaults",
        "_GOMP_parallel",
        "__Z16__kmp_suspend_64ILb0ELb1EEviP11kmp_flag_64IXT_EXT0_EE",
        "__ZN11kmp_flag_64ILb0ELb1EE4waitEP8kmp_infoiPv",
        "__ZTV11kmp_flag_64ILb0ELb1EE",
        "__ZTI11kmp_flag_64ILb0ELb1EE",
        "__ZTS11kmp_flag_64ILb0ELb1EE",
    ],
)
def test_audit_detects_openmp_symbols(symbol: str) -> None:
    assert audit._is_openmp_symbol(symbol)


@pytest.mark.parametrize(
    "symbol",
    [
        "_PyInit__clifft_core",
        "_my_kmp_helper",
        "_my_omp_helper",
        "__ZN6clifft15omp_diagnosticsEv",
        "_memcmp",
    ],
)
def test_audit_allows_unrelated_symbols(symbol: str) -> None:
    assert not audit._is_openmp_symbol(symbol)
