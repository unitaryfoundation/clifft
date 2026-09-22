"""Exercise conformance onboarding and omission checks through pytest collection."""

from dataclasses import replace
from pathlib import Path

import pytest
import sampling_conformance as conformance
from test_sampling_integration import (
    test_syndrome_outputs_preserve_record_snapshots as check_record_snapshots,
)
from utils_conformance import CPU_SAMPLING_MODES, CpuSamplingMode

import clifft

pytest_plugins = ("pytester",)

_MEASUREMENT_TEST = """
import numpy as np
import pytest

@pytest.mark.sampling_conformance("terminal-measurements")
def test_measurement(sampling_api):
    result = sampling_api.sample(sampling_api.compile("X 0\\nM 0"), 65, seed=1)
    np.testing.assert_array_equal(result.measurements, np.ones((65, 1), dtype=np.uint8))
"""

_RESET_TEST = """
@pytest.mark.sampling_conformance("measurement-reset")
def test_reset(sampling_api):
    result = sampling_api.sample(sampling_api.compile("X 0\\nR 0\\nM 0"), 65, seed=1)
    np.testing.assert_array_equal(result.measurements, np.zeros((65, 1), dtype=np.uint8))
"""


@pytest.fixture
def matrix(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    # These isolated collections exercise our plugin without unrelated
    # installed plugins changing collection or warning policies.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.syspathinsert(Path(__file__).parent)
    pytester.makeconftest('pytest_plugins = ("sampling_conformance",)')
    monkeypatch.setattr(conformance, "FEATURES", frozenset({"terminal-measurements"}))
    monkeypatch.setattr(
        conformance,
        "CONFIGURATIONS",
        tuple(
            replace(configuration, support={"terminal-measurements": None})
            for configuration in conformance.CONFIGURATIONS
        ),
    )
    pytester.makepyfile(test_behaviors=_MEASUREMENT_TEST)
    return pytester


def test_mode_and_feature_additions_inherit_existing_tests(
    matrix: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix.runpytest("-q", "--sampling-coverage").assert_outcomes(passed=3)

    monkeypatch.setattr(conformance, "FEATURES", conformance.FEATURES | {"measurement-reset"})
    configurations = tuple(
        replace(configuration, support={**configuration.support, "measurement-reset": None})
        for configuration in conformance.CONFIGURATIONS
    )
    extra = conformance.SamplingConfiguration(
        CpuSamplingMode("packed-17", 17), dict(configurations[0].support)
    )
    monkeypatch.setattr(conformance, "CONFIGURATIONS", (*configurations, extra))
    matrix.makepyfile(test_behaviors=_MEASUREMENT_TEST + _RESET_TEST)
    result = matrix.runpytest("-v", "--sampling-coverage")
    result.assert_outcomes(passed=8)
    result.stdout.fnmatch_lines(
        ["*test_measurement*packed-17*PASSED*", "*test_reset*packed-17*PASSED*"]
    )


@pytest.mark.parametrize(
    ("omission", "message"),
    [
        ("mode", "Required CPU sampling configuration missing"),
        ("decision", "Sampling support decision required"),
        ("test", "No shared tests for supported behavior"),
        ("case-mode", "Incomplete sampling modes"),
        ("skip", "Declare sampling exclusions instead of skipping"),
    ],
)
def test_collection_rejects_coverage_omissions(
    matrix: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, omission: str, message: str
) -> None:
    if omission == "mode":
        monkeypatch.setattr(conformance, "CONFIGURATIONS", conformance.CONFIGURATIONS[1:])
    elif omission == "decision":
        first, *rest = conformance.CONFIGURATIONS
        monkeypatch.setattr(conformance, "CONFIGURATIONS", (replace(first, support={}), *rest))
    elif omission == "test":
        monkeypatch.setattr(conformance, "FEATURES", conformance.FEATURES | {"measurement-reset"})
        monkeypatch.setattr(
            conformance,
            "CONFIGURATIONS",
            tuple(
                replace(configuration, support={**configuration.support, "measurement-reset": None})
                for configuration in conformance.CONFIGURATIONS
            ),
        )
    elif omission == "case-mode":
        matrix.makeconftest("""
import pytest
pytest_plugins = ("sampling_conformance",)

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    items[:] = [item for item in items if "packed-65" not in item.nodeid]
""")
    else:
        matrix.makepyfile(
            test_behaviors=_MEASUREMENT_TEST.replace(
                "def test_measurement",
                '@pytest.mark.skip(reason="missing coverage")\ndef test_measurement',
            )
        )
    result = matrix.runpytest("--collect-only", "--sampling-coverage", "-q")
    assert result.ret != 0
    assert message in result.stdout.str() + result.stderr.str()


def test_targeted_runs_do_not_require_the_entire_inventory(
    matrix: pytest.Pytester,
) -> None:
    matrix.runpytest("test_behaviors.py::test_measurement[packed-65]", "-q").assert_outcomes(
        passed=1
    )


def test_shared_test_cannot_bypass_its_selected_mode(matrix: pytest.Pytester) -> None:
    matrix.makepyfile(
        test_behaviors="import clifft\n"
        + _MEASUREMENT_TEST.replace("sampling_api.sample(", "clifft.sample(")
    )
    result = matrix.runpytest("--sampling-coverage", "-q")
    result.assert_outcomes(passed=3, errors=3)
    assert "Shared test did not sample through its selected mode" in result.stdout.str()


@pytest.mark.parametrize("failure_site", ["setup", "body", "sampling"])
def test_original_failures_do_not_add_a_coverage_error(
    matrix: pytest.Pytester, failure_site: str
) -> None:
    if failure_site == "setup":
        source = _MEASUREMENT_TEST.replace(
            "def test_measurement(sampling_api):",
            "def test_measurement(sampling_api, broken_setup):",
        )
        source += """
@pytest.fixture
def broken_setup(sampling_api):
    raise RuntimeError("failure before sampling")
"""
    elif failure_site == "body":
        source = _MEASUREMENT_TEST.replace(
            "    result =", '    raise RuntimeError("failure before sampling")\n    result ='
        )
    else:
        source = """
import pytest

@pytest.mark.sampling_conformance("terminal-measurements")
def test_measurement(sampling_api):
    program = sampling_api.compile("M 0\\nDETECTOR rec[-1]", postselection_mask=[1])
    sampling_api.sample(program, 65, seed=1)
"""
    matrix.makepyfile(test_behaviors=source)
    result = matrix.runpytest("--sampling-coverage", "-q")
    result.assert_outcomes(
        errors=3 if failure_site == "setup" else 0,
        failed=0 if failure_site == "setup" else 3,
    )
    assert "Shared test did not sample through its selected mode" not in result.stdout.str()


def test_rejected_calls_do_not_establish_execution_coverage(matrix: pytest.Pytester) -> None:
    matrix.makepyfile(
        test_behaviors="""
import pytest

@pytest.mark.sampling_conformance("terminal-measurements")
def test_measurement(sampling_api):
    program = sampling_api.compile("M 0\\nDETECTOR rec[-1]", postselection_mask=[1])
    with pytest.raises(ValueError, match="sample_survivors"):
        sampling_api.sample(program, 65, seed=1)
"""
    )
    result = matrix.runpytest("--sampling-coverage", "-q")
    result.assert_outcomes(passed=3, errors=3)
    assert "Shared test did not sample through its selected mode" in result.stdout.str()


def test_explicit_exclusion_is_reported_without_claiming_coverage(
    matrix: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, *rest = conformance.CONFIGURATIONS
    monkeypatch.setattr(
        conformance,
        "CONFIGURATIONS",
        (
            replace(first, support={"terminal-measurements": "synthetic unsupported behavior"}),
            *rest,
        ),
    )
    result = matrix.runpytest("--sampling-coverage", "-q")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*single-shot: excluded: synthetic unsupported behavior*"])


def test_inherited_snapshot_assertions_reject_a_packed_output_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = clifft.sample

    def stale_observable(program, shots, **kwargs):
        result = sample(program, shots, **kwargs)
        if kwargs.get("batch_size") == 65:
            # Reproduce the observable using the latest record instead of
            # the historical value captured before readout noise.
            result.observables[:, 0] = result.measurements[:, 0] ^ 1
        return result

    monkeypatch.setattr(clifft, "sample", stale_observable)
    for mode in CPU_SAMPLING_MODES:
        if mode.name == "packed-65":
            with pytest.raises(AssertionError):
                check_record_snapshots(mode)
        else:
            check_record_snapshots(mode)


@pytest.mark.parametrize("mode", CPU_SAMPLING_MODES, ids=lambda mode: mode.name)
@pytest.mark.parametrize("keep_records", [False, True])
def test_survivor_adapter_preserves_execution_and_output_options(
    mode: CpuSamplingMode, keep_records: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    program = object()
    monkeypatch.setattr(
        clifft, "sample_survivors", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    mode.sample_survivors(program, 257, seed=41, keep_records=keep_records)
    assert calls == [
        (
            (program, 257),
            {"seed": 41, "keep_records": keep_records, "threads": 1, "batch_size": mode.batch_size},
        )
    ]
