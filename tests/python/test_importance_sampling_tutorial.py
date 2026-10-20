"""Execute the cultivation tutorial setup without its full statistical workload."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from utils_conformance import CpuSamplingMode

ROOT = Path(__file__).resolve().parents[2]


def test_tutorial_compile_block(
    importance_sampling_mode: CpuSamplingMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    guide = ROOT / "docs/guide/importance-sampling-tutorial.md"
    source = guide.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    namespace: dict[str, Any] = {}
    monkeypatch.chdir(ROOT)
    # The rest of this guide intentionally skips codeblocks because it runs
    # millions of shots. Execute its actual setup so API drift still fails CI.
    exec(compile(source, str(guide), "exec"), namespace)
    program = namespace["prog"]
    assert program.num_detectors == 20
    assert program.num_observables == 1
    assert len(program.noise_site_probabilities) == 518
    result = importance_sampling_mode.sample_k_survivors(program, 65, k=0, seed=42)
    assert result.total_shots == result.passed_shots == 65
    assert result.logical_errors == 0


@pytest.fixture(scope="module")
def tutorial() -> ModuleType:
    path = ROOT / "docs/guide/scripts/importance_sampling_tutorial.py"
    spec = importlib.util.spec_from_file_location("importance_sampling_tutorial", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("gate", ["T", "S"])
def test_tutorial_script_strata(
    tutorial: ModuleType, gate: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tutorial, "MAX_K", 1)
    monkeypatch.setattr(tutorial, "SHOTS_PER_STRATUM", 65)
    path = getattr(tutorial, f"{gate}_GATE_CIRCUIT")
    pmf, strata, n_sites = tutorial.run_stratified_is(path, f"{gate}-gate")
    assert n_sites == 518
    assert len(pmf) == len(strata) == 2
    assert strata[0] == {"k": 0, "total": 65, "passed": 65, "errors": 0}
    assert strata[1]["k"] == 1
    assert strata[1]["total"] == 65
    assert strata[1]["passed"] < 65
    assert 0 <= strata[1]["errors"] <= strata[1]["passed"]
    sweep = tutorial.compute_sweep(strata, n_sites)
    assert len(sweep["discard_rates"]) == len(tutorial.P_VALUES)
    assert np.isfinite(sweep["error_rates"]).all()
