"""Complete physical-history checks and rejection of unsupported coherent records."""

# ruff: noqa: E402

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("cirq")

from audit_folded_protocol import validate_trace
from compile_folded_protocol import compile_protocol, monomial_plan
from folded_check_contraction import FoldedChecks
from folded_msc import Builder
from folded_msc_family import make_circuit


@pytest.mark.parametrize("distance,native_growth", [(3, False), (5, False), (7, False), (7, True)])
@pytest.mark.parametrize("probability", [0, 0.03])
def test_native_complete_attempt_matches_physical_replay(
    tmp_path, distance, native_growth, probability
):
    sampler = Path("build-study/sample_folded_protocol").resolve()
    reference = Path("build-study/replay_cultivation").resolve()
    if not sampler.is_file() or not reference.is_file():
        pytest.skip("build the complete folded sampler and physical replay harness")
    circuit = make_circuit(probability, distance=distance)
    bundle = tmp_path / "bundle"
    metadata = compile_protocol(circuit, distance, bundle, native_growth=native_growth)
    output = subprocess.check_output(
        [str(sampler), str(bundle), "1", "4", "651", "0", "off", "0"], text=True
    )
    rows = [json.loads(line) for line in output.splitlines()]
    assert rows[-1]["normalization_error"] < 1e-10
    assert rows[-1]["continuation_error"] < 1e-10
    if native_growth:
        assert rows[-1]["prefix_peak_width"] <= 8
    for i, sample in enumerate(rows[:-1]):
        result = validate_trace(circuit, metadata, sample, reference, oracle=i == 0)
        if not probability:
            assert result["accepted"] and not result["observable"]
    if probability:
        assert any(sample["prefix_faults"] and sample["suffix_faults"] for sample in rows[:-1])
        if native_growth:
            assert any(sample["earlier_faults"] and sample["growth_faults"] for sample in rows[:-1])


def test_compiler_rejects_measurement_that_distinguishes_coherent_cat_branches():
    builder = Builder(0)
    builder.logical_check("check")
    operations = [op for op in builder.circuit.operations if op.probability is None]
    preparation = next(i for i, op in enumerate(operations) if op.name == "H")
    operations.insert(preparation + 1, replace(operations[preparation], name="M"))
    indexed = list(enumerate(operations))
    slots = {i: i for i, op in indexed if op.name in {"M", "R"}}
    with pytest.raises(ValueError, match="branch leaks"):
        monomial_plan(indexed, FoldedChecks(3), slots, {}, 1)
