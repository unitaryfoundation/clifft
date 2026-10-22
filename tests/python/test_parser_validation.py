"""Validate circuit text at public Python entry points before execution."""

import numpy as np
import pytest
import stim

import clifft
from clifft import noncomp


@pytest.fixture(params=["parse", "parse_file", "compile", "noncomp"])
def read_circuit(request, tmp_path):
    def read(text):
        if request.param == "parse":
            return clifft.parse(text)
        if request.param == "parse_file":
            path = tmp_path / "circuit.stim"
            path.write_text(text)
            return clifft.parse_file(str(path))
        if request.param == "compile":
            return clifft.compile(text)
        return noncomp.sample(text, noncomp.Model(), shots=4, seed=3)

    return read


@pytest.mark.parametrize(
    "gate",
    [
        "CX",
        "CNOT",
        "CY",
        "CZ",
        "SWAP",
        "ISWAP",
        "ISWAP_DAG",
        "SQRT_XX",
        "SQRT_XX_DAG",
        "SQRT_YY",
        "SQRT_YY_DAG",
        "SQRT_ZZ",
        "SQRT_ZZ_DAG",
        "CXSWAP",
        "CZSWAP",
        "SWAPCX",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "R_XX(0.125)",
        "R_YY(0.125)",
        "R_ZZ(0.125)",
    ],
)
@pytest.mark.parametrize("targets", ["0 0", "0 1 2 2"])
def test_duplicate_physical_pair_rejected(gate, targets, read_circuit):
    with pytest.raises(clifft.ParseError, match="Line 2: .*requires distinct qubit targets"):
        read_circuit(f"H 0\n{gate} {targets}\nM 0")


def test_duplicate_pair_inside_repeat_rejected(read_circuit):
    with pytest.raises(clifft.ParseError, match="Line 3: .*requires distinct qubit targets"):
        read_circuit("H 0\nREPEAT 2 {\nCX 0 0\n}\nM 0")


@pytest.mark.parametrize("probability", ["-0.1", "1.1"])
@pytest.mark.parametrize(
    ("prefix", "gate"),
    [
        ("", "CORRELATED_ERROR"),
        ("", "E"),
        ("E(0.5) X0\n", "ELSE_CORRELATED_ERROR"),
        ("E(1) X0\n", "ELSE_CORRELATED_ERROR"),
        ("E(0) X0\nELSE_CORRELATED_ERROR(1) Y0\n", "ELSE_CORRELATED_ERROR"),
    ],
)
@pytest.mark.parametrize("targets", ["X0", "", "X0*X0"])
def test_correlated_probability_rejected_before_lowering(
    probability, prefix, gate, targets, read_circuit
):
    line = prefix.count("\n") + 1
    with pytest.raises(
        clifft.ParseError,
        match=rf"Line {line}: .*probability must be finite and lie in \[0, 1\]",
    ):
        read_circuit(f"{prefix}{gate}({probability}) {targets}\nM 0")


@pytest.mark.parametrize("probability", ["nan", "inf", "-inf", "1e999"])
@pytest.mark.parametrize("prefix", ["", "E(1) X0\n"])
def test_nonfinite_correlated_probability_rejected(probability, prefix, read_circuit):
    gate = "ELSE_CORRELATED_ERROR" if prefix else "CORRELATED_ERROR"
    line = prefix.count("\n") + 1
    with pytest.raises(clifft.ParseError, match=rf"Line {line}:"):
        read_circuit(f"{prefix}{gate}({probability}) X0\nM 0")


@pytest.fixture(
    params=[
        "X 0\nCX 0 1 0 2\nM 0 1 2",
        "X 0\nM 0\nCX rec[-1] 0\nM 0",
        "X 0\nM 0\nH 0\nCZ rec[-1] 0\nMX 0",
        "E(0) X0\nELSE_CORRELATED_ERROR(0) Y0\nM 0",
        "E(0) X0\nELSE_CORRELATED_ERROR(1) Y0\nM 0",
        "E(1) X0\nELSE_CORRELATED_ERROR(0) Y0\nM 0",
        "E(1) X0\nELSE_CORRELATED_ERROR(1) Y0\nM 0",
    ],
    ids=["reuse-qubit", "cx-feedback", "cz-feedback", "p00", "p01", "p10", "p11"],
)
def valid_boundary_circuit(request):
    text = request.param
    expected = stim.Circuit(text).compile_sampler(seed=3).sample(32)
    return text, expected


def test_valid_boundaries_sample_like_stim(valid_boundary_circuit, sampling_mode):
    text, expected = valid_boundary_circuit
    result = sampling_mode.sample(sampling_mode.compile(text), shots=32, seed=3)
    np.testing.assert_array_equal(result.measurements, expected)


@pytest.mark.parametrize("parsed", [False, True], ids=["text", "parsed"])
def test_valid_boundaries_sample_noncomp_like_stim(
    valid_boundary_circuit, parsed, noncomp_sampling_api
):
    text, expected = valid_boundary_circuit
    source = clifft.parse(text) if parsed else text
    result = noncomp_sampling_api(source, noncomp.Model(), shots=32, seed=3)
    np.testing.assert_array_equal(result.measurements, expected)
