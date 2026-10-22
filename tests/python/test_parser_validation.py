"""Validate circuit text at public Python entry points before execution."""

import numpy as np
import pytest
import stim

import clifft
from clifft import noncomp

_PAIR_GATES = [
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
    "MXX",
    "MYY",
    "MZZ",
    "DEPOLARIZE2(0.1)",
    "PAULI_CHANNEL_2(" + ",".join(["0.1"] + ["0"] * 14) + ")",
]

_TRIPLE_NOISE_GATES = [
    "DEPOLARIZE3(0.1)",
    "PAULI_CHANNEL_3(" + ",".join(["0.1"] + ["0"] * 62) + ")",
]

_PAULI_CHANNEL_WIDTHS = [1, 2, 3]

_SINGLE_PROBABILITY_NOISE = [
    ("X_ERROR", "0"),
    ("Y_ERROR", "0"),
    ("Z_ERROR", "0"),
    ("DEPOLARIZE1", "0"),
    ("DEPOLARIZE2", "0 1"),
    ("DEPOLARIZE3", "0 1 2"),
]

_INLINE_MEASUREMENTS = [
    ("M", "0"),
    ("MZ", "0"),
    ("MX", "0"),
    ("MY", "0"),
    ("MR", "0"),
    ("MRZ", "0"),
    ("MRX", "0"),
    ("MRY", "0"),
    ("MPAD", "0"),
    ("MPP", "X0*Z1"),
    ("MXX", "0 1"),
    ("MYY", "0 1"),
    ("MZZ", "0 1"),
]

_CORRELATED_ERROR_CASES = [
    ("", "CORRELATED_ERROR"),
    ("", "E"),
    ("E(0.5) X0\n", "ELSE_CORRELATED_ERROR"),
    ("E(1) X0\n", "ELSE_CORRELATED_ERROR"),
    ("E(0) X0\nELSE_CORRELATED_ERROR(1) Y0\n", "ELSE_CORRELATED_ERROR"),
]


def assert_gate_coverage(expected, cases, exclusions=None):
    # Keep aliases in the regression cases; metadata uses canonical gate names.
    aliases = {"CNOT": "CX", "MZ": "M", "MRZ": "MR", "E": "CORRELATED_ERROR"}
    names = {case.split("(", 1)[0] for case in cases}
    covered = {aliases.get(name, name) for name in names}
    excluded = set(exclusions or {})
    missing = expected - covered - excluded
    stale = (covered | excluded) - expected
    overlap = covered & excluded
    assert not (missing or stale or overlap), (
        f"Missing regression cases or explicit exclusions: {sorted(missing)}; "
        f"stale cases or exclusions: {sorted(stale)}; "
        f"both covered and excluded: {sorted(overlap)}"
    )


@pytest.fixture(scope="module")
def gate_metadata():
    from clifft._clifft_core import _gate_metadata

    return _gate_metadata()


@pytest.mark.parametrize(
    ("arity", "cases", "exclusions"),
    [
        (
            "pair",
            _PAIR_GATES,
            {"CH": "Dedicated parser rewrite; malformed targets covered in test_parser.cc."},
        ),
        (
            "triple",
            _TRIPLE_NOISE_GATES,
            {
                "CCX": "Dedicated parser rewrite; malformed targets covered in test_parser.cc.",
                "CCZ": "Dedicated parser rewrite; malformed targets covered in test_parser.cc.",
            },
        ),
    ],
    ids=["pair", "triple"],
)
def test_physical_operand_regression_coverage(gate_metadata, arity, cases, exclusions):
    expected = {
        name
        for name, traits in gate_metadata.items()
        if traits["arity"] == arity
        and (traits["unitary"] or traits["noise"] or traits["physical_measurement"])
        and not traits["identity_noop"]
    }
    assert_gate_coverage(expected, cases, exclusions)


def test_inline_measurement_regression_coverage(gate_metadata):
    # MPAD accepts readout noise on classical bits despite not measuring a qubit.
    expected = {name for name, traits in gate_metadata.items() if traits["physical_measurement"]}
    assert_gate_coverage(expected | {"MPAD"}, [gate for gate, _ in _INLINE_MEASUREMENTS])


def test_pauli_channel_regression_coverage(gate_metadata):
    expected = {name for name in gate_metadata if name.startswith("PAULI_CHANNEL_")}
    assert_gate_coverage(expected, [f"PAULI_CHANNEL_{width}" for width in _PAULI_CHANNEL_WIDTHS])


def test_single_probability_noise_regression_coverage(gate_metadata):
    expected = {
        name
        for name, traits in gate_metadata.items()
        if traits["noise"] and traits["arity"] != "multi" and not name.startswith("PAULI_CHANNEL_")
    }
    assert_gate_coverage(
        expected,
        [gate for gate, _ in _SINGLE_PROBABILITY_NOISE],
        {"READOUT_NOISE": "Accepts one or two probabilities; validated separately."},
    )


def test_correlated_error_regression_coverage(gate_metadata):
    expected = {
        name
        for name, traits in gate_metadata.items()
        if traits["noise"] and traits["arity"] == "multi"
    }
    assert_gate_coverage(expected, [gate for _, gate in _CORRELATED_ERROR_CASES])


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


@pytest.mark.parametrize("gate", _PAIR_GATES)
@pytest.mark.parametrize("targets", ["0 0", "0 1 2 2"])
def test_duplicate_physical_pair_rejected(gate, targets, read_circuit):
    with pytest.raises(clifft.ParseError, match="Line 2: .*requires distinct qubit targets"):
        read_circuit(f"H 0\n{gate} {targets}\nM 0")


def test_duplicate_pair_inside_repeat_rejected(read_circuit):
    with pytest.raises(clifft.ParseError, match="Line 3: .*requires distinct qubit targets"):
        read_circuit("H 0\nREPEAT 2 {\nCX 0 0\n}\nM 0")


@pytest.mark.parametrize("gate", _TRIPLE_NOISE_GATES)
@pytest.mark.parametrize("targets", ["0 0 1", "0 1 0", "0 1 1", "0 1 2 3 4 3"])
def test_duplicate_noise_triple_rejected(gate, targets, read_circuit):
    with pytest.raises(clifft.ParseError, match="Line 2: .*requires distinct qubit targets"):
        read_circuit(f"H 0\n{gate} {targets}\nM 0")


@pytest.mark.parametrize("width", _PAULI_CHANNEL_WIDTHS)
@pytest.mark.parametrize("position", [0, -1])
@pytest.mark.parametrize("probability", ["-0.1", "1.1"])
def test_pauli_channel_checks_every_probability(width, position, probability, read_circuit):
    probabilities = ["0"] * (4**width - 1)
    probabilities[position] = probability
    targets = " ".join(map(str, range(width)))
    with pytest.raises(clifft.ParseError, match="Line 2: .*probabilities must be finite"):
        read_circuit(f"H 0\nPAULI_CHANNEL_{width}({','.join(probabilities)}) {targets}\nM 0")


@pytest.mark.parametrize("width", _PAULI_CHANNEL_WIDTHS)
def test_pauli_channel_rejects_excess_total_probability(width, read_circuit):
    probabilities = ["0"] * (4**width - 1)
    probabilities[0], probabilities[-1] = "0.5", "0.75"
    targets = " ".join(map(str, range(width)))
    with pytest.raises(clifft.ParseError, match="Line 2: .*probabilities must sum to at most 1"):
        read_circuit(f"H 0\nPAULI_CHANNEL_{width}({','.join(probabilities)}) {targets}\nM 0")


@pytest.mark.parametrize(("gate", "targets"), _SINGLE_PROBABILITY_NOISE)
@pytest.mark.parametrize("arguments", ["", "()", "(0.1,0.2)"])
def test_single_probability_noise_requires_one_argument(gate, targets, arguments, read_circuit):
    with pytest.raises(clifft.ParseError, match=f"Line 2: {gate} requires exactly 1 probability"):
        read_circuit(f"H 0\n{gate}{arguments} {targets}\nM 0")


@pytest.mark.parametrize(("gate", "targets"), _SINGLE_PROBABILITY_NOISE)
@pytest.mark.parametrize("probability", ["-0.1", "1.5"])
def test_single_probability_noise_rejects_invalid_probability(
    gate, targets, probability, read_circuit
):
    with pytest.raises(clifft.ParseError, match=f"Line 2: {gate} probability must be finite"):
        read_circuit(f"H 0\n{gate}({probability}) {targets}\nM 0")


@pytest.mark.parametrize(("gate", "targets"), _SINGLE_PROBABILITY_NOISE)
@pytest.mark.parametrize("probability", ["nan", "inf", "-inf", "1e999"])
def test_single_probability_noise_rejects_nonfinite_probability(
    gate, targets, probability, read_circuit
):
    with pytest.raises(clifft.ParseError, match="Line 2:"):
        read_circuit(f"H 0\n{gate}({probability}) {targets}\nM 0")


@pytest.mark.parametrize(("gate", "targets"), _SINGLE_PROBABILITY_NOISE)
@pytest.mark.parametrize("probability", [0, 1])
def test_single_probability_noise_accepts_endpoints(gate, targets, probability, read_circuit):
    text = f"H 0\n{gate}({probability}) {targets}\nM 0"
    if gate != "DEPOLARIZE3":  # Stim has no three-qubit depolarizing instruction.
        stim.Circuit(text)
    read_circuit(text)


@pytest.mark.parametrize(("gate", "targets"), _INLINE_MEASUREMENTS)
@pytest.mark.parametrize("arguments", ["-0.1", "1.1", "0.1,0.2"])
def test_inline_measurement_noise_rejects_invalid_arguments(gate, targets, arguments, read_circuit):
    with pytest.raises(clifft.ParseError, match="Line 2: .*probability"):
        read_circuit(f"H 0\n{gate}({arguments}) {targets}")


@pytest.mark.parametrize("probability", ["-0.1", "1.1"])
@pytest.mark.parametrize(("prefix", "gate"), _CORRELATED_ERROR_CASES)
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
        "M(0) 0\nM(1) 0\nMR(1) 0\nM 0",
        "H 0\nMX(0) 0\nMX(1) 0\nMRX(1) 0\nMX 0",
        "H 0\nS 0\nMY(0) 0\nMY(1) 0\nMRY(1) 0\nMY 0",
        "MPAD(0) 0 1\nMPAD(1) 0 1",
        "MPP(0) Z0*Z1\nMPP(1) Z0*Z1",
        "H 0 1 2\nMXX(0) 0 1 0 2\nMXX(1) 0 1 0 2",
        "H 0 1\nS 0 1\nMYY(0) 0 1\nMYY(1) 0 1",
        "MZZ(0) 0 1\nMZZ(1) 0 1",
        "PAULI_CHANNEL_1(0,0,0) 0\nM 0\nPAULI_CHANNEL_1(0.25,0.75,0) 0\nM 0",
        "PAULI_CHANNEL_2(" + ",".join(["0"] * 15) + ") 0 1\nM 0 1",
        "PAULI_CHANNEL_2(" + ",".join(["0"] * 4 + ["1"] + ["0"] * 10) + ") 0 1\nM 0 1",
        "DEPOLARIZE2(0) 0 1 0 2\nM 0 1 2",
    ],
    ids=[
        "reuse-qubit",
        "cx-feedback",
        "cz-feedback",
        "p00",
        "p01",
        "p10",
        "p11",
        "mz-noise",
        "mx-noise",
        "my-noise",
        "mpad-noise",
        "mpp-noise",
        "mxx-noise",
        "myy-noise",
        "mzz-noise",
        "pauli1",
        "pauli2-zero",
        "pauli2-certain",
        "noise-reuse",
    ],
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


@pytest.fixture(params=[0, 1])
def three_qubit_channel_boundary(request):
    probabilities = ["0"] * 63
    probabilities[20] = str(request.param)  # XXX in lexicographic I/X/Y/Z order.
    text = f"PAULI_CHANNEL_3({','.join(probabilities)}) 0 1 2 0 3 4\nM 0 1 2 3 4"
    expected = [0, request.param, request.param, request.param, request.param]
    return text, expected


def test_three_qubit_channel_boundaries(three_qubit_channel_boundary, sampling_mode):
    text, expected = three_qubit_channel_boundary
    result = sampling_mode.sample(sampling_mode.compile(text), shots=32, seed=3)
    np.testing.assert_array_equal(result.measurements, np.tile(expected, (32, 1)))


def test_three_qubit_channel_boundaries_noncomp(three_qubit_channel_boundary, noncomp_sampling_api):
    text, expected = three_qubit_channel_boundary
    result = noncomp_sampling_api(text, noncomp.Model(), shots=32, seed=3)
    np.testing.assert_array_equal(result.measurements, np.tile(expected, (32, 1)))
