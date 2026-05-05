"""Tests for exact computational-basis probability queries."""

import numpy as np
import pytest

import clifft


def test_program_num_qubits_property() -> None:
    prog = clifft.compile("H 0\nCX 0 2")

    assert prog.num_qubits == 3


def test_bell_state_probabilities() -> None:
    prog = clifft.compile("H 0\nCX 0 1")

    probs = clifft.probabilities(prog, ["00", "01", "10", "11"])

    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)
    assert probs.dtype == np.float64
    assert probs.shape == (4,)


def test_bit_order_big_maps_first_position_to_qubit_zero() -> None:
    prog_x0 = clifft.compile("X 0\nH 1\nH 1")
    prog_x1 = clifft.compile("X 1")

    np.testing.assert_allclose(clifft.probabilities(prog_x0, ["10", "01"]), [1.0, 0.0])
    np.testing.assert_allclose(clifft.probabilities(prog_x1, ["10", "01"]), [0.0, 1.0])


def test_bit_order_little_maps_last_position_to_qubit_zero() -> None:
    prog_x0 = clifft.compile("X 0\nH 1\nH 1")
    prog_x1 = clifft.compile("X 1")

    np.testing.assert_allclose(
        clifft.probabilities(prog_x0, ["01", "10"], bit_order="little"), [1.0, 0.0]
    )
    np.testing.assert_allclose(
        clifft.probabilities(prog_x1, ["01", "10"], bit_order="little"), [0.0, 1.0]
    )


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8])
def test_array_input_matches_string_input(dtype: np.dtype) -> None:
    prog = clifft.compile("X 0\nH 1\nH 1")
    bits = np.array([[1, 0], [0, 1]], dtype=dtype)

    np.testing.assert_allclose(
        clifft.probabilities(prog, bits),
        clifft.probabilities(prog, ["10", "01"]),
    )
    np.testing.assert_allclose(
        clifft.probabilities(prog, bits, bit_order="little"),
        clifft.probabilities(prog, ["10", "01"], bit_order="little"),
    )


def test_probability_input_validation() -> None:
    prog = clifft.compile("H 0\nCX 0 1")

    with pytest.raises(ValueError, match="length 1, expected 2"):
        clifft.probabilities(prog, ["0"])
    with pytest.raises(ValueError, match="expected only '0' and '1'"):
        clifft.probabilities(prog, ["0x"])
    with pytest.raises(ValueError, match="bit_order"):
        clifft.probabilities(prog, ["00"], bit_order="middle")
    with pytest.raises(TypeError, match="strings or a 2D"):
        clifft.probabilities(prog, [[0, 0]])
    with pytest.raises(ValueError, match="array must be 2D"):
        clifft.probabilities(prog, np.array([0, 1], dtype=np.uint8))
    with pytest.raises(ValueError, match="3 columns, expected 2"):
        clifft.probabilities(prog, np.array([[0, 1, 0]], dtype=np.uint8))
    with pytest.raises(TypeError, match="dtype must be bool or uint8"):
        clifft.probabilities(prog, np.array([[0, 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="contain only 0 and 1"):
        clifft.probabilities(prog, np.array([[0, 2]], dtype=np.uint8))


@pytest.mark.parametrize(
    "circuit,kwargs",
    [
        ("M 0", {}),
        ("M(0.1) 0", {}),
        ("X_ERROR(0.1) 0", {}),
        ("M 0\nDETECTOR rec[-1]", {}),
        ("M 0\nDETECTOR rec[-1]", {"postselection_mask": [1]}),
        ("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]", {}),
        ("EXP_VAL Z0", {}),
        ("M 0\nCX rec[-1] 1", {}),
    ],
)
def test_probabilities_rejects_non_unitary_programs(circuit: str, kwargs: dict[str, object]) -> None:
    prog = clifft.compile(circuit, **kwargs)

    with pytest.raises(ValueError, match="requires a unitary program"):
        clifft.probabilities(prog, ["0" * prog.num_qubits])


def test_make_unitary_pass_enables_querying_unitary_skeleton() -> None:
    passes = clifft.HirPassManager()
    passes.add(clifft.MakeUnitaryPass())
    prog = clifft.compile(
        """
        H 0
        M 0
        X_ERROR(0.25) 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        EXP_VAL X0
        """,
        hir_passes=passes,
    )

    assert prog.num_measurements == 0
    assert prog.num_detectors == 0
    assert prog.num_observables == 0
    assert prog.num_exp_vals == 0
    np.testing.assert_allclose(clifft.probabilities(prog, ["0", "1"]), [0.5, 0.5], atol=1e-12)


def test_probabilities_match_dense_statevector_for_small_circuit() -> None:
    circuit = """
    H 0
    CX 0 1
    T 1
    H 2
    CX 2 0
    """
    prog = clifft.compile(circuit)

    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    expected = np.abs(clifft.get_statevector(prog, state)) ** 2
    bitstrings = [format(i, f"0{prog.num_qubits}b")[::-1] for i in range(1 << prog.num_qubits)]

    np.testing.assert_allclose(clifft.probabilities(prog, bitstrings), expected, atol=1e-12)


def test_probabilities_supports_active_rank_beyond_dense_statevector_limit() -> None:
    circuit = "\n".join(f"H {q}\nT {q}" for q in range(12))
    prog = clifft.compile(circuit)

    assert prog.num_qubits == 12
    assert prog.peak_rank > 10
    with pytest.raises(RuntimeError, match="Statevector expansion limited"):
        state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        clifft.execute(prog, state)
        clifft.get_statevector(prog, state)

    np.testing.assert_allclose(
        clifft.probabilities(prog, ["0" * 12, "1" * 12]),
        [2**-12, 2**-12],
        atol=1e-15,
    )
