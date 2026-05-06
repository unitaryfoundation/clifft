"""Tests for exact computational-basis probability queries."""

import math
from typing import Any

import numpy as np
import pytest
from conftest import random_dense_clifford_t_circuit
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

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


def test_probabilities_supports_high_qubit_string_bitstrings() -> None:
    prog = clifft.compile("H 70")
    zero = "0" * 71
    one_q70_big = ("0" * 70) + "1"
    one_q70_little = "1" + ("0" * 70)

    np.testing.assert_allclose(
        clifft.probabilities(prog, [zero, one_q70_big]),
        [0.5, 0.5],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        clifft.probabilities(prog, [zero, one_q70_little], bit_order="little"),
        [0.5, 0.5],
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8])
def test_probabilities_supports_high_qubit_array_bitstrings(dtype: np.dtype) -> None:
    prog = clifft.compile("X 70")
    bits = np.zeros((2, 71), dtype=dtype)
    bits[1, 70] = 1
    little_bits = np.zeros((2, 71), dtype=dtype)
    little_bits[1, 0] = 1

    np.testing.assert_allclose(clifft.probabilities(prog, bits), [0.0, 1.0])
    np.testing.assert_allclose(
        clifft.probabilities(prog, little_bits, bit_order="little"), [0.0, 1.0]
    )


def test_probabilities_supports_multiword_array_bitstrings() -> None:
    prog = clifft.compile("H 199\nH 199")
    bits = np.zeros((3, 200), dtype=np.uint8)
    bits[1, 0] = 1
    bits[2, 199] = 1
    little_bits = np.zeros((3, 200), dtype=np.uint8)
    little_bits[1, 199] = 1
    little_bits[2, 0] = 1

    np.testing.assert_allclose(clifft.probabilities(prog, bits), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        clifft.probabilities(prog, little_bits, bit_order="little"),
        [1.0, 0.0, 0.0],
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
        invalid_sequence: Any = [[0, 0]]
        clifft.probabilities(prog, invalid_sequence)
    with pytest.raises(ValueError, match="array must be 2D"):
        clifft.probabilities(prog, np.array([0, 1], dtype=np.uint8))
    with pytest.raises(ValueError, match="3 columns, expected 2"):
        clifft.probabilities(prog, np.array([[0, 1, 0]], dtype=np.uint8))
    with pytest.raises(TypeError, match="dtype must be bool or uint8"):
        invalid_dtype: Any = np.array([[0, 1]], dtype=np.int64)
        clifft.probabilities(prog, invalid_dtype)
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
        ("M 0\nCX rec[-1] 1", {}),
    ],
)
def test_probabilities_rejects_non_unitary_programs(circuit: str, kwargs: dict[str, Any]) -> None:
    prog = clifft.compile(circuit, **kwargs)

    with pytest.raises(ValueError, match="requires pure-state evolution"):
        clifft.probabilities(prog, ["0" * prog.num_qubits])


def test_probabilities_allows_exp_val_probes() -> None:
    with_probe = clifft.compile("H 0\nEXP_VAL X0")
    without_probe = clifft.compile("H 0")

    np.testing.assert_allclose(
        clifft.probabilities(with_probe, ["0", "1"]),
        clifft.probabilities(without_probe, ["0", "1"]),
        atol=1e-12,
    )


def test_drop_non_unitary_pass_enables_querying_unitary_skeleton() -> None:
    passes = clifft.HirPassManager()
    passes.add(clifft.DropNonUnitaryPass())
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


@pytest.mark.parametrize("num_qubits,seed", [(2, 101), (3, 202), (4, 303)])
def test_probabilities_match_qiskit_for_random_small_circuits(num_qubits: int, seed: int) -> None:
    circuit = random_dense_clifford_t_circuit(num_qubits, depth=18, seed=seed)
    prog = clifft.compile(circuit)
    qiskit_sv = qiskit_statevector(stim_to_qiskit_noiseless(circuit))
    bitstrings = (
        (
            np.arange(1 << num_qubits, dtype=np.uint64)[:, None]
            >> np.arange(num_qubits, dtype=np.uint64)
        )
        & np.uint64(1)
    ).astype(np.uint8)

    np.testing.assert_allclose(
        clifft.probabilities(prog, bitstrings),
        np.abs(qiskit_sv) ** 2,
        atol=1e-12,
    )


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


def test_probabilities_match_formula_for_continuous_z_rotation() -> None:
    # H R_Z(alpha) H |0> -> cos(pi*alpha/2) |0> - i sin(pi*alpha/2) |1>.
    # Exercises the OP_ARRAY_ROT / PHASE_ROTATION path the random Clifford+T
    # circuits never touch.
    alpha = 0.123
    prog = clifft.compile(f"H 0\nR_Z({alpha}) 0\nH 0")
    half_angle = math.pi * alpha / 2.0
    np.testing.assert_allclose(
        clifft.probabilities(prog, ["0", "1"]),
        [math.cos(half_angle) ** 2, math.sin(half_angle) ** 2],
        atol=1e-12,
    )


def test_probabilities_sum_to_one_at_high_active_rank() -> None:
    # Unitarity invariant on a circuit whose peak_rank exceeds the
    # statevector-expansion limit, so this catches normalization regressions
    # that the small-circuit statevector cross-check cannot.
    prog = clifft.compile("\n".join(f"H {q}\nT {q}" for q in range(12)))
    assert prog.peak_rank > 10
    n = prog.num_qubits
    bitstrings = [format(i, f"0{n}b") for i in range(1 << n)]
    np.testing.assert_allclose(clifft.probabilities(prog, bitstrings).sum(), 1.0, atol=1e-10)


def test_probabilities_handles_empty_and_singleton_inputs() -> None:
    prog = clifft.compile("H 0")

    empty_list = clifft.probabilities(prog, [])
    assert empty_list.shape == (0,)
    assert empty_list.dtype == np.float64

    empty_arr = clifft.probabilities(prog, np.zeros((0, 1), dtype=np.uint8))
    assert empty_arr.shape == (0,)

    single = clifft.probabilities(prog, "0")
    assert single.shape == (1,)
    np.testing.assert_allclose(single, [0.5], atol=1e-12)


def test_probabilities_match_statevector_for_fused_unitaries() -> None:
    # Construct a circuit that triggers both single-axis (OP_ARRAY_U2) and
    # tile (OP_ARRAY_U4) fusion so the supported-opcode listing for those
    # paths is actually exercised end-to-end.
    src = (
        "H 0\nT 0\nS 0\nH 0\nT 0\nS 0\nH 0\nT 0\nS 0\n"
        "H 1\nT 1\nS 1\nH 1\nT 1\nS 1\nH 1\nT 1\nS 1\n"
        "CX 0 1\nH 0\nT 0\nS 0"
    )
    prog = clifft.compile(src)
    opcodes = {instr.opcode for instr in prog}
    assert clifft.Opcode.OP_ARRAY_U2 in opcodes, "test circuit no longer triggers U2 fusion"
    assert clifft.Opcode.OP_ARRAY_U4 in opcodes, "test circuit no longer triggers U4 fusion"

    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    expected = np.abs(clifft.get_statevector(prog, state)) ** 2
    n = prog.num_qubits
    bitstrings = [format(i, f"0{n}b")[::-1] for i in range(1 << n)]
    np.testing.assert_allclose(clifft.probabilities(prog, bitstrings), expected, atol=1e-12)
