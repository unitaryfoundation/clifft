"""Keep the permissive Qiskit oracle from silently dropping supported gates."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from conftest import assert_statevectors_equiv
from utils_qiskit import _SKIP_GATES, qiskit_statevector, stim_to_qiskit

import clifft


@pytest.mark.parametrize(
    "source",
    [
        "H 0",
        "S 0",
        "S_DAG 0",
        "T 0",
        "T_DAG 0",
        "X 0",
        "Y 0",
        "Z 0",
        "CX 0 1",
        "CY 0 1",
        "CZ 0 1",
        "CH 0 1",
        "CCX 0 1 2",
        "CCZ 0 1 2",
        "R_X(0.23) 0",
        "R_Y(0.23) 0",
        "R_Z(0.23) 0",
        "U3(0.23, 0.37, 0.41) 0",
        "R_XX(0.23) 0 1",
        "R_YY(0.23) 0 1",
        "R_ZZ(0.23) 0 1",
        "R_PAULI(0.23) X0*Y1*Z2",
    ],
    ids=lambda source: source.split("(")[0].split()[0],
)
def test_supported_unitaries_are_not_skipped(
    source: str,
    statevector_from_circuit: Callable[[str], npt.NDArray[np.complex128]],
) -> None:
    gate = source.split("(")[0].split()[0]
    assert gate not in _SKIP_GATES, f"Supported gate {gate} is in the oracle skip list"
    converted = stim_to_qiskit(source)
    assert converted.data, f"Supported gate {gate} disappeared during Qiskit conversion"

    # Phase gates and controlled gates can look like identity on the all-zero state.
    preparation = "\n".join(
        f"R_Y(0.173) {qubit}\nR_Z(0.319) {qubit}" for qubit in range(converted.num_qubits)
    )
    circuit = f"{preparation}\n{source}"
    assert_statevectors_equiv(
        statevector_from_circuit(circuit),
        qiskit_statevector(stim_to_qiskit(circuit)),
        msg=f"Qiskit conversion of {gate}",
    )


@pytest.mark.parametrize(
    ("source", "operations"),
    [
        pytest.param("M 0", ("measure",), id="M"),
        pytest.param("MX 0", ("h", "measure", "h"), id="MX"),
        pytest.param("MR 0", ("measure", "reset"), id="MR"),
        pytest.param("R 0", ("reset",), id="R"),
        pytest.param("RX 0", ("reset", "h"), id="RX"),
        pytest.param("RY 0", ("reset", "h", "s"), id="RY"),
    ],
)
def test_supported_measurements_and_resets_are_not_skipped(
    source: str, operations: tuple[str, ...]
) -> None:
    gate = source.split()[0]
    assert gate not in _SKIP_GATES, f"Supported gate {gate} is in the oracle skip list"
    converted = stim_to_qiskit(source)
    assert tuple(instruction.operation.name for instruction in converted.data) == operations
    clifft.compile(source)


@pytest.mark.parametrize("ignored", ["TICK", "X_ERROR(0.25) 0", "DETECTOR rec[-1]"])
def test_annotations_and_noise_remain_skipped(ignored: str) -> None:
    converted = stim_to_qiskit(f"H 0\n{ignored}\nM 0")
    assert tuple(instruction.operation.name for instruction in converted.data) == ("h", "measure")
