"""End-to-end Clifford gate validation against test-only Stim."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import stim
from conftest import assert_statevectors_equiv

# These inventories match the fixed nonidentity Cliffords in GateType: 23 single-qubit
# gates and 21 two-qubit gates. I and II are parser-discarded no-ops; variable-width
# SPP and SPP_DAG operations are covered separately below.
SINGLE_QUBIT_CLIFFORDS = [
    "H",
    "S",
    "S_DAG",
    "X",
    "Y",
    "Z",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "H_XY",
    "H_YZ",
    "H_NXY",
    "H_NXZ",
    "H_NYZ",
    "C_XYZ",
    "C_ZYX",
    "C_NXYZ",
    "C_NZYX",
    "C_XNYZ",
    "C_XYNZ",
    "C_ZNYX",
    "C_ZYNX",
]

TWO_QUBIT_CLIFFORDS = [
    "CX",
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
]

# The projectors onto these four states span the one-qubit operator space. Their tensor
# products therefore distinguish every two-qubit channel, including relative phases.
SINGLE_QUBIT_PREPARATIONS = ["I {q}", "X {q}", "H {q}", "H {q}\nS {q}"]


def _stim_statevector(circuit: str) -> npt.NDArray[np.complex128]:
    tableau = stim.Tableau.from_circuit(stim.Circuit(circuit))
    return np.asarray(tableau.to_state_vector(endian="little"), dtype=np.complex128)


@pytest.mark.parametrize("gate", SINGLE_QUBIT_CLIFFORDS)
@pytest.mark.parametrize("preparation", SINGLE_QUBIT_PREPARATIONS)
def test_named_single_qubit_clifford_matches_stim(
    gate: str,
    preparation: str,
    statevector_from_circuit: Callable[[str], npt.NDArray[np.complex128]],
) -> None:
    circuit = f"{preparation.format(q=0)}\n{gate} 0"
    assert_statevectors_equiv(
        statevector_from_circuit(circuit),
        _stim_statevector(circuit),
        atol=1e-6,
        msg=circuit,
    )


@pytest.mark.parametrize("gate", TWO_QUBIT_CLIFFORDS)
@pytest.mark.parametrize("first_preparation", SINGLE_QUBIT_PREPARATIONS)
@pytest.mark.parametrize("second_preparation", SINGLE_QUBIT_PREPARATIONS)
def test_named_two_qubit_clifford_matches_stim(
    gate: str,
    first_preparation: str,
    second_preparation: str,
    statevector_from_circuit: Callable[[str], npt.NDArray[np.complex128]],
) -> None:
    circuit = "\n".join(
        [
            first_preparation.format(q=0),
            second_preparation.format(q=1),
            f"{gate} 0 1",
        ]
    )
    assert_statevectors_equiv(
        statevector_from_circuit(circuit),
        _stim_statevector(circuit),
        atol=1e-6,
        msg=circuit,
    )


@pytest.mark.parametrize(
    "operation",
    [
        "SPP X0",
        "SPP_DAG Y0",
        "SPP !Z0",
        "SPP X0*X1",
        "SPP_DAG Y0*Z1",
        "SPP !X0*Y1*Z2",
        "SPP_DAG X0*!Y1*Z2",
    ],
)
def test_pauli_product_clifford_matches_stim(
    operation: str,
    statevector_from_circuit: Callable[[str], npt.NDArray[np.complex128]],
) -> None:
    circuit = f"H 0\nH 1\nH 2\nS 1\n{operation}"
    assert_statevectors_equiv(
        statevector_from_circuit(circuit),
        _stim_statevector(circuit),
        atol=1e-6,
        msg=circuit,
    )
