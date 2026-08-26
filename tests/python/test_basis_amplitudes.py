"""Exact phase-aware computational-basis amplitude queries."""

import random

import numpy as np
import pytest
import stim
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import clifft


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


def output_bits(basis: int, num_qubits: int) -> str:
    """Encode a little-endian integer using the public qubit-first convention."""
    return "".join(str((basis >> qubit) & 1) for qubit in range(num_qubits))


def exact_statevector(circuit_text: str, num_qubits: int) -> np.ndarray:
    """Evaluate every output row through independently compiled target queries."""
    return np.asarray(
        [
            clifft.evaluate_amplitude(
                clifft.compile_basis_amplitude(circuit_text, [output_bits(basis, num_qubits)])
            )
            for basis in range(1 << num_qubits)
        ],
        dtype=np.complex128,
    )


@pytest.mark.parametrize(
    ("gate", "num_qubits"),
    [(gate, 1) for gate in SINGLE_QUBIT_CLIFFORDS]
    + [(gate, 2) for gate in TWO_QUBIT_CLIFFORDS],
)
def test_named_clifford_matrices_match_stim(gate: str, num_qubits: int) -> None:
    """Every column retains Stim's canonical Clifford global phase."""
    expected = np.asarray(stim.gate_data(gate).unitary_matrix, dtype=np.complex128)
    for input_basis in range(1 << num_qubits):
        preparation = [
            f"X {qubit}" for qubit in range(num_qubits) if (input_basis >> qubit) & 1
        ]
        targets = " ".join(str(qubit) for qubit in range(num_qubits))
        actual = exact_statevector("\n".join([*preparation, f"{gate} {targets}"]), num_qubits)
        np.testing.assert_allclose(actual, expected[:, input_basis], rtol=0.0, atol=1e-7)


def test_random_clifford_t_amplitudes_match_qiskit_aer_componentwise() -> None:
    """Random complex amplitudes match an independent strong simulator without phase alignment."""
    rng = random.Random(0x408)
    simulator = AerSimulator(method="statevector")
    for _ in range(30):
        num_qubits = rng.randrange(1, 5)
        reference = QuantumCircuit(num_qubits)
        source: list[str] = []
        for _ in range(rng.randrange(5, 25)):
            gate = rng.choice(["h", "s", "t", "tdg", "x", "y", "z", "cx", "cz"])
            if gate in {"cx", "cz"} and num_qubits > 1:
                q1, q2 = rng.sample(range(num_qubits), 2)
                source.append(f"{gate.upper()} {q1} {q2}")
                getattr(reference, gate)(q1, q2)
            else:
                if gate in {"cx", "cz"}:
                    gate = rng.choice(["h", "s", "t", "tdg", "x", "y", "z"])
                qubit = rng.randrange(num_qubits)
                source_gate = "T_DAG" if gate == "tdg" else gate.upper()
                source.append(f"{source_gate} {qubit}")
                getattr(reference, gate)(qubit)

        reference.save_statevector()
        expected = np.asarray(simulator.run(reference).result().get_statevector())
        actual = exact_statevector("\n".join(source), num_qubits)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)


def test_qasm2_query_restores_source_global_phase() -> None:
    """The QASM phase sidecar is consumed only by the phase-aware compiler."""
    source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        t q[1];
        cx q[0],q[1];
        u3(0.31,-0.47,0.59) q[0];
    """
    reference = QuantumCircuit(2)
    reference.h(0)
    reference.t(1)
    reference.cx(0, 1)
    reference.u(0.31, -0.47, 0.59, 0)
    reference.save_statevector()
    expected = np.asarray(AerSimulator(method="statevector").run(reference).result().get_statevector())
    actual = np.asarray(
        [
            clifft.evaluate_amplitude(
                clifft.compile_basis_amplitude(
                    source,
                    [output_bits(basis, 2)],
                    input_format="qasm2",
                )
            )
            for basis in range(4)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_output_effect_can_reduce_reported_peak_width() -> None:
    """The target basis state can make adjoint rotations deterministic."""
    query = clifft.compile_basis_amplitude("H 1\nH 0\nT 1\nT 0", ["10"])
    assert query.peak_active_width == 0
    assert clifft.evaluate_amplitude(query) == pytest.approx(0.5 * np.exp(1j * np.pi / 4))


def test_amplitude_magnitude_matches_existing_probability_query() -> None:
    """The new scalar agrees with the existing phase-insensitive API."""
    source = "H 0\nCX 0 1\nT 0\nH 1\nT_DAG 1"
    program = clifft.compile(source, hir_passes=None)
    bitstrings = [output_bits(basis, 2) for basis in range(4)]
    probabilities = clifft.basis_probabilities(program, bitstrings)
    amplitudes = np.asarray(
        [
            clifft.evaluate_amplitude(clifft.compile_basis_amplitude(source, [bitstring]))
            for bitstring in bitstrings
        ]
    )
    np.testing.assert_allclose(np.abs(amplitudes) ** 2, probabilities, rtol=0.0, atol=1e-12)


def test_amplitude_query_validates_target_and_unitarity() -> None:
    """The initial API accepts one complete target for a pure unitary only."""
    with pytest.raises(ValueError, match="exactly one"):
        clifft.compile_basis_amplitude("H 0", ["0", "1"])
    with pytest.raises(ValueError, match="length"):
        clifft.compile_basis_amplitude("H 0", ["00"])
    with pytest.raises(ValueError, match="pure-unitary"):
        clifft.compile_basis_amplitude("M 0", ["0"])
