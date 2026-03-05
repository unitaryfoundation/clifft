"""Independent Qiskit-Aer validation of UCC statevector correctness.

Proves that UCC's Clifford+T amplitude interference matches Qiskit-Aer's
statevector simulator exactly (fidelity > 0.9999). This provides a third-party
oracle independent of both Stim and UCC's own numpy oracle.
"""

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equal,
    random_clifford_t_circuit,
    random_dense_clifford_t_circuit,
)
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

import ucc


def _ucc_statevector(circuit_str: str) -> np.ndarray:
    """Compile and execute circuit in UCC, return dense statevector."""
    prog = ucc.compile(circuit_str)
    state = ucc.State(prog.peak_rank, prog.num_measurements)
    ucc.execute(prog, state)
    sv: np.ndarray = ucc.get_statevector(prog, state)
    return sv


class TestQiskitStatevectorOracle:
    """Validate UCC statevectors against Qiskit-Aer."""

    def test_single_h(self) -> None:
        """H|0> = |+> matches Qiskit."""
        circuit = "H 0"
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)

    def test_single_t(self) -> None:
        """H-T circuit matches Qiskit."""
        circuit = "H 0\nT 0"
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)

    def test_t_dagger(self) -> None:
        """H-T_DAG circuit matches Qiskit."""
        circuit = "H 0\nT_DAG 0"
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)

    def test_bell_plus_t(self) -> None:
        """Bell state + T gate matches Qiskit."""
        circuit = "H 0\nCX 0 1\nT 0"
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)

    def test_two_t_equals_s(self) -> None:
        """T*T = S identity matches Qiskit."""
        circuit = "H 0\nT 0\nT 0"
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)

    def test_four_t_equals_z(self) -> None:
        """T^4 = Z identity matches Qiskit."""
        circuit = "H 0\nT 0\nT 0\nT 0\nT 0"
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)

    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
    @pytest.mark.parametrize("seed", range(5))
    def test_random_clifford_t_small(self, num_qubits: int, seed: int) -> None:
        """Random Clifford+T circuits up to 5 qubits match Qiskit."""
        circuit = random_clifford_t_circuit(num_qubits, depth=15, seed=seed)
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv, msg=f"{num_qubits}q seed={seed}\n{circuit}")

    @pytest.mark.parametrize("seed", range(3))
    def test_random_clifford_t_medium(self, seed: int) -> None:
        """Random Clifford+T circuits at 6 qubits match Qiskit."""
        for num_qubits in [6]:
            circuit = random_clifford_t_circuit(num_qubits, depth=20, seed=seed)
            ucc_sv = _ucc_statevector(circuit)
            qc = stim_to_qiskit_noiseless(circuit)
            qiskit_sv = qiskit_statevector(qc)
            assert_statevectors_equal(
                ucc_sv, qiskit_sv, msg=f"{num_qubits}q seed={seed}\n{circuit}"
            )

    def test_multi_qubit_entangled_t(self) -> None:
        """Entangled circuit with T gates on multiple qubits matches Qiskit."""
        circuit = "\n".join(
            [
                "H 0",
                "H 1",
                "H 2",
                "CX 0 1",
                "CX 1 2",
                "T 0",
                "T 1",
                "T_DAG 2",
                "CX 2 0",
                "T 0",
            ]
        )
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)

    def test_deep_t_circuit(self) -> None:
        """Deep circuit with many T gates tests accumulation accuracy."""
        lines = ["H 0", "H 1"]
        for i in range(8):
            lines.append(f"T {i % 2}")
            lines.append("CX 0 1")
        circuit = "\n".join(lines)
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv)


class TestDenseCliffordTFuzzer:
    """Validate UCC against Qiskit-Aer with dense entanglement circuits.

    Uses higher 2-qubit gate probability (50%) and all three 2-qubit gates
    (CX, CY, CZ) to stress the compiler's Pauli compression pipeline.
    """

    @pytest.mark.parametrize("num_qubits", [3, 4, 5])
    @pytest.mark.parametrize("seed", range(5))
    def test_dense_entanglement_small(self, num_qubits: int, seed: int) -> None:
        """Dense Clifford+T circuits at 3-5 qubits match Qiskit."""
        circuit = random_dense_clifford_t_circuit(num_qubits, depth=30, seed=seed)
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(
            ucc_sv, qiskit_sv, msg=f"dense {num_qubits}q seed={seed}\n{circuit}"
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_dense_entanglement_medium(self, seed: int) -> None:
        """Dense Clifford+T at 6 qubits match Qiskit."""
        circuit = random_dense_clifford_t_circuit(6, depth=40, seed=seed)
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv, msg=f"dense 6q seed={seed}")

    @pytest.mark.parametrize("seed", range(3))
    def test_deep_phase_accumulation(self, seed: int) -> None:
        """Deep circuits (depth=100) stress T-gate phase arithmetic."""
        circuit = random_dense_clifford_t_circuit(4, depth=100, seed=seed, two_qubit_prob=0.3)
        ucc_sv = _ucc_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(ucc_sv, qiskit_sv, msg=f"deep 4q seed={seed}")
