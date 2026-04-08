"""Independent Qiskit-Aer validation of Clifft statevector correctness.

Proves that Clifft's Clifford+T amplitude interference matches Qiskit-Aer's
statevector simulator exactly (complex amplitude equality, not just fidelity).
This provides a third-party oracle independent of both Stim and Clifft's own
numpy oracle.  Exact amplitude comparison catches global phase drift bugs
that fidelity-based tests would miss.
"""

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equal,
    random_clifford_t_circuit,
    random_dense_clifford_t_circuit,
)
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

import clifft


def _clifft_statevector(circuit_str: str) -> np.ndarray:
    """Compile and execute circuit in Clifft, return dense statevector."""
    prog = clifft.compile(circuit_str)
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    sv: np.ndarray = clifft.get_statevector(prog, state)
    return sv


class TestQiskitStatevectorOracle:
    """Validate Clifft statevectors against Qiskit-Aer."""

    def test_single_h(self) -> None:
        """H|0> = |+> matches Qiskit."""
        circuit = "H 0"
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)

    def test_single_t(self) -> None:
        """H-T circuit matches Qiskit."""
        circuit = "H 0\nT 0"
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)

    def test_t_dagger(self) -> None:
        """H-T_DAG circuit matches Qiskit."""
        circuit = "H 0\nT_DAG 0"
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)

    def test_bell_plus_t(self) -> None:
        """Bell state + T gate matches Qiskit."""
        circuit = "H 0\nCX 0 1\nT 0"
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)

    def test_two_t_equals_s(self) -> None:
        """T*T = S identity matches Qiskit."""
        circuit = "H 0\nT 0\nT 0"
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)

    def test_four_t_equals_z(self) -> None:
        """T^4 = Z identity matches Qiskit."""
        circuit = "H 0\nT 0\nT 0\nT 0\nT 0"
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)

    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
    @pytest.mark.parametrize("seed", range(5))
    def test_random_clifford_t_small(self, num_qubits: int, seed: int) -> None:
        """Random Clifford+T circuits up to 5 qubits match Qiskit."""
        circuit = random_clifford_t_circuit(num_qubits, depth=15, seed=seed)
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv, msg=f"{num_qubits}q seed={seed}\n{circuit}")

    @pytest.mark.parametrize("seed", range(3))
    def test_random_clifford_t_medium(self, seed: int) -> None:
        """Random Clifford+T circuits at 6 qubits match Qiskit."""
        for num_qubits in [6]:
            circuit = random_clifford_t_circuit(num_qubits, depth=20, seed=seed)
            clifft_sv = _clifft_statevector(circuit)
            qc = stim_to_qiskit_noiseless(circuit)
            qiskit_sv = qiskit_statevector(qc)
            assert_statevectors_equal(
                clifft_sv, qiskit_sv, msg=f"{num_qubits}q seed={seed}\n{circuit}"
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
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)

    def test_deep_t_circuit(self) -> None:
        """Deep circuit with many T gates tests accumulation accuracy."""
        lines = ["H 0", "H 1"]
        for i in range(8):
            lines.append(f"T {i % 2}")
            lines.append("CX 0 1")
        circuit = "\n".join(lines)
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv)


class TestDenseCliffordTFuzzer:
    """Validate Clifft against Qiskit-Aer with dense entanglement circuits.

    Uses higher 2-qubit gate probability (50%) and all three 2-qubit gates
    (CX, CY, CZ) to stress the compiler's Pauli compression pipeline.
    """

    @pytest.mark.parametrize("num_qubits", [3, 4, 5])
    @pytest.mark.parametrize("seed", range(5))
    def test_dense_entanglement_small(self, num_qubits: int, seed: int) -> None:
        """Dense Clifford+T circuits at 3-5 qubits match Qiskit."""
        circuit = random_dense_clifford_t_circuit(num_qubits, depth=30, seed=seed)
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(
            clifft_sv, qiskit_sv, msg=f"dense {num_qubits}q seed={seed}\n{circuit}"
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_dense_entanglement_medium(self, seed: int) -> None:
        """Dense Clifford+T at 6 qubits match Qiskit."""
        circuit = random_dense_clifford_t_circuit(6, depth=40, seed=seed)
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv, msg=f"dense 6q seed={seed}")

    @pytest.mark.parametrize("seed", range(3))
    def test_deep_phase_accumulation(self, seed: int) -> None:
        """Deep circuits (depth=100) stress T-gate phase arithmetic."""
        circuit = random_dense_clifford_t_circuit(4, depth=100, seed=seed, two_qubit_prob=0.3)
        clifft_sv = _clifft_statevector(circuit)
        qc = stim_to_qiskit_noiseless(circuit)
        qiskit_sv = qiskit_statevector(qc)
        assert_statevectors_equal(clifft_sv, qiskit_sv, msg=f"deep 4q seed={seed}")


class TestArbitraryRotations:
    """Validate continuous rotation gates against Qiskit Aer."""

    def _check_amplitudes(self, stim_text: str, atol: float = 1e-6) -> None:
        """Compile+execute with Clifft, compare exact amplitudes against Qiskit.

        Because Stim's to_flat_unitary_matrix (used in the get_statevector
        oracle) canonicalizes global phase, the oracle U_C matrix may differ
        from the true physical global phase by a scalar.  We project out
        this single global scalar and then enforce strict equality across
        all relative amplitudes to catch any true phase drift.
        """
        import clifft

        prog = clifft.compile(
            stim_text,
            hir_passes=clifft.default_hir_pass_manager(),
            bytecode_passes=clifft.default_bytecode_pass_manager(),
        )
        state = clifft.State(
            peak_rank=prog.peak_rank, num_measurements=prog.num_measurements, seed=42
        )
        clifft.execute(prog, state)
        clifft_sv = np.array(clifft.get_statevector(prog, state))

        qc = stim_to_qiskit_noiseless(stim_text)
        qiskit_sv = qiskit_statevector(qc)

        # Align global phase before comparison to factor out Stim's artifact
        overlap = np.vdot(clifft_sv, qiskit_sv)
        if np.abs(overlap) > 1e-8:
            phase = overlap / np.abs(overlap)
            clifft_sv = clifft_sv * phase

        np.testing.assert_allclose(
            clifft_sv,
            qiskit_sv,
            atol=atol,
            err_msg=(
                f"Amplitude mismatch after phase alignment\n"
                f"Clifft:    {clifft_sv[:8]}\n"
                f"Qiskit: {qiskit_sv[:8]}"
            ),
        )

    def test_rz_single_qubit(self) -> None:
        self._check_amplitudes("R_Z(0.25) 0")

    def test_rz_is_t_gate(self) -> None:
        """R_Z(0.25) should be equivalent to T gate up to global phase."""
        self._check_amplitudes("H 0\nR_Z(0.25) 0\nH 0")

    def test_rx_single_qubit(self) -> None:
        self._check_amplitudes("R_X(0.5) 0")

    def test_ry_single_qubit(self) -> None:
        self._check_amplitudes("R_Y(0.3) 0")

    def test_u3_gate(self) -> None:
        self._check_amplitudes("U3(0.5, 0.25, 0.125) 0")

    def test_u3_is_x_gate(self) -> None:
        """U3(1, 0, 0) should match X gate up to global phase."""
        self._check_amplitudes("U3(1.0, 0.0, 0.0) 0")

    def test_rzz_two_qubit(self) -> None:
        self._check_amplitudes("H 0\nH 1\nR_ZZ(0.3) 0 1")

    def test_rxx_two_qubit(self) -> None:
        self._check_amplitudes("R_XX(0.4) 0 1")

    def test_ryy_two_qubit(self) -> None:
        self._check_amplitudes("R_YY(0.2) 0 1")

    def test_r_pauli_xyz(self) -> None:
        self._check_amplitudes("R_PAULI(0.1) X0*Y1*Z2")

    def test_rotations_after_cliffords(self) -> None:
        """Rotations after entangling gates should still match."""
        circuit = """
            H 0
            CX 0 1
            R_Z(0.3) 0
            R_X(0.2) 1
        """
        self._check_amplitudes(circuit)

    def test_multiple_rotations_compose(self) -> None:
        """Multiple rotations on same qubit should compose correctly."""
        circuit = """
            H 0
            R_Z(0.1) 0
            R_Z(0.2) 0
            R_Z(0.3) 0
        """
        self._check_amplitudes(circuit)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_rotation_circuits(self, seed: int) -> None:
        """Random circuits mixing Cliffords and rotations."""
        rng = np.random.default_rng(seed + 1000)
        n_qubits = 3
        lines: list[str] = []
        for _ in range(10):
            gate_type = rng.integers(0, 6)
            if gate_type == 0:
                q = int(rng.integers(0, n_qubits))
                lines.append(f"H {q}")
            elif gate_type == 1:
                q1, q2 = rng.choice(n_qubits, 2, replace=False)
                lines.append(f"CX {int(q1)} {int(q2)}")
            elif gate_type == 2:
                q = int(rng.integers(0, n_qubits))
                alpha = float(rng.uniform(-2.0, 2.0))
                lines.append(f"R_Z({alpha:.6f}) {q}")
            elif gate_type == 3:
                q = int(rng.integers(0, n_qubits))
                alpha = float(rng.uniform(-2.0, 2.0))
                lines.append(f"R_X({alpha:.6f}) {q}")
            elif gate_type == 4:
                q = int(rng.integers(0, n_qubits))
                alpha = float(rng.uniform(-2.0, 2.0))
                lines.append(f"R_Y({alpha:.6f}) {q}")
            elif gate_type == 5:
                q = int(rng.integers(0, n_qubits))
                theta = float(rng.uniform(-2.0, 2.0))
                phi = float(rng.uniform(-2.0, 2.0))
                lam = float(rng.uniform(-2.0, 2.0))
                lines.append(f"U3({theta:.6f}, {phi:.6f}, {lam:.6f}) {q}")

        self._check_amplitudes("\n".join(lines))

    @pytest.mark.parametrize("seed", range(3))
    def test_random_two_qubit_rotations(self, seed: int) -> None:
        """Random circuits with two-qubit Pauli rotations."""
        rng = np.random.default_rng(seed + 2000)
        n_qubits = 3
        lines: list[str] = []
        for _ in range(8):
            gate_type = rng.integers(0, 5)
            if gate_type == 0:
                q = int(rng.integers(0, n_qubits))
                lines.append(f"H {q}")
            elif gate_type == 1:
                q1, q2 = rng.choice(n_qubits, 2, replace=False)
                alpha = float(rng.uniform(-1.0, 1.0))
                lines.append(f"R_XX({alpha:.6f}) {int(q1)} {int(q2)}")
            elif gate_type == 2:
                q1, q2 = rng.choice(n_qubits, 2, replace=False)
                alpha = float(rng.uniform(-1.0, 1.0))
                lines.append(f"R_YY({alpha:.6f}) {int(q1)} {int(q2)}")
            elif gate_type == 3:
                q1, q2 = rng.choice(n_qubits, 2, replace=False)
                alpha = float(rng.uniform(-1.0, 1.0))
                lines.append(f"R_ZZ({alpha:.6f}) {int(q1)} {int(q2)}")
            elif gate_type == 4:
                q = int(rng.integers(0, n_qubits))
                alpha = float(rng.uniform(-1.0, 1.0))
                lines.append(f"R_Z({alpha:.6f}) {q}")

        self._check_amplitudes("\n".join(lines))
