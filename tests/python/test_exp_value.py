"""Python integration tests for EXP_VAL expectation value probes.

Tests cover:
  - Qiskit-independent exact oracle (numpy statevector Pauli expectation)
  - Statistical equivalence to destructive MPP measurement
  - Noise and Pauli-frame trajectory interaction
  - Regression on circuits without EXP_VAL
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_clifford_t_circuit

import ucc

# =============================================================================
# Helpers
# =============================================================================

# Single-qubit Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def pauli_expectation(sv: np.ndarray, pauli_str: str, num_qubits: int) -> float:
    """Compute <sv| P |sv> for a Pauli product string like 'X0*Z2'.

    Qubits not mentioned get identity. Uses little-endian ordering
    (qubit 0 = LSB) matching UCC convention.
    """
    # Parse pauli string into per-qubit operators
    ops: dict[int, str] = {}
    for term in pauli_str.split("*"):
        pauli_char = term[0]
        qubit = int(term[1:])
        ops[qubit] = pauli_char

    # Build full Pauli matrix via kronecker product (qubit 0 = rightmost)
    mat = np.array([[1.0]], dtype=np.complex128)
    for q in range(num_qubits):
        p = _PAULI[ops.get(q, "I")]
        mat = np.asarray(np.kron(p, mat), dtype=np.complex128)

    return float(np.real(sv.conj() @ mat @ sv))


def ucc_statevector(circuit_str: str) -> np.ndarray:
    """Compile and execute circuit in UCC, return dense statevector."""
    prog = ucc.compile(circuit_str)
    state = ucc.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    ucc.execute(prog, state)
    return np.array(ucc.get_statevector(prog, state))


def random_pauli_product(num_qubits: int, rng: np.random.Generator) -> str:
    """Generate a random non-identity Pauli product on the given qubits."""
    paulis = ["X", "Y", "Z"]
    # Pick 1 to num_qubits qubits to act on
    k = rng.integers(1, num_qubits + 1)
    qubits = rng.choice(num_qubits, size=k, replace=False)
    terms = [f"{rng.choice(paulis)}{q}" for q in sorted(qubits)]
    return "*".join(terms)


# =============================================================================
# Exact oracle: numpy statevector Pauli expectations
# =============================================================================


class TestExactOracle:
    """Compare EXP_VAL results to numpy statevector Pauli expectations."""

    def test_single_qubit_x_on_plus(self) -> None:
        """<X> on |+> = +1."""
        circuit = "H 0"
        sv = ucc_statevector(circuit)
        expected = pauli_expectation(sv, "X0", 1)

        result = ucc.sample(ucc.compile(f"{circuit}\nEXP_VAL X0"), 1, seed=0)
        np.testing.assert_allclose(result.exp_vals[0, 0], expected, atol=1e-12)

    def test_single_qubit_z_on_plus(self) -> None:
        """<Z> on |+> = 0."""
        circuit = "H 0"
        sv = ucc_statevector(circuit)
        expected = pauli_expectation(sv, "Z0", 1)

        result = ucc.sample(ucc.compile(f"{circuit}\nEXP_VAL Z0"), 1, seed=0)
        np.testing.assert_allclose(result.exp_vals[0, 0], expected, atol=1e-12)

    def test_bell_zz(self) -> None:
        """<Z0*Z1> on Bell state = +1."""
        circuit = "H 0\nCX 0 1"
        sv = ucc_statevector(circuit)
        expected = pauli_expectation(sv, "Z0*Z1", 2)

        result = ucc.sample(ucc.compile(f"{circuit}\nEXP_VAL Z0*Z1"), 1, seed=0)
        np.testing.assert_allclose(result.exp_vals[0, 0], expected, atol=1e-12)

    def test_bell_xx(self) -> None:
        """<X0*X1> on Bell state = +1."""
        circuit = "H 0\nCX 0 1"
        sv = ucc_statevector(circuit)
        expected = pauli_expectation(sv, "X0*X1", 2)

        result = ucc.sample(ucc.compile(f"{circuit}\nEXP_VAL X0*X1"), 1, seed=0)
        np.testing.assert_allclose(result.exp_vals[0, 0], expected, atol=1e-12)

    def test_t_gate_expectation(self) -> None:
        """<X> after H-T on |0> = cos(pi/4) = 1/sqrt(2)."""
        circuit = "H 0\nT 0"
        sv = ucc_statevector(circuit)
        expected = pauli_expectation(sv, "X0", 1)

        result = ucc.sample(ucc.compile(f"{circuit}\nEXP_VAL X0"), 1, seed=0)
        np.testing.assert_allclose(result.exp_vals[0, 0], expected, atol=1e-10)

    def test_multi_qubit_product(self) -> None:
        """<X0*Y1*Z2> on a 3-qubit state matches numpy oracle."""
        circuit = "H 0\nCX 0 1\nS 1\nH 2"
        sv = ucc_statevector(circuit)
        expected = pauli_expectation(sv, "X0*Y1*Z2", 3)

        result = ucc.sample(ucc.compile(f"{circuit}\nEXP_VAL X0*Y1*Z2"), 1, seed=0)
        np.testing.assert_allclose(result.exp_vals[0, 0], expected, atol=1e-10)

    @pytest.mark.parametrize("seed", range(8))
    def test_random_clifford_t_oracle(self, seed: int) -> None:
        """Random Clifford+T circuit with random Pauli product matches oracle."""
        rng = np.random.default_rng(seed + 1000)
        num_qubits = int(rng.integers(3, 6))
        depth = int(rng.integers(10, 21))

        circuit = random_clifford_t_circuit(num_qubits, depth, seed)
        sv = ucc_statevector(circuit)

        pauli = random_pauli_product(num_qubits, rng)
        expected = pauli_expectation(sv, pauli, num_qubits)

        prog = ucc.compile(f"{circuit}\nEXP_VAL {pauli}")
        result = ucc.sample(prog, 1, seed=0)
        np.testing.assert_allclose(
            result.exp_vals[0, 0],
            expected,
            atol=1e-8,
            err_msg=f"Pauli={pauli}, circuit seed={seed}, nq={num_qubits}, depth={depth}",
        )


# =============================================================================
# Statistical equivalence to destructive measurement
# =============================================================================


class TestStatisticalEquivalence:
    """EXP_VAL should agree with MPP measurement statistics."""

    @pytest.mark.parametrize(
        "circuit,pauli",
        [
            ("H 0", "X0"),
            ("H 0\nT 0", "X0"),
            ("H 0\nT 0", "Z0"),
            ("H 0\nCX 0 1", "Z0*Z1"),
            ("H 0\nCX 0 1\nT 0", "X0"),
        ],
    )
    def test_exp_val_matches_mpp_mean(self, circuit: str, pauli: str) -> None:
        """mean(exp_vals) ≈ mean(1 - 2*mpp_bits) over many shots."""
        shots = 5000

        # EXP_VAL path
        ev_prog = ucc.compile(f"{circuit}\nEXP_VAL {pauli}")
        ev_result = ucc.sample(ev_prog, shots, seed=42)
        ev_mean = float(np.mean(ev_result.exp_vals[:, 0]))

        # MPP path (destructive measurement of same Pauli)
        mpp_prog = ucc.compile(f"{circuit}\nMPP {pauli}")
        mpp_result = ucc.sample(mpp_prog, shots, seed=42)
        mpp_mean = float(np.mean(1.0 - 2.0 * mpp_result.measurements[:, -1].astype(np.float64)))

        # For deterministic states, both should be exact
        # For non-deterministic (T gate), use statistical tolerance
        atol = 5.0 * np.sqrt(1.0 / shots)  # 5-sigma
        np.testing.assert_allclose(
            ev_mean,
            mpp_mean,
            atol=atol,
            err_msg=f"EXP_VAL vs MPP mismatch: circuit='{circuit}', pauli='{pauli}'",
        )


# =============================================================================
# Noise and Pauli-frame trajectory tests
# =============================================================================


class TestPauliFrameInteraction:
    """Verify EXP_VAL correctly reads the Pauli frame."""

    def test_z_error_flips_x_expectation(self) -> None:
        """Z_ERROR(1.0) anti-commutes with X, flipping <X> from +1 to -1."""
        prog = ucc.compile("H 0\nZ_ERROR(1.0) 0\nEXP_VAL X0")
        result = ucc.sample(prog, 10, seed=0)
        np.testing.assert_allclose(result.exp_vals[:, 0], -1.0, atol=1e-12)

    def test_x_error_flips_z_expectation(self) -> None:
        """X_ERROR(1.0) anti-commutes with Z, flipping <Z> from +1 to -1."""
        prog = ucc.compile("X_ERROR(1.0) 0\nEXP_VAL Z0")
        result = ucc.sample(prog, 10, seed=0)
        np.testing.assert_allclose(result.exp_vals[:, 0], -1.0, atol=1e-12)

    def test_z_error_commutes_with_z(self) -> None:
        """Z_ERROR(1.0) commutes with Z, so <Z> on |0> stays +1."""
        prog = ucc.compile("Z_ERROR(1.0) 0\nEXP_VAL Z0")
        result = ucc.sample(prog, 10, seed=0)
        np.testing.assert_allclose(result.exp_vals[:, 0], 1.0, atol=1e-12)

    def test_measurement_feedback_cx(self) -> None:
        """EXP_VAL reads post-measurement Pauli frame via CX feedback.

        Circuit: H 0 / M 0 / CX rec[-1] 1 / EXP_VAL Z1
        Per shot: exp[0] == 1 - 2*meas[0]
        """
        prog = ucc.compile("H 0\nM 0\nCX rec[-1] 1\nEXP_VAL Z1")
        result = ucc.sample(prog, 100, seed=42)
        expected = 1.0 - 2.0 * result.measurements[:, 0].astype(np.float64)
        np.testing.assert_allclose(result.exp_vals[:, 0], expected, atol=1e-12)

    def test_measurement_feedback_cz(self) -> None:
        """EXP_VAL reads post-measurement Pauli frame via CZ feedback.

        Circuit: H 1 / H 0 / M 0 / CZ rec[-1] 1 / EXP_VAL X1
        Per shot: exp[0] == 1 - 2*meas[0]
        """
        prog = ucc.compile("H 1\nH 0\nM 0\nCZ rec[-1] 1\nEXP_VAL X1")
        result = ucc.sample(prog, 100, seed=42)
        expected = 1.0 - 2.0 * result.measurements[:, 0].astype(np.float64)
        np.testing.assert_allclose(result.exp_vals[:, 0], expected, atol=1e-12)

    def test_depolarize_reduces_expectation(self) -> None:
        """DEPOLARIZE1(1.0) on |0> applies X/Y/Z uniformly, <Z> averages to -1/3."""
        prog = ucc.compile("DEPOLARIZE1(1.0) 0\nEXP_VAL Z0")
        result = ucc.sample(prog, 10000, seed=0)
        mean_z = float(np.mean(result.exp_vals[:, 0]))
        # DEPOLARIZE1(1.0) applies one of {X, Y, Z} with equal probability.
        # X|0>=|1>: <Z>=-1, Y|0>=i|1>: <Z>=-1, Z|0>=|0>: <Z>=+1.
        # Average: (-1 + -1 + 1) / 3 = -1/3.
        assert (
            abs(mean_z - (-1.0 / 3.0)) < 0.05
        ), f"Expected ~-1/3 after full depolarization, got {mean_z}"


# =============================================================================
# No-EXP_VAL regression
# =============================================================================


class TestNoExpValRegression:
    """Circuits without EXP_VAL should behave identically to before."""

    def test_exp_vals_shape_empty(self) -> None:
        """exp_vals has shape (shots, 0) when no EXP_VAL in circuit."""
        prog = ucc.compile("H 0\nM 0")
        result = ucc.sample(prog, 10, seed=0)
        assert result.exp_vals.shape == (10, 0)

    def test_num_exp_vals_zero(self) -> None:
        """Program reports num_exp_vals == 0."""
        prog = ucc.compile("H 0\nM 0")
        assert prog.num_exp_vals == 0

    def test_no_exp_val_opcode_in_bytecode(self) -> None:
        """No OP_EXP_VAL appears in bytecode for a plain circuit."""
        prog = ucc.compile("H 0\nCX 0 1\nM 0\nM 1")
        for inst in prog:
            d = inst.as_dict()
            assert d["opcode"] != "OP_EXP_VAL"

    def test_measurements_unchanged(self) -> None:
        """Measurement outcomes match with and without EXP_VAL in circuit."""
        base_circuit = "H 0\nM 0"
        shots = 200

        prog_no_ev = ucc.compile(base_circuit)
        result_no_ev = ucc.sample(prog_no_ev, shots, seed=42)

        prog_with_ev = ucc.compile("H 0\nEXP_VAL Z0\nM 0")
        result_with_ev = ucc.sample(prog_with_ev, shots, seed=42)

        np.testing.assert_array_equal(
            result_no_ev.measurements,
            result_with_ev.measurements,
            err_msg="EXP_VAL changed measurement outcomes",
        )

    def test_detectors_observables_unchanged(self) -> None:
        """Detector and observable records are unaffected by EXP_VAL."""
        base = "H 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        shots = 100

        prog_no_ev = ucc.compile(base)
        result_no_ev = ucc.sample(prog_no_ev, shots, seed=42)

        prog_with_ev = ucc.compile(
            "H 0\nEXP_VAL X0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        )
        result_with_ev = ucc.sample(prog_with_ev, shots, seed=42)

        np.testing.assert_array_equal(result_no_ev.detectors, result_with_ev.detectors)
        np.testing.assert_array_equal(result_no_ev.observables, result_with_ev.observables)
