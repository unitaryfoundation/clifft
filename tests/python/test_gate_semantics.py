"""End-to-end semantics for Stim gates absorbed by the Clifford frontend."""

import numpy as np

import clifft


def _statevector(circuit: str) -> np.ndarray:
    return np.asarray(clifft.get_statevector(clifft.compile(circuit)))


def _measurements(circuit: str, *, seed: int = 1) -> np.ndarray:
    return np.asarray(clifft.sample(clifft.compile(circuit), 1, seed=seed).measurements[0])


def _assert_statevectors_equal(actual_circuit: str, expected_circuit: str) -> None:
    np.testing.assert_allclose(
        _statevector(actual_circuit),
        _statevector(expected_circuit),
        atol=1e-12,
        rtol=0,
    )


def _assert_statevectors_equivalent(actual_circuit: str, expected_circuit: str) -> None:
    actual = _statevector(actual_circuit)
    expected = _statevector(expected_circuit)
    overlap = np.vdot(expected, actual)
    assert not np.isclose(abs(overlap), 0.0, atol=1e-12)
    phase = overlap / abs(overlap)
    np.testing.assert_allclose(actual, phase * expected, atol=1e-12, rtol=0)


def _assert_statevectors_differ(first_circuit: str, second_circuit: str) -> None:
    first = _statevector(first_circuit)
    second = _statevector(second_circuit)
    overlap = abs(np.vdot(first, second))
    assert not np.isclose(overlap, 1.0, atol=1e-12)


def test_pauli_product_phase_gates_match_named_single_qubit_gates() -> None:
    _assert_statevectors_equal("H 0\nSPP Z0", "H 0\nS 0")
    _assert_statevectors_equal("H 0\nSPP_DAG Z0", "H 0\nS_DAG 0")
    _assert_statevectors_equal("H 0\nTPP Z0", "H 0\nT 0")
    _assert_statevectors_equal("H 0\nTPP_DAG Z0", "H 0\nT_DAG 0")


def test_inverted_pauli_product_matches_conjugated_gate() -> None:
    _assert_statevectors_equal("SPP !X0", "Z 0\nSPP X0\nZ 0")


def test_inverted_pauli_products_preserve_named_gate_phase() -> None:
    _assert_statevectors_equal("H 0\nSPP !Z0", "H 0\nX 0\nSPP Z0\nX 0")
    _assert_statevectors_equal("H 0\nSPP_DAG !Z0", "H 0\nX 0\nSPP_DAG Z0\nX 0")
    _assert_statevectors_equal("H 0\nTPP !Z0", "H 0\nX 0\nTPP Z0\nX 0")
    _assert_statevectors_equal("H 0\nTPP_DAG !Z0", "H 0\nX 0\nTPP_DAG Z0\nX 0")


def test_negative_rewound_tpp_preserves_named_gate_phase() -> None:
    _assert_statevectors_equal("H 0\nX 0\nTPP Z0\nX 0", "H 0\nX 0\nT 0\nX 0")


def test_spp_clifford_action_matches_named_square_root_gates() -> None:
    _assert_statevectors_equivalent("SPP Y0", "SQRT_Y 0")
    _assert_statevectors_equivalent("SPP Y0*Y1", "SQRT_YY 0 1")
    _assert_statevectors_equivalent("SPP_DAG Y0*Y1", "SQRT_YY_DAG 0 1")
    _assert_statevectors_equivalent("H 0\nH 1\nSPP Z0*Z1", "H 0\nH 1\nSQRT_ZZ 0 1")


def test_spp_matches_two_tpp_gates_componentwise() -> None:
    _assert_statevectors_equal("H 0\nH 1\nSPP Z0*Z1", "H 0\nH 1\nTPP Z0*Z1\nTPP Z0*Z1")


def test_spp_xx_matches_the_named_square_root_gate() -> None:
    _assert_statevectors_equivalent("H 0\nSPP X0*X1", "H 0\nSQRT_XX 0 1")


def test_nontrivial_pauli_product_phase_gates_match_their_decompositions() -> None:
    basis_change = "H 0\nH_YZ 1\nCX 1 0\nCX 2 0\n"
    uncompute = "CX 2 0\nCX 1 0\nH_YZ 1\nH 0"
    _assert_statevectors_equal(
        "H 2\nSPP X0*Y1*Z2", "H 2\n" + basis_change + "S 0\n" + uncompute
    )
    _assert_statevectors_equal(
        "H 2\nTPP X0*Y1*Z2", "H 2\n" + basis_change + "T 0\n" + uncompute
    )


def test_multiple_tpp_products_are_applied_in_order() -> None:
    _assert_statevectors_equal("H 0\nTPP Z0 X1", "H 0\nT 0\nH 1\nT 1\nH 1")


def test_multiple_spp_products_are_applied_in_order() -> None:
    _assert_statevectors_equal("SPP X0 Z0", "SPP X0\nSPP Z0")
    _assert_statevectors_equal("SPP Z0 X0", "SPP Z0\nSPP X0")
    _assert_statevectors_differ("SPP X0 Z0", "SPP Z0 X0")
    _assert_statevectors_differ("SPP Z0 X0", "I 0")


def test_pauli_product_phase_gates_are_independent_of_term_order() -> None:
    _assert_statevectors_equal("H 0\nH 1\nSPP X0*Y1*Z2", "H 0\nH 1\nSPP Z2*X0*Y1")
    _assert_statevectors_equal("H 0\nH 1\nTPP X0*Y1*Z2", "H 0\nH 1\nTPP Z2*X0*Y1")


def test_clifford_aliases_match() -> None:
    np.testing.assert_allclose(_statevector("H 0"), _statevector("H_XZ 0"), atol=1e-12)
    np.testing.assert_allclose(_statevector("H 0\nS 0"), _statevector("H 0\nSQRT_Z 0"), atol=1e-12)
    np.testing.assert_allclose(
        _statevector("H 0\nX 1\nCZSWAP 0 1"),
        _statevector("H 0\nX 1\nSWAPCZ 0 1"),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _statevector("H 0\nZCX 0 1"),
        np.array([2**-0.5, 0, 0, 2**-0.5], dtype=np.complex128),
        atol=1e-12,
    )


def test_identity_is_a_noop_but_sets_circuit_width() -> None:
    np.testing.assert_allclose(_statevector("H 0\nI 0\nT 0"), _statevector("H 0\nT 0"), atol=1e-12)
    state = _statevector("I 3\nH 0")
    assert state.shape == (16,)
    np.testing.assert_allclose(state[:2], [2**-0.5, 2**-0.5], atol=1e-12)
    np.testing.assert_array_equal(state[2:], 0)


def test_iswap_phase_and_inverse() -> None:
    np.testing.assert_allclose(
        _statevector("X 0\nISWAP 0 1"),
        np.array([0, 0, 1j, 0], dtype=np.complex128),
        atol=1e-12,
    )
    circuit = "H 0\nCX 0 1"
    np.testing.assert_allclose(
        _statevector(circuit + "\nISWAP 0 1\nISWAP_DAG 0 1"),
        _statevector(circuit),
        atol=1e-12,
    )


def test_swap_exchanges_qubit_amplitudes() -> None:
    np.testing.assert_allclose(
        _statevector("X 0\nSWAP 0 1"),
        np.array([0, 0, 1, 0], dtype=np.complex128),
        atol=1e-12,
    )


def test_absorbed_single_qubit_cliffords() -> None:
    np.testing.assert_allclose(_statevector("SQRT_X 0\nSQRT_X 0"), _statevector("X 0"), atol=1e-12)
    np.testing.assert_allclose(
        np.abs(_statevector("H 0\nC_XYZ 0\nC_XYZ 0\nC_XYZ 0")),
        np.abs(_statevector("H 0")),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.abs(_statevector("H_XY 0")), [0, 1], atol=1e-12)


def test_mpad_and_inverted_measurements() -> None:
    np.testing.assert_array_equal(_measurements("MPAD 1 0 1 0"), [1, 0, 1, 0])
    np.testing.assert_array_equal(_measurements("MPAD !0 !1"), [1, 0])
    np.testing.assert_array_equal(_measurements("M !0"), [1])


def test_pair_measurement_aliases_match_mpp() -> None:
    preparations = {
        "XX": "H 0\nCX 0 1",
        "YY": "H 0\nCX 0 1",
        "ZZ": "H 0\nCX 0 1",
    }
    for basis, preparation in preparations.items():
        pair = _measurements(f"{preparation}\nM{basis} 0 1")
        product = _measurements(f"{preparation}\nMPP {basis[0]}0*{basis[1]}1")
        np.testing.assert_array_equal(pair, product)


def test_y_reset_uses_a_z_correction() -> None:
    for seed in range(20):
        assert _measurements("S 0\nH 0\nRY 0\nMY 0", seed=seed)[0] == 0
        assert _measurements("S 0\nH 0\nMRY 0\nMY 0", seed=seed)[1] == 0
        assert _measurements("H 0\nCX 0 1\nRY 0\nMY 0\nM 1", seed=seed)[0] == 0
