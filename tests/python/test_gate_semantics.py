"""End-to-end semantics for Stim gates absorbed by the Clifford frontend."""

import numpy as np

import clifft


def _statevector(circuit: str) -> np.ndarray:
    return clifft.get_statevector(clifft.compile(circuit))


def _measurements(circuit: str, *, seed: int = 1) -> np.ndarray:
    return clifft.sample(clifft.compile(circuit), 1, seed=seed).measurements[0]


def test_clifford_aliases_match() -> None:
    np.testing.assert_allclose(_statevector("H 0"), _statevector("H_XZ 0"), atol=1e-12)
    np.testing.assert_allclose(
        _statevector("H 0\nS 0"), _statevector("H 0\nSQRT_Z 0"), atol=1e-12
    )
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
    np.testing.assert_allclose(
        _statevector("H 0\nI 0\nT 0"), _statevector("H 0\nT 0"), atol=1e-12
    )
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


def test_absorbed_single_qubit_cliffords() -> None:
    np.testing.assert_allclose(
        _statevector("SQRT_X 0\nSQRT_X 0"), _statevector("X 0"), atol=1e-12
    )
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
