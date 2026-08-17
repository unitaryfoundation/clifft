"""Focused contracts for experimental symbolic-coordinate statevectors."""

from typing import Any

import numpy as np
import pytest

import clifft
import clifft.experimental as experimental


def test_statevector_preserves_exact_global_phase() -> None:
    program = experimental.compile("H 0\nT 0\nT 0\nH 0")
    np.testing.assert_allclose(
        experimental.get_statevector(program),
        np.array([0.5 + 0.5j, 0.5 - 0.5j]),
        atol=1e-6,
        rtol=0,
    )


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
def test_statevector_rejects_nonunitary_programs(circuit: str, kwargs: dict[str, Any]) -> None:
    program = experimental.compile(circuit, **kwargs)
    with pytest.raises(ValueError, match="requires pure-state unitary evolution"):
        experimental.get_statevector(program)


def test_statevector_allows_expectation_probes() -> None:
    with_probe = experimental.compile("H 0\nEXP_VAL X0")
    without_probe = experimental.compile("H 0")
    np.testing.assert_allclose(
        experimental.get_statevector(with_probe),
        experimental.get_statevector(without_probe),
        atol=1e-12,
        rtol=0,
    )


def test_drop_non_unitary_pass_enables_unitary_skeleton() -> None:
    passes = clifft.HirPassManager()
    passes.add(clifft.DropNonUnitaryPass())
    program = experimental.compile("H 0\nM 0\nX_ERROR(0.1) 0", hir_passes=passes)
    expected = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(
        experimental.get_statevector(program), [expected, expected], atol=1e-6, rtol=0
    )


def test_statevector_keeps_existing_dense_limit() -> None:
    program = experimental.compile("H 10")
    with pytest.raises(RuntimeError, match="limited to 10 qubits"):
        experimental.get_statevector(program)
