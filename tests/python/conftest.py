"""Shared test fixtures and utilities for Clifft Python tests."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import clifft
import clifft.experimental as experimental


@pytest.fixture(params=[clifft, experimental], ids=["legacy", "experimental"])
def sampling_api(request: pytest.FixtureRequest) -> Any:
    """Run supported sampling conformance tests against both Python APIs."""
    return cast(Any, request.param)


@pytest.fixture(params=[clifft, experimental], ids=["legacy", "experimental"])
def basis_probabilities_api(request: pytest.FixtureRequest) -> Any:
    """Run shared exact basis-query tests against both Python APIs."""
    return cast(Any, request.param)


@pytest.fixture(params=["legacy", "experimental"], ids=["legacy", "experimental"])
def statevector_from_circuit(
    request: pytest.FixtureRequest,
) -> Callable[[str], npt.NDArray[np.complex128]]:
    """Compile and expand a pure-state circuit through either backend."""
    if request.param == "experimental":

        def experimental_statevector(stim_text: str) -> npt.NDArray[np.complex128]:
            return cast(
                npt.NDArray[np.complex128],
                experimental.get_statevector(experimental.compile(stim_text)),
            )

        return experimental_statevector

    def legacy_statevector(stim_text: str) -> npt.NDArray[np.complex128]:
        program = clifft.compile(stim_text)
        state = clifft.State(
            peak_rank=program.peak_rank,
            num_measurements=program.num_measurements,
        )
        clifft.execute(program, state)
        return cast(npt.NDArray[np.complex128], clifft.get_statevector(program, state))

    return legacy_statevector


@pytest.fixture(
    params=[clifft.noncomp.sample, experimental.sample_noncomputational],
    ids=["svm", "symbolic-coordinate"],
)
def noncomp_sampling_api(request: pytest.FixtureRequest) -> Any:
    """Run supported noncomputational trajectories through both executors."""
    return cast(Any, request.param)


def noncomp_transition_matrix(
    entries: Mapping[tuple[int, int], float],
) -> list[list[float]]:
    """Build a five-level T[to][from] matrix from its nonzero entries."""
    matrix = [[0.0] * 5 for _ in range(5)]
    for (destination, source), probability in entries.items():
        matrix[destination][source] = probability
    return matrix


def noncomp_classifier_matrix_with_column(
    level: int, probabilities: Sequence[float]
) -> list[list[float]]:
    """Build a faithful computational classifier with one replaced column."""
    matrix = [[0.0] * 5 for _ in probabilities]
    for current_level in range(5):
        matrix[0][current_level] = 1.0
    matrix[0][1], matrix[1][1] = 0.0, 1.0
    for symbol, probability in enumerate(probabilities):
        matrix[symbol][level] = probability
    return matrix


def assert_statevectors_equiv(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float = 1e-4, msg: str = ""
) -> None:
    """Assert two statevectors are equivalent up to global phase.

    Uses fidelity: abs(|<psi|phi>|^2 - 1) <= rtol.
    Catches both underflow (imperfect overlap) and overflow (numerical error).
    """
    fidelity = float(np.abs(np.vdot(expected, actual)) ** 2)
    if abs(fidelity - 1.0) > rtol:
        raise AssertionError(f"Fidelity {fidelity:.6f}, expected ~1.0 (rtol={rtol}). {msg}")


def assert_statevectors_componentwise_equal(
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
    *,
    atol: float = 1e-6,
    rtol: float = 0.0,
    msg: str = "",
) -> None:
    """Assert amplitudes match componentwise, including global phase."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        atol=atol,
        rtol=rtol,
        err_msg=msg,
    )


def binomial_tolerance(p: float, n: int, *, sigma: float = 5.0) -> float:
    """Compute tolerance for binomial proportion estimate.

    Returns sigma standard deviations of the binomial standard error.
    Default 5-sigma gives <1 in 3.5 million false positive rate per assertion.

    Args:
        p: Expected probability (0 to 1 inclusive)
        n: Number of samples (shots)
        sigma: Number of standard deviations for the bound

    Returns:
        Tolerance value such that |observed - p| < tolerance with high probability.
        Returns a tiny epsilon (1e-12) for deterministic probabilities (p == 0
        or p == 1) so that exact matches pass strict-less-than comparisons.
    """
    if p == 0.0 or p == 1.0:
        return 1e-12
    std_err = float(np.sqrt((p * (1 - p)) / n))
    return sigma * std_err


def cross_binomial_tolerance(p: float, n: int, *, sigma: float = 5.0) -> float:
    """Tolerance for comparing proportions from two independent samplers.

    When comparing p_hat_a - p_hat_b where both are independent binomial
    proportions with the same underlying p and sample size n, the
    standard error of the difference is sqrt(2) * StdErr(single).

    Args:
        p: Pooled probability estimate
        n: Number of samples per sampler
        sigma: Number of standard deviations for the bound

    Returns:
        Tolerance for |p_hat_a - p_hat_b| < tolerance
    """
    return float(np.sqrt(2.0)) * binomial_tolerance(p, n, sigma=sigma)


def random_clifford_t_circuit(num_qubits: int, depth: int, seed: int) -> str:
    """Generate a random universal Clifford+T circuit (noiseless, no measurements)."""
    rng = np.random.default_rng(seed)
    gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG"]

    lines: list[str] = []
    for _ in range(depth):
        if num_qubits > 1 and rng.random() < 0.3:
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"CX {q1} {q2}")
        else:
            gate = rng.choice(gates_1q)
            q = rng.integers(0, num_qubits)
            lines.append(f"{gate} {q}")
    return "\n".join(lines)


def random_dense_clifford_t_circuit(
    num_qubits: int, depth: int, seed: int, *, two_qubit_prob: float = 0.5
) -> str:
    """Generate a random Clifford+T circuit with dense entanglement.

    Higher 2-qubit gate probability and includes CY/CZ alongside CX.
    Produces circuits with heavy multi-qubit interference that stress
    the compiler's Pauli localization and virtual axis allocation.

    Args:
        num_qubits: Number of qubits.
        depth: Number of gate layers.
        seed: Random seed.
        two_qubit_prob: Probability of emitting a 2-qubit gate (default 0.5).

    Returns:
        Circuit string in .stim format.
    """
    rng = np.random.default_rng(seed)
    gates_1q = ["H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z"]
    gates_2q = ["CX", "CY", "CZ"]

    lines: list[str] = []
    for _ in range(depth):
        if num_qubits > 1 and rng.random() < two_qubit_prob:
            gate = rng.choice(gates_2q)
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"{gate} {q1} {q2}")
        else:
            gate = rng.choice(gates_1q)
            q = rng.integers(0, num_qubits)
            lines.append(f"{gate} {q}")
    return "\n".join(lines)


def random_clifford_circuit(num_qubits: int, depth: int, seed: int) -> str:
    """Generate a random pure-Clifford circuit (no T gates, no measurements)."""
    rng = np.random.default_rng(seed)
    gates_1q = ["H", "S", "S_DAG", "X", "Y", "Z"]
    gates_2q = ["CX", "CY", "CZ"]

    lines: list[str] = []
    for _ in range(depth):
        if num_qubits > 1 and rng.random() < 0.4:
            gate = rng.choice(gates_2q)
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            lines.append(f"{gate} {q1} {q2}")
        else:
            gate = rng.choice(gates_1q)
            q = rng.integers(0, num_qubits)
            lines.append(f"{gate} {q}")
    return "\n".join(lines)
