"""Shared test fixtures and utilities for UCC Python tests."""

import numpy as np


def assert_statevectors_equal(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float = 1e-4, msg: str = ""
) -> None:
    """Assert two statevectors are equal up to global phase.

    Uses fidelity: |<psi|phi>|^2 >= 1 - rtol
    """
    fidelity = float(np.abs(np.vdot(expected, actual)) ** 2)
    if fidelity < 1.0 - rtol:
        raise AssertionError(f"Fidelity {fidelity:.6f} < {1.0 - rtol}. {msg}")


def binomial_tolerance(p: float, n: int, *, sigma: float = 5.0) -> float:
    """Compute tolerance for binomial proportion estimate.

    Returns sigma standard deviations of the binomial standard error.
    Default 5-sigma gives <1 in 3.5 million false positive rate per assertion.

    Args:
        p: Expected probability (0 < p < 1)
        n: Number of samples (shots)
        sigma: Number of standard deviations for the bound

    Returns:
        Tolerance value such that |observed - p| < tolerance with high probability
    """
    # Clamp p to avoid zero variance for p=0 or p=1
    p_clamped = max(min(p, 0.99), 0.01)
    std_err = float(np.sqrt((p_clamped * (1 - p_clamped)) / n))
    return sigma * std_err


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
    the compiler's Pauli compression and virtual axis allocation.

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
