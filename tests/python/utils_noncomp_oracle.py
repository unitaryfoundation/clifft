"""Tiny dense reference for cross-checking noncomputational sampling.

Test-only. A minimal density-matrix simulator over the computational subspace
(<= 3 qubits), built from first principles -- statevector unitaries, Born-rule
Z measurement, and partial trace -- and deliberately independent of clifft's
sampler, rewriter, and SVM. Combined with explicit classical probabilities for
initial levels, transitions, and the classifier, it yields expected output
distributions to compare against ``clifft.noncomp.sample`` within shot noise.

Scope: the exact supported subset only -- small Clifford circuits, a Z-basis
binary classifier, and state-independent loss handled as a true partial trace.
This is not a general simulator and must not grow into one; the lossless
self-check tests guard that its quantum core is correct before it is trusted to
judge the noncomputational pipeline.

Qubit/bit convention: qubit 0 is the most significant factor in the Kronecker
product, matching how the self-check compares against clifft's records.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_I = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)

GATES_1Q: dict[str, npt.NDArray[np.complex128]] = {
    "I": _I,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
}


def zero_state(n: int) -> npt.NDArray[np.complex128]:
    """The |0...0> statevector for n qubits."""
    v = np.zeros(2**n, dtype=complex)
    v[0] = 1.0
    return v


def _embed(u: npt.NDArray[np.complex128], q: int, n: int) -> npt.NDArray[np.complex128]:
    """Embed a single-qubit operator on qubit q into the n-qubit space."""
    ops = [_I] * n
    ops[q] = u
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full


def apply_1q(
    state: npt.NDArray[np.complex128], gate: str, q: int, n: int
) -> npt.NDArray[np.complex128]:
    """Apply a named single-qubit gate to qubit q."""
    return _embed(GATES_1Q[gate], q, n) @ state


def apply_cx(
    state: npt.NDArray[np.complex128], control: int, target: int, n: int
) -> npt.NDArray[np.complex128]:
    """Apply CX(control, target)."""
    u = _embed(_P0, control, n) + _embed(_P1, control, n) @ _embed(GATES_1Q["X"], target, n)
    return u @ state


def prob_one(state: npt.NDArray[np.complex128], q: int, n: int) -> float:
    """Born-rule probability that a Z measurement of qubit q yields 1."""
    return float(np.real(state.conj() @ (_embed(_P1, q, n) @ state)))


def reduced_density(
    state: npt.NDArray[np.complex128], keep: int, n: int
) -> npt.NDArray[np.complex128]:
    """Reduced 2x2 density matrix of qubit `keep`, tracing out the rest."""
    psi = state.reshape([2] * n)
    # Move the kept axis to the front, flatten the rest, then rho = sum_r |a_r><a_r|.
    psi = np.moveaxis(psi, keep, 0).reshape(2, -1)
    return psi @ psi.conj().T


def marginal_one_after_trace_out(
    state: npt.NDArray[np.complex128], lost: int, survivor: int, n: int
) -> float:
    """P(survivor Z = 1) after tracing out the `lost` qubit (a partial trace)."""
    # Tracing out one qubit then reading another reduces to the survivor's own
    # reduced density matrix (the trace is linear and the lost qubit factors out
    # of the survivor marginal), so reuse reduced_density on the survivor.
    rho = reduced_density(state, survivor, n)
    return float(np.real(rho[1, 1]))
