"""Tiny dense reference for cross-checking noncomputational sampling.

Test-only. A minimal quantum core over the computational subspace (<= 5
qubits), built from first principles -- statevector unitaries, Born-rule
Z measurement, partial trace, and the per-site exact transition channel
(source-conditioned collapse plus the sqrt(1 - p) no-fire damping filter)
-- and deliberately independent of clifft's sampler, rewriter, driver, and
executor. Combined with explicit classical probabilities for initial levels,
transitions, and the classifier, it yields expected output distributions
to compare against ``clifft.noncomp.sample`` within shot noise.

The channel here is the *physical* Kraus map, applied uniformly to any
computational qubit: it knows nothing of clifft's known/unknown status
tracking, dormant/active instrument forms, traps, or continuations --
every one of those implementation strategies must reproduce it.

Scope: the exact supported subset only -- small Clifford circuits, a
Z-basis binary classifier, loss as a true partial trace, and the exact
per-site channel above. This is not a general simulator and must not grow
into one; the lossless self-check tests guard that its quantum core is
correct before it is trusted to judge the noncomputational pipeline.

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
    """P(survivor Z = 1) after the `lost` qubit is traced out.

    The survivor's single-qubit marginal is its reduced density matrix, which
    traces out every other qubit regardless, so the result does not depend on
    which other qubit is named `lost`. The parameter documents the scenario
    (lose `lost`, read `survivor`) and must differ from `survivor`.
    """
    if lost == survivor:
        raise ValueError("lost and survivor must be different qubits")
    rho = reduced_density(state, survivor, n)
    return float(np.real(rho[1, 1]))


# Exact per-site transition channel.
#
# The physical channel of one annotation target on a computational qubit,
# as Kraus operators on that qubit's factor:
#
#   fire, source s, destination d:  K = sqrt(p[d][s]) |after_d><s|
#   no fire:                        K0 = diag(sqrt(1 - ptot_g), sqrt(1 - ptot_e))
#
# where ptot_s is column s's total jump probability. A fire collapses the
# qubit onto its source; a noncomputational destination then removes the
# (now factored) qubit from the quantum register, a computational one
# re-prepares it at |d>. The branch helpers below return (weight, state)
# pairs with the state renormalized, so callers assemble the mixture with
# explicit weights.


def collapse(
    state: npt.NDArray[np.complex128], q: int, bit: int, n: int
) -> tuple[float, npt.NDArray[np.complex128]]:
    """Project qubit q onto |bit> and renormalize: (Born weight, post state)."""
    proj = _embed(_P1 if bit else _P0, q, n)
    post = proj @ state
    weight = float(np.real(post.conj() @ post))
    if weight <= 0.0:
        return 0.0, post
    return weight, post / np.sqrt(weight)


def damp_no_fire(
    state: npt.NDArray[np.complex128], q: int, ptot_g: float, ptot_e: float, n: int
) -> tuple[float, npt.NDArray[np.complex128]]:
    """Apply the no-fire filter K0 on qubit q: (branch weight, post state)."""
    k0 = np.array([[np.sqrt(1.0 - ptot_g), 0.0], [0.0, np.sqrt(1.0 - ptot_e)]], dtype=complex)
    post = _embed(k0, q, n) @ state
    weight = float(np.real(post.conj() @ post))
    if weight <= 0.0:
        return 0.0, post
    return weight, post / np.sqrt(weight)


def set_collapsed_qubit(
    state: npt.NDArray[np.complex128], q: int, source: int, dest: int, n: int
) -> npt.NDArray[np.complex128]:
    """Re-prepare a just-collapsed (factored) qubit from |source> to |dest>."""
    if source == dest:
        return state
    return _embed(GATES_1Q["X"], q, n) @ state
