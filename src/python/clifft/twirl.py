"""Pauli-twirling helpers for coherent control errors.

A coherent control error -- an over-rotation, a phase miscalibration -- is a
unitary, not a noncomputational status change. Clifft can simulate a coherent
error exactly at the usual active-rank cost; the standard fast-path
alternative is to *twirl* it: replace the error unitary ``U`` with the Pauli
channel whose probabilities are

    p_P = |tr(U P)|^2 / 4,    P in {I, X, Y, Z}.

The twirled channel preserves the diagonal of ``U``'s Pauli transfer matrix
and discards the coherences between Pauli components. That is an
approximation, not an equivalence: circuit statistics under the twirled
channel generally differ from the exact unitary even for a single
application (twirling one Hadamard turns a deterministic H-H identity into a
coin flip), and they coincide only in selected configurations where the
discarded coherences never reach a measurement.

These helpers only compute probabilities. Insert them into circuits as
ordinary noise instructions (``Z_ERROR(p)``, ``PAULI_CHANNEL_1(px, py, pz)``),
which both the plain sampler and the noncomputational path treat like any
other Pauli noise.

This module is experimental. It operates on one unitary at a time and leaves
the circuit editing to the caller; a more robust interface -- a circuit
annotated with the sites to twirl, with the noise nodes substituted
internally -- may supersede these free functions, changing or removing this
surface.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = [
    "pauli_probabilities",
    "rotation",
]

_PAULIS = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def pauli_probabilities(unitary: npt.ArrayLike) -> tuple[float, float, float]:
    """Twirl a single-qubit unitary into Pauli probabilities ``(p_x, p_y, p_z)``.

    The identity probability is the complement ``1 - p_x - p_y - p_z``. The
    result is insensitive to the unitary's global phase. Suitable as the
    arguments of a ``PAULI_CHANNEL_1`` instruction (or a single ``*_ERROR``
    when only one component is nonzero).

    Raises:
        ValueError: if ``unitary`` is not a 2x2 unitary matrix.
    """
    u = np.asarray(unitary, dtype=complex)
    if u.shape != (2, 2):
        raise ValueError(f"pauli_probabilities: expected a 2x2 matrix, got shape {u.shape}")
    if not np.allclose(u @ u.conj().T, np.eye(2), atol=1e-9):
        raise ValueError("pauli_probabilities: matrix is not unitary")
    return tuple(float(abs(np.trace(u @ p)) ** 2 / 4.0) for p in _PAULIS)  # type: ignore[return-value]


def rotation(axis: str, radians: float) -> tuple[float, float, float]:
    """Twirl of the rotation ``exp(-i * radians/2 * P_axis)``.

    Closed form: probability ``sin^2(radians / 2)`` on the rotation axis and
    zero elsewhere. This is the per-gate channel for an over-rotation by
    ``radians`` about ``axis`` ("X", "Y", or "Z").
    """
    p = math.sin(radians / 2.0) ** 2
    try:
        index = ("X", "Y", "Z").index(axis.upper())
    except ValueError:
        raise ValueError(f"rotation: axis must be 'X', 'Y', or 'Z', got {axis!r}") from None
    out = [0.0, 0.0, 0.0]
    out[index] = p
    return (out[0], out[1], out[2])
