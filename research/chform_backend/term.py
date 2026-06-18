"""Term backends for the low-rank engine.

A `Term` is one (sub-normalised) stabilizer state in the decomposition
`|psi> = sum_j term_j`. Two implementations satisfy the same interface:

  * `DenseTerm` -- a dense 2^n statevector. Trivially correct; the original
    prototype and the validation oracle for the CH-form.
  * `CHForm` (in `chform.py`) -- the real CH-form tableau, O(n^2) bits/term.

The engine is written against this interface, so swapping the per-term store
changes only memory/gate cost, never chi or the branching/measurement logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_S_DAG = np.array([[1, 0], [0, -1j]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_1Q = {"H": _H, "S": _S, "S_DAG": _S_DAG, "X": _X, "Y": _Y, "Z": _Z}

_ZERO_TOL = 1e-12


def _apply_1q(vec: np.ndarray, n: int, q: int, u: np.ndarray) -> np.ndarray:
    """Apply a 2x2 unitary to qubit q (LSB = qubit 0) of a 2^n statevector."""
    v = vec.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("ab,xbz->xaz", u, v).reshape(-1)


@runtime_checkable
class Term(Protocol):
    """The operations the engine needs from a single stabilizer term."""

    def copy(self) -> "Term": ...
    def clifford_1q(self, name: str, q: int) -> None: ...
    def cx(self, c: int, t: int) -> None: ...
    def cz(self, c: int, t: int) -> None: ...
    def swap(self, a: int, b: int) -> None: ...
    def project(self, q: int, bit: int) -> "Term | None": ...
    def scale(self, c: complex) -> None: ...
    def norm2(self) -> float: ...
    def amplitude(self, x: int) -> complex: ...
    def statevector(self) -> np.ndarray: ...
    def canonical_key(self) -> bytes: ...
    def merge_key(self) -> bytes: ...
    def merge_add(self, other: "Term") -> None: ...


class DenseTerm:
    """A stabilizer term stored as a dense 2^n statevector."""

    __slots__ = ("n", "v")

    def __init__(self, n: int, vec: np.ndarray):
        self.n = n
        self.v = vec

    def copy(self) -> "DenseTerm":
        return DenseTerm(self.n, self.v.copy())

    def clifford_1q(self, name: str, q: int) -> None:
        self.v = _apply_1q(self.v, self.n, q, _1Q[name])

    def cx(self, c: int, t: int) -> None:
        idx = np.arange(2 ** self.n)
        self.v = self.v[idx ^ (((idx >> c) & 1) << t)]

    def cz(self, c: int, t: int) -> None:
        idx = np.arange(2 ** self.n)
        self.v = self.v * np.where(((idx >> c) & 1) & ((idx >> t) & 1), -1.0, 1.0)

    def swap(self, a: int, b: int) -> None:
        idx = np.arange(2 ** self.n)
        ba, bb = (idx >> a) & 1, (idx >> b) & 1
        self.v = self.v[idx ^ ((ba ^ bb) << a) ^ ((ba ^ bb) << b)]

    def project(self, q: int, bit: int) -> "DenseTerm | None":
        idx = np.arange(2 ** self.n)
        keep = (((idx >> q) & 1) == bit)
        w = np.where(keep, self.v, 0.0)
        if np.vdot(w, w).real <= _ZERO_TOL:
            return None
        return DenseTerm(self.n, w)

    def scale(self, c: complex) -> None:
        self.v = self.v * c

    def norm2(self) -> float:
        return float(np.vdot(self.v, self.v).real)

    def amplitude(self, x: int) -> complex:
        return complex(self.v[x])

    def statevector(self) -> np.ndarray:
        return self.v

    def canonical_key(self) -> bytes:
        nrm = np.sqrt(np.vdot(self.v, self.v).real)
        if nrm < _ZERO_TOL:
            return b"0"
        d = self.v / nrm
        k = int(np.argmax(np.abs(d)))
        d = d * np.exp(-1j * np.angle(d[k]))
        # round + add 0.0 to collapse -0.0 -> 0.0 (signed zeros from complex
        # multiplies break byte-equality of otherwise identical vectors).
        return (np.round(d, 9) + (0.0 + 0.0j)).tobytes()

    def merge_key(self) -> bytes:
        return self.canonical_key()

    def merge_add(self, other: "DenseTerm") -> None:
        self.v = self.v + other.v
