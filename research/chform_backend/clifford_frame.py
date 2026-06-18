"""Global Clifford frame for the composition (clifft-style free Clifford evolution).

The composition idea (the original stab-rank bet): factor the state as

    |state> = F . ( sum_a term_a )

where F is a SINGLE Clifford unitary shared across all terms (the "frame", like
clifft's symbolic Clifford layer) and the term_a are the residual stabilizer
states (CH-form). A Clifford gate G is then absorbed into the frame for free,

    G |state> = (G F) (sum_a term_a)        -- update F, terms UNTOUCHED,

so the residual backend never pays the O(chi n^2)-per-Clifford cost a *pure*
stab-rank simulator pays. Only non-Clifford (T / magic) gates and measurements
touch the terms, and they are conjugated through F:

    T_q F (sum term) = F . (F^-1 T_q F) (sum term),  and  F^-1 T_q F is a
    rotation about the Pauli  P' = F^-1 Z_q F,

so the frame exposes exactly one thing to the residual layer: `conj_Z(q)`, the
Pauli F^-1 Z_q F. That is what this module provides.

Representation: the inverse-conjugation tableau -- for each qubit q we store the
Paulis  xc[q] = F^-1 X_q F  and  zc[q] = F^-1 Z_q F  (a symplectic basis).
Left-multiplying F by a gate G (F <- G F) updates only the rows of the qubit(s)
G touches, each an O(n) Pauli product:
    new(F^-1 P_q F) = F^-1 (G^-1 P_q G) F,
and G^-1 P_q G is a short Pauli (substitute X_j -> xc[j], Z_j -> zc[j]). So a
Clifford gate costs O(n) on the frame, vs O(chi n^2) on the residual -- the win.

A Pauli is (x in {0,1}^n, z in {0,1}^n, phase in Z_4 meaning i^phase X(x) Z(z)).
Hermitian Paulis have phase in {0,2}; products transiently carry i's (mod 4).
Qubit 0 = LSB, matching the rest of the engine.
"""

from __future__ import annotations

import numpy as np


def _pauli_mul(p1, x1, z1, p2, x2, z2):
    """(i^p1 X(x1)Z(z1)) (i^p2 X(x2)Z(z2)) = i^p X(x)Z(z).
    Moving Z(z1) past X(x2): Z(z1)X(x2) = (-1)^{z1.x2} X(x2)Z(z1)."""
    p = (p1 + p2 + 2 * int(np.dot(z1, x2) % 2)) % 4
    return p, x1 ^ x2, z1 ^ z2


# G^-1 P_q G for the basic gates, as (phase, [(qubit, 'X'|'Z'), ...]) factor lists
# (substitute X_j -> xc[j], Z_j -> zc[j] to get F^-1 (G^-1 P_q G) F).
def _gate_images(name, qs):
    """Return {(q,'X'): (phase, factors), (q,'Z'): (...)} for the gate's support."""
    if name in ("H",):
        (q,) = qs
        return {(q, "X"): (0, [(q, "Z")]), (q, "Z"): (0, [(q, "X")])}
    if name == "S":  # S^-1 X S = -Y = i^3 XZ ; Z -> Z
        (q,) = qs
        return {(q, "X"): (3, [(q, "X"), (q, "Z")]), (q, "Z"): (0, [(q, "Z")])}
    if name == "S_DAG":  # S X S^-1 = Y = i^1 XZ ; Z -> Z
        (q,) = qs
        return {(q, "X"): (1, [(q, "X"), (q, "Z")]), (q, "Z"): (0, [(q, "Z")])}
    if name == "X":  # X: X->X, Z->-Z
        (q,) = qs
        return {(q, "X"): (0, [(q, "X")]), (q, "Z"): (2, [(q, "Z")])}
    if name == "Z":  # Z: X->-X, Z->Z
        (q,) = qs
        return {(q, "X"): (2, [(q, "X")]), (q, "Z"): (0, [(q, "Z")])}
    if name == "Y":  # Y: X->-X, Z->-Z
        (q,) = qs
        return {(q, "X"): (2, [(q, "X")]), (q, "Z"): (2, [(q, "Z")])}
    if name == "CX":  # control c, target t: X_c->X_cX_t, Z_t->Z_cZ_t (others fixed)
        c, t = qs
        return {(c, "X"): (0, [(c, "X"), (t, "X")]), (t, "Z"): (0, [(c, "Z"), (t, "Z")])}
    if name == "CZ":  # X_c->X_c Z_t, X_t->Z_c X_t (Z's fixed)
        c, t = qs
        return {(c, "X"): (0, [(c, "X"), (t, "Z")]), (t, "X"): (0, [(c, "Z"), (t, "X")])}
    if name == "SWAP":
        a, b = qs
        return {(a, "X"): (0, [(b, "X")]), (a, "Z"): (0, [(b, "Z")]),
                (b, "X"): (0, [(a, "X")]), (b, "Z"): (0, [(a, "Z")])}
    raise ValueError(name)


class CliffordFrame:
    """A Clifford unitary F, stored by F^-1 X_q F (xc) and F^-1 Z_q F (zc)."""

    __slots__ = ("n", "xc_x", "xc_z", "xc_p", "zc_x", "zc_z", "zc_p", "gates")

    def __init__(self, n: int):
        self.n = n
        # xc[q] = F^-1 X_q F, zc[q] = F^-1 Z_q F ; init F = I.
        self.xc_x = np.eye(n, dtype=np.uint8)
        self.xc_z = np.zeros((n, n), dtype=np.uint8)
        self.xc_p = np.zeros(n, dtype=np.int64)
        self.zc_x = np.zeros((n, n), dtype=np.uint8)
        self.zc_z = np.eye(n, dtype=np.uint8)
        self.zc_p = np.zeros(n, dtype=np.int64)
        # gate history (in application-to-ket order), for one-time readout F.term
        self.gates: list[tuple] = []

    def _row(self, q, which):
        if which == "X":
            return self.xc_p[q], self.xc_x[q], self.xc_z[q]
        return self.zc_p[q], self.zc_x[q], self.zc_z[q]

    def _subst(self, phase, factors):
        """i^phase * prod_j (xc[j] or zc[j]) -- the stored Pauli for each factor."""
        n = self.n
        p, x, z = phase % 4, np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
        for (q, which) in factors:
            rp, rx, rz = self._row(q, which)
            p, x, z = _pauli_mul(p, x, z, rp, rx, rz)
        return p, x, z

    def apply_left(self, name: str, *qs: int) -> None:
        """F <- (gate) F. Only the touched qubits' rows change; each O(n)."""
        images = _gate_images(name, qs)
        updates = {}
        for (q, which), (phase, factors) in images.items():
            updates[(q, which)] = self._subst(phase, factors)
        for (q, which), (p, x, z) in updates.items():
            if which == "X":
                self.xc_p[q], self.xc_x[q], self.xc_z[q] = p, x, z
            else:
                self.zc_p[q], self.zc_x[q], self.zc_z[q] = p, x, z
        self.gates.append((name, qs))

    def apply_to(self, term) -> None:
        """Apply F itself to a residual term (in place): term <- F . term, by
        replaying the recorded gates. Used only at readout (one-time, O(|gates|))
        -- evolution uses the O(n) tableau (conj_Z), never this."""
        for name, qs in self.gates:
            if name in ("H", "S", "S_DAG", "X", "Y", "Z"):
                term.clifford_1q(name, qs[0])
            elif name == "CX":
                term.cx(qs[0], qs[1])
            elif name == "CZ":
                term.cz(qs[0], qs[1])
            elif name == "SWAP":
                term.swap(qs[0], qs[1])

    # convenience names matching the engine's gate set
    def h(self, q): self.apply_left("H", q)
    def s(self, q): self.apply_left("S", q)
    def s_dag(self, q): self.apply_left("S_DAG", q)
    def x(self, q): self.apply_left("X", q)
    def y(self, q): self.apply_left("Y", q)
    def z(self, q): self.apply_left("Z", q)
    def cx(self, c, t): self.apply_left("CX", c, t)
    def cz(self, c, t): self.apply_left("CZ", c, t)
    def swap(self, a, b): self.apply_left("SWAP", a, b)

    def conj_Z(self, q: int):
        """F^-1 Z_q F as (phase in Z_4, x bits, z bits) -- what magic/measurement
        in the frame need (T_q is diagonal, a function of Z_q)."""
        return int(self.zc_p[q]) % 4, self.zc_x[q].copy(), self.zc_z[q].copy()

    def conj_X(self, q: int):
        return int(self.xc_p[q]) % 4, self.xc_x[q].copy(), self.xc_z[q].copy()
