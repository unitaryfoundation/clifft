"""CH-form stabilizer state -- the compact O(n^2)-per-term store.

This is the "real" term backend the dense prototype was a stand-in for. A
stabilizer state on n qubits is stored as

    |phi> = omega * U_C * U_H * |s>

following the CH-form of Bravyi, Browne, Calpin, Campbell, Gosset, Howard,
"Simulation of quantum circuits by low-rank stabilizer decompositions",
Quantum 3, 181 (2019); arXiv:1808.00128, Section 4.1 (PDF in research/refs/).
Equation numbers below refer to that paper.

  * s in {0,1}^n        -- a computational basis string,
  * v in {0,1}^n        -- a Hadamard layer, U_H = H(v) = tensor_q H^{v_q}  (Eq.44),
  * omega in C          -- a global amplitude (carries sub-normalisation after
                           projection; ||phi||^2 = |omega|^2 always, since
                           U_C U_H|s> is a unit vector),
  * U_C                 -- a Clifford in the <S, CZ, CX> subgroup (it fixes
                           |0..0>, Eq.41), described by its stabilizer tableau
                           via n x n binary matrices F, G, M and gamma in Z_4^n:

      U_C^{-1} Z_p U_C = prod_j Z_j^{G[p,j]}
      U_C^{-1} X_p U_C = i^{gamma_p} prod_j X_j^{F[p,j]} Z_j^{M[p,j]}          (Eq.43)

    (note: the conjugation is by U_C^{-1}, the paper's convention -- this matters
    for every update rule, so the whole module follows it exactly). U_C
    preserves Pauli commutation, hence F G^T = I (mod 2).

Storage is O(n^2) bits per term instead of the dense 2^n complex amplitudes --
the point of this increment. S/CZ/CX act in O(n), H in O(n^2), a single
amplitude <x|phi> in O(n^2) (Eq.56), and `statevector` materialises the dense
vector only when explicitly asked (validation / cross-term sums).

Qubit convention matches the rest of the engine: qubit 0 is the least
significant bit, so bit q of an integer basis label x is (x >> q) & 1.
"""

from __future__ import annotations

import numpy as np

_SQRT1_2 = 1.0 / np.sqrt(2.0)
_I_POW = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j], dtype=complex)  # i^k, k mod 4
_H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)


class CHForm:
    """A single (sub-normalised) stabilizer state in CH-form (paper convention)."""

    __slots__ = ("n", "F", "G", "M", "g", "v", "s", "w")

    def __init__(self, n: int):
        self.n = n
        self.F = np.eye(n, dtype=np.uint8)
        self.G = np.eye(n, dtype=np.uint8)
        self.M = np.zeros((n, n), dtype=np.uint8)
        self.g = np.zeros(n, dtype=np.int64)  # gamma in Z_4
        self.v = np.zeros(n, dtype=np.uint8)
        self.s = np.zeros(n, dtype=np.uint8)
        self.w = 1.0 + 0j

    # ------------------------------------------------------------------ utils
    def copy(self) -> "CHForm":
        c = CHForm.__new__(CHForm)
        c.n, c.w = self.n, self.w
        c.F, c.G, c.M = self.F.copy(), self.G.copy(), self.M.copy()
        c.g, c.v, c.s = self.g.copy(), self.v.copy(), self.s.copy()
        return c

    def scale(self, c: complex) -> None:
        self.w *= c

    def norm2(self) -> float:
        """||phi||^2.  U_C U_H|s> is always a unit vector, so this is |omega|^2."""
        return float(abs(self.w) ** 2)

    # ---------------------------------------------- left-multiplication: L[Gamma]
    # Applying a gate to the STATE is U_C <- Gamma U_C (Eq.46), the left-multiply
    # rules of the paper (end of Sec 4.1). These act on ROWS of the tableau.

    def _L_S(self, q: int) -> None:
        self.M[q, :] ^= self.G[q, :]
        self.g[q] = (self.g[q] - 1) % 4

    def _L_CZ(self, q: int, r: int) -> None:
        self.M[q, :] ^= self.G[r, :]
        self.M[r, :] ^= self.G[q, :]

    def _L_CX(self, q: int, r: int) -> None:  # control q, target r
        mft_qr = int(self.M[q, :] @ self.F[r, :]) & 1  # (M F^T)_{q,r}, pre-update
        gnew = (self.g[q] + self.g[r] + 2 * mft_qr) % 4
        self.G[r, :] ^= self.G[q, :]
        self.F[q, :] ^= self.F[r, :]
        self.M[q, :] ^= self.M[r, :]
        self.g[q] = gnew

    # --------------------------------------------- right-multiplication: R[Gamma]
    # Used only inside the desuperposition (U_C <- U_C W_C). Acts on COLUMNS.

    def _R_S(self, q: int) -> None:
        self.M[:, q] ^= self.F[:, q]
        self.g = (self.g - self.F[:, q].astype(np.int64)) % 4

    def _R_CZ(self, q: int, r: int) -> None:
        self.M[:, q] ^= self.F[:, r]
        self.M[:, r] ^= self.F[:, q]
        self.g = (self.g + 2 * (self.F[:, q] & self.F[:, r]).astype(np.int64)) % 4

    def _R_CX(self, q: int, r: int) -> None:  # control q, target r
        self.G[:, q] ^= self.G[:, r]
        self.F[:, r] ^= self.F[:, q]
        self.M[:, q] ^= self.M[:, r]

    # ------------------------------------------------------ public Clifford gates
    def s_gate(self, q: int) -> None:
        self._L_S(q)

    def s_dag(self, q: int) -> None:
        for _ in range(3):  # S^dagger = S^3
            self._L_S(q)

    def z(self, q: int) -> None:
        self._L_S(q); self._L_S(q)  # Z = S^2

    def cz(self, q: int, r: int) -> None:
        self._L_CZ(q, r)

    def cx(self, q: int, r: int) -> None:
        self._L_CX(q, r)

    def h(self, q: int) -> None:
        self._apply_h(q)

    def x(self, q: int) -> None:
        self.h(q); self.z(q); self.h(q)  # X = H Z H

    def y(self, q: int) -> None:
        self.z(q); self.x(q); self.w *= 1j  # Y = i X Z

    def swap(self, a: int, b: int) -> None:
        self.cx(a, b); self.cx(b, a); self.cx(a, b)

    def clifford_1q(self, name: str, q: int) -> None:
        {"H": self.h, "S": self.s_gate, "S_DAG": self.s_dag,
         "X": self.x, "Y": self.y, "Z": self.z}[name](q)

    # --------------------------------------------- desuperposition (Proposition 4)
    def _desuperpose(self, t: np.ndarray, u: np.ndarray, delta: int) -> complex:
        """In-place build the CH data of U_C U_H(|t> + i^delta |u>), returning the
        scalar omega_P4 (|omega_P4| = sqrt(2)) of Eq.50. Assumes t != u and that
        self currently holds the U_C / U_H (= H(v)) context. Mutates F,G,M,g (via
        R[] right-multiplications by W_C), v and s; does NOT touch self.w."""
        n, v = self.n, self.v
        diff = (t ^ u).astype(bool)
        V0 = [i for i in range(n) if v[i] == 0 and diff[i]]
        V1 = [i for i in range(n) if v[i] == 1 and diff[i]]
        # Build V_C with Eq.54: collapse the branch difference onto a single qubit q.
        wc: list[tuple] = []
        if V0:
            q = V0[0]
            wc += [("CX", q, i) for i in V0 if i != q]
            wc += [("CZ", q, i) for i in V1]
        else:  # V0 empty, V1 non-empty
            q = V1[0]
            wc += [("CX", i, q) for i in V1 if i != q]
        # y, z differ only on bit q (paper, end of Prop.4 proof).
        if t[q] == 1:
            y = u.copy(); y[q] ^= 1; zz = u.copy()
        else:
            y = t.copy(); zz = t.copy(); zz[q] ^= 1
        # single-qubit residual on q:  H^{v_q}(|y_q> + i^delta |z_q>) = w_q S^a H^b |c>
        ket = np.zeros(2, dtype=complex)
        ket[int(y[q])] += 1.0
        ket[int(zz[q])] += _I_POW[delta % 4]
        if v[q] == 1:
            ket = _H2 @ ket
        a, b, c, omega_q = _match_single_qubit(ket)
        # U_C <- U_C W_C = U_C V_C S_q^a  (right multiplications)
        for kind, x_, y_ in wc:
            (self._R_CX if kind == "CX" else self._R_CZ)(x_, y_)
        for _ in range(a):
            self._R_S(q)
        # U_H' = H(v with v_q <- b);  s' = y except s'_q = c   (Prop.4 proof)
        self.v = v.copy()
        self.v[q] = b
        s_new = y.copy()
        s_new[q] = c
        self.s = s_new
        return omega_q

    def _apply_h(self, p: int) -> None:
        """Apply H_p to the state (Eqs.47-52)."""
        n = self.n
        s, v, F, G, M = self.s, self.v, self.F, self.G, self.M
        vbar = 1 - v
        t = (s ^ (G[p, :] & v)).astype(np.uint8)               # Eq.48
        u = (s ^ (F[p, :] & vbar) ^ (M[p, :] & v)).astype(np.uint8)
        alpha = int(np.sum(G[p, :] & vbar & s)) & 1            # Eq.49
        beta = int(np.sum((M[p, :] & vbar & s)
                          + (F[p, :] & v) * (M[p, :] ^ s))) & 1
        gp = int(self.g[p]) % 4
        if np.array_equal(t, u):  # trivial case (Eq.47 with t == u)
            self.w *= _SQRT1_2 * ((-1.0) ** alpha + _I_POW[gp] * (-1.0) ** beta)
            self.s = t
            return
        delta = (gp + 2 * (alpha + beta)) % 4
        omega_q = self._desuperpose(t, u, delta)
        self.w *= _SQRT1_2 * ((-1.0) ** alpha) * omega_q       # Eqs.51-52

    # ------------------------------------------------------ projective measurement
    def project(self, q: int, bit: int) -> "CHForm | None":
        """Project onto qubit q == bit (the gate (I + (-1)^bit Z_q)/2). Returns a
        new sub-normalised CHForm, or None if the outcome has zero support.

        Q = U_H^{-1} U_C^{-1} P U_C U_H with P = (-1)^bit Z_q; since
        U_C^{-1} Z_q U_C = Z(G[q]) (Eq.43), Q = (-1)^bit X(G[q]&v) Z(G[q]&~v).
        Then (I+Q)|s> = |s> + i^delta |t>, handled like H via Proposition 4."""
        out = self.copy()
        G, v, s = out.G, out.v, out.s
        a = (G[q, :] & v).astype(np.uint8)          # X-part of Q
        b = (G[q, :] & (1 - v)).astype(np.uint8)     # Z-part of Q
        parity = (int(bit) + int(np.sum(b & s))) & 1
        delta = (2 * parity) % 4
        t = (s ^ a).astype(np.uint8)
        if not a.any():  # outcome deterministic for this qubit
            if parity:   # (I+Q)|s> = 0  -> zero support
                return None
            return out   # full weight, ||.||^2 unchanged ((w/2)*2 = w)
        omega_q = out._desuperpose(s.copy(), t, delta)
        out.w *= 0.5 * omega_q
        return out

    # ------------------------------------------------------- amplitude readout
    def amplitude(self, x: int) -> complex:
        """<x|phi> via Eq.56.  Q = prod_{p: x_p=1} U_C^{-1} X_p U_C = i^mu Z(t)X(u),
        accumulated as a Pauli product; then
            <x|phi> = w * 2^{-|v|/2} * i^mu * prod_{v_j=1}(-1)^{u_j s_j}
                                            * prod_{v_j=0} [u_j == s_j]."""
        n = self.n
        mu = 0
        tz = np.zeros(n, dtype=np.uint8)  # Z-part of Q
        ux = np.zeros(n, dtype=np.uint8)  # X-part of Q
        for p in range(n):
            if not (x >> p) & 1:
                continue
            Fp, Mp = self.F[p, :], self.M[p, :]
            # multiply Q (= i^mu Z(tz) X(ux)) by i^{g_p} X(Fp) Z(Mp)
            mu = (mu + int(self.g[p]) + 2 * int(((ux ^ Fp) & Mp).sum())) % 4
            ux = ux ^ Fp
            tz = tz ^ Mp
        # u = ux should equal x F (mod 2); now apply <0|Q U_H|s>
        v, s = self.v, self.s
        nonh = v == 0
        if np.any((ux[nonh] ^ s[nonh]) != 0):
            return 0.0 + 0j
        amp = self.w * (_SQRT1_2 ** int(v.sum())) * _I_POW[mu]
        hsign = int(np.sum(ux[v == 1] & s[v == 1])) & 1
        return complex(-amp if hsign else amp)

    def statevector(self) -> np.ndarray:
        out = np.empty(2 ** self.n, dtype=complex)
        for x in range(2 ** self.n):
            out[x] = self.amplitude(x)
        return out

    def support_point(self) -> int:
        """A basis state x with <x|phi> != 0, in O(n^2) from the tableau (no
        scan): the amplitude formula constrains u = xF to equal s on the
        non-v positions; choosing u = s everywhere and inverting with
        F^{-1} = G^T (the G F^T = I invariant) gives x = G s (mod 2)."""
        x_bits = (self.G @ self.s) % 2
        return int(sum(int(b) << j for j, b in enumerate(x_bits)))

    def canonical_key(self) -> bytes:
        """Hashable key up to global phase, for the dedup recompression
        (materialises; native CH-form inner-product dedup is a later increment)."""
        vec = self.statevector()
        nrm = np.sqrt(np.vdot(vec, vec).real)
        if nrm < 1e-12:
            return b"0"
        d = vec / nrm
        k = int(np.argmax(np.abs(d)))
        d = d * np.exp(-1j * np.angle(d[k]))
        return (np.round(d, 9) + (0.0 + 0.0j)).tobytes()

    # --- cheap (O(n^2), non-materialising) duplicate merge for sparsify ---
    def merge_key(self) -> bytes:
        """Identifies the CH data exactly (excluding omega): identical tableaux
        represent the identical unit state, so their omegas add coherently."""
        return (self.F.tobytes() + self.G.tobytes() + self.M.tobytes()
                + self.g.tobytes() + self.v.tobytes() + self.s.tobytes())

    def merge_add(self, other: "CHForm") -> None:
        self.w += other.w


def _match_single_qubit(ket: np.ndarray):
    """Find (a in Z_4, b,c in {0,1}, omega) with ket == omega * S^a H^b |c>."""
    for b in (0, 1):
        for c in (0, 1):
            base = np.zeros(2, dtype=complex)
            base[c] = 1.0
            if b:
                base = _H2 @ base
            for a in range(4):
                vec = base.copy()
                vec[1] *= _I_POW[a]
                omega = complex(np.vdot(vec, ket))
                if np.allclose(ket, omega * vec, atol=1e-9):
                    return a, b, c, omega
    raise ValueError(f"no single-qubit stabilizer match for {ket}")
