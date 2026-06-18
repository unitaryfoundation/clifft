"""Native (non-materialising) inner products and norm estimation for the
low-rank engine -- arXiv:1808.00128 Sec 4.3, Lemmas 2/3/4.

The CH-form term store is already O(n^2) bits, but `statevector`, measurement
and dedup still build a 2^n vector to get norms / amplitudes. This module
removes that: it computes <phi | phi_A> between a CH-form stabilizer state and an
*equatorial* state phi_A in O(n^3) (Lemma 3 + the exponential sum Lemma 4), and
uses it for the Bravyi-Gosset-Howard norm estimator (Lemma 2)

    eta_A = 2^n |<phi_A | psi>|^2,   E_A[eta_A] = ||psi||^2,   Var <= ||psi||^4,

so ||psi||^2 = ||sum_a term_a||^2 is estimated by averaging eta over random
equatorial A -- with no 2^n vector anywhere. Equatorial states (Eq.60):

    |phi_A> = 2^{-n/2} sum_x i^{x A x^T} |x>,   A symmetric, diag in Z_4, off in Z_2.

Everything here is validated against direct dense computation in test_normest.py.
"""

from __future__ import annotations

import numpy as np

from .chform import CHForm, _I_POW


# ----------------------------------------------------- exponential sums Z(B)
def exp_sum_direct(B: np.ndarray) -> complex:
    """Z(B) = sum_{x in {0,1}^m} i^{x B x^T}, by brute force (validation oracle).
    B symmetric with diag in Z_4, off-diagonal in Z_2."""
    m = B.shape[0]
    tot = 0.0 + 0j
    for xb in range(2 ** m):
        x = [(xb >> k) & 1 for k in range(m)]
        e = 0
        for a in range(m):
            if x[a]:
                e += int(B[a, a])
                for b in range(a + 1, m):
                    if x[b]:
                        e += 2 * int(B[a, b])
        tot += _I_POW[e % 4]
    return tot


def _real_exp_sum(M: np.ndarray, L: np.ndarray) -> int:
    """sum_{x in {0,1}^n} (-1)^{x M x^T + L x^T}, M binary (possibly
    non-symmetric, non-zero diagonal), L binary (Lemma 4, real case)."""
    M = (np.asarray(M, dtype=np.int64) % 2).copy()
    L = (np.asarray(L, dtype=np.int64) % 2).copy()
    n = len(L)
    factor = 1
    while True:
        if n == 0:
            return factor
        # find an asymmetric off-diagonal pair (i<j, M[i,j] != M[j,i])
        piv = None
        for i in range(n):
            row, col = M[i, i + 1:], M[i + 1:, i]
            diff = np.nonzero(row != col)[0]
            if diff.size:
                piv = (i, i + 1 + int(diff[0]))
                break
        if piv is None:  # symmetric -> the form is linear, Q = (diag(M)+L).x
            if np.any((np.diag(M) + L) % 2):
                return 0
            return factor * (1 << n)
        i, j = piv
        Mii, Mjj, Li, Lj = int(M[i, i]), int(M[j, j]), int(L[i]), int(L[j])
        others = [k for k in range(n) if k != i and k != j]
        oth = np.array(others, dtype=np.int64)
        m1 = (M[i, oth] + M[oth, i]) % 2
        m2 = (M[j, oth] + M[oth, j]) % 2
        Melse = M[np.ix_(oth, oth)]
        Lelse = L[oth]
        ai, aj = (Li + Mii) % 2, (Lj + Mjj) % 2
        Mnew = (Melse + np.outer(m1, m2)) % 2
        Lnew = (Lelse + aj * m1 + ai * m2) % 2
        factor *= 2 * (-1) ** (ai * aj)
        M, L, n = Mnew, Lnew, len(others)


def exp_sum(B: np.ndarray) -> complex:
    """Z(B) = sum_{x in {0,1}^m} i^{x B x^T} in O(m^3) (Lemma 4 + Prop.6).
    B symmetric, diag in Z_4, off-diagonal in Z_2."""
    n = B.shape[0]
    if n == 0:
        return 1.0 + 0j
    diag = np.array([int(B[a, a]) for a in range(n)], dtype=np.int64)
    Kvec = diag % 2          # B_aa = 2 L_a + K_a
    Lvec = (diag // 2) % 2
    # Z_2 form Q over n+1 variables (Eq.76):
    #   Q = sum_{a<b}(B_ab + K_a K_b) x_a x_b + sum_a K_a x_a x_{n+1} + sum_a L_a x_a
    N = n + 1
    QM = np.zeros((N, N), dtype=np.int64)  # upper-triangular quadratic coeffs
    for a in range(n):
        for b in range(a + 1, n):
            QM[a, b] = (int(B[a, b]) + Kvec[a] * Kvec[b]) % 2
        QM[a, n] = Kvec[a] % 2
    QL = np.zeros(N, dtype=np.int64)
    QL[:n] = Lvec
    re = _real_exp_sum(QM, QL) / 2.0                  # Re(Z)
    QLi = QL.copy(); QLi[n] ^= 1
    im = _real_exp_sum(QM, QLi) / 2.0                 # Im(Z)
    return complex(re + 1j * im)


# ------------------------------------------------------ equatorial states / A
def equatorial_state(A: np.ndarray) -> np.ndarray:
    """Dense |phi_A> = 2^{-n/2} sum_x i^{x A x^T} |x>  (validation only)."""
    n = A.shape[0]
    out = np.empty(2 ** n, dtype=complex)
    for xb in range(2 ** n):
        x = [(xb >> k) & 1 for k in range(n)]
        e = 0
        for a in range(n):
            if x[a]:
                e += int(A[a, a])
                for b in range(a + 1, n):
                    if x[b]:
                        e += 2 * int(A[a, b])
        out[xb] = _I_POW[e % 4]
    return out / (np.sqrt(2.0) ** n)


def random_equatorial(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform A in M_n: diagonal in Z_4, off-diagonal in Z_2 (symmetric)."""
    A = np.zeros((n, n), dtype=np.int64)
    for a in range(n):
        A[a, a] = int(rng.integers(4))
        for b in range(a + 1, n):
            val = int(rng.integers(2))
            A[a, b] = A[b, a] = val
    return A


# -------------------------------------------------- Lemma 3: <phi | phi_A>
def inner_equatorial(phi: CHForm, A: np.ndarray) -> complex:
    """<phi | phi_A> for CH-form phi (its omega included) and equatorial A (Eq.62).

    J: diag = gamma, off = (M F^T) mod 2;  K = G^T (A + J) G;  the |v|-restriction
    B = (K + 2 diag(s + s K))|_{v=1};  then
        <phi|phi_A> = conj(omega) 2^{-(n+|v|)/2} i^{s K s^T} (-1)^{s.v} Z(B)."""
    n = phi.n
    F = phi.F.astype(np.int64)
    G = phi.G.astype(np.int64)
    M = phi.M.astype(np.int64)
    g = phi.g.astype(np.int64)
    v = phi.v.astype(np.int64)
    s = phi.s.astype(np.int64)
    # J and A+J, reduced to M_n (diag mod 4, off mod 2)
    J = (M @ F.T) % 2
    np.fill_diagonal(J, g % 4)
    AJ = (A + J).astype(np.int64)
    for a in range(n):
        AJ[a, a] %= 4
        for b in range(a + 1, n):
            AJ[a, b] = AJ[b, a] = AJ[a, b] % 2
    K = G.T @ AJ @ G
    sK = s @ K
    vidx = [j for j in range(n) if v[j] == 1]
    m = len(vidx)
    B = np.zeros((m, m), dtype=np.int64)
    for ai, a in enumerate(vidx):
        B[ai, ai] = (K[a, a] + 2 * (s[a] + sK[a])) % 4
        for bi in range(ai + 1, m):
            b = vidx[bi]
            B[ai, bi] = B[bi, ai] = K[a, b] % 2
    Z = exp_sum(B)
    sKsT = int(s @ K @ s) % 4
    sv = int(s @ v) % 2
    val = _I_POW[sKsT] * ((-1.0) ** sv) * Z * (2.0 ** (-(n + m) / 2.0))
    return np.conj(phi.w) * val


# -------------------------------------------------- Lemma 2: norm estimation
def estimate_norm2(terms, n: int, samples: int, rng: np.random.Generator) -> float:
    """Estimate ||sum_a term_a||^2 by averaging eta_A = 2^n |<phi_A|psi>|^2 over
    `samples` random equatorial A. Unbiased, Var <= ||psi||^4 (Lemma 2)."""
    acc = 0.0
    for _ in range(samples):
        A = random_equatorial(n, rng)
        # <psi|phi_A> = sum_a <term_a|phi_A>; eta = 2^n |<phi_A|psi>|^2
        inner = sum(inner_equatorial(t, A) for t in terms)
        acc += (2 ** n) * abs(inner) ** 2
    return acc / samples
