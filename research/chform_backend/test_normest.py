"""Validate the native inner product / norm estimation (norm_est.py) against
direct dense computation.

Run: python -m research.chform_backend.test_normest
"""

from __future__ import annotations

import numpy as np

from .chform import CHForm
from .engine import LowRankState
from . import norm_est as ne


def _rand_chform(n, depth, rng):
    """A random stabilizer state as a single CH-form (random Clifford on |0>)."""
    c = CHForm(n)
    names = ["H", "S", "S_DAG", "X", "Y", "Z"]
    for _ in range(depth):
        for q in range(n):
            c.clifford_1q(str(rng.choice(names)), q)
        for q in range(0, n - 1, 2):
            (c.cx if rng.random() < 0.5 else c.cz)(q, q + 1)
    c.scale(np.exp(1j * rng.uniform(0, 2 * np.pi)) * rng.uniform(0.3, 2.0))
    return c


def test_exp_sum(trials=300):
    """Lemma 4 O(n^3) exponential sum vs brute force."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        m = int(rng.integers(0, 7))
        B = np.zeros((m, m), dtype=np.int64)
        for a in range(m):
            B[a, a] = rng.integers(4)
            for b in range(a + 1, m):
                B[a, b] = B[b, a] = rng.integers(2)
        worst = max(worst, abs(ne.exp_sum(B) - ne.exp_sum_direct(B)))
    assert worst < 1e-9, f"exp_sum mismatch {worst}"
    print(f"[OK] Lemma 4 exponential sum vs brute force on {trials} matrices "
          f"(max abs err {worst:.2e})")


def test_inner_equatorial(trials=300):
    """Lemma 3 <phi|phi_A> vs materialised inner product."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(1, 6))
        phi = _rand_chform(n, int(rng.integers(1, 4)), rng)
        A = ne.random_equatorial(n, rng)
        got = ne.inner_equatorial(phi, A)
        ref = np.vdot(phi.statevector(), ne.equatorial_state(A))  # <phi|phi_A>
        worst = max(worst, abs(got - ref))
    assert worst < 1e-9, f"inner_equatorial mismatch {worst}"
    print(f"[OK] Lemma 3 <phi|phi_A> vs dense on {trials} states "
          f"(max abs err {worst:.2e})")


def test_norm_estimate(trials=12):
    """Lemma 2 norm estimator -> true ||psi||^2 (Clifford+T low-rank states)."""
    rng = np.random.default_rng(2)
    worst_rel = 0.0
    for _ in range(trials):
        n = int(rng.integers(2, 5))
        s = LowRankState(n, backend="chform")
        for q in range(n):
            s.clifford_1q("H", q)
        for q in range(n):
            s.t(q, dagger=bool(rng.integers(2)))
        for q in range(n - 1):
            s.cz(q, q + 1)
        exact = float(np.vdot(s.statevector(), s.statevector()).real)
        est = ne.estimate_norm2(s.terms, n, samples=4000, rng=rng)
        rel = abs(est - exact) / exact
        worst_rel = max(worst_rel, rel)
    # estimator std ~ ||psi||^2 / sqrt(L); L=4000 -> ~1.6% typical, allow margin
    assert worst_rel < 0.15, f"norm estimate off by {worst_rel:.3f}"
    print(f"[OK] Lemma 2 norm estimate -> ||psi||^2 on {trials} low-rank states "
          f"(worst rel err {worst_rel:.3f}, L=4000)")


if __name__ == "__main__":
    test_exp_sum()
    test_inner_equatorial()
    test_norm_estimate()
