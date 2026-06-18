"""Benchmark the low-extent magic decomposition + sparsification.

Run: python -u -m research.chform_backend.bench_sparsify

Demonstrates the two facts that let a stabilizer-rank backend beat clifft's 2^k
dense block in the magic-sparse regime:

  1. EXTENT. With T branched in the {I,S} Clifford basis (T = alpha I + beta S),
     the decomposition's L1 weight grows as (|alpha|+|beta|)^t = 2^{0.114 t}, so
     the stabilizer extent ||c||_1^2 = 2^{0.228 t} -- NOT the 2^t of the naive
     |0>/|1> split. 0.228 is the BGH sampling exponent.

  2. SPARSIFICATION. Random importance-sampling to k terms is an unbiased
     estimator with E||psi - omega||^2 = (||c||_1^2 - 1)/k, so k ~ 2^{0.228 t}/d^2
     terms approximate the state to L2 error d. The exact rank 2^t collapses to
     the extent scale -- at the cost of clifft's exactness.

Sizes are kept small enough to build the exact 2^t state and materialise the
true statevector as ground truth. This shows the SCALING and validates the
estimator; it is not yet the past-clifft benchmark (that needs the
non-materialising inner product / measurement, increment 2a). The extent section
runs the real CH-form backend; the error sections use the dense backend purely
for speed -- sparsify() is backend-agnostic and validated identical on both.
"""

from __future__ import annotations

import numpy as np

from .engine import LowRankState


def _iqp_after_t(n: int, seed: int, backend: str) -> LowRankState:
    """H^n ; T^{+-} on each qubit -- stop right after the magic
    (chi = 2^n, ||c||_1 = 1.082^n)."""
    s = LowRankState(n, backend=backend)
    r = np.random.default_rng(seed)
    for q in range(n):
        s.clifford_1q("H", q)
    for q in range(n):
        s.t(q, dagger=bool(r.integers(2)))
    return s


def _finish(s: LowRankState, n: int, seed: int) -> None:
    """CZ chain + H^n -- the Clifford tail (applies to the current chi)."""
    r = np.random.default_rng(seed + 1)
    for q in range(n - 1):
        if r.random() < 0.6:
            s.cz(q, q + 1)
    for q in range(n):
        s.clifford_1q("H", q)


def _l1(terms) -> float:
    return float(sum(np.sqrt(t.norm2()) for t in terms))


def _sparse_from_base(base, weights, l1, k, n, rng) -> LowRankState:
    """Importance-sample k terms from a prebuilt base (the after-T term list),
    without rebuilding the exact 2^t state each time."""
    s = LowRankState(n, backend="dense")
    picks = rng.choice(len(base), size=k, p=weights / l1)
    s.terms = []
    for a in picks:
        nt = base[a].copy()
        nt.scale(l1 / (k * weights[a]))
        s.terms.append(nt)
    s.recompress_dedup()  # merge repeated samples (exact)
    return s


def extent_table():
    print("=" * 72)
    print("1) EXTENT (real CH-form backend): L1 ~ 1.082^t => extent = 2^{0.228 t}")
    print("=" * 72)
    print(f"  {'t':>3} {'chi=2^t':>9} {'||c||_1':>10} {'||c||_1^(1/t)':>14} "
          f"{'log2(extent)/t':>16}")
    for t in (2, 4, 6, 8, 10, 12):
        s = _iqp_after_t(t, seed=t, backend="chform")
        l1 = _l1(s.terms)
        print(f"  {t:>3} {s.chi:>9} {l1:>10.3f} {l1 ** (1.0 / t):>14.4f} "
              f"{np.log2(l1 ** 2) / t:>16.4f}")
    print("  rate -> 1.0824 = 2^0.114, exponent/t -> 0.2284 (the BGH exponent)")


def error_vs_k():
    n, seed, reps = 12, 12, 20
    print()
    print("=" * 72)
    print(f"2) SPARSIFICATION error vs k at t={n} (chi_exact = {2 ** n})")
    print("=" * 72)
    base = _iqp_after_t(n, seed, backend="dense")
    base_terms = base.terms
    weights = np.array([np.sqrt(t.norm2()) for t in base_terms])
    l1 = float(weights.sum())
    true = _iqp_after_t(n, seed, backend="dense")
    _finish(true, n, seed)
    psi_true = true.statevector()
    print(f"  ||c||_1 = {l1:.2f}, ||c||_1^2 = {l1 ** 2:.1f}, ||psi||^2 = 1")
    print(f"  {'k':>6} {'mean ||psi-omega||^2':>22} {'bound (L1^2-1)/k':>18} "
          f"{'mean fidelity':>14} {'mean chi':>9}")
    rng = np.random.default_rng(0)
    for k in (32, 64, 128, 256, 512, 1024):
        errs, fids, chis = [], [], []
        for _ in range(reps):
            s = _sparse_from_base(base_terms, weights, l1, k, n, rng)
            chis.append(s.chi)
            _finish(s, n, seed)
            omega = s.statevector()
            errs.append(float(np.vdot(psi_true - omega, psi_true - omega).real))
            fids.append(abs(complex(np.vdot(psi_true, omega))) ** 2)
        print(f"  {k:>6} {np.mean(errs):>22.4f} {(l1 ** 2 - 1) / k:>18.4f} "
              f"{np.mean(fids):>14.4f} {np.mean(chis):>9.0f}")
    print(f"  measured error tracks the (||c||_1^2-1)/k bound; k << {2 ** n} already")
    print("  gives high fidelity -- the rank collapses to the extent scale.")


def unbiasedness():
    n, seed, R, k = 10, 10, 2000, 16
    print()
    print("=" * 72)
    print("3) UNBIASEDNESS: averaging sparsified runs -> the true state")
    print("=" * 72)
    base = _iqp_after_t(n, seed, backend="dense")
    base_terms = base.terms
    weights = np.array([np.sqrt(t.norm2()) for t in base_terms])
    l1 = float(weights.sum())
    true = _iqp_after_t(n, seed, backend="dense")
    _finish(true, n, seed)
    psi_true = true.statevector()
    rng = np.random.default_rng(1)
    acc = np.zeros(2 ** n, dtype=complex)
    for _ in range(R):
        s = _sparse_from_base(base_terms, weights, l1, k, n, rng)
        _finish(s, n, seed)
        acc += s.statevector()
    acc /= R
    print(f"  t={n}, k={k} (chi_exact={2 ** n}), averaged over R={R} runs")
    print(f"  || mean(omega) - psi_true || = {np.linalg.norm(acc - psi_true):.4f}"
          f"  (-> 0 as R grows: the estimator is unbiased)")


if __name__ == "__main__":
    extent_table()
    error_vs_k()
    unbiasedness()
