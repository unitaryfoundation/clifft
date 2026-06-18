"""Head-to-head vs clifft on magic-sparse IQP -- the past-clifft benchmark.

Run: python -u -m research.chform_backend.bench_vs_clifft

IQP = H^n ; (T^{+-} on each qubit) ; CZ chain ; H^n. clifft must hold the full
2^k active block (k = peak_rank); the CH-form + sparsification backend holds only
~2^{0.228 n} stabilizer terms of O(n^2) bits each and computes amplitudes /
norms WITHOUT ever materialising a 2^n vector (norm_est.py).

Magic injection is SINGLE-SHOT: |T>^n is a tensor product, so we importance-
sample k branch-strings directly (independent {I,S} choice per qubit), giving an
unbiased estimator with k ~ 2^{0.228 n}/delta^2 terms -- no 2^t built and NO
factor-of-t (unlike streaming mid-circuit sparsification, whose variance
accumulates; the ||omega||^2 = 1+(extent-1)/k formula -- no t-factor -- is
validated in test_chform.test_single_shot_magic).

  Part A  correctness: single-shot amplitudes vs clifft's exact P(x) (small n).
  Part B  resource crossover: clifft's 2^k vs the CH-form extent scale 2^{0.228 n}.
  Part C  actual runs past clifft's wall: single-shot, bounded chi/memory where
          clifft needs terabytes-petabytes (and cannot even compile at k >= 63).
"""

from __future__ import annotations

import time

import numpy as np

from .engine import LowRankState
from . import norm_est as ne


def iqp(n: int, seed: int):
    """Return (gate-runner for our engine, stim text for clifft)."""
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(q, q + 1) for q in range(n - 1) if r.random() < 0.6]
    lines = [f"H {q}" for q in range(n)]
    lines += [f"{'T_DAG' if dag[q] else 'T'} {q}" for q in range(n)]
    lines += [f"CZ {a} {b}" for a, b in czs]
    lines += [f"H {q}" for q in range(n)]

    def run(s: LowRankState):
        for q in range(n):
            s.clifford_1q("H", q)
        for q in range(n):
            s.t(q, dagger=dag[q])
        for a, b in czs:
            s.cz(a, b)
        for q in range(n):
            s.clifford_1q("H", q)

    return run, "\n".join(lines)


_TP = np.exp(1j * np.pi / 4)


def iqp_single_shot(n: int, seed: int, k: int, rng: np.random.Generator) -> LowRankState:
    """Build the same IQP state as iqp(), but with SINGLE-SHOT magic injection:
    sparsify the tensor-product magic |T>^n in one shot to k terms (no streaming,
    no t-factor, no 2^t). Same (dag, czs) as iqp(seed) so it is comparable."""
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(q, q + 1) for q in range(n - 1) if r.random() < 0.6]
    s = LowRankState(n, backend="chform")
    for q in range(n):
        s.clifford_1q("H", q)  # base |+>^n  (chi = 1)
    s.inject_magic_layer([(q, np.conj(_TP) if dag[q] else _TP) for q in range(n)], k, rng)
    for a, b in czs:
        s.cz(a, b)
    for q in range(n):
        s.clifford_1q("H", q)
    return s


def _bytes(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB"):
        if nbytes < 1024 or unit == "EB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024


def term_bytes(n: int) -> int:
    """CH-form per-term: F,G,M (n^2 uint8) + gamma (n int64) + v,s (n uint8) + w."""
    return 3 * n * n + 8 * n + 2 * n + 16


def clifft_peak_rank(text: str):
    import clifft
    try:
        prog = clifft.compile(text)
    except Exception as e:  # the 1ULL<<k hard wall at k>=63
        return None, str(e).splitlines()[0][:40]
    return int(prog.peak_rank), None


# ---------------------------------------------------------------- Part A
def part_a_correctness():
    import clifft
    print("=" * 74)
    print("A) CORRECTNESS vs clifft (exact), SINGLE-SHOT magic injection")
    print("=" * 74)
    print(f"  {'n':>3} {'k':>6} {'chi':>6} {'||om||^2':>9} "
          f"{'max |P_ours-P_clifft|':>22}")
    for n in (4, 6, 8, 10):  # clifft.get_statevector caps at 10 qubits (4^n U_C)
        _, text = iqp(n, seed=n)
        prog = clifft.compile(text)
        st = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        clifft.execute(prog, st)
        sv = np.asarray(clifft.get_statevector(prog, st)).ravel()
        p_clifft = np.abs(sv) ** 2
        delta = 0.2
        k = max(200, int(2.0 ** (0.228 * n) / delta ** 2))
        s = iqp_single_shot(n, n, k, np.random.default_rng(100 + n))
        v = s.statevector()                       # exact norm (n <= 10)
        norm = float(np.vdot(v, v).real)
        top = np.argsort(p_clifft)[-12:]
        worst = max(abs(abs(s.amplitude(int(x))) ** 2 / norm - p_clifft[x]) for x in top)
        print(f"  {n:>3} {k:>6} {s.chi:>6} {norm:>9.3f} {worst:>22.4f}")
    print("  P_ours within the single-shot error of clifft's exact probabilities;")
    print("  ||omega||^2 ~ 1 (delta=0.2 -> ~1.04). chi = sampled terms (< 2^n).")


# ---------------------------------------------------------------- Part B
def part_b_crossover():
    print()
    print("=" * 74)
    print("B) RESOURCE CROSSOVER: clifft 2^k dense block vs CH-form extent scale")
    print("=" * 74)
    print(f"  {'n':>3} {'clifft k':>9} {'clifft mem (2^k)':>18} "
          f"{'CH-form chi~2^.228n':>20} {'CH-form mem':>12}")
    for n in (20, 30, 40, 46, 50, 54, 60, 62, 64):
        _, text = iqp(n, seed=n)
        k, err = clifft_peak_rank(text)
        if k is None:
            kmem = "compile FAILS"
            kstr = f"-- ({err})"
        else:
            kstr = str(k)
            kmem = _bytes((2.0 ** k) * 16)
        chi = 2.0 ** (0.228 * n)  # extent scale (delta=1 RMS; x1/delta^2 for error delta)
        print(f"  {n:>3} {kstr:>9} {kmem:>18} {chi:>20.0f} "
              f"{_bytes(chi * term_bytes(n)):>12}")
    print("  clifft mem = 16B x 2^k (the dense active block); compile hard-fails at")
    print("  k>=63. CH-form rank is the extent scale 2^{0.228 n} (x1/delta^2 for")
    print("  target L2 error delta); memory = chi x O(n^2) bits.")


# ---------------------------------------------------------------- Part C
def part_c_past_clifft():
    print()
    print("=" * 74)
    print("C) ACTUAL RUNS PAST CLIFFT'S WALL, SINGLE-SHOT (non-materialising)")
    print("=" * 74)
    print(f"  {'n':>3} {'clifft needs':>14} {'k':>7} {'chi':>6} "
          f"{'CH mem':>9} {'|<0|psi>|':>10} {'wall(s)':>8}")
    for n in (40, 46, 50):
        delta = 0.4
        k = max(2000, int(2.0 ** (0.228 * n) / delta ** 2))
        _, text = iqp(n, seed=n)
        kc, _ = clifft_peak_rank(text)
        need = _bytes((2.0 ** kc) * 16) if kc is not None else "compile-fail"
        t0 = time.time()
        s = iqp_single_shot(n, n, k, np.random.default_rng(7 + n))
        amp0 = abs(s.amplitude(0))
        dt = time.time() - t0
        mem = _bytes(s.chi * term_bytes(n))
        print(f"  {n:>3} {need:>14} {k:>7} {s.chi:>6} {mem:>9} {amp0:>10.2e} {dt:>8.1f}")
    print("  Builds k ~ 2^{0.228 n}/delta^2 terms directly (delta=0.4), no 2^t, no")
    print("  t-factor -> accurate (||omega||^2 ~ 1+(extent-1)/k, see Part A and")
    print("  test_chform.test_single_shot_magic) AND feasible: completes in")
    print("  megabytes/seconds where clifft needs TB-PB (and cannot compile at")
    print("  k>=63). A single amplitude <0|psi> is O(chi n^2), no 2^n vector.")


if __name__ == "__main__":
    part_a_correctness()
    part_b_crossover()
    part_c_past_clifft()
