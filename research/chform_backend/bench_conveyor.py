"""The conveyor op-count experiment -- the composition's UNIQUE advantage.

Run: python -u -m research.chform_backend.bench_conveyor

A "magic conveyor": a width-w register reused across R measured rounds, each
    H^w ; T^{+-} on each ; CZ chain ; H^w ; measure all w ; classical feedforward
so every round's magic is measured out (the rank collapses) and the rounds are
linked into ONE connected computation by feedforward (classically-controlled X).

The point (from the profiling phase): the 2^{0.228 n} EXPONENT win is matched by
any *measurement-aware* stabilizer-rank simulator -- our plain CH-form engine IS
one, it collapses chi on each measurement. The composition's UNIQUE edge is a
constant/poly factor: clifft's symbolic frame evolves Cliffords for free, while a
pure stab-rank sim re-applies every Clifford to every one of chi terms. This
experiment measures exactly that:

    pure stab-rank (plain): clifford_term_ops = sum over Clifford gates of chi
    frame + residual:       clifford_term_ops = 0  (Cliffords absorbed into F in
                                                    O(n) each, terms untouched)

Both compute the SAME state (validated at small w with forced outcomes); only the
per-term Clifford work differs. This is the free-Clifford advantage the profiling
phase named as clifft's genuine edge over the best stab-rank baseline -- now
realized end to end (frame + residual + frame-conjugated magic + measurement).
"""

from __future__ import annotations

import numpy as np

from .engine import LowRankState


def run_conveyor(w: int, R: int, frame: bool, seed: int, forced=None):
    """Returns (state, outcomes). `forced` (a flat outcome list) pins the
    measurement results so frame and plain runs are comparable."""
    rg = np.random.default_rng(seed)
    dag = [[bool(b) for b in rg.integers(0, 2, size=w)] for _ in range(R)]
    s = LowRankState(w, backend="chform", frame=frame)
    dummy = np.random.default_rng(seed + 999)
    outs, fi = [], 0
    for r in range(R):
        for q in range(w):
            s.clifford_1q("H", q)
        for q in range(w):
            s.t(q, dagger=dag[r][q])
        for q in range(w - 1):
            s.cz(q, q + 1)
        for q in range(w):
            s.clifford_1q("H", q)
        round_outs = []
        for q in range(w):
            f = forced[fi] if forced is not None else None
            round_outs.append(s.measure_z(q, dummy, force=f)); fi += 1
        outs.extend(round_outs)
        for q in range(w):                       # feedforward: free Cliffords
            if round_outs[q]:
                s.clifford_1q("X", (q + 1) % w)
    return s, outs


def correctness():
    print("=" * 74)
    print("CORRECTNESS: frame conveyor == plain conveyor (forced outcomes)")
    print("=" * 74)
    worst = 0.0
    for w, R in ((2, 3), (3, 2), (4, 2)):
        sp, outs = run_conveyor(w, R, frame=False, seed=w * 7 + R)
        sf, _ = run_conveyor(w, R, frame=True, seed=w * 7 + R, forced=outs)
        worst = max(worst, float(np.max(np.abs(sp.statevector() - sf.statevector()))))
    assert worst < 1e-9, f"conveyor frame vs plain mismatch {worst}"
    print(f"  [OK] identical final state across w/R (max abs err {worst:.2e})")


def opcount_table():
    print()
    print("=" * 74)
    print("OP-COUNT: per-term Clifford work, pure stab-rank vs frame + residual")
    print("=" * 74)
    print(f"  {'w':>3} {'R':>3} {'peak_chi':>9} {'plain cliff_term_ops':>21} "
          f"{'frame cliff_term_ops':>21} {'frame gates (O(n))':>19}")
    R = 4
    for w in (4, 6, 8):
        sp, outs = run_conveyor(w, R, frame=False, seed=w * 7 + R)
        sf, _ = run_conveyor(w, R, frame=True, seed=w * 7 + R, forced=outs)
        assert sf.ctr.clifford_term_ops == 0
        print(f"  {w:>3} {R:>3} {sp.ctr.peak_chi:>9} {sp.ctr.clifford_term_ops:>21} "
              f"{sf.ctr.clifford_term_ops:>21} {sf.ctr.frame_clifford_gates:>19}")
    print("  plain pays sum_clifford chi per-term updates (grows with R, w, 2^w);")
    print("  frame pays ZERO -- every Clifford absorbed into F in O(n)/gate. Same")
    print("  state, same chi collapse on measurement; only the Clifford cost differs.")
    print("  This is the composition's unique edge over measurement-aware stab-rank")
    print("  (a constant/poly factor, NOT the exponent -- exactly as profiled).")


if __name__ == "__main__":
    correctness()
    opcount_table()
