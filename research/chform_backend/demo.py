"""Demonstrate the composition mechanism on the low-rank engine.

Run: python -m research.chform_backend.demo

Shows two things on circuits small enough to run the dense-backed engine exactly:

  1. UNITARY Clifford+T (no mid-circuit measurement): the stabilizer rank chi
     grows monotonically toward 2^(#T) -- the naive, un-reduced case.

  2. CONVEYOR (a register re-used across measured rounds): chi peaks at
     ~2^(per-round magic) each round and then COLLAPSES back when the round is
     measured out. Peak chi is bounded by ONE round's magic, NOT the total --
     this is clifft's measurement-driven reduction expressed as a rank bound.

Also reports `clifford_term_ops` = sum over Clifford gates of chi-at-that-time:
the number of per-term Clifford updates a pure stab-rank simulator must perform.
clifft's frame handles the Clifford evolution of dormant axes for ~free, so this
count is precisely the work the composition (clifft frame + residual stab-rank)
removes relative to a pure stab-rank backend.
"""

from __future__ import annotations

import numpy as np

from .engine import LowRankState


def run_unitary_iqp(n: int, seed: int = 0, backend: str = "dense") -> LowRankState:
    """H^n ; (T on each qubit) ; CZ chain ; H^n   -- no measurement."""
    rng = np.random.default_rng(seed)
    s = LowRankState(n, backend=backend)
    for q in range(n):
        s.clifford_1q("H", q)
    for q in range(n):
        s.t(q, dagger=bool(rng.integers(2)))
    for q in range(n - 1):
        if rng.random() < 0.6:
            s.cz(q, q + 1)
    for q in range(n):
        s.clifford_1q("H", q)
    return s


def run_conveyor(width: int, rounds: int, seed: int = 0, backend: str = "dense"):
    """A width-qubit register re-used across `rounds` measured mini-IQP rounds.

    Each round: H all; T on each (fresh magic -- the prior round's measurement +
    this H avoid T*T=S); CZ chain; H all; measure all (collapse). Returns the
    state plus the peak chi observed within each round.
    """
    rng = np.random.default_rng(seed)
    s = LowRankState(width, backend=backend)
    per_round_peak = []
    for _ in range(rounds):
        base = s.chi
        for q in range(width):
            s.clifford_1q("H", q)
        for q in range(width):
            s.t(q, dagger=bool(rng.integers(2)))
        for q in range(width - 1):
            s.cz(q, q + 1)
        for q in range(width):
            s.clifford_1q("H", q)
        peak_in_round = max(c for _, c in s.chi_trace[-(3 * width + 1):])
        for q in range(width):
            s.measure_z(q, rng)
        per_round_peak.append((base, peak_in_round, s.chi))
    return s, per_round_peak


def main():
    print("=" * 72)
    print("1) UNITARY Clifford+T: chi grows toward 2^(#T) (no reduction)")
    print("=" * 72)
    print(f"{'n':>3} {'#T':>3} {'peak_chi':>9} {'2^#T':>8} {'cliff_term_ops':>15}")
    for n in (4, 8, 10):
        s = run_unitary_iqp(n, seed=n)
        c = s.ctr
        print(f"{n:>3} {c.t_gates:>3} {c.peak_chi:>9} {2**c.t_gates:>8} {c.clifford_term_ops:>15}")

    print()
    print("=" * 72)
    print("2) CONVEYOR: chi peaks per round (~2^width) then COLLAPSES on measure")
    print("=" * 72)
    for width, rounds in ((6, 4), (8, 6)):
        s, rounds_info = run_conveyor(width, rounds, seed=width * 10 + rounds)
        total_T = s.ctr.t_gates
        print(f"\nwidth={width}, rounds={rounds}: total T = {total_T}")
        print(f"  {'round':>5} {'chi_start':>9} {'chi_peak':>9} {'chi_after_meas':>14}")
        for i, (b, pk, after) in enumerate(rounds_info):
            print(f"  {i:>5} {b:>9} {pk:>9} {after:>14}")
        print(f"  => overall peak_chi = {s.ctr.peak_chi}  "
              f"(bounded by ONE round: ~2^{width} = {2**width}; "
              f"NOT 2^total_T = 2^{total_T})")
        print(f"  => clifford_term_ops (pure-stab-rank Clifford work) = "
              f"{s.ctr.clifford_term_ops}")

    print()
    print("Takeaway: measurement collapses chi back each round, so peak rank is")
    print("set by per-episode magic, not total -- the composition keeps the rank")
    print("bounded exactly where clifft's measurement reduction shrinks k. The")
    print("clifford_term_ops figure is the per-term Clifford work a pure stab-rank")
    print("backend pays and clifft's frame removes for dormant axes.")

    print()
    print("=" * 72)
    print("3) CH-FORM backend: identical chi, O(n^2)-bits/term store (not 2^n)")
    print("=" * 72)
    print(f"  {'case':>16} {'dense peak_chi':>14} {'chform peak_chi':>16} "
          f"{'bytes/term: dense':>18} {'chform':>9}")
    for width, rounds in ((6, 4), (8, 6)):
        seed = width * 10 + rounds
        sd, _ = run_conveyor(width, rounds, seed=seed, backend="dense")
        sc, _ = run_conveyor(width, rounds, seed=seed, backend="chform")
        # per-term storage: dense = 2^n complex128 (16 B each);
        # CH-form = F,G,M (n^2 uint8 each) + gamma,v,s (n int) + omega.
        dense_bytes = (2 ** width) * 16
        chform_bytes = 3 * width * width + 3 * width + 16
        assert sd.ctr.peak_chi == sc.ctr.peak_chi, "backend chi mismatch!"
        print(f"  conveyor w{width}_r{rounds} {sd.ctr.peak_chi:>14} "
              f"{sc.ctr.peak_chi:>16} {dense_bytes:>18} {chform_bytes:>9}")
    print()
    print("Same chi (the cost driver is storage-independent), but each term is a")
    print("CH-form tableau: 2^n complex amplitudes -> O(n^2) bits. Validated to")
    print("machine precision vs the dense oracle and clifft (see validate.py).")


if __name__ == "__main__":
    main()
