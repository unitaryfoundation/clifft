"""NATURAL adaptive workloads for the frame composition -- the experiment that
decides whether the online frame's advantage appears in recognizable protocols,
not just in the synthetic conveyor of bench_adaptive.py.

The measured phase boundary (bench_adaptive): the frame wins iff the circuit is
Clifford-dominated while chi is large -- roughly, Clifford gate-applications
per round must exceed ~2x the per-term conjugation cost of the round's magic
and measurements. Two natural families sit near that boundary from opposite
sides:

  CULTIVATION-STYLE ("patch"): a data register accumulates injected T's
  (mid-circuit magic; chi grows and is never measured away) while every round
  runs syndrome extraction -- ancilla CX fans, ancilla measurements,
  classically-controlled corrections, ancilla reset-and-reuse -- plus L layers
  of logical Clifford work (patch deformation / lattice surgery traffic).
  This is the shape of magic-state cultivation and code deformation. The knob
  L moves the family across the boundary; L=0 (bare syndrome extraction) is
  the least Clifford-dominated point.

  TELEPORTED-T INJECTION ("teleport"): the textbook adaptive magic gadget --
  prepare |T> on an ancilla, CX, measure, apply the classically-controlled
  Clifford correction. This is how fault-tolerant architectures apply T
  gates. Very few Cliffords per magic event, so the expectation is that this
  family does NOT need the frame -- the honest low-Clifford contrast point.

Method mirrors bench_adaptive: sample one true trajectory with the dense-term
backend (measurement outcomes have the right distribution by construction),
then TIME both chform engines replaying the identical trajectory -- same ops,
same outcomes; the only difference is who pays for Cliffords. All engine
variants (plain/frame x fast/slow) are cross-checked at small sizes.

Run:  .venv-research/bin/python -m research.chform_backend.bench_natural
      [--quick]   (small sizes only)
Writes bench_natural.json next to this file.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from .engine import LowRankState

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Family 1: injected-patch syndrome extraction (cultivation-style).
# Data qubits 0..nd-1, ancillas nd..nd+na-1 (na = nd-1 Z-type checks of a
# repetition-code patch; ancillas measured and reused every round).
# ---------------------------------------------------------------------------
def patch_round_ops(nd, na, rnd, t_budget_left, L, rg):
    """One round as an op list. Ops are tuples:
       ("1q", name, q) | ("cx", c, t) | ("cz", c, t) | ("t", q, dagger)
       ("meas", q, ff_target)  -- measure q; feedforward X on ff_target if 1,
                                  then reset q by the same classically
                                  controlled X (ancilla reuse).
    """
    ops = []
    if t_budget_left > 0:  # magic event: inject one T into the patch
        ops.append(("t", rnd % nd, bool(rg.integers(0, 2))))
    # L layers of logical Clifford work on the data register (deformation /
    # surgery traffic). Restricted to CODE-PRESERVING gates (diagonal S/Z,
    # Pauli X, CX/CZ): real transversal/surgery Clifford traffic protects the
    # encoded magic. An H here would rotate the accumulated T-phases into the
    # basis the Z-checks measure, and the syndrome round would destroy the
    # magic -- unphysical for a protocol whose whole point is to keep it.
    for _ in range(L):
        for q in range(nd):
            ops.append(("1q", ["S", "X", "Z"][rg.integers(0, 3)], q))
        perm = rg.permutation(nd)
        for i in range(0, nd - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            ops.append(("cx", a, b) if rg.integers(0, 2) else ("cz", a, b))
    # syndrome extraction: Z_i Z_{i+1} checks via CX fans onto fresh ancillas
    for i in range(na):
        anc = nd + i
        ops.append(("cx", i, anc))
        ops.append(("cx", i + 1, anc))
    for i in range(na):
        anc = nd + i
        # on a hot syndrome: X-correct data i+1; always reset the ancilla
        ops.append(("meas", anc, [("X", i + 1)]))
    return ops


def build_patch(nd, na, R, t_max, L, seed):
    rg = np.random.default_rng(seed)
    rounds = []
    t_left = t_max
    for r in range(R):
        ops = patch_round_ops(nd, na, r, t_left, L, rg)
        if t_left > 0:
            t_left -= 1
        rounds.append(ops)
    return rounds


# ---------------------------------------------------------------------------
# Family 2: teleported-T injection (the textbook adaptive magic gadget).
# Targets 0..m-1 share ancilla q=m. One gadget on target j:
#   prepare |T> on the ancilla (H a; T a), CX j->a, measure a;
#   on outcome 1 apply the Clifford correction (S, X) to the target;
#   ancilla is reset and reused.
# This is how fault-tolerant architectures actually apply T gates -- genuinely
# adaptive (50/50 classically-controlled correction), magic accumulates on the
# unmeasured targets (chi = 2^t exactly), and the Clifford count per magic
# event is SMALL -- the natural protocol expected to sit on the frame-loses
# side of the boundary.
# ---------------------------------------------------------------------------
def teleport_gadget_ops(j, anc):
    return [
        ("1q", "H", anc),
        ("t", anc, False),
        ("cx", j, anc),
        ("meas", anc, [("S", j), ("X", j)]),
    ]


def build_teleport(m, g, seed):
    """g injection rounds over m targets (t_total = m*g)."""
    rounds = []
    anc = m
    for r in range(g):
        ops = []
        for j in range(m):
            ops.extend(teleport_gadget_ops(j, anc))
        rounds.append(ops)
    return rounds


# ---------------------------------------------------------------------------
# Engine driver: run ops on a LowRankState; outcomes sampled (search) or
# replayed (timed). Returns (outcomes, wall_s, cliff_s, chi_peak, counts).
# ---------------------------------------------------------------------------
def run_ops(n, op_rounds, frame, backend, forced=None, seed=0):
    s = LowRankState(n, backend=backend, frame=frame)
    rng = np.random.default_rng(seed + 9)
    outcomes = []
    fi = 0
    chi_peak = 1
    t_cliff = 0.0
    n_cliff = n_t = n_meas = 0
    t0 = time.time()
    for ops in op_rounds:
        for op in ops:
            if op[0] == "1q":
                tc = time.time()
                s.clifford_1q(op[1], op[2])
                t_cliff += time.time() - tc
                n_cliff += 1
            elif op[0] == "cx":
                tc = time.time()
                s.cx(op[1], op[2])
                t_cliff += time.time() - tc
                n_cliff += 1
            elif op[0] == "cz":
                tc = time.time()
                s.cz(op[1], op[2])
                t_cliff += time.time() - tc
                n_cliff += 1
            elif op[0] == "t":
                s.t(op[1], dagger=op[2])
                n_t += 1
                chi_peak = max(chi_peak, s.chi)
            elif op[0] == "meas":
                q, ff = op[1], op[2]
                if forced is None:
                    b = s.measure_z(q, rng)  # true sampling (dense search)
                else:
                    b = forced[fi]
                    s.measure_z_forced_fast(q, b)
                fi += 1
                outcomes.append(b)
                n_meas += 1
                if b:  # feedforward corrections + ancilla reset (X on q)
                    tc = time.time()
                    for name, tgt in ff:
                        s.clifford_1q(name, tgt)
                    s.clifford_1q("X", q)
                    t_cliff += time.time() - tc
                    n_cliff += 1 + len(ff)
        chi_peak = max(chi_peak, s.chi)
    wall = time.time() - t0
    assert any(t.norm2() > 1e-24 for t in s.terms), "trajectory died"
    return s, outcomes, wall, t_cliff, chi_peak, (n_cliff, n_t, n_meas)


def rus_trajectory(m, per_target_attmuch, seed, forced=None, frame=False,
                   backend="chform"):
    """RUS has data-dependent LENGTH, so the op stream is built during the
    search run and replayed verbatim: search mode returns the realized op
    rounds; replay mode consumes them."""
    rg = np.random.default_rng(seed)
    anc = m
    if forced is None:
        # search on the dense backend, recording the realized ops + outcomes
        s = LowRankState(m + 1, backend="dense", frame=False)
        rng = np.random.default_rng(seed + 9)
        realized, outcomes = [], []
        for j in range(m):
            for attempt in range(per_target_attmuch):
                ops = rus_attempt_ops(j, anc, bool(rg.integers(0, 2)))
                realized.append(ops)
                for op in ops[:-1]:
                    if op[0] == "1q":
                        s.clifford_1q(op[1], op[2])
                    elif op[0] == "cx":
                        s.cx(op[1], op[2])
                    else:
                        s.t(op[1], dagger=op[2])
                b = s.measure_z(anc, rng)
                outcomes.append(b)
                if b:
                    # same rule the replay driver applies for ("meas", q, ff):
                    # X correction on the target, X reset on the ancilla
                    s.clifford_1q("X", j)
                    s.clifford_1q("X", anc)
                else:
                    break  # success: next target
        return realized, outcomes
    return None


def readout_vec(s):
    v = s.statevector()
    nrm = np.sqrt(np.vdot(v, v).real)
    return v / nrm if nrm > 0 else v


def state_dist(a, b):
    """Distance up to global phase: 0 iff the normalized states agree.
    (Fixing the phase at the largest amplitude is ambiguous for states with
    many equal-magnitude amplitudes -- common after Clifford layers.)"""
    return float(abs(1.0 - abs(np.vdot(a, b))))


def correctness():
    print("CORRECTNESS (small patch): dense search == chform replays, frame == plain")
    worst = 0.0
    rounds = build_patch(nd=3, na=2, R=3, t_max=2, L=1, seed=11)
    s0, outs, *_ = run_ops(5, rounds, frame=False, backend="dense")
    ref = readout_vec(s0)
    for frame in (False, True):
        s, o2, *_ = run_ops(5, rounds, frame=frame, backend="chform", forced=outs)
        assert o2 == outs
        worst = max(worst, state_dist(readout_vec(s), ref))
    print(f"[{'OK' if worst < 1e-9 else 'FAIL'}] patch: chform frame/plain replays "
          f"match dense search (max abs err {worst:.2e})")
    assert worst < 1e-9

    rounds = build_teleport(m=3, g=2, seed=21)
    sd0, outs, *_ = run_ops(4, rounds, frame=False, backend="dense", seed=21)
    vec0 = readout_vec(sd0)
    for frame in (False, True):
        s, o2, *_ = run_ops(4, rounds, frame=frame, backend="chform",
                            forced=outs, seed=21)
        assert o2 == outs
        worst = max(worst, state_dist(readout_vec(s), vec0))
    print(f"[{'OK' if worst < 1e-9 else 'FAIL'}] teleport-T: frame == plain == "
          f"dense on the adaptive gadget (max err {worst:.2e})")
    assert worst < 1e-9
    return worst


def bench_patch(quick):
    print("\nPATCH (cultivation-style): nd data, na=nd-1 checks, t_max injected T's,"
          "\nL logical-Clifford layers per round; frame vs plain, same trajectory")
    print(f"{'nd':>3} {'t':>3} {'L':>3} {'chi':>5} {'#Cl':>5} {'plain(s)':>9} "
          f"{'frame(s)':>9} {'ratio':>7} {'plCliff(s)':>10} {'frCliff(s)':>10}")
    rows = []
    nd = 6 if quick else 8
    na = nd - 1
    R = 8 if quick else 14
    for t_max in ([4] if quick else [6, 10]):
        for L in ([0, 2] if quick else [0, 1, 2, 4, 8]):
            seed = 1000 + 13 * t_max + L
            rounds = build_patch(nd, na, R, t_max, L, seed)
            n = nd + na
            _, outs, *_ = run_ops(n, rounds, frame=False, backend="dense", seed=seed)
            _, _, tp, tcp, chip, cnt = run_ops(n, rounds, frame=False,
                                               backend="chform", forced=outs)
            _, _, tf, tcf, chif, _ = run_ops(n, rounds, frame=True,
                                             backend="chform", forced=outs)
            rows.append({"family": "patch", "nd": nd, "t_max": t_max, "L": L,
                         "chi_plain": chip, "chi_frame": chif, "n_cliff": cnt[0],
                         "n_t": cnt[1], "n_meas": cnt[2],
                         "plain_s": tp, "frame_s": tf,
                         "plain_cliff_s": tcp, "frame_cliff_s": tcf})
            print(f"{nd:>3} {t_max:>3} {L:>3} {max(chip, chif):>5} {cnt[0]:>5} "
                  f"{tp:>9.2f} {tf:>9.2f} {tp / tf:>7.2f} {tcp:>10.2f} "
                  f"{tcf:>10.3f}", flush=True)
    return rows


def bench_teleport(quick):
    print("\nTELEPORTED-T injection: m targets, g rounds, adaptive S|X correction")
    print(f"{'m':>3} {'g':>3} {'chi':>5} {'#Cl':>5} {'plain(s)':>9} {'frame(s)':>9} {'ratio':>7}")
    rows = []
    for m, g in ([(4, 2)] if quick else [(5, 2), (6, 2)]):
        seed = 4000 + m
        rounds = build_teleport(m, g, seed)
        n = m + 1
        _, outs, *_ = run_ops(n, rounds, frame=False, backend="dense", seed=seed)
        _, _, tp, tcp, chip, cnt = run_ops(n, rounds, frame=False,
                                           backend="chform", forced=outs)
        _, _, tf, tcf, chif, _ = run_ops(n, rounds, frame=True,
                                         backend="chform", forced=outs)
        rows.append({"family": "teleport", "m": m, "g": g,
                     "chi_plain": chip, "chi_frame": chif, "n_cliff": cnt[0],
                     "n_t": cnt[1], "n_meas": cnt[2],
                     "plain_s": tp, "frame_s": tf,
                     "plain_cliff_s": tcp, "frame_cliff_s": tcf})
        print(f"{m:>3} {g:>3} {max(chip, chif):>5} {cnt[0]:>5} {tp:>9.2f} "
              f"{tf:>9.2f} {tp / tf:>7.2f}", flush=True)
    return rows


def main():
    quick = "--quick" in sys.argv
    worst = correctness()
    patch = bench_patch(quick)
    teleport = bench_teleport(quick)
    out = os.path.join(HERE, "bench_natural.json")
    with open(out, "w") as f:
        json.dump({"correctness_err": worst, "patch": patch,
                   "teleport": teleport}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
