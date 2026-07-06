"""The adaptive/mid-circuit workload -- the composition's defensible claim,
measured in WALL-CLOCK (the earlier conveyor result was op-counts only).

Workload: an adaptive Clifford-dominated computation on a width-w register,
R rounds of
    H^w ; feedforward-X (previous round's outcomes) ; T^{+-} layer ;
    [D global Clifford scrambling layers, applied while chi = 2^w] ;
    CZ chain ; H^w ; measure all w qubits
The scrambling layers sit BETWEEN the magic and the measurements -- where the
term count is at its peak -- because that is where a per-term simulator pays
chi * (gate cost) and a frame pays O(n). (Cliffords applied while chi ~ 1 are
cheap for everyone; realistic adaptive circuits -- syndrome extraction,
teleportation networks -- interleave Cliffords with live magic exactly so.)
The mid-circuit measurements + classically-controlled corrections make this a
DYNAMIC circuit: static strong-simulation pipelines (gadgetized-amplitude
methods a la Pashayan et al., per-outcome ZX/WMC runs) must either postselect
gadgets or unroll outcome branches; a sampling simulator must follow the
trajectory online, which is exactly what this engine and qiskit-aer's
extended_stabilizer do.

Contestants (identical trajectories, forced to the same outcome bits, same
CH-form term store, same measurement code -- the ONLY difference is who pays
for Cliffords):
  plain : every Clifford applied to all chi terms -- the architecture of
          qiskit-aer's extended_stabilizer (verified per-term in its source).
  frame : Cliffords absorbed into the global O(n)-per-gate frame; only magic
          and measurement touch the terms.
Both use measure_z_forced_fast (no dense materialization, no normalization --
shared and equal). Expected: t_plain - t_frame grows ~ chi * (Clifford gates),
i.e. the frame advantage is the per-term Clifford work, now in seconds rather
than op-counts.

External arm: we attempt the same adaptive circuit on qiskit-aer
extended_stabilizer (mid-circuit measurement + if_test feedforward). If the
backend rejects dynamic circuits, that is itself the finding (the CH-form
incumbent cannot enter this workload); if it runs, we report its wall-clock
with the language caveat (aer is C++, this engine is Python -- compare
scaling in chi, not absolute constants).

Run:  .venv-research/bin/python -m research.chform_backend.bench_adaptive
Writes bench_adaptive.json next to this file.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from .engine import LowRankState

HERE = os.path.dirname(os.path.abspath(__file__))


def scramble_layers(w, D, rg):
    """D layers: random 1q Cliffords on every qubit + a random 2q pairing."""
    layers = []
    for _ in range(D):
        ops = [("1q", ["H", "S", "X", "Z"][rg.integers(0, 4)], q) for q in range(w)]
        perm = rg.permutation(w)
        for i in range(0, w - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            ops.append(("2q", ["CX", "CZ"][rg.integers(0, 2)], (a, b)))
        layers.append(ops)
    return layers


def build_program(w, R, D, seed):
    """The full adaptive program, generated once so every contestant runs the
    identical circuit: per round (scramble, dag pattern)."""
    rg = np.random.default_rng(seed)
    return [
        {"scramble": scramble_layers(w, D, rg),
         "dag": [bool(b) for b in rg.integers(0, 2, size=w)]}
        for _ in range(R)
    ]


def run_engine(w, R, D, seed, frame, fast=True, forced=None):
    """forced=None: force the pseudo-random trajectory, flipping any bit that
    lands on a zero-support branch (plain reference run). forced=<list>: replay
    exactly that outcome sequence (frame runs replay the plain run's
    trajectory, so both engines do identical work up to who pays Cliffords).
    After each fully-measured round the state is provably rank 1 and both
    engines collapse to one term (collapse_to_rank1) -- without it, terms
    proportional to the same state but with different tableaux compound chi
    across rounds."""
    prog = build_program(w, R, D, seed)
    want = list(forced) if forced is not None else None
    backend = "chform" if forced is not None else "dense"
    s = LowRankState(w, backend=backend, frame=frame)
    rng = np.random.default_rng(seed + 9)
    t_cliff = 0.0
    chi_peak = 1
    used = []
    t0 = time.time()
    fi = 0
    prev = [0] * w
    for rnd in prog:
        for q in range(w):
            s.clifford_1q("H", q)
        for q in range(w):                      # feedforward (adaptive part)
            if prev[q]:
                s.clifford_1q("X", (q + 1) % w)
        for q in range(w):
            s.t(q, dagger=rnd["dag"][q])
        tc = time.time()                        # Cliffords at peak chi
        for layer in rnd["scramble"]:
            for kind, name, qs in layer:
                if kind == "1q":
                    s.clifford_1q(name, qs)
                elif name == "CX":
                    s.cx(*qs)
                else:
                    s.cz(*qs)
        for q in range(w - 1):
            s.cz(q, q + 1)
        for q in range(w):
            s.clifford_1q("H", q)
        t_cliff += time.time() - tc
        chi_peak = max(chi_peak, s.chi)
        prev = []
        for q in range(w):
            if forced is None:
                # TRAJECTORY SEARCH (untimed): true sampling on the dense-term
                # backend -- 2^w vectors are tiny at benchmark widths, and the
                # sampled trajectory has nonzero probability BY CONSTRUCTION.
                # (Forcing pseudo-random bits instead dies: a round's outcome
                # support can be an exponentially small affine subspace, which
                # blind redraws never hit.)
                b = s.measure_z(q, rng)
            else:
                b = want[fi]
                if fast:
                    s.measure_z_forced_fast(q, b)
                else:
                    s.measure_z(q, rng, force=b)
            fi += 1
            used.append(b)
            prev.append(b)
        if fast and forced is not None:
            s.collapse_to_rank1()   # all w qubits just measured => rank 1
        assert any(t.norm2() > 1e-24 for t in s.terms), \
            "trajectory died on replay (should be impossible)"
    return s, time.time() - t0, t_cliff, chi_peak, used


def readout_vec(s):
    v = s.statevector()
    n = np.sqrt(np.vdot(v, v).real)
    if n > 0:
        v = v / n
    k = int(np.argmax(np.abs(v)))
    return v * np.exp(-1j * np.angle(v[k]))


def correctness():
    print("CORRECTNESS (small w): fast forced path == validated slow path; frame == plain")
    worst = 0.0
    for w, R, D in ((3, 2, 1), (4, 2, 2), (4, 3, 1)):
        s0, _, _, _, outs = run_engine(w, R, D, seed=w * 100 + R, frame=False)
        vecs = [readout_vec(s0)]
        for frame, fast in ((False, False), (True, True), (True, False)):
            s, _, _, _, _ = run_engine(w, R, D, seed=w * 100 + R, frame=frame,
                                       fast=fast, forced=outs)
            vecs.append(readout_vec(s))
        for v in vecs[1:]:
            worst = max(worst, float(np.max(np.abs(v - vecs[0]))))
    print(f"[{'OK' if worst < 1e-9 else 'FAIL'}] 4 engine variants agree "
          f"(max abs err {worst:.2e})")
    assert worst < 1e-9
    return worst


def wallclock():
    print(f"\nWALL-CLOCK (R=4, D=4): the frame advantage in seconds")
    print(f"{'w':>3} {'chi_peak':>8} {'plain(s)':>9} {'frame(s)':>9} {'ratio':>7} "
          f"{'plainCliff(s)':>13} {'frameCliff(s)':>13}")
    rows = []
    for w in (4, 6, 8, 10, 12):
        # pass 1 (untimed): find a live trajectory; passes 2-3: timed replays
        _, _, _, _, outs = run_engine(w, 4, 4, seed=w * 100 + 4, frame=False)
        _, tp, tcp, chip, _ = run_engine(w, 4, 4, seed=w * 100 + 4, frame=False,
                                         forced=outs)
        _, tf, tcf, chif, _ = run_engine(w, 4, 4, seed=w * 100 + 4, frame=True,
                                         forced=outs)
        # chi trajectories may differ slightly (the definite-basis T shortcut
        # fires differently under frame conjugation); report both.
        rows.append({"w": w, "chi_peak_plain": chip, "chi_peak_frame": chif,
                     "plain_s": tp, "frame_s": tf,
                     "plain_cliff_s": tcp, "frame_cliff_s": tcf})
        print(f"{w:>3} {max(chip, chif):>8} {tp:>9.2f} {tf:>9.2f} {tp / tf:>7.1f} "
              f"{tcp:>13.2f} {tcf:>13.3f}", flush=True)
    return rows


def depth_sweep():
    """The phase boundary: at fixed width, sweep the Clifford scrambling depth
    D. Plain time grows linearly in D (chi per-term gate applications); frame
    time is flat in D (absorption is O(n) per gate) -- but the frame pays a
    D-INDEPENDENT per-term overhead on magic and measurement (the conjugated
    Pauli P' = F^-1 Z F has weight ~w, so each T/measurement costs O(w) gate
    applications per term). The composition wins iff the circuit is
    sufficiently Clifford-dominated: measured crossover D* below."""
    w, R = 8, 4
    print(f"\nDEPTH SWEEP (w={w}, R={R}): frame wins iff Clifford-dominated")
    print(f"{'D':>4} {'plain(s)':>9} {'frame(s)':>9} {'ratio':>7}")
    rows = []
    for D in (4, 8, 16, 32, 64, 96):
        seed = w * 100 + D
        _, _, _, _, outs = run_engine(w, R, D, seed=seed, frame=False)
        _, tp, _, _, _ = run_engine(w, R, D, seed=seed, frame=False, forced=outs)
        _, tf, _, _, _ = run_engine(w, R, D, seed=seed, frame=True, forced=outs)
        rows.append({"D": D, "plain_s": tp, "frame_s": tf})
        print(f"{D:>4} {tp:>9.2f} {tf:>9.2f} {tp / tf:>7.2f}", flush=True)
    return rows


def aer_arm():
    """Same adaptive circuit on qiskit-aer extended_stabilizer (1 shot)."""
    print("\nAER extended_stabilizer on the same adaptive circuit (shots=1):")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    results = {}
    for w in (4, 6, 8, 10, 12):
        prog = build_program(w, 4, 4, seed=w * 100 + 4)
        qc = QuantumCircuit(w, w * len(prog))
        ci = 0
        for r, rnd in enumerate(prog):
            for q in range(w):
                qc.h(q)
            if r > 0:  # feedforward on previous round's clbits
                for q in range(w):
                    with qc.if_test((qc.clbits[(r - 1) * w + q], 1)):
                        qc.x((q + 1) % w)
            for q in range(w):
                (qc.tdg if rnd["dag"][q] else qc.t)(q)
            for layer in rnd["scramble"]:
                for kind, name, qs in layer:
                    if kind == "1q":
                        getattr(qc, name.lower())(qs)
                    elif name == "CX":
                        qc.cx(*qs)
                    else:
                        qc.cz(*qs)
            for q in range(w - 1):
                qc.cz(q, q + 1)
            for q in range(w):
                qc.h(q)
            for q in range(w):
                qc.measure(q, ci); ci += 1
        sim = AerSimulator(method="extended_stabilizer",
                           extended_stabilizer_metropolis_mixing_time=50)
        t0 = time.time()
        try:
            res = sim.run(qc, shots=1).result()
            ok = res.success
            err = "" if ok else str(res.status)
        except Exception as e:  # noqa: BLE001 -- report whatever aer raises
            ok, err = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        results[str(w)] = {"success": ok, "time_s": dt, "error": err[:300]}
        print(f"  w={w}: {'OK' if ok else 'REJECTED/FAILED'} in {dt:.2f}s {err[:120]}",
              flush=True)
    return results


def main():
    worst = correctness()
    rows = wallclock()
    depth = depth_sweep()
    aer = aer_arm()
    out = os.path.join(HERE, "bench_adaptive.json")
    with open(out, "w") as f:
        json.dump({"correctness_err": worst, "rows": rows, "depth": depth,
                   "aer": aer}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
