"""Scale benchmark for gadgetized general Clifford+T strong simulation
(the C++ arm; math + Python reference in gadgetize.py).

Three parts:
  1. Cross-validation: the C++ gadget binary against the dense reference at
     small n (TV at the sparsification scale) -- guards the spec plumbing.
  2. HIDDEN SHIFT at scale: the circuit's output is |s> deterministically, so
     P(s) = 1 is exact ground truth at ANY size with no reference simulator --
     the classic stabilizer-rank benchmark family, now with interleaved magic
     handled by gadgetization + single-shot sampling (no streaming t-factor).
     Note clifft itself compiles hidden shift to a tiny active block (its
     measurement-driven reduction is strong on this family) -- the point here
     is validating OUR generality at scale, not beating clifft on its own turf.
  3. RANDOM Clifford+T (interleaved magic): head-to-head against clifft's
     exact record_probabilities on the measured circuit, where clifft's
     compiled peak_rank is the honest cost driver.

Run:  .venv-research/bin/python -m research.chform_backend.bench_general
Writes bench_general.json next to this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

import numpy as np

from .gadgetize import (count_t, dense_reference, hidden_shift,
                        random_cliffordT)
from .hir_bridge import optimize as hir_optimize

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = "/tmp/chf_gadget"


def build_binary():
    src = os.path.join(HERE, "..", "chform_cpp", "bench_gadget.cpp")
    subprocess.run(["clang++", "-std=c++20", "-O3", "-o", BIN, src], check=True)


def run_gadget(n, ops, k, targets, seed):
    t = count_t(ops)
    daggers = "".join("1" if op[0] == "T_DAG" else "0"
                      for op in ops if op[0] in ("T", "T_DAG"))
    with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
        f.write(f"{n} {t} {k} {seed}\n{daggers}\n{len(ops)}\n")
        for op in ops:
            if op[0] in ("CX", "CZ"):
                f.write(f"{op[0]} {op[1]} {op[2]}\n")
            elif op[0] in ("T", "T_DAG"):
                f.write(f"T {op[1]}\n")   # dagger carried by the pattern
            elif op[0] == "S_DAG":
                f.write(f"SDG {op[1]}\n")
            else:
                f.write(f"{op[0]} {op[1]}\n")
        f.write(f"{len(targets)}\n")
        for x in targets:
            f.write(f"{int(x)}\n")
        spec = f.name
    out = subprocess.run([BIN, spec], capture_output=True, text=True, check=True).stdout
    os.unlink(spec)
    P, kv = {}, {}
    for line in out.splitlines():
        p = line.split()
        if p[0] == "P":
            P[int(p[1])] = float(p[2])
        else:
            kv[p[0]] = float(p[1])
    return P, kv


def budget(t, delta):
    return max(2000, int(2.0 ** (0.228 * t) / delta ** 2))


def crossvalidate():
    print("CROSS-VALIDATION: C++ gadget binary vs dense reference (small n)")
    worst = 0.0
    for n, depth, tt, seed in ((6, 4, 8, 1), (8, 5, 12, 2)):
        ops = random_cliffordT(n, depth, tt, seed)
        vd = np.abs(dense_reference(n, ops)) ** 2
        k = budget(tt, 0.05)
        P, _ = run_gadget(n, ops, k, list(range(2 ** n)), seed=99)
        tv = 0.5 * sum(abs(P[x] - vd[x]) for x in range(2 ** n))
        print(f"  n={n} t={tt} k={k}: TV vs dense = {tv:.4f}")
        worst = max(worst, tv)
    assert worst < 0.05, worst
    print(f"[OK] C++ gadget pipeline agrees with dense (worst TV {worst:.4f} "
          f"at delta=0.05)")


def bench_hidden_shift():
    print("\nHIDDEN SHIFT at scale: exact answer P(s)=1 built into the family")
    print(f"{'n':>3} {'t':>3} {'N':>4} {'k':>7} {'P(s)':>7} {'bg P(x!=s)':>11} "
          f"{'time(s)':>8} {'clifft k/t(s)':>14}")
    rows = []
    for n, n_ccz in ((16, 2), (24, 3), (32, 4), (40, 4)):
        ops, s = hidden_shift(n, n_ccz, seed=50 + n)
        tt = count_t(ops)
        delta = 0.3
        k = budget(tt, delta)
        rng = np.random.default_rng(5)
        others = [int(x) for x in rng.integers(0, 2 ** n, size=8) if int(x) != s]
        t0 = time.time()
        P, kv = run_gadget(n, ops, k, [s] + others, seed=7 + n)
        dt = time.time() - t0
        # clifft exact on the measured circuit, for context (expected: tiny k)
        import clifft
        lines = []
        for op in ops:
            if op[0] in ("CX", "CZ"):
                lines.append(f"{op[0]} {op[1]} {op[2]}")
            else:
                lines.append(f"{op[0]} {op[1]}")
        lines += [f"M {q}" for q in range(n)]
        prog = clifft.compile("\n".join(lines))
        rec = np.array([[(s >> q) & 1 for q in range(n)]], dtype=bool)
        t0 = time.time()
        pc = float(np.asarray(clifft.record_probabilities(prog, rec))[0])
        tclifft = time.time() - t0
        assert abs(pc - 1.0) < 1e-9  # sanity: the construction is right
        bg = float(np.mean([P[x] for x in others]))
        rows.append({"n": n, "t": tt, "k": k, "P_s": P[s], "bg": bg,
                     "time_s": dt, "clifft_peak_rank": prog.peak_rank,
                     "clifft_s": tclifft})
        print(f"{n:>3} {tt:>3} {n + tt:>4} {k:>7} {P[s]:>7.3f} {bg:>11.2e} "
              f"{dt:>8.1f} {prog.peak_rank:>6}/{tclifft:>6.3f}", flush=True)
    return rows


def bench_random():
    print("\nRANDOM Clifford+T (interleaved magic) vs clifft exact "
          "(record_probabilities, measured circuit)")
    print(f"{'n':>3} {'traw':>4} {'tlive':>5} {'kmeas':>5} {'clifft(s)':>10} "
          f"{'ours(s)':>8} {'TV':>7} {'k':>7}")
    import clifft
    rows = []
    for n, depth, tt in ((20, 6, 32), (26, 6, 40), (30, 6, 48)):
        ops = random_cliffordT(n, depth, tt, seed=800 + n)
        lines = []
        for op in ops:
            if op[0] in ("CX", "CZ"):
                lines.append(f"{op[0]} {op[1]} {op[2]}")
            else:
                lines.append(f"{op[0]} {op[1]}")
        lines += [f"M {q}" for q in range(n)]
        prog = clifft.compile("\n".join(lines))
        # Targets = outcomes clifft actually observes (unique records from a
        # short sampling run): the likelihood use-case, and it guarantees the
        # exact subset mass is nonzero (uniform targets all miss the support
        # when the compiled program's output lives on a small affine subspace).
        samp = clifft.sample(prog, shots=256)
        recs_all = np.asarray(samp.measurements, dtype=bool)
        uniq = sorted(set(tuple(r) for r in recs_all.tolist()))[:64]
        recs = np.array(uniq, dtype=bool)
        targets = [sum(int(b) << q for q, b in enumerate(r)) for r in uniq]
        t0 = time.time()
        pc = np.asarray(clifft.record_probabilities(prog, recs))
        tclifft = time.time() - t0
        delta = 0.15
        k = budget(tt, delta)
        t0 = time.time()
        P, kv = run_gadget(n, ops, k, targets, seed=13 + n)
        dt = time.time() - t0
        denom = float(np.sum(pc))
        tv = 0.5 * sum(abs(P[x] - p) for x, p in zip(targets, pc)) / denom
        # The compile-time decision rule uses the LIVE T-count after clifft's
        # HIR optimization (t_live <= t_raw; using t_raw is biased against the
        # backend). Both exponents come from the same compilation.
        _, t_raw, t_live = hir_optimize("\n".join(lines))
        rows.append({"n": n, "t_raw": tt, "t_live": t_live,
                     "clifft_peak_rank": prog.peak_rank,
                     "clifft_s": tclifft, "ours_s": dt, "tv": tv, "k": k})
        rule = "backend" if 0.228 * t_live < prog.peak_rank else "clifft"
        print(f"{n:>3} {tt:>4} {t_live:>5} {prog.peak_rank:>5} {tclifft:>10.3f} "
              f"{dt:>8.1f} {tv:>7.3f} {k:>7}   0.228*tlive={0.228 * t_live:5.1f} "
              f"vs k={prog.peak_rank} -> {rule}", flush=True)
    return rows


def main():
    build_binary()
    crossvalidate()
    hs = bench_hidden_shift()
    rc = bench_random()
    out = os.path.join(HERE, "bench_general.json")
    with open(out, "w") as f:
        json.dump({"hidden_shift": hs, "random_cliffordT": rc}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
