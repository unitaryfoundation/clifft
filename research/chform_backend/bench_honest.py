"""The HONEST head-to-head: CH-form backend vs clifft's real P(x) fast path.

This replaces bench_overlap.py's comparison, which had two flaws (found in
review):

  1. WRONG clifft BASELINE. It timed `basis_probabilities` on the *unitary*
     program (peak_rank = n). A user who wants P(x) compiles the *measured*
     program and calls `record_probabilities` -- clifft's exact forced-replay
     API. On bench_overlap's nearest-neighbour-CZ IQP the measured program
     compiles to peak_rank = 1 (lightcone sweep) and clifft answers the same
     96 exact probabilities ~1e5x faster than the number the old benchmark
     charged it. Here we (a) use `record_probabilities` as the baseline and
     (b) use DENSE random IQP (CZ w.p. 1/2, the standard hard ensemble), where
     the measured program compiles to peak_rank = n/2 -- so the baseline is
     exponential but honestly so (2^{n/2}, not 2^n).

  2. UNFAIR ERROR METRIC. Its "norm-free TV" renormalized our P over the
     target subset, exactly cancelling the norm-estimation error (11-18%
     relative at the budgets quoted). Here the primary metric charges the
     full pipeline:  TV = 0.5 * sum_x |P_hat(x) - P(x)| / sum_x P(x)
     (only the *exact* subset mass in the denominator, nothing of ours), and
     we report the norm-estimation error separately.

Ground truth is the exact O(n 2^{n/2}) meet-in-the-middle tool
(chform_cpp/mitm_iqp.cpp) -- which doubles as the yardstick for "hard":
this family costs 2^{n/2} EXACTLY for clifft *and* for MitM, so the CH-form
backend's asymptotic edge (2^{0.228n}) is real, but the crossover must be
measured against 2^{n/2}, not against the unitary path's 2^n.

Backend modes per (n, delta):
  - "analytic":  samples=0, E||omega||^2 normalization (valid: unitary product
                 magic). The full pipeline error is sparsification only.
  - "est-L30":   samples=30 norm estimation, as the old report ran it -- shows
                 the norm-error contribution the old metric hid.

Run:  .venv-research/bin/python -m research.chform_backend.bench_honest
      [--quick]  (n <= 40 only, 1 rep -- a few minutes)
Writes bench_honest.json next to this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(HERE, "..", "chform_cpp")
OVERLAP = "/tmp/chf_overlap_h"
MITM = "/tmp/chf_mitm_h"

NTARGET = 96
DELTAS = (0.4, 0.15)
PRECISION_N = 44                      # the delta-dial demo
PRECISION_DELTAS = (0.3, 0.2, 0.1, 0.05)
CLIFFT_MAX_STATE_BYTES = 5 * 2**30    # skip the exact baseline past ~5 GB/replay


def build_binaries():
    for out, src in ((OVERLAP, "bench_overlap.cpp"), (MITM, "mitm_iqp.cpp")):
        subprocess.run(
            ["clang++", "-std=c++20", "-O3", "-o", out, os.path.join(CPP_DIR, src)],
            check=True,
        )


def make_dense_iqp(n, seed):
    """Dense random IQP: one T/T_dag per qubit, CZ on each pair w.p. 1/2."""
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(a, b) for a in range(n) for b in range(a + 1, n) if r.random() < 0.5]
    return dag, czs


def write_spec(n, k, samples, dag, czs, targets):
    f = tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False)
    f.write(f"{n} {k} {samples}\n")
    f.write("".join("1" if d else "0" for d in dag) + "\n")
    f.write(f"{len(czs)}\n")
    for a, b in czs:
        f.write(f"{a} {b}\n")
    f.write(f"{len(targets)}\n")
    for x in targets:
        f.write(f"{int(x)}\n")
    f.close()
    return f.name


def parse_kv(out):
    kv, P = {}, {}
    for line in out.splitlines():
        p = line.split()
        if p[0] == "P":
            P[int(p[1])] = float(p[2])
        elif len(p) == 2:
            try:
                kv[p[0]] = float(p[1])
            except ValueError:
                kv[p[0]] = p[1]
    return kv, P


def run_mitm(n, dag, czs, targets):
    spec = write_spec(n, 0, 0, dag, czs, targets)
    out = subprocess.run([MITM, spec], capture_output=True, text=True, check=True).stdout
    os.unlink(spec)
    kv, P = parse_kv(out)
    return P, kv.get("table_s", 0.0) + kv.get("targets_s", 0.0)


def run_backend(n, delta, samples, dag, czs, targets, seed):
    k = max(2000, int(2.0 ** (0.228 * n) / delta**2))
    spec = write_spec(n, k, samples, dag, czs, targets)
    out = subprocess.run(
        [OVERLAP, spec, str(seed)], capture_output=True, text=True, check=True
    ).stdout
    os.unlink(spec)
    kv, P = parse_kv(out)
    return {
        "k": k,
        "P": P,
        "build_s": kv["build_s"],
        "norm_s": kv["norm_s"],
        "amps_s": kv["amps_s"],
        "total_s": kv["total_s"],
        "norm2": kv["norm2"],
    }


def tv_honest(targets, P_hat, P_exact):
    """0.5 * sum|P_hat - P| / sum P  -- full pipeline error, exact-mass scale."""
    denom = sum(P_exact[x] for x in targets)
    return 0.5 * sum(abs(P_hat[x] - P_exact[x]) for x in targets) / denom


def tv_normfree(targets, P_hat, P_exact):
    """The OLD metric (both sides renormalized over the subset) -- reported
    only to show how much error it hides."""
    so = sum(P_hat[x] for x in targets) or 1.0
    sc = sum(P_exact[x] for x in targets) or 1.0
    return 0.5 * sum(abs(P_hat[x] / so - P_exact[x] / sc) for x in targets)


def clifft_baseline(n, dag, czs, targets, max_targets=None):
    import clifft

    lines = [f"H {q}" for q in range(n)]
    lines += [f"{'T_DAG' if dag[q] else 'T'} {q}" for q in range(n)]
    lines += [f"CZ {a} {b}" for a, b in czs]
    lines += [f"H {q}" for q in range(n)]
    lines += [f"M {q}" for q in range(n)]
    t0 = time.time()
    prog = clifft.compile("\n".join(lines))
    t_compile = time.time() - t0

    run_targets = targets if max_targets is None else targets[:max_targets]
    records = np.array(
        [[(x >> q) & 1 for q in range(n)] for x in run_targets], dtype=bool
    )
    t0 = time.time()
    probs = np.asarray(clifft.record_probabilities(prog, records))
    t_query = time.time() - t0
    scaled = t_query * (len(targets) / len(run_targets))
    return {
        "peak_rank": prog.peak_rank,
        "compile_s": t_compile,
        "query_s": scaled,
        "query_measured_s": t_query,
        "n_targets_run": len(run_targets),
        "P": {int(x): float(p) for x, p in zip(run_targets, probs)},
    }


def one_point(n, rep, deltas=DELTAS):
    seed = 1000 + 17 * n + rep
    dag, czs = make_dense_iqp(n, seed)
    rng_t = np.random.default_rng(seed + 1)
    targets = sorted(set(int(x) for x in rng_t.integers(0, 2**n, size=NTARGET)))

    P_exact, t_mitm = run_mitm(n, dag, czs, targets)

    row = {"n": n, "rep": rep, "ncz": len(czs), "mitm_s": t_mitm, "backend": {}}

    state_bytes = 16 * 2 ** (n - n // 2)  # clifft's measured block: peak_rank ~ ceil(n/2)
    if state_bytes <= CLIFFT_MAX_STATE_BYTES:
        cb = clifft_baseline(n, dag, czs, targets)
        rel = max(
            abs(cb["P"][x] - P_exact[x]) / max(P_exact[x], 1e-300) for x in cb["P"]
        )
        assert rel < 1e-6, f"clifft vs MitM mismatch at n={n}: {rel}"
        row["clifft"] = {k: v for k, v in cb.items() if k != "P"}
        row["clifft"]["max_rel_vs_mitm"] = rel
    else:
        row["clifft"] = None  # exact baseline infeasible at this size (memory)

    for delta in deltas:
        for mode, samples in (("analytic", 0), ("est-L30", 30)):
            if mode == "est-L30" and delta != 0.4:
                continue  # norm-estimation cost scales as chi*L; one budget suffices
            r = run_backend(n, delta, samples, dag, czs, targets, seed + 2)
            row["backend"][f"d{delta}-{mode}"] = {
                "k": r["k"],
                "total_s": r["total_s"],
                "build_s": r["build_s"],
                "norm_s": r["norm_s"],
                "amps_s": r["amps_s"],
                "norm2": r["norm2"],
                "tv": tv_honest(targets, r["P"], P_exact),
                "tv_normfree_OLD": tv_normfree(targets, r["P"], P_exact),
                "max_rel": max(
                    abs(r["P"][x] - P_exact[x]) / max(P_exact[x], 1e-300)
                    for x in targets
                ),
            }
    return row


def main():
    quick = "--quick" in sys.argv
    build_binaries()
    ns = [24, 28, 32, 36, 40] if quick else [24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    rows = []
    print(
        f"{'n':>3} {'rep':>3} {'k_meas':>6} {'clifft(s)':>10} {'mitm(s)':>8} "
        f"{'ours d=.4(s)':>12} {'TV.4':>7} {'ours d=.15(s)':>13} {'TV.15':>7} {'TVest.4':>8}"
    )
    for n in ns:
        reps = 1 if (quick or n >= 52) else 3
        for rep in range(reps):
            row = one_point(n, rep)
            rows.append(row)
            c = row["clifft"]
            b4 = row["backend"]["d0.4-analytic"]
            b15 = row["backend"]["d0.15-analytic"]
            be = row["backend"]["d0.4-est-L30"]
            print(
                f"{n:>3} {rep:>3} {(c or {}).get('peak_rank', -1):>6} "
                f"{(c or {}).get('query_s', float('nan')):>10.3f} {row['mitm_s']:>8.1f} "
                f"{b4['total_s']:>12.2f} {b4['tv']:>7.3f} "
                f"{b15['total_s']:>13.2f} {b15['tv']:>7.3f} {be['tv']:>8.3f}",
                flush=True,
            )

    # precision dial at fixed n (analytic mode: full-pipeline TV vs delta)
    dial = []
    if not quick:
        n = PRECISION_N
        seed = 5000 + n
        dag, czs = make_dense_iqp(n, seed)
        rng_t = np.random.default_rng(seed + 1)
        targets = sorted(set(int(x) for x in rng_t.integers(0, 2**n, size=256)))
        P_exact, _ = run_mitm(n, dag, czs, targets)
        for delta in PRECISION_DELTAS:
            tvs, ts = [], []
            for rep in range(4):
                r = run_backend(n, delta, 0, dag, czs, targets, seed + 10 + rep)
                tvs.append(tv_honest(targets, r["P"], P_exact))
                ts.append(r["total_s"])
            dial.append(
                {
                    "n": n,
                    "delta": delta,
                    "k": r["k"],
                    "tv_mean": float(np.mean(tvs)),
                    "tv_std": float(np.std(tvs)),
                    "time_mean_s": float(np.mean(ts)),
                }
            )
            print(
                f"dial n={n} delta={delta:.2f} k={r['k']:>7} "
                f"TV={np.mean(tvs):.4f}+-{np.std(tvs):.4f} t={np.mean(ts):.2f}s",
                flush=True,
            )

    out = os.path.join(HERE, "bench_honest.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "dial": dial}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
