"""Accuracy/runtime tradeoff: sweep the sparsification budget k (equivalently the
target error delta) at fixed n, measuring total-variation distance vs clifft's
EXACT Born probabilities and wall-clock. Produces a Pareto plot.

Run: python -u -m research.chform_backend.bench_precision

budget k = 2^{0.228 n} / delta^2, so smaller delta -> larger k -> lower TV but
higher runtime. Each k is averaged over R independent single-shot realizations
(the estimator is random); we report mean +/- std.
"""

from __future__ import annotations

import subprocess
import tempfile
import time

import numpy as np

CPP_BIN = "/tmp/chf_overlap"
N = 24
NTARGET = 256
KS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
R = 6                      # realizations per k
EXTENT = 2.0 ** (0.228 * N)


def make_iqp(n, seed):
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(q, q + 1) for q in range(n - 1) if r.random() < 0.6]
    return dag, czs


def run_cpp(n, k, dag, czs, targets, seed):
    with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
        f.write(f"{n} {k} 8\n")
        f.write("".join("1" if d else "0" for d in dag) + "\n")
        f.write(f"{len(czs)}\n")
        for a, b in czs:
            f.write(f"{a} {b}\n")
        f.write(f"{len(targets)}\n")
        for x in targets:
            f.write(f"{int(x)}\n")
        spec = f.name
    t0 = time.time()
    out = subprocess.run([CPP_BIN, spec, str(seed)], capture_output=True, text=True, check=True).stdout
    wall = time.time() - t0
    P = {int(p[1]): float(p[2]) for p in (l.split() for l in out.splitlines()) if p[0] == "P"}
    return wall, P


def tv(targets, Po, Pc):
    so = sum(Po[x] for x in targets) or 1.0
    sc = sum(Pc[x] for x in targets) or 1.0
    return 0.5 * sum(abs(Po[x] / so - Pc[x] / sc) for x in targets)


def main():
    import clifft
    dag, czs = make_iqp(N, 100 + N)
    text = "\n".join([f"H {q}" for q in range(N)]
                     + [f"{'T_DAG' if dag[q] else 'T'} {q}" for q in range(N)]
                     + [f"CZ {a} {b}" for a, b in czs]
                     + [f"H {q}" for q in range(N)])
    prog = clifft.compile(text)
    rng_t = np.random.default_rng(7)
    targets = [int(x) for x in np.unique(rng_t.integers(0, 2 ** N, size=NTARGET))]
    t0 = time.time()
    pc = np.asarray(clifft.basis_probabilities(prog, [format(x, f"0{N}b") for x in targets],
                                               bit_order="little"))
    print(f"clifft exact P(x): {time.time() - t0:.2f}s (once)")
    Pc = {x: float(p) for x, p in zip(targets, pc)}

    print(f"{'k':>8} {'delta':>7} {'mean TV':>9} {'std TV':>8} {'mean t(s)':>10}")
    rows = []
    for k in KS:
        delta = (EXTENT / k) ** 0.5
        tvs, ts = [], []
        for r in range(R):
            wall, Po = run_cpp(N, k, dag, czs, targets, seed=1000 + r)
            tvs.append(tv(targets, Po, Pc)); ts.append(wall)
        mtv, stv, mt = np.mean(tvs), np.std(tvs), np.mean(ts)
        print(f"{k:>8} {delta:>7.3f} {mtv:>9.4f} {stv:>8.4f} {mt:>10.3f}")
        rows.append((k, delta, mtv, stv, mt))
    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = [r[0] for r in rows]; delta = [r[1] for r in rows]
    mtv = [r[2] for r in rows]; stv = [r[3] for r in rows]; mt = [r[4] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.errorbar(mt, mtv, yerr=stv, fmt="o-", capsize=3)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("wall-clock per run (s)")
    ax1.set_ylabel("total-variation dist. vs clifft exact P(x)")
    ax1.set_title(f"Accuracy vs runtime (Pareto), n={N}")
    ax1.grid(True, which="both", alpha=0.3)
    for kk, x, y in zip(k, mt, mtv):
        ax1.annotate(f"k={kk//1000}k", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax2.loglog(delta, mtv, "s-", label="measured TV")
    ax2.loglog(delta, delta, "k--", alpha=0.5, label=r"$\propto\delta$ (slope 1)")
    ax2.set_xlabel(r"target error $\delta=\sqrt{2^{0.228n}/k}$")
    ax2.set_ylabel("total-variation dist. vs clifft"); ax2.set_title("TV scales ~ delta")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = "research/chform_cpp/precision_bench.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
