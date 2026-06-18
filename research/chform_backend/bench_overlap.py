"""Overlapping-regime head-to-head vs clifft (k=10-20) -- the first SAME-TASK
comparison: clifft's EXACT Born probabilities vs our APPROXIMATE ones, on the
identical magic-sparse IQP circuits, with wall-clock for clifft, the Python
backend, and the C++ backend. Produces a plot.

Run: python -u -m research.chform_backend.bench_overlap

In this regime clifft's 2^k active block is tiny (16 KB - 16 MB), so clifft is
fast AND exact -- our approximate backend has no speed edge here (its niche is
large k, where clifft cannot run). This benchmark shows exactly that boundary,
and validates our P(x) against clifft's exact P(x) beyond the n<=10 statevector
cap (basis_probabilities works at any n).
"""

from __future__ import annotations

import subprocess
import tempfile
import time

import numpy as np

from .engine import LowRankState
from . import norm_est as ne

CPP_BIN = "/tmp/chf_overlap"
NS = [10, 12, 14, 16, 18, 20, 22, 24, 26]
NTARGET = 96
SAMPLES = 80
DELTA = 0.15
SKIP_PYTHON = True  # the Python backend is the slow prototype; skip past n~20


def make_iqp(n, seed):
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(q, q + 1) for q in range(n - 1) if r.random() < 0.6]
    return dag, czs


def stim_text(n, dag, czs):
    lines = [f"H {q}" for q in range(n)]
    lines += [f"{'T_DAG' if dag[q] else 'T'} {q}" for q in range(n)]
    lines += [f"CZ {a} {b}" for a, b in czs]
    lines += [f"H {q}" for q in range(n)]
    return "\n".join(lines)


def build_ours(n, dag, czs, k, rng):
    tp = np.exp(1j * np.pi / 4)
    s = LowRankState(n, backend="chform")
    for q in range(n):
        s.clifford_1q("H", q)
    s.inject_magic_layer([(q, np.conj(tp) if dag[q] else tp) for q in range(n)], k, rng)
    for a, b in czs:
        s.cz(a, b)
    for q in range(n):
        s.clifford_1q("H", q)
    return s


def run_cpp(n, k, dag, czs, targets):
    with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
        f.write(f"{n} {k} {SAMPLES}\n")
        f.write("".join("1" if d else "0" for d in dag) + "\n")
        f.write(f"{len(czs)}\n")
        for a, b in czs:
            f.write(f"{a} {b}\n")
        f.write(f"{len(targets)}\n")
        for x in targets:
            f.write(f"{int(x)}\n")
        spec = f.name
    out = subprocess.run([CPP_BIN, spec], capture_output=True, text=True, check=True).stdout
    bt = nt = 0.0
    P = {}
    for line in out.splitlines():
        p = line.split()
        if p[0] == "build_s": bt = float(p[1])
        elif p[0] == "norm_s": nt = float(p[1])
        elif p[0] == "P": P[int(p[1])] = float(p[2])
    return bt + nt, P


def tv_distance(targets, P_ours, P_clifft):
    """Norm-free total variation on the sampled targets (isolates sparsification
    error from norm-estimation noise)."""
    so = sum(P_ours[x] for x in targets) or 1.0
    sc = sum(P_clifft[x] for x in targets) or 1.0
    return 0.5 * sum(abs(P_ours[x] / so - P_clifft[x] / sc) for x in targets)


def main():
    import clifft
    print(f"{'n':>3} {'k':>5} {'clifft(s)':>10} {'cpp(s)':>9} {'py(s)':>9} "
          f"{'TV(cpp)':>9} {'TV(py)':>9} {'||om||^2':>9}")
    rows = []
    for n in NS:
        dag, czs = make_iqp(n, 100 + n)
        text = stim_text(n, dag, czs)
        prog = clifft.compile(text)
        k = max(2000, int(2.0 ** (0.228 * n) / DELTA ** 2))
        rng_t = np.random.default_rng(7)
        targets = list(np.unique(rng_t.integers(0, 2 ** n, size=NTARGET)))
        bitstrings = [format(int(x), f"0{n}b") for x in targets]

        t0 = time.time()
        pc = np.asarray(clifft.basis_probabilities(prog, bitstrings, bit_order="little"))
        t_clifft = time.time() - t0
        P_clifft = {int(x): float(p) for x, p in zip(targets, pc)}

        t_cpp, P_cpp = run_cpp(n, k, dag, czs, targets)
        tv_cpp = tv_distance([int(x) for x in targets], P_cpp, P_clifft)

        if SKIP_PYTHON:
            t_py = tv_py = norm = float("nan")
        else:
            t0 = time.time()
            s = build_ours(n, dag, czs, k, np.random.default_rng(9 + n))
            norm = ne.estimate_norm2(s.terms, n, SAMPLES, np.random.default_rng(3))
            P_py = {int(x): abs(s.amplitude(int(x))) ** 2 / norm for x in targets}
            t_py = time.time() - t0
            tv_py = tv_distance([int(x) for x in targets], P_py, P_clifft)
        print(f"{n:>3} {k:>5} {t_clifft:>10.4f} {t_cpp:>9.4f} {t_py:>9.4f} "
              f"{tv_cpp:>9.4f} {tv_py:>9.4f} {norm:>9.3f}")
        rows.append((n, k, t_clifft, t_cpp, t_py, tv_cpp, tv_py))

    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = [r[0] for r in rows]
    has_py = not all(np.isnan(r[4]) for r in rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.semilogy(n, [r[2] for r in rows], "o-", label="clifft (exact)")
    ax1.semilogy(n, [r[3] for r in rows], "s-", label="ours C++ (approx, C++ norm-est)")
    if has_py:
        ax1.semilogy(n, [r[4] for r in rows], "^-", label="ours Python (approx, Python norm-est)")
    ax1.set_xlabel("n qubits  (k = peak_rank ≈ n)"); ax1.set_ylabel("wall-clock (s)")
    ax1.set_title("Time: Born probs of 96 bitstrings (approx lines incl. norm estimation)")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
    ax2.plot(n, [r[5] for r in rows], "s-", label="ours C++ (C++ norm-est)")
    if has_py:
        ax2.plot(n, [r[6] for r in rows], "^-", label="ours Python (Python norm-est)")
    ax2.set_xlabel("n qubits"); ax2.set_ylabel("total-variation dist. vs clifft (norm-free)")
    ax2.set_title("Correctness: approx P(x) vs clifft exact P(x)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "research/chform_cpp/overlap_bench.png" if has_py else "research/chform_cpp/overlap_bench_to26.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
