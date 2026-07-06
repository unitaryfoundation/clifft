"""Validate the exact meet-in-the-middle IQP tool (chform_cpp/mitm_iqp.cpp):

  1. vs a direct dense-numpy IQP statevector at n <= 12 (all 2^n targets);
  2. vs clifft.record_probabilities on the *measured* dense-IQP program at
     n = 24 and 30 (96 uniform targets) -- cross-validates both tools.

Run:  .venv-research/bin/python -m research.chform_backend.validate_mitm
(Expects the binary at /tmp/chf_mitm; builds it if missing.)
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

MITM = "/tmp/chf_mitm"
HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_built():
    if not os.path.exists(MITM):
        src = os.path.join(HERE, "..", "chform_cpp", "mitm_iqp.cpp")
        subprocess.run(["clang++", "-std=c++20", "-O3", "-o", MITM, src], check=True)


def make_dense_iqp(n, seed, p=0.5):
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(a, b) for a in range(n) for b in range(a + 1, n) if r.random() < p]
    return dag, czs


def run_mitm(n, dag, czs, targets):
    with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
        f.write(f"{n} 0 0\n")
        f.write("".join("1" if d else "0" for d in dag) + "\n")
        f.write(f"{len(czs)}\n")
        for a, b in czs:
            f.write(f"{a} {b}\n")
        f.write(f"{len(targets)}\n")
        for x in targets:
            f.write(f"{int(x)}\n")
        spec = f.name
    out = subprocess.run([MITM, spec], capture_output=True, text=True, check=True).stdout
    os.unlink(spec)
    P = {}
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "P":
            P[int(parts[1])] = float(parts[2])
    return P


def dense_iqp_probs(n, dag, czs):
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = 1.0
    idx = np.arange(2 ** n)

    def H(q):
        nonlocal v
        lo = (idx >> q) & 1 == 0
        x0, x1 = idx[lo], idx[lo] | (1 << q)
        a, b = v[x0].copy(), v[x1].copy()
        v[x0] = (a + b) / np.sqrt(2)
        v[x1] = (a - b) / np.sqrt(2)

    for q in range(n):
        H(q)
    tp = np.exp(1j * np.pi / 4)
    for q in range(n):
        ph = np.conj(tp) if dag[q] else tp
        v[(idx >> q) & 1 == 1] *= ph
    for a, b in czs:
        v[((idx >> a) & 1 == 1) & ((idx >> b) & 1 == 1)] *= -1
    for q in range(n):
        H(q)
    return np.abs(v) ** 2


def main():
    ensure_built()

    # 1) vs dense
    worst = 0.0
    for n in range(2, 13):
        dag, czs = make_dense_iqp(n, 500 + n)
        P = run_mitm(n, dag, czs, list(range(2 ** n)))
        ref = dense_iqp_probs(n, dag, czs)
        err = max(abs(P[x] - ref[x]) for x in range(2 ** n))
        worst = max(worst, err)
    print(f"[{'OK' if worst < 1e-9 else 'FAIL'}] MitM vs dense numpy, n=2..12, all targets "
          f"(max abs err {worst:.2e})")

    # 2) vs clifft record_probabilities (measured program, exact)
    import clifft

    for n in (24, 30):
        dag, czs = make_dense_iqp(n, 500 + n)
        lines = [f"H {q}" for q in range(n)]
        lines += [f"{'T_DAG' if dag[q] else 'T'} {q}" for q in range(n)]
        lines += [f"CZ {a} {b}" for a, b in czs]
        lines += [f"H {q}" for q in range(n)]
        lines += [f"M {q}" for q in range(n)]
        prog = clifft.compile("\n".join(lines))
        rng = np.random.default_rng(7)
        targets = sorted(set(int(x) for x in rng.integers(0, 2 ** n, size=96)))
        records = np.array([[(x >> q) & 1 for q in range(n)] for x in targets], dtype=bool)
        pc = np.asarray(clifft.record_probabilities(prog, records))
        P = run_mitm(n, dag, czs, targets)
        rel = max(abs(P[x] - p) / max(p, 1e-300) for x, p in zip(targets, pc))
        print(f"[{'OK' if rel < 1e-5 else 'FAIL'}] MitM vs clifft record_probabilities, "
              f"n={n} (peak_rank={prog.peak_rank}), 96 targets (max rel err {rel:.2e})")


if __name__ == "__main__":
    main()
