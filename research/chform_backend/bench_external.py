"""External baseline: QuiZX (ZX-calculus reduced stabilizer decompositions,
Kissinger & van de Wetering, QST 2022) on the SAME dense-IQP instances as
bench_honest.py (same seeds, same targets).

Task framing -- read before quoting numbers:
  * QuiZX computes each amplitude EXACTLY (a stronger guarantee than our
    additive-error delta-approximation), one decompose run per target.
  * Our backend builds one sparsified state and answers all targets from it
    (cost amortizes across targets); QuiZX pays per target.
  We therefore report BOTH per-amplitude time and the 96-target total, and the
  comparison must be read as exact-vs-approximate, not like-for-like.
  * Everything is validated against the exact MitM evaluator.
  * Single-threaded throughout (Decomposer.decompose, not decompose_parallel).

Quokka# (weighted model counting, CAV'24) is pip-installed but requires the
external GPMC solver binary (github.com/System-Verification-Lab/GPMC) on PATH;
if `gpmc` is present we run it on the same instances, else we skip and say so.

Run:  .venv-research/bin/python -m research.chform_backend.bench_external
Writes bench_external.json next to this file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(HERE, "..", "chform_cpp")
OVERLAP = "/tmp/chf_overlap_h"
MITM = "/tmp/chf_mitm_h"

NS = [24, 32, 40, 48, 56]
NT_QUIZX = 8          # exact amplitudes are expensive; report per-amplitude
NT_FULL = 96          # ours + ground truth use the full bench_honest target set
QUIZX_TIME_CAP_S = 900.0  # stop the sweep when one target exceeds this


def build_binaries():
    for out, src in ((OVERLAP, "bench_overlap.cpp"), (MITM, "mitm_iqp.cpp")):
        subprocess.run(
            ["clang++", "-std=c++20", "-O3", "-o", out, os.path.join(CPP_DIR, src)],
            check=True,
        )


def make_dense_iqp(n, seed):
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


def parse_out(out):
    kv, P = {}, {}
    for line in out.splitlines():
        p = line.split()
        if p[0] == "P":
            P[int(p[1])] = float(p[2])
        elif len(p) == 2:
            try:
                kv[p[0]] = float(p[1])
            except ValueError:
                pass
    return kv, P


def run_tool(binary, n, k, samples, dag, czs, targets, seed=None):
    spec = write_spec(n, k, samples, dag, czs, targets)
    cmd = [binary, spec] + ([str(seed)] if seed is not None else [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    os.unlink(spec)
    return parse_out(out)


def iqp_qasm(n, dag, czs):
    # header split over lines: quokka#'s parser requires it (quizx accepts both)
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    lines += [f"h q[{q}];" for q in range(n)]
    lines += [f"{'tdg' if dag[q] else 't'} q[{q}];" for q in range(n)]
    lines += [f"cz q[{a}],q[{b}];" for a, b in czs]
    lines += [f"h q[{q}];" for q in range(n)]
    return "\n".join(lines)


def quizx_prob(qasm, n, x):
    import quizx

    g = quizx.qasm(qasm)
    g.apply_state("0" * n)
    g.apply_effect("".join(str((x >> q) & 1) for q in range(n)))
    quizx.full_simp(g)
    d = quizx.Decomposer(g)
    d.with_full_simp()
    d.decompose()
    return abs(complex(d.get_scalar())) ** 2, d.get_nterms()


def main():
    build_binaries()
    have_gpmc = shutil.which("gpmc") is not None
    results = []
    quizx_dead = False

    print(f"{'n':>3} {'quizx s/amp':>12} {'nterms':>8} {'quizx 96 (est)':>14} "
          f"{'ours d=.15 96':>13} {'TV':>6} {'exact(MitM) 96':>14}")
    for n in NS:
        seed = 1000 + 17 * n  # rep 0 of bench_honest
        dag, czs = make_dense_iqp(n, seed)
        rng_t = np.random.default_rng(seed + 1)
        targets = sorted(set(int(x) for x in rng_t.integers(0, 2**n, size=NT_FULL)))

        # exact ground truth (all targets)
        t0 = time.time()
        _, P_exact = run_tool(MITM, n, 0, 0, dag, czs, targets)
        t_mitm = time.time() - t0

        # ours, delta=0.15, analytic norm, all targets
        delta = 0.15
        k = max(2000, int(2.0 ** (0.228 * n) / delta**2))
        t0 = time.time()
        _, P_ours = run_tool(OVERLAP, n, k, 0, dag, czs, targets, seed=seed + 2)
        t_ours = time.time() - t0
        denom = sum(P_exact[x] for x in targets)
        tv = 0.5 * sum(abs(P_ours[x] - P_exact[x]) for x in targets) / denom

        # quizx, exact, per amplitude on the first NT_QUIZX targets
        row = {"n": n, "ours_96_s": t_ours, "ours_tv": tv, "mitm_96_s": t_mitm}
        if not quizx_dead:
            qasm = iqp_qasm(n, dag, czs)
            times, nterms, worst = [], [], 0.0
            for x in targets[:NT_QUIZX]:
                t0 = time.time()
                p, nt = quizx_prob(qasm, n, x)
                times.append(time.time() - t0)
                nterms.append(nt)
                worst = max(worst, abs(p - P_exact[x]) / max(P_exact[x], 1e-300))
                if times[-1] > QUIZX_TIME_CAP_S:
                    break
            per_amp = float(np.mean(times))
            row.update(
                quizx_s_per_amp=per_amp,
                quizx_nterms_mean=float(np.mean(nterms)),
                quizx_96_est_s=per_amp * NT_FULL,
                quizx_max_rel_err=worst,
                quizx_n_run=len(times),
            )
            assert worst < 1e-6, f"quizx vs MitM mismatch at n={n}: {worst}"
            if per_amp * NT_FULL > 4 * QUIZX_TIME_CAP_S or times[-1] > QUIZX_TIME_CAP_S:
                quizx_dead = True
            print(f"{n:>3} {per_amp:>12.2f} {np.mean(nterms):>8.0f} "
                  f"{per_amp * NT_FULL:>14.1f} {t_ours:>13.2f} {tv:>6.3f} {t_mitm:>14.1f}",
                  flush=True)
        else:
            print(f"{n:>3} {'(capped)':>12} {'-':>8} {'-':>14} "
                  f"{t_ours:>13.2f} {tv:>6.3f} {t_mitm:>14.1f}", flush=True)
        results.append(row)

    quokka = {"available": have_gpmc}
    if have_gpmc:
        import quokka_sharp as qk

        print("\nquokka# (gpmc found): exact P(x), 1 target per run")
        for n in NS:
            seed = 1000 + 17 * n
            dag, czs = make_dense_iqp(n, seed)
            rng_t = np.random.default_rng(seed + 1)
            targets = sorted(set(int(x) for x in rng_t.integers(0, 2**n, size=NT_FULL)))
            _, P_exact = run_tool(MITM, n, 0, 0, dag, czs, targets)
            qasm = iqp_qasm(n, dag, czs)
            with tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False) as f:
                f.write(qasm)
                qasm_path = f.name
            times, worst = [], 0.0
            for x in targets[:NT_QUIZX]:
                circuit = qk.QASMparser(qasm_path)
                cnf = qk.QASM2CNF(circuit, computational_basis=True, weighted=True)
                cnf.leftProjectAllZero()
                cnf.add_measurement({q: (x >> q) & 1 for q in range(n)})
                t0 = time.time()
                p = qk.Simulate(cnf)
                times.append(time.time() - t0)
                if p not in (None, "TIMEOUT"):
                    worst = max(worst, abs(float(p) - P_exact[x]) / max(P_exact[x], 1e-300))
            os.unlink(qasm_path)
            quokka[str(n)] = {"s_per_amp": float(np.mean(times)), "max_rel_err": worst}
            print(f"  n={n}: {np.mean(times):.2f} s/amp (max rel err {worst:.1e})", flush=True)
    else:
        print("\nquokka#: SKIPPED -- `gpmc` binary not on PATH "
              "(build github.com/System-Verification-Lab/GPMC and retry)")

    out = os.path.join(HERE, "bench_external.json")
    with open(out, "w") as f:
        json.dump({"quizx_rows": results, "quokka": quokka}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
