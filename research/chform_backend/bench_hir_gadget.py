"""Gadgetize the COMPILED circuit: T-teleportation wired to clifft's
optimized HIR (hir_bridge.run_hir_record_gadgetized).

Previously two separate routes existed: gadgetization consumed RAW circuits
(single-shot sampler + analytic norm, but at t_raw), and the HIR bridge
consumed the optimized IR (t_live, but streaming T's with mid-run
resampling). This benchmark measures their composition -- the product-magic
sampler at the LIVE T-count:

  1. exactness: gadgetized-HIR == clifft record_probabilities on random
     interleaved Clifford+T and hidden shift (machine precision);
  2. the saving: at EQUAL budget k, accuracy of raw-gadgetized (t_raw
     ancillas) vs HIR-gadgetized (t_live ancillas) against exact clifft.
     The live route's extent is 2^{0.228 t_live}, so it should match the raw
     route's accuracy at ~2^{0.228 (t_raw - t_live)} smaller budget.

Run:  .venv-research/bin/python -m research.chform_backend.bench_hir_gadget
Writes bench_hir_gadget.json next to this file.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .gadgetize import build_gadgetized, hidden_shift, probability, random_cliffordT
from .hir_bridge import (
    ops_to_stim,
    optimize,
    run_hir_record,
    run_hir_record_gadgetized,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def clifft_records(text, n_targets, seed):
    """Records sampled from the output distribution + exact probabilities."""
    import clifft

    prog = clifft.compile(text)
    samp = clifft.sample(prog, shots=4 * n_targets, seed=seed)
    recs = sorted(set(tuple(r) for r in
                      np.asarray(samp.measurements, dtype=bool).tolist()))
    recs = recs[:n_targets]
    pc = np.asarray(clifft.record_probabilities(
        prog, np.array(recs, dtype=bool)))
    return recs, pc


def validate_exact():
    print("1) EXACTNESS: gadgetized-HIR == clifft == streaming-HIR")
    worst = 0.0
    cases = [("random", n, random_cliffordT(n, 5, tt, seed=70 + n))
             for n, tt in ((5, 8), (7, 12), (9, 16))]
    ops_hs, _ = hidden_shift(6, n_ccz=1, seed=4)
    cases.append(("hidden-shift", 6, ops_hs))
    for name, n, ops in cases:
        text = ops_to_stim(ops, n)
        hir, t_raw, t_live = optimize(text)
        # exact mode costs 2^{t_live} per record: cap the record count so the
        # heavy hidden-shift case (t_live=14, chi=16384) validates in minutes.
        # The cheap cases exercise the statevector norm; the heavy one the
        # rank-1 norm (all qubits are measured, so both are exact).
        n_rec = 12 if t_live <= 10 else 2
        recs, pc = clifft_records(text, n_rec, seed=1)
        for rec, p_exact in zip(recs, pc):
            p_gad, chi, _ = run_hir_record_gadgetized(
                hir, rec, k=0, exact=True, final_norm_rank1=t_live > 10)
            worst = max(worst, abs(p_gad - p_exact))
            if t_live <= 10:      # streaming cross-check on the cheap cases
                p_str, _, _ = run_hir_record(hir, rec)
                worst = max(worst, abs(p_gad - p_str))
        print(f"   {name:>12} n={n} t_raw={t_raw:>3} -> t_live={t_live:>3}  "
              f"chi_exact={chi:>5}  max err {worst:.2e}", flush=True)
    ok = worst < 1e-9
    print(f"[{'OK' if ok else 'FAIL'}] exact gadgetized-HIR matches clifft "
          f"and the streaming path ({worst:.2e})")
    assert ok
    return worst


def bench_saving():
    print("\n2) EQUAL-BUDGET ACCURACY: raw-gadgetized (t_raw) vs "
          "HIR-gadgetized (t_live), 12 records, 2 reps")
    n, t_total = 9, 32
    ops = random_cliffordT(n, 5, t_total, seed=97)
    text = ops_to_stim(ops, n)
    recs, pc = clifft_records(text, 12, seed=2)
    hir, t_raw, t_live = optimize(text)
    ratio = 2.0 ** (0.228 * (t_raw - t_live))
    print(f"   t_raw={t_raw} -> t_live={t_live}: extent ratio "
          f"2^{{0.228*{t_raw - t_live}}} = {ratio:.0f}x")
    rows = []
    for k in (500, 2000):
        tv_raw, tv_hir, dt_raw, dt_hir = [], [], 0.0, 0.0
        for rep in range(2):
            rng = np.random.default_rng(300 + rep)
            t0 = time.time()
            st, t, l1sq = build_gadgetized(n, ops, k, rng)
            p_raw = np.array([probability(
                st, n, t, sum(int(b) << q for q, b in enumerate(rec)),
                k=k, l1sq=l1sq) for rec in recs])
            dt_raw += time.time() - t0
            t0 = time.time()
            p_hir = np.array([run_hir_record_gadgetized(
                hir, rec, k, np.random.default_rng(600 + 97 * rep + i),
                final_norm_rank1=True)[0]
                for i, rec in enumerate(recs)])
            dt_hir += time.time() - t0
            tv_raw.append(0.5 * np.abs(p_raw - pc).sum() / pc.sum())
            tv_hir.append(0.5 * np.abs(p_hir - pc).sum() / pc.sum())
        row = {"k": k, "tv_raw": float(np.mean(tv_raw)),
               "tv_hir": float(np.mean(tv_hir)),
               "s_raw": dt_raw / 2, "s_hir": dt_hir / 2}
        rows.append(row)
        print(f"   k={k:>5}: raw TV {row['tv_raw']:.3f} ({row['s_raw']:.1f}s)"
              f"   |   live TV {row['tv_hir']:.3f} ({row['s_hir']:.1f}s)",
              flush=True)
    return {"t_raw": t_raw, "t_live": t_live, "extent_ratio": ratio,
            "rows": rows}


def bench_hidden_shift():
    print("\n3) HIDDEN SHIFT at the live T-count (P(s)=1 ground truth)")
    rows = []
    for n in (16, 24):
        ops, shift = hidden_shift(n, n_ccz=n // 8, seed=11)
        text = ops_to_stim(ops, n)
        hir, t_raw, t_live = optimize(text)
        rec = [(shift >> q) & 1 for q in range(n)]
        k = max(200, int(2.0 ** (0.228 * t_live) / 0.3 ** 2))
        t0 = time.time()
        p, chi, _ = run_hir_record_gadgetized(
            hir, rec, k, np.random.default_rng(5), final_norm_rank1=True)
        dt = time.time() - t0
        rows.append({"n": n, "t_raw": t_raw, "t_live": t_live, "k": k,
                     "p_shift": float(p), "time_s": dt})
        print(f"   n={n}: t_raw={t_raw} -> t_live={t_live}, k={k}: "
              f"P(shift) = {p:.3f} (true 1.0)  [{dt:.1f}s]", flush=True)
    return rows


def main():
    w = validate_exact()
    s = bench_saving()
    h = bench_hidden_shift()
    out = os.path.join(HERE, "bench_hir_gadget.json")
    with open(out, "w") as f:
        json.dump({"exact_worst_err": w, "saving": s, "hidden_shift": h},
                  f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
