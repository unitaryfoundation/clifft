"""Approximate EPISODIC execution -- measurement of the piece the dispatch
rule projected: per-episode sparsification with mid-run resampling.

Mechanism (hir_bridge.run_hir_record with sparsify_budget=k): T rotations
auto-sparsify when chi exceeds 2k (the engine's streaming trigger); after
every forced measurement the decomposition is resampled down to k if above
budget -- at episode boundaries the surviving terms are near-parallel, so the
boundary resample is nearly free and each episode restarts from a fresh
budget. Every step is the unbiased BGH resample, so P(record) is unbiased;
what must be MEASURED (this file) is how the error grows with the number of
episode boundaries R crossed, and that chi indeed stays at the budget scale
rather than 2^{t_live}.

Three measurements, all against clifft's exact record probabilities on
feedforward conveyors (adaptive circuits -- CONDITIONAL_PAULI replayed):
  1. unbiasedness + error vs budget (fixed circuit, growing k);
  2. error vs episode count R at fixed budget (the compounding law the
     dispatch rule's R^2 prefactor is based on);
  3. a 72-qubit end-to-end run where nothing 2^n is ever built (final norm by
     the BGH estimator) -- exact-mode chi would be 2^72; episodic runs at
     chi <= 2*budget.

Run:  .venv-research/bin/python -m research.chform_backend.bench_episodic
Writes bench_episodic.json next to this file.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from ..stabrank_profiling.circuits import magic_conveyor
from .dispatch import analyze
from .hir_bridge import cross_record_probability, optimize, run_hir_record

HERE = os.path.dirname(os.path.abspath(__file__))


def exact_record(text, seed):
    """One sampled record + its exact probability from clifft."""
    import clifft

    prog = clifft.compile(text)
    samp = clifft.sample(prog, shots=1, seed=seed)
    rec = [int(b) for b in np.asarray(samp.measurements, dtype=bool)[0]]
    p = float(np.asarray(clifft.record_probabilities(
        prog, np.array([rec], dtype=bool)))[0])
    return rec, p


def bench_budget():
    print("1) UNBIASEDNESS + ERROR vs BUDGET (conveyor r=6 w=8: per-episode "
          "exact chi=256; 24 reps each)")
    text = magic_conveyor(6, 8, 8, seed=11)
    rec, p_exact = exact_record(text, seed=3)
    hir, _, t_live = optimize(text, episodic=True)
    print(f"   exact P(record) = {p_exact:.6e}   (exact-mode chi would be "
          f"2^{t_live})")
    rows = []
    for budget in (32, 64, 128, 256):
        naive, cross, chis = [], [], []
        for rep in range(24):
            p, chi, _ = run_hir_record(hir, rec, sparsify_budget=budget,
                                       rng=np.random.default_rng(100 + rep),
                                       final_norm_rank1=True)
            pc, _ = cross_record_probability(
                hir, rec, budget, np.random.default_rng(4000 + rep),
                np.random.default_rng(8000 + rep))
            naive.append(p)
            cross.append(pc)
            chis.append(chi)
        naive, cross = np.array(naive), np.array(cross)
        row = {"budget": budget, "chi_peak": int(max(chis))}
        for tag, est in (("naive", naive), ("cross", cross)):
            row[f"{tag}_rel_rms"] = float(np.sqrt(np.mean((est / p_exact - 1) ** 2)))
            row[f"{tag}_bias"] = float(est.mean() / p_exact - 1)
        rows.append(row)
        print(f"   k={budget:>4}: naive rel-rms {row['naive_rel_rms']:6.3f} "
              f"bias {row['naive_bias']:+6.3f}   |   cross rel-rms "
              f"{row['cross_rel_rms']:6.3f} bias {row['cross_bias']:+6.3f}"
              f"   chi_peak {max(chis):>4}", flush=True)
    return rows


def bench_compounding():
    print("\n2) ERROR vs EPISODE COUNT R (w=8, budget=128 -- 2x per-episode "
          "compression; 24 reps): compounding, incl. the breakdown regime")
    rows = []
    for R in (2, 4, 8, 16):
        text = magic_conveyor(R, 8, 8, seed=21)
        rec, p_exact = exact_record(text, seed=5)
        hir, _, _ = optimize(text, episodic=True)
        est = []
        for rep in range(24):
            pc, _ = cross_record_probability(
                hir, rec, 128, np.random.default_rng(500 + rep),
                np.random.default_rng(700 + rep))
            est.append(pc)
        est = np.array(est)
        rel_rms = float(np.sqrt(np.mean((est / p_exact - 1) ** 2)))
        bias = float(est.mean() / p_exact - 1)
        rows.append({"R": R, "rel_rms": rel_rms, "bias": bias})
        print(f"   R={R:>3}: cross rel-rms {rel_rms:6.3f}  bias {bias:+6.3f}"
              f"   (rel-rms/sqrt(R) = {rel_rms / np.sqrt(R):.3f})", flush=True)
    return rows


def bench_scale():
    print("\n3) 72-QUBIT END-TO-END (conveyor r=6 w=12; exact-mode chi = 2^72 "
          "-- impossible): full non-materializing pipeline")
    text = magic_conveyor(6, 12, 12, seed=31)
    a = analyze(text, "conveyor r=6 w=12")
    rec, p_exact = exact_record(text, seed=7)
    hir, _, _ = optimize(text)  # default HIR (squeeze) measured best here
    t0 = time.time()
    ests = []
    chi = 0
    for rep in range(4):
        pc, c = cross_record_probability(
            hir, rec, 200, np.random.default_rng(9 + rep),
            np.random.default_rng(90 + rep))
        ests.append(pc)
        chi = max(chi, c)
    dt = time.time() - t0
    p = float(np.mean(ests))
    rel = abs(p / p_exact - 1)
    print(f"   n={a['episodes'] * 12} qubits, t_live={a['t_live']}, "
          f"mblk={a['mblk']}: chi_peak={chi} (budget 200)")
    print(f"   P_cross (mean of 4) = {p:.3e} vs exact {p_exact:.3e}  "
          f"(rel err {rel:.2f})  [{dt:.0f}s]")
    ok = rel < 1.0 and chi <= 512
    print(f"[{'OK' if ok else 'FAIL'}] debiased pipeline, no 2^n object, "
          f"chi at budget scale")
    assert ok
    return {"t_live": a["t_live"], "mblk": a["mblk"], "chi_peak": chi,
            "p": p, "p_exact": p_exact, "rel_err": rel, "time_s": dt}


def main():
    b = bench_budget()
    c = bench_compounding()
    s = bench_scale()
    out = os.path.join(HERE, "bench_episodic.json")
    with open(out, "w") as f:
        json.dump({"budget": b, "compounding": c, "scale": s}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
