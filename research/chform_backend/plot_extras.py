"""Figure for the external head-to-head and the adaptive phase boundary
(reads bench_external.json + bench_adaptive.json; writes
../chform_cpp/external_adaptive.png).

Run:  .venv-research/bin/python -m research.chform_backend.plot_extras
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "chform_cpp", "external_adaptive.png")


def main():
    with open(os.path.join(HERE, "bench_external.json")) as f:
        ext = json.load(f)
    with open(os.path.join(HERE, "bench_adaptive.json")) as f:
        ada = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    rows = ext["quizx_rows"]
    ns = [r["n"] for r in rows]
    ax1.semilogy(ns, [r["quizx_96_est_s"] for r in rows], "o-", color="#1f4e8c",
                 label="QuiZX (exact, per-target x 96)")
    ax1.semilogy(ns, [r["mitm_96_s"] for r in rows], "^-", color="#3a7d44",
                 label="MitM (exact, ours)")
    ax1.semilogy(ns, [r["ours_96_s"] for r in rows], "s-", color="#c04a2e",
                 label="CH-form $\\delta$=0.15 (TV$\\approx$0.08)")
    ax1.set_xlabel("n (dense random IQP, shared instances)")
    ax1.set_ylabel("seconds for 96 Born probabilities")
    ax1.set_title("External head-to-head (exact vs approximate)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    dep = ada["depth"]
    Ds = [r["D"] for r in dep]
    ax2.plot(Ds, [r["plain_s"] for r in dep], "o-", color="#1f4e8c",
             label="plain (per-term Cliffords, Aer-style)")
    ax2.plot(Ds, [r["frame_s"] for r in dep], "s-", color="#c04a2e",
             label="frame (online composition)")
    ax2.set_xlabel("Clifford scrambling depth D per round (w=8, R=4, adaptive)")
    ax2.set_ylabel("seconds per trajectory")
    ax2.set_title("Adaptive workload: the frame pays off iff Clifford-dominated")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
