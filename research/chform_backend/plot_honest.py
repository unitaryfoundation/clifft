"""Figure for the honest benchmark (reads bench_honest.json; writes
../chform_cpp/honest_bench.png). Left: wall-clock vs n -- clifft's exact
record_probabilities path (2^{n/2}) vs the CH-form backend at two accuracy
settings. Right: the accuracy dial -- full-pipeline TV vs delta at n=44.

Run:  .venv-research/bin/python -m research.chform_backend.plot_honest
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "chform_cpp", "honest_bench.png")


def main():
    with open(os.path.join(HERE, "bench_honest.json")) as f:
        data = json.load(f)

    agg = defaultdict(lambda: defaultdict(list))
    for row in data["rows"]:
        n = row["n"]
        if row.get("clifft"):
            agg[n]["clifft"].append(row["clifft"]["query_s"])
        agg[n]["b4"].append(row["backend"]["d0.4-analytic"]["total_s"])
        agg[n]["b15"].append(row["backend"]["d0.15-analytic"]["total_s"])
        agg[n]["tv4"].append(row["backend"]["d0.4-analytic"]["tv"])
        agg[n]["tv15"].append(row["backend"]["d0.15-analytic"]["tv"])

    ns = sorted(agg)
    cl_ns = [n for n in ns if agg[n]["clifft"]]
    cl = [np.mean(agg[n]["clifft"]) for n in cl_ns]
    b4 = [np.mean(agg[n]["b4"]) for n in ns]
    b15 = [np.mean(agg[n]["b15"]) for n in ns]
    tv4 = np.mean([np.mean(agg[n]["tv4"]) for n in ns])
    tv15 = np.mean([np.mean(agg[n]["tv15"]) for n in ns])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.semilogy(cl_ns, cl, "o-", color="#1f4e8c", label="clifft record_probabilities (exact)")
    # 2^{n/2} guide through the last clifft point
    guide_n = np.array([32, 62])
    guide = cl[-1] * 2.0 ** ((guide_n - cl_ns[-1]) / 2.0)
    ax1.semilogy(guide_n, guide, ":", color="#1f4e8c", alpha=0.5, label=r"$2^{n/2}$ guide")
    ax1.semilogy(ns, b4, "s-", color="#c04a2e",
                 label=f"CH-form, $\\delta$=0.4 (TV$\\approx${tv4:.2f})")
    ax1.semilogy(ns, b15, "d-", color="#e08a30",
                 label=f"CH-form, $\\delta$=0.15 (TV$\\approx${tv15:.2f})")
    ax1.axvline(60, color="gray", lw=0.8, ls="--")
    ax1.text(60.4, 0.02, "exact needs\n$\\geq$16 GB", fontsize=8, color="gray")
    ax1.set_xlabel("n (dense random IQP, measured; peak_rank = n/2)")
    ax1.set_ylabel("seconds for 96 Born probabilities")
    ax1.set_title("Honest head-to-head: exact fast path vs approximate backend")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    dial = data["dial"]
    ds = [d["delta"] for d in dial]
    tvs = [d["tv_mean"] for d in dial]
    errs = [d["tv_std"] for d in dial]
    ax2.errorbar(ds, tvs, yerr=errs, fmt="o-", color="#c04a2e", label="measured TV (full pipeline)")
    dd = np.linspace(min(ds), max(ds), 50)
    ax2.plot(dd, 0.5 * dd, ":", color="gray", label=r"TV $= 0.50\,\delta$")
    ax2.set_xlabel(r"target error $\delta$  (budget $k = 2^{0.228n}/\delta^2$)")
    ax2.set_ylabel("total-variation distance to exact $P(x)$")
    ax2.set_title("Accuracy dial at n=44 (256 targets, 4 realizations)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
