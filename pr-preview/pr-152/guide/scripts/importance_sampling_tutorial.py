#!/usr/bin/env python3
"""Importance sampling tutorial script for UCC documentation.

Runs the full importance sampling workflow on the d=3 magic state
cultivation circuit and generates plots for the tutorial.

Usage:
    uv run python docs/guide/scripts/importance_sampling_tutorial.py

Generates:
    docs/guide/images/is_pmf_and_error_rate.png
    docs/guide/images/is_weighted_contributions.png
    docs/guide/images/is_error_rate_sweep.png
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom  # type: ignore[import-not-found]

import ucc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CIRCUIT_PATH = Path(__file__).resolve().parent.parent / "circuits" / "circuit_d3_p0.001.stim"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGE_DIR.mkdir(exist_ok=True)

MAX_K = 16  # Maximum fault count to simulate
SHOTS_PER_STRATUM = 750_000  # Shots per stratum (matches paper Fig 13 caption)
SEED_BASE = 42

# Physical error rates for reweighting
P_VALUES = np.array([0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01])

matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.facecolor": "white",
    }
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class StratumRecord(TypedDict):
    k: int
    total: int
    passed: int
    errors: int


# ---------------------------------------------------------------------------
# Poisson-Binomial PMF via dynamic programming
# ---------------------------------------------------------------------------


def poisson_binomial_pmf(probs: NDArray[np.float64], max_k: int) -> NDArray[np.float64]:
    """Compute the Poisson-Binomial PMF P(K=k) for k = 0, ..., max_k."""
    dp = np.zeros(max_k + 1)
    dp[0] = 1.0
    for p in probs:
        for k in range(max_k, 0, -1):
            dp[k] = dp[k] * (1.0 - p) + dp[k - 1] * p
        dp[0] *= 1.0 - p
    return dp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading circuit: {CIRCUIT_PATH}")
    circuit_text = CIRCUIT_PATH.read_text()

    # Build all-detector postselection mask (one flag per detector)
    prog_probe = ucc.compile(
        circuit_text,
        normalize_syndromes=True,
        hir_passes=ucc.default_hir_pass_manager(),
        bytecode_passes=ucc.default_bytecode_pass_manager(),
    )
    num_det: int = prog_probe.num_detectors
    mask = [1] * num_det

    prog = ucc.compile(
        circuit_text,
        normalize_syndromes=True,
        postselection_mask=mask,
        hir_passes=ucc.default_hir_pass_manager(),
        bytecode_passes=ucc.default_bytecode_pass_manager(),
    )

    site_probs: NDArray[np.float64] = prog.noise_site_probabilities
    n_sites = len(site_probs)
    print(f"Circuit: peak_rank={prog.peak_rank}, {num_det} detectors, {n_sites} noise sites")
    print(f"All site probabilities equal: {len(np.unique(site_probs)) == 1}")
    print(f"Site probability: {site_probs[0]:.6f}")

    # -----------------------------------------------------------------------
    # Step 1: Compute Poisson-Binomial PMF
    # -----------------------------------------------------------------------
    P_K = poisson_binomial_pmf(site_probs, MAX_K)
    print(f"\nPoisson-Binomial PMF (N={n_sites}, p={site_probs[0]})")
    for k in range(MAX_K + 1):
        print(f"  P(K={k:2d}) = {P_K[k]:.6e}")
    print(f"  sum(P_K) = {sum(P_K):.10f}")
    print(f"  tail P(K>{MAX_K}) = {1.0 - sum(P_K):.3e}")

    # -----------------------------------------------------------------------
    # Step 2: Run stratified importance sampling
    # -----------------------------------------------------------------------
    print(f"\nRunning stratified sampling: {MAX_K + 1} strata x {SHOTS_PER_STRATUM} shots")
    t0 = time.time()

    stratum_data: list[StratumRecord] = []
    for k in range(MAX_K + 1):
        r = ucc.sample_k_survivors(prog, shots=SHOTS_PER_STRATUM, k=k, seed=SEED_BASE + k)
        total: int = r["total_shots"]
        passed: int = r["passed_shots"]
        errors: int = r["logical_errors"]
        stratum_data.append(StratumRecord(k=k, total=total, passed=passed, errors=errors))
        p_fail_k = errors / total if total > 0 else 0.0
        p_surv_k = passed / total if total > 0 else 0.0
        print(
            f"  k={k:2d}: P(K=k)={P_K[k]:.3e}, "
            f"pass={passed}/{total}, err={errors}, "
            f"p_fail|k={p_fail_k:.6f}, p_surv|k={p_surv_k:.4f}"
        )

    elapsed = time.time() - t0
    total_shots_all = sum(d["total"] for d in stratum_data)
    print(f"Total time: {elapsed:.2f}s ({total_shots_all:,} shots)")

    # Compute weighted error rate at the circuit's native p
    weighted_errors = 0.0
    weighted_survival = 0.0
    for d in stratum_data:
        k = d["k"]
        total_k = d["total"]
        if total_k == 0 or P_K[k] < 1e-30:
            continue
        weighted_errors += P_K[k] * d["errors"] / total_k
        weighted_survival += P_K[k] * d["passed"] / total_k

    p_fail_is = weighted_errors / weighted_survival if weighted_survival > 0 else 0.0
    print(f"\nImportance sampling estimate: p_fail = {p_fail_is:.6e}")

    # -----------------------------------------------------------------------
    # Step 3: Reweight for different physical error rates
    # -----------------------------------------------------------------------
    error_rates: list[float] = []
    error_bars: list[float] = []
    discard_rates: list[float] = []

    print(f"\n{'p':>8s} {'p_fail':>12s} {'95% CI':>14s} {'discard%':>10s}")
    print("-" * 50)
    for p in P_VALUES:
        pmf: NDArray[np.float64] = binom.pmf(np.arange(MAX_K + 1), n_sites, p)
        w_err = 0.0
        w_surv = 0.0
        w_total = 0.0
        var_w_err = 0.0

        for d in stratum_data:
            k = d["k"]
            total_k = d["total"]
            if total_k == 0 or pmf[k] < 1e-30:
                continue

            p_fail_k = d["errors"] / total_k
            p_surv_k = d["passed"] / total_k

            w_err += pmf[k] * p_fail_k
            w_surv += pmf[k] * p_surv_k
            w_total += pmf[k]

            # Variance of numerator (Eq. 55 from Tuloup & Ayral)
            var_w_err += (pmf[k] ** 2) * (p_fail_k * (1.0 - p_fail_k)) / total_k

        rate = w_err / w_surv if w_surv > 0 else 0.0
        se_rate = np.sqrt(var_w_err) / w_surv if w_surv > 0 else 0.0
        ci_95 = 1.96 * float(se_rate)
        disc = 1.0 - w_surv / w_total if w_total > 0 else 0.0

        error_rates.append(rate)
        error_bars.append(ci_95)
        discard_rates.append(disc)
        print(f"{p:8.4f} {rate:12.3e}  +/- {ci_95:10.3e} {disc * 100:9.2f}%")

    # -----------------------------------------------------------------------
    # Plot 1: PMF and conditional error rate (dual-axis, like Fig 12a)
    # Two sets of PMF bars: p=0.001 and p=0.01
    # -----------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    k_vals = np.arange(MAX_K + 1)
    bar_width = 0.35

    # PMF bars for p=0.001 (the native circuit probability)
    ax1.bar(
        k_vals - bar_width / 2,
        P_K,
        width=bar_width,
        color="#ce93d8",
        alpha=0.8,
        label="P(k), p=0.001",
        zorder=3,
    )

    # PMF bars for p=0.01
    pmf_001: NDArray[np.float64] = binom.pmf(k_vals, n_sites, 0.01)
    ax1.bar(
        k_vals + bar_width / 2,
        pmf_001,
        width=bar_width,
        color="#5c6bc0",
        alpha=0.8,
        label="P(k), p=0.01",
        zorder=3,
    )

    ax1.set_xlabel("Number of faults k")
    ax1.set_ylabel("P(k)")
    ax1.tick_params(axis="y")
    ax1.set_xticks(k_vals)

    # Right axis: conditional error rate (errors / total shots)
    ax2 = ax1.twinx()
    p_fail_per_k: list[float] = []
    p_fail_per_k_err: list[float] = []
    for d in stratum_data:
        total_k = d["total"]
        errors_k = d["errors"]
        if total_k > 0:
            rate_k = errors_k / total_k
            se = np.sqrt(rate_k * (1 - rate_k) / total_k)
            p_fail_per_k.append(rate_k)
            p_fail_per_k_err.append(1.96 * float(se))
        else:
            p_fail_per_k.append(0.0)
            p_fail_per_k_err.append(0.0)

    ax2.errorbar(
        k_vals,
        p_fail_per_k,
        yerr=p_fail_per_k_err,
        fmt="o-",
        color="#e53935",
        label="Error rate",
        zorder=4,
        capsize=3,
    )
    ax2.set_ylabel("Error rate", color="#e53935")
    ax2.tick_params(axis="y", labelcolor="#e53935")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("d=3 Magic State Cultivation: PMF and Conditional Error Rate")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "is_pmf_and_error_rate.png", bbox_inches="tight")
    print(f"\nSaved: {IMAGE_DIR / 'is_pmf_and_error_rate.png'}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: Weighted contributions P(K=k) * p_fail|k (like Fig 12b)
    # Three sets of bars: p=0.001, p=0.003, p=0.01
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))

    plot_ps = [0.001, 0.003, 0.01]
    plot_colors = ["#7b1fa2", "#1565c0", "#00bcd4"]
    n_bars = len(plot_ps)
    total_bar_width = 0.7
    single_bar_width = total_bar_width / n_bars

    for i, (p_val, color) in enumerate(zip(plot_ps, plot_colors)):
        pmf_p: NDArray[np.float64] = binom.pmf(k_vals, n_sites, p_val)
        contribs: list[float] = []
        for d in stratum_data:
            k = d["k"]
            total_k = d["total"]
            if total_k > 0:
                contribs.append(pmf_p[k] * d["errors"] / total_k)
            else:
                contribs.append(0.0)
        offset = (i - (n_bars - 1) / 2) * single_bar_width
        ax.bar(
            k_vals + offset,
            contribs,
            width=single_bar_width,
            color=color,
            alpha=0.85,
            label=f"p={p_val}",
            zorder=3,
        )

    ax.set_xlabel("Number of faults k")
    ax.set_ylabel(r"P(k) $\times$ p$_{fail|k}$")
    ax.set_title("Weighted Error Contributions by Stratum")
    ax.set_xticks(k_vals)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "is_weighted_contributions.png", bbox_inches="tight")
    print(f"Saved: {IMAGE_DIR / 'is_weighted_contributions.png'}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 3: Logical error rate vs attempts per kept shot (like Fig 13a)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))

    attempts_per_kept = [1.0 / (1.0 - d) for d in discard_rates]

    ax.errorbar(
        attempts_per_kept,
        error_rates,
        yerr=error_bars,
        fmt="s-",
        color="#e53935",
        label="d=3, gate=T",
        capsize=4,
        markersize=6,
        linewidth=1.5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Attempts per kept shot")
    ax.set_ylabel("Logical error rate (per kept shot)")
    ax.set_title("Logical Error Rate vs. Overhead")

    for p, x, y in zip(P_VALUES, attempts_per_kept, error_rates):
        ax.annotate(
            f"p={p}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color="#555555",
        )

    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "is_error_rate_sweep.png", bbox_inches="tight")
    print(f"Saved: {IMAGE_DIR / 'is_error_rate_sweep.png'}")
    plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
