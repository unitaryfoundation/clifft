#!/usr/bin/env python3
"""Importance sampling tutorial script for UCC documentation.

Runs the full importance sampling workflow on both the d=3 T-gate and
S-gate magic state cultivation circuits, then generates comparison plots.
This reproduces the d=3 versions of Figs. 12 and 13 from "Computing
logical error thresholds with the Pauli Frame Sparse Representation" by
Thomas Tuloup and Thomas Ayral (arXiv:2603.14670).

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

CIRCUIT_DIR = Path(__file__).resolve().parent.parent / "circuits"
T_GATE_CIRCUIT = CIRCUIT_DIR / "circuit_d3_t_gate_p0.001.stim"
S_GATE_CIRCUIT = CIRCUIT_DIR / "circuit_d3_s_gate_p0.001.stim"

IMAGE_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGE_DIR.mkdir(exist_ok=True)

MAX_K = 16  # Maximum fault count to simulate
SHOTS_PER_STRATUM = 750_000  # Shots per stratum (matches paper Fig 13 caption)
SEED_BASE = 42

# Physical error rates for reweighting
P_VALUES = np.array([0.001, 0.002, 0.003, 0.005, 0.007, 0.01])

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


class SweepResult(TypedDict):
    error_rates: list[float]
    error_bars: list[float]
    discard_rates: list[float]


# ---------------------------------------------------------------------------
# Binomial PMF for the uniform-p circuits used in this tutorial
# ---------------------------------------------------------------------------


def uniform_fault_pmf(probs: NDArray[np.float64], max_k: int) -> NDArray[np.float64]:
    """Compute P(K=k) for a circuit where every noise site shares one fault rate."""
    if probs.size == 0:
        return np.zeros(max_k + 1, dtype=np.float64)
    if not np.allclose(probs, probs[0]):
        raise ValueError("tutorial expects a uniform fault probability across all sites")
    # probs[i] is the total fault probability for the i-th noise site/trial.
    return np.asarray(binom.pmf(np.arange(max_k + 1), probs.size, probs[0]), dtype=np.float64)


# ---------------------------------------------------------------------------
# Run stratified IS on a single circuit
# ---------------------------------------------------------------------------


def run_stratified_is(
    circuit_path: Path,
    label: str,
) -> tuple[NDArray[np.float64], list[StratumRecord], int]:
    """Compile and run stratified importance sampling on a circuit.

    Returns (pmf, stratum_data, n_sites).
    """
    print(f"\n{'='*60}")
    print(f"Circuit: {label} ({circuit_path.name})")
    print(f"{'='*60}")
    circuit_text = circuit_path.read_text()

    prog_probe = ucc.compile(
        circuit_text,
        normalize_syndromes=True,
        hir_passes=ucc.default_hir_pass_manager(),
        bytecode_passes=ucc.default_bytecode_pass_manager(),
    )
    # Today we compile once to discover the detector count, then again with an
    # all-detector postselection mask. A front-end detector count query would
    # remove this extra compile.
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
    print(f"  peak_rank={prog.peak_rank}, {num_det} detectors, {n_sites} noise sites")

    pmf = uniform_fault_pmf(site_probs, MAX_K)

    print(f"  Running {MAX_K + 1} strata x {SHOTS_PER_STRATUM} shots")
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
            f"  k={k:2d}: P(K=k)={pmf[k]:.3e}, "
            f"pass={passed}/{total}, err={errors}, "
            f"p_fail|k={p_fail_k:.6f}, p_surv|k={p_surv_k:.4f}"
        )

    elapsed = time.time() - t0
    total_shots = sum(d["total"] for d in stratum_data)
    print(f"  Time: {elapsed:.2f}s ({total_shots:,} shots)")

    weighted_errors = 0.0
    weighted_survival = 0.0
    for d in stratum_data:
        k_val = d["k"]
        total_k = d["total"]
        if total_k == 0 or pmf[k_val] < 1e-30:
            continue
        weighted_errors += pmf[k_val] * d["errors"] / total_k
        weighted_survival += pmf[k_val] * d["passed"] / total_k

    p_fail_is = weighted_errors / weighted_survival if weighted_survival > 0 else 0.0
    print(f"  IS estimate: p_fail = {p_fail_is:.6e}")

    return pmf, stratum_data, n_sites


def compute_sweep(
    stratum_data: list[StratumRecord],
    n_sites: int,
) -> SweepResult:
    """Reweight stratum data across P_VALUES. Returns rates, CIs, and discard rates."""
    k_range = np.arange(MAX_K + 1)
    error_rates: list[float] = []
    error_bars: list[float] = []
    discard_rates: list[float] = []

    for p in P_VALUES:
        pmf: NDArray[np.float64] = binom.pmf(k_range, n_sites, p)
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
            var_w_err += (pmf[k] ** 2) * (p_fail_k * (1.0 - p_fail_k)) / total_k

        rate = w_err / w_surv if w_surv > 0 else 0.0
        se_rate = np.sqrt(var_w_err) / w_surv if w_surv > 0 else 0.0
        ci_95 = 1.96 * float(se_rate)
        disc = 1.0 - w_surv / w_total if w_total > 0 else 0.0

        error_rates.append(rate)
        error_bars.append(ci_95)
        discard_rates.append(disc)

    return SweepResult(error_rates=error_rates, error_bars=error_bars, discard_rates=discard_rates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # -------------------------------------------------------------------
    # Run IS on both circuits
    # -------------------------------------------------------------------
    t_pmf, t_strata, t_nsites = run_stratified_is(T_GATE_CIRCUIT, "T-gate")
    s_pmf, s_strata, s_nsites = run_stratified_is(S_GATE_CIRCUIT, "S-gate")

    t_sweep = compute_sweep(t_strata, t_nsites)
    s_sweep = compute_sweep(s_strata, s_nsites)

    # Print sweep tables
    for label, sweep in [("T-gate", t_sweep), ("S-gate", s_sweep)]:
        print(f"\n{label} sweep:")
        print(f"{'p':>8s} {'p_fail':>12s} {'95% CI':>14s} {'discard%':>10s}")
        print("-" * 50)
        for i, p in enumerate(P_VALUES):
            print(
                f"{p:8.4f} {sweep['error_rates'][i]:12.3e}"
                f"  +/- {sweep['error_bars'][i]:10.3e}"
                f" {sweep['discard_rates'][i] * 100:9.2f}%"
            )

    k_vals = np.arange(MAX_K + 1)

    # -------------------------------------------------------------------
    # Plot 1: PMF and conditional error rate (T-gate, like Fig 12a)
    # -------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    bar_width = 0.35
    ax1.bar(
        k_vals - bar_width / 2,
        t_pmf,
        width=bar_width,
        color="#ce93d8",
        alpha=0.8,
        label="P(k), p=0.001",
        zorder=3,
    )
    pmf_001: NDArray[np.float64] = binom.pmf(k_vals, t_nsites, 0.01)
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

    ax2 = ax1.twinx()
    p_fail_per_k: list[float] = []
    p_fail_per_k_err: list[float] = []
    for d in t_strata:
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
        label="Error rate (T-gate)",
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

    # -------------------------------------------------------------------
    # Plot 2: Weighted contributions (T-gate, like Fig 12b)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))

    plot_ps = [0.001, 0.003, 0.01]
    plot_colors = ["#7b1fa2", "#1565c0", "#00bcd4"]
    n_bars = len(plot_ps)
    total_bar_width = 0.7
    single_bar_width = total_bar_width / n_bars

    for i, (p_val, color) in enumerate(zip(plot_ps, plot_colors)):
        pmf_p: NDArray[np.float64] = binom.pmf(k_vals, t_nsites, p_val)
        contribs: list[float] = []
        for d in t_strata:
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

    # -------------------------------------------------------------------
    # Plot 3: Logical error rate vs overhead -- both T and S gates
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, sweep, marker, color in [
        ("d=3, gate=T", t_sweep, "s", "#e53935"),
        ("d=3, gate=S", s_sweep, "D", "#1565c0"),
    ]:
        attempts = [1.0 / (1.0 - d) for d in sweep["discard_rates"]]
        ax.errorbar(
            attempts,
            sweep["error_rates"],
            yerr=sweep["error_bars"],
            fmt=f"{marker}-",
            color=color,
            label=label,
            capsize=4,
            markersize=6,
            linewidth=1.5,
        )

    # Label only T-gate points in black (both curves share the same p values)
    t_attempts = [1.0 / (1.0 - d) for d in t_sweep["discard_rates"]]
    for p, x, y in zip(P_VALUES, t_attempts, t_sweep["error_rates"]):
        ax.annotate(
            f"p={p}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color="#333333",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Attempts per kept shot")
    ax.set_ylabel("Logical error rate (per kept shot)")
    ax.set_title("Logical Error Rate vs. Overhead")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "is_error_rate_sweep.png", bbox_inches="tight")
    print(f"Saved: {IMAGE_DIR / 'is_error_rate_sweep.png'}")
    plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
