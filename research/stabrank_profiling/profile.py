"""Run the (k, t_live) profiling suite and emit a decision table.

Usage:
    python -m research.stabrank_profiling.profile            # full suite, table
    python -m research.stabrank_profiling.profile --json out.json
    python -m research.stabrank_profiling.profile --no-optimize

The headline question this answers: across realistic circuits, does clifft's
peak active dimension k grow large while the live magic count t_live stays
moderate? If so, a residual stabilizer-rank decomposition of the active block
extends clifft's reach. The table puts the competing log2-costs side by side:

    dense      = peak_active_k                (clifft today; hard wall ~30)
    global-SR  = 0.228 * total_T              (whole-circuit stabilizer rank)
    resid-SR   = 0.228 * peak_live_magic_axes (stab-rank on the active residual)

plus ``infl`` = Clifford-inflation fraction of the active dimension.
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyzer import profile_program
from .circuits import build_suite, large_k_sweep, composition_demo


def _validate(program, res) -> str:
    """Cross-check the reconstructed peak against Program.peak_rank.

    If these disagree, the static opcode model is incomplete (some opcode
    changes active_k in a way the analyzer doesn't model) and every derived
    number is suspect. Returns "" on match, else a warning string.
    """
    if res.peak_active_k != res.program_peak_rank:
        return (
            f"  !! PEAK MISMATCH for {res.name}: "
            f"reconstructed={res.peak_active_k} vs Program.peak_rank={res.program_peak_rank}"
        )
    return ""


def run(optimize: bool = True, fused_as_magic: bool = True, sweep: bool = False, demo: bool = False):
    import clifft

    if demo:
        suite = composition_demo()
    elif sweep:
        suite = large_k_sweep()
    else:
        suite = build_suite()
    results = []
    warnings = []
    over_wall = []
    for name, stim_text in suite:
        try:
            if optimize:
                program = clifft.compile(stim_text)
            else:
                program = clifft.compile(stim_text, hir_passes=None, bytecode_passes=None)
        except RuntimeError as e:
            # clifft refuses peak rank >= 63 (1ULL << k UB guard). These are
            # exactly the circuits a stabilizer-rank backend would target: k is
            # past clifft's hard wall. Record and continue.
            over_wall.append(f"  {name}: clifft cannot compile -- {e}")
            continue
        res = profile_program(program, name=name, fused_treated_as_magic=fused_as_magic)
        results.append(res)
        w = _validate(program, res)
        if w:
            warnings.append(w)
    if over_wall:
        warnings.append("OVER clifft's k=63 wall (uncompilable; stab-rank territory):")
        warnings.extend(over_wall)
    return results, warnings


def print_table(results, warnings) -> None:
    # Columns (all log2-costs, i.e. exponents):
    #   k       = peak active dimension (dense cost; clifft today)
    #   Ttot    = total T-injections in the circuit
    #   mblk    = peak T-injections folded into one active episode (block magic)
    #   sat     = mblk / k  (magic saturation; <=4.4 => residual stab-rank wins)
    #   dense   = k
    #   gSR     = 0.228 * Ttot      (whole-circuit stabilizer rank, sampling)
    #   rSR     = min(k, 0.228*mblk)  (stab-rank on the active residual)
    #   win     = cheapest of {dense, gSR, rSR}
    hdr = (
        f"{'circuit':28} {'n':>3} {'k':>4} {'Ttot':>5} {'mblk':>5} {'sat':>5} "
        f"{'dense':>6} {'gSR':>6} {'rSR':>6} {'win':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: x.peak_active_k, reverse=True):
        costs = {
            "dense": r.log2_dense,
            "gSR": r.log2_global_stabrank_sample,
            "rSR": r.log2_residual_stabrank_sample,
        }
        verdict = min(costs, key=costs.get)
        print(
            f"{r.name:28} {r.num_qubits:>3} {r.peak_active_k:>4} {r.n_nonclifford_total:>5} "
            f"{r.peak_t_in_episode:>5} {r.magic_saturation:>5.1f} "
            f"{r.log2_dense:>6.1f} {r.log2_global_stabrank_sample:>6.1f} "
            f"{r.log2_residual_stabrank_sample:>6.1f} {verdict:>7}"
        )
    print()
    if warnings:
        print("VALIDATION WARNINGS (analyzer model incomplete -- numbers suspect):")
        for w in warnings:
            print(w)
    else:
        print("validation: all reconstructed peak_active_k == Program.peak_rank  [OK]")

    # Aggregate signal for the research bet.
    print()
    big_k = [r for r in results if r.peak_active_k >= 8]
    if big_k:
        wins = [r for r in big_k if r.residual_beats_dense]
        clifford_infl = [r for r in big_k if r.clifford_inflation > 0.01]
        mean_sat = sum(r.magic_saturation for r in big_k) / len(big_k)
        print(
            f"Of {len(big_k)} circuits with k>=8:\n"
            f"  - residual stab-rank beats dense in {len(wins)}/{len(big_k)} "
            f"(needs magic-saturation mblk/k < 4.4)\n"
            f"  - mean magic-saturation mblk/k = {mean_sat:.1f}\n"
            f"  - Clifford-inflated active dimension in {len(clifford_infl)}/{len(big_k)} "
            f"(plain OP_EXPAND is otherwise absent: k is built from magic)"
        )
        gsr_worse = [r for r in big_k if r.log2_global_stabrank_sample > r.log2_dense]
        print(
            f"  - whole-circuit stab-rank is WORSE than clifft's dense block in "
            f"{len(gsr_worse)}/{len(big_k)} (clifft's k << Ttot reduction already beats it)"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH", help="write full results as JSON")
    ap.add_argument("--no-optimize", action="store_true", help="profile naive lowering (skip passes)")
    ap.add_argument(
        "--sweep", action="store_true", help="run the large-k / low-saturation IQP+CCZ sweep"
    )
    ap.add_argument(
        "--demo", action="store_true", help="run the composition-advantage conveyor demo"
    )
    ap.add_argument(
        "--no-fused-magic",
        action="store_true",
        help="treat fused U2/U4 as Clifford (lower bound on magic)",
    )
    args = ap.parse_args(argv)

    results, warnings = run(
        optimize=not args.no_optimize,
        fused_as_magic=not args.no_fused_magic,
        sweep=args.sweep,
        demo=args.demo,
    )
    print_table(results, warnings)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nwrote {args.json}")
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
