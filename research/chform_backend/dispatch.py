"""The compile-time dispatcher, with the per-episode (mblk) rule.

Three execution strategies, three cost exponents, all computable from one
clifft compilation before anything runs:

  dense engine        2^{peak_rank}                exact
  backend, global     2^{0.228 t_live} / delta^2   one sparsified sum over ALL
                                                   surviving magic
  backend, episodic   R^2 2^{0.228 mblk} / delta^2 re-sparsify at each episode
                                                   boundary (k -> 0); errors
                                                   compound over R episodes, so
                                                   delta_episode = delta/sqrt(R)
                                                   -> the R^2 prefactor

where t_live = HIR live T-count, mblk = peak T-injections inside any single
active episode, and R = number of magic-carrying episodes (both from the
static opcode walk of research/stabrank_profiling). Since mblk <= t_live with
equality iff the circuit is one episode, the episodic rule only ever moves
circuits TOWARD the backend -- dramatically so for long computations that
recycle a bounded amount of live magic (the conveyor / fault-tolerance
shape), where 0.228*t_live is absurd but 0.228*mblk is small.

Episodic execution is IMPLEMENTED and measured (bench_episodic.py). The
measurements split the strategy in two:

  backend-episodic-exact   2^{mblk} terms, ZERO error: budget at the
                           per-episode exact rank; T-time growth never
                           exceeds it and the boundary collapse is exact.
                           Measured: chi_peak = 2^{mblk}, P == clifft.
  backend-episodic (appr.) budgets below 2^{mblk}: the sound schedule
                           (T-time sparsify only -- never resample after a
                           projection, which is ill-conditioned -- exact
                           boundary collapse, two-run debiased estimator)
                           works at mild compression (72 qubits, 10x
                           compression: rel err 0.4) but for FULL-RECORD
                           probabilities the error turns on sharply below
                           the exact budget and compounds steeply with R --
                           the R^2 model is optimistic there. Use for mild
                           compression or non-record targets.

Run the demo + validation:
  .venv-research/bin/python -m research.chform_backend.dispatch
"""

from __future__ import annotations

import math

import numpy as np

from ..stabrank_profiling.analyzer import profile_stim
from .hir_bridge import optimize, run_hir_record

ALPHA = 0.228
MEM_WALL_BYTES = 16e9  # dense feasibility: 16 B * 2^k <= this


def count_episodes(k_trajectory) -> int:
    """Number of maximal k>0 stretches (episodes that held any active block)."""
    R = 0
    up = False
    for k in k_trajectory:
        if k > 0 and not up:
            R += 1
            up = True
        elif k == 0:
            up = False
    return max(R, 1)


def analyze(stim_text: str, name: str = "") -> dict:
    """All dispatch inputs from one compilation pass."""
    prof = profile_stim(stim_text, name=name, keep_trajectory=True)
    _, t_raw, t_live = optimize(stim_text)
    return {
        "name": name,
        "num_qubits": prof.num_qubits,
        "peak_rank": prof.program_peak_rank,
        "t_raw": t_raw,
        "t_live": t_live,
        "mblk": prof.peak_t_in_episode,
        "episodes": count_episodes(prof.k_trajectory),
    }


def dispatch(stim_text: str, delta: float = 0.15, name: str = "",
             mem_bytes: float = MEM_WALL_BYTES) -> dict:
    """Route a program to its cheapest feasible strategy.

    Cost model (log2 of the dominant term count / block size; the two backend
    strategies share the 1/delta^2 factor, the episodic one pays R^2 total):
        dense     : peak_rank                      (feasible iff within memory)
        global    : 0.228 * t_live + 2 log2(1/delta)
        episodic  : 0.228 * mblk  + 2 log2(R/delta)   [R > 1 only]
    """
    a = analyze(stim_text, name)
    k, tl, mblk, R = a["peak_rank"], a["t_live"], a["mblk"], a["episodes"]
    log2_inv_d2 = 2.0 * math.log2(1.0 / delta)
    n_q = 1 + max(1, a.get("num_qubits", 64))  # term size ~ O(n^2) bits
    costs = {
        "dense": (float(k), 16.0 * 2.0 ** k <= mem_bytes),
        "backend-global": (ALPHA * tl + log2_inv_d2, True),
        # exact episodic: 2^mblk terms, zero error (measured); feasible while
        # the term store fits
        "backend-episodic-exact": (float(mblk),
                                   R > 1 and (2.0 ** mblk) * (n_q ** 2 / 4.0)
                                   <= mem_bytes),
        # approximate episodic: mild compression only for record targets
        # (measured caveat -- see module docstring)
        "backend-episodic": (ALPHA * mblk + 2.0 * math.log2(R / delta),
                             R > 1),
    }
    feasible = {s: c for s, (c, ok) in costs.items() if ok}
    choice = min(feasible, key=feasible.get)
    a.update({
        "delta": delta,
        "log2_cost": {s: round(c, 1) for s, (c, ok) in costs.items()},
        "feasible": {s: ok for s, (c, ok) in costs.items()},
        "choice": choice,
    })
    return a


# ---------------------------------------------------------------------------
# demo + validation
# ---------------------------------------------------------------------------
def _families():
    from ..stabrank_profiling.circuits import magic_conveyor
    from .bench_honest import make_dense_iqp
    from .gadgetize import hidden_shift, random_cliffordT
    from .hir_bridge import ops_to_stim

    fams = []
    n = 40
    dag, czs = make_dense_iqp(n, 1000 + 17 * n)
    lines = [f"H {q}" for q in range(n)]
    lines += [f"{'T_DAG' if dag[q] else 'T'} {q}" for q in range(n)]
    lines += [f"CZ {a} {b}" for a, b in czs]
    lines += [f"H {q}" for q in range(n)]
    lines += [f"M {q}" for q in range(n)]
    fams.append(("dense IQP n=40", "\n".join(lines)))

    ops = random_cliffordT(30, 6, 48, seed=830)
    fams.append(("random n=30 t=48", ops_to_stim(ops, 30)))

    ops, _ = hidden_shift(32, 4, seed=82)
    fams.append(("hidden shift n=32", ops_to_stim(ops, 32)))

    fams.append(("conveyor r=8 w=8", magic_conveyor(8, 8, 8, seed=5)))
    fams.append(("conveyor r=12 w=24", magic_conveyor(12, 24, 24, seed=5)))
    # the flip case: per-round blocks too big for the dense wall, total magic
    # absurd for the global backend -- only the episodic strategy survives
    fams.append(("conveyor r=12 w=128", magic_conveyor(12, 128, 128, seed=5)))
    return fams


def demo():
    print("COMPILE-TIME DISPATCH (delta=0.15): one compilation, four "
          "strategies, cheapest feasible wins")
    print(f"{'family':>20} {'k':>4} {'tlive':>5} {'mblk':>4} {'R':>3} "
          f"{'dense':>6} {'global':>7} {'epi-ex':>7} {'epi-ap':>7}  choice")
    for name, text in _families():
        d = dispatch(text, delta=0.15, name=name)
        c = d["log2_cost"]
        dense = f"{c['dense']:.0f}" + ("" if d["feasible"]["dense"] else "!")
        epi_ex = (f"{c['backend-episodic-exact']:.0f}"
                  if d["feasible"]["backend-episodic-exact"] else "--")
        epi_ap = (f"{c['backend-episodic']:.1f}"
                  if d["feasible"]["backend-episodic"] else "--")
        print(f"{name:>20} {d['peak_rank']:>4} {d['t_live']:>5} "
              f"{d['mblk']:>4} {d['episodes']:>3} {dense:>6} "
              f"{c['backend-global']:>7.1f} {epi_ex:>7} {epi_ap:>7}  "
              f"{d['choice']}")
    print("  (! = dense infeasible at the 16 GB wall; log2 of dominant "
          "term count / block size)")


def validate_episodic_premise():
    """The episodic rule presumes the backend's chi collapses at episode
    boundaries. Demonstrate on a small conveyor via the HIR bridge with
    canonical recompression (validation scale): chi_peak tracks 2^mblk, not
    2^t_live, and P(record) still matches clifft exactly."""
    import clifft

    from ..stabrank_profiling.circuits import magic_conveyor

    text = magic_conveyor(4, 3, 3, seed=9)  # R=4 episodes, mblk=3, t_live=12
    a = analyze(text, "conveyor r=4 w=3")
    prog = clifft.compile(text)
    samp = clifft.sample(prog, shots=32)
    recs = sorted(set(tuple(r) for r in
                      np.asarray(samp.measurements, dtype=bool).tolist()))[:6]
    pc = np.asarray(clifft.record_probabilities(prog, np.array(recs, dtype=bool)))
    hir_dict, _, _ = optimize(text)
    worst = 0.0
    chi_peak = 0
    for rec, p_exact in zip(recs, pc):
        p_hir, chi, _ = run_hir_record(hir_dict, rec, recompress=True)
        worst = max(worst, abs(p_hir - p_exact))
        chi_peak = max(chi_peak, chi)
    ok = worst < 1e-9 and chi_peak < 2 ** a["t_live"]
    print(f"\n[{'OK' if ok else 'FAIL'}] episodic premise (conveyor r=4 w=3): "
          f"chi_peak={chi_peak} ~ 2^mblk={2 ** a['mblk']} << "
          f"2^t_live={2 ** a['t_live']}; max|P - clifft| = {worst:.2e}")
    assert ok


if __name__ == "__main__":
    demo()
    validate_episodic_premise()
