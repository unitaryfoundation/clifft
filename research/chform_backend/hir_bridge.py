"""Consume clifft's OPTIMIZED HIR in the CH-form backend -- the wiring that
was scaffolded at the start of this research thread and never connected.

clifft's Python API exposes the full pipeline: parse -> trace (Heisenberg IR)
-> HirPassManager.run (the optimizer). The optimized HIR is remarkable input
for this backend:

  * it contains NO Clifford gates at all -- the compiler has absorbed every
    Clifford into the Pauli strings of the remaining ops (the frame idea,
    performed statically and exactly);
  * its ops are exactly the two things the engine's explicit-Pauli entry
    points implement: T_GATE = a diagonal rotation about an arbitrary Pauli
    (engine.rz_about_pauli) and MEASURE = a projective Pauli measurement
    (engine.measure_pauli_forced_fast);
  * its T-count (`HirModule.num_t_gates`) is the LIVE magic after
    optimization -- the honest backend cost exponent, often far below the raw
    T-count. The compile-time dispatch rule should compare peak_rank against
    0.228 * t_live, not 0.228 * t_raw.

So running the backend on the optimized HIR means: chi tracks 2^{t_live}
instead of 2^{t_raw}, and the backend performs ZERO Clifford gate
applications -- the online frame is subsumed by the compiler for
compile-known circuits (it remains the mechanism for dynamic/adaptive ones).

Record probabilities: HIR MEASURE ops are forced to the record's bits; each
projection's norm decay carries the probability, recovered from the final
term-sum norm and the accumulated rescale factors.

Validation:  .venv-research/bin/python -m research.chform_backend.hir_bridge
"""

from __future__ import annotations

import numpy as np

from .engine import LowRankState

TPHASE = np.exp(1j * np.pi / 4)


def parse_pauli(s: str, n: int):
    """'+X0*Z1*Y3' -> (pp, ax, az) with P = i^pp X(ax) Z(az); Y = i X Z."""
    pp = 0
    if s[0] in "+-":
        if s[0] == "-":
            pp = 2
        s = s[1:]
    ax = np.zeros(n, dtype=np.int64)
    az = np.zeros(n, dtype=np.int64)
    for term in s.split("*"):
        kind, q = term[0], int(term[1:])
        if kind == "X":
            ax[q] = 1
        elif kind == "Z":
            az[q] = 1
        elif kind == "Y":
            ax[q] = 1
            az[q] = 1
            pp = (pp + 1) % 4
        else:
            raise ValueError(f"unknown Pauli factor {term!r}")
    return pp, ax, az


def optimize(stim_text: str, episodic: bool = False):
    """parse -> trace -> HIR passes; returns (hir_dict, t_raw, t_live).

    episodic=True runs PeepholeFusionPass ONLY: it performs the full T-count
    reduction but -- unlike the default manager's StatevectorSqueezePass --
    does not hoist measurements, so episode boundaries stay contiguous in the
    op stream. Episodic execution needs that: hoisting interleaves episodes,
    the boundary rank-1 collapse stops firing, and the ill-conditioned
    post-projection resamples return (measured in bench_episodic.py)."""
    import clifft

    circ = clifft.parse(stim_text)
    hir = clifft.trace(circ)
    t_raw = hir.num_t_gates
    if episodic:
        pm = clifft._clifft_core.HirPassManager()
        pm.add(clifft._clifft_core.PeepholeFusionPass())
    else:
        pm = clifft.default_hir_pass_manager()
    pm.run(hir)
    return hir.as_dict(), t_raw, hir.num_t_gates


def run_hir_record(hir_dict: dict, record, backend: str = "chform",
                   recompress: bool = False, sparsify_budget: int | None = None,
                   rng: np.random.Generator | None = None,
                   norm_samples: int | None = None,
                   final_norm_rank1: bool = False,
                   return_state: bool = False):
    """Execute the optimized HIR with all measurements forced to `record`
    (indexed by meas_record_idx). Returns (P(record), chi_peak, n_t_applied).

    The state is |0^n> evolved by the T rotations; each forced Pauli
    measurement projects (I + (-1)^bit P)/2. P(record) is the squared norm of
    the final (unnormalized) state, with the engine's stability rescales
    divided back out.

    EXACT mode (default): chi doubles per T (validation scale), norm by
    materializing the term sum.

    EPISODIC APPROXIMATE mode (sparsify_budget=k): the T path auto-sparsifies
    whenever chi exceeds 2k (the engine's streaming trigger), and after every
    measurement the decomposition is resampled down to k terms if it exceeds
    the budget -- at episode boundaries (full magic collapse) the surviving
    terms are near-parallel, so this boundary resample is nearly free and
    resets the budget for the next episode. Both steps are the unbiased BGH
    resample, so P(record) remains an unbiased estimate; errors compound over
    boundaries (measured in bench_episodic.py). With norm_samples=L the final
    norm uses the non-materializing BGH estimator (norm_est.estimate_norm2)
    -- the full pipeline then never builds a 2^n object."""
    n = hir_dict["num_qubits"]
    s = LowRankState(n, backend=backend, sparsify_budget=sparsify_budget,
                     rng=rng)
    scale = 1.0  # product of stability rescale factors
    chi_peak = 1
    n_t = 0
    for op in hir_dict["ops"]:
        assert not op.get("is_hidden", False), "hidden measurement unsupported"
        # the string's +/- prefix and the `sign` field are the same datum;
        # parse_pauli already folds the prefix into pp
        pp, ax, az = parse_pauli(op["pauli_string"], n)
        if op["op_type"] == "T_GATE":
            phase = np.conj(TPHASE) if op["is_dagger"] else TPHASE
            s.rz_about_pauli(pp, ax, az, phase)
            n_t += 1
        elif op["op_type"] == "CONDITIONAL_PAULI":
            # classically-controlled Pauli (feedforward): with the record
            # forced, the condition is deterministic -- apply i^pp X(ax) Z(az)
            # to every term when the controlling bit is set
            if int(record[op["controlling_meas"]]):
                for t in s.terms:
                    for j in np.nonzero(az)[0]:
                        t.clifford_1q("Z", int(j))
                    for j in np.nonzero(ax)[0]:
                        t.clifford_1q("X", int(j))
                    t.scale(1j ** (pp % 4))
        elif op["op_type"] == "MEASURE":
            bit = int(record[op["meas_record_idx"]])
            scale *= s.measure_pauli_forced_fast(pp, ax, az, bit)
            if recompress:
                # canonical (materializing) recompression -- validation-scale
                # only; collapses parallel terms at episode boundaries so chi
                # tracks per-episode magic (the episodic-dispatch premise)
                s.recompress_dedup()
            elif sparsify_budget is not None and s.chi > 1:
                # episodic mode: NEVER resample after a projection (forced
                # measurements inflate ||c||_1/||psi|| and make the BGH
                # resample ill-conditioned). Instead, collapse EXACTLY when
                # the episode has fully collapsed (terms mutually parallel);
                # within-episode chi is capped by the T-time auto-trigger,
                # whose decomposition is extent-controlled from the collapsed
                # base and hence well-conditioned.
                s.collapse_if_parallel()
        else:
            raise ValueError(f"unsupported HIR op {op['op_type']}")
        chi_peak = max(chi_peak, s.chi)
    if final_norm_rank1:
        # VALID ONLY when the circuit ends fully collapsed (all magic measured
        # out; e.g. a conveyor's last round): the exact final state is rank 1,
        # so the amplitude-ratio collapse computes |sum_i c_i|^2 exactly in
        # O(chi n^2) -- no 2^n object, no estimator noise.
        s.collapse_to_rank1()
        norm2 = float(s.terms[0].norm2())
    elif norm_samples is not None:
        from .norm_est import estimate_norm2

        norm2 = estimate_norm2(s.terms, n, norm_samples,
                               s._rng if rng is None else rng)
    else:
        vec = s.statevector()
        norm2 = float(np.vdot(vec, vec).real)
    if return_state:
        return norm2 / (scale * scale), chi_peak, n_t, s, scale
    return norm2 / (scale * scale), chi_peak, n_t


def run_hir_record_gadgetized(hir_dict: dict, record, k: int,
                              rng: np.random.Generator | None = None,
                              exact: bool = False,
                              final_norm_rank1: bool = False,
                              backend: str = "chform"):
    """P(record) from the OPTIMIZED HIR with every T_GATE gadgetized -- the
    composition of the two previously separate routes: gadgetization ran on
    raw circuits (t_raw ancillas), the HIR bridge streamed T rotations
    (t_live but with mid-run resampling). This function teleports the
    COMPILED magic, so the single-shot product sampler and the analytic
    normalization run at the live T-count.

    A compiled T_GATE is a rotation about an arbitrary Pauli,
    R(P) = (I+P)/2 + e^{i theta}(I-P)/2. Its gadget generalizes the in-line
    one: prepare an ancilla in (|0> + e^{i theta}|1>)/sqrt2, entangle with
    Lambda_P(X_a) = H_a . (a-controlled-P) . H_a, force the ancilla to 0 --
    the surviving branch contributes exactly R(P)/sqrt2, and the entangler is
    entirely Clifford (controlled-P = S_a^pp then CZ/CX per Pauli factor).
    All magic is then ONE up-front product layer on t_live ancillas, sampled
    single-shot at budget k ~ 2^{0.228 t_live}/delta^2; the data evolution is
    Clifford + forced projections, so chi never exceeds k.

    Returns (P(record), chi_peak, t_live)."""
    n = hir_dict["num_qubits"]
    ops = hir_dict["ops"]
    tops = [op for op in ops if op["op_type"] == "T_GATE"]
    t = len(tops)
    N = n + t
    s = LowRankState(N, backend=backend)
    for i in range(t):
        s.clifford_1q("H", n + i)
    gates = [(n + i, np.conj(TPHASE) if op["is_dagger"] else TPHASE)
             for i, op in enumerate(tops)]
    if exact or k == 0:
        for q, ph in gates:
            s.rz_diag(q, ph)          # exact branching: chi -> 2^{t_live}
        norm_corr = 1.0
    else:
        s.inject_magic_layer(gates, k, rng or np.random.default_rng())
        l1sq = (2.0 * abs(0.5 + 1j * (np.sqrt(2) - 1) / 2)) ** (2 * t)
        norm_corr = 1.0 + (l1sq - 1.0) / k   # analytic E||omega||^2
    scale = 1.0
    chi_peak = s.chi
    ti = 0
    for op in ops:
        assert not op.get("is_hidden", False), "hidden measurement unsupported"
        pp, ax, az = parse_pauli(op["pauli_string"], N)
        if op["op_type"] == "T_GATE":
            a = n + ti
            ti += 1
            # Lambda_P(X_a) = H_a . (a-controlled-P) . H_a with
            # P = i^pp X(ax) Z(az); controlled-(i^pp) = S_a^pp. Applying the
            # CZ factors before the CX factors realizes controlled-[X(ax)Z(az)].
            s.clifford_1q("H", a)
            for _ in range(pp % 4):
                s.clifford_1q("S", a)
            for j in np.nonzero(az)[0]:
                s.cz(a, int(j))
            for j in np.nonzero(ax)[0]:
                s.cx(a, int(j))
            s.clifford_1q("H", a)
        elif op["op_type"] == "CONDITIONAL_PAULI":
            if int(record[op["controlling_meas"]]):
                for term in s.terms:
                    for j in np.nonzero(az)[0]:
                        term.clifford_1q("Z", int(j))
                    for j in np.nonzero(ax)[0]:
                        term.clifford_1q("X", int(j))
                    term.scale(1j ** (pp % 4))
        elif op["op_type"] == "MEASURE":
            bit = int(record[op["meas_record_idx"]])
            scale *= s.measure_pauli_forced_fast(pp, ax, az, bit)
        else:
            raise ValueError(f"unsupported HIR op {op['op_type']}")
        chi_peak = max(chi_peak, s.chi)
    # close the gadgets: force every ancilla to 0 (each contributes R(P)/sqrt2,
    # recovered by the 2^t factor below)
    ax0 = np.zeros(N, dtype=np.int64)
    for i in range(t):
        az_a = np.zeros(N, dtype=np.int64)
        az_a[n + i] = 1
        scale *= s.measure_pauli_forced_fast(0, ax0, az_a, 0)
    if final_norm_rank1:
        # valid when the record + gadget forcings fully determine the state
        # (all qubits measured): every surviving term is parallel
        s.collapse_to_rank1()
        norm2 = float(s.terms[0].norm2())
    else:
        vec = s.statevector()
        norm2 = float(np.vdot(vec, vec).real)
    return (2.0 ** t) * norm2 / (scale * scale) / norm_corr, chi_peak, t


def cross_record_probability(hir_dict, record, sparsify_budget, rng_a, rng_b):
    """Debiased P(record) for circuits that END FULLY COLLAPSED (rank-1 final
    state), via two INDEPENDENT episodic runs.

    The naive estimator ||Pi omega||^2 is biased upward by the sparsification
    variance (E||Pi omega||^2 = ||Pi psi||^2 + E||Pi(omega-psi)||^2). With two
    independent unbiased runs omega, omega', the cross product
    <Pi omega|Pi omega'> is unbiased. At a rank-1 ending both runs' states are
    proportional to the SAME exact direction sigma (the projector image), so
    the cross product reduces to amplitudes at one fixed support point x*:
        P = Re[ A(x*) conj(A'(x*)) ] / |sigma(x*)|^2 ,
    with A = <x*|omega_final>/scale and |sigma(x*)|^2 exact from either run's
    collapsed term. Returns (P_cross, chi_peak)."""
    _, chi_a, _, sa, scale_a = run_hir_record(
        hir_dict, record, sparsify_budget=sparsify_budget, rng=rng_a,
        final_norm_rank1=True, return_state=True)
    _, chi_b, _, sb, scale_b = run_hir_record(
        hir_dict, record, sparsify_budget=sparsify_budget, rng=rng_b,
        final_norm_rank1=True, return_state=True)
    ta, tb = sa.terms[0], sb.terms[0]
    xstar = ta.support_point()
    amp_a = ta.amplitude(xstar) / scale_a
    amp_b = tb.amplitude(xstar) / scale_b
    sigma_x2 = abs(ta.amplitude(xstar)) ** 2 / ta.norm2()  # exact: rank-1 dir
    p = float(np.real(amp_a * np.conj(amp_b))) / sigma_x2
    return p, max(chi_a, chi_b)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def ops_to_stim(ops, n, measure_all=True):
    lines = []
    for op in ops:
        if op[0] in ("CX", "CZ"):
            lines.append(f"{op[0]} {op[1]} {op[2]}")
        else:
            lines.append(f"{op[0]} {op[1]}")
    if measure_all:
        lines += [f"M {q}" for q in range(n)]
    return "\n".join(lines)


def validate():
    import clifft

    from .gadgetize import count_t, hidden_shift, random_cliffordT

    print("HIR bridge validation: backend on clifft-OPTIMIZED circuits")
    worst = 0.0
    rows = []
    cases = [("random", n, random_cliffordT(n, 5, tt, seed=70 + n))
             for n, tt in ((5, 8), (7, 12), (9, 16))]
    ops_hs, _ = hidden_shift(6, n_ccz=1, seed=4)
    cases.append(("hidden-shift", 6, ops_hs))
    for name, n, ops in cases:
        text = ops_to_stim(ops, n)
        prog = clifft.compile(text)
        samp = clifft.sample(prog, shots=64)
        recs = sorted(set(tuple(r) for r in
                          np.asarray(samp.measurements, dtype=bool).tolist()))[:12]
        pc = np.asarray(clifft.record_probabilities(
            prog, np.array(recs, dtype=bool)))
        hir_dict, t_raw, t_live = optimize(text)
        for rec, p_exact in zip(recs, pc):
            p_hir, chi_peak, n_t = run_hir_record(hir_dict, rec)
            worst = max(worst, abs(p_hir - p_exact))
        rows.append((name, n, t_raw, t_live, chi_peak))
        print(f"  {name:>12} n={n} t_raw={t_raw:>3} -> t_live={t_live:>3}  "
              f"chi_peak={chi_peak:>4} (=2^t_live: {chi_peak == 2 ** t_live})  "
              f"max|P_hir - P_clifft| so far {worst:.2e}")
    print(f"[{'OK' if worst < 1e-9 else 'FAIL'}] backend-on-optimized-HIR == "
          f"clifft record_probabilities (max abs err {worst:.2e})")
    assert worst < 1e-9
    print("Note: zero Clifford gate applications occurred in any backend run "
          "-- the compiler absorbed them all.")


if __name__ == "__main__":
    validate()
