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


def optimize(stim_text: str):
    """parse -> trace -> default HIR passes; returns (hir_dict, t_raw, t_live)."""
    import clifft

    circ = clifft.parse(stim_text)
    hir = clifft.trace(circ)
    t_raw = hir.num_t_gates
    pm = clifft.default_hir_pass_manager()
    pm.run(hir)
    return hir.as_dict(), t_raw, hir.num_t_gates


def run_hir_record(hir_dict: dict, record, backend: str = "chform",
                   recompress: bool = False):
    """Execute the optimized HIR with all measurements forced to `record`
    (indexed by meas_record_idx). Returns (P(record), chi_peak, n_t_applied).

    The state is |0^n> evolved by the T rotations; each forced Pauli
    measurement projects (I + (-1)^bit P)/2. P(record) is the squared norm of
    the final (unnormalized) state, with the engine's stability rescales
    divided back out. Norm evaluation materializes the term sum -- validation
    scale (the C++ scale path is future work; see findings)."""
    n = hir_dict["num_qubits"]
    s = LowRankState(n, backend=backend)
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
            chi_peak = max(chi_peak, s.chi)
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
        else:
            raise ValueError(f"unsupported HIR op {op['op_type']}")
    vec = s.statevector()
    norm2 = float(np.vdot(vec, vec).real)
    return norm2 / (scale * scale), chi_peak, n_t


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
