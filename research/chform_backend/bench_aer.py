"""Reproducible qiskit-aer `extended_stabilizer` comparison.

The original findings quoted aer numbers with no committed script (flagged in
review as unauditable). This reproduces both claims at auditable scale:

  A. SAMPLING: shots/s on the nearest-neighbour IQP family, clifft.sample
     (exact; the measured program compiles to peak_rank ~ 1 on this family)
     vs extended_stabilizer (approximate CH-form Metropolis sampler).
     Settings disclosed inline; extended_stabilizer_mixing_time trades sample
     quality for speed and aer gives no quality certificate -- we run mixing=50
     (the old report's setting) and note the default is much larger.

  B. STRONG SIMULATION: estimating P(x) for 96 uniform targets of a DENSE IQP
     circuit at n=24 (true P ~ 1e-7): shot-frequency estimation needs ~1/P
     shots per expected hit, so a 30k-shot run hits ~0 targets; the CH-form
     backend computes the same quantities directly in ~1 s, and clifft's
     record_probabilities computes them EXACTLY in ~ms. (The honest summary is
     three-way: clifft owns exact P(x) while 2^{n/2} fits; our backend owns
     approximate P(x) beyond; extended_stabilizer cannot enter this task at
     anticoncentrated P.)

Run:  .venv-research/bin/python -m research.chform_backend.bench_aer
Requires qiskit + qiskit-aer (in .venv-research: qiskit 2.4.2, aer 0.17.2).
"""

from __future__ import annotations

import time

import numpy as np


def nn_iqp_lines(n, seed):
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(q, q + 1) for q in range(n - 1) if r.random() < 0.6]
    return dag, czs


def dense_iqp_lines(n, seed):
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(a, b) for a in range(n) for b in range(a + 1, n) if r.random() < 0.5]
    return dag, czs


def qiskit_iqp(n, dag, czs):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.h(q)
    for q in range(n):
        (qc.tdg if dag[q] else qc.t)(q)
    for a, b in czs:
        qc.cz(a, b)
    for q in range(n):
        qc.h(q)
    qc.measure(range(n), range(n))
    return qc


def clifft_iqp(n, dag, czs):
    import clifft

    lines = [f"H {q}" for q in range(n)]
    lines += [f"{'T_DAG' if dag[q] else 'T'} {q}" for q in range(n)]
    lines += [f"CZ {a} {b}" for a, b in czs]
    lines += [f"H {q}" for q in range(n)]
    lines += [f"M {q}" for q in range(n)]
    return clifft.compile("\n".join(lines))


def bench_sampling():
    import clifft
    from qiskit_aer import AerSimulator

    SHOTS = 2000
    MIXING = 50  # aer default is far larger; smaller = faster, lower quality
    print(f"A. sampling throughput, nn-IQP, {SHOTS} shots "
          f"(aer: extended_stabilizer, mixing_time={MIXING}, approx err 0.1)")
    print(f"{'n':>4} {'clifft shots/s':>15} {'aer shots/s':>12} {'ratio':>10} {'clifft k':>8}")
    sim = AerSimulator(
        method="extended_stabilizer",
        extended_stabilizer_metropolis_mixing_time=MIXING,
        extended_stabilizer_approximation_error=0.1,
    )
    for n in (10, 16, 22, 28):
        dag, czs = nn_iqp_lines(n, 100 + n)
        prog = clifft_iqp(n, dag, czs)
        t0 = time.time()
        clifft.sample(prog, shots=SHOTS)
        cl = SHOTS / (time.time() - t0)
        qc = qiskit_iqp(n, dag, czs)
        t0 = time.time()
        sim.run(qc, shots=SHOTS).result()
        ae = SHOTS / (time.time() - t0)
        print(f"{n:>4} {cl:>15.0f} {ae:>12.1f} {cl / ae:>9.0f}x {prog.peak_rank:>8}")


def bench_strong():
    import clifft
    from qiskit_aer import AerSimulator

    n, SHOTS = 24, 30000
    dag, czs = dense_iqp_lines(n, 500 + n)
    prog = clifft_iqp(n, dag, czs)
    rng = np.random.default_rng(7)
    targets = sorted(set(int(x) for x in rng.integers(0, 2**n, size=96)))
    records = np.array([[(x >> q) & 1 for q in range(n)] for x in targets], dtype=bool)

    t0 = time.time()
    p_exact = np.asarray(clifft.record_probabilities(prog, records))
    t_cl = time.time() - t0

    sim = AerSimulator(
        method="extended_stabilizer",
        extended_stabilizer_metropolis_mixing_time=50,
        extended_stabilizer_approximation_error=0.1,
    )
    qc = qiskit_iqp(n, dag, czs)
    t0 = time.time()
    res = sim.run(qc, shots=SHOTS).result().get_counts()
    t_aer = time.time() - t0
    # aer bitstrings are big-endian (clbit n-1 first)
    hits = sum(
        res.get(format(x, f"0{n}b")[::-1], 0) + res.get(format(x, f"0{n}b"), 0)
        for x in targets
    )
    print(f"\nB. strong simulation, dense IQP n={n}, 96 uniform targets "
          f"(true P: mean {p_exact.mean():.1e}, max {p_exact.max():.1e})")
    print(f"  clifft record_probabilities (exact): {t_cl * 1e3:.1f} ms")
    print(f"  aer ext_stab {SHOTS} shots: {t_aer:.0f} s, target hits = {hits} "
          f"(need ~1/P ~ {1 / max(p_exact.mean(), 1e-300):.0e} shots per expected hit)")


if __name__ == "__main__":
    bench_sampling()
    bench_strong()
