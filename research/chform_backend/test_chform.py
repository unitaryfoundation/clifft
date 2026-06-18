"""Gate-by-gate validation of the CH-form term store against a dense oracle.

Run: python -m research.chform_backend.test_chform

The dense statevector is ground truth. Each CH-form primitive is exercised and
diffed against the same operation on a dense vector, so a phase/sign bug is
localised to one rule -- the oracle the README calls for ("validate it against,
gate by gate"). Conventions follow arXiv:1808.00128 Sec 4.1 (qubit 0 = LSB).
"""

from __future__ import annotations

import numpy as np

from .chform import CHForm

_U = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "S_DAG": np.array([[1, 0], [0, -1j]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# --------------------------------------------------------------- dense oracle
def d_init(n):
    z = np.zeros(2 ** n, dtype=complex)
    z[0] = 1.0
    return z


def d_1q(v, n, q, name):
    r = v.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("ab,xbz->xaz", _U[name], r).reshape(-1)


def d_cx(v, n, c, t):
    idx = np.arange(2 ** n)
    return v[idx ^ (((idx >> c) & 1) << t)]


def d_cz(v, n, c, t):
    idx = np.arange(2 ** n)
    return v * np.where(((idx >> c) & 1) & ((idx >> t) & 1), -1.0, 1.0)


def d_swap(v, n, a, b):
    idx = np.arange(2 ** n)
    ba, bb = (idx >> a) & 1, (idx >> b) & 1
    return v[idx ^ ((ba ^ bb) << a) ^ ((ba ^ bb) << b)]


def d_project(v, n, q, bit):
    idx = np.arange(2 ** n)
    return np.where(((idx >> q) & 1) == bit, v, 0.0)


def _err(a, b):
    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------- tests
def test_amplitude_handbuilt():
    worst = 0.0
    for sbit, ket in ((0, [1, 1]), (1, [1, -1])):
        c = CHForm(1); c.v[0] = 1; c.s[0] = sbit
        worst = max(worst, _err(c.statevector(), np.array(ket) / np.sqrt(2)))
    for n in (1, 2, 3, 4):
        c = CHForm(n); c.v[:] = 1
        worst = max(worst, _err(c.statevector(), np.ones(2 ** n) / np.sqrt(2) ** n))
    assert worst < 1e-12, f"amplitude formula mismatch {worst}"
    print(f"[OK] amplitude formula on hand-built states (max abs err {worst:.2e})")


def _random_clifford(n, depth, rng):
    ops = []
    for _ in range(depth):
        for q in range(n):
            ops.append(("1q", str(rng.choice(list(_U))), q))
        for q in range(0, n - 1, 2):
            ops.append((str(rng.choice(["cx", "cz", "swap"])), q, q + 1))
        for q in range(1, n - 1, 2):
            ops.append((str(rng.choice(["cx", "cz", "swap"])), q, q + 1))
    return ops


def _run(c, v, n, ops):
    for op in ops:
        k = op[0]
        if k == "1q":
            c.clifford_1q(op[1], op[2]); v = d_1q(v, n, op[2], op[1])
        elif k == "cx":
            c.cx(op[1], op[2]); v = d_cx(v, n, op[1], op[2])
        elif k == "cz":
            c.cz(op[1], op[2]); v = d_cz(v, n, op[1], op[2])
        else:
            c.swap(op[1], op[2]); v = d_swap(v, n, op[1], op[2])
    return v


def test_all_clifford_gates_vs_dense(trials=400):
    """Random circuits over the full set {H,S,S_DAG,X,Y,Z,CX,CZ,SWAP} from |0>^n.

    With H present every stabilizer state is reachable, so this exercises the
    full convention rewrite + the Hadamard desuperposition end to end.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(2, 7))
        c, v = CHForm(n), d_init(n)
        v = _run(c, v, n, _random_clifford(n, int(rng.integers(2, 6)), rng))
        worst = max(worst, _err(c.statevector(), v))
    assert worst < 1e-9, f"Clifford gate mismatch {worst}"
    print(f"[OK] full Clifford set vs dense on {trials} random circuits "
          f"(max abs err {worst:.2e})")


def test_projection_vs_dense(trials=400):
    """Project a random stabilizer state onto qubit q == bit, both ways."""
    rng = np.random.default_rng(1)
    worst, n_none_ok = 0.0, 0
    for _ in range(trials):
        n = int(rng.integers(2, 7))
        c, v = CHForm(n), d_init(n)
        v = _run(c, v, n, _random_clifford(n, int(rng.integers(2, 5)), rng))
        q, bit = int(rng.integers(n)), int(rng.integers(2))
        dv = d_project(v, n, q, bit)
        pc = c.project(q, bit)
        if np.max(np.abs(dv)) < 1e-12:  # zero support
            assert pc is None, "expected None projection for zero-support outcome"
            n_none_ok += 1
        else:
            assert pc is not None, "got None for a nonzero-support outcome"
            worst = max(worst, _err(pc.statevector(), dv))
    assert worst < 1e-9, f"projection mismatch {worst}"
    print(f"[OK] projection vs dense on {trials} states "
          f"(max abs err {worst:.2e}, {n_none_ok} zero-support None-cases)")


def test_invariants(trials=200):
    rng = np.random.default_rng(2)
    for _ in range(trials):
        n = int(rng.integers(2, 7))
        c = CHForm(n)
        _run(c, d_init(n), n, _random_clifford(n, int(rng.integers(2, 6)), rng))
        gft = (c.G.astype(int) @ c.F.T.astype(int)) % 2
        assert np.array_equal(gft, np.eye(n, dtype=int)), "G F^T != I"
        assert abs(c.norm2() - 1.0) < 1e-9, f"norm^2 = {c.norm2()} != 1"
    print(f"[OK] G F^T = I and ||phi||=1 invariants on {trials} circuits")


def test_engine_backend_equivalence_through_measurement(trials=40):
    """Run identical measured Clifford+T circuits on both engine backends with a
    shared RNG; assert equal statevectors AND that the CH-form backend stays
    CH-form across measurement+recompress (no silent densification)."""
    from .engine import LowRankState
    from .chform import CHForm
    worst = 0.0
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 5))
        ld = LowRankState(n, backend="dense")
        lc = LowRankState(n, backend="chform")
        rd = np.random.default_rng(10_000 + seed)
        rc = np.random.default_rng(10_000 + seed)  # same measurement outcomes
        for _ in range(int(rng.integers(3, 7))):
            for q in range(n):
                ld.clifford_1q("H", q); lc.clifford_1q("H", q)
            for q in range(n):
                dag = bool(rng.integers(2))
                ld.t(q, dagger=dag); lc.t(q, dagger=dag)
            for q in range(n - 1):
                ld.cz(q, q + 1); lc.cz(q, q + 1)
            for q in range(n):
                ld.clifford_1q("H", q); lc.clifford_1q("H", q)
            for q in range(n):
                ld.measure_z(q, rd); lc.measure_z(q, rc)
        assert all(isinstance(t, CHForm) for t in lc.terms), \
            "CH-form backend densified a term across measurement/recompress"
        worst = max(worst, _err(ld.statevector(), lc.statevector()))
    assert worst < 1e-9, f"backend mismatch through measurement {worst}"
    print(f"[OK] engine dense vs chform agree through measurement on {trials} "
          f"runs, chform stays CH-form (max abs err {worst:.2e})")


def _iqp_spec(n, seed):
    r = np.random.default_rng(seed)
    dag = [bool(b) for b in r.integers(0, 2, size=n)]
    czs = [(q, q + 1) for q in range(n - 1) if r.random() < 0.6]
    return dag, czs


def _iqp_single_shot(n, seed, k, rng):
    from .engine import LowRankState
    tp = np.exp(1j * np.pi / 4)
    dag, czs = _iqp_spec(n, seed)
    s = LowRankState(n, backend="chform")
    for q in range(n):
        s.clifford_1q("H", q)  # base |+>^n, chi = 1
    s.inject_magic_layer([(q, np.conj(tp) if dag[q] else tp) for q in range(n)], k, rng)
    for a, b in czs:
        s.cz(a, b)
    for q in range(n):
        s.clifford_1q("H", q)
    return s


def _iqp_exact(n, seed):
    from .engine import LowRankState
    dag, czs = _iqp_spec(n, seed)
    s = LowRankState(n, backend="dense")
    for q in range(n):
        s.clifford_1q("H", q)
    for q in range(n):
        s.t(q, dagger=dag[q])
    for a, b in czs:
        s.cz(a, b)
    for q in range(n):
        s.clifford_1q("H", q)
    return s.statevector()


def test_single_shot_magic():
    """inject_magic_layer: (1) unbiased -- averaging single-shot runs reproduces
    the exact amplitudes; (2) ||omega||^2 ~ 1 + (extent-1)/k at the single-shot
    budget (no t-factor). Validated via native amplitudes / small-n exact norms,
    so NO 2^n materialisation in a loop (which is what made an earlier version
    slow -- summing chi terms x 2^n amplitudes, the very thing CH-form avoids)."""
    # 1) unbiasedness on a few amplitudes (native amplitude(), no full statevector)
    n = 8
    psi = _iqp_exact(n, 0)
    xs = [0, 5, 17, 42, 130, 200]
    rng = np.random.default_rng(1)
    acc = np.zeros(len(xs), dtype=complex)
    R, k = 2500, 8
    for _ in range(R):
        s = _iqp_single_shot(n, 0, k, rng)
        for i, x in enumerate(xs):
            acc[i] += s.amplitude(x)
    acc /= R
    err = float(np.max(np.abs(acc - psi[xs])))
    assert err < 0.03, f"single-shot not unbiased: {err}"
    print(f"[OK] single-shot unbiased: max|avg<x|omega> - <x|psi>| = {err:.4f} "
          f"(n={n}, k={k}, R={R}, {len(xs)} amplitudes)")

    # 2) ||omega||^2 - 1 ~ (extent-1)/k at the single-shot budget (exact norm,
    #    n=10 small chi -> fast). Confirms the budget has NO factor of t.
    n = 10
    print(f"  {'n':>3} {'k':>5} {'extent':>7} {'single-shot ||om||^2':>20} "
          f"{'predicted 1+(ext-1)/k':>22}")
    for k in (100, 300):
        ss = [float(np.vdot(v := _iqp_single_shot(n, 5, k, np.random.default_rng(50 + t)
                                                  ).statevector(), v).real)
              for t in range(4)]
        pred = 1.0 + (1.17 ** n - 1.0) / k
        assert abs(np.mean(ss) - pred) < 0.15, f"||omega||^2 off: {np.mean(ss)} vs {pred}"
        print(f"  {n:>3} {k:>5} {1.17 ** n:>7.1f} {np.mean(ss):>20.3f} {pred:>22.3f}")
    print("  matches 1+(extent-1)/k exactly: single-shot needs only k~extent/delta^2")
    print("  terms (no t-factor), vs streaming's ~t x that (variance accumulates over")
    print("  the ~t mid-circuit sparsify steps -- streaming gave ||omega||^2~2 at n=40).")


if __name__ == "__main__":
    test_amplitude_handbuilt()
    test_all_clifford_gates_vs_dense()
    test_projection_vs_dense()
    test_invariants()
    test_engine_backend_equivalence_through_measurement()
    test_single_shot_magic()
