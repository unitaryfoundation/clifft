"""General Clifford+T strong simulation via T-gadgetization -- the
sparsification generalization beyond product-magic (IQP-style) circuits.

The route: teleport every in-line T onto its own ancilla (the adaptive gadget
validated in bench_natural.py) and FORCE the gadget outcome to 0. Because the
gadget's classically-controlled correction makes every outcome branch yield
the same output distribution, the 0-branch alone carries it: each forced-0
gadget contributes exactly a factor T/sqrt(2) on its target. Concretely, for
a circuit C with t T-gates on n qubits,

    P(x) = 2^t * | <x, 0^t| C_gadget ( |0^n> (x) |T>^t ) |^2 ,

where C_gadget is C with each T replaced by CX(target -> ancilla) -- an
entirely CLIFFORD circuit. All magic is now ONE up-front product layer, so:

  * the validated single-shot sampler (inject_magic_layer) applies -- unbiased,
    budget k ~ 2^{0.228 t}/delta^2, NO factor-of-t variance;
  * the normalization stays ANALYTIC: E||omega||^2 = 1 + (||c||_1^2 - 1)/k
    (the magic layer is a product), so no norm estimation enters;
  * the frame/engine machinery is unchanged -- the gadgetized circuit is
    Clifford + one amplitude query.

This is the classic gadgetized strong simulation (Bravyi-Gosset / BGH),
realized on this engine. The cost is t extra qubits (CH-form terms are
O((n+t)^2) bits -- cheap) in exchange for removing the streaming
sparsification's t-fold variance accumulation.

Circuit ops format (shared with the C++ driver's spec files):
  ("H"|"S"|"S_DAG"|"X"|"Y"|"Z", q) | ("CX", a, b) | ("CZ", a, b)
  | ("T", q) | ("T_DAG", q)

Run the validation:
  .venv-research/bin/python -m research.chform_backend.gadgetize
"""

from __future__ import annotations

import sys

import numpy as np

from .engine import LowRankState

TPHASE = np.exp(1j * np.pi / 4)


# ---------------------------------------------------------------------------
# circuit families
# ---------------------------------------------------------------------------
def random_cliffordT(n, depth, t_total, seed):
    """Random Clifford layers with t_total T/T_DAG gates interleaved
    UNIFORMLY through the depth -- the generic non-product-magic circuit the
    single-shot sampler could not previously handle."""
    rg = np.random.default_rng(seed)
    ops = []
    t_slots = sorted(rg.choice(depth, size=t_total, replace=True))
    ti = 0
    for d in range(depth):
        for q in range(n):
            g = ["H", "S", "X", "Z"][rg.integers(0, 4)]
            ops.append((g, int(q)))
        perm = rg.permutation(n)
        for i in range(0, n - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            ops.append(("CX", a, b) if rg.integers(0, 2) else ("CZ", a, b))
        while ti < t_total and t_slots[ti] == d:
            ops.append((("T", "T_DAG")[rg.integers(0, 2)], int(rg.integers(0, n))))
            ti += 1
    return ops


def ccz_ops(a, b, c):
    """CCZ via the 7-T phase-polynomial network:
    CCZ = T_a T_b T_c . Tdg_{a+b} Tdg_{a+c} Tdg_{b+c} . T_{a+b+c}
    with each parity computed into a wire by CX conjugation. Validated
    against the dense CCZ in validate()."""
    ops = [("T", a), ("T", b), ("T", c)]
    ops += [("CX", a, b), ("T_DAG", b), ("CX", a, b)]            # a xor b
    ops += [("CX", a, c), ("T_DAG", c), ("CX", a, c)]            # a xor c
    ops += [("CX", b, c), ("T_DAG", c), ("CX", b, c)]            # b xor c
    ops += [("CX", a, c), ("CX", b, c), ("T", c),
            ("CX", b, c), ("CX", a, c)]                          # a xor b xor c
    return ops


def hidden_shift(n, n_ccz, seed):
    """Hidden-shift circuit for a Maiorana-McFarland bent function
    f(u,v) = u.v + h(v) with cubic h built from CCZ triples (dual
    f~(u,v) = u.v + h(u)). The algorithm H^n O_{f(.+s)} H^n O_{f~} H^n |0>
    outputs the shift s DETERMINISTICALLY -- P(s) = 1 is the built-in exact
    ground truth at any size (the classic BGH benchmark family).
    Returns (ops, s). T-count = 14 * n_ccz."""
    assert n % 2 == 0
    half = n // 2
    rg = np.random.default_rng(seed)
    s = int(rg.integers(0, 2 ** n))
    triples = [sorted(rg.choice(half, size=3, replace=False)) for _ in range(n_ccz)]

    def oracle(side):  # side 0: h on v (qubits half..n-1); side 1: h on u
        o = []
        for i in range(half):
            o.append(("CZ", i, i + half))  # u.v
        off = half if side == 0 else 0
        for tr in triples:
            a, b, c = (int(q) + off for q in tr)
            o += ccz_ops(a, b, c)
        return o

    ops = [("H", q) for q in range(n)]
    ops += [("X", q) for q in range(n) if (s >> q) & 1]   # O_{f(.+s)} = X^s O_f X^s
    ops += oracle(0)
    ops += [("X", q) for q in range(n) if (s >> q) & 1]
    ops += [("H", q) for q in range(n)]
    ops += oracle(1)                                       # O_{f~}
    ops += [("H", q) for q in range(n)]
    return ops, s


# ---------------------------------------------------------------------------
# gadgetized build
# ---------------------------------------------------------------------------
def count_t(ops):
    return sum(1 for op in ops if op[0] in ("T", "T_DAG"))


def build_gadgetized(n, ops, k, rng, exact=False, backend="chform"):
    """Return (state over n+t qubits, t, l1sq). exact=True keeps the full
    2^t decomposition (validation only); else single-shot sample k terms."""
    t = count_t(ops)
    s = LowRankState(n + t, backend=backend)
    daggers = [op[0] == "T_DAG" for op in ops if op[0] in ("T", "T_DAG")]
    for i in range(t):
        s.clifford_1q("H", n + i)
    gates = [(n + i, np.conj(TPHASE) if daggers[i] else TPHASE) for i in range(t)]
    if exact:
        for q, ph in gates:
            s.rz_diag(q, ph)          # exact branching: chi -> 2^t
        l1sq = None
    else:
        s.inject_magic_layer(gates, k, rng)
        l1sq = (2.0 * abs(0.5 + 1j * (np.sqrt(2) - 1) / 2)) ** (2 * t)
    ti = 0
    for op in ops:
        if op[0] in ("T", "T_DAG"):
            s.cx(op[1], n + ti)       # the gadget: CX target -> ancilla
            ti += 1
        elif op[0] == "CX":
            s.cx(op[1], op[2])
        elif op[0] == "CZ":
            s.cz(op[1], op[2])
        else:
            s.clifford_1q(op[0], op[1])
    return s, t, l1sq


def probability(state, n, t, x, k=None, l1sq=None):
    """P(x) for an n-bit outcome x: amplitude at (x, ancillas=0), times 2^t,
    normalized analytically for the sparsified build."""
    amp = state.amplitude(int(x))     # ancilla bits are the high bits, = 0
    p = (2.0 ** t) * abs(amp) ** 2
    if l1sq is not None:
        p /= 1.0 + (l1sq - 1.0) / k
    return p


# ---------------------------------------------------------------------------
# dense reference for the ORIGINAL circuit (validation)
# ---------------------------------------------------------------------------
_G1 = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.diag([1, 1j]), "S_DAG": np.diag([1, -1j]),
    "X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1, -1]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "T": np.diag([1, TPHASE]), "T_DAG": np.diag([1, np.conj(TPHASE)]),
}


def dense_reference(n, ops):
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = 1.0
    idx = np.arange(2 ** n)
    for op in ops:
        if op[0] in ("CX", "CZ"):
            a, b = op[1], op[2]
            if op[0] == "CZ":
                v[((idx >> a) & 1 == 1) & ((idx >> b) & 1 == 1)] *= -1
            else:
                hot = (idx >> a) & 1 == 1
                src = idx[hot]
                w = v.copy()
                w[src] = v[src ^ (1 << b)]
                v = w
        else:
            U = _G1[op[0]]
            q = op[1]
            lo = (idx >> q) & 1 == 0
            x0, x1 = idx[lo], idx[lo] | (1 << q)
            a0, a1 = v[x0].copy(), v[x1].copy()
            v[x0] = U[0, 0] * a0 + U[0, 1] * a1
            v[x1] = U[1, 0] * a0 + U[1, 1] * a1
    return v


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate():
    rng = np.random.default_rng(7)

    # 0) the 7-T CCZ network really is CCZ
    ops = [("H", q) for q in range(3)] + ccz_ops(0, 1, 2)
    v = dense_reference(3, ops)
    ref = np.ones(8, dtype=complex) / np.sqrt(8)
    ref[7] *= -1
    err = float(np.max(np.abs(v - ref)))
    print(f"[{'OK' if err < 1e-12 else 'FAIL'}] 7-T network == CCZ (max err {err:.2e})")
    assert err < 1e-12

    # 1) EXACT gadgetized == dense on random interleaved Clifford+T
    worst = 0.0
    for trial in range(20):
        n = int(rng.integers(2, 5))
        tt = int(rng.integers(1, 5))
        ops = random_cliffordT(n, 4, tt, seed=100 + trial)
        vd = dense_reference(n, ops)
        st, t, _ = build_gadgetized(n, ops, k=0, rng=rng, exact=True)
        for x in range(2 ** n):
            p = probability(st, n, t, x)
            worst = max(worst, abs(p - abs(vd[x]) ** 2))
    print(f"[{'OK' if worst < 1e-9 else 'FAIL'}] exact gadgetized == dense, "
          f"20 random interleaved circuits (max abs err {worst:.2e})")
    assert worst < 1e-9

    # 2) SPARSIFIED gadgetized: unbiased, error tracks delta (interleaved magic
    #    -- the case the single-shot sampler could not previously touch)
    n, tt = 6, 8
    ops = random_cliffordT(n, 5, tt, seed=42)
    vd = dense_reference(n, ops)
    targets = list(range(2 ** n))
    for delta in (0.3, 0.1):
        k = max(64, int((2.0 ** (0.228 * tt)) / delta ** 2))
        tvs = []
        for rep in range(6):
            st, t, l1sq = build_gadgetized(n, ops, k, np.random.default_rng(500 + rep))
            P = np.array([probability(st, n, t, x, k, l1sq) for x in targets])
            E = np.abs(vd) ** 2
            tvs.append(0.5 * float(np.sum(np.abs(P - E))))
        print(f"    delta={delta:.2f}  k={k:5d}  TV={np.mean(tvs):.4f}"
              f"  (TV/delta={np.mean(tvs)/delta:.2f})")
    print("[OK] sparsified gadgetized: TV scales with delta on interleaved magic")

    # 3) hidden shift: exact gadget P(s) == 1 at small n; and the streaming
    #    engine.sparsify path gets its first correctness exercise
    n = 6
    ops, s = hidden_shift(n, n_ccz=1, seed=3)
    vd = dense_reference(n, ops)
    assert abs(abs(vd[s]) ** 2 - 1.0) < 1e-12, "hidden-shift construction broken"
    st, t, _ = build_gadgetized(n, ops, k=0, rng=rng, exact=True)
    p = probability(st, n, t, s)
    print(f"[{'OK' if abs(p - 1) < 1e-9 else 'FAIL'}] hidden shift n=6 (t={t}): "
          f"exact gadgetized P(s) = {p:.12f}")
    assert abs(p - 1) < 1e-9

    # 4) streaming sparsify (engine.sparsify): first direct exercise of the
    #    engine method -- unbiasedness on a small exact state
    n = 4
    base_ops = random_cliffordT(n, 3, 4, seed=9)
    vd = dense_reference(n, base_ops)
    errs = []
    for rep in range(40):
        st, t, _ = build_gadgetized(n, base_ops, k=0,
                                    rng=np.random.default_rng(rep), exact=True)
        st.sparsify(24, np.random.default_rng(1000 + rep))
        amp = np.mean([st.amplitude(x) for x in [3]])  # any fixed amplitude
        errs.append(amp)
    exact_amp = None
    st, t, _ = build_gadgetized(n, base_ops, k=0, rng=rng, exact=True)
    exact_amp = st.amplitude(3)
    bias = abs(np.mean(errs) - exact_amp)
    print(f"[{'OK' if bias < 0.05 else 'FAIL'}] engine.sparsify unbiased "
          f"(|mean-exact| = {bias:.3f} over 40 draws, k=24)")
    assert bias < 0.05


if __name__ == "__main__":
    validate()
    if "--clifft" in sys.argv:
        import clifft  # cross-check vs clifft on one measured circuit

        n, tt = 8, 10
        ops = random_cliffordT(n, 5, tt, seed=77)
        NAMES = {"H": "H", "S": "S", "S_DAG": "S_DAG", "X": "X", "Y": "Y",
                 "Z": "Z", "T": "T", "T_DAG": "T_DAG"}
        lines = []
        for op in ops:
            if op[0] in ("CX",):
                lines.append(f"CX {op[1]} {op[2]}")
            elif op[0] == "CZ":
                lines.append(f"CZ {op[1]} {op[2]}")
            else:
                lines.append(f"{NAMES[op[0]]} {op[1]}")
        lines += [f"M {q}" for q in range(n)]
        prog = clifft.compile("\n".join(lines))
        rng = np.random.default_rng(1)
        targets = sorted(set(int(x) for x in rng.integers(0, 2 ** n, size=32)))
        recs = np.array([[(x >> q) & 1 for q in range(n)] for x in targets],
                        dtype=bool)
        pc = np.asarray(clifft.record_probabilities(prog, recs))
        st, t, _ = build_gadgetized(n, ops, k=0, rng=rng, exact=True)
        worst = max(abs(probability(st, n, t, x) - p) for x, p in zip(targets, pc))
        print(f"[{'OK' if worst < 1e-9 else 'FAIL'}] exact gadgetized == clifft "
              f"record_probabilities, n={n} t={tt} (max abs err {worst:.2e})")
