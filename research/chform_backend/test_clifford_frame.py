"""Validate CliffordFrame's inverse-conjugation tableau against dense linear
algebra: for random Clifford circuits, frame.conj_Z(q) must equal F^-1 Z_q F and
frame.conj_X(q) must equal F^-1 X_q F, computed densely.

Run: python -m research.chform_backend.test_clifford_frame
"""

from __future__ import annotations

import numpy as np

from .clifford_frame import CliffordFrame
from .chform import CHForm

_U = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "S_DAG": np.array([[1, 0], [0, -1j]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_I2 = np.eye(2, dtype=complex)
_IPOW = np.array([1, 1j, -1, -1j], dtype=complex)


def _dense_1q(n, q, U):
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for x in range(2 ** n):
        b = (x >> q) & 1
        for bp in range(2):
            if U[bp, b]:
                y = (x & ~(1 << q)) | (bp << q)
                M[y, x] += U[bp, b]
    return M


def _dense_cx(n, c, t):
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for x in range(2 ** n):
        M[x ^ (((x >> c) & 1) << t), x] = 1.0
    return M


def _dense_cz(n, c, t):
    M = np.diag([(-1.0) ** (((x >> c) & 1) & ((x >> t) & 1)) for x in range(2 ** n)]).astype(complex)
    return M


def _dense_swap(n, a, b):
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for x in range(2 ** n):
        ba, bb = (x >> a) & 1, (x >> b) & 1
        M[x ^ ((ba ^ bb) << a) ^ ((ba ^ bb) << b), x] = 1.0
    return M


def _dense_gate(n, name, qs):
    if name in _U:
        return _dense_1q(n, qs[0], _U[name])
    if name == "CX":
        return _dense_cx(n, qs[0], qs[1])
    if name == "CZ":
        return _dense_cz(n, qs[0], qs[1])
    if name == "SWAP":
        return _dense_swap(n, qs[0], qs[1])
    raise ValueError(name)


def _pauli_dense(n, phase, x, z):
    ops = []
    for q in range(n):  # MSB..LSB kron so that bit q = (idx>>q)&1
        op = (_U["X"] if x[q] else _I2) @ (_U["Z"] if z[q] else _I2)
        ops.append(op)
    full = ops[n - 1]
    for q in range(n - 2, -1, -1):
        full = np.kron(full, ops[q])
    return _IPOW[phase % 4] * full


def test_frame_conjugation(trials=400):
    rng = np.random.default_rng(0)
    names1 = ["H", "S", "S_DAG", "X", "Y", "Z"]
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(1, 6))
        frame = CliffordFrame(n)
        F = np.eye(2 ** n, dtype=complex)
        for _ in range(int(rng.integers(2, 12))):
            if n >= 2 and rng.random() < 0.4:
                name = str(rng.choice(["CX", "CZ", "SWAP"]))
                a, b = rng.choice(n, size=2, replace=False)
                qs = (int(a), int(b))
            else:
                name = str(rng.choice(names1))
                qs = (int(rng.integers(n)),)
            frame.apply_left(name, *qs)
            F = _dense_gate(n, name, qs) @ F           # F <- gate . F
        Finv = F.conj().T
        for q in range(n):
            for which, conj in (("Z", frame.conj_Z), ("X", frame.conj_X)):
                p, xb, zb = conj(q)
                got = _pauli_dense(n, p, xb, zb)
                P = _dense_1q(n, q, _U[which])
                ref = Finv @ P @ F                     # F^-1 P_q F
                worst = max(worst, float(np.max(np.abs(got - ref))))
    assert worst < 1e-9, f"frame conjugation mismatch {worst}"
    print(f"[OK] CliffordFrame conj_Z/conj_X == F^-1 P F on {trials} random "
          f"Clifford circuits (max abs err {worst:.2e})")


def test_symplectic_invariant(trials=200):
    """xc[q], zc[q] must anticommute (conjugates of X_q, Z_q); xc[q],zc[r!=q]
    commute -- the symplectic basis is preserved."""
    from .clifford_frame import _pauli_mul
    rng = np.random.default_rng(1)

    def commute(p, ax, az, bx, bz):  # 0 if commute, 1 if anticommute
        return (int(np.dot(az, bx) + np.dot(bz, ax)) % 2)

    for _ in range(trials):
        n = int(rng.integers(2, 6))
        frame = CliffordFrame(n)
        for _ in range(int(rng.integers(3, 12))):
            if rng.random() < 0.4:
                a, b = rng.choice(n, size=2, replace=False)
                frame.apply_left(str(rng.choice(["CX", "CZ", "SWAP"])), int(a), int(b))
            else:
                frame.apply_left(str(rng.choice(["H", "S", "S_DAG", "X", "Y", "Z"])),
                                 int(rng.integers(n)))
        for q in range(n):
            assert commute(0, frame.xc_x[q], frame.xc_z[q], frame.zc_x[q], frame.zc_z[q]) == 1
            for r in range(n):
                if r != q:
                    assert commute(0, frame.xc_x[q], frame.xc_z[q],
                                   frame.zc_x[r], frame.zc_z[r]) == 0
    print(f"[OK] symplectic basis preserved on {trials} random circuits")


def test_frame_engine_vs_plain(trials=60):
    """Frame-mode engine == plain engine (both exact) on unitary Clifford+T
    circuits, and the frame pays ZERO per-term Clifford ops (the free-Clifford
    win) while plain pays sum_clifford chi."""
    from .engine import LowRankState
    rng = np.random.default_rng(0)
    names1 = ["H", "S", "S_DAG", "X", "Y", "Z"]
    worst = 0.0
    saved_total, frame_total = 0, 0
    for _ in range(trials):
        n = int(rng.integers(2, 6))
        ops = []
        for _ in range(int(rng.integers(3, 7))):
            for q in range(n):
                r = rng.random()
                if r < 0.30:
                    ops.append(("t", q, bool(rng.integers(2))))
                else:
                    ops.append(("1q", str(rng.choice(names1)), q))
            for q in range(0, n - 1, 2):
                ops.append((str(rng.choice(["cx", "cz"])), q, q + 1))

        def run(s):
            for op in ops:
                if op[0] == "1q":
                    s.clifford_1q(op[1], op[2])
                elif op[0] == "t":
                    s.t(op[1], dagger=op[2])
                else:
                    (s.cx if op[0] == "cx" else s.cz)(op[1], op[2])

        sp = LowRankState(n, backend="chform")
        sf = LowRankState(n, backend="chform", frame=True)
        run(sp)
        run(sf)
        worst = max(worst, float(np.max(np.abs(sp.statevector() - sf.statevector()))))
        saved_total += sp.ctr.clifford_term_ops      # per-term Clifford work plain pays
        frame_total += sf.ctr.clifford_term_ops       # frame pays 0 of these
        assert sf.ctr.clifford_term_ops == 0, "frame should pay no per-term Clifford ops"
    assert worst < 1e-9, f"frame vs plain mismatch {worst}"
    print(f"[OK] frame engine == plain engine on {trials} unitary Clifford+T "
          f"circuits (max abs err {worst:.2e})")
    print(f"     per-term Clifford ops: plain = {saved_total}, frame = {frame_total} "
          f"(all absorbed into the O(n)/gate frame)")


def test_frame_measurement_vs_plain(trials=50):
    """Frame-mode MEASUREMENT == plain engine on measured Clifford+T circuits.
    The plain run picks outcomes; the frame run is forced to the same outcomes,
    so the final statevectors must match exactly (both exact)."""
    from .engine import LowRankState
    rng = np.random.default_rng(0)
    names1 = ["H", "S", "S_DAG", "X", "Y", "Z"]
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(2, 5))
        ops = []
        for _ in range(int(rng.integers(2, 4))):       # a few measured rounds
            for q in range(n):
                ops.append(("1q", "H", q))
            for q in range(n):
                if rng.random() < 0.6:
                    ops.append(("t", q, bool(rng.integers(2))))
            for q in range(0, n - 1, 2):
                ops.append(("cz", q, q + 1))
            for q in range(n):
                ops.append(("1q", "H", q))
            for q in range(n):
                ops.append(("m", q))

        def run(s, outcomes):
            rec, oi = [], 0
            dummy = np.random.default_rng(123)
            for op in ops:
                if op[0] == "1q":
                    s.clifford_1q(op[1], op[2])
                elif op[0] == "t":
                    s.t(op[1], dagger=op[2])
                elif op[0] == "cz":
                    s.cz(op[1], op[2])
                elif op[0] == "m":
                    f = outcomes[oi] if outcomes is not None else None
                    rec.append(s.measure_z(op[1], dummy, force=f)); oi += 1
            return rec

        sp = LowRankState(n, backend="chform")
        outs = run(sp, None)                            # plain picks outcomes
        sf = LowRankState(n, backend="chform", frame=True)
        run(sf, outs)                                   # frame forced to match
        assert all(isinstance(t, CHForm) for t in sf.terms), "frame densified a term"
        worst = max(worst, float(np.max(np.abs(sp.statevector() - sf.statevector()))))
    assert worst < 1e-9, f"frame measurement mismatch {worst}"
    print(f"[OK] frame-mode measurement == plain on {trials} measured Clifford+T "
          f"circuits, forced outcomes (max abs err {worst:.2e})")


if __name__ == "__main__":
    test_frame_conjugation()
    test_symplectic_invariant()
    test_frame_engine_vs_plain()
    test_frame_measurement_vs_plain()
