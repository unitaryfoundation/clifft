"""Validate the low-rank engine's exactness against a dense reference + clifft.

Run: python -m research.chform_backend.validate
"""

from __future__ import annotations

import numpy as np

from .engine import LowRankState, _T_PHASE
from .term import _apply_1q, _1Q


# --- a dead-simple dense reference simulator (single statevector) ---
class DenseRef:
    def __init__(self, n: int):
        self.n = n
        self.v = np.zeros(2**n, dtype=complex)
        self.v[0] = 1.0

    def clifford_1q(self, name, q):
        self.v = _apply_1q(self.v, self.n, q, _1Q[name])

    def cx(self, c, t):
        idx = np.arange(2**self.n)
        self.v = self.v[idx ^ (((idx >> c) & 1) << t)]

    def cz(self, c, t):
        idx = np.arange(2**self.n)
        self.v = self.v * np.where(((idx >> c) & 1) & ((idx >> t) & 1), -1.0, 1.0)

    def t(self, q, dagger=False):
        idx = np.arange(2**self.n)
        ph = np.conj(_T_PHASE) if dagger else _T_PHASE
        self.v = self.v * np.where((idx >> q) & 1, ph, 1.0)

    def rz(self, q, theta):
        idx = np.arange(2**self.n)
        self.v = self.v * np.where((idx >> q) & 1, np.exp(1j * theta), 1.0)


def _random_circuit(n, depth, seed):
    rng = np.random.default_rng(seed)
    gates = []
    for _ in range(depth):
        for q in range(n):
            r = rng.random()
            if r < 0.25:
                gates.append(("t", q, bool(rng.integers(2))))
            elif r < 0.45:
                gates.append(("rz", q, float(rng.uniform(0.1, 1.2))))
            else:
                gates.append(("c1", rng.choice(list(_1Q)), q))
        for q in range(0, n - 1, 2):
            gates.append(("cx", q, q + 1) if rng.random() < 0.5 else ("cz", q, q + 1))
    return gates


def _run(state, gates):
    for g in gates:
        if g[0] == "c1":
            state.clifford_1q(g[1], g[2])
        elif g[0] == "cx":
            state.cx(g[1], g[2])
        elif g[0] == "cz":
            state.cz(g[1], g[2])
        elif g[0] == "t":
            state.t(g[1], dagger=g[2])
        elif g[0] == "rz":
            state.rz(g[1], g[2])


def validate_vs_dense(trials=25, backend="dense"):
    worst = 0.0
    for seed in range(trials):
        n = int(np.random.default_rng(seed).integers(3, 9))
        depth = 3
        gates = _random_circuit(n, depth, seed)
        lr = LowRankState(n, backend=backend)
        ref = DenseRef(n)
        _run(lr, gates)
        _run(ref, gates)
        err = float(np.max(np.abs(lr.statevector() - ref.v)))
        worst = max(worst, err)
        assert err < 1e-9, f"seed {seed}: statevector mismatch {err}"
    print(f"[OK] low-rank({backend}) == dense reference on {trials} random "
          f"Clifford+T circuits (max abs err {worst:.2e})")


def validate_vs_clifft(trials=12, backend="dense"):
    try:
        import clifft
    except ImportError:
        print("[skip] clifft not importable")
        return
    worst = 0.0
    n_checked = 0
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        n = int(rng.integers(3, 7))
        gates = _random_circuit(n, 2, 1000 + seed)
        # build stim text (unitary: no measurement) for clifft
        lines = []
        for g in gates:
            if g[0] == "c1":
                lines.append(f"{g[1]} {g[2]}")
            elif g[0] == "cx":
                lines.append(f"CX {g[1]} {g[2]}")
            elif g[0] == "cz":
                lines.append(f"CZ {g[1]} {g[2]}")
            elif g[0] == "t":
                lines.append(f"{'T_DAG' if g[2] else 'T'} {g[1]}")
            elif g[0] == "rz":
                # clifft R_z angle is in half-turns: phase on |1> is e^{i*pi*theta}
                lines.append(f"R_Z({g[1] / np.pi:.10f}) {g[2] if isinstance(g[2], int) else g[1]}")
        # rz target fix: rebuild cleanly to avoid the tuple-shape confusion
        lines = []
        for g in gates:
            if g[0] == "c1":
                lines.append(f"{g[1]} {g[2]}")
            elif g[0] == "cx":
                lines.append(f"CX {g[1]} {g[2]}")
            elif g[0] == "cz":
                lines.append(f"CZ {g[1]} {g[2]}")
            elif g[0] == "t":
                lines.append(f"{'T_DAG' if g[2] else 'T'} {g[1]}")
            elif g[0] == "rz":
                lines.append(f"R_Z({g[2] / np.pi:.10f}) {g[1]}")
        prog = clifft.compile("\n".join(lines), hir_passes=None, bytecode_passes=None)
        st = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
        clifft.execute(prog, st)
        sv_clifft = np.asarray(clifft.get_statevector(prog, st)).ravel()
        lr = LowRankState(n, backend=backend)
        _run(lr, gates)
        sv_lr = lr.statevector()
        # compare up to a global phase
        if np.max(np.abs(sv_clifft)) > 1e-12:
            k = int(np.argmax(np.abs(sv_clifft)))
            ph = sv_lr[k] / sv_clifft[k]
            err = float(np.max(np.abs(sv_lr - ph * sv_clifft)))
            worst = max(worst, err)
            n_checked += 1
            assert err < 1e-7, f"seed {seed}: clifft mismatch {err}"
    print(f"[OK] low-rank({backend}) == clifft.get_statevector on {n_checked} "
          f"circuits (max abs err {worst:.2e}, up to global phase)")


if __name__ == "__main__":
    for be in ("dense", "chform"):
        validate_vs_dense(backend=be)
        validate_vs_clifft(backend=be)
