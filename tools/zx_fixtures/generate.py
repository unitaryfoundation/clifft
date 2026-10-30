"""Regenerate the single-copy ZX fixtures for tests/test_zx_simplify.cc.

The fixtures pin Clifft's port of zxparam's rewrite rules (zxparam 0.1.0,
https://pypi.org/project/zxparam/). Each
case holds a closed single-copy parametric ZX diagram, the hash of the
canonical text after every rule pass that changed it, the final scalar terms,
and, for small random circuits, the trajectory amplitudes |<o|U_e|0>| from an
independent statevector simulation.

Diagrams are built from Stim-format circuits (with Tsim's T tags) as amplitudes
<o|U_e|0>: every measurement outcome o and noise bit e is a Boolean phase
parameter, a Z measurement is the projector |o><o| (a Z spider with an X(pi o)
leaf), a reset is an unrecorded effect followed by a fresh |0>, and every qubit
is projected at the end. All vertices are classical-domain, which zxparam
treats as one copy.

Run from the repository root with an environment that has Tsim
(https://github.com/QuEraComputing/tsim) and zxparam 0.1.0 installed, for
example:

    /path/to/tsim/.venv/bin/python tools/zx_fixtures/generate.py \\
        --bench /path/to/clifft-bench/workloads/circuits
"""

from __future__ import annotations

import argparse
import importlib.metadata
from fractions import Fraction
from pathlib import Path

import numpy as np
import tsim  # type: ignore[import-not-found]
import zxparam  # type: ignore[import-not-found]
from tsim.core.parse import parse_parametric_tag  # type: ignore[import-not-found]
from tsim.noise.channels import (  # type: ignore[import-not-found]
    error_probs,
    pauli_channel_1_probs,
    pauli_channel_2_probs,
)
from zxparam import (
    Domain,
    EdgeType,
    Graph,
    Phase,
    VertexType,
    _core,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "zx"
OUT = 1_000_000  # parameter IDs of outcomes start here; noise bits start at 0
Z, X = VertexType.Z, VertexType.X
S, H = EdgeType.SIMPLE, EdgeType.HADAMARD

MAX_ORACLE_PARAMS = 8
BENCH = ("distillation.stim", "msc_d3_inject_cultivate_p1e-3.stim", "coherent_d3_r3.stim")


# ---------------------------------------------------------------------------
# Single-copy builder.


class Builder:
    def __init__(self) -> None:
        self.g = Graph()
        self.last: dict[int, object] = {}
        self.etype: dict[int, EdgeType] = {}
        self.num_e = 0
        self.outcomes: list[str] = []
        self.records: list[int] = []

    def lane(self, q: int, t: VertexType = X) -> None:
        self.last[q] = self.g.add_vertex(t, Phase.zero())
        self.etype[q] = S

    def append(self, q: int, t: VertexType, phase: Phase | None = None) -> object:
        if q not in self.last:
            self.lane(q)
        w = self.g.add_vertex(t, phase if phase is not None else Phase.zero())
        self.g.add_edge(self.last[q], w, self.etype[q])
        self.last[q], self.etype[q] = w, S
        return w

    def h(self, q: int) -> None:
        if q not in self.last:
            self.lane(q)
        self.etype[q] = H if self.etype[q] == S else S

    def outcome(self, kind: str) -> int:
        k = len(self.outcomes)
        self.outcomes.append(kind)
        if kind == "rec":
            self.records.append(k)
        return k

    def param(self, k: int, flip: bool = False) -> Phase:
        return Phase(Fraction(int(flip)), frozenset({OUT + k}))

    def err(self, bit: int) -> Phase:
        return Phase(Fraction(0), frozenset({bit}))

    def channel(self, probs: np.ndarray) -> int:
        first = self.num_e
        self.num_e += int(round(np.log2(len(probs))))
        return first

    def measure(self, q: int, basis: str, flip: bool, p: float) -> Phase:
        k = self.outcome("rec")
        a = self.append(q, Z if basis == "Z" else X)
        ph = self.param(k, flip)
        if p:
            # Readout noise: the record is the physical outcome XOR an error bit.
            ph = ph + self.err(self.channel(error_probs(p)))
        leaf = self.g.add_vertex(X if basis == "Z" else Z, ph)
        self.g.add_edge(a, leaf)
        return ph

    def reset(self, q: int, basis: str) -> None:
        if q in self.last:
            k = self.outcome("silent")
            leaf = self.g.add_vertex(X if basis == "Z" else Z, self.param(k))
            self.g.add_edge(self.last[q], leaf, self.etype[q])
        self.lane(q, X if basis == "Z" else Z)

    def mpp(self, paulis: list[tuple[str, int]], flip: bool) -> None:
        k = self.outcome("rec")
        hub = self.g.add_vertex(X, self.param(k, flip))
        for p, q in paulis:
            # Y is S_DAG then H to Z: S^dag Y S = X and H X H = Z.
            if p == "Y":
                self.append(q, Z, Phase(Fraction(3, 2)))
            if p in "XY":
                self.h(q)
            self.g.add_edge(self.append(q, Z), hub)
            if p in "XY":
                self.h(q)
            if p == "Y":
                self.append(q, Z, Phase(Fraction(1, 2)))

    def finish(self) -> None:
        for q in sorted(self.last):
            leaf = self.g.add_vertex(X, self.param(self.outcome("final")))
            self.g.add_edge(self.last[q], leaf, self.etype[q])


def build(text: str) -> Builder:
    b = Builder()
    single = {
        "X": (X, Fraction(1)),
        "Z": (Z, Fraction(1)),
        "SQRT_X": (X, Fraction(1, 2)),
        "SQRT_X_DAG": (X, Fraction(3, 2)),
        "S": (Z, Fraction(1, 2)),
        "S_DAG": (Z, Fraction(3, 2)),
    }
    for ins in tsim.Circuit(text)._stim_circ.flattened():
        name, tg, args = ins.name, ins.targets_copy(), ins.gate_args_copy()
        qs = [t.value for t in tg]
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
            continue
        if name == "I" and ins.tag:
            gate, params = parse_parametric_tag(ins)
            for q in qs:
                b.append(q, Z if gate == "R_Z" else X, Phase(params["theta"]))
        elif name in ("S", "S_DAG") and "T" in ins.tag:
            for q in qs:
                b.append(q, Z, Phase(Fraction(1, 4) if name == "S" else Fraction(7, 4)))
        elif name in ("M", "MZ", "MX", "MR", "MRX"):
            for t in tg:
                basis = "X" if name in ("MX", "MRX") else "Z"
                ph = b.measure(t.value, basis, t.is_inverted_result_target, args[0] if args else 0)
                if name in ("MR", "MRX"):
                    b.append(t.value, X if basis == "Z" else Z, ph)
        elif name in ("R", "RZ", "RX"):
            for q in qs:
                b.reset(q, "X" if name == "RX" else "Z")
        elif name == "MPP":
            prods: list[tuple[list[tuple[str, int]], bool]] = []
            cur: list[tuple[str, int]] = []
            flip, prev_comb = False, True
            for t in tg:
                if t.is_combiner:
                    prev_comb = True
                    continue
                if not prev_comb and cur:
                    prods.append((cur, flip))
                    cur, flip = [], False
                cur.append(("X" if t.is_x_target else "Y" if t.is_y_target else "Z", t.value))
                flip ^= t.is_inverted_result_target
                prev_comb = False
            if cur:
                prods.append((cur, flip))
            for paulis, fl in prods:
                b.mpp(paulis, fl)
        elif name in ("CX", "CZ"):
            for c, t in zip(tg[::2], tg[1::2]):
                if c.is_measurement_record_target:
                    r = b.param(b.records[len(b.records) + c.value])
                    b.append(t.value, X if name == "CX" else Z, r)
                    continue
                a, w = b.append(c.value, Z), b.append(t.value, X if name == "CX" else Z)
                b.g.add_edge(a, w, S if name == "CX" else H)
        elif name == "H":
            for q in qs:
                b.h(q)
        elif name in single:
            t, ph = single[name]
            for q in qs:
                b.append(q, t, Phase(ph))
        elif name in ("SQRT_Y", "SQRT_Y_DAG"):
            # SQRT_Y = H then X, SQRT_Y_DAG = H then Z, up to global phase.
            for q in qs:
                b.h(q)
                b.append(q, X if name == "SQRT_Y" else Z, Phase.pi())
        elif name in ("X_ERROR", "Z_ERROR"):
            for q in qs:
                e = b.channel(error_probs(args[0]))
                b.append(q, X if name == "X_ERROR" else Z, b.err(e))
        elif name == "DEPOLARIZE1":
            for q in qs:
                e = b.channel(pauli_channel_1_probs(args[0] / 3, args[0] / 3, args[0] / 3))
                b.append(q, Z, b.err(e))
                b.append(q, X, b.err(e + 1))
        elif name == "DEPOLARIZE2":
            for qi, qj in zip(qs[::2], qs[1::2]):
                e = b.channel(pauli_channel_2_probs(*([args[0] / 15] * 15)))
                b.append(qi, Z, b.err(e))
                b.append(qi, X, b.err(e + 1))
                b.append(qj, Z, b.err(e + 2))
                b.append(qj, X, b.err(e + 3))
        else:
            raise NotImplementedError(name)
    b.finish()
    return b


# ---------------------------------------------------------------------------
# zxparam's full_reduce, one leaf step at a time.

# Rules that act only across domains or on boundaries; they must never fire here.
INERT = ("decohere", "parity_tie", "ground_pauli", "ground_gadget", "pivot_boundary")


def phase_text(value: Fraction) -> str:
    quarters = value * 4
    if quarters.denominator == 1:
        return f"g{int(quarters)}"
    return f"{float(value):.9f}"


def canonical_text(g: Graph) -> str:
    lines = []
    for v in sorted(g.vertices(), key=int):
        ph = g.phase(v)
        t = "Z" if g.vertex_type(v) == Z else "X"
        params = ",".join(str(p) for p in sorted(ph.parameters))
        nbrs = "".join(f" {int(n)}{'s' if g.edge_type(v, n) == S else 'h'}" for n in g.neighbors(v))
        lines.append(f"{int(v)} {t} {phase_text(ph.value)} {params} |{nbrs}\n")
    return "".join(lines)


def fnv1a64(text: str) -> int:
    h = 0xCBF29CE484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def traced_full_reduce(g: Graph) -> list[tuple[str, int]]:
    steps: list[tuple[str, int]] = []

    def leaf(name: str) -> bool:
        if name == "to_gh":
            changed = _core.to_gh_step(g)
        elif name == "remove_isolated":
            changed = _core.remove_isolated_step(g)
        else:
            changed = _core.rule_pass(g, name)
        if changed:
            if name in INERT:
                raise AssertionError(f"{name} fired on a single-copy diagram")
            steps.append((name, fnv1a64(canonical_text(g))))
        return bool(changed)

    def sat(name: str) -> bool:
        changed = False
        while leaf(name):
            changed = True
        return changed

    def interior() -> bool:
        changed = False
        while True:
            r = False
            for name in ("decohere", "parity_tie", "id_remove", "spider_fusion", "pivot", "lcomp"):
                r |= sat(name)
            r |= sat("ground_pauli")
            r |= sat("ground_gadget")
            if not r:
                return changed
            changed = True

    def run() -> bool:
        changed = leaf("to_gh")
        changed |= sat("spider_fusion")
        changed |= sat("copy")
        changed |= interior()
        changed |= sat("pivot_gadget")
        changed |= sat("pivot_boundary")
        while True:
            r = interior()
            r |= sat("gadget_normalize")
            r |= leaf("gadget_fusion")
            r |= leaf("copy")
            if not r:
                break
            changed = True
        removed = leaf("remove_isolated")
        return changed or removed

    while run():
        pass
    return steps


# ---------------------------------------------------------------------------
# Random circuits and their statevector amplitudes.

GATES1 = ["H", "S", "S_DAG", "T", "T_DAG", "X", "Z", "SQRT_X"]


def random_circuit(
    rng: np.random.Generator, nq: int, depth: int, dense_t: bool, clifford_heavy: bool = False
) -> list:
    ops: list = []
    nrec = 0
    for _ in range(depth):
        r = rng.random()
        q = int(rng.integers(nq))
        if clifford_heavy and r < 0.8:
            # Dense entangling Clifford structure: pivots with overlapping
            # neighbourhoods and long local-complementation chains.
            if r < 0.35:
                ops.append((["H", "S", "S_DAG", "SQRT_X"][rng.integers(4)], [q]))
            elif r < 0.7:
                a, b = rng.choice(nq, 2, replace=False)
                ops.append((["CX", "CZ"][rng.integers(2)], [int(a), int(b)]))
            else:
                ops.append((["T", "T_DAG"][rng.integers(2)], [q]))
            continue
        if dense_t and r < 0.45:
            name = ["T", "T_DAG", "CX", "X_ERROR", "Z_ERROR"][rng.integers(5)]
            if name == "CX":
                a, b = rng.choice(nq, 2, replace=False)
                ops.append(("CX", [int(a), int(b)]))
            else:
                ops.append((name, [q]))
        elif r < 0.35:
            ops.append((GATES1[rng.integers(len(GATES1))], [q]))
        elif r < 0.5:
            a, b = rng.choice(nq, 2, replace=False)
            ops.append((["CX", "CZ"][rng.integers(2)], [int(a), int(b)]))
        elif r < 0.6:
            ops.append((["X_ERROR", "Z_ERROR", "DEPOLARIZE1"][rng.integers(3)], [q]))
        elif r < 0.72:
            ops.append((["M", "MX"][rng.integers(2)], [q]))
            nrec += 1
        elif r < 0.8:
            ops.append((["R", "RX", "MR"][rng.integers(3)], [q]))
            nrec += ops[-1][0] == "MR"
        elif r < 0.88:
            a, b = rng.choice(nq, 2, replace=False)
            ops.append(("MPP", [("XZ"[rng.integers(2)], int(a)), ("XZ"[rng.integers(2)], int(b))]))
            nrec += 1
        elif nrec:
            ops.append(
                ("FEEDBACK", [-int(rng.integers(1, nrec + 1)), q, ["CX", "CZ"][rng.integers(2)]])
            )
    return ops


def to_stim(ops: list) -> str:
    lines = []
    for name, t in ops:
        if name in ("T", "T_DAG"):
            lines.append(f"{'S' if name == 'T' else 'S_DAG'}[T] {t[0]}")
        elif name in ("X_ERROR", "Z_ERROR", "DEPOLARIZE1"):
            lines.append(f"{name}(0.1) {t[0]}")
        elif name == "MPP":
            lines.append("MPP " + "*".join(f"{p}{q}" for p, q in t))
        elif name == "FEEDBACK":
            lines.append(f"{t[2]} rec[{t[0]}] {t[1]}")
        else:
            lines.append(f"{name} " + " ".join(map(str, t)))
    return "\n".join(lines) + "\n"


def reference_amplitudes(ops: list, nq: int, b: Builder, params: list[int]) -> np.ndarray:
    """|<o|U_e|0>| for every assignment, bit k of the index being params[k]."""
    pidx = {p: k for k, p in enumerate(params)}
    hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    xm, zm = np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    one = {
        "H": hm,
        "S": np.diag([1, 1j]),
        "S_DAG": np.diag([1, -1j]),
        "T": np.diag([1, np.exp(1j * np.pi / 4)]),
        "T_DAG": np.diag([1, np.exp(-1j * np.pi / 4)]),
        "X": xm,
        "Z": zm,
        "SQRT_X": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    }
    two = {"CX": np.eye(4)[[0, 1, 3, 2]], "CZ": np.diag([1, 1, 1, -1])}

    def apply1(psi: np.ndarray, m: np.ndarray, q: int) -> np.ndarray:
        return np.moveaxis(np.tensordot(m, np.moveaxis(psi, q, 0), axes=1), 0, q)

    def apply2(psi: np.ndarray, m: np.ndarray, a: int, c: int) -> np.ndarray:
        psi = np.moveaxis(psi, [a, c], [0, 1])
        shape = psi.shape
        psi = (m @ psi.reshape(4, -1)).reshape(shape)
        return np.moveaxis(psi, [0, 1], [a, c])

    def project(psi: np.ndarray, q: int, basis: str, val: int) -> np.ndarray:
        v = np.eye(2)[val]
        if basis == "X":
            v = hm @ v
        return apply1(psi, np.outer(v, v.conj()), q)

    out = np.zeros(1 << len(params))
    for k in range(len(out)):

        def bit(p: int) -> int:
            return (k >> pidx[p]) & 1 if p in pidx else 0

        psi = np.zeros((2,) * nq, dtype=complex)
        psi[(0,) * nq] = 1
        used: set[int] = set()
        e = o = 0
        rec: list[int] = []
        for name, t in ops:
            if name == "FEEDBACK":
                if rec[t[0]]:
                    psi = apply1(psi, xm if t[2] == "CX" else zm, t[1])
                used.add(t[1])
                continue
            if name in one:
                psi = apply1(psi, one[name], t[0])
            elif name in two:
                psi = apply2(psi, two[name], t[0], t[1])
            elif name in ("X_ERROR", "Z_ERROR"):
                if bit(e):
                    psi = apply1(psi, xm if name == "X_ERROR" else zm, t[0])
                e += 1
            elif name == "DEPOLARIZE1":
                if bit(e):
                    psi = apply1(psi, zm, t[0])
                if bit(e + 1):
                    psi = apply1(psi, xm, t[0])
                e += 2
            elif name in ("M", "MX", "MR"):
                val = bit(OUT + o)
                o += 1
                rec.append(val)
                psi = project(psi, t[0], "X" if name == "MX" else "Z", val)
                if name == "MR" and val:
                    psi = apply1(psi, xm, t[0])
            elif name in ("R", "RX"):
                basis = "Z" if name == "R" else "X"
                if t[0] in used:
                    val = bit(OUT + o)
                    o += 1
                    psi = project(psi, t[0], basis, val)
                    if val:
                        psi = apply1(psi, xm if basis == "Z" else zm, t[0])
                elif basis == "X":
                    psi = apply1(psi, hm, t[0])
            elif name == "MPP":
                val = bit(OUT + o)
                o += 1
                rec.append(val)
                full = np.eye(1)
                for qq in range(nq):
                    m = np.eye(2)
                    for p, q in t:
                        if q == qq:
                            m = xm if p == "X" else zm
                    full = np.kron(full, m)
                proj = (np.eye(1 << nq) + (-1) ** val * full) / 2
                psi = (proj @ psi.reshape(-1)).reshape((2,) * nq)
            qs = [q for _, q in t] if name == "MPP" else t
            used.update(qs)
        for q in sorted(b.last):
            psi = project(psi, q, "Z", bit(OUT + o))
            o += 1
        out[k] = np.sqrt((np.abs(psi) ** 2).sum())
    return out


# ---------------------------------------------------------------------------
# Fixture writing.


def graph_lines(g: Graph) -> list[str]:
    lines = []
    ids = sorted(g.vertices(), key=int)
    if [int(v) for v in ids] != list(range(len(ids))):
        raise AssertionError("builder IDs must be 0, 1, 2, ...")
    for v in ids:
        if g.vertex_domain(v) != Domain.CLASSICAL:
            raise AssertionError("single-copy diagrams use classical-domain vertices only")
        ph = g.phase(v)
        t = "Z" if g.vertex_type(v) == Z else "X"
        params = ",".join(str(p) for p in sorted(ph.parameters)) or "-"
        lines.append(f"v {int(v)} {t} {ph.value.numerator}/{ph.value.denominator} {params}")
    for v in ids:
        nbrs = " ".join(f"{int(n)}{'s' if g.edge_type(v, n) == S else 'h'}" for n in g.neighbors(v))
        lines.append(f"a {int(v)} {nbrs}".rstrip())
    return lines


def scalar_lines(g: Graph) -> list[str]:
    lines = []
    for n in g.scalar.nodes:
        params = ",".join(str(p) for p in sorted(n.a)) or "-"
        lines.append(f"node {n.alpha.numerator}/{n.alpha.denominator} {params}")
    for q in g.scalar.phase_pairs:
        a = ",".join(str(p) for p in sorted(q.a)) or "-"
        b = ",".join(str(p) for p in sorted(q.b)) or "-"
        lines.append(
            f"pair {q.alpha.numerator}/{q.alpha.denominator} "
            f"{q.beta.numerator}/{q.beta.denominator} {a} {b}"
        )
    return lines


def case_lines(name: str, text: str, oracle: tuple | None = None) -> list[str]:
    b = build(text)
    lines = [f"case {name}", *graph_lines(b.g)]
    if oracle is not None:
        ops, nq = oracle
        params = sorted(
            {p for v in b.g.vertices() for p in b.g.phase(v).parameters}
            | {OUT + k for k in range(len(b.outcomes))}
        )
        ref = reference_amplitudes(ops, nq, b, params)
        lines.append("params " + " ".join(f"{p}{'o' if p >= OUT else 'e'}" for p in params))
        lines.append("ref " + " ".join("0" if x < 1e-12 else f"{x:.6e}" for x in ref))
    native = b.g.copy()
    zxparam.full_reduce(native)
    steps = traced_full_reduce(b.g)
    if canonical_text(b.g) != canonical_text(native):
        raise AssertionError(f"{name}: the traced strategy differs from zxparam.full_reduce")
    lines += [f"step {rule} {h:016x}" for rule, h in steps]
    lines.append(f"final {fnv1a64(canonical_text(b.g)):016x}")
    lines += scalar_lines(b.g)
    lines.append("end")
    return lines


def write(path: Path, header: list[str], cases: list[list[str]]) -> None:
    text = "\n".join(header + [line for case in cases for line in case]) + "\n"
    path.write_text(text)
    print(f"{path.relative_to(ROOT)}: {len(cases)} cases, {len(text) // 1024} KiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, help="clifft-bench workloads/circuits directory")
    ap.add_argument("--random", type=int, default=25, help="random circuits per family")
    a = ap.parse_args()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    header = [
        "# Generated by tools/zx_fixtures/generate.py; do not edit.",
        f"# zxparam {importlib.metadata.version('zxparam')}.",
    ]
    families = (
        ("random", 3, False, False, 14, 1),
        ("random_t", 3, True, False, 20, 2),
        ("random_clifford", 4, False, True, 26, 3),
    )
    for family, nq, dense_t, clifford_heavy, depth, seed in families:
        rng = np.random.default_rng(seed)
        cases = []
        kept = 0
        while kept < a.random:
            ops = random_circuit(rng, nq, depth, dense_t, clifford_heavy)
            text = to_stim(ops)
            b = build(text)
            # The oracle enumerates every assignment of the parameters.
            if b.num_e + len(b.outcomes) > MAX_ORACLE_PARAMS:
                continue
            cases.append(case_lines(f"{family}_{kept}", text, (ops, nq)))
            kept += 1
        write(FIXTURES / f"single_copy_{family}.txt", header, cases)
    if a.bench:
        for name in BENCH:
            text = (a.bench / name).read_text()
            write(FIXTURES / f"single_copy_{Path(name).stem}.txt", header, [case_lines(name, text)])


if __name__ == "__main__":
    main()
