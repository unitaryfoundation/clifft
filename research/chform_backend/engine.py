"""Low-rank stabilizer-decomposition engine (residual-backend prototype).

This is the prototype of the "stab-rank back-end" that would replace clifft's
dense active block. It maintains the state as a low-rank superposition of
stabilizer states

    |psi> = sum_j  term_j        (each term_j is a (sub-normalized) stabilizer state)

and exposes the operations the active block sees: Clifford gates, T / R_z
(which BRANCH the decomposition), and measurement (which can REDUCE the rank).
The number of terms chi is the stabilizer rank -- the quantity that decides cost.

Term backends
-------------
Each term is a `Term` (see `term.py`): either a `DenseTerm` (a 2^n statevector --
trivially correct, the original prototype) or a `CHForm` (the real CH-form
stabilizer tableau, O(n^2) bits/term -- arXiv:1808.00128). chi (the rank) and the
branching / measurement logic are identical for both; only the per-term storage
and gate cost differ. Select with `LowRankState(n, backend="chform")`. The dense
backend is the validation oracle the CH-form is checked against (see
`test_chform.py`); both are checked against clifft in `validate.py`.

No re-compression / sparsification is performed beyond exact dedup: T-gates
branch the rank by up to x2 each, so chi reaches the *exact* (un-sparsified)
stabilizer rank. The stabilizer-extent improvement (2^0.228t instead of 2^t) is
a separate sparsification pass, noted as a TODO. The composition claim under
test here -- that measurement collapses the rank back down per episode -- is
independent of sparsification.
"""

from __future__ import annotations

import numpy as np

from .term import DenseTerm, Term
from .chform import CHForm, _I_POW
from .clifford_frame import CliffordFrame

_T_PHASE = np.exp(1j * np.pi / 4)
_ZERO_TOL = 1e-12

# Diagonal single-qubit Cliffords by the phase they put on |1>, at angles
# 0, pi/2, pi, 3pi/2.  A diagonal R_z(theta) = diag(1, e^{i theta}) decomposes
# into the two ADJACENT diagonal Cliffords bracketing theta -- the minimal-L1
# (minimal-extent) Clifford pair. For T (theta = pi/4) this is the {I, S}
# decomposition T = alpha I + beta S with |alpha| + |beta| = 2^{0.114}, i.e.
# extent 2^{0.228} per T -- the source of the 0.228 stabilizer-rank exponent.
_DIAG_NAMES = ["I", "S", "Z", "S_DAG", "I"]
_DIAG_PHASE = [1 + 0j, 1j, -1 + 0j, -1j, 1 + 0j]
_DIAG_ANGLE = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]


def _clifford_pair_decompose(phase1: complex) -> list[tuple[complex, str]]:
    """diag(1, phase1) = c1 * D1 + c2 * D2 for the adjacent diagonal-Clifford
    pair (D in {I,S,Z,S_DAG}) bracketing arg(phase1). Returns the nonzero
    (coeff, gate-name) terms -- one entry if phase1 is itself a Clifford phase."""
    theta = float(np.angle(phase1)) % (2 * np.pi)
    for i in range(4):
        if _DIAG_ANGLE[i] - 1e-9 <= theta <= _DIAG_ANGLE[i + 1] + 1e-9:
            p1, p2 = _DIAG_PHASE[i], _DIAG_PHASE[i + 1]
            n1, n2 = _DIAG_NAMES[i], _DIAG_NAMES[i + 1]
            break
    c1 = (phase1 - p2) / (p1 - p2)
    c2 = 1.0 - c1
    out = []
    if abs(c1) > _ZERO_TOL:
        out.append((complex(c1), n1))
    if abs(c2) > _ZERO_TOL:
        out.append((complex(c2), n2))
    return out


def _apply_pauli_dense(vec: np.ndarray, p: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(i^p X(a) Z(b)) . vec  on a dense statevector (validation-scale only)."""
    n = len(a)
    a_int = int(sum(int(a[j]) << j for j in range(n)))
    idx = np.arange(len(vec))
    sign = np.ones(len(vec))
    for j in np.nonzero(b)[0]:
        sign = sign * np.where((idx >> int(j)) & 1, -1.0, 1.0)
    out = np.zeros_like(vec)
    out[idx ^ a_int] = _I_POW[p % 4] * sign * vec
    return out


_GATE_DAG = {"H": "H", "S": "S_DAG", "S_DAG": "S", "X": "X", "Y": "Y", "Z": "Z",
             "CX": "CX", "CZ": "CZ", "SWAP": "SWAP"}


def _apply_gate_to_term(term, name, qs):
    if name in ("H", "S", "S_DAG", "X", "Y", "Z"):
        term.clifford_1q(name, qs[0])
    elif name == "CX":
        term.cx(qs[0], qs[1])
    elif name == "CZ":
        term.cz(qs[0], qs[1])
    elif name == "SWAP":
        term.swap(qs[0], qs[1])


def _reduce_pauli_to_Z(n: int, p: int, a: np.ndarray, b: np.ndarray):
    """Clifford W (gate list) with W (i^p X(a)Z(b)) W^dagger = sign * Z_j.
    Returns (gates, j, sign in {+1,-1}). Gates from {H,S,S_DAG,CX}; conjugation
    rules tracked so the residual sign is exact."""
    a, b, p = a.copy().astype(np.int64), b.copy().astype(np.int64), int(p) % 4
    gates = []
    for i in range(n):
        if a[i] and b[i]:           # Y_i -> apply S_i (S Y S^dagger = -X): clears Z
            gates.append(("S", (i,)))
            p = (p + 1) % 4; b[i] ^= 1
        if a[i]:                    # X_i -> apply H_i: X->Z
            gates.append(("H", (i,)))
            p = (p + 2 * (a[i] & b[i])) % 4; a[i], b[i] = b[i], a[i]
    support = [i for i in range(n) if b[i]]
    assert not a.any(), "reduction left an X component"
    j = support[0]
    for i in support[1:]:           # CX(control i, target j): removes Z_i, keeps Z_j
        gates.append(("CX", (i, j)))
        a[j] ^= a[i]; b[i] ^= b[j]
    assert p % 2 == 0, f"non-real residual phase {p}"
    return gates, j, (1 if p % 4 == 0 else -1)


class Counters:
    """Operation-count instrumentation (the free-Clifford cost comparison)."""

    def __init__(self) -> None:
        self.clifford_gates = 0  # Clifford gates applied to the state
        self.clifford_term_ops = 0  # gates x chi-at-that-time (pure-stab-rank cost)
        self.frame_clifford_gates = 0  # Cliffords absorbed by the frame (O(n) each, free per-term)
        self.t_gates = 0
        self.t_branches = 0  # actual rank increase from T-branching
        self.measurements = 0
        self.peak_chi = 1

    def as_dict(self) -> dict:
        return {
            "clifford_gates": self.clifford_gates,
            "clifford_term_ops": self.clifford_term_ops,
            "frame_clifford_gates": self.frame_clifford_gates,
            "t_gates": self.t_gates,
            "t_branches": self.t_branches,
            "measurements": self.measurements,
            "peak_chi": self.peak_chi,
        }


def _new_term(n: int, backend: str) -> Term:
    """The |0...0> term in the requested backend."""
    if backend == "dense":
        z = np.zeros(2 ** n, dtype=complex)
        z[0] = 1.0
        return DenseTerm(n, z)
    if backend == "chform":
        return CHForm(n)  # CHForm() initialises to |0...0>
    raise ValueError(f"unknown backend {backend!r}")


class LowRankState:
    """A low-rank stabilizer decomposition of an n-qubit state."""

    def __init__(self, n: int, backend: str = "dense",
                 sparsify_budget: int | None = None,
                 rng: np.random.Generator | None = None,
                 frame: bool = False):
        self.n = n
        self.backend = backend
        self.terms: list[Term] = [_new_term(n, backend)]
        self.ctr = Counters()
        self.chi_trace: list[tuple[str, int]] = [("init", 1)]
        # if set, rz/T auto-sparsifies down to this budget whenever chi exceeds
        # 2x it -- the streaming mode that keeps the rank at the extent scale
        # without ever building the exact 2^t decomposition.
        self.sparsify_budget = sparsify_budget
        self._rng = rng if rng is not None else np.random.default_rng()
        # composition mode: |state> = F . (sum terms), F a global Clifford frame.
        # Cliffords are absorbed into F (free per-term); magic/measurement are
        # conjugated through F (Pauli P' = F^-1 Z_q F touches the terms).
        self.frame = CliffordFrame(n) if frame else None

    # --- rank bookkeeping ---
    @property
    def chi(self) -> int:
        return len(self.terms)

    def _record(self, label: str) -> None:
        self.ctr.peak_chi = max(self.ctr.peak_chi, self.chi)
        self.chi_trace.append((label, self.chi))

    # --- Clifford gates ---
    # Frame mode: absorbed into F in O(n), terms untouched (the free-Clifford win).
    # Plain mode: applied to every term, O(chi) per gate (pure stab-rank cost).
    def _frame_clifford(self, name: str, *qs: int) -> bool:
        if self.frame is None:
            return False
        self.frame.apply_left(name, *qs)
        self.ctr.frame_clifford_gates += 1
        self._record(f"F:{name}{qs}")
        return True

    def clifford_1q(self, name: str, q: int) -> None:
        if self._frame_clifford(name, q):
            return
        for t in self.terms:
            t.clifford_1q(name, q)
        self.ctr.clifford_gates += 1
        self.ctr.clifford_term_ops += self.chi
        self._record(f"{name}({q})")

    def cx(self, c: int, t: int) -> None:
        if self._frame_clifford("CX", c, t):
            return
        for term in self.terms:
            term.cx(c, t)
        self.ctr.clifford_gates += 1
        self.ctr.clifford_term_ops += self.chi
        self._record(f"CX({c},{t})")

    def cz(self, c: int, t: int) -> None:
        if self._frame_clifford("CZ", c, t):
            return
        for term in self.terms:
            term.cz(c, t)
        self.ctr.clifford_gates += 1
        self.ctr.clifford_term_ops += self.chi
        self._record(f"CZ({c},{t})")

    def swap(self, a: int, b: int) -> None:
        if self._frame_clifford("SWAP", a, b):
            return
        for term in self.terms:
            term.swap(a, b)
        self.ctr.clifford_gates += 1
        self.ctr.clifford_term_ops += self.chi
        self._record(f"SWAP({a},{b})")

    # --- non-Clifford diagonal: BRANCHES the decomposition ---
    def rz_diag(self, q: int, phase1: complex) -> None:
        """Apply diag(1, phase1) on qubit q by the minimal-extent Clifford-pair
        decomposition diag(1, phase1) = c1 D1 + c2 D2 (D in {I,S,Z,S_DAG}). Each
        term branches into c1 D1|term> + c2 D2|term>, BOTH stabilizer states.

        This is the low-extent magic decomposition: each branch keeps unit norm
        and the L1 weight grows by |c1|+|c2| per gate (~1.082 for T), so the
        whole-state extent is ~2^{0.228 t} -- exactly what `sparsify` exploits.
        (The earlier |0>/|1> projection split cost sqrt(2) per T, extent 2^t,
        which sparsification cannot improve.)

        Frame mode: R_z(theta) = e^{i th/2}(cos(th/2) I - i sin(th/2) P') with
        P' = F^-1 Z_q F (conj_Z), so each term branches into
            c0 . term  +  c1 . (P' . term),   c0 = e^{i th/2} cos(th/2),
                                              c1 = -i e^{i th/2} sin(th/2),
        and P'.term is just Pauli gates on the term (extent 1.707/T here -- the
        single-Pauli branch; the optimal {I,S} 1.17/T needs the Clifford sqrt(P')
        instead, a later refinement)."""
        if self.frame is not None:
            self._rz_diag_frame(q, phase1)
            return
        old_chi = self.chi
        decomp = _clifford_pair_decompose(phase1)
        new_terms: list[Term] = []
        for term in self.terms:
            for coeff, gate in decomp:
                nt = term.copy()
                if gate != "I":
                    nt.clifford_1q(gate, q)
                nt.scale(coeff)
                new_terms.append(nt)
        self.terms = new_terms
        self.ctr.t_gates += 1
        self.ctr.t_branches += max(0, self.chi - old_chi)
        self._record(f"Tdiag({q})")
        if self.sparsify_budget is not None and self.chi > 2 * self.sparsify_budget:
            self.sparsify(self.sparsify_budget, self._rng)

    def _rz_diag_frame(self, q: int, phase1: complex) -> None:
        """Frame-conjugated magic: delegate to the explicit-Pauli rotation with
        P' = F^-1 Z_q F (see rz_about_pauli for the mechanism)."""
        pp, ax, az = self.frame.conj_Z(q)            # P' = i^pp X(ax) Z(az)
        self.rz_about_pauli(pp, ax, az, phase1, label=f"Tframe({q})")

    def rz_about_pauli(self, pp: int, ax, az, phase1: complex,
                       label: str | None = None) -> None:
        """Diagonal rotation diag(1, phase1) about an EXPLICIT Pauli
        P = i^pp X(ax) Z(az), at the OPTIMAL {I,S} extent (1.17/T, not 1.707).

        diag(1,phase1) = c1 D1 + c2 D2 with D in {I,S,Z,S_DAG}; each diagonal
        Clifford D is applied about P as D(P) = W^dagger D_j W, where W is a
        Clifford mapping P -> +Z_j (functional calculus commutes with
        conjugation -- exact, no phase). The reduction's sign is normalised to
        +1 by appending X_j to W, so the coefficients (hence the 2^{0.228 t}
        extent) match the plain path exactly.

        This is also the entry point for consuming clifft's OPTIMIZED HIR,
        whose T_GATE ops are rotations about arbitrary conjugated Paulis
        (see hir_bridge.py)."""
        old_chi = self.chi
        decomp = _clifford_pair_decompose(phase1)
        gates, j, sign = _reduce_pauli_to_Z(self.n, pp, np.asarray(ax), np.asarray(az))
        if sign < 0:
            gates = gates + [("X", (j,))]            # X_j Z_j X_j = -Z_j -> normalise to +Z_j
        wdag = [(_GATE_DAG[nm], qs) for nm, qs in reversed(gates)]
        new_terms: list[Term] = []
        for term in self.terms:
            for coeff, gate in decomp:
                nt = term.copy()
                if gate != "I":                      # F^-1 D_q F = W^dagger D_j W
                    for nm, qs in gates:
                        _apply_gate_to_term(nt, nm, qs)
                    nt.clifford_1q(gate, j)
                    for nm, qs in wdag:
                        _apply_gate_to_term(nt, nm, qs)
                nt.scale(coeff)
                new_terms.append(nt)
        self.terms = new_terms
        self.ctr.t_gates += 1
        self.ctr.t_branches += max(0, self.chi - old_chi)
        self._record(label or "Tpauli")
        if self.sparsify_budget is not None and self.chi > 2 * self.sparsify_budget:
            self.sparsify(self.sparsify_budget, self._rng)

    # --- sparsification: random sub-sampling to ~extent terms (APPROXIMATE) ---
    def sparsify(self, k: int, rng: np.random.Generator) -> None:
        """Replace the decomposition by an unbiased random estimator with at most
        `k` terms (Bravyi-Gosset-Howard sparsification).

        With |psi> = sum_a c_a |phi_a> (|phi_a> unit, c_a = term's omega), sample
        k indices i.i.d. from p_a = |c_a| / ||c||_1 and form
            |omega> = (||c||_1 / k) sum_j (c_{a_j}/|c_{a_j}|) |phi_{a_j}>.
        Then E|omega> = |psi> and E|| |psi> - |omega> ||^2 = (||c||_1^2 - ||psi||^2)/k.
        Because the low-extent T branching keeps ||c||_1 ~ 2^{0.114 t}, taking
        k ~ 2^{0.228 t}/delta^2 bounds the error by delta -- the rank stays at the
        stabilizer-extent scale instead of the exact 2^t. This trades clifft's
        exactness for sub-2^t rank: the whole point of the stab-rank regime.

        No-op if chi <= k. Duplicate samples merge in `recompress_dedup`."""
        weights = np.array([np.sqrt(t.norm2()) for t in self.terms])
        l1 = float(weights.sum())
        if l1 < _ZERO_TOL or self.chi <= k:
            return
        p = weights / l1
        picks = rng.choice(self.chi, size=k, p=p)
        # merge repeated samples by exact key (O(n^2) for CH-form, no 2^n vector)
        groups: dict[bytes, Term] = {}
        merged: list[Term] = []
        for a in picks:
            nt = self.terms[a].copy()
            nt.scale(l1 / (k * weights[a]))  # (||c||_1/k) * (c_a/|c_a|) folded in
            key = nt.merge_key()
            rep = groups.get(key)
            if rep is None:
                groups[key] = nt
                merged.append(nt)
            else:
                rep.merge_add(nt)
        self.terms = merged
        self._record(f"sparsify({k})")

    # --- single-shot tensor-product magic injection (no streaming, no t-factor) ---
    def inject_magic_layer(self, gates: list[tuple[int, complex]], k: int,
                           rng: np.random.Generator) -> None:
        """Sparsify a whole tensor-product magic layer in ONE shot.

        Precondition: chi == 1 (a single base stabilizer state, e.g. after the
        pre-magic Cliffords). `gates` is a list of (qubit, phase1) diagonal
        rotations whose Clifford-pair decompositions tensor-multiply (distinct
        qubits / commuting branches) -- the magic register is then a product
        sum  prod_q (c_{q,0} D_{q,0} + c_{q,1} D_{q,1}) |base>  over 2^t branch
        strings, with L1 weight ||c||_1 = prod_q (|c_{q,0}|+|c_{q,1}|).

        Instead of enumerating 2^t and sparsifying (streaming, which accumulates
        variance over t steps), we IMPORTANCE-SAMPLE k branch strings directly:
        each qubit's branch is drawn independently with P(branch 1) =
        |c_{q,1}|/(|c_{q,0}|+|c_{q,1}|), so the string b is drawn with
        probability |c_b|/||c||_1. The estimator
            |omega> = (||c||_1/k) sum_j (c_{b_j}/|c_{b_j}|) D_{b_j}|base>
        is unbiased with E|| |psi>-|omega> ||^2 = (||c||_1^2 - ||psi||^2)/k, so
        k ~ ||c||_1^2/delta^2 = 2^{0.228 t}/delta^2 bounds the error -- with NO
        factor of t and NO 2^t ever built. This is the BGH-optimal route for
        product magic (IQP); streaming `sparsify` is the fallback when magic is
        interleaved with entangling Cliffords."""
        if self.chi != 1:
            raise ValueError("inject_magic_layer requires chi == 1 (single base)")
        base = self.terms[0]
        branchers: list[tuple[int, list[tuple[complex, str]]]] = []
        for q, phase1 in gates:
            decomp = _clifford_pair_decompose(phase1)
            if len(decomp) == 1:  # phase1 is itself Clifford -> deterministic
                c, gate = decomp[0]
                if gate != "I":
                    base.clifford_1q(gate, q)
                base.scale(c)
            else:
                branchers.append((q, decomp))
        self.ctr.t_gates += len(branchers)
        if not branchers:
            return
        l1 = 1.0
        probs1 = []
        for _, d in branchers:
            w0, w1 = abs(d[0][0]), abs(d[1][0])
            l1 *= (w0 + w1)
            probs1.append(w1 / (w0 + w1))
        groups: dict[bytes, Term] = {}
        new_terms: list[Term] = []
        for _ in range(k):
            nt = base.copy()
            phase = 1.0 + 0j
            for (q, decomp), p1 in zip(branchers, probs1):
                c, gate = decomp[1] if rng.random() < p1 else decomp[0]
                if gate != "I":
                    nt.clifford_1q(gate, q)
                phase *= c / abs(c)
            nt.scale(l1 / k * phase)
            key = nt.merge_key()
            rep = groups.get(key)
            if rep is None:
                groups[key] = nt
                new_terms.append(nt)
            else:
                rep.merge_add(nt)
        self.terms = new_terms
        self._record(f"magic_layer({len(branchers)}->{self.chi})")

    def t(self, q: int, dagger: bool = False) -> None:
        self.rz_diag(q, np.conj(_T_PHASE) if dagger else _T_PHASE)

    def rz(self, q: int, theta: float) -> None:
        """R_z(theta) = diag(1, e^{i theta}) in clifft half-turn convention;
        theta is the literal phase on |1> (caller converts turns->radians)."""
        self.rz_diag(q, np.exp(1j * theta))

    # --- recompression: merge parallel terms (exact, partial) ---
    def recompress_dedup(self) -> None:
        """Merge terms that are scalar multiples of one another.

        c_j|s> + c_k|s> = (c_j+c_k)|s> is exact. This collapses chi precisely
        when measurement has disentangled / aligned the surviving stabilizer
        states (e.g. a fully measured-out round leaves a single basis state).
        It is a CHEAP, EXACT lower-effort stand-in for true sparsification /
        stabilizer-extent recompression (the random-walk methods), which reach
        the 2^0.228t extent bound; dedup does not, but it suffices to exhibit
        the per-episode rank collapse. Without ANY recompression chi never
        collapses on measurement -- the surviving terms stay linearly dependent
        but distinct -- so recompression is REQUIRED, not optional.

        Keyed on each term's canonical (global-phase-fixed) statevector; terms
        sharing a key are scalar multiples of one stabilizer state. Each group is
        collapsed onto its first member IN PLACE (rescaled by the summed/own
        amplitude ratio), so a term keeps its native backend -- a CH-form group
        stays a single CH-form, it is not silently densified. The rescale is
        exact because same-key members are genuine scalar multiples.
        """
        groups: dict[bytes, list[tuple[Term, np.ndarray]]] = {}
        order: list[bytes] = []
        for term in self.terms:
            vec = term.statevector()
            if np.vdot(vec, vec).real < _ZERO_TOL:
                continue
            key = term.canonical_key()
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((term, vec))
        new_terms: list[Term] = []
        for key in order:
            members = groups[key]
            rep, v0 = members[0]
            if len(members) > 1:
                merged = sum((v for _, v in members[1:]), v0.copy())
                k = int(np.argmax(np.abs(v0)))
                rep.scale(complex(merged[k] / v0[k]))  # rep := (sum c_i / c_0) rep
            new_terms.append(rep)
        if not new_terms:
            zero = _new_term(self.n, self.backend)
            zero.scale(0.0)
            new_terms = [zero]
        self.terms = new_terms
        self._record("recompress")

    # --- readout: in frame mode the frame F is applied to the terms (one-time) ---
    def _readout_terms(self) -> list[Term]:
        """The residual terms with the frame folded in: F . term. In plain mode
        this is just the terms. O(|gates| * chi), used only at readout."""
        if self.frame is None:
            return self.terms
        out = []
        for term in self.terms:
            ft = term.copy()
            self.frame.apply_to(ft)
            out.append(ft)
        return out

    # --- non-materialising single amplitude: <x|psi> = sum_a omega_a <x|phi_a> ---
    def amplitude(self, x: int) -> complex:
        """<x|psi> for basis label x, O(chi * n^2), never builds a 2^n vector.
        With `estimate_norm2` (norm_est.py) this gives output probabilities
        P(x) = |<x|psi>|^2 / ||psi||^2 past clifft's dense wall."""
        return sum((t.amplitude(x) for t in self._readout_terms()), 0j)

    # --- measurement: projects every term; can REDUCE the rank ---
    def statevector(self) -> np.ndarray:
        out = np.zeros(2 ** self.n, dtype=complex)
        for term in self._readout_terms():
            out += term.statevector()
        return out

    def measure_z(self, q: int, rng: np.random.Generator, force: int | None = None) -> int:
        """Sample a Z-measurement on qubit q, project all terms, drop terms with
        zero support on the outcome, renormalize. Returns the outcome bit.
        `force` pins the outcome (for deterministic frame-vs-plain validation)."""
        if self.frame is not None:
            return self._measure_z_frame(q, rng, force)
        full = self.statevector()
        idx = np.arange(2 ** self.n)
        is1 = ((idx >> q) & 1).astype(bool)
        p1 = float(np.vdot(full[is1], full[is1]).real)
        ptot = float(np.vdot(full, full).real)
        if force is not None:
            outcome = int(force)
        else:
            outcome = 1 if (ptot > 0 and rng.random() < p1 / ptot) else 0
        new_terms: list[Term] = []
        for term in self.terms:
            proj = term.project(q, outcome)
            if proj is not None and proj.norm2() > _ZERO_TOL:
                new_terms.append(proj)
        self.terms = new_terms if new_terms else [_new_term(self.n, self.backend)]
        if not new_terms:
            self.terms[0].scale(0.0)
        # renormalize the surviving superposition to unit norm
        nrm = np.sqrt(np.vdot(self.statevector(), self.statevector()).real)
        if nrm > _ZERO_TOL:
            for term in self.terms:
                term.scale(1.0 / nrm)
        self.ctr.measurements += 1
        self._record(f"M({q})->{outcome}")
        self.recompress_dedup()
        return outcome

    def collapse_to_rank1(self) -> None:
        """Collapse chi parallel terms onto one representative -- VALID ONLY
        when the state is provably rank 1 (e.g. every register qubit has just
        been measured, so the state is a single stabilizer state; the caller
        owns that argument). Non-materialising: finds a support point x* of the
        first surviving term by scanning amplitudes (O(2^n) worst case but
        each query is O(n^2) and generic states hit support immediately), then
        sums the coefficient ratios t_i(x*)/t_0(x*) -- exact because all terms
        are scalar multiples of one stabilizer state.

        This replaces the tableau-key dedup at round boundaries, where terms
        proportional to the same state generically have DIFFERENT tableaux
        (merge_key cannot see it) and chi would otherwise compound per round."""
        live = [t for t in self.terms if t.norm2() > _ZERO_TOL]
        if len(live) <= 1:
            self.terms = live or self.terms[:1]
            return
        t0 = live[0]
        a0 = 0.0 + 0.0j
        xstar = -1
        for x in range(2 ** self.n):
            a0 = t0.amplitude(x)
            if abs(a0) > 1e-12:
                xstar = x
                break
        if xstar < 0:  # t0 numerically null after all; fall back untouched
            return
        ratio = sum(t.amplitude(xstar) for t in live) / a0
        t0.scale(complex(ratio))
        self.terms = [t0]
        self._record("collapse_rank1")

    def measure_pauli_forced_fast(self, pp: int, ax, az, outcome: int) -> float:
        """Forced projective measurement of an EXPLICIT Pauli P = i^pp X(ax) Z(az):
        project every term onto the `outcome` eigenspace via the W-reduction
        (P -> Z_j), dedup by tableau key, and apply a common rescale so the
        largest term stays O(1). Returns the applied rescale factor f: callers
        computing record probabilities recover the true projected norm as
        stored_norm / prod(f). Non-materialising; the HIR-consumption entry
        point (clifft's optimized MEASURE ops are arbitrary Pauli
        measurements -- see hir_bridge.py)."""
        outcome = int(outcome)
        gates, j, sign = _reduce_pauli_to_Z(self.n, pp, np.asarray(ax), np.asarray(az))
        bit_j = outcome ^ (1 if sign < 0 else 0)
        wdag = [(_GATE_DAG[nm], qs) for nm, qs in reversed(gates)]
        new_terms: list[Term] = []
        for term in self.terms:
            ft = term.copy()
            for nm, qs in gates:
                _apply_gate_to_term(ft, nm, qs)
            pj = ft.project(j, bit_j)
            if pj is not None and pj.norm2() > _ZERO_TOL:
                for nm, qs in wdag:
                    _apply_gate_to_term(pj, nm, qs)
                new_terms.append(pj)
        if not new_terms:
            zero = _new_term(self.n, self.backend)
            zero.scale(0.0)
            new_terms = [zero]
        merged: dict[bytes, Term] = {}
        order: list[bytes] = []
        for t in new_terms:
            key = t.merge_key()
            if key in merged:
                merged[key].merge_add(t)
            else:
                merged[key] = t
                order.append(key)
        self.terms = [merged[k] for k in order
                      if merged[k].norm2() > _ZERO_TOL] or new_terms[:1]
        m = max((t.norm2() for t in self.terms), default=0.0)
        f = 1.0
        if m > 0.0:
            f = 1.0 / np.sqrt(m)
            for t in self.terms:
                t.scale(f)
        self.ctr.measurements += 1
        self._record(f"Mpauli->{outcome}")
        return f

    def measure_z_forced_fast(self, q: int, outcome: int,
                              flip_if_dead: bool = False) -> int:
        """Forced-outcome Z measurement with NO dense materialization anywhere:
        project every term, drop zero-support terms, merge exact duplicates by
        tableau key (merge_key, O(n^2)); no outcome probability, no
        renormalization (all downstream operations are linear, so the state
        stays a consistent unnormalized trajectory -- frame and plain engines
        skip the same normalization, keeping wall-clock comparisons fair).

        This is the scale path for the adaptive-workload benchmark
        (bench_adaptive.py); sampling-mode measurement still uses measure_z,
        whose probability computation materializes 2^n (validation scale) --
        replacing that with norm estimation is the known remaining step."""
        outcome = int(outcome)

        def project_all(bit: int) -> list[Term]:
            if self.frame is not None:
                pp, ax, az = self.frame.conj_Z(q)
                gates, j, sign = _reduce_pauli_to_Z(self.n, pp, ax, az)
                bit_j = bit ^ (1 if sign < 0 else 0)
                wdag = [(_GATE_DAG[nm], qs) for nm, qs in reversed(gates)]
                out: list[Term] = []
                for term in self.terms:
                    ft = term.copy()
                    for nm, qs in gates:
                        _apply_gate_to_term(ft, nm, qs)
                    pj = ft.project(j, bit_j)
                    if pj is not None and pj.norm2() > _ZERO_TOL:
                        for nm, qs in wdag:
                            _apply_gate_to_term(pj, nm, qs)
                        out.append(pj)
                return out
            out = []
            for term in self.terms:
                pj = term.project(q, bit)
                if pj is not None and pj.norm2() > _ZERO_TOL:
                    out.append(pj)
            return out

        new_terms = project_all(outcome)
        if not new_terms and flip_if_dead:
            # The forced branch has zero support; the complementary outcome
            # is then certain (P(0)+P(1) = ||psi||^2 > 0). Used by benchmark
            # drivers that force pseudo-random trajectories.
            outcome ^= 1
            new_terms = project_all(outcome)
        if not new_terms:
            zero = _new_term(self.n, self.backend)
            zero.scale(0.0)
            new_terms = [zero]
        # exact duplicate merge on the tableau key (non-materialising)
        merged: dict[bytes, Term] = {}
        order: list[bytes] = []
        for t in new_terms:
            key = t.merge_key()
            if key in merged:
                merged[key].merge_add(t)
            else:
                merged[key] = t
                order.append(key)
        self.terms = [merged[k] for k in order
                      if merged[k].norm2() > _ZERO_TOL] or new_terms[:1]
        # Common rescale so the largest term stays O(1): the unnormalized
        # trajectory otherwise decays ~2x per measurement and healthy terms
        # eventually sink below the absolute _ZERO_TOL and get misclassified
        # as dead. A shared factor is correctness-neutral (state is
        # unnormalized by contract) and O(chi) scalar work.
        m = max((t.norm2() for t in self.terms), default=0.0)
        if m > 0.0:
            f = 1.0 / np.sqrt(m)
            for t in self.terms:
                t.scale(f)
        self.ctr.measurements += 1
        self._record(f"Mfast({q})->{outcome}")
        return outcome

    def _measure_z_frame(self, q: int, rng: np.random.Generator,
                         force: int | None = None) -> int:
        """Measure Z_q on F.(sum term) = a Pauli measurement of P' = F^-1 Z_q F on
        the RESIDUAL; F is unchanged. Project via W (P' -> Z_j), CHForm.project, W^dagger."""
        pp, ax, az = self.frame.conj_Z(q)
        # outcome probability on the residual (materialised here, validation-scale;
        # norm estimation replaces this at scale -- same status as the plain path).
        res = np.zeros(2 ** self.n, dtype=complex)
        for term in self.terms:
            res += term.statevector()
        ntot = float(np.vdot(res, res).real)
        pres = _apply_pauli_dense(res, pp, ax, az)
        proj0 = (res + pres) / 2.0                       # P' = +1  <=>  Z_q outcome 0
        p0 = float(np.vdot(proj0, proj0).real)
        if force is not None:
            outcome = int(force)
        else:
            outcome = 0 if (ntot > 0 and rng.random() < p0 / ntot) else 1
        gates, j, sign = _reduce_pauli_to_Z(self.n, pp, ax, az)
        bit_j = outcome ^ (1 if sign < 0 else 0)
        wdag = [(_GATE_DAG[nm], qs) for nm, qs in reversed(gates)]
        new_terms: list[Term] = []
        for term in self.terms:
            ft = term.copy()
            for nm, qs in gates:
                _apply_gate_to_term(ft, nm, qs)
            pj = ft.project(j, bit_j)
            if pj is not None and pj.norm2() > _ZERO_TOL:
                for nm, qs in wdag:
                    _apply_gate_to_term(pj, nm, qs)
                new_terms.append(pj)
        self.terms = new_terms if new_terms else [_new_term(self.n, self.backend)]
        if not new_terms:
            self.terms[0].scale(0.0)
        res2 = np.zeros(2 ** self.n, dtype=complex)
        for term in self.terms:
            res2 += term.statevector()
        nrm = np.sqrt(np.vdot(res2, res2).real)
        if nrm > _ZERO_TOL:
            for term in self.terms:
                term.scale(1.0 / nrm)
        self.ctr.measurements += 1
        self._record(f"Mframe({q})->{outcome}")
        self.recompress_dedup()
        return outcome
