"""Unrotated planar surface code [[d^2+(d-1)^2,1,d]] with Moussa's fold-transversal logical S.

Qubits at (i,j), 0<=i,j<=2d-2, i+j even. X-checks at (odd i, even j), Z-checks at (even i, odd j).
Z_L = column j=0, X_L = row i=0. Transpose swaps X<->Z checks and X_L<->Z_L.
Fold-transversal S: S^{+-1} on diagonal qubits (i,i), CZ on mirror pairs (i,j)<->(j,i).
Verified with stim: stabilizers preserved, X_L -> +-Y_L on |+_L>.
Emits Stim circuits for controlled versions (CS = T c; T t; CX; T_DAG t; CX ; CCZ = 7 T decomposition).
"""
import itertools, json, sys
import stim

def planar(d):
    L = 2 * d - 1
    pts = [(i, j) for i in range(L) for j in range(L) if (i + j) % 2 == 0]
    idx = {p: k for k, p in enumerate(pts)}
    n = len(pts)
    def nbrs(i, j):
        return [idx[(a, b)] for (a, b) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if (a, b) in idx]
    xs = [nbrs(i, j) for i in range(L) for j in range(L) if i % 2 == 1 and j % 2 == 0]
    zs = [nbrs(i, j) for i in range(L) for j in range(L) if i % 2 == 0 and j % 2 == 1]
    XL = [idx[(0, j)] for j in range(0, L, 2)]
    ZL = [idx[(i, 0)] for i in range(0, L, 2)]
    diag = [idx[(i, i)] for i in range(L)]
    pairs = sorted({tuple(sorted((idx[(i, j)], idx[(j, i)]))) for (i, j) in pts if i < j})
    assert n == d * d + (d - 1) ** 2 and len(xs) + len(zs) == n - 1
    return dict(d=d, n=n, pts=pts, xs=xs, zs=zs, XL=XL, ZL=ZL, diag=diag, pairs=pairs)

def pstr(n, kind, sup):
    s = ["_"] * n
    for q in sup: s[q] = kind
    return stim.PauliString("".join(s))

def prep_plus_L(code):
    """TableauSimulator in |+_L> with all stabilizers +1 (deterministic tableau-derived encoder)."""
    n = code["n"]
    gens = [pstr(n, "X", s) for s in code["xs"]] + [pstr(n, "Z", s) for s in code["zs"]]
    xl = pstr(n, "X", code["XL"])
    tab = stim.Tableau.from_stabilizers(gens + [xl], allow_redundant=False, allow_underconstrained=False)
    sim = stim.TableauSimulator(); sim.set_num_qubits(n)
    sim.do(tab.to_circuit())
    assert all(sim.peek_observable_expectation(g) == 1 for g in gens) and sim.peek_observable_expectation(xl) == 1
    return sim, gens, xl

def gf2_solve(rows, rhs, n):
    """rows: list of int bitmasks (equations), rhs: list of bits. Return x bitmask or None."""
    eqs = [(r, b) for r, b in zip(rows, rhs)]
    piv = {}
    for r, b in eqs:
        for p, (pr, pb) in piv.items():
            if r >> p & 1: r ^= pr; b ^= pb
        if r == 0:
            if b: return None
            continue
        p = r.bit_length() - 1
        for q in list(piv):
            qr, qb = piv[q]
            if qr >> p & 1: piv[q] = (qr ^ r, qb ^ b)
        piv[p] = (r, b)
    x = 0
    for p, (r, b) in piv.items():
        if b: x |= 1 << p
    return x

def verify_fold(code):
    """All-S ansatz on the diagonal + CZ on mirror pairs, then a Z-string fixing X-check signs. Returns (signs, y, lines)."""
    n, d = code["n"], code["d"]
    yl = pstr(n, "X", code["XL"]) * pstr(n, "Z", code["ZL"]); yl = stim.PauliString(str(yl).lstrip("+-i"))
    signs = tuple(1 for _ in code["diag"])
    sim, gens, xl = prep_plus_L(code)
    lines = [f"S {q}" for q in code["diag"]] + [f"CZ {a} {b}" for a, b in code["pairs"]]
    sim.do(stim.Circuit("\n".join(lines)))
    st = [sim.peek_observable_expectation(g) for g in gens]
    assert all(s != 0 for s in st), "stabilizer group not preserved"
    bad = [k for k, s in enumerate(st) if s == -1]
    # fix with Z-string v: for each X-check k, parity(v & support_k) = [k in bad]; Z-checks unaffected
    rows = [sum(1 << q for q in s) for s in code["xs"]]
    rhs = [1 if k in bad else 0 for k in range(len(code["xs"]))]
    v = gf2_solve(rows, rhs, n)
    assert v is not None
    zfix = [q for q in range(n) if v >> q & 1]
    if zfix: sim.do(stim.Circuit("Z " + " ".join(map(str, zfix))))
    st = [sim.peek_observable_expectation(g) for g in gens]
    assert all(s == 1 for s in st), st
    y = sim.peek_observable_expectation(yl)
    assert y != 0 and sim.peek_observable_expectation(xl) == 0
    code["zfix"] = zfix
    return signs, y, lines + ([f"Z {q}" for q in zfix])

def ccz_lines(c, a, b, scratch=None):
    """CCZ via phase polynomial: T on a,b,c ; T_DAG on pairs ; T on triple, parities built with CX (no scratch needed)."""
    L = [f"T {c}", f"T {a}", f"T {b}",
         f"CX {a} {b}", f"T_DAG {b}", f"CX {a} {b}",          # a^b
         f"CX {c} {b}", f"T_DAG {b}", f"CX {c} {b}",          # c^b
         f"CX {c} {a}", f"T_DAG {a}", f"CX {c} {a}",          # c^a
         f"CX {a} {b}", f"CX {c} {b}", f"T {b}", f"CX {c} {b}", f"CX {a} {b}"]  # a^b^c
    return L

def cs_lines(c, t, dag=False):
    return [f"{'T_DAG' if dag else 'T'} {c}", f"{'T_DAG' if dag else 'T'} {t}", f"CX {c} {t}", f"{'T' if dag else 'T_DAG'} {t}", f"CX {c} {t}"]

def controlled_fold_S(code, signs, ctrl, p):
    L = []
    for q, s in zip(code["diag"], signs):
        L += cs_lines(ctrl, q, dag=(s == -1))
        if p: L.append(f"DEPOLARIZE2({p}) {ctrl} {q}")
    for a, b in code["pairs"]:
        L += ccz_lines(ctrl, a, b)
        if p: L.append(f"DEPOLARIZE2({p}) {a} {b}"); L.append(f"DEPOLARIZE1({p}) {ctrl}")
    for q in code.get("zfix", []):
        L.append(f"CZ {ctrl} {q}")
        if p: L.append(f"DEPOLARIZE2({p}) {ctrl} {q}")
    return L

def logical_T_gadget(code, p, dag=False):
    """Exact logical T on the code space: accumulate Z_L parity on ZL[0] with CX, T, un-CX (CNOT+diagonal stand-in for injection)."""
    tgt, rest = code["ZL"][0], code["ZL"][1:]
    L = [f"CX {q} {tgt}" for q in rest] + [f"{'T_DAG' if dag else 'T'} {tgt}"] + [f"CX {q} {tgt}" for q in reversed(rest)]
    if p: L.append(f"DEPOLARIZE1({p}) " + " ".join(map(str, code["ZL"])))
    return L

def circuit(code, signs, *, checks=1, inject_T=False, p=0.001, readout="MX"):
    n = code["n"]
    L = [f"RX {' '.join(map(str, range(n)))}"]
    for s in code["zs"]: L.append("MPP " + "*".join(f"Z{q}" for q in s))
    for s in code["xs"]: L.append("MPP " + "*".join(f"X{q}" for q in s))
    if p: L.append(f"DEPOLARIZE1({p}) {' '.join(map(str, range(n)))}")
    if inject_T:
        L += logical_T_gadget(code, p)
    for k in range(checks):
        c = n + k
        L.append(f"RX {c}")
        L += controlled_fold_S(code, signs, c, p)
        L.append(f"MX {c}")
        # re-measure stabilizers between checks (Clifford bridge, like a QEC round)
        if k + 1 < checks:
            for s in code["zs"]: L.append("MPP " + "*".join(f"Z{q}" for q in s))
            for s in code["xs"]: L.append("MPP " + "*".join(f"X{q}" for q in s))
            if p: L.append(f"DEPOLARIZE1({p}) {' '.join(map(str, range(n)))}")
    L.append(f"{readout} {' '.join(map(str, range(n)))}")
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    import clifft
    out = {}
    for d in (3, 5):
        code = planar(d)
        signs, y, lines = verify_fold(code)
        print(f"d={d}: n={code['n']} #X={len(code['xs'])} #Z={len(code['zs'])} diag={code['diag']} pairs={len(code['pairs'])} zfix={code['zfix']} Y_L->{y}")
        for name, kw in {
            "check1": dict(checks=1), "check2": dict(checks=2),
            "T_check1": dict(checks=1, inject_T=True), "T_check2": dict(checks=2, inject_T=True),
        }.items():
            for noise, p in (("ideal", 0.0), ("noisy", 0.001)):
                src = circuit(code, signs, p=p, **kw)
                fn = f"planar_d{d}_{name}_{noise}.stim"
                open(fn, "w").write(src)
                w = clifft.compile(src).peak_active_width
                tcount = sum(1 for l in src.splitlines() if l.split()[0] in ("T", "T_DAG"))
                print(f"  {fn:34s} T={tcount:4d} ordinary peak_active_width={w}")
                out[fn] = dict(width=w, t=tcount)
    json.dump(out, open("ordinary_widths.json", "w"), indent=1)
