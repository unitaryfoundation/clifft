"""Hypothesis test: across fault histories, does the prototype's reduced circuit differ only in
rotation signs (T<->T_DAG, S<->S_DAG), Paulis (X/Y/Z, record values), or also in structure (CX/CZ wiring)?"""
import sys, random, re, collections
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools", "profile"))
from conditional_phase_frontend import ConditionalPhase
from automatic_specialization import FaultModel

def classify(a, b):
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return "LENGTH", len(la), len(lb)
    kinds = collections.Counter()
    for x, y in zip(la, lb):
        if x == y: continue
        gx, gy = x.split()[0], y.split()[0]
        tx, ty = x.split()[1:], y.split()[1:]
        if {gx, gy} <= {"T", "T_DAG"} and tx == ty: kinds["T-sign"] += 1
        elif {gx, gy} <= {"S", "S_DAG"} and tx == ty: kinds["S-sign"] += 1
        elif gx == gy and gx in ("MPAD",): kinds["record-value"] += 1
        elif gx == gy and gx in ("X", "Y", "Z"): kinds["pauli-targets"] += 1
        elif {gx, gy} <= {"X", "Y", "Z", "I"}: kinds["pauli"] += 1
        else: kinds[f"STRUCT:{gx}->{gy}"] += 1
    return "SAME-LENGTH", dict(kinds)

for name in sys.argv[1:]:
    src = open(name).read()
    model = FaultModel(src)
    front = ConditionalPhase(src)
    if front.first is None:
        print(name, "REJECTED:", front.rejection); continue
    rng = random.Random(7)
    hist = [model.draw(rng) for _ in range(6)]
    # force at least some faults: draw a few single-site histories too
    sites = len(model.sites)
    hist += [((s, 1),) for s in rng.sample(range(sites), min(4, sites))]
    outs = [front.rewrite(h, seed=11, choose_prefix=lambda shared, k: 0) for h in hist]
    base = outs[0]
    print(f"== {name}: sites={sites} stop={base.stop_reason} steps={len(base.steps)} residual_t={[s['residual_t'] for s in base.steps]}")
    gates = collections.Counter(l.split()[0] for l in base.source.splitlines() if l.strip())
    print("   reduced alphabet:", dict(gates.most_common(12)))
    for h, o in zip(hist[1:], outs[1:]):
        print(f"   faults={len(h):2d} stop={o.stop_reason:16s} diff vs base:", classify(base.source, o.source))
