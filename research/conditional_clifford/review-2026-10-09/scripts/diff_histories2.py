"""Multiset comparison of reduced circuits across fault histories: is the T-core (T targets + CX wiring) invariant,
with differences confined to rotation signs, diagonal Cliffords, Paulis and record values?"""
import sys, random, collections
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools", "profile"))
from conditional_phase_frontend import ConditionalPhase
from automatic_specialization import FaultModel

def buckets(src):
    b = collections.defaultdict(collections.Counter)
    for l in src.splitlines():
        if not l.strip(): continue
        g, *t = l.split()
        key = {"T": "T", "T_DAG": "T", "S": "S", "S_DAG": "S"}.get(g, g)
        b[key][(g, tuple(t))] += 1
    return b

def compare(a, b):
    out = {}
    for k in sorted(set(a) | set(b)):
        if a[k] == b[k]: continue
        if k == "T":   # same targets, only T<->T_DAG?
            ta = collections.Counter(t for (g, t) in a[k].elements()); tb = collections.Counter(t for (g, t) in b[k].elements())
            out["T"] = "sign-only" if ta == tb else f"TARGETS-DIFFER (+{sum((tb-ta).values())} -{sum((ta-tb).values())})"
        elif k == "S":
            ta = collections.Counter(t for (g, t) in a[k].elements()); tb = collections.Counter(t for (g, t) in b[k].elements())
            out["S"] = "sign-only" if ta == tb else f"targets-differ(+{sum((tb-ta).values())} -{sum((ta-tb).values())})"
        else:
            out[k] = f"+{sum((b[k]-a[k]).values())} -{sum((a[k]-b[k]).values())}"
    return out

for name in sys.argv[1:]:
    src = open(name).read()
    model = FaultModel(src)
    front = ConditionalPhase(src)
    if front.first is None:
        print(name, "REJECTED:", front.rejection); continue
    rng = random.Random(7)
    sites = len(model.sites)
    hist = [()] + [model.draw(rng) for _ in range(4)] + [((s, k),) for s in rng.sample(range(sites), min(6, sites)) for k in (1,)]
    hist += [tuple((s, 1 + (i % 3)) for i, s in enumerate(rng.sample(range(sites), min(5, sites))))]  # multi-fault
    outs = [front.rewrite(h, seed=11, choose_prefix=lambda shared, k: 0) for h in hist]
    base = buckets(outs[0].source)
    print(f"== {name}: sites={sites} stop={outs[0].stop_reason} residual_t={[s['residual_t'] for s in outs[0].steps]} "
          f"T={sum(base['T'].values())} CX={sum(base['CX'].values())} CZ={sum(base['CZ'].values())} S={sum(base['S'].values())}")
    for h, o in zip(hist[1:], outs[1:]):
        print(f"   faults={len(h)} stop={o.stop_reason:15s}", compare(base, buckets(o.source)))
