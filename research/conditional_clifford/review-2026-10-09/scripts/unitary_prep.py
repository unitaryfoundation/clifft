"""Does production clifft (ed2a125a, TOHPE + PR552) reduce the controlled fold-S check when |+_L> is prepared by a
deterministic unitary encoder (signed constraints provable from |0>) instead of random-sign MPP projection?"""
import stim, clifft
from planar_fold import planar, pstr, controlled_fold_S, verify_fold
for d in (3, 5):
    code = planar(d); n = code["n"]
    signs, y, _ = verify_fold(code)
    gens = [pstr(n, "X", s) for s in code["xs"]] + [pstr(n, "Z", s) for s in code["zs"]]
    enc = stim.Tableau.from_stabilizers(gens + [pstr(n, "X", code["XL"])]).to_circuit()
    for checks in (1, 2):
        L = [str(enc)]
        for k in range(checks):
            c = n + k
            L.append(f"RX {c}"); L += controlled_fold_S(code, signs, c, 0.0); L.append(f"MX {c}")
        L.append("MX " + " ".join(map(str, range(n))))
        src = "\n".join(L) + "\n"
        open(f"planar_d{d}_uprep_check{checks}_ideal.stim", "w").write(src)
        prog = clifft.compile(src)
        print(f"d={d} unitary-prep check{checks} ideal: ordinary peak_active_width={prog.peak_active_width}")
