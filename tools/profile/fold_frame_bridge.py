"""Compile cat measurement of a shared Clifford fault frame into two monomials."""

from fold_blocks import Fold, project_mask
from fold_check import Check, Event, PhaseMonomial, branch_operator
from fold_contraction import compose
from fold_fault_frame import FramePlan


class Bridge:
    def __init__(self, fold: Fold):
        self.fold = fold
        self.frame = FramePlan(fold.core)
        inverse = {q: j for j, q in enumerate(fold.mapping + fold.cats)}
        check = Check(
            fold.plan.width,
            len(fold.cats),
            tuple(
                Event("CZ" if op.name == "CCZ" else op.name, tuple(inverse[q] for q in op.targets))
                for op in fold.core.operations
            ),
        )
        check.validate()
        self.ideal = [branch_operator(check, (b,) * len(fold.cats))[0] for b in (0, 1)]
        self.routes = []
        for edge, (a, b) in enumerate(self.frame.edges):
            if a in fold.data_index and b in fold.data_index:
                q, r = sorted((fold.data_index[a], fold.data_index[b]))
                if (q, r) not in fold.plan.edges:
                    raise ValueError("frame edge is outside the contraction geometry")
                self.routes.append((edge, 0, q, r, fold.plan.edges.index((q, r))))
            elif a in fold.cat_index and b in fold.cat_index:
                self.routes.append((edge, 2, 0, 0, 0))
            else:
                q = fold.data_index[b] if a in fold.cat_index else fold.data_index[a]
                self.routes.append((edge, 1, q, 0, 0))

    def branches(self, preparation_faults, core_faults, decode_faults):
        fold = self.fold
        x, z, flags = fold.preparation.evaluate(preparation_faults)
        if flags not in (0, fold.equal_flag_mask):
            return []
        initial = self.frame.empty()
        for q in fold.cats:
            initial.flips |= x & (1 << q)
            initial.linear[q] = 4 * ((z >> q) & 1)
        correction = self.frame.evaluate(core_faults, initial)
        _, _, records = fold.decode.evaluate(decode_faults)
        return self.contract(correction, records)

    def contract(self, correction, records):
        fold = self.fold
        cat_flip = project_mask(correction.flips, fold.cats)
        result = []
        for bit in (0, 1):
            cat = (bit * ((1 << len(fold.cats)) - 1)) ^ cat_flip
            weight = 0.5 * int(fold.decode_table[records, cat])
            if not weight:
                continue
            op = PhaseMonomial(
                fold.plan.width,
                project_mask(correction.flips, fold.mapping),
                correction.phase + bit * sum(correction.linear[q] for q in fold.cats),
                [correction.linear[q] for q in fold.mapping],
            )
            for edge, kind, q, r, _ in self.routes:
                if not (correction.edges >> edge) & 1:
                    continue
                if kind == 0:
                    op.edges.add((q, r))
                elif kind == 1:
                    op.linear[q] += 4 * bit
                else:
                    op.global_phase += 4 * bit
            result.append((weight, compose(op, self.ideal[bit])))
        return result
