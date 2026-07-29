"""Exact output-distribution enumerator for small noncomputational circuits.

Test-only companion to ``utils_noncomp_oracle``: walks a small circuit over
the default five-level set and returns the exact joint distribution of the
visible measurement record, by branching a pure statevector over every
classical event -- initial levels, transition fires, Born outcomes -- with
explicit probability weights. Transition sites apply the physical per-site
channel (source-conditioned collapse plus the sqrt(1 - p) no-fire filter),
so the result is the dense reference the trajectory sampler must match to
shot noise; ``damping="neglect"`` reproduces that policy's one documented
omission by skipping the no-fire filter while keeping fire weights exact.

Deliberately independent of clifft: its own line parser (a small subset)
and its own statement of the semantics -- operations touching a leaked or
lost operand drop whole (measurements never drop; a classifier column
supplies their record bit), R restores a leaked qubit and never a lost
one, LOSS is a source-independent trace-out. Exponential in circuit
size by construction; the branch cap guards against feeding it a scenario
it was never meant to hold.

Level ids match ``clifft.noncomp.Level``: g=0, e=1, leak_g=2, leak_e=3,
lost=4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import utils_noncomp_oracle as oracle

G, E, LEAK_G, LEAK_E, LOST = range(5)
_COMPUTATIONAL = (G, E)

_PRUNE_EPS = 1e-12
_MAX_BRANCHES = 200_000


@dataclass
class _Op:
    kind: str  # "gate" | "reset" | "measure" | "measure_reset" | "transition" | "loss"
    targets: list[int]
    gate: str = ""
    tag: str = ""
    prob: float = 0.0


_GATE_1Q = {"H", "S", "X", "Y", "Z"}


def parse_circuit(text: str) -> tuple[list[_Op], int]:
    """Parse the supported subset; returns (ops, num_qubits)."""
    ops: list[_Op] = []
    max_q = -1
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, *args = line.split()
        qubits = [int(a) for a in args]
        max_q = max(max_q, *qubits) if qubits else max_q
        if name in _GATE_1Q:
            ops.extend(_Op("gate", [q], gate=name) for q in qubits)
        elif name == "CX":
            assert len(qubits) % 2 == 0
            ops.extend(
                _Op("gate", [qubits[i], qubits[i + 1]], gate="CX") for i in range(0, len(qubits), 2)
            )
        elif name == "R":
            ops.extend(_Op("reset", [q]) for q in qubits)
        elif name == "M":
            ops.extend(_Op("measure", [q]) for q in qubits)
        elif name == "MR":
            ops.extend(_Op("measure_reset", [q]) for q in qubits)
        elif name.startswith("LEVEL_TRANSITION["):
            m = re.fullmatch(r"LEVEL_TRANSITION\[([^\]]+)\]", name)
            assert m is not None, line
            ops.extend(_Op("transition", [q], tag=m.group(1)) for q in qubits)
        elif name.startswith("LOSS("):
            m = re.fullmatch(r"LOSS\(([^)]+)\)", name)
            assert m is not None, line
            ops.extend(_Op("loss", [q], prob=float(m.group(1))) for q in qubits)
        else:
            raise ValueError(f"enumerator does not support: {line}")
    return ops, max_q + 1


@dataclass
class _Branch:
    weight: float
    state: npt.NDArray[np.complex128]
    status: tuple[int, ...]  # level id per qubit
    record: tuple[int, ...]


class ExactDistribution:
    """The enumeration result: record joint, noncomputational-level
    marginals per qubit, and truncated mass.

    Only noncomputational levels appear in ``noncomp_level_probs``: for a
    computational qubit the classical ledger tracks the *category* (which
    is all the branching semantics need), while the level itself is
    quantum information living in the state -- so per-level computational
    marginals are not representable here by design. The computational
    mass per qubit is one minus the noncomputational sum.
    """

    def __init__(self) -> None:
        self.record_probs: dict[tuple[int, ...], float] = {}
        self.noncomp_level_probs: list[dict[int, float]] = []
        self.dropped_mass = 0.0

    def _absorb(self, branch: _Branch, num_qubits: int) -> None:
        if not self.noncomp_level_probs:
            self.noncomp_level_probs = [{} for _ in range(num_qubits)]
        self.record_probs[branch.record] = self.record_probs.get(branch.record, 0.0) + branch.weight
        for q, level in enumerate(branch.status):
            if level in _COMPUTATIONAL:
                continue
            self.noncomp_level_probs[q][level] = (
                self.noncomp_level_probs[q].get(level, 0.0) + branch.weight
            )


def _hidden_collapse(branch: _Branch, q: int, n: int) -> list[tuple[float, np.ndarray, int]]:
    """Born-collapse qubit q without recording: [(weight, state, bit), ...]."""
    out = []
    for bit in (0, 1):
        w, post = oracle.collapse(branch.state, q, bit, n)
        if w > 0.0:
            out.append((w, post, bit))
    return out


def _prep_zero(state: np.ndarray, q: int, bit_now: int, n: int) -> np.ndarray:
    return oracle.set_collapsed_qubit(state, q, bit_now, 0, n)


def enumerate_exact(
    text: str,
    *,
    initial: list[float],
    transitions: dict[str, list[list[float]]],
    classifier: list[list[float]],
    damping: str = "exact",
) -> ExactDistribution:
    """Exact record/status distribution for `text` under the given model.

    `classifier` is P[symbol][level] with two symbols; computational
    measurements are faithful Born readouts (identity columns required).
    """
    assert damping in ("exact", "neglect")
    assert len(classifier) == 2
    assert (
        classifier[0][G] == 1.0 and classifier[1][E] == 1.0
    ), "the enumerator models faithful computational readout only"
    ops, n = parse_circuit(text)

    # Initial branches: product over qubits of the level distribution.
    # Computational e is an X-prep; noncomputational levels park the factor
    # at |0> with the level recorded classically.
    branches = [_Branch(1.0, oracle.zero_state(n), (G,) * n, ())]
    for q in range(n):
        seeded: list[_Branch] = []
        for br in branches:
            for level, p in enumerate(initial):
                if p <= 0.0:
                    continue
                state = br.state
                if level == E:
                    state = oracle.apply_1q(state, "X", q, n)
                status = list(br.status)
                status[q] = level
                seeded.append(_Branch(br.weight * p, state, tuple(status), ()))
        branches = seeded

    result = ExactDistribution()

    def emit(br: _Branch) -> None:
        result._absorb(br, n)

    for op in ops:
        nxt: list[_Branch] = []

        for br in branches:
            noncomp = [q for q in op.targets if br.status[q] not in _COMPUTATIONAL]

            if op.kind == "gate":
                if noncomp:
                    nxt.append(br)  # the whole operation drops
                    continue
                if op.gate == "CX":
                    state = oracle.apply_cx(br.state, op.targets[0], op.targets[1], n)
                else:
                    state = oracle.apply_1q(br.state, op.gate, op.targets[0], n)
                nxt.append(_Branch(br.weight, state, br.status, br.record))
                continue

            q = op.targets[0]
            level = br.status[q]

            if op.kind == "reset":
                if level == LOST:
                    nxt.append(br)  # reset does not restore a lost qubit
                    continue
                for w, post, bit in _hidden_collapse(br, q, n):
                    status = list(br.status)
                    status[q] = G
                    nxt.append(
                        _Branch(
                            br.weight * w,
                            _prep_zero(post, q, bit, n),
                            tuple(status),
                            br.record,
                        )
                    )
                continue

            if op.kind in ("measure", "measure_reset"):
                if level in _COMPUTATIONAL:
                    for w, post, bit in _hidden_collapse(br, q, n):
                        state = post
                        if op.kind == "measure_reset":
                            state = _prep_zero(post, q, bit, n)
                        nxt.append(_Branch(br.weight * w, state, br.status, br.record + (bit,)))
                else:
                    for bit in (0, 1):
                        p_bit = classifier[bit][level]
                        if p_bit <= 0.0:
                            continue
                        status = list(br.status)
                        if op.kind == "measure_reset" and level != LOST:
                            status[q] = G  # reset restores the leaked qubit
                        nxt.append(
                            _Branch(
                                br.weight * p_bit,
                                br.state,
                                tuple(status),
                                br.record + (bit,),
                            )
                        )
                continue

            if op.kind == "loss":
                if level == LOST:
                    nxt.append(br)
                    continue
                p = op.prob
                if p < 1.0:
                    nxt.append(_Branch(br.weight * (1.0 - p), br.state, br.status, br.record))
                if p > 0.0:
                    status = list(br.status)
                    status[q] = LOST
                    if level in _COMPUTATIONAL:
                        # Source-independent trace-out: hidden collapse, no filter.
                        for w, post, _bit in _hidden_collapse(br, q, n):
                            nxt.append(_Branch(br.weight * p * w, post, tuple(status), br.record))
                    else:
                        nxt.append(_Branch(br.weight * p, br.state, tuple(status), br.record))
                continue

            assert op.kind == "transition"
            column = transitions[op.tag]

            if level not in _COMPUTATIONAL:
                # Classical consult: draw from the status level's column.
                stay = 1.0 - sum(column[d][level] for d in range(5))
                if stay > 0.0:
                    nxt.append(_Branch(br.weight * stay, br.state, br.status, br.record))
                for d in range(5):
                    p = column[d][level]
                    if p <= 0.0:
                        continue
                    status = list(br.status)
                    status[q] = d
                    if d in _COMPUTATIONAL:
                        # Recapture: materialize the carrier at |d>.
                        for w, post, bit in _hidden_collapse(br, q, n):
                            state = oracle.set_collapsed_qubit(
                                _prep_zero(post, q, bit, n), q, 0, d, n
                            )
                            nxt.append(_Branch(br.weight * p * w, state, tuple(status), br.record))
                    else:
                        nxt.append(_Branch(br.weight * p, br.state, tuple(status), br.record))
                continue

            # Quantum consult: the physical per-site channel.
            ptot = [sum(column[d][s] for d in range(5)) for s in _COMPUTATIONAL]

            # Fire branches: collapse onto the source, then land the destination.
            pops = {}
            for s in _COMPUTATIONAL:
                w_s, collapsed = oracle.collapse(br.state, q, s, n)
                pops[s] = w_s
                if w_s <= 0.0:
                    continue
                for d in range(5):
                    p = column[d][s]
                    if p <= 0.0:
                        continue
                    status = list(br.status)
                    if d in _COMPUTATIONAL:
                        state = oracle.set_collapsed_qubit(collapsed, q, s, d, n)
                    else:
                        status[q] = d
                        state = collapsed  # factored; parked at |s>
                    nxt.append(_Branch(br.weight * w_s * p, state, tuple(status), br.record))

            # No-fire branch.
            if damping == "exact":
                w0, post = oracle.damp_no_fire(br.state, q, ptot[G], ptot[E], n)
                if w0 > 0.0:
                    nxt.append(_Branch(br.weight * w0, post, br.status, br.record))
            else:
                w0 = 1.0 - ptot[G] * pops[G] - ptot[E] * pops[E]
                if w0 > 0.0:
                    nxt.append(_Branch(br.weight * w0, br.state, br.status, br.record))

        # Prune and cap.
        kept = []
        for br in nxt:
            if br.weight < _PRUNE_EPS:
                result.dropped_mass += br.weight
            else:
                kept.append(br)
        branches = kept
        if len(branches) > _MAX_BRANCHES:
            raise RuntimeError(f"branch explosion: {len(branches)} branches")

    for br in branches:
        emit(br)
    return result


def tvd(p: dict[tuple[int, ...], float], q: dict[tuple[int, ...], float]) -> float:
    """Total variation distance between two record distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def empirical_record_probs(measurements: np.ndarray) -> dict[tuple[int, ...], float]:
    """Empirical record joint from a (shots x slots) 0/1 array."""
    m = np.asarray(measurements)
    probs: dict[tuple[int, ...], float] = {}
    inv = 1.0 / m.shape[0]
    for row in m:
        key = tuple(int(b) for b in row)
        probs[key] = probs.get(key, 0.0) + inv
    return probs
