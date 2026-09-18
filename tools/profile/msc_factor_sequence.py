"""Synthetic five-check noisy CSS sequence with explicit physical-gate reference.

The generated geometry is a code block, not an authors' cultivation circuit.
Every round contains a T-conjugated all-Y measurement and actual CSS projection.
Readout errors affect reported bits only. No postselection or syndrome correction
is inserted; a physical frame carries each measured sector into the next round.
"""

import random
from pathlib import Path

import numpy as np
import stim
from export_msc_sampler import Writer
from fold_check import PhaseMonomial
from fold_contraction import ROOTS
from msc_factor_sampling import FactorCode, generated_color_checks
from msc_growth_contraction import FactoredTerm
from msc_sampling import draw
from study_msc_factor_sampling import export_native, write_factor_plan

ROUNDS = 5


def local_table():
    table = []
    for faults in range(256):
        branches = []
        for branch in (0, 1):
            op = PhaseMonomial(1)
            for layer in range(4):
                axis = (faults >> (2 * layer)) & 3
                if axis:
                    op.append("_XYZ"[axis], (0,))
                if layer == 0:
                    op.append("T", (0,))
                elif layer == 1 and branch:
                    op.append("Y", (0,))
                elif layer == 2:
                    op.append("T_DAG", (0,))
            branches.append((op.flips, op.global_phase, op.linear[0]))
        table.append(branches)
    return table


class Sequence:
    def __init__(self, distance, noise=0.001, axis="Z"):
        if not 0 <= noise <= 1 or axis not in "XYZ" or len(axis) != 1:
            raise ValueError("invalid noise or terminal axis")
        self.checks, self.width = generated_color_checks(distance)
        if self.width >= 64:
            raise ValueError("native sequence requires at most 63 data wires")
        self.code = FactorCode(self.checks, self.width)
        if self.code.rank != len(self.code.zchecks):
            raise ValueError("native sequence requires balanced CSS")
        self.noise, self.axis = noise, axis
        self.table = local_table()
        self.stride = 1 + len(self.checks)
        self.record_count = ROUNDS * self.stride + 1

    def history(self, seed):
        rng = random.Random(seed)
        faults = [
            [rng.randrange(1, 4) if rng.random() < self.noise else 0 for _ in range(4 * self.width)]
            for _ in range(ROUNDS)
        ]
        return faults, [int(rng.random() < self.noise) for _ in range(self.record_count)]

    def payload(self, faults, bit):
        terms = []
        for branch in (0, 1):
            op = PhaseMonomial(self.width)
            for q in range(self.width):
                index = sum(faults[layer * self.width + q] << (2 * layer) for layer in range(4))
                flip, phase, linear = self.table[index][branch]
                op.flips |= flip << q
                op.global_phase = (op.global_phase + phase) % 8
                op.linear[q] = linear
            terms.append(
                FactoredTerm(op, np.empty((0, 2), dtype=complex), (-1) ** (branch * bit) / 2)
            )
        return terms

    def frame(self, syndrome):
        x = z = 0
        for bit, (cx, cz) in zip(syndrome, self.code.duals, strict=True):
            if bit:
                x ^= cx
                z ^= cz
        return x, z

    def terminal_probability(self, logical, frame):
        x, z = frame
        if self.axis == "Z":
            expectation = abs(logical[0]) ** 2 - abs(logical[1]) ** 2
            sign = x.bit_count()
        elif self.axis == "X":
            expectation = 2 * (logical[0].conjugate() * logical[1]).real
            sign = z.bit_count()
        else:
            expectation = 2 * (logical[0].conjugate() * logical[1]).imag
            sign = (x ^ z).bit_count() + (self.width - 1) // 2
        return float((1 + (-1) ** sign * expectation) / 2)

    def outputs(self, outcomes, flips):
        records = [int(a) ^ int(b) for a, b in zip(outcomes, flips, strict=True)]
        detectors = [
            records[r * self.stride + k] ^ records[(r - 1) * self.stride + k]
            for r in range(1, ROUNDS)
            for k in range(1, self.stride)
        ]
        return dict(records=records, detectors=detectors, observables=[records[-1]])

    def run(self, seed=0, history=None, outcomes=None):
        rng = random.Random(seed)
        faults, flips = self.history(seed + 1) if history is None else history
        logical = np.asarray([1, ROOTS[1]]) / np.sqrt(2)
        frame = (0, 0)
        result, rounds = [], []
        probability = 1.0
        for r in range(ROUNDS):
            payloads = [self.payload(faults[r], bit) for bit in (0, 1)]
            weights = [self.code.norm(logical, frame, terms) for terms in payloads]
            if abs(sum(weights) - 1) > 1e-10:
                raise AssertionError("gadget is not trace preserving")
            bit = draw(weights, rng)[0] if outcomes is None else outcomes[r * self.stride]
            terms = payloads[bit]
            if outcomes is None:
                syndrome, _ = self.code.sample_syndrome(logical, frame, terms, rng)
            else:
                syndrome = outcomes[r * self.stride + 1 : (r + 1) * self.stride]
            output = self.code.contract(logical, frame, terms, syndrome)
            weight = float(np.vdot(output, output).real)
            if weight <= 0:
                raise ValueError("unreachable sequence history")
            logical = output / np.sqrt(weight)
            frame = self.frame(syndrome)
            probability *= weight
            result += [bit, *syndrome]
            rounds.append(dict(probability=weight, logical=logical.copy(), frame=frame))
        zero = self.terminal_probability(logical, frame)
        bit = draw([zero, 1 - zero], rng)[0] if outcomes is None else outcomes[-1]
        probability *= 1 - zero if bit else zero
        result.append(bit)
        return dict(
            probability=probability,
            logical=logical,
            frame=frame,
            rounds=rounds,
            outcomes=result,
            faults=faults,
            flips=flips,
            **self.outputs(result, flips),
        )

    def circuit(self, history=None):
        """Retain elementary T gates; fixing faults removes stochastic channels."""
        n = self.width
        prep = stim.Tableau.from_stabilizers([*self.checks, stim.PauliString("X" * n)]).to_circuit()
        lines = [str(prep).rstrip()]
        ladder = [f"CX {q} {n - 1}" for q in range(n - 1)]
        lines += [*ladder, f"T {n - 1}", *reversed(ladder)]
        wires = " ".join(map(str, range(n)))
        record = 0

        def noise(r, layer):
            if history is None:
                lines.append(f"DEPOLARIZE1({self.noise}) {wires}")
            else:
                for q in range(n):
                    axis = history[0][r][layer * n + q]
                    if axis:
                        lines.append(f"{'_XYZ'[axis]} {q}")

        def measure(pauli):
            nonlocal record
            probability = self.noise if history is None else 0
            lines.append(f"MPP({probability}) " + pauli)
            record += 1

        for r in range(ROUNDS):
            noise(r, 0)
            lines.append(f"T {wires}")
            noise(r, 1)
            measure("*".join(f"Y{q}" for q in range(n)))
            noise(r, 2)
            lines.append(f"T_DAG {wires}")
            noise(r, 3)
            for p in self.checks:
                measure("*".join(f"{'_XYZ'[p[q]]}{q}" for q in range(n) if p[q]))
            if r:
                for k in range(1, self.stride):
                    current = r * self.stride + k - record
                    prior = current - self.stride
                    lines.append(f"DETECTOR rec[{current}] rec[{prior}]")
        measure("*".join(f"{self.axis}{q}" for q in range(n)))
        lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
        return "\n".join(lines) + "\n"

    def program_history(self, history):
        from msc_protocol import Program

        program = Program(self.circuit())
        physical = iter(axis for row in history[0] for axis in row)
        flips = iter(history[1])
        bound = {}
        for k, site in enumerate(program.sites):
            choice = ("flip" if next(flips) else None) if site.readout else "_XYZ"[next(physical)]
            if choice and choice != "_":
                bound[k] = choice
        return program, bound

    def export(self, path):
        identity = FactoredTerm(PhaseMonomial(self.width), np.empty((0, 2), complex), 1)
        export_native(self.code, [(np.array([1, 0]), (0, 0), [identity])], path)
        w = Writer()
        w.put(self.noise, "XYZ".index(self.axis))
        write_factor_plan(w, self.code.single)
        for values in (self.code.xchecks, self.code.zchecks, self.code.coordinates):
            w.vector(values)
        w.put(len(self.code.order))
        for (x, k), (cx, cz) in zip(self.code.order, self.code.duals, strict=True):
            w.put(int(x), k, cx, cz)
        for row in self.table:
            for branch in row:
                w.put(*branch)
        with Path(path).open("a") as stream:
            stream.write("\n".join(w.tokens) + "\n")
