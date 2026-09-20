"""Certify a signed-code handoff through the physical d5-to-d7 Clifford bridge."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import stim
from folded_msc_f7 import F7Builder, make_f7
from folded_msc_family import Surface
from validate_folded_msc import noise_labels


def pauli(axis, support, width=99):
    result = stim.PauliString(width)
    for q in support:
        result[q] = axis
    return result


def logicals(surface):
    end = 2 * surface.distance - 1
    x = pauli("X", [surface.index[0, y] for y in range(0, end, 2)])
    z = pauli("Z", [surface.index[x, 0] for x in range(0, end, 2)])
    return [x, 1j * x * z, z]


def bits(p):
    return sum((p[q] in (1, 2)) << q for q in range(len(p))) | sum(
        (p[q] in (2, 3)) << (q + len(p)) for q in range(len(p))
    )


def decompose(target, generators):
    basis = {}
    for i, generator in enumerate(generators):
        row, combination = bits(generator), 1 << i
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row, combination
                break
            r, c = basis[pivot]
            row, combination = row ^ r, combination ^ c
        if not row:
            raise ValueError("dependent bridge generators")
    row, combination = bits(target), 0
    while row:
        pivot = row.bit_length() - 1
        if pivot not in basis:
            raise ValueError("growth does not preserve the required operator span")
        r, c = basis[pivot]
        row, combination = row ^ r, combination ^ c
    product = stim.PauliString(len(target))
    for i, generator in enumerate(generators):
        if combination >> i & 1:
            product *= generator
    ratio = target.sign / product.sign
    if ratio not in (1, -1):
        raise ValueError("non-Hermitian bridge relation")
    return combination, int(ratio == -1)


class GrowthHandoff:
    def __init__(self):
        small, large = Surface(5), Surface(7)
        self.mapping = {q: large.index[x + 2, y + 2] for q, (x, y) in enumerate(small.points)}
        self.input_checks = [
            pauli(axis, [self.mapping[q] for q in row])
            for axis in "XZ"
            for row in small.checks(axis)
        ]
        self.input_logicals = []
        for p in logicals(small):
            mapped = stim.PauliString(99)
            mapped.sign = p.sign
            for q in range(len(small.points)):
                mapped[self.mapping[q]] = p[q]
            self.input_logicals.append(mapped)
        self.new_qubits = [q for q in range(85) if q not in self.mapping.values()]
        self.preparations = [pauli("Z", [q]) for q in self.new_qubits]
        self.output_checks = [pauli(a, row) for a in "XZ" for row in large.checks(a)]
        self.output_logicals = logicals(large)
        self.ancillas = [pauli("Z", [q]) for q in range(85, 99)]
        builder = F7Builder(0.001)
        builder.growth_to_seven()
        growth = stim.Circuit()
        for op in builder.circuit.operations:
            if op.probability is None and op.name != "R":
                growth.append(op.name, op.qubits)
        builder.stabilizers("d7_pre_checks", noisy=True)
        self.operations = builder.circuit.operations
        physical_bridge = [
            op
            for op in make_f7(0.001).operations
            if op.stage in {"grow_regular5_to_regular7", "d7_pre_checks"}
        ]
        if self.operations != physical_bridge:
            raise ValueError("bridge differs from the complete physical protocol")
        input_generators = self.input_checks + self.preparations
        inverse = growth.inverse()
        check_relations = [
            decompose(p.after(inverse), input_generators) for p in self.output_checks
        ]
        # Both bases have 84 generators; containment therefore proves equality.
        for p in input_generators:
            decompose(p.after(growth), self.output_checks)
        logical_relations = []
        for source, target in zip(self.input_logicals, self.output_logicals, strict=True):
            mask, sign = decompose(target.after(inverse), input_generators + [source])
            if not mask >> 84:
                raise ValueError("logical operator was lost in growth")
            logical_relations.append((mask & ((1 << 40) - 1), sign))
        if logical_relations[1] != (
            logical_relations[0][0] ^ logical_relations[2][0],
            logical_relations[0][1] ^ logical_relations[2][1],
        ):
            raise ValueError("logical bridge is not a Pauli frame update")
        self.relations = []
        ancilla_masks = [1 << (40 + i) for i in range(14)]
        ancilla_signs = [0] * 14
        check = 0
        for op in self.operations:
            if op.probability is not None or op.name not in {"R", "M"}:
                continue
            q = op.qubits[0]
            if q < 85:
                if op.name != "R" or q not in self.new_qubits:
                    raise ValueError("unsupported bridge data measurement")
                relation = (0, 0)
            elif op.name == "R":
                relation = (ancilla_masks[q - 85], ancilla_signs[q - 85])
                ancilla_masks[q - 85] = ancilla_signs[q - 85] = 0
            else:
                mask, sign = check_relations[check]
                relation = (mask & ((1 << 40) - 1), sign)
                ancilla_masks[q - 85], ancilla_signs[q - 85] = relation
                check += 1
            self.relations.append(relation)
        if check != 84:
            raise ValueError("unexpected bridge syndrome record count")
        self.record_count = len(self.relations)
        self.relations += [(mask & ((1 << 40) - 1), sign) for mask, sign in check_relations]
        self.relations += logical_relations
        self.relations += list(zip(ancilla_masks, ancilla_signs, strict=True))
        self.probes = self.output_checks + self.output_logicals + self.ancillas
        self.probe_masks = [bits(p) for p in self.probes]
        self.sites = {}
        for position, op in enumerate(self.operations):
            if op.probability is not None:
                self.sites[position] = {
                    label: self.fault_effect(position, label) for label in noise_labels(op)
                }

    def fault_effect(self, position, label):
        op = self.operations[position]
        x = sum((axis in "XY") << q for q, axis in zip(op.qubits, label, strict=True))
        z = sum((axis in "YZ") << q for q, axis in zip(op.qubits, label, strict=True))
        flipped, record = 0, sum(p.name in {"R", "M"} for p in self.operations[: position + 1])
        for op in self.operations[position + 1 :]:
            if op.probability is not None:
                continue
            q = op.qubits[0]
            if op.name == "H":
                change = ((x ^ z) >> q & 1) << q
                x, z = x ^ change, z ^ change
            elif op.name == "CX":
                target = op.qubits[1]
                x ^= ((x >> q) & 1) << target
                z ^= ((z >> target) & 1) << q
            elif op.name in {"R", "M"}:
                flipped |= ((x >> q) & 1) << record
                record += 1
                if op.name == "R":
                    x, z = x & ~(1 << q), z & ~(1 << q)
            else:
                raise ValueError("unsupported bridge operation")
        for i, p in enumerate(self.probe_masks):
            anticommutes = ((x & (p >> 99)).bit_count() + (z & p).bit_count()) % 2
            flipped |= anticommutes << (self.record_count + i)
        return flipped

    def apply(self, input_bits, faults):
        result = sum(
            (((input_bits & mask).bit_count() % 2) ^ sign) << i
            for i, (mask, sign) in enumerate(self.relations)
        )
        for position, label in faults.items():
            result ^= self.sites[position][label]
        return result

    def reference(self, input_bits, faults, axis):
        generators = [
            p * (-1 if input_bits >> i & 1 else 1) for i, p in enumerate(self.input_checks)
        ]
        generators += self.preparations + [self.input_logicals[axis]]
        generators += [
            p * (-1 if input_bits >> (40 + i) & 1 else 1) for i, p in enumerate(self.ancillas)
        ]
        sim = stim.TableauSimulator()
        sim.set_inverse_tableau(stim.Tableau.from_stabilizers(generators).inverse())
        result, record = 0, 0
        for i, op in enumerate(self.operations):
            if op.probability is not None:
                for q, p in zip(op.qubits, faults.get(i, "I" * len(op.qubits)), strict=True):
                    if p != "I":
                        sim.do(stim.CircuitInstruction(p, [q]))
                continue
            if op.name in {"M", "R"}:
                value = sim.peek_z(op.qubits[0])
                if abs(value) != 1:
                    raise AssertionError("bridge record is not deterministic after sector sampling")
                result |= int(value < 0) << record
                record += 1
            sim.do(stim.CircuitInstruction(op.name, op.qubits))
        checked = (1 << self.record_count) - 1
        for i, p in enumerate(self.probes):
            value = sim.peek_observable_expectation(p)
            if 84 <= i < 87 and i != 84 + axis:
                if value != 0:
                    raise AssertionError("unexpected logical axis rotation")
                continue
            if abs(value) != 1:
                raise AssertionError("bridge leaves the signed output code")
            bit = self.record_count + i
            result |= int(value < 0) << bit
            checked |= 1 << bit
        return result, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    bridge = GrowthHandoff()
    compile_seconds = time.perf_counter() - started
    rng = random.Random(48319)
    checks = []
    for probability in (0, 0.001, 0.03):
        for axis in range(3):
            for _ in range(4):
                inputs = rng.getrandbits(54)
                faults = {
                    i: rng.choice(list(channels))
                    for i, channels in bridge.sites.items()
                    if rng.random() < probability
                }
                actual, mask = bridge.reference(inputs, faults, axis)
                if (actual ^ bridge.apply(inputs, faults)) & mask:
                    raise AssertionError("compiled bridge differs from physical Stim replay")
                checks.append(dict(probability=probability, axis="XYZ"[axis], faults=len(faults)))
    result = dict(
        compile_seconds=compile_seconds,
        total_seconds=time.perf_counter() - started,
        bridge_sha256=hashlib.sha256(
            "\n".join(op.text() for op in bridge.operations).encode()
        ).hexdigest(),
        input_css_generators=40,
        prepared_qubits=44,
        output_css_generators=84,
        classical_input_bits=54,
        record_bits=bridge.record_count,
        noise_sites=len(bridge.sites),
        pauli_channels=sum(len(channels) for channels in bridge.sites.values()),
        packed_transfer_bytes=8 * len(bridge.relations)
        + 8
        * ((len(bridge.relations) + 63) // 64)
        * (1 + sum(len(channels) for channels in bridge.sites.values())),
        logical_action="input-syndrome-dependent Pauli frame",
        checks=checks,
        integrated_native_sampler=False,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
