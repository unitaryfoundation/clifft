"""Certify a signed-code handoff through the physical d5-to-d7 Clifford bridge."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import stim
from folded_growth import GrowthHandoff


def reference(bridge, input_bits, faults, axis):
    generators = [p * (-1 if input_bits >> i & 1 else 1) for i, p in enumerate(bridge.input_checks)]
    generators += bridge.preparations + [bridge.input_logicals[axis]]
    generators += [
        p * (-1 if input_bits >> (40 + i) & 1 else 1) for i, p in enumerate(bridge.ancillas)
    ]
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(stim.Tableau.from_stabilizers(generators).inverse())
    result, record = 0, 0
    for i, op in enumerate(bridge.operations):
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
    checked = (1 << bridge.record_count) - 1
    for i, p in enumerate(bridge.probes):
        value = sim.peek_observable_expectation(p)
        if 84 <= i < 87 and i != 84 + axis:
            if value != 0:
                raise AssertionError("unexpected logical axis rotation")
            continue
        if abs(value) != 1:
            raise AssertionError("bridge leaves the signed output code")
        bit = bridge.record_count + i
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
                actual, mask = reference(bridge, inputs, faults, axis)
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
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
